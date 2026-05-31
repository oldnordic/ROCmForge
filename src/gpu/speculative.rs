//! GPU Speculative Decoding (Draft-and-Verify) Engine.
//!
//! Handles safe loading of target/draft models, dual-cache isolations,
//! draft generation steps, and parallel batched verification passes.

use super::cache::{GpuForwardScratch, GpuKvCache, GpuPrefillScratch};
use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::weights::{GpuBuffer, GpuModelWeights};
use crate::config::ModelConfig;
use crate::cpu::cache::CpuForwardScratch;
use crate::cpu::weights::CpuModelWeights;

/// Safe matrix/buffer-non-overlapping address isolation helper.
fn verify_cache_isolation(target_kv: &GpuKvCache, draft_kv: &GpuKvCache) -> GpuResult<()> {
    for layer in 0..target_kv.num_layers {
        let t_k = target_kv.k_ptr(layer)? as usize;
        let t_v = target_kv.v_ptr(layer)? as usize;
        let t_size = target_kv.max_seq_len * target_kv.kv_size * std::mem::size_of::<f32>();

        for d_layer in 0..draft_kv.num_layers {
            let d_k = draft_kv.k_ptr(d_layer)? as usize;
            let d_v = draft_kv.v_ptr(d_layer)? as usize;
            let d_size = draft_kv.max_seq_len * draft_kv.kv_size * std::mem::size_of::<f32>();

            // Assert K caches do not overlap
            let diff_k = (t_k as isize - d_k as isize).abs() as usize;
            if diff_k < std::cmp::max(t_size, d_size) {
                return Err(GpuError::HipApiError {
                    code: -1,
                    description: format!(
                        "Safety violation: Overlapping Key caches between target layer {} and draft layer {}",
                        layer, d_layer
                    ),
                });
            }

            // Assert V caches do not overlap
            let diff_v = (t_v as isize - d_v as isize).abs() as usize;
            if diff_v < std::cmp::max(t_size, d_size) {
                return Err(GpuError::HipApiError {
                    code: -1,
                    description: format!(
                        "Safety violation: Overlapping Value caches between target layer {} and draft layer {}",
                        layer, d_layer
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Dynamic checker to verify if a weight matrix type supports batched GPU operations.
fn supports_batched_gemm_type(wtype: crate::loader::GgmlType) -> bool {
    matches!(
        wtype,
        crate::loader::GgmlType::Q4_0 | crate::loader::GgmlType::Q4_1
    )
}

/// The Speculative Engine coordinating dual-model co-execution.
pub struct SpeculativeEngine {
    pub target_model: GpuModelWeights,
    pub target_cpu_weights: CpuModelWeights,
    pub draft_model: GpuModelWeights,
    pub draft_cpu_weights: CpuModelWeights,
    pub target_config: ModelConfig,
    pub draft_config: ModelConfig,
    pub target_kv: GpuKvCache,
    pub draft_kv: GpuKvCache,
    pub target_prefill_scratch: GpuPrefillScratch,
    pub target_scratch: GpuForwardScratch,
    pub target_host_scratch: CpuForwardScratch,
    pub draft_scratch: GpuForwardScratch,
    pub draft_host_scratch: CpuForwardScratch,
    pub draft_prefill_scratch: GpuPrefillScratch,
}

impl SpeculativeEngine {
    /// Instantiate the Speculative Engine, co-loading both models with VRAM safety limit checks.
    pub fn new(
        device: &GpuDevice,
        target_path: &str,
        draft_path: &str,
        max_seq_len: usize,
        prompt_len: usize,
    ) -> GpuResult<Self> {
        let device_id = device.device_id();

        // 0. Pre-flight VRAM headroom check before loading weights
        let vram_session = crate::gpu::vram_budget::VramSession::new(device_id)?;
        let target_size = std::fs::metadata(target_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let draft_size = std::fs::metadata(draft_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let estimated_weights = target_size + draft_size;

        if estimated_weights > (vram_session.inference_budget as f64 * 0.85) as usize {
            return Err(GpuError::OutOfMemory {
                requested: estimated_weights,
                available: vram_session.inference_budget,
                hint: format!(
                    "Co-loading target model ({:.1} MB) and draft model ({:.1} MB) exceeds 85% of usable VRAM session budget ({:.1} MB free, {:.1} MB total). Aborting for safety.",
                    target_size as f64 / (1024.0 * 1024.0),
                    draft_size as f64 / (1024.0 * 1024.0),
                    vram_session.inference_budget as f64 / (1024.0 * 1024.0),
                    vram_session.total as f64 / (1024.0 * 1024.0)
                ),
            });
        }

        // 1. Load target model config and weights
        let (target_config, target_cpu_weights, target_model) = if target_path.ends_with(".rfm") {
            let file =
                crate::loader::RfmFile::open(target_path).map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("Failed to open target RFM: {}", e),
                })?;
            let config =
                ModelConfig::from_rfm(&file.metadata).map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("Target config error: {}", e),
                })?;
            let cpu =
                CpuModelWeights::load_rfm(&file, &config).map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("Target CPU weights error: {}", e),
                })?;
            let gpu = GpuModelWeights::load_rfm_for_device(&file, &config, device_id)?;
            (config, cpu, gpu)
        } else {
            let file =
                crate::loader::GgufFile::open(target_path).map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("Failed to open target GGUF: {}", e),
                })?;
            let config = ModelConfig::from_gguf(&file).map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("Target config error: {}", e),
            })?;
            let cpu = CpuModelWeights::load(&file, &config).map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("Target CPU weights error: {}", e),
            })?;
            let gpu = GpuModelWeights::load_for_device(&file, &config, device_id)?;
            (config, cpu, gpu)
        };

        // 2. Load draft model config and weights
        let (draft_config, draft_cpu_weights, draft_model) = if draft_path.ends_with(".rfm") {
            let file =
                crate::loader::RfmFile::open(draft_path).map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("Failed to open draft RFM: {}", e),
                })?;
            let config =
                ModelConfig::from_rfm(&file.metadata).map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("Draft config error: {}", e),
                })?;
            let cpu =
                CpuModelWeights::load_rfm(&file, &config).map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("Draft CPU weights error: {}", e),
                })?;
            let gpu = GpuModelWeights::load_rfm_for_device(&file, &config, device_id)?;
            (config, cpu, gpu)
        } else {
            let file =
                crate::loader::GgufFile::open(draft_path).map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("Failed to open draft GGUF: {}", e),
                })?;
            let config = ModelConfig::from_gguf(&file).map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("Draft config error: {}", e),
            })?;
            let cpu = CpuModelWeights::load(&file, &config).map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("Draft CPU weights error: {}", e),
            })?;
            let gpu = GpuModelWeights::load_for_device(&file, &config, device_id)?;
            (config, cpu, gpu)
        };

        // 3. Allocate KV Caches
        let target_kv = GpuKvCache::new(&target_config, max_seq_len)?;
        let draft_kv = GpuKvCache::new(&draft_config, max_seq_len)?;

        // 4. Verify absolute pointer address cache bounds isolation
        verify_cache_isolation(&target_kv, &draft_kv)?;

        // 5. Pre-Allocation VRAM Guardrail: Target weights + Draft weights + caches + 512MB <= 90% total VRAM
        let target_vram = target_model.vram_bytes() + target_kv.vram_bytes();
        let draft_vram = draft_model.vram_bytes() + draft_kv.vram_bytes();
        let scratch_headroom = 512 * 1024 * 1024;
        let total_required = target_vram + draft_vram + scratch_headroom;

        let (_, total_vram) = super::ffi::hip_get_mem_info(device_id)?;
        if total_required as f64 > 0.90 * total_vram as f64 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "VRAM Safety limit exceeded: Target model ({:.1} MB) + Draft model ({:.1} MB) + caches ({:.1} MB) require {:.1} MB total, exceeding 90% limit ({:.1} MB).",
                    target_model.vram_bytes() as f64 / (1024.0 * 1024.0),
                    draft_model.vram_bytes() as f64 / (1024.0 * 1024.0),
                    (target_kv.vram_bytes() + draft_kv.vram_bytes()) as f64 / (1024.0 * 1024.0),
                    total_required as f64 / (1024.0 * 1024.0),
                    (total_vram as f64 * 0.90) / (1024.0 * 1024.0)
                ),
            });
        }

        // 6. Allocate reusable prefill and forward scratches
        let prefill_capacity = std::cmp::max(prompt_len, 16);
        let target_prefill_scratch = GpuPrefillScratch::new(&target_config, prefill_capacity)?;
        let target_scratch = GpuForwardScratch::new(&target_config)?;
        let target_host_scratch = CpuForwardScratch::new(&target_config);

        let draft_scratch = GpuForwardScratch::new(&draft_config)?;
        let draft_host_scratch = CpuForwardScratch::new(&draft_config);
        let draft_prefill_scratch = GpuPrefillScratch::new(&draft_config, prefill_capacity)?;

        Ok(Self {
            target_model,
            target_cpu_weights,
            draft_model,
            draft_cpu_weights,
            target_config,
            draft_config,
            target_kv,
            draft_kv,
            target_prefill_scratch,
            target_scratch,
            target_host_scratch,
            draft_scratch,
            draft_host_scratch,
            draft_prefill_scratch,
        })
    }

    /// Autoregressively draft N speculative tokens using the draft model on the GPU.
    pub fn draft_tokens(
        &mut self,
        device: &GpuDevice,
        start_pos: usize,
        draft_count: usize,
        first_token: u32,
    ) -> GpuResult<Vec<u32>> {
        let mut drafted = Vec::with_capacity(draft_count);
        let mut current_token = first_token;

        for i in 0..draft_count {
            let pos = start_pos + i;

            // Check bounds: draft position must not exceed max_seq_len
            if pos >= self.draft_kv.max_seq_len {
                return Err(GpuError::HipApiError {
                    code: -1,
                    description: format!(
                        "Draft position {} exceeds maximum sequence length {}",
                        pos, self.draft_kv.max_seq_len
                    ),
                });
            }

            // Embed the current input token
            super::forward::gpu_embed_token_hybrid(
                device,
                current_token,
                &self.draft_model,
                &self.draft_cpu_weights,
                &mut self.draft_scratch,
                &mut self.draft_host_scratch,
                &self.draft_config,
            )?;

            // Run a single forward pass of the draft model
            let opt_token = super::forward::gpu_full_forward_hybrid(
                device,
                &self.draft_model,
                &self.draft_cpu_weights,
                &mut self.draft_kv,
                &mut self.draft_scratch,
                &mut self.draft_host_scratch,
                pos,
                &self.draft_config,
                super::forward::GpuLogitsMode::GreedyArgmax,
            )?;

            let token = if let Some(t) = opt_token {
                t
            } else {
                device.synchronize()?;
                crate::cpu::sampler::cpu_sample_greedy(
                    &self.draft_host_scratch.logits[..self.draft_config.vocab_size],
                )
            };

            drafted.push(token);
            current_token = token;
        }

        Ok(drafted)
    }

    /// Run the parallel batched verification pass on the target model.
    pub fn verify_tokens(
        &mut self,
        device: &GpuDevice,
        start_pos: usize,
        draft_tokens: &[u32],
        last_verified_token: u32,
    ) -> GpuResult<(Vec<u32>, usize)> {
        if start_pos == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_tokens: start_pos must be >= 1".to_string(),
            });
        }

        // Build token IDs batch: [last_verified_token] + draft_tokens
        let mut token_ids = Vec::with_capacity(draft_tokens.len() + 1);
        token_ids.push(last_verified_token);
        token_ids.extend_from_slice(draft_tokens);

        let seq_len = token_ids.len();
        let target_start = start_pos - 1;

        // Assert target verification sequence boundaries to prevent cache bounds overrun
        if target_start + seq_len > self.target_kv.max_seq_len {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "Target verification sequence [{}-{}] exceeds maximum sequence length {}",
                    target_start,
                    target_start + seq_len,
                    self.target_kv.max_seq_len
                ),
            });
        }

        // Run target model verification via sequential decode path (matching draft_tokens),
        // so that both paths use identical kernels and produce bit-identical logits.
        // This is the key fix: prefill kernels differ from decode kernels; using
        // the same decode path guarantees deterministic acceptance/rejection.
        let draft_count = draft_tokens.len();
        let mut num_accepted = 0;
        let mut accepted_tokens = Vec::with_capacity(draft_count + 1);

        for i in 0..draft_count {
            let pos = start_pos + i;
            if pos >= self.target_kv.max_seq_len {
                return Err(GpuError::HipApiError {
                    code: -1,
                    description: format!(
                        "Target verify position {} exceeds maximum sequence length {}",
                        pos, self.target_kv.max_seq_len
                    ),
                });
            }

            let input_token = if i == 0 {
                last_verified_token
            } else {
                draft_tokens[i - 1]
            };

            super::forward::gpu_embed_token_hybrid(
                device,
                input_token,
                &self.target_model,
                &self.target_cpu_weights,
                &mut self.target_scratch,
                &mut self.target_host_scratch,
                &self.target_config,
            )?;

            let target_token = super::forward::gpu_full_forward_hybrid(
                device,
                &self.target_model,
                &self.target_cpu_weights,
                &mut self.target_kv,
                &mut self.target_scratch,
                &mut self.target_host_scratch,
                pos,
                &self.target_config,
                super::forward::GpuLogitsMode::GreedyArgmax,
            )?;

            let target_token = target_token.unwrap_or_else(|| {
                // fallback should not happen with GreedyArgmax, but handle defensively
                crate::cpu::sampler::cpu_sample_greedy(
                    &self.target_host_scratch.logits[..self.target_config.vocab_size],
                )
            });

            if draft_tokens[i] == target_token {
                num_accepted += 1;
                accepted_tokens.push(target_token);
            } else {
                // Divergence found! Accept target's corrected output
                accepted_tokens.push(target_token);
                break;
            }
        }

        // If all N draft tokens are accepted, sample the (N+1)-th token from target
        if num_accepted == draft_count {
            let next_pos = start_pos + draft_count;
            let input_token = draft_tokens[draft_count - 1];

            super::forward::gpu_embed_token_hybrid(
                device,
                input_token,
                &self.target_model,
                &self.target_cpu_weights,
                &mut self.target_scratch,
                &mut self.target_host_scratch,
                &self.target_config,
            )?;

            let next_token = super::forward::gpu_full_forward_hybrid(
                device,
                &self.target_model,
                &self.target_cpu_weights,
                &mut self.target_kv,
                &mut self.target_scratch,
                &mut self.target_host_scratch,
                next_pos,
                &self.target_config,
                super::forward::GpuLogitsMode::GreedyArgmax,
            )?;
            let next_token = next_token.unwrap_or_else(|| {
                crate::cpu::sampler::cpu_sample_greedy(
                    &self.target_host_scratch.logits[..self.target_config.vocab_size],
                )
            });
            accepted_tokens.push(next_token);
        }

        // Synchronize the draft model's KV Cache with the last accepted target token.
        let last_accepted_token = *accepted_tokens
            .last()
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: "No accepted tokens returned".to_string(),
            })?;

        let sync_input_token = if num_accepted == 0 {
            last_verified_token
        } else {
            draft_tokens[num_accepted - 1]
        };

        super::forward::gpu_embed_token_hybrid(
            device,
            sync_input_token,
            &self.draft_model,
            &self.draft_cpu_weights,
            &mut self.draft_scratch,
            &mut self.draft_host_scratch,
            &self.draft_config,
        )?;

        let sync_pos = start_pos + num_accepted;
        super::forward::gpu_full_forward_hybrid(
            device,
            &self.draft_model,
            &self.draft_cpu_weights,
            &mut self.draft_kv,
            &mut self.draft_scratch,
            &mut self.draft_host_scratch,
            sync_pos,
            &self.draft_config,
            super::forward::GpuLogitsMode::Skip,
        )?;

        Ok((accepted_tokens, num_accepted))
    }
}
