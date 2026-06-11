//! Synchronous GPU inference wrapper for the HTTP server.
//!
//! Extracts the core prefill+decode loop from `main.rs::run_gpu_inference`
//! into a clean sync function with no CLI side effects.  Retains VRAM
//! pre-flight (critical on a display-attached GPU).

#[cfg(feature = "gpu")]
use crate::config::ModelConfig;
#[cfg(feature = "gpu")]
use crate::cpu::cache::CpuForwardScratch;
#[cfg(feature = "gpu")]
use crate::cpu::sampler::{cpu_sample_greedy, cpu_sample_top_p};
#[cfg(feature = "gpu")]
use crate::cpu::weights::CpuModelWeights;
#[cfg(feature = "gpu")]
use crate::gpu;
#[cfg(feature = "gpu")]
use crate::tokenizer::BpeTokenizer;
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
pub fn run_gpu_sync_inference(
    gpu_weights_arc: &Arc<crate::gpu::GpuModelWeights>,
    cpu_weights: &Arc<CpuModelWeights>,
    config: &ModelConfig,
    tok: &BpeTokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> crate::error::RocmForgeResult<(String, usize)> {
    // ── 1. GPU detection ────────────────────────────────────────────────────────
    let gpu_caps = gpu::detect().ok_or("GPU requested but no AMD GPU detected")?;

    // ── 2. VRAM pre-flight ──────────────────────────────────────────────────────
    let vram_session = gpu::VramSession::new(gpu_caps.device_id)
        ?;

    let max_seq_estimate = config.max_seq_len.min(max_tokens + 2048);
    let kv_estimate = gpu::GpuKvCache::estimate_bytes(config, max_seq_estimate);
    let scratch_estimate = gpu::GpuForwardScratch::estimate_bytes(config);

    vram_session
        .check_fits(0, kv_estimate, scratch_estimate)
        ?;

    // ── 3. Device init ─────────────────────────────────────────────────────────
    let device =
        gpu::GpuDevice::get_or_init(gpu_caps.device_id)?;

    let gpu_weights = gpu_weights_arc.as_ref();

    // ── 4. Allocate KV cache and scratch ─────────────────────────────────────────
    let max_seq = (prompt_tokens.len() + max_tokens).min(config.max_seq_len);
    let mut kv = gpu::GpuKvCache::new(config, max_seq)?;
    let mut gpu_scratch =
        gpu::GpuForwardScratch::new(config)?;

    // Expert scratch sized for the largest gate/up/down expert dims.
    'expert_scratch: for layer in &gpu_weights.layers {
        let all_compressed = [
            layer.ffn_gate_compressed.as_ref(),
            layer.ffn_up_compressed.as_ref(),
            layer.ffn_down_compressed.as_ref(),
        ];
        if all_compressed.iter().all(|x| x.is_some()) {
            let k = layer
                .ffn_gate_compressed
                .as_ref()
                .map(|c| c.k)
                .unwrap_or(32);
            let max_rows = all_compressed
                .iter()
                .filter_map(|x| *x)
                .map(|c| c.rows)
                .max()
                .unwrap_or(1);
            let max_cols = all_compressed
                .iter()
                .filter_map(|x| *x)
                .map(|c| c.cols)
                .max()
                .unwrap_or(1);
            let max_nnz = all_compressed
                .iter()
                .filter_map(|x| *x)
                .map(|c| c.max_nnz())
                .max()
                .unwrap_or(1);
            gpu_scratch
                .init_expert_scratch(k as u32, max_rows, max_cols, max_nnz)
                ?;
            break 'expert_scratch;
        }
    }

    let mut host_scratch = CpuForwardScratch::new(config);
    let use_greedy = top_p >= 1.0;
    let use_gpu_greedy_fastpath = use_greedy;

    let final_prompt_logits_mode = if use_gpu_greedy_fastpath {
        gpu::GpuLogitsMode::GreedyArgmax
    } else {
        gpu::GpuLogitsMode::DownloadToHost
    };

    // ── 5. Hotpath router ────────────────────────────────────────────────────────
    let profile = gpu::ModelProfile::from_weights(&gpu_weights, config);
    let path = gpu::select_path(&profile, prompt_tokens.len(), &vram_session);

    if let Err(_e) = gpu::check_path_vram(&path, config, prompt_tokens.len(), &vram_session) {
        // silently fall back to decode-style
    }

    // ── 6. Prefill ───────────────────────────────────────────────────────────────
    let prompt_next_token = match path {
        gpu::InferencePath::BatchedPrefill { .. } => {
            match gpu::GpuPrefillScratch::new(config, prompt_tokens.len()) {
                Ok(mut prefill_scratch) => {
                    match gpu::gpu_batched_prefill_forward_q4_0(
                        &device,
                        &gpu_weights,
                        &cpu_weights,
                        &mut kv,
                        &mut prefill_scratch,
                        &mut host_scratch,
                        prompt_tokens,
                        0,
                        config,
                        final_prompt_logits_mode,
                    ) {
                        Ok(token) => token,
                        Err(_err) => {
                            // Fallback to decode-style prompt processing
                            let mut prompt_next_token = None;
                            for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                                gpu::gpu_embed_token_hybrid(
                                    &device,
                                    token_id,
                                    &gpu_weights,
                                    &cpu_weights,
                                    &mut gpu_scratch,
                                    &mut host_scratch,
                                    config,
                                )
                                ?;
                                let logits_mode = if pos + 1 == prompt_tokens.len() {
                                    final_prompt_logits_mode
                                } else {
                                    gpu::GpuLogitsMode::Skip
                                };
                                prompt_next_token = gpu::gpu_full_forward_hybrid(
                                    &device,
                                    &gpu_weights,
                                    &cpu_weights,
                                    &mut kv,
                                    &mut gpu_scratch,
                                    &mut host_scratch,
                                    pos,
                                    config,
                                    logits_mode,
                                )
                                ?;
                            }
                            prompt_next_token
                        }
                    }
                }
                Err(_err) => {
                    // Fallback to decode-style prompt processing
                    let mut prompt_next_token = None;
                    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                        gpu::gpu_embed_token_hybrid(
                            &device,
                            token_id,
                            &gpu_weights,
                            &cpu_weights,
                            &mut gpu_scratch,
                            &mut host_scratch,
                            config,
                        )
                        ?;
                        let logits_mode = if pos + 1 == prompt_tokens.len() {
                            final_prompt_logits_mode
                        } else {
                            gpu::GpuLogitsMode::Skip
                        };
                        prompt_next_token = gpu::gpu_full_forward_hybrid(
                            &device,
                            &gpu_weights,
                            &cpu_weights,
                            &mut kv,
                            &mut gpu_scratch,
                            &mut host_scratch,
                            pos,
                            config,
                            logits_mode,
                        )
                        ?;
                    }
                    prompt_next_token
                }
            }
        }
        gpu::InferencePath::SvdOptimized => {
            let mut prompt_next_token = None;
            for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                gpu::gpu_embed_token_hybrid(
                    &device,
                    token_id,
                    &gpu_weights,
                    &cpu_weights,
                    &mut gpu_scratch,
                    &mut host_scratch,
                    config,
                )
                ?;
                let logits_mode = if pos + 1 == prompt_tokens.len() {
                    final_prompt_logits_mode
                } else {
                    gpu::GpuLogitsMode::Skip
                };
                prompt_next_token = gpu::gpu_full_forward_hybrid(
                    &device,
                    &gpu_weights,
                    &cpu_weights,
                    &mut kv,
                    &mut gpu_scratch,
                    &mut host_scratch,
                    pos,
                    config,
                    logits_mode,
                )
                ?;
            }
            prompt_next_token
        }
        gpu::InferencePath::DecodeStyle | gpu::InferencePath::CpuFallback { .. } => {
            let mut prompt_next_token = None;
            for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                gpu::gpu_embed_token_hybrid(
                    &device,
                    token_id,
                    &gpu_weights,
                    &cpu_weights,
                    &mut gpu_scratch,
                    &mut host_scratch,
                    config,
                )
                ?;
                let logits_mode = if pos + 1 == prompt_tokens.len() {
                    final_prompt_logits_mode
                } else {
                    gpu::GpuLogitsMode::Skip
                };
                prompt_next_token = gpu::gpu_full_forward_hybrid(
                    &device,
                    &gpu_weights,
                    &cpu_weights,
                    &mut kv,
                    &mut gpu_scratch,
                    &mut host_scratch,
                    pos,
                    config,
                    logits_mode,
                )
                ?;
            }
            prompt_next_token
        }
    };

    // ── 7. Decode loop ───────────────────────────────────────────────────────────
    let mut pos = prompt_tokens.len();
    let mut n_generated = 0usize;
    let mut seed = 0xdeadbeef_u64;

    let mut next_token = if use_greedy {
        if use_gpu_greedy_fastpath {
            prompt_next_token.expect("greedy GPU prompt pass should return next token")
        } else {
            cpu_sample_greedy(&host_scratch.logits)
        }
    } else {
        seed = seed.wrapping_add(1);
        cpu_sample_top_p(&host_scratch.logits, temperature, top_p, seed)
    };

    let mut output_tokens = Vec::with_capacity(max_tokens);

    loop {
        if tok.is_eog(next_token) || n_generated >= max_tokens || pos >= max_seq {
            break;
        }
        output_tokens.push(next_token);
        n_generated += 1;

        gpu::gpu_embed_token_hybrid(
            &device,
            next_token,
            &gpu_weights,
            &cpu_weights,
            &mut gpu_scratch,
            &mut host_scratch,
            config,
        )
        ?;
        let logits_mode = if use_gpu_greedy_fastpath {
            gpu::GpuLogitsMode::GreedyArgmax
        } else {
            gpu::GpuLogitsMode::DownloadToHost
        };
        let decode_next_token = gpu::gpu_full_forward_hybrid(
            &device,
            &gpu_weights,
            &cpu_weights,
            &mut kv,
            &mut gpu_scratch,
            &mut host_scratch,
            pos,
            config,
            logits_mode,
        )
        ?;
        pos += 1;

        next_token = if let Some(token) = decode_next_token {
            token
        } else {
            // SYNC POINT: wait for GPU forward + argmax download (non-graph path)
            device
                .synchronize()
                ?;

            if use_greedy {
                if use_gpu_greedy_fastpath {
                    let token = gpu_scratch.argmax_result_index.as_slice::<i32>()[0];
                    if token < 0 || (token as usize) >= config.vocab_size {
                        return Err(format!("gpu argmax returned out-of-range index {}", token));
                    }
                    token as u32
                } else {
                    cpu_sample_greedy(&host_scratch.logits)
                }
            } else {
                seed = seed.wrapping_add(1);
                cpu_sample_top_p(&host_scratch.logits, temperature, top_p, seed)
            }
        };
    }

    let text = tok.decode(&output_tokens, false);
    Ok((text, output_tokens.len()))
}

#[cfg(feature = "gpu")]
pub fn run_gpu_stream_inference(
    gpu_weights_arc: &Arc<crate::gpu::GpuModelWeights>,
    cpu_weights: &Arc<CpuModelWeights>,
    config: &ModelConfig,
    tok: &BpeTokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    tx: tokio::sync::mpsc::UnboundedSender<String>,
) -> crate::error::RocmForgeResult<()> {
    // ── 1. GPU detection & VRAM pre-flight ──────────────────────────────────────
    let gpu_caps = gpu::detect().ok_or("GPU requested but no AMD GPU detected")?;
    let vram_session = gpu::VramSession::new(gpu_caps.device_id)
        ?;

    let max_seq_estimate = config.max_seq_len.min(max_tokens + 2048);
    let kv_estimate = gpu::GpuKvCache::estimate_bytes(config, max_seq_estimate);
    let scratch_estimate = gpu::GpuForwardScratch::estimate_bytes(config);

    vram_session
        .check_fits(0, kv_estimate, scratch_estimate)
        ?;

    // ── 2. Device init ─────────────────────────────────────────────────────────
    let device =
        gpu::GpuDevice::get_or_init(gpu_caps.device_id)?;
    let gpu_weights = gpu_weights_arc.as_ref();

    // ── 3. Allocate KV cache and scratch ─────────────────────────────────────────
    let max_seq = (prompt_tokens.len() + max_tokens).min(config.max_seq_len);
    let mut kv = gpu::GpuKvCache::new(config, max_seq)?;
    let mut gpu_scratch =
        gpu::GpuForwardScratch::new(config)?;

    // Expert scratch sized for the largest gate/up/down expert dims.
    'expert_scratch: for layer in &gpu_weights.layers {
        let all_compressed = [
            layer.ffn_gate_compressed.as_ref(),
            layer.ffn_up_compressed.as_ref(),
            layer.ffn_down_compressed.as_ref(),
        ];
        if all_compressed.iter().all(|x| x.is_some()) {
            let k = layer
                .ffn_gate_compressed
                .as_ref()
                .map(|c| c.k)
                .unwrap_or(32);
            let max_rows = all_compressed
                .iter()
                .filter_map(|x| *x)
                .map(|c| c.rows)
                .max()
                .unwrap_or(1);
            let max_cols = all_compressed
                .iter()
                .filter_map(|x| *x)
                .map(|c| c.cols)
                .max()
                .unwrap_or(1);
            let max_nnz = all_compressed
                .iter()
                .filter_map(|x| *x)
                .map(|c| c.max_nnz())
                .max()
                .unwrap_or(1);
            gpu_scratch
                .init_expert_scratch(k as u32, max_rows, max_cols, max_nnz)
                ?;
            break 'expert_scratch;
        }
    }

    let mut host_scratch = CpuForwardScratch::new(config);
    let use_greedy = top_p >= 1.0;
    let use_gpu_greedy_fastpath = use_greedy;

    let final_prompt_logits_mode = if use_gpu_greedy_fastpath {
        gpu::GpuLogitsMode::GreedyArgmax
    } else {
        gpu::GpuLogitsMode::DownloadToHost
    };

    // ── 4. Hotpath router ────────────────────────────────────────────────────────
    let profile = gpu::ModelProfile::from_weights(gpu_weights, config);
    let path = gpu::select_path(&profile, prompt_tokens.len(), &vram_session);

    if let Err(_e) = gpu::check_path_vram(&path, config, prompt_tokens.len(), &vram_session) {
        // silently fall back to decode-style
    }

    // ── 5. Prefill ───────────────────────────────────────────────────────────────
    let prompt_next_token = match path {
        gpu::InferencePath::BatchedPrefill { .. } => {
            match gpu::GpuPrefillScratch::new(config, prompt_tokens.len()) {
                Ok(mut prefill_scratch) => {
                    match gpu::gpu_batched_prefill_forward_q4_0(
                        device,
                        gpu_weights,
                        cpu_weights,
                        &mut kv,
                        &mut prefill_scratch,
                        &mut host_scratch,
                        prompt_tokens,
                        0,
                        config,
                        final_prompt_logits_mode,
                    ) {
                        Ok(token) => token,
                        Err(_err) => {
                            // Fallback to decode-style prompt processing
                            let mut prompt_next_token = None;
                            for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                                gpu::gpu_embed_token_hybrid(
                                    device,
                                    token_id,
                                    gpu_weights,
                                    cpu_weights,
                                    &mut gpu_scratch,
                                    &mut host_scratch,
                                    config,
                                )
                                ?;
                                let logits_mode = if pos + 1 == prompt_tokens.len() {
                                    final_prompt_logits_mode
                                } else {
                                    gpu::GpuLogitsMode::Skip
                                };
                                prompt_next_token = gpu::gpu_full_forward_hybrid(
                                    device,
                                    gpu_weights,
                                    cpu_weights,
                                    &mut kv,
                                    &mut gpu_scratch,
                                    &mut host_scratch,
                                    pos,
                                    config,
                                    logits_mode,
                                )
                                ?;
                            }
                            prompt_next_token
                        }
                    }
                }
                Err(_err) => {
                    // Fallback to decode-style prompt processing
                    let mut prompt_next_token = None;
                    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                        gpu::gpu_embed_token_hybrid(
                            device,
                            token_id,
                            gpu_weights,
                            cpu_weights,
                            &mut gpu_scratch,
                            &mut host_scratch,
                            config,
                        )
                        ?;
                        let logits_mode = if pos + 1 == prompt_tokens.len() {
                            final_prompt_logits_mode
                        } else {
                            gpu::GpuLogitsMode::Skip
                        };
                        prompt_next_token = gpu::gpu_full_forward_hybrid(
                            device,
                            gpu_weights,
                            cpu_weights,
                            &mut kv,
                            &mut gpu_scratch,
                            &mut host_scratch,
                            pos,
                            config,
                            logits_mode,
                        )
                        ?;
                    }
                    prompt_next_token
                }
            }
        }
        gpu::InferencePath::SvdOptimized => {
            let mut prompt_next_token = None;
            for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                gpu::gpu_embed_token_hybrid(
                    device,
                    token_id,
                    gpu_weights,
                    cpu_weights,
                    &mut gpu_scratch,
                    &mut host_scratch,
                    config,
                )
                ?;
                let logits_mode = if pos + 1 == prompt_tokens.len() {
                    final_prompt_logits_mode
                } else {
                    gpu::GpuLogitsMode::Skip
                };
                prompt_next_token = gpu::gpu_full_forward_hybrid(
                    device,
                    gpu_weights,
                    cpu_weights,
                    &mut kv,
                    &mut gpu_scratch,
                    &mut host_scratch,
                    pos,
                    config,
                    logits_mode,
                )
                ?;
            }
            prompt_next_token
        }
        gpu::InferencePath::DecodeStyle | gpu::InferencePath::CpuFallback { .. } => {
            let mut prompt_next_token = None;
            for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                gpu::gpu_embed_token_hybrid(
                    device,
                    token_id,
                    gpu_weights,
                    cpu_weights,
                    &mut gpu_scratch,
                    &mut host_scratch,
                    config,
                )
                ?;
                let logits_mode = if pos + 1 == prompt_tokens.len() {
                    final_prompt_logits_mode
                } else {
                    gpu::GpuLogitsMode::Skip
                };
                prompt_next_token = gpu::gpu_full_forward_hybrid(
                    device,
                    gpu_weights,
                    cpu_weights,
                    &mut kv,
                    &mut gpu_scratch,
                    &mut host_scratch,
                    pos,
                    config,
                    logits_mode,
                )
                ?;
            }
            prompt_next_token
        }
    };

    // ── 6. Decode loop ───────────────────────────────────────────────────────────
    let mut pos = prompt_tokens.len();
    let mut n_generated = 0usize;
    let mut seed = 0xdeadbeef_u64;

    let mut next_token = if use_greedy {
        if use_gpu_greedy_fastpath {
            prompt_next_token.expect("greedy GPU prompt pass should return next token")
        } else {
            cpu_sample_greedy(&host_scratch.logits)
        }
    } else {
        seed = seed.wrapping_add(1);
        cpu_sample_top_p(&host_scratch.logits, temperature, top_p, seed)
    };

    let mut output_tokens = Vec::with_capacity(max_tokens);
    let mut previous_text = String::new();

    loop {
        if tok.is_eog(next_token) || n_generated >= max_tokens || pos >= max_seq {
            break;
        }
        output_tokens.push(next_token);
        n_generated += 1;

        // Decode incremental text and stream it over the channel
        let text = tok.decode(&output_tokens, false);
        let new_chars = &text[previous_text.len().min(text.len())..];
        if !new_chars.is_empty() {
            let _ = tx.send(new_chars.to_string());
        }
        previous_text = text;

        gpu::gpu_embed_token_hybrid(
            device,
            next_token,
            gpu_weights,
            cpu_weights,
            &mut gpu_scratch,
            &mut host_scratch,
            config,
        )
        ?;
        let logits_mode = if use_gpu_greedy_fastpath {
            gpu::GpuLogitsMode::GreedyArgmax
        } else {
            gpu::GpuLogitsMode::DownloadToHost
        };
        let decode_next_token = gpu::gpu_full_forward_hybrid(
            device,
            gpu_weights,
            cpu_weights,
            &mut kv,
            &mut gpu_scratch,
            &mut host_scratch,
            pos,
            config,
            logits_mode,
        )
        ?;
        pos += 1;

        next_token = if let Some(token) = decode_next_token {
            token
        } else {
            // SYNC POINT: wait for GPU forward + argmax download (non-graph path)
            device
                .synchronize()
                ?;

            if use_greedy {
                if use_gpu_greedy_fastpath {
                    let token = gpu_scratch.argmax_result_index.as_slice::<i32>()[0];
                    if token < 0 || (token as usize) >= config.vocab_size {
                        return Err(format!("gpu argmax returned out-of-range index {}", token));
                    }
                    token as u32
                } else {
                    cpu_sample_greedy(&host_scratch.logits)
                }
            } else {
                seed = seed.wrapping_add(1);
                cpu_sample_top_p(&host_scratch.logits, temperature, top_p, seed)
            }
        };
    }

    Ok(())
}
