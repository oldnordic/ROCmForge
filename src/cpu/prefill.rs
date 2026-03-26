//! CPU batched prefill — processes entire prompt in cache-sized batches.
//!
//! Uses GEMM (batched matrix multiply) for QKV projections instead of GEMV.
//! Populates KV cache for subsequent decode steps.
//!
//! Processing is split into batches that fit in L3 cache for optimal performance.
//! For multi-core systems, batches are distributed across physical cores.

use super::cache::{CpuForwardScratch, CpuKvCache};
use super::ops::{add_bias_batched, dispatch_gemm, dispatch_gemv, flash_attn_prefill, rms_norm, silu_fuse};
use super::quant::{embed_f32_batch, embed_q4_0_batch, embed_q4_1_batch, embed_q4_k_batch, embed_q5_0_batch, embed_q6_k_batch, embed_q8_0_batch};
use super::weights::{CpuLayerWeights, CpuModelWeights};
use super::CpuError;
use crate::config::ModelConfig;
use crate::loader::GgmlType;
use crate::hardware::BatchConfig;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::marker::PhantomData;

// Wrapper for Send-unsafe raw pointers
#[repr(C)]
struct SendPtr<T> {
    ptr: *mut T,
    _phantom: PhantomData<T>,
}

unsafe impl<T> Send for SendPtr<T> {}

// ── PrefillScratch ────────────────────────────────────────────────────────────

/// Scratch buffers for batched prefill.
///
/// All buffers are sized for [batch_len, dim] layout.
struct CpuPrefillScratch {
    /// Hidden states [batch_len * hidden_size]
    hidden: Vec<f32>,
    /// Normalized hidden [batch_len * hidden_size]
    normed: Vec<f32>,
    /// Query [batch_len * q_size]
    q: Vec<f32>,
    /// Key [batch_len * kv_size]
    k: Vec<f32>,
    /// Value [batch_len * kv_size]
    v: Vec<f32>,
    /// Attention output [batch_len * q_size]
    attn_out: Vec<f32>,
    /// Layer output [batch_len * hidden_size]
    layer_out: Vec<f32>,
    /// FFN gate [batch_len * intermediate_size]
    gate: Vec<f32>,
    /// FFN SwiGLU [batch_len * intermediate_size]
    swiglu: Vec<f32>,
    /// Q8_0 scratch buffer for GEMV quantization
    q8_scratch: Vec<u8>,
}

impl CpuPrefillScratch {
    fn new(config: &ModelConfig, batch_len: usize) -> Self {
        let n = batch_len;
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;

        // Q8_0 scratch buffer
        use super::quant::Q8_BLOCK_ELEMS;
        use super::quant::Q8_BLOCK_BYTES;
        let num_blocks = h / Q8_BLOCK_ELEMS;
        let q8_scratch = vec![0u8; num_blocks * Q8_BLOCK_BYTES];

        Self {
            hidden: vec![0.0; n * h],
            normed: vec![0.0; n * h],
            q: vec![0.0; n * q],
            k: vec![0.0; n * kv],
            v: vec![0.0; n * kv],
            attn_out: vec![0.0; n * q],
            layer_out: vec![0.0; n * h],
            gate: vec![0.0; n * ff],
            swiglu: vec![0.0; n * ff],
            q8_scratch,
        }
    }
}

// ── CpuParallelPrefillScratch ───────────────────────────────────────────────

/// Scratch buffers for parallel batched prefill.
///
/// Each thread gets its own set of buffers to avoid contention.
/// All buffers are sized for [sub_batch_len, dim] layout where
/// sub_batch_len = batch_len / num_cores.
struct CpuParallelPrefillScratch {
    /// Per-thread hidden states [num_threads][sub_batch_len * hidden_size]
    per_thread_hidden: Vec<Vec<f32>>,
    /// Per-thread normalized hidden [num_threads][sub_batch_len * hidden_size]
    per_thread_normed: Vec<Vec<f32>>,
    /// Per-thread query [num_threads][sub_batch_len * q_size]
    per_thread_q: Vec<Vec<f32>>,
    /// Per-thread key [num_threads][sub_batch_len * kv_size]
    per_thread_k: Vec<Vec<f32>>,
    /// Per-thread value [num_threads][sub_batch_len * kv_size]
    per_thread_v: Vec<Vec<f32>>,
    /// Per-thread attention output [num_threads][sub_batch_len * q_size]
    per_thread_attn_out: Vec<Vec<f32>>,
    /// Per-thread layer output [num_threads][sub_batch_len * hidden_size]
    per_thread_layer_out: Vec<Vec<f32>>,
    /// Per-thread FFN gate [num_threads][sub_batch_len * intermediate_size]
    per_thread_gate: Vec<Vec<f32>>,
    /// Per-thread FFN SwiGLU [num_threads][sub_batch_len * intermediate_size]
    per_thread_swiglu: Vec<Vec<f32>>,
    /// Per-thread Q8_0 scratch buffers for GEMV quantization
    per_thread_q8_scratch: Vec<Vec<u8>>,
    /// Number of threads (should be num_cores)
    num_threads: usize,
}

impl CpuParallelPrefillScratch {
    fn new(config: &ModelConfig, num_threads: usize) -> Self {
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;

        // All threads get same sub-batch size
        let sub_batch_len = 1; // Will be set dynamically per batch
        let n = sub_batch_len;

        // Q8_0 scratch buffer size
        use super::quant::Q8_BLOCK_ELEMS;
        use super::quant::Q8_BLOCK_BYTES;
        let num_blocks = h / Q8_BLOCK_ELEMS;
        let q8_scratch_size = num_blocks * Q8_BLOCK_BYTES;

        Self {
            per_thread_hidden: (0..num_threads).map(|_| vec![0.0; n * h]).collect(),
            per_thread_normed: (0..num_threads).map(|_| vec![0.0; n * h]).collect(),
            per_thread_q: (0..num_threads).map(|_| vec![0.0; n * q]).collect(),
            per_thread_k: (0..num_threads).map(|_| vec![0.0; n * kv]).collect(),
            per_thread_v: (0..num_threads).map(|_| vec![0.0; n * kv]).collect(),
            per_thread_attn_out: (0..num_threads).map(|_| vec![0.0; n * q]).collect(),
            per_thread_layer_out: (0..num_threads).map(|_| vec![0.0; n * h]).collect(),
            per_thread_gate: (0..num_threads).map(|_| vec![0.0; n * ff]).collect(),
            per_thread_swiglu: (0..num_threads).map(|_| vec![0.0; n * ff]).collect(),
            per_thread_q8_scratch: (0..num_threads).map(|_| vec![0u8; q8_scratch_size]).collect(),
            num_threads,
        }
    }

    /// Resize all per-thread buffers for a specific sub-batch length.
    fn resize_for_batch(&mut self, sub_batch_len: usize, config: &ModelConfig) {
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;

        // Q8_0 scratch buffer size (doesn't depend on batch_len)
        use super::quant::Q8_BLOCK_ELEMS;
        use super::quant::Q8_BLOCK_BYTES;
        let num_blocks = h / Q8_BLOCK_ELEMS;
        let q8_scratch_size = num_blocks * Q8_BLOCK_BYTES;

        for i in 0..self.num_threads {
            if self.per_thread_hidden[i].len() != sub_batch_len * h {
                self.per_thread_hidden[i] = vec![0.0; sub_batch_len * h];
                self.per_thread_normed[i] = vec![0.0; sub_batch_len * h];
                self.per_thread_q[i] = vec![0.0; sub_batch_len * q];
                self.per_thread_k[i] = vec![0.0; sub_batch_len * kv];
                self.per_thread_v[i] = vec![0.0; sub_batch_len * kv];
                self.per_thread_attn_out[i] = vec![0.0; sub_batch_len * q];
                self.per_thread_layer_out[i] = vec![0.0; sub_batch_len * h];
                self.per_thread_gate[i] = vec![0.0; sub_batch_len * ff];
                self.per_thread_swiglu[i] = vec![0.0; sub_batch_len * ff];
                // Q8_0 scratch buffer size is fixed, only need to ensure it exists
                if self.per_thread_q8_scratch[i].len() != q8_scratch_size {
                    self.per_thread_q8_scratch[i] = vec![0u8; q8_scratch_size];
                }
            }
        }
    }
}

// ── prefill_layer_forward ─────────────────────────────────────────────────────

fn prefill_layer_forward(
    ps: &mut CpuPrefillScratch,
    weights: &CpuLayerWeights,
    kv: &mut CpuKvCache,
    layer: usize,
    start_pos: usize,
    config: &ModelConfig,
    batch_len: usize,
) -> Result<(), CpuError> {
    let h = config.hidden_size;
    let q_s = config.num_heads * config.head_dim;
    let kv_s = config.num_kv_heads * config.head_dim;
    let ff = config.intermediate_size;
    let eps = config.rms_norm_eps;

    // 1. RMS norm each row: normed[s, :] = rms_norm(hidden[s, :], attn_norm)
    for s in 0..batch_len {
        let xr = &ps.hidden[s * h..(s + 1) * h];
        let or = &mut ps.normed[s * h..(s + 1) * h];
        rms_norm(xr, &weights.attn_norm, or, eps);
    }

    // 2. QKV GEMM
    dispatch_gemm(
        &weights.attn_q,
        &weights.attn_q_meta,
        &ps.normed,
        &mut ps.q,
        q_s,
        h,
    )?;
    dispatch_gemm(
        &weights.attn_k,
        &weights.attn_k_meta,
        &ps.normed,
        &mut ps.k,
        kv_s,
        h,
    )?;
    dispatch_gemm(
        &weights.attn_v,
        &weights.attn_v_meta,
        &ps.normed,
        &mut ps.v,
        kv_s,
        h,
    )?;

    // 3. Optional biases
    if let Some(bq) = &weights.attn_q_bias {
        add_bias_batched(&mut ps.q, bq, q_s, batch_len);
    }
    if let Some(bk) = &weights.attn_k_bias {
        add_bias_batched(&mut ps.k, bk, kv_s, batch_len);
    }
    if let Some(bv) = &weights.attn_v_bias {
        add_bias_batched(&mut ps.v, bv, kv_s, batch_len);
    }

    // 4. RoPE batched
    let row_len = config.num_heads * config.head_dim;
    for s in 0..batch_len {
        let qr = &mut ps.q[s * row_len..(s + 1) * row_len];
        super::ops::rope(
            qr,
            config.num_heads,
            config.head_dim,
            start_pos + s,
            config.rope_theta,
            config.rope_neox,
        );
    }
    let kv_row_len = config.num_kv_heads * config.head_dim;
    for s in 0..batch_len {
        let kr = &mut ps.k[s * kv_row_len..(s + 1) * kv_row_len];
        super::ops::rope(
            kr,
            config.num_kv_heads,
            config.head_dim,
            start_pos + s,
            config.rope_theta,
            config.rope_neox,
        );
    }

    // 5. KV write batched
    for s in 0..batch_len {
        let pos = start_pos + s;
        let k_row = &ps.k[s * kv_s..(s + 1) * kv_s];
        let v_row = &ps.v[s * kv_s..(s + 1) * kv_s];
        kv.write_k(layer, pos, k_row);
        kv.write_v(layer, pos, v_row);
    }

    // 6. Causal flash attention (prefill: Q, K, V are all [batch_len, ...])
    flash_attn_prefill(
        &ps.q,
        &ps.k,
        &ps.v,
        &mut ps.attn_out,
        batch_len,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
    );

    // 7. O projection GEMM
    dispatch_gemm(
        &weights.attn_o,
        &weights.attn_o_meta,
        &ps.attn_out,
        &mut ps.layer_out,
        h,
        q_s,
    )?;

    // 8. Residual: hidden += layer_out
    for i in 0..batch_len * h {
        ps.hidden[i] += ps.layer_out[i];
    }

    // 9. Post-attention RMS norm batched
    for s in 0..batch_len {
        let xr = &ps.hidden[s * h..(s + 1) * h];
        let or = &mut ps.normed[s * h..(s + 1) * h];
        rms_norm(xr, &weights.ffn_norm, or, eps);
    }

    // 10. MLP: gate + up projections
    dispatch_gemm(
        &weights.ffn_gate,
        &weights.ffn_gate_meta,
        &ps.normed,
        &mut ps.gate,
        ff,
        h,
    )?;
    dispatch_gemm(
        &weights.ffn_up,
        &weights.ffn_up_meta,
        &ps.normed,
        &mut ps.swiglu,
        ff,
        h,
    )?;

    // 11. SwiGLU: silu(gate) * up
    silu_fuse(&ps.gate, &mut ps.swiglu);

    // 12. Down projection GEMM - uses metadata to determine if transposition is needed
    dispatch_gemm(
        &weights.ffn_down,
        &weights.ffn_down_meta,
        &ps.swiglu,
        &mut ps.layer_out,
        h,  // out_dim (hidden_size)
        ff,  // in_dim (intermediate_size)
    )?;

    // 13. Residual: hidden += layer_out
    for i in 0..batch_len * h {
        ps.hidden[i] += ps.layer_out[i];
    }

    Ok(())
}

// ── cpu_prefill_forward ───────────────────────────────────────────────────────

/// Process entire prompt in cache-sized batches.
///
/// Splits the prompt into batches that fit in L3 cache for optimal performance.
/// Each batch is processed through all layers sequentially.
///
/// Fills `scratch.logits` with the next-token prediction from the last prompt position.
/// KV cache populated for positions start_pos..start_pos+tokens.len().
///
/// # Arguments
/// * `tokens` - Prompt token IDs
/// * `weights` - Model weights
/// * `kv` - KV cache (will be populated)
/// * `scratch` - Scratch buffers for final output
/// * `start_pos` - Starting position in KV cache (0 for new conversation)
/// * `config` - Model configuration
/// * `batch_config` - Batch configuration (max tokens per batch)
pub fn cpu_prefill_forward(
    tokens: &[u32],
    weights: &CpuModelWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    start_pos: usize,
    config: &ModelConfig,
    batch_config: &BatchConfig,
) -> Result<(), CpuError> {
    let seq_len = tokens.len();
    assert!(seq_len > 0, "cpu_prefill_forward: empty token list");
    assert!(
        start_pos + seq_len <= kv.max_seq_len,
        "prompt longer than KV cache"
    );

    let h = config.hidden_size;
    let batch_size = batch_config.max_tokens_per_batch;

    // Process in batches to keep working set in L3 cache
    for batch_start in (0..seq_len).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(seq_len);
        let batch_tokens = &tokens[batch_start..batch_end];
        let batch_len = batch_tokens.len();
        let batch_pos = start_pos + batch_start;

        let mut ps = CpuPrefillScratch::new(config, batch_len);

        // 1. Gather embeddings for this batch
        match weights.token_emb_meta.wtype {
            GgmlType::F32 => {
                let wf: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        weights.token_emb.as_ptr() as *const f32,
                        weights.token_emb.len() / 4,
                    )
                };
                embed_f32_batch(batch_tokens, wf, &mut ps.hidden, h);
            }
            GgmlType::Q4_0 => {
                embed_q4_0_batch(batch_tokens, &weights.token_emb, &mut ps.hidden, h);
            }
            GgmlType::Q4_K => {
                embed_q4_k_batch(batch_tokens, &weights.token_emb, &mut ps.hidden, h);
            }
            GgmlType::Q5_0 => {
                embed_q5_0_batch(batch_tokens, &weights.token_emb, &mut ps.hidden, h);
            }
            GgmlType::Q6_K => {
                embed_q6_k_batch(batch_tokens, &weights.token_emb, &mut ps.hidden, h);
            }
            GgmlType::Q8_0 => {
                embed_q8_0_batch(batch_tokens, &weights.token_emb, &mut ps.hidden, h);
            }
            other => {
                return Err(CpuError::UnsupportedWeightType(other));
            }
        }

        // 2. All layers for this batch
        for layer_idx in 0..config.num_layers {
            prefill_layer_forward(
                &mut ps,
                weights.layer(layer_idx),
                kv,
                layer_idx,
                batch_pos,
                config,
                batch_len,
            )?;
        }

        // 3. For the final batch, extract last row for logits
        if batch_end == seq_len {
            let last_row = &ps.hidden[(batch_len - 1) * h..batch_len * h];

            // Final RMS norm on last row
            rms_norm(
                last_row,
                &weights.output_norm,
                &mut scratch.normed,
                config.rms_norm_eps,
            );

            // LM head GEMV (use decode path for single token)
            let v = config.vocab_size;
            // Use metadata to automatically select correct kernel (regular or transposed)
            super::ops::dispatch_gemv(
                &weights.lm_head,
                &weights.lm_head_meta,
                &scratch.normed,
                &mut scratch.logits,
                v,  // vocab_size (out_dim)
                h,  // hidden_size (in_dim)
                Some(&mut scratch.q8_scratch),
            )?;
        }
    }

    Ok(())
}

// ── cpu_prefill_forward_parallel ─────────────────────────────────────────────

/// Process entire prompt with inter-batch parallel execution.
///
/// Uses parallel processing for multiple independent batches when sequence length allows.
/// Each batch goes through all layers sequentially.
///
/// Batches are independent due to causal masking:
/// - Batch 0 (tokens 0-127) only needs attention within itself
/// - Batch 1 (tokens 128-255) reads Batch 0's KV cache (already written)
/// - Therefore, batches can run in parallel if their position ranges don't overlap
///
/// For sequences shorter than 2*batch_size, falls back to sequential processing.
///
/// # Safety
///
/// Different batches write to disjoint KV cache positions, so concurrent writes are safe
/// without mutex overhead. The KV cache is passed as immutable reference since
/// each thread writes to different [layer][pos] ranges.
///
/// # Arguments
/// * `tokens` - Prompt token IDs
/// * `weights` - Model weights (shared immutable)
/// * `kv` - KV cache (will be populated with concurrent writes)
/// * `scratch` - Scratch buffers for final output
/// * `start_pos` - Starting position in KV cache (0 for new conversation)
/// * `config` - Model configuration
/// * `batch_config` - Batch configuration (max tokens per batch, num cores)
pub fn cpu_prefill_forward_parallel(
    tokens: &[u32],
    weights: &CpuModelWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    start_pos: usize,
    config: &ModelConfig,
    batch_config: &BatchConfig,
) -> Result<(), CpuError> {
    let seq_len = tokens.len();
    assert!(seq_len > 0, "cpu_prefill_forward_parallel: empty token list");
    assert!(
        start_pos + seq_len <= kv.max_seq_len,
        "prompt longer than KV cache"
    );

    let h = config.hidden_size;
    let batch_size = batch_config.max_tokens_per_batch;
    let num_cores = batch_config.num_cores;

    // Calculate batch start positions
    let batch_starts: Vec<usize> = (0..seq_len).step_by(batch_size).collect();

    // For short sequences (< 2 batches), sequential is faster (less threading overhead)
    if batch_starts.len() < 2 {
        // Fall back to sequential processing for better performance
        return cpu_prefill_forward(tokens, weights, kv, scratch, start_pos, config, batch_config);
    }

    // Wrap KV cache pointer in Arc<Mutex<SendPtr<CpuKvCache>>> for shared access
    // SAFETY: Each thread writes to disjoint KV positions (different batch ranges).
    // Raw pointer is used because we can't safely share &mut across threads.
    // Mutex ensures only one thread accesses at a time.
    // SendPtr implements Send, allowing the pointer to be shared across threads.
    let kv_ptr: *mut CpuKvCache = kv as *const _ as *mut CpuKvCache;
    let kv_shared: Arc<Mutex<SendPtr<CpuKvCache>>> = Arc::new(Mutex::new(SendPtr {
        ptr: kv_ptr,
        _phantom: PhantomData,
    }));

    // Wrap weights and config in Arc for shared immutable access
    let weights_shared = Arc::new(weights);
    let config_shared = Arc::new(config);

    // Process batches in parallel using rayon
    // Each batch is independent - different KV positions, causal masking handles inter-batch attention
    let results: Vec<Result<(Vec<f32>, usize), String>> = batch_starts
        .par_iter()
        .enumerate()
        .map(|(batch_idx, &batch_start)| -> Result<(Vec<f32>, usize), String> {
            let batch_end = (batch_start + batch_size).min(seq_len);
            let batch_tokens = &tokens[batch_start..batch_end];
            let batch_len = batch_tokens.len();
            let batch_pos = start_pos + batch_start;

            // Create scratch for this batch
            let mut ps = CpuPrefillScratch::new(&config_shared, batch_len);

            // 1. Gather embeddings for this batch
            match weights_shared.token_emb_meta.wtype {
                GgmlType::F32 => {
                    let wf: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            weights_shared.token_emb.as_ptr() as *const f32,
                            weights_shared.token_emb.len() / 4,
                        )
                    };
                    embed_f32_batch(batch_tokens, wf, &mut ps.hidden, h);
                }
                GgmlType::Q4_0 => {
                    embed_q4_0_batch(batch_tokens, &weights_shared.token_emb, &mut ps.hidden, h);
                }
                GgmlType::Q4_K => {
                    embed_q4_k_batch(batch_tokens, &weights_shared.token_emb, &mut ps.hidden, h);
                }
                GgmlType::Q5_0 => {
                    embed_q5_0_batch(batch_tokens, &weights_shared.token_emb, &mut ps.hidden, h);
                }
                GgmlType::Q6_K => {
                    embed_q6_k_batch(batch_tokens, &weights_shared.token_emb, &mut ps.hidden, h);
                }
                GgmlType::Q8_0 => {
                    embed_q8_0_batch(batch_tokens, &weights_shared.token_emb, &mut ps.hidden, h);
                }
                other => {
                    return Err(format!("Unsupported weight type: {:?}", other));
                }
            }

            // 2. All layers for this batch
            // SAFETY: Each batch writes to disjoint KV positions: [layer, pos_start..pos_end]
            // Lock KV cache briefly for writes (disjoint positions, minimal contention)
            for layer_idx in 0..config_shared.num_layers {
                let mut kv_guard = kv_shared.lock()
                    .map_err(|e| format!("KV cache lock failed: {}", e))?;
                // SAFETY: Get mutable reference from UnsafeCell
                // Safe because we hold the lock
                // SAFETY: Dereference raw pointer to get &mut CpuKvCache
                // Safe because we hold the mutex lock and only this thread accesses
                let kv_ref: &mut CpuKvCache = unsafe { &mut *kv_guard.ptr };
                prefill_layer_forward(
                    &mut ps,
                    weights_shared.layer(layer_idx),
                    kv_ref,
                    layer_idx,
                    batch_pos,
                    &config_shared,
                    batch_len,
                ).map_err(|e| format!("Layer {} failed: {}", layer_idx, e))?;
            }

            // 3. Extract hidden state if this is the last batch
            let hidden_state = if batch_end == seq_len {
                let last_row_start = (batch_len - 1) * h;
                let last_row_end = batch_len * h;
                ps.hidden[last_row_start..last_row_end].to_vec()
            } else {
                Vec::new() // Not last batch, no need to keep hidden state
            };

            Ok((hidden_state, batch_idx))
        })
        .collect();

    // Check for errors
    for result in &results {
        if let Err(e) = result {
            return Err(CpuError::InvalidOperation(format!("Parallel batch failed: {}", e)));
        }
    }

    // Find the last batch (highest batch_idx) and extract its logits
    let last_batch_idx = results.iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|(_, idx)| *idx)
        .max()
        .ok_or_else(|| CpuError::InvalidOperation("No batches processed".to_string()))?;

    // Re-run the last batch to populate scratch.logits
    // (We could modify prefill_layer_forward to return hidden state, but this is simpler)
    let batch_start = batch_starts[last_batch_idx];
    let batch_end = (batch_start + batch_size).min(seq_len);
    let batch_tokens = &tokens[batch_start..batch_end];
    let batch_len = batch_tokens.len();
    let batch_pos = start_pos + batch_start;

    let mut ps_last = CpuPrefillScratch::new(config, batch_len);

    // Embed last batch
    match weights.token_emb_meta.wtype {
        GgmlType::F32 => {
            let wf: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    weights.token_emb.as_ptr() as *const f32,
                    weights.token_emb.len() / 4,
                )
            };
            embed_f32_batch(batch_tokens, wf, &mut ps_last.hidden, h);
        }
        GgmlType::Q4_0 => {
            embed_q4_0_batch(batch_tokens, &weights.token_emb, &mut ps_last.hidden, h);
        }
        GgmlType::Q4_1 => {
            super::quant::embed_q4_1_batch(batch_tokens, &weights.token_emb, &mut ps_last.hidden, h);
        }
        GgmlType::Q4_K => {
            embed_q4_k_batch(batch_tokens, &weights.token_emb, &mut ps_last.hidden, h);
        }
        GgmlType::Q5_0 => {
            embed_q5_0_batch(batch_tokens, &weights.token_emb, &mut ps_last.hidden, h);
        }
        GgmlType::Q6_K => {
            embed_q6_k_batch(batch_tokens, &weights.token_emb, &mut ps_last.hidden, h);
        }
        GgmlType::Q8_0 => {
            embed_q8_0_batch(batch_tokens, &weights.token_emb, &mut ps_last.hidden, h);
        }
        other => {
            return Err(CpuError::UnsupportedWeightType(other));
        }
    }

    // Last batch through all layers
    for layer_idx in 0..config.num_layers {
        prefill_layer_forward(
            &mut ps_last,
            weights.layer(layer_idx),
            kv,
            layer_idx,
            batch_pos,
            config,
            batch_len,
        )?;
    }

    // Extract last row and compute logits
    let last_row = &ps_last.hidden[(batch_len - 1) * h..batch_len * h];
    rms_norm(
        last_row,
        &weights.output_norm,
        &mut scratch.normed,
        config.rms_norm_eps,
    );
    dispatch_gemv(
        &weights.lm_head,
        &weights.lm_head_meta,
        &scratch.normed,
        &mut scratch.logits,
        config.vocab_size,
        h,
        Some(&mut scratch.q8_scratch),
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config() -> ModelConfig {
        ModelConfig {
            num_layers: 2,
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            vocab_size: 100,
            max_seq_len: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_neox: true,
            use_attention_bias: false,
            attention_layout: crate::config::AttentionLayout::SplitQkv,
            architecture: "qwen2".to_string(),
        }
    }

    #[test]
    fn prefill_scratch_sizes() {
        let config = make_test_config();
        let ps = CpuPrefillScratch::new(&config, 4);
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;

        assert_eq!(ps.hidden.len(), 4 * h);
        assert_eq!(ps.normed.len(), 4 * h);
        assert_eq!(ps.q.len(), 4 * q);
        assert_eq!(ps.k.len(), 4 * kv);
        assert_eq!(ps.v.len(), 4 * kv);
        assert_eq!(ps.attn_out.len(), 4 * q);
        assert_eq!(ps.layer_out.len(), 4 * h);
        assert_eq!(ps.gate.len(), 4 * ff);
        assert_eq!(ps.swiglu.len(), 4 * ff);
    }

    #[test]
    fn prefill_scratch_with_batch_size_one() {
        let config = make_test_config();
        let ps = CpuPrefillScratch::new(&config, 1);
        let h = config.hidden_size;

        assert_eq!(ps.hidden.len(), h);
        assert_eq!(ps.normed.len(), h);
    }

    #[test]
    fn prefill_scratch_with_large_batch() {
        let config = make_test_config();
        let ps = CpuPrefillScratch::new(&config, 256);
        let h = config.hidden_size;

        assert_eq!(ps.hidden.len(), 256 * h);
    }
}
