use crate::config::ModelConfig;
use crate::gpu::cache::{GpuForwardScratch, GpuKvCache};
use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernels::attention::{
    flash_attn_decode_strided_multi_head_from_state_on_stream,
    flash_attn_decode_strided_multi_head_on_stream, flash_attn_decode_turboquant,
};

/// Emit attention edges from a downloaded [num_heads, seq_len] weight matrix.
fn record_attention_edges_from_weights(
    recorder: &mut crate::cpu::forward_graph_trace::ForwardGraphRecorder,
    weights: &[f32],
    k_prefix: &[f32],
    v_prefix: &[f32],
    layer: usize,
    pos: usize,
    seq_len: usize,
    num_heads: usize,
    kv_stride: usize,
) {
    let threshold = recorder.attention_threshold();
    for h in 0..num_heads {
        let base = h * seq_len;
        for t in 0..seq_len {
            let weight = weights[base + t];
            if weight > threshold {
                let k = &k_prefix[t * kv_stride..(t + 1) * kv_stride];
                let v = &v_prefix[t * kv_stride..(t + 1) * kv_stride];
                recorder.ensure_kv_nodes(layer, t, k, v);
                recorder.record_attention_edge(Some(h), layer, pos, t, weight);
            }
        }
    }
}

pub(in crate::gpu::forward) fn gpu_attention_decode(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    kv: &GpuKvCache,
    layer_idx: usize,
    pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let seq_len = pos + 1;
    let num_q_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads_for_layer(layer_idx);
    let head_dim = config.head_dim_for_layer(layer_idx);
    let kv_size = config.kv_size(layer_idx);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_cache = kv.k_ptr(layer_idx)? as *const f32;
    let v_cache = kv.v_ptr(layer_idx)? as *const f32;
    let q_base = scratch.q.as_ptr() as *const f32;
    let out_base = scratch.attn_out.as_ptr() as *mut f32;

    let kv_lora_dim = kv.kv_lora_dim.unwrap_or(0);
    let w_up_k = kv
        .w_up_k
        .as_ref()
        .map(|w| w[layer_idx].as_ptr() as *const f32)
        .unwrap_or(std::ptr::null());
    let w_up_v = kv
        .w_up_v
        .as_ref()
        .map(|w| w[layer_idx].as_ptr() as *const f32)
        .unwrap_or(std::ptr::null());

    let trace_active = scratch.forward_graph_recorder().is_some();

    if let Some(bits) = kv.kv_quant_bits {
        if trace_active {
            return Err(GpuError::UnsupportedOperation {
                operation: "forward graph trace".to_string(),
                reason: "TurboQuant KV cache does not expose attention weights for tracing"
                    .to_string(),
            });
        }
        let centroids = kv.centroids_ptr()?;
        let num_centroids = 1 << bits;
        flash_attn_decode_turboquant(
            out_base,
            q_base,
            k_cache as *const u8,
            v_cache as *const u8,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            kv_lora_dim,
            bits as i32,
            num_centroids,
            centroids,
            kv.qjl_scale,
            w_up_k,
            w_up_v,
            device.stream(),
        )
    } else {
        let attn_weights_ptr = if trace_active {
            scratch.ensure_attn_weights(config)?
        } else {
            std::ptr::null_mut()
        };
        flash_attn_decode_strided_multi_head_on_stream(
            out_base,
            attn_weights_ptr,
            q_base,
            k_cache,
            v_cache,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            kv_size,                              // Pass per-layer cache stride
            scale,
            kv_lora_dim,
            kv.adastate_anchors_enabled,
            w_up_k,
            w_up_v,
            device.stream(),
        )?;

        if trace_active {
            device.synchronize()?;
            let weights = {
                let buf = scratch
                    .attn_weights_buf()
                    .ok_or_else(|| GpuError::InvalidOperation {
                        message:
                            "decode_attention_q4_0: attention trace requested without attn_weights buffer"
                                .to_string(),
                    })?;
                buf.copy_to_host_vec()?
            };
            let weights = &weights[..num_q_heads * seq_len];
            let kv_stride = kv.kv_lora_dim.unwrap_or(kv.kv_size);
            let mut k_prefix = vec![0.0f32; seq_len * kv_stride];
            let mut v_prefix = vec![0.0f32; seq_len * kv_stride];
            kv.copy_kv_prefix_to_host(layer_idx, seq_len, &mut k_prefix, &mut v_prefix)?;
            if let Some(recorder) = scratch.forward_graph_recorder() {
                record_attention_edges_from_weights(
                    recorder,
                    weights,
                    &k_prefix,
                    &v_prefix,
                    layer_idx,
                    pos,
                    seq_len,
                    num_q_heads,
                    kv_stride,
                );
            }
        }
        Ok(())
    }
}

pub(in crate::gpu::forward) fn gpu_attention_decode_from_state(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    kv: &GpuKvCache,
    layer_idx: usize,
    pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let num_q_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads_for_layer(layer_idx);
    let head_dim = config.head_dim_for_layer(layer_idx);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_cache = kv.k_ptr(layer_idx)? as *const f32;
    let v_cache = kv.v_ptr(layer_idx)? as *const f32;
    let q_base = scratch.q.as_ptr() as *const f32;
    let out_base = scratch.attn_out.as_ptr() as *mut f32;

    let kv_lora_dim = kv.kv_lora_dim.unwrap_or(0);
    let w_up_k = kv
        .w_up_k
        .as_ref()
        .map(|w| w[layer_idx].as_ptr() as *const f32)
        .unwrap_or(std::ptr::null());
    let w_up_v = kv
        .w_up_v
        .as_ref()
        .map(|w| w[layer_idx].as_ptr() as *const f32)
        .unwrap_or(std::ptr::null());

    let trace_active = scratch.forward_graph_recorder().is_some();

    if kv.kv_quant_bits.is_some() {
        if trace_active {
            return Err(GpuError::UnsupportedOperation {
                operation: "forward graph trace".to_string(),
                reason: "TurboQuant KV cache does not expose attention weights for tracing"
                    .to_string(),
            });
        }
        let seq_len = scratch.decode_state_next_pos().unwrap_or(0) + 1;
        let centroids = kv.centroids_ptr()?;
        let bits = kv.kv_quant_bits.unwrap_or(3);
        let num_centroids = 1 << bits;
        flash_attn_decode_turboquant(
            out_base,
            q_base,
            k_cache as *const u8,
            v_cache as *const u8,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            kv_lora_dim,
            bits as i32,
            num_centroids,
            centroids,
            kv.qjl_scale,
            w_up_k,
            w_up_v,
            device.stream(),
        )
    } else {
        let attn_weights_ptr = if trace_active {
            scratch.ensure_attn_weights(config)?
        } else {
            std::ptr::null_mut()
        };
        let kv_size = config.kv_size(layer_idx);
        flash_attn_decode_strided_multi_head_from_state_on_stream(
            out_base,
            attn_weights_ptr,
            q_base,
            k_cache,
            v_cache,
            scratch.decode_seq_len_ptr(),
            num_q_heads,
            num_kv_heads,
            head_dim,
            kv_size,                              // Pass per-layer cache stride
            scale,
            kv_lora_dim,
            kv.adastate_anchors_enabled,
            w_up_k,
            w_up_v,
            device.stream(),
        )?;

        if trace_active {
            device.synchronize()?;
            let seq_len = pos + 1;
            let weights = {
                let buf = scratch
                    .attn_weights_buf()
                    .ok_or_else(|| GpuError::InvalidOperation {
                        message:
                            "prefill_attention_q4_0: attention trace requested without attn_weights buffer"
                                .to_string(),
                    })?;
                buf.copy_to_host_vec()?
            };
            let weights = &weights[..num_q_heads * seq_len];
            let kv_stride = kv.kv_lora_dim.unwrap_or(kv.kv_size);
            let mut k_prefix = vec![0.0f32; seq_len * kv_stride];
            let mut v_prefix = vec![0.0f32; seq_len * kv_stride];
            kv.copy_kv_prefix_to_host(layer_idx, seq_len, &mut k_prefix, &mut v_prefix)?;
            if let Some(recorder) = scratch.forward_graph_recorder() {
                record_attention_edges_from_weights(
                    recorder,
                    weights,
                    &k_prefix,
                    &v_prefix,
                    layer_idx,
                    pos,
                    seq_len,
                    num_q_heads,
                    kv_stride,
                );
            }
        }
        Ok(())
    }
}
