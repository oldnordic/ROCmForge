use crate::gpu::cache::{GpuForwardScratch, GpuKvCache};
use crate::gpu::device::GpuDevice;
use crate::gpu::error::GpuResult;
use crate::gpu::kernels::attention::{
    flash_attn_decode_strided_multi_head_from_state_on_stream,
    flash_attn_decode_strided_multi_head_on_stream, flash_attn_decode_turboquant,
};

pub(in crate::gpu::forward) fn gpu_attention_decode(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    kv: &GpuKvCache,
    layer_idx: usize,
    pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> GpuResult<()> {
    let seq_len = pos + 1;
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

    if let Some(bits) = kv.kv_quant_bits {
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
        flash_attn_decode_strided_multi_head_on_stream(
            out_base,
            q_base,
            k_cache,
            v_cache,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            kv_lora_dim,
            kv.adastate_anchors_enabled,
            w_up_k,
            w_up_v,
            device.stream(),
        )
    }
}

pub(in crate::gpu::forward) fn gpu_attention_decode_from_state(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    kv: &GpuKvCache,
    layer_idx: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> GpuResult<()> {
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

    if kv.kv_quant_bits.is_some() {
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
        flash_attn_decode_strided_multi_head_from_state_on_stream(
            out_base,
            q_base,
            k_cache,
            v_cache,
            scratch.decode_seq_len_ptr(),
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            kv_lora_dim,
            kv.adastate_anchors_enabled,
            w_up_k,
            w_up_v,
            device.stream(),
        )
    }
}
