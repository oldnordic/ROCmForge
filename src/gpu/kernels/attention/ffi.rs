use crate::gpu::ffi::{hipError_t, hipStream_t};
use std::os::raw::{c_float, c_int};

unsafe extern "C" {
    pub fn gpu_kv_write(
        d_k_cache: *mut f32,
        d_v_cache: *mut f32,
        d_k: *const f32,
        d_v: *const f32,
        pos: c_int,
        kv_size: c_int,
        max_seq: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn gpu_kv_write_state(
        d_k_cache: *mut f32,
        d_v_cache: *mut f32,
        d_k: *const f32,
        d_v: *const f32,
        d_pos: *const c_int,
        kv_size: c_int,
        max_seq: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn gpu_kv_write_rope(
        d_k_cache: *mut f32,
        d_v_cache: *mut f32,
        d_k: *const f32,
        d_v: *const f32,
        pos: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        theta_base: f32,
        neox: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn gpu_kv_write_rope_state(
        d_k_cache: *mut f32,
        d_v_cache: *mut f32,
        d_k: *const f32,
        d_v: *const f32,
        d_pos: *const c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        theta_base: c_float,
        neox: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn gpu_kv_write_batched(
        d_k_cache: *mut f32,
        d_v_cache: *mut f32,
        d_k: *const f32,
        d_v: *const f32,
        start_pos: c_int,
        kv_size: c_int,
        max_seq: c_int,
        seq_len: c_int,
    ) -> hipError_t;

    pub fn gpu_flash_attn_prefill_strided(
        d_out: *mut f32,
        d_q: *const f32,
        d_k: *const f32,
        d_v: *const f32,
        seq_len: c_int,
        head_dim: c_int,
        out_stride: c_int,
        q_stride: c_int,
        kv_stride: c_int,
        out_head_offset: c_int,
        q_head_offset: c_int,
        kv_head_offset: c_int,
        scale: c_float,
        kv_lora_dim: c_int,
        w_up_k: *const f32,
        w_up_v: *const f32,
    ) -> hipError_t;

    pub fn gpu_flash_attn_decode_strided_multi_head(
        d_out: *mut f32,
        d_q: *const f32,
        d_k_cache: *const f32,
        d_v_cache: *const f32,
        seq_len: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        scale: f32,
        kv_lora_dim: c_int,
        adastate_anchors_enabled: c_int,
        w_up_k: *const f32,
        w_up_v: *const f32,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn gpu_flash_attn_decode_strided_multi_head_state(
        d_out: *mut f32,
        d_q: *const f32,
        d_k_cache: *const f32,
        d_v_cache: *const f32,
        d_seq_len: *const c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        scale: f32,
        kv_lora_dim: c_int,
        adastate_anchors_enabled: c_int,
        w_up_k: *const f32,
        w_up_v: *const f32,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn gpu_kv_write_compressed(
        d_k_cache: *mut u8,
        d_v_cache: *mut u8,
        d_k: *const f32,
        d_v: *const f32,
        d_pos: *const c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        theta_base: f32,
        neox: c_int,
        kv_lora_dim: c_int,
        kv_frame_codec_enabled: c_int,
        w_down_k: *const f32,
        w_down_v: *const f32,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn gpu_kv_write_turboquant(
        d_k_cache: *mut u8,
        d_v_cache: *mut u8,
        d_k: *const f32,
        d_v: *const f32,
        d_pos: *const c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        theta_base: f32,
        neox: c_int,
        kv_lora_dim: c_int,
        bits: c_int,
        num_centroids: c_int,
        d_centroids: *const f32,
        qjl_scale: f32,
        w_down_k: *const f32,
        w_down_v: *const f32,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn gpu_flash_attn_decode_turboquant(
        d_out: *mut f32,
        d_q: *const f32,
        d_k_cache: *const u8,
        d_v_cache: *const u8,
        seq_len: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        scale: f32,
        kv_lora_dim: c_int,
        bits: c_int,
        num_centroids: c_int,
        d_centroids: *const f32,
        qjl_scale: f32,
        w_up_k: *const f32,
        w_up_v: *const f32,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn gpu_kv_write_batched_compressed(
        d_k_cache: *mut u8,
        d_v_cache: *mut u8,
        d_k: *const f32,
        d_v: *const f32,
        start_pos: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        seq_len: c_int,
        kv_lora_dim: c_int,
        kv_frame_codec_enabled: c_int,
        w_down_k: *const f32,
        w_down_v: *const f32,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn gpu_reconstruct_kv_cache_prefix_sum(
        d_k_cache: *mut f32,
        d_v_cache: *mut f32,
        d_pos: *const c_int,
        start_pos: c_int,
        seq_len: c_int,
        kv_lora_dim: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}
