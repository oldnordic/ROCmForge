//! Stable binding-tag helpers for decode-graph key construction.

use crate::gpu::weights::GpuBuffer;

#[inline]
fn mix_binding_tag(tag: u64, ptr: usize) -> u64 {
    tag.rotate_left(13) ^ (ptr as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

pub(crate) fn compute_kv_binding_tag(k: &[GpuBuffer], v: &[GpuBuffer]) -> u64 {
    let mut tag = 0u64;
    for buffer in k {
        tag = mix_binding_tag(tag, buffer.as_ptr() as usize);
    }
    for buffer in v {
        tag = mix_binding_tag(tag, buffer.as_ptr() as usize);
    }
    tag
}
