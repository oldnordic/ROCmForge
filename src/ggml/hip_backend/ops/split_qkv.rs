//! HIP QKV split using strided device-to-device copies.

use crate::backend::{HipBuffer, HipError, HipResult};

pub fn split_qkv(
    input: &HipBuffer,
    output_q: &HipBuffer,
    output_k: &HipBuffer,
    output_v: &HipBuffer,
    seq_len: usize,
    hidden: usize,
) -> HipResult<()> {
    let elem_bytes = std::mem::size_of::<f32>();
    let src_pitch_bytes = hidden
        .checked_mul(3)
        .and_then(|v| v.checked_mul(elem_bytes))
        .ok_or_else(|| HipError::GenericError("QKV source pitch overflow".to_string()))?;
    let dst_pitch_bytes = hidden
        .checked_mul(elem_bytes)
        .ok_or_else(|| HipError::GenericError("QKV dest pitch overflow".to_string()))?;
    let width_bytes = dst_pitch_bytes;

    let offsets = [
        (0usize, output_q),
        (hidden, output_k),
        (hidden * 2, output_v),
    ];

    for (col_offset, output) in offsets {
        let src_offset_bytes = col_offset
            .checked_mul(elem_bytes)
            .ok_or_else(|| HipError::GenericError("QKV source offset overflow".to_string()))?;
        output.copy_from_buffer_strided_2d(
            0,
            dst_pitch_bytes,
            input,
            src_offset_bytes,
            src_pitch_bytes,
            width_bytes,
            seq_len,
        )?;
    }

    Ok(())
}
