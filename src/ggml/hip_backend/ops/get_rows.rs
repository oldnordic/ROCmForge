//! HIP GetRows implementation using device-to-device copies.

use crate::backend::{HipBackend, HipError, HipResult, HipBuffer};
use crate::ggml::Layout;

pub fn get_rows(
    _backend: &HipBackend,
    weights: &HipBuffer,
    tokens: &[u32],
    n_embd: usize,
    n_vocab: usize,
    layout: Layout,
    output: &HipBuffer,
) -> HipResult<()> {
    let elem_bytes = std::mem::size_of::<f32>();
    for (i, &token_id) in tokens.iter().enumerate() {
        let token_index = token_id as usize;
        let dst_offset = i * n_embd;
        if token_index >= n_vocab {
            return Err(HipError::GenericError(format!(
                "Token ID {} exceeds vocab size {}",
                token_id, n_vocab
            )));
        }
        let dst_offset_bytes = dst_offset * elem_bytes;
        match layout {
            Layout::RowMajor => {
                let src_offset = token_index * n_embd;
                let byte_len = n_embd * elem_bytes;
                let src_offset_bytes = src_offset * elem_bytes;
                output.copy_from_buffer_region(dst_offset_bytes, weights, src_offset_bytes, byte_len)?;
            }
            Layout::ColMajor => {
                let src_offset_bytes = token_index * elem_bytes;
                let src_pitch_bytes = n_vocab * elem_bytes;
                output.copy_from_buffer_strided_2d(
                    dst_offset_bytes,
                    elem_bytes,
                    weights,
                    src_offset_bytes,
                    src_pitch_bytes,
                    elem_bytes,
                    n_embd,
                )?;
            }
            Layout::Strided => {
                return Err(HipError::GenericError(
                    "Strided layout not supported for GetRows".to_string(),
                ));
            }
        }
    }
    Ok(())
}

pub fn validate_token_ids(tokens: &[u32], vocab_size: usize) -> HipResult<()> {
    for &token_id in tokens {
        if token_id as usize >= vocab_size {
            return Err(HipError::GenericError(format!(
                "Token ID {} exceeds vocab size {}",
                token_id, vocab_size
            )));
        }
    }
    Ok(())
}
