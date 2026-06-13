use super::super::super::error::{GpuError, GpuResult};
use super::super::super::ffi::hipError_t;
use std::os::raw::c_int;

extern "C" {
    fn gpu_embed_q8_0_token(
        embedding_q8_0: *const u8,
        out: *mut f32,
        hidden_size: c_int,
        vocab_size: c_int,
        token_id: c_int,
    ) -> hipError_t;

    fn gpu_embed_q4_0_token(
        embedding_q4_0: *const u8,
        out: *mut f32,
        hidden_size: c_int,
        vocab_size: c_int,
        token_id: c_int,
    ) -> hipError_t;

    fn gpu_embed_q8_0_batch(
        embedding_q8_0: *const u8,
        token_ids: *const i32,
        out: *mut f32,
        hidden_size: c_int,
        vocab_size: c_int,
        seq_len: c_int,
    ) -> hipError_t;

    fn gpu_embed_q4_0_batch(
        embedding_q4_0: *const u8,
        token_ids: *const i32,
        out: *mut f32,
        hidden_size: c_int,
        vocab_size: c_int,
        seq_len: c_int,
    ) -> hipError_t;
}

/// Decode one Q8_0 token embedding directly on GPU into an f32 hidden buffer.
pub fn embed_q8_0_token(
    embedding_q8_0: *const u8,
    out: *mut f32,
    hidden_size: usize,
    vocab_size: usize,
    token_id: u32,
) -> GpuResult<()> {
    if hidden_size == 0 || vocab_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "embed_q8_0_token: hidden_size and vocab_size cannot be zero".to_string(),
        });
    }
    if !hidden_size.is_multiple_of(32) {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "embed_q8_0_token: hidden_size must be a multiple of 32, got {}",
                hidden_size
            ),
        });
    }
    if embedding_q8_0.is_null() || out.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "embed_q8_0_token: embedding and output pointers must be non-null"
                .to_string(),
        });
    }
    if token_id as usize >= vocab_size {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "embed_q8_0_token: token_id {} out of range for vocab_size {}",
                token_id, vocab_size
            ),
        });
    }

    let result = unsafe {
        gpu_embed_q8_0_token(
            embedding_q8_0,
            out,
            hidden_size as c_int,
            vocab_size as c_int,
            token_id as c_int,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("embed_q8_0_token kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Decode one Q4_0 token embedding directly on GPU into an f32 hidden buffer.
pub fn embed_q4_0_token(
    embedding_q4_0: *const u8,
    out: *mut f32,
    hidden_size: usize,
    vocab_size: usize,
    token_id: u32,
) -> GpuResult<()> {
    if hidden_size == 0 || vocab_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "embed_q4_0_token: hidden_size and vocab_size cannot be zero".to_string(),
        });
    }
    if !hidden_size.is_multiple_of(32) {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "embed_q4_0_token: hidden_size must be a multiple of 32, got {}",
                hidden_size
            ),
        });
    }
    if embedding_q4_0.is_null() || out.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "embed_q4_0_token: embedding and output pointers must be non-null"
                .to_string(),
        });
    }
    if token_id as usize >= vocab_size {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "embed_q4_0_token: token_id {} out of range for vocab_size {}",
                token_id, vocab_size
            ),
        });
    }

    let result = unsafe {
        gpu_embed_q4_0_token(
            embedding_q4_0,
            out,
            hidden_size as c_int,
            vocab_size as c_int,
            token_id as c_int,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("embed_q4_0_token kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Decode a batch of Q8_0 token embeddings directly on GPU into an f32 output buffer.
pub fn embed_q8_0_batch(
    embedding_q8_0: *const u8,
    token_ids: *const i32,
    out: *mut f32,
    hidden_size: usize,
    vocab_size: usize,
    seq_len: usize,
) -> GpuResult<()> {
    if hidden_size == 0 || vocab_size == 0 || seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "embed_q8_0_batch: hidden_size, vocab_size, and seq_len cannot be zero"
                .to_string(),
        });
    }
    if !hidden_size.is_multiple_of(32) {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "embed_q8_0_batch: hidden_size must be a multiple of 32, got {}",
                hidden_size
            ),
        });
    }
    if embedding_q8_0.is_null() || token_ids.is_null() || out.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "embed_q8_0_batch: embedding, token_ids, and output pointers must be non-null"
                    .to_string(),
        });
    }

    let result = unsafe {
        gpu_embed_q8_0_batch(
            embedding_q8_0,
            token_ids,
            out,
            hidden_size as c_int,
            vocab_size as c_int,
            seq_len as c_int,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("embed_q8_0_batch kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Decode a batch of Q4_0 token embeddings directly on GPU into an f32 output buffer.
pub fn embed_q4_0_batch(
    embedding_q4_0: *const u8,
    token_ids: *const i32,
    out: *mut f32,
    hidden_size: usize,
    vocab_size: usize,
    seq_len: usize,
) -> GpuResult<()> {
    if hidden_size == 0 || vocab_size == 0 || seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "embed_q4_0_batch: hidden_size, vocab_size, and seq_len cannot be zero"
                .to_string(),
        });
    }
    if !hidden_size.is_multiple_of(32) {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "embed_q4_0_batch: hidden_size must be a multiple of 32, got {}",
                hidden_size
            ),
        });
    }
    if embedding_q4_0.is_null() || token_ids.is_null() || out.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "embed_q4_0_batch: embedding, token_ids, and output pointers must be non-null"
                    .to_string(),
        });
    }

    let result = unsafe {
        gpu_embed_q4_0_batch(
            embedding_q4_0,
            token_ids,
            out,
            hidden_size as c_int,
            vocab_size as c_int,
            seq_len as c_int,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("embed_q4_0_batch kernel failed: {:?}", result),
        });
    }

    Ok(())
}
