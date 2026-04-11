//! Element-wise kernel wrappers.
//!
//! Safety-first: bounds checked before kernel launch.

use super::super::error::{GpuError, GpuResult};
use super::super::ffi::{hipError_t, hipStream_t};
use super::super::GpuDevice;
use std::os::raw::c_int;

/// Element-wise add: out = x + y
pub fn add(x: *const f32, y: *const f32, out: *mut f32, n: usize) -> GpuResult<()> {
    add_on_stream(x, y, out, n, hipStream_t::null())
}

/// Element-wise add on an explicit HIP stream.
pub fn add_on_stream(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise add: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { gpu_add(x, y, out, n as c_int, stream) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("add kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Element-wise multiply: out = x * y
pub fn mul(x: *const f32, y: *const f32, out: *mut f32, n: usize) -> GpuResult<()> {
    mul_on_stream(x, y, out, n, hipStream_t::null())
}

/// Element-wise multiply on an explicit HIP stream.
pub fn mul_on_stream(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise mul: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { gpu_mul_on_stream(x, y, out, n as c_int, stream) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("mul kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Element-wise scale: out = x * scale
pub fn scale(x: *const f32, out: *mut f32, scale: f32, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise scale: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { gpu_scale(x, out, scale, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("scale kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// GELU activation: out = gelu(x)
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: *const f32, out: *mut f32, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise gelu: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { gpu_gelu(x, out, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gelu kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// SiLU activation: out = x / (1 + exp(-x))
pub fn silu(x: *const f32, out: *mut f32, n: usize) -> GpuResult<()> {
    silu_on_stream(x, out, n, hipStream_t::null())
}

/// SiLU activation on an explicit HIP stream.
pub fn silu_on_stream(
    x: *const f32,
    out: *mut f32,
    n: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise silu: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { gpu_silu_on_stream(x, out, n as c_int, stream) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("silu kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Batched element-wise add for prefill: out[s, i] = x[s, i] + y[i]
/// where x is [seq_len, n] and y is [n] (broadcast)
pub fn add_batched(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
    seq_len: usize,
) -> GpuResult<()> {
    if n == 0 || seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise add_batched: n and seq_len cannot be zero".to_string(),
        });
    }

    let result = unsafe { gpu_add_batched(x, y, out, n as c_int, seq_len as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("add_batched kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Batched element-wise multiply for prefill: out[s, i] = x[s, i] * y[i]
pub fn mul_batched(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
    seq_len: usize,
) -> GpuResult<()> {
    if n == 0 || seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise mul_batched: n and seq_len cannot be zero".to_string(),
        });
    }

    let result = unsafe { gpu_mul_batched(x, y, out, n as c_int, seq_len as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("mul_batched kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Argmax reduction over logits: returns the index of the maximum value.
///
/// Uses reusable GPU workspace to avoid per-token allocations.
pub fn argmax_f32(
    input: *const f32,
    partial_values: *mut f32,
    partial_indices: *mut i32,
    output_index: *mut i32,
    n: usize,
) -> GpuResult<()> {
    argmax_f32_on_stream(
        input,
        partial_values,
        partial_indices,
        output_index,
        n,
        hipStream_t::null(),
    )
}

/// Argmax reduction over logits on an explicit HIP stream.
///
/// Uses reusable GPU workspace to avoid per-token allocations.
pub fn argmax_f32_on_stream(
    input: *const f32,
    partial_values: *mut f32,
    partial_indices: *mut i32,
    output_index: *mut i32,
    n: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "argmax_f32: n cannot be zero".to_string(),
        });
    }
    if input.is_null()
        || partial_values.is_null()
        || partial_indices.is_null()
        || output_index.is_null()
    {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "argmax_f32: all pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gpu_argmax_f32_on_stream(
            input,
            partial_values,
            partial_indices,
            output_index,
            n as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("argmax_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
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
    if hidden_size % 32 != 0 {
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
    if hidden_size % 32 != 0 {
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
    if hidden_size % 32 != 0 {
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
    if hidden_size % 32 != 0 {
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

/// Zero-fill GPU memory: ptr[i] = 0.0f for i in 0..n
///
/// Launches asynchronously on device's stream.
/// Caller must call device.synchronize() if sync needed.
pub fn zero_fill(ptr: *mut f32, n: usize, device: &GpuDevice) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "zero_fill: n cannot be zero".to_string(),
        });
    }

    let stream = device.stream();

    let result = unsafe { gpu_zero_fill(ptr, n as c_int, stream) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("zero_fill kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Increment in-device decode state `[pos, seq_len]` by 1 on an explicit stream.
pub fn increment_decode_state_on_stream(state: *mut i32, stream: hipStream_t) -> GpuResult<()> {
    if state.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "increment_decode_state_on_stream: state pointer must be non-null"
                .to_string(),
        });
    }

    let result = unsafe { gpu_increment_decode_state_on_stream(state, stream) };
    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!(
                "increment_decode_state_on_stream kernel failed: {:?}",
                result
            ),
        });
    }

    Ok(())
}

// FFI declarations - will be linked from compiled HIP kernels
unsafe extern "C" {
    fn gpu_add(
        x: *const f32,
        y: *const f32,
        out: *mut f32,
        n: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_mul(x: *const f32, y: *const f32, out: *mut f32, n: c_int) -> hipError_t;

    fn gpu_mul_on_stream(
        x: *const f32,
        y: *const f32,
        out: *mut f32,
        n: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_scale(x: *const f32, out: *mut f32, scale: f32, n: c_int) -> hipError_t;

    fn gpu_gelu(x: *const f32, out: *mut f32, n: c_int) -> hipError_t;

    fn gpu_silu(x: *const f32, out: *mut f32, n: c_int) -> hipError_t;

    fn gpu_silu_on_stream(
        x: *const f32,
        out: *mut f32,
        n: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_add_batched(
        x: *const f32,
        y: *const f32,
        out: *mut f32,
        n: c_int,
        seq_len: c_int,
    ) -> hipError_t;

    fn gpu_mul_batched(
        x: *const f32,
        y: *const f32,
        out: *mut f32,
        n: c_int,
        seq_len: c_int,
    ) -> hipError_t;

    fn gpu_argmax_f32(
        input: *const f32,
        partial_values: *mut f32,
        partial_indices: *mut i32,
        output_index: *mut i32,
        n: c_int,
    ) -> hipError_t;

    fn gpu_argmax_f32_on_stream(
        input: *const f32,
        partial_values: *mut f32,
        partial_indices: *mut i32,
        output_index: *mut i32,
        n: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

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

    fn gpu_zero_fill(ptr: *mut f32, n: c_int, stream: hipStream_t) -> hipError_t;

    fn gpu_increment_decode_state_on_stream(state: *mut i32, stream: hipStream_t) -> hipError_t;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_rejects_zero_n() {
        let result = add(std::ptr::null(), std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn mul_on_stream_rejects_zero_n() {
        let result = mul_on_stream(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            hipStream_t::null(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn gelu_rejects_zero_n() {
        let result = gelu(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn silu_on_stream_rejects_zero_n() {
        let result = silu_on_stream(
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            hipStream_t::null(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn add_batched_rejects_zero_seq_len() {
        let result = add_batched(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            128,
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn argmax_rejects_zero_n() {
        let result = argmax_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn embed_q8_0_rejects_unaligned_hidden_size() {
        let result = embed_q8_0_token(std::ptr::null(), std::ptr::null_mut(), 33, 10, 0);
        assert!(result.is_err());
    }

    #[test]
    fn embed_q8_0_batch_rejects_zero_seq_len() {
        let result = embed_q8_0_batch(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            32,
            10,
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn zero_fill_rejects_zero_n() {
        use super::super::super::GpuDevice;

        let device = GpuDevice::init(0);
        let result = match device {
            Ok(d) => zero_fill(std::ptr::null_mut(), 0, &d),
            Err(_) => return, // Skip test if GPU unavailable
        };
        assert!(result.is_err());
    }
}
