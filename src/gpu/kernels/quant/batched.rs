//! Batched Q4_0 quantization kernels for prefill processing.
//!
//! Safety-first: bounds checked before kernel launch.

use super::super::super::error::{GpuError, GpuResult};
use super::super::super::ffi::{hipError_t, hipStream_t};
use std::os::raw::c_int;

// ── External HIP Kernel Declarations ───────────────────────────────────────────────

extern "C" {
    /// Batched Q4_0 × f32 GEMM for prefill processing.
    ///
    /// Computes: output[seq_len][ncols_dst] = gemm(input[seq_len][n_rows], weights)
    ///
    /// # Arguments
    /// * `weights_q4_0` - Weight matrix in Q4_0 format [ncols_dst][n_rows/32][18]
    /// * `input` - Input activations [seq_len][n_rows] (row-major)
    /// * `output` - Output buffer [seq_len][ncols_dst] (row-major)
    /// * `n_rows` - Number of rows in weight matrix (input dimension)
    /// * `ncols_dst` - Number of columns in weight matrix (output dimension)
    /// * `seq_len` - Number of tokens in batch (sequence length)
    /// * `stream` - HIP stream for kernel execution
    ///
    /// # Returns
    /// hipSuccess on success, error code otherwise
    fn batched_gemm_q4_0_f32_prefill(
        weights_q4_0: *const std::ffi::c_void,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        seq_len: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_wmma_matmul_q4_0(
        weights_q4_0: *const std::ffi::c_void,
        input: *const f32,
        output: *mut f32,
        in_dim: c_int,
        out_dim: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    /// Batched Q4_1 × f32 GEMM for prefill processing.
    ///
    /// Computes: output[seq_len][ncols_dst] = gemm(input[seq_len][n_rows], weights)
    ///
    /// # Arguments
    /// * `weights_q4_1` - Weight matrix in Q4_1 format [ncols_dst][n_rows/32][20]
    /// * `input` - Input activations [seq_len][n_rows] (row-major)
    /// * `output` - Output buffer [seq_len][ncols_dst] (row-major)
    /// * `n_rows` - Number of rows in weight matrix (input dimension)
    /// * `ncols_dst` - Number of columns in weight matrix (output dimension)
    /// * `seq_len` - Number of tokens in batch (sequence length)
    /// * `stream` - HIP stream for kernel execution
    ///
    /// # Returns
    /// hipSuccess on success, error code otherwise
    fn batched_gemm_q4_1_f32_prefill(
        weights_q4_1: *const std::ffi::c_void,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        seq_len: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    /// Batched fused Gate/Up GEMV + SwiGLU for prefill processing.
    ///
    /// Computes for all sequence positions in parallel:
    ///   output[seq_len][ff_size] = SiLU(gate[seq_len][ff_size]) * up[seq_len][ff_size]
    ///   where gate = input × W_gate and up = input × W_up
    ///
    /// # Arguments
    /// * `w_gate_q4_0` - Gate weights in Q4_0 format [ff_size][n_rows/32][18]
    /// * `w_up_q4_0` - Up weights in Q4_0 format [ff_size][n_rows/32][18]
    /// * `input` - Input activations [seq_len][n_rows] (row-major)
    /// * `output` - Output buffer [seq_len][ff_size] (row-major)
    /// * `n_rows` - Input dimension (hidden size)
    /// * `ff_size` - Feed-forward dimension (gate/up output size)
    /// * `seq_len` - Number of tokens in batch
    /// * `stream` - HIP stream for kernel execution
    ///
    /// # Returns
    /// hipSuccess on success, error code otherwise
    fn batched_fused_gate_up_q4_0_f32_prefill(
        w_gate_q4_0: *const std::ffi::c_void,
        w_up_q4_0: *const std::ffi::c_void,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ff_size: c_int,
        seq_len: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}

// ── Public Rust API ─────────────────────────────────────────────────────────────────

/// Batched Q4_0 × f32 GEMM for prefill processing.
///
/// This function computes matrix multiplication for multiple input vectors in parallel,
/// which is the primary workload during prompt prefill. Each token in the prompt is
/// processed independently through the same weight matrix.
///
/// # Arguments
/// * `weights_q4_0` - GPU pointer to Q4_0 weight matrix [ncols_dst][n_rows/32][18]
/// * `input` - GPU pointer to input activations [seq_len][n_rows] (row-major)
/// * `output` - GPU pointer to output buffer [seq_len][ncols_dst] (row-major)
/// * `n_rows` - Number of rows in weight matrix (input dimension)
/// * `ncols_dst` - Number of columns in weight matrix (output dimension)
/// * `seq_len` - Number of tokens in batch (sequence length)
/// * `stream` - HIP stream for kernel execution
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
/// - seq_len must be > 0
///
/// # Thread Block Organization
/// - Grid: (ncols_dst, seq_len)
/// - Block: 256 threads
/// - Each block processes one (column, sequence_position) pair
pub fn batched_gemm_q4_0_f32(
    weights_q4_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    seq_len: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_gemm_q4_0_f32: seq_len cannot be zero".to_string(),
        });
    }

    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_gemm_q4_0_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    if weights_q4_0.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_gemm_q4_0_f32: weights pointer is null".to_string(),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_gemm_q4_0_f32: input pointer is null".to_string(),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_gemm_q4_0_f32: output pointer is null".to_string(),
        });
    }

    // Validate dimensions are within i32 range for HIP kernel
    let n_rows_i32 = c_int::try_from(n_rows).map_err(|_| GpuError::HipApiError {
        code: -1,
        description: format!("batched_gemm_q4_0_f32: n_rows {} exceeds i32 range", n_rows),
    })?;

    let ncols_dst_i32 = c_int::try_from(ncols_dst).map_err(|_| GpuError::HipApiError {
        code: -1,
        description: format!(
            "batched_gemm_q4_0_f32: ncols_dst {} exceeds i32 range",
            ncols_dst
        ),
    })?;

    let seq_len_i32 = c_int::try_from(seq_len).map_err(|_| GpuError::HipApiError {
        code: -1,
        description: format!(
            "batched_gemm_q4_0_f32: seq_len {} exceeds i32 range",
            seq_len
        ),
    })?;

    let result = unsafe {
        batched_gemm_q4_0_f32_prefill(
            weights_q4_0 as *const std::ffi::c_void,
            input,
            output,
            n_rows_i32,
            ncols_dst_i32,
            seq_len_i32,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("batched_gemm_q4_0_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Batched Q4_1 × f32 GEMM for prefill processing.
///
/// This function computes matrix multiplication for multiple input vectors in parallel,
/// which is the primary workload during prompt prefill. Each token in the prompt is
/// processed independently through the same weight matrix.
///
/// # Arguments
/// * `weights_q4_1` - GPU pointer to Q4_1 weight matrix [ncols_dst][n_rows/32][20]
/// * `input` - GPU pointer to input activations [seq_len][n_rows] (row-major)
/// * `output` - GPU pointer to output buffer [seq_len][ncols_dst] (row-major)
/// * `n_rows` - Number of rows in weight matrix (input dimension)
/// * `ncols_dst` - Number of columns in weight matrix (output dimension)
/// * `seq_len` - Number of tokens in batch (sequence length)
/// * `stream` - HIP stream for kernel execution
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
/// - seq_len must be > 0
///
/// # Thread Block Organization
/// - Grid: (ncols_dst, seq_len)
/// - Block: 256 threads
/// - Each block processes one (column, sequence_position) pair
pub fn batched_gemm_q4_1_f32(
    weights_q4_1: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    seq_len: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_gemm_q4_1_f32: seq_len cannot be zero".to_string(),
        });
    }

    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_gemm_q4_1_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    if weights_q4_1.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_gemm_q4_1_f32: weights pointer is null".to_string(),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_gemm_q4_1_f32: input pointer is null".to_string(),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_gemm_q4_1_f32: output pointer is null".to_string(),
        });
    }

    // Validate dimensions are within i32 range for HIP kernel
    let n_rows_i32 = c_int::try_from(n_rows).map_err(|_| GpuError::HipApiError {
        code: -1,
        description: format!("batched_gemm_q4_1_f32: n_rows {} exceeds i32 range", n_rows),
    })?;

    let ncols_dst_i32 = c_int::try_from(ncols_dst).map_err(|_| GpuError::HipApiError {
        code: -1,
        description: format!(
            "batched_gemm_q4_1_f32: ncols_dst {} exceeds i32 range",
            ncols_dst
        ),
    })?;

    let seq_len_i32 = c_int::try_from(seq_len).map_err(|_| GpuError::HipApiError {
        code: -1,
        description: format!(
            "batched_gemm_q4_1_f32: seq_len {} exceeds i32 range",
            seq_len
        ),
    })?;

    let result = unsafe {
        batched_gemm_q4_1_f32_prefill(
            weights_q4_1 as *const std::ffi::c_void,
            input,
            output,
            n_rows_i32,
            ncols_dst_i32,
            seq_len_i32,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("batched_gemm_q4_1_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Batched fused Gate/Up GEMV + SwiGLU for prefill processing.
///
/// This function computes the fused gate-up projection with SwiGLU activation
/// for multiple input vectors in parallel, which is the primary workload during
/// prompt prefill. Each token in the prompt is processed independently through
/// the same gate and up weight matrices.
///
/// # Arguments
/// * `w_gate_q4_0` - GPU pointer to Q4_0 gate weight matrix [ff_size][n_rows/32][18]
/// * `w_up_q4_0` - GPU pointer to Q4_0 up weight matrix [ff_size][n_rows/32][18]
/// * `input` - GPU pointer to input activations [seq_len][n_rows] (row-major)
/// * `output` - GPU pointer to output buffer [seq_len][ff_size] (row-major)
/// * `n_rows` - Input dimension (hidden size)
/// * `ff_size` - Feed-forward dimension (gate/up output size)
/// * `seq_len` - Number of tokens in batch (sequence length)
/// * `stream` - HIP stream for kernel execution
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
/// - seq_len must be > 0
///
/// # Thread Block Organization
/// - Grid: (ff_size, seq_len)
/// - Block: 256 threads
/// - Each block processes one (column, sequence_position) pair
pub fn batched_fused_gate_up_q4_0_f32(
    w_gate_q4_0: *const u8,
    w_up_q4_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ff_size: usize,
    seq_len: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_fused_gate_up_q4_0_f32: seq_len cannot be zero".to_string(),
        });
    }

    if n_rows == 0 || ff_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_fused_gate_up_q4_0_f32: n_rows and ff_size cannot be zero"
                .to_string(),
        });
    }

    if w_gate_q4_0.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_fused_gate_up_q4_0_f32: gate weights pointer is null".to_string(),
        });
    }

    if w_up_q4_0.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_fused_gate_up_q4_0_f32: up weights pointer is null".to_string(),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_fused_gate_up_q4_0_f32: input pointer is null".to_string(),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "batched_fused_gate_up_q4_0_f32: output pointer is null".to_string(),
        });
    }

    // Validate dimensions are within i32 range for HIP kernel
    let n_rows_i32 = c_int::try_from(n_rows).map_err(|_| GpuError::HipApiError {
        code: -1,
        description: format!(
            "batched_fused_gate_up_q4_0_f32: n_rows {} exceeds i32 range",
            n_rows
        ),
    })?;

    let ff_size_i32 = c_int::try_from(ff_size).map_err(|_| GpuError::HipApiError {
        code: -1,
        description: format!(
            "batched_fused_gate_up_q4_0_f32: ff_size {} exceeds i32 range",
            ff_size
        ),
    })?;

    let seq_len_i32 = c_int::try_from(seq_len).map_err(|_| GpuError::HipApiError {
        code: -1,
        description: format!(
            "batched_fused_gate_up_q4_0_f32: seq_len {} exceeds i32 range",
            seq_len
        ),
    })?;

    let result = unsafe {
        batched_fused_gate_up_q4_0_f32_prefill(
            w_gate_q4_0 as *const std::ffi::c_void,
            w_up_q4_0 as *const std::ffi::c_void,
            input,
            output,
            n_rows_i32,
            ff_size_i32,
            seq_len_i32,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("batched_fused_gate_up_q4_0_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Accelerated Q4_0 matrix multiplication using GPU/matrix cores.
pub fn wmma_matmul_q4_0_f32(
    weights_q4_0: *const std::ffi::c_void,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "wmma_matmul_q4_0_f32: batch_size cannot be zero".to_string(),
        });
    }
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "wmma_matmul_q4_0_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }
    if weights_q4_0.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "wmma_matmul_q4_0_f32: weights pointer is null".to_string(),
        });
    }
    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "wmma_matmul_q4_0_f32: input pointer is null".to_string(),
        });
    }
    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "wmma_matmul_q4_0_f32: output pointer is null".to_string(),
        });
    }

    let n_rows_i32 = n_rows.try_into().map_err(|_| GpuError::HipApiError {
        code: -1,
        description: format!("wmma_matmul_q4_0_f32: n_rows {} exceeds i32 range", n_rows),
    })?;

    let ncols_dst_i32 = ncols_dst.try_into().map_err(|_| GpuError::HipApiError {
        code: -1,
        description: format!(
            "wmma_matmul_q4_0_f32: ncols_dst {} exceeds i32 range",
            ncols_dst
        ),
    })?;

    let batch_size_i32 = batch_size.try_into().map_err(|_| GpuError::HipApiError {
        code: -1,
        description: format!(
            "wmma_matmul_q4_0_f32: batch_size {} exceeds i32 range",
            batch_size
        ),
    })?;

    let result = unsafe {
        gpu_wmma_matmul_q4_0(
            weights_q4_0,
            input,
            output,
            n_rows_i32,
            ncols_dst_i32,
            batch_size_i32,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gpu_wmma_matmul_q4_0 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_gemm_rejects_zero_seq_len() {
        // Test that seq_len=0 is rejected
        let result = batched_gemm_q4_0_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            1024,
            4096,
            0,
            hipStream_t::null(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn batched_gemm_rejects_zero_dimensions() {
        // Test that n_rows=0 is rejected
        let result = batched_gemm_q4_0_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            4096,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());

        // Test that ncols_dst=0 is rejected
        let result = batched_gemm_q4_0_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            1024,
            0,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn batched_gemm_rejects_null_pointers() {
        // Test that null weights pointer is rejected
        let result = batched_gemm_q4_0_f32(
            std::ptr::null(),
            0x1000 as *const f32,
            std::ptr::null_mut(),
            1024,
            4096,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());

        // Test that null input pointer is rejected
        let result = batched_gemm_q4_0_f32(
            0x1000 as *const u8,
            std::ptr::null(),
            std::ptr::null_mut(),
            1024,
            4096,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());

        // Test that null output pointer is rejected
        let result = batched_gemm_q4_0_f32(
            0x1000 as *const u8,
            0x2000 as *const f32,
            std::ptr::null_mut(),
            1024,
            4096,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn batched_gemm_q4_1_rejects_zero_seq_len() {
        // Test that seq_len=0 is rejected
        let result = batched_gemm_q4_1_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            1024,
            4096,
            0,
            hipStream_t::null(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn batched_gemm_q4_1_rejects_zero_dimensions() {
        // Test that n_rows=0 is rejected
        let result = batched_gemm_q4_1_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            4096,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());

        // Test that ncols_dst=0 is rejected
        let result = batched_gemm_q4_1_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            1024,
            0,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn batched_gemm_q4_1_rejects_null_pointers() {
        // Test that null weights pointer is rejected
        let result = batched_gemm_q4_1_f32(
            std::ptr::null(),
            0x1000 as *const f32,
            std::ptr::null_mut(),
            1024,
            4096,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());

        // Test that null input pointer is rejected
        let result = batched_gemm_q4_1_f32(
            0x1000 as *const u8,
            std::ptr::null(),
            std::ptr::null_mut(),
            1024,
            4096,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());

        // Test that null output pointer is rejected
        let result = batched_gemm_q4_1_f32(
            0x1000 as *const u8,
            0x2000 as *const f32,
            std::ptr::null_mut(),
            1024,
            4096,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn batched_fused_gate_up_q4_0_rejects_zero_seq_len() {
        // Test that seq_len=0 is rejected
        let result = batched_fused_gate_up_q4_0_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            1024,
            4096,
            0,
            hipStream_t::null(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn batched_fused_gate_up_q4_0_rejects_zero_dimensions() {
        // Test that n_rows=0 is rejected
        let result = batched_fused_gate_up_q4_0_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            4096,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());

        // Test that ff_size=0 is rejected
        let result = batched_fused_gate_up_q4_0_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            1024,
            0,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn batched_fused_gate_up_q4_0_rejects_null_pointers() {
        // Test that null gate weights pointer is rejected
        let result = batched_fused_gate_up_q4_0_f32(
            std::ptr::null(),
            0x1000 as *const u8,
            0x2000 as *const f32,
            std::ptr::null_mut(),
            1024,
            4096,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());

        // Test that null up weights pointer is rejected
        let result = batched_fused_gate_up_q4_0_f32(
            0x1000 as *const u8,
            std::ptr::null(),
            0x2000 as *const f32,
            std::ptr::null_mut(),
            1024,
            4096,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());

        // Test that null input pointer is rejected
        let result = batched_fused_gate_up_q4_0_f32(
            0x1000 as *const u8,
            0x1000 as *const u8,
            std::ptr::null(),
            std::ptr::null_mut(),
            1024,
            4096,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());

        // Test that null output pointer is rejected
        let result = batched_fused_gate_up_q4_0_f32(
            0x1000 as *const u8,
            0x1000 as *const u8,
            0x2000 as *const f32,
            std::ptr::null_mut(),
            1024,
            4096,
            10,
            hipStream_t::null(),
        );
        assert!(result.is_err());
    }
}
