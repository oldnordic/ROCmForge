//! Sparse CSR GEMV kernel dispatch.
//!
//! Wraps the HIP kernel for y = A * x where A is in CSR format.

use super::super::error::{GpuError, GpuResult};
use super::super::ffi::hipStream_t;

extern "C" {
    fn gpu_sparse_csr_gemv_f32(
        values: *const f32,
        col_indices: *const u32,
        row_offsets: *const u32,
        nnz: i32,
        rows: i32,
        cols: i32,
        x: *const f32,
        y: *mut f32,
        stream: hipStream_t,
    ) -> i32;
}

/// Dispatch sparse CSR GEMV: y = A * x
///
/// # Arguments
/// * `values` — nonzero values array [nnz]
/// * `col_indices` — column indices for each nonzero [nnz]
/// * `row_offsets` — row pointer array [rows + 1]
/// * `nnz` — number of nonzeros
/// * `rows` — number of rows in A
/// * `cols` — number of columns in A (and length of x)
/// * `x` — input vector [cols]
/// * `y` — output vector [rows]
/// * `stream` — HIP stream
pub fn dispatch_sparse_csr_gemv_f32(
    values: *const f32,
    col_indices: *const u32,
    row_offsets: *const u32,
    nnz: usize,
    rows: usize,
    cols: usize,
    x: *const f32,
    y: *mut f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if values.is_null()
        || col_indices.is_null()
        || row_offsets.is_null()
        || x.is_null()
        || y.is_null()
    {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "sparse_csr_gemv: null pointer argument".to_string(),
        });
    }
    if nnz == 0 || rows == 0 || cols == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "sparse_csr_gemv: invalid dimensions nnz={} rows={} cols={}",
                nnz, rows, cols
            ),
        });
    }

    let code = unsafe {
        gpu_sparse_csr_gemv_f32(
            values,
            col_indices,
            row_offsets,
            nnz as i32,
            rows as i32,
            cols as i32,
            x,
            y,
            stream,
        )
    };
    if code != 0 {
        return Err(GpuError::HipApiError {
            code,
            description: format!("gpu_sparse_csr_gemv_f32 failed with code {}", code),
        });
    }
    Ok(())
}
