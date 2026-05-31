//! rocSOLVER GPU-accelerated SVD for the model converter.
//!
//! Provides `gpu_svd_batch` which processes one expert matrix at a time
//! using the non-batched `rocsolver_sgesvd`, keeping the API surface small
//! and avoiding strided-batched alignment requirements that vary by m/n shape.
//!
//! # Performance
//! GPU SVD of a [2048, 512] matrix: ~2-5 ms.  256 experts × 3 tensors ×
//! 40 layers = 30,720 calls → ~60-150 s total, vs 8-16 h on CPU.
//!
//! # Safety
//! - VRAM checked before each call; returns `Err` if below desktop-safe threshold.
//! - All GPU buffers are RAII-freed after each expert.
//! - `hipDeviceSynchronize` + info check after every SVD call.
//! - No graph capture; safe on a display-attached GPU.

use super::error::{GpuError, GpuResult};
use super::ffi;
use super::vram_budget::desktop_vram_reservation;
use super::weights::GpuBuffer;
use std::os::raw::c_int;

// ── Safety ────────────────────────────────────────────────────────────────────

/// Minimum free VRAM before each GPU SVD allocation.
fn vram_guard_bytes() -> usize {
    desktop_vram_reservation() + 512 * 1024 * 1024 // + 512 MB safety margin
}

// ── FFI ───────────────────────────────────────────────────────────────────────

#[expect(non_camel_case_types, reason = "ROCSolver C API uses C naming conventions")]
type rocblas_handle = *mut std::ffi::c_void;
#[expect(non_camel_case_types, reason = "ROCSolver C API uses C naming conventions")]
type rocblas_int = c_int;

/// Values from /opt/rocm/include/rocblas/rocblas-types.h — NOT ASCII codes.
#[allow(non_camel_case_types, dead_code)]
#[repr(i32)]
enum rocblas_svect {
    All = 191,
    Singular = 192,
    Overwrite = 193,
    None = 194,
}

/// Algorithm variant for sgesvd fast_alg.
/// Values from /opt/rocm/include/rocsolver/rocsolver-extra-types.h —
/// NOT ASCII character codes.
#[allow(non_camel_case_types, dead_code)]
#[repr(i32)]
enum rocblas_workmode {
    OutOfPlace = 201,
    InPlace = 202,
}

extern "C" {
    fn rocblas_create_handle(handle: *mut rocblas_handle) -> c_int;
    fn rocblas_destroy_handle(handle: rocblas_handle) -> c_int;

    /// Non-batched thin SVD: A = U * diag(S) * V^T.
    /// V (not Vt) is returned — shape [n, min(m,n)] stored col-major.
    fn rocsolver_sgesvd(
        handle: rocblas_handle,
        left_svect: rocblas_svect,
        right_svect: rocblas_svect,
        m: rocblas_int,
        n: rocblas_int,
        a: *mut f32,
        lda: rocblas_int,
        s: *mut f32,
        u: *mut f32,
        ldu: rocblas_int,
        v: *mut f32,
        ldv: rocblas_int,
        e: *mut f32,
        fast_alg: rocblas_workmode,
        info: *mut rocblas_int,
    ) -> c_int;
}

// ── Handle RAII ───────────────────────────────────────────────────────────────

struct RocblasHandle(rocblas_handle);

impl RocblasHandle {
    fn new() -> GpuResult<Self> {
        let mut h: rocblas_handle = std::ptr::null_mut();
        let rc = unsafe { rocblas_create_handle(&mut h) };
        if rc != 0 || h.is_null() {
            return Err(GpuError::HipApiError {
                code: rc,
                description: format!("rocblas_create_handle failed (code {})", rc),
            });
        }
        Ok(RocblasHandle(h))
    }
}

impl Drop for RocblasHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { rocblas_destroy_handle(self.0) };
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute top-k truncated SVD for a single row-major `[rows, cols]` matrix.
///
/// Returns `(u_scaled [rows * k], vt [k * cols])` — same layout as `top_k_svd_quant`.
/// Thin wrapper around `gpu_svd_batch` with `batch_count = 1`.
pub fn gpu_svd_single(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    k: usize,
) -> GpuResult<(Vec<f32>, Vec<f32>)> {
    gpu_svd_batch(matrix, rows, cols, k, 1)
}

/// Compute top-k truncated SVD for `batch_count` row-major matrices using GPU.
///
/// Input: `batch_count` row-major `[rows, cols]` matrices concatenated.
/// Returns `(u_scaled, vt)`:
/// - `u_scaled`: `[batch_count * rows * k]` — left singular vectors, each
///   column j scaled by σⱼ.  Row-major `[rows, k]` per expert.
/// - `vt`:       `[batch_count * k * cols]` — row-major `[k, cols]` per expert.
///
/// One GPU call per expert (non-batched). VRAM ~= one expert's U + V + A
/// (~50-200 MB depending on shape) — freed after each expert.
pub fn gpu_svd_batch(
    matrices: &[f32],
    rows: usize,
    cols: usize,
    k: usize,
    batch_count: usize,
) -> GpuResult<(Vec<f32>, Vec<f32>)> {
    assert_eq!(matrices.len(), batch_count * rows * cols);
    let k = k.min(rows.min(cols));
    let mn = rows.min(cols);

    // rocSOLVER's m < n code path accesses memory differently than m >= n.
    // Strategy for m < n: pass the data as-is (row-major A [rows, cols] equals
    // col-major A^T [cols, rows]), swap m↔n, and swap which buffer receives U vs V.
    // The singular values and the final u_scaled/vt layout are unchanged.
    let transposed = rows < cols;
    let (m, n) = if transposed {
        (cols as rocblas_int, rows as rocblas_int) // rocSOLVER sees [cols × rows] = A^T
    } else {
        (rows as rocblas_int, cols as rocblas_int)
    };

    let handle = RocblasHandle::new()?;

    let mut all_u = Vec::<f32>::with_capacity(batch_count * rows * k);
    let mut all_vt = Vec::<f32>::with_capacity(batch_count * k * cols);

    // Reuse GPU buffers across experts to avoid repeated allocation overhead.
    // Sizes are fixed for the current (rows, cols) shape, so allocate once.
    // u_buf stores U of A col-major [rows, mn]; v_buf stores V of A col-major [cols, mn].
    // When transposed, these map to rocSOLVER's V and U respectively (sizes still match).
    let a_bytes = rows * cols * 4;
    let u_bytes = rows * mn * 4;
    let v_bytes = cols * mn * 4;
    let s_bytes = mn * 4;
    let e_bytes = mn.saturating_sub(1).max(1) * 4;
    let needed = a_bytes + u_bytes + v_bytes + s_bytes + e_bytes;

    let (free_vram, _) = ffi::hip_get_mem_info(0)?;
    if free_vram < vram_guard_bytes() + needed {
        return Err(GpuError::OutOfMemory {
            requested: needed,
            available: free_vram.saturating_sub(vram_guard_bytes()),
            hint: format!(
                "GPU SVD: not enough VRAM for [{rows}×{cols}] expert (need {} MB, \
                 {} MB safely available). Lower ROCMFORGE_DESKTOP_VRAM_GB.",
                needed / (1024 * 1024),
                free_vram.saturating_sub(vram_guard_bytes()) / (1024 * 1024),
            ),
        });
    }

    // Persistent GPU buffers (reused per expert).
    let mut a_buf = GpuBuffer::alloc(a_bytes)?;
    let mut u_buf = GpuBuffer::alloc(u_bytes)?;
    let mut v_buf = GpuBuffer::alloc(v_bytes)?;
    let mut s_buf = GpuBuffer::alloc(s_bytes)?;
    let mut e_buf = GpuBuffer::alloc(e_bytes)?;
    let mut info_buf = GpuBuffer::alloc(4)?;

    let mut s_host = vec![0.0f32; mn];
    let mut u_host = vec![0.0f32; rows * mn];
    let mut v_host = vec![0.0f32; cols * mn];
    let mut info_host = 0i32;

    for e in 0..batch_count {
        let src = &matrices[e * rows * cols..(e + 1) * rows * cols];

        // Upload input to GPU.
        //
        // Standard case (rows >= cols): transpose row-major [rows, cols] → col-major [rows, cols].
        //   rocSOLVER sees the matrix A as intended.
        //
        // Transposed case (rows < cols): upload row-major A [rows, cols] as-is.
        //   In col-major layout this is A^T [cols, rows], and since we pass m=cols, n=rows,
        //   rocSOLVER computes SVD of A^T.  U(A^T)=V(A), V(A^T)=U(A) — see buffer swap below.
        let col_major_storage = if transposed {
            None
        } else {
            let mut cm = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    cm[j * rows + i] = src[i * cols + j];
                }
            }
            Some(cm)
        };
        let input_bytes: &[u8] = if transposed {
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, a_bytes) }
        } else {
            let cm_ref = col_major_storage.as_ref().expect("invariant: col_major_storage set when !transposed");
            unsafe { std::slice::from_raw_parts(cm_ref.as_ptr() as *const u8, a_bytes) }
        };
        a_buf.copy_from_host(input_bytes)?;

        // Assign U/V buffers and leading dimensions.
        // Standard: rocSOLVER writes U→u_buf (ldu=m=rows), V→v_buf (ldv=n=cols).
        // Transposed: rocSOLVER writes U(A^T)=V(A)→v_buf (ldu=m=cols), V(A^T)=U(A)→u_buf (ldv=n=rows).
        let (u_ptr, ldu, v_ptr, ldv) = if transposed {
            (
                v_buf.as_ptr() as *mut f32,
                m, // v_buf receives V(A), col-major [cols, mn]
                u_buf.as_ptr() as *mut f32,
                n,
            ) // u_buf receives U(A), col-major [rows, mn]
        } else {
            (u_buf.as_ptr() as *mut f32, m, v_buf.as_ptr() as *mut f32, n)
        };

        let rc = unsafe {
            rocsolver_sgesvd(
                handle.0,
                rocblas_svect::Singular,
                rocblas_svect::Singular,
                m,
                n,
                a_buf.as_ptr() as *mut f32,
                m, // lda = m (col-major)
                s_buf.as_ptr() as *mut f32,
                u_ptr,
                ldu,
                v_ptr,
                ldv,
                e_buf.as_ptr() as *mut f32,
                rocblas_workmode::OutOfPlace,
                info_buf.as_ptr() as *mut c_int,
            )
        };
        if rc != 0 {
            return Err(GpuError::HipApiError {
                code: rc,
                description: format!(
                    "rocsolver_sgesvd failed (code {}) for expert {e} [{rows}×{cols}]",
                    rc
                ),
            });
        }

        ffi::hip_device_synchronize()?;

        // Check convergence
        let info_bytes =
            unsafe { std::slice::from_raw_parts_mut(&mut info_host as *mut i32 as *mut u8, 4) };
        ffi::hip_memcpy_d2h(info_bytes.as_mut_ptr(), info_buf.as_ptr(), 4)?;
        if info_host != 0 {
            return Err(GpuError::HipApiError {
                code: info_host,
                description: format!(
                    "rocSOLVER SVD did not converge for expert {e} (info={})",
                    info_host
                ),
            });
        }

        // Download S, U(A), V(A).
        // Regardless of the transposed flag, u_buf holds U(A) and v_buf holds V(A)
        // (the buffer swap above ensures this).
        let dl = |buf: &GpuBuffer, dst: &mut [f32]| -> GpuResult<()> {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len() * 4)
            };
            ffi::hip_memcpy_d2h(bytes.as_mut_ptr(), buf.as_ptr(), dst.len() * 4)
        };
        dl(&s_buf, &mut s_host)?;
        dl(&u_buf, &mut u_host)?;
        dl(&v_buf, &mut v_host)?;

        // u_buf: col-major U(A) [rows, mn] in standard case, col-major VT(A^T) [rows, rows] in transposed case.
        if transposed {
            for i in 0..rows {
                for j in 0..k {
                    all_u.push(u_host[i * rows + j] * s_host[j]);
                }
            }
        } else {
            for i in 0..rows {
                for j in 0..k {
                    all_u.push(u_host[j * rows + i] * s_host[j]);
                }
            }
        }

        // v_buf: col-major VT(A) [mn, cols] in standard case, col-major V(A) [cols, mn] in transposed case.
        if transposed {
            for j in 0..k {
                for i in 0..cols {
                    all_vt.push(v_host[j * cols + i]);
                }
            }
        } else {
            for j in 0..k {
                for i in 0..cols {
                    all_vt.push(v_host[i * cols + j]);
                }
            }
        }

        if e % 32 == 0 {
            eprint!(".");
        }
    }
    eprintln!(); // newline after dots
    Ok((all_u, all_vt))
}
