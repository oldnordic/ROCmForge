#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::gpu::kernels::dispatch_sparse_csr_gemv_f32;
use rocmforge::gpu::{GpuBuffer, GpuDevice};
use serial_test::serial;

fn upload_f32(data: &[f32]) -> rocmforge::gpu::GpuResult<GpuBuffer> {
    let mut buf = GpuBuffer::alloc(std::mem::size_of_val(data))?;
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    buf.copy_from_host(bytes)?;
    Ok(buf)
}

fn upload_u32(data: &[u32]) -> rocmforge::gpu::GpuResult<GpuBuffer> {
    let mut buf = GpuBuffer::alloc(std::mem::size_of_val(data))?;
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    buf.copy_from_host(bytes)?;
    Ok(buf)
}

fn download_f32(buf: &GpuBuffer, len: usize) -> rocmforge::gpu::GpuResult<Vec<f32>> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    buf.copy_to_host(&mut bytes)?;
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() })
}

fn init_device() -> GpuDevice {
    GpuDevice::init(0).expect("GPU should initialize")
}

fn expect<T>(r: rocmforge::gpu::GpuResult<T>) -> T {
    r.expect("GPU operation should succeed")
}

/// Build a sparse CSR matrix from dense, returning (values, col_indices, row_offsets, nnz)
fn dense_to_csr(matrix: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<u32>, Vec<u32>, usize) {
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_offsets = vec![0u32; rows + 1];

    for r in 0..rows {
        for c in 0..cols {
            let v = matrix[r * cols + c];
            if v != 0.0 {
                values.push(v);
                col_indices.push(c as u32);
            }
        }
        row_offsets[r + 1] = values.len() as u32;
    }

    let nnz = values.len();
    (values, col_indices, row_offsets, nnz)
}

#[test]
#[serial]
fn test_sparse_csr_gemv_basic() {
    let device = init_device();
    let rows = 8usize;
    let cols = 16usize;

    // Build a sparse matrix with ~25% nonzeros
    let mut dense = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            if (r + c * 3) % 4 == 0 {
                dense[r * cols + c] = ((r * cols + c) as f32) * 0.01 - 0.5;
            }
        }
    }

    let mut x = vec![0.0f32; cols];
    for (i, v) in x.iter_mut().enumerate() {
        *v = (i as f32) * 0.05 - 0.3;
    }

    // CPU reference: y = A * x
    let mut ref_y = vec![0.0f32; rows];
    for r in 0..rows {
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += dense[r * cols + c] * x[c];
        }
        ref_y[r] = sum;
    }

    // Convert to CSR
    let (values, col_indices, row_offsets, nnz) = dense_to_csr(&dense, rows, cols);

    // GPU execution
    let gpu_values = expect(upload_f32(&values));
    let gpu_col_idx = expect(upload_u32(&col_indices));
    let gpu_row_off = expect(upload_u32(&row_offsets));
    let gpu_x = expect(upload_f32(&x));
    let gpu_y = expect(upload_f32(&vec![0.0f32; rows]));

    expect(dispatch_sparse_csr_gemv_f32(
        gpu_values.as_ptr() as *const f32,
        gpu_col_idx.as_ptr() as *const u32,
        gpu_row_off.as_ptr() as *const u32,
        nnz,
        rows,
        cols,
        gpu_x.as_ptr() as *const f32,
        gpu_y.as_ptr() as *mut f32,
        device.stream(),
    ));

    expect(device.synchronize());

    let out_y = expect(download_f32(&gpu_y, rows));

    for i in 0..rows {
        assert!(
            (out_y[i] - ref_y[i]).abs() < 1e-4,
            "sparse CSR GEMV mismatch at {}: actual={}, ref={}",
            i,
            out_y[i],
            ref_y[i]
        );
    }
}

#[test]
#[serial]
fn test_sparse_csr_gemv_all_nonzero() {
    let device = init_device();
    let rows = 4usize;
    let cols = 8usize;

    // Dense matrix (all non-zero → still valid CSR)
    let mut dense = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dense[r * cols + c] = (r as f32) * 0.1 + (c as f32) * 0.05 - 0.3;
        }
    }

    let mut x = vec![0.0f32; cols];
    for (i, v) in x.iter_mut().enumerate() {
        *v = (i as f32) * 0.02 - 0.1;
    }

    // CPU reference
    let mut ref_y = vec![0.0f32; rows];
    for r in 0..rows {
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += dense[r * cols + c] * x[c];
        }
        ref_y[r] = sum;
    }

    let (values, col_indices, row_offsets, nnz) = dense_to_csr(&dense, rows, cols);

    let gpu_values = expect(upload_f32(&values));
    let gpu_col_idx = expect(upload_u32(&col_indices));
    let gpu_row_off = expect(upload_u32(&row_offsets));
    let gpu_x = expect(upload_f32(&x));
    let gpu_y = expect(upload_f32(&vec![0.0f32; rows]));

    expect(dispatch_sparse_csr_gemv_f32(
        gpu_values.as_ptr() as *const f32,
        gpu_col_idx.as_ptr() as *const u32,
        gpu_row_off.as_ptr() as *const u32,
        nnz,
        rows,
        cols,
        gpu_x.as_ptr() as *const f32,
        gpu_y.as_ptr() as *mut f32,
        device.stream(),
    ));

    expect(device.synchronize());

    let out_y = expect(download_f32(&gpu_y, rows));

    for i in 0..rows {
        assert!(
            (out_y[i] - ref_y[i]).abs() < 1e-4,
            "dense-as-CSR GEMV mismatch at {}: actual={}, ref={}",
            i,
            out_y[i],
            ref_y[i]
        );
    }
}

#[test]
#[serial]
fn test_sparse_csr_gemv_empty_rows() {
    let device = init_device();
    let rows = 6usize;
    let cols = 8usize;

    // Row 2 and row 4 are completely zero
    let mut dense = vec![0.0f32; rows * cols];
    for r in 0..rows {
        if r == 2 || r == 4 {
            continue;
        }
        for c in 0..cols {
            if c % 2 == 0 {
                dense[r * cols + c] = (r as f32) * 0.2 + (c as f32) * 0.1;
            }
        }
    }

    let mut x = vec![0.0f32; cols];
    for (i, v) in x.iter_mut().enumerate() {
        *v = (i as f32) * 0.03 - 0.12;
    }

    // CPU reference
    let mut ref_y = vec![0.0f32; rows];
    for r in 0..rows {
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += dense[r * cols + c] * x[c];
        }
        ref_y[r] = sum;
    }

    let (values, col_indices, row_offsets, nnz) = dense_to_csr(&dense, rows, cols);

    let gpu_values = expect(upload_f32(&values));
    let gpu_col_idx = expect(upload_u32(&col_indices));
    let gpu_row_off = expect(upload_u32(&row_offsets));
    let gpu_x = expect(upload_f32(&x));
    let gpu_y = expect(upload_f32(&vec![0.0f32; rows]));

    expect(dispatch_sparse_csr_gemv_f32(
        gpu_values.as_ptr() as *const f32,
        gpu_col_idx.as_ptr() as *const u32,
        gpu_row_off.as_ptr() as *const u32,
        nnz,
        rows,
        cols,
        gpu_x.as_ptr() as *const f32,
        gpu_y.as_ptr() as *mut f32,
        device.stream(),
    ));

    expect(device.synchronize());

    let out_y = expect(download_f32(&gpu_y, rows));

    for i in 0..rows {
        assert!(
            (out_y[i] - ref_y[i]).abs() < 1e-4,
            "empty-row CSR GEMV mismatch at {}: actual={}, ref={}",
            i,
            out_y[i],
            ref_y[i]
        );
    }
}
