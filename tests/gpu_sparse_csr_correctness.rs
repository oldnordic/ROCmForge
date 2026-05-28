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

/// CPU reference: y = A * x where A is dense
fn dense_gemv_reference(a: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    for i in 0..rows {
        let mut sum = 0.0f32;
        for j in 0..cols {
            sum += a[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
    y
}

/// Convert dense matrix to CSR format
fn dense_to_csr(a: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_offsets = vec![0u32; rows + 1];

    for i in 0..rows {
        for j in 0..cols {
            let v = a[i * cols + j];
            if v != 0.0f32 {
                values.push(v);
                col_indices.push(j as u32);
            }
        }
        row_offsets[i + 1] = values.len() as u32;
    }

    (values, col_indices, row_offsets)
}

#[test]
#[serial]
fn test_sparse_csr_gemv_basic() {
    let device = init_device();
    let rows = 8usize;
    let cols = 6usize;

    // Random-ish dense matrix
    let mut a = vec![0.0f32; rows * cols];
    for (i, v) in a.iter_mut().enumerate() {
        *v = (i as f32) * 0.01 - 0.15;
    }
    // Make it sparse: zero out ~half
    for i in (0..a.len()).step_by(3) {
        a[i] = 0.0f32;
    }

    let mut x = vec![0.0f32; cols];
    for (i, v) in x.iter_mut().enumerate() {
        *v = (i as f32) * 0.03 - 0.1;
    }

    let ref_y = dense_gemv_reference(&a, rows, cols, &x);

    let (values, col_indices, row_offsets) = dense_to_csr(&a, rows, cols);
    let nnz = values.len();

    let gpu_values = expect(upload_f32(&values));
    let gpu_cols = expect(upload_u32(&col_indices));
    let gpu_rows = expect(upload_u32(&row_offsets));
    let gpu_x = expect(upload_f32(&x));
    let gpu_y = expect(upload_f32(&vec![0.0f32; rows]));

    expect(dispatch_sparse_csr_gemv_f32(
        gpu_values.as_ptr() as *const f32,
        gpu_cols.as_ptr() as *const u32,
        gpu_rows.as_ptr() as *const u32,
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
            "Sparse CSR GEMV mismatch at {}: actual={}, ref={}",
            i,
            out_y[i],
            ref_y[i]
        );
    }
}

#[test]
#[serial]
fn test_sparse_csr_gemv_identity() {
    let device = init_device();
    let n = 8usize;

    // Identity matrix as sparse CSR
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0f32;
    }

    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 0.5).collect();
    let ref_y = dense_gemv_reference(&a, n, n, &x);

    let (values, col_indices, row_offsets) = dense_to_csr(&a, n, n);

    let gpu_values = expect(upload_f32(&values));
    let gpu_cols = expect(upload_u32(&col_indices));
    let gpu_rows = expect(upload_u32(&row_offsets));
    let gpu_x = expect(upload_f32(&x));
    let gpu_y = expect(upload_f32(&vec![0.0f32; n]));

    expect(dispatch_sparse_csr_gemv_f32(
        gpu_values.as_ptr() as *const f32,
        gpu_cols.as_ptr() as *const u32,
        gpu_rows.as_ptr() as *const u32,
        values.len(),
        n,
        n,
        gpu_x.as_ptr() as *const f32,
        gpu_y.as_ptr() as *mut f32,
        device.stream(),
    ));

    expect(device.synchronize());

    let out_y = expect(download_f32(&gpu_y, n));

    for i in 0..n {
        assert!(
            (out_y[i] - ref_y[i]).abs() < 1e-4,
            "Sparse CSR identity mismatch at {}: actual={}, ref={}",
            i,
            out_y[i],
            ref_y[i]
        );
    }
}

#[test]
#[serial]
fn test_sparse_csr_gemv_diagonal() {
    let device = init_device();
    let n = 16usize;

    // Diagonal matrix with varying values
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        a[i * n + i] = (i as f32 + 1.0) * 0.5;
    }

    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let ref_y = dense_gemv_reference(&a, n, n, &x);

    let (values, col_indices, row_offsets) = dense_to_csr(&a, n, n);

    let gpu_values = expect(upload_f32(&values));
    let gpu_cols = expect(upload_u32(&col_indices));
    let gpu_rows = expect(upload_u32(&row_offsets));
    let gpu_x = expect(upload_f32(&x));
    let gpu_y = expect(upload_f32(&vec![0.0f32; n]));

    expect(dispatch_sparse_csr_gemv_f32(
        gpu_values.as_ptr() as *const f32,
        gpu_cols.as_ptr() as *const u32,
        gpu_rows.as_ptr() as *const u32,
        values.len(),
        n,
        n,
        gpu_x.as_ptr() as *const f32,
        gpu_y.as_ptr() as *mut f32,
        device.stream(),
    ));

    expect(device.synchronize());

    let out_y = expect(download_f32(&gpu_y, n));

    for i in 0..n {
        assert!(
            (out_y[i] - ref_y[i]).abs() < 1e-4,
            "Sparse CSR diagonal mismatch at {}: actual={}, ref={}",
            i,
            out_y[i],
            ref_y[i]
        );
    }
}

#[test]
#[serial]
fn test_sparse_csr_gemv_empty_row() {
    let device = init_device();
    let rows = 6usize;
    let cols = 4usize;

    // Matrix where row 2 is all zeros
    let mut a = vec![0.0f32; rows * cols];
    for i in 0..rows {
        if i == 2 {
            continue;
        }
        for j in 0..cols {
            a[i * cols + j] = (i as f32) * 0.1 + (j as f32) * 0.05;
        }
    }

    let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.2 + 0.1).collect();
    let ref_y = dense_gemv_reference(&a, rows, cols, &x);

    let (values, col_indices, row_offsets) = dense_to_csr(&a, rows, cols);

    let gpu_values = expect(upload_f32(&values));
    let gpu_cols = expect(upload_u32(&col_indices));
    let gpu_rows = expect(upload_u32(&row_offsets));
    let gpu_x = expect(upload_f32(&x));
    let gpu_y = expect(upload_f32(&vec![0.0f32; rows]));

    expect(dispatch_sparse_csr_gemv_f32(
        gpu_values.as_ptr() as *const f32,
        gpu_cols.as_ptr() as *const u32,
        gpu_rows.as_ptr() as *const u32,
        values.len(),
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
            "Sparse CSR empty-row mismatch at {}: actual={}, ref={}",
            i,
            out_y[i],
            ref_y[i]
        );
    }
}
