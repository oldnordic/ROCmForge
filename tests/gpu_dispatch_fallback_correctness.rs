#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::gpu::device::GpuDevice;
use rocmforge::gpu::ops::gpu_dispatch_gemv_with_fallback_on_stream;
use rocmforge::gpu::weights::{GpuBuffer, GpuMpoWeights, GpuSparseCsrWeights, WeightMeta};
use rocmforge::gpu::TensorRole;
use rocmforge::loader::GgmlType;
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

#[test]
#[serial]
fn test_fallback_sparse_csr() {
    if !rocmforge::gpu::safety::run_experimental_gpu_tests_enabled() {
        eprintln!("Skipping test_fallback_sparse_csr — set ROCMFORGE_RUN_EXPERIMENTAL_GPU_TESTS=1 to enable");
        return;
    }
    let device = init_device();
    let rows = 4usize;
    let cols = 4usize;
    let nnz = 4usize;

    let values: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let col_idx: Vec<u32> = vec![0, 1, 2, 3];
    let row_ptr: Vec<u32> = vec![0, 1, 2, 3, 4];
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let values_buf = expect(upload_f32(&values));
    let col_idx_buf = expect(upload_u32(&col_idx));
    let row_ptr_buf = expect(upload_u32(&row_ptr));
    let input = expect(upload_f32(&input_data));
    let gpu_y = expect(upload_f32(&vec![0.0f32; rows]));

    let sparse_weights = GpuSparseCsrWeights {
        values: values_buf,
        col_idx: col_idx_buf,
        row_ptr: row_ptr_buf,
        rows,
        cols,
        nnz,
    };

    // Dummy dense weights/meta (should be ignored)
    let dummy_weights = GpuBuffer::empty();
    let dummy_meta = WeightMeta {
        wtype: GgmlType::F32,
        dims: vec![rows as u64, cols as u64],
        needs_transpose: false,
        role: TensorRole::Generic,
        svd_k: None,
    };

    expect(gpu_dispatch_gemv_with_fallback_on_stream(
        &device,
        &dummy_weights,
        &dummy_meta,
        None,
        Some(&sparse_weights),
        None,
        input.as_ptr() as *const f32,
        gpu_y.as_ptr() as *mut f32,
        rows,
        cols,
        std::ptr::null_mut(),
        device.stream(),
    ));

    expect(device.synchronize());

    let out_y = expect(download_f32(&gpu_y, rows));

    for i in 0..rows {
        assert!(
            (out_y[i] - input_data[i]).abs() < 1e-4,
            "sparse fallback mismatch at {}: expected {} got {}",
            i,
            input_data[i],
            out_y[i]
        );
    }
}

#[test]
#[serial]
fn test_fallback_mpo() {
    if !rocmforge::gpu::safety::run_experimental_gpu_tests_enabled() {
        eprintln!(
            "Skipping test_fallback_mpo — set ROCMFORGE_RUN_EXPERIMENTAL_GPU_TESTS=1 to enable"
        );
        return;
    }
    let device = init_device();

    // 2-site MPO: d1=2, d2=2, chi=2
    let d1 = 2u32;
    let d2 = 2u32;
    let chi = 2u32;
    let out_dim = (d1 * d2) as usize;
    let in_dim = out_dim;

    // Site dims: [1, d1, chi, 1, chi, d2, 1, 1] for n_sites=2
    let site_dims: Vec<u32> = vec![1, d1, chi, 1, chi, d2, 1, 1];

    // Site data: 2 sites
    // Site 0: [1, d1, chi] = [1, 2, 2] = 4 floats (identity-like)
    // Site 1: [chi, d2, 1] = [2, 2, 1] = 4 floats (identity-like)
    let site_data: Vec<f32> = vec![
        // Site 0: 1x2x2 = 4 floats
        1.0, 0.0, 0.0, 1.0, // Site 1: 2x2x1 = 4 floats
        1.0, 0.0, 0.0, 1.0,
    ];

    let input_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];

    let site_data_buf = expect(upload_f32(&site_data));
    let input = expect(upload_f32(&input_data));
    let gpu_y = expect(upload_f32(&vec![0.0f32; d1 as usize]));

    let site_dims_buf = expect(upload_u32(&site_dims));

    let mpo_weights = GpuMpoWeights {
        site_data: site_data_buf,
        site_dims: site_dims_buf,
        n_sites: 2,
    };

    // Dummy dense weights/meta (should be ignored)
    let dummy_weights = GpuBuffer::empty();
    let dummy_meta = WeightMeta {
        wtype: GgmlType::F32,
        dims: vec![out_dim as u64, in_dim as u64],
        needs_transpose: false,
        role: TensorRole::Generic,
        svd_k: None,
    };

    expect(gpu_dispatch_gemv_with_fallback_on_stream(
        &device,
        &dummy_weights,
        &dummy_meta,
        None,
        None,
        Some(&mpo_weights),
        input.as_ptr() as *const f32,
        gpu_y.as_ptr() as *mut f32,
        d1 as usize, // out_dim = d1
        d2 as usize, // in_dim = d2
        std::ptr::null_mut(),
        device.stream(),
    ));

    expect(device.synchronize());

    let out_y = expect(download_f32(&gpu_y, d1 as usize));

    // With identity-like MPO, output should equal input (first d1 elements)
    for i in 0..(d1 as usize) {
        assert!(
            (out_y[i] - input_data[i]).abs() < 1e-3,
            "mpo fallback mismatch at {}: expected {} got {}",
            i,
            input_data[i],
            out_y[i]
        );
    }
}
