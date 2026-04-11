#![cfg(feature = "gpu")]

mod common;

use rocmforge::gpu::{self, GpuBuffer, GpuDevice, TensorRole, WeightMeta};
use rocmforge::loader::GgmlType;
use serial_test::serial;

#[test]
#[serial]
fn test_gemv_q4_0_lds_fallback() {
    require_gpu!();
    require_vram!(1);

    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("Failed to init GPU");
    let h = 32768; // Large enough to exceed typical 64KB LDS limit (32768 * 4 = 128KB)
    let out_dim = 128;

    // Create weights
    let weights_data = vec![0u8; (h * out_dim / 32) * 18];
    let mut gpu_weights = GpuBuffer::alloc(weights_data.len()).unwrap();
    gpu_weights.copy_from_host(&weights_data).unwrap();

    let meta = WeightMeta {
        dims: vec![out_dim as u64, h as u64],
        wtype: GgmlType::Q4_0,
        role: TensorRole::Generic,
        needs_transpose: false,
    };

    // Create input/output
    let input_data = vec![1.0f32; h];
    let mut gpu_input = GpuBuffer::alloc(h * 4).unwrap();
    gpu_input
        .copy_from_host(unsafe {
            std::slice::from_raw_parts(input_data.as_ptr() as *const u8, h * 4)
        })
        .unwrap();

    let gpu_output = GpuBuffer::alloc(out_dim * 4).unwrap();

    // This call should hit the LDS fallback in the orchestrator
    let result = gpu::ops::gpu_dispatch_gemv(
        &device,
        &gpu_weights,
        &meta,
        gpu_input.as_ptr() as *const f32,
        gpu_output.as_ptr() as *mut f32,
        out_dim,
        h,
    );

    assert!(
        result.is_ok(),
        "GEMV should succeed via fallback even if LDS limit exceeded"
    );
}

#[test]
#[serial]
fn test_rms_norm_alignment_fallback() {
    require_gpu!();
    require_vram!(1);

    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("Failed to init GPU");
    let n = 1024;

    let x = vec![1.0f32; n + 4]; // Extra padding to allow unaligned offset
    let weight = vec![1.0f32; n];

    let mut gpu_x_full = GpuBuffer::alloc((n + 4) * 4).unwrap();
    let mut gpu_weight = GpuBuffer::alloc(n * 4).unwrap();
    let gpu_out = GpuBuffer::alloc(n * 4).unwrap();

    gpu_x_full
        .copy_from_host(unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, (n + 4) * 4) })
        .unwrap();
    gpu_weight
        .copy_from_host(unsafe { std::slice::from_raw_parts(weight.as_ptr() as *const u8, n * 4) })
        .unwrap();

    // Get an unaligned pointer by adding 4 bytes (1 float) offset
    let unaligned_x = unsafe { (gpu_x_full.as_ptr() as *const f32).add(1) };

    // This call should hit the alignment fallback in the orchestrator
    let result = gpu::ops::gpu_dispatch_rms_norm(
        &device,
        unaligned_x,
        gpu_weight.as_ptr() as *const f32,
        gpu_out.as_ptr() as *mut f32,
        n,
        1e-5,
        device.stream(),
    );

    assert!(
        result.is_ok(),
        "RMS Norm should succeed via fallback even if pointers are unaligned"
    );
}
