#![cfg(feature = "gpu")]

mod common;

use rocmforge::gpu::{self, GpuBuffer, GpuDevice};
use serial_test::serial;
use std::time::Instant;

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max)
}

fn upload_f32(data: &[f32]) -> GpuBuffer {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    let mut buffer = GpuBuffer::alloc(std::mem::size_of_val(data)).expect("alloc GPU buffer");
    buffer.copy_from_host(bytes).expect("upload GPU buffer");
    buffer
}

fn download_f32(buf: &GpuBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    buf.copy_to_host(&mut bytes).expect("download GPU buffer");
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() }
}

fn average_kernel_ms<F>(device: &GpuDevice, iters: usize, mut launch: F) -> f64
where
    F: FnMut(),
{
    let start = Instant::now();
    for _ in 0..iters {
        launch();
        device.synchronize().expect("stream synchronize");
    }
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

#[test]
#[serial]
fn test_gpu_rms_norm_vulkan_style_shuffles() {
    let _lock = match common::GpuLock::acquire() {
        Ok(lock) => lock,
        Err(err) => {
            eprintln!("Skipping test: {}", err);
            return;
        }
    };

    require_gpu!();
    require_vram!(1);

    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let n = 4096; // Standard hidden size
    let eps = 1e-5f32;

    let mut x = vec![0.0f32; n];
    let mut weight = vec![0.0f32; n];
    for i in 0..n {
        x[i] = (i as f32).sin();
        weight[i] = (i as f32).cos().abs();
    }

    let d_x = upload_f32(&x);
    let d_weight = upload_f32(&weight);
    let d_out_production = GpuBuffer::alloc(n * 4).expect("alloc production out");
    let d_out_vulkan = GpuBuffer::alloc(n * 4).expect("alloc vulkan out");

    // Warmup
    gpu::kernels::norm::rms_norm_on_stream(
        d_x.as_ptr() as *const f32,
        d_weight.as_ptr() as *const f32,
        d_out_production.as_ptr() as *mut f32,
        n,
        eps,
        device.stream(),
    )
    .expect("production warmup");

    gpu::kernels::norm::rms_norm_vulkan_style(
        d_x.as_ptr() as *const f32,
        d_weight.as_ptr() as *const f32,
        d_out_vulkan.as_ptr() as *mut f32,
        n,
        eps,
        device.stream(),
    )
    .expect("vulkan warmup");

    device.synchronize().expect("warmup sync");

    // Benchmark Production
    let production_ms = average_kernel_ms(&device, 1000, || {
        gpu::kernels::norm::rms_norm_on_stream(
            d_x.as_ptr() as *const f32,
            d_weight.as_ptr() as *const f32,
            d_out_production.as_ptr() as *mut f32,
            n,
            eps,
            device.stream(),
        )
        .expect("production run");
    });

    // Benchmark Vulkan-Style
    let vulkan_ms = average_kernel_ms(&device, 1000, || {
        gpu::kernels::norm::rms_norm_vulkan_style(
            d_x.as_ptr() as *const f32,
            d_weight.as_ptr() as *const f32,
            d_out_vulkan.as_ptr() as *mut f32,
            n,
            eps,
            device.stream(),
        )
        .expect("vulkan run");
    });

    let out_prod = download_f32(&d_out_production, n);
    let out_vulk = download_f32(&d_out_vulkan, n);
    let cross_err = max_abs_error(&out_prod, &out_vulk);

    eprintln!(
        "rms_norm_vulkan_style_shuffles n={} production_ms={:.6} vulkan_ms={:.6} speedup={:.3} cross_err={:.8}",
        n,
        production_ms,
        vulkan_ms,
        production_ms / vulkan_ms,
        cross_err
    );

    assert!(
        cross_err <= 1e-5,
        "Vulkan-style RMS Norm diverged: cross_err={}",
        cross_err
    );
}
