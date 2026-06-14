#![cfg(feature = "gpu")]
#![allow(warnings)]

//! Performance and correctness test for FFN layers using synthetic data.

mod common;

use common::helpers::*;
use rocmforge::config::ModelConfig;
use rocmforge::gpu::{GpuBuffer, GpuDevice, GpuForwardScratch, WeightMeta};
use rocmforge::loader::GgmlType;
use serial_test::serial;
use std::time::Instant;

#[test]
#[serial]
fn test_ffn_performance_synthetic() {
    require_gpu!();
    let device = GpuDevice::init(0).expect("GPU init failed");
    let dev_id = device.device_id();
    let config = mock_model_config();
    let h = config.hidden_size;
    let ff = config.intermediate_size;

    eprintln!("=== FFN Performance (Synthetic) ===");
    eprintln!("h={}, ff={}", h, ff);

    // 1. Setup synthetic weights (Q4_0)
    let up_data = (0..ff * h)
        .map(|i| (i as f32).sin() * 0.1)
        .collect::<Vec<f32>>();
    let down_data = (0..h * ff)
        .map(|i| (i as f32).cos() * 0.1)
        .collect::<Vec<f32>>();

    let up_q4 = quantize_q4_0(&up_data);
    let down_q4 = quantize_q4_0(&down_data);

    let up_gpu = upload_raw(dev_id, &up_q4);
    let down_gpu = upload_raw(dev_id, &down_q4);

    let up_meta = mock_gpu_meta(ff, h, GgmlType::Q4_0, rocmforge::gpu::TensorRole::Generic);
    let down_meta = mock_gpu_meta(h, ff, GgmlType::Q4_0, rocmforge::gpu::TensorRole::Generic);

    // 2. Setup input
    let input_f32 = (0..h).map(|i| (i as f32) * 0.01).collect::<Vec<f32>>();
    let input_gpu = upload_f32(dev_id, &input_f32);
    let mut output_gpu = GpuBuffer::alloc_for_device(h * 4, dev_id).expect("alloc output_gpu");
    let mut swiglu_gpu = GpuBuffer::alloc_for_device(ff * 4, dev_id).expect("alloc swiglu_gpu");

    // 3. Benchmark
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        rocmforge::gpu::ops::gpu_dispatch_gemv_on_stream(
            &device,
            &up_gpu,
            &up_meta,
            input_gpu.as_ptr() as *const f32,
            swiglu_gpu.as_ptr() as *mut f32,
            ff,
            h,
            device.stream(),
        )
        .expect("gemv up");

        rocmforge::gpu::ops::gpu_dispatch_gemv_on_stream(
            &device,
            &down_gpu,
            &down_meta,
            swiglu_gpu.as_ptr() as *const f32,
            output_gpu.as_ptr() as *mut f32,
            h,
            ff,
            device.stream(),
        )
        .expect("gemv down");
    }
    device.synchronize().expect("sync");
    let elapsed = start.elapsed();

    eprintln!(
        "Avg FFN (up+down) time: {:?} / iter",
        elapsed / (iters as u32)
    );
}

fn upload_raw(device_id: i32, bytes: &[u8]) -> GpuBuffer {
    let mut buf =
        GpuBuffer::alloc_for_device(bytes.len(), device_id).expect("helper: upload_raw alloc");
    buf.copy_from_host(bytes).expect("helper: upload_raw copy");
    buf
}
