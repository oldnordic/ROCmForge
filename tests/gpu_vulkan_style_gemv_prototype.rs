#![cfg(feature = "gpu")]

mod common;

use rocmforge::config::ModelConfig;
use rocmforge::cpu::{cache::CpuForwardScratch, weights::CpuModelWeights};
use rocmforge::gpu::{self, GpuBuffer, GpuDevice};
use rocmforge::loader::{GgmlType, GgufFile};
use serial_test::serial;
use std::time::Instant;

const MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

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
fn test_gpu_vulkan_style_ffn_down_multi_row_q4_0() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    require_experimental_gpu_tests!();
    require_gpu!();
    require_vram!(4);

    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse model config");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("GPU weights should load");
    let mut layer_idx = 0;
    for i in 0..config.num_layers {
        if gpu_weights.layer(i).ffn_down_meta.wtype == GgmlType::Q4_0 {
            layer_idx = i;
            break;
        }
    }
    let layer = gpu_weights.layer(layer_idx);

    if layer.ffn_down_meta.wtype != GgmlType::Q4_0 {
        eprintln!("Skipping test: no Q4_0 ffn_down layer found in this model");
        return;
    }
    eprintln!("Testing on layer {}", layer_idx);

    let h = config.hidden_size;
    let ff_size = config.intermediate_size;

    // Create random input
    let mut input = vec![0.0f32; ff_size];
    for i in 0..ff_size {
        input[i] = (i as f32).sin();
    }

    let d_input = upload_f32(&input);
    let d_out_production = GpuBuffer::alloc(h * 4).expect("alloc production out");
    let d_out_vulkan = GpuBuffer::alloc(h * 4).expect("alloc vulkan out");

    let n_waves = 8; // ffn_down is larger, usually 8 waves is better

    // Warmup
    gpu::kernels::quant::gemv_q4_0_f32_on_stream(
        layer.ffn_down.as_ptr(),
        d_input.as_ptr() as *const f32,
        d_out_production.as_ptr() as *mut f32,
        ff_size,
        h,
        device.stream(),
    )
    .expect("production warmup");

    gpu::kernels::quant::gemv_q4_0_f32_vulkan_style(
        &device,
        layer.ffn_down.as_ptr(),
        d_input.as_ptr() as *const f32,
        d_out_vulkan.as_ptr() as *mut f32,
        ff_size,
        h,
        n_waves,
        device.stream(),
    )
    .expect("vulkan warmup");

    device.synchronize().expect("warmup sync");

    // Benchmark Production
    let production_ms = average_kernel_ms(&device, 200, || {
        gpu::kernels::quant::gemv_q4_0_f32_on_stream(
            layer.ffn_down.as_ptr(),
            d_input.as_ptr() as *const f32,
            d_out_production.as_ptr() as *mut f32,
            ff_size,
            h,
            device.stream(),
        )
        .expect("production run");
    });

    // Benchmark Vulkan-Style
    let vulkan_ms = average_kernel_ms(&device, 200, || {
        gpu::kernels::quant::gemv_q4_0_f32_vulkan_style(
            &device,
            layer.ffn_down.as_ptr(),
            d_input.as_ptr() as *const f32,
            d_out_vulkan.as_ptr() as *mut f32,
            ff_size,
            h,
            n_waves,
            device.stream(),
        )
        .expect("vulkan run");
    });

    let production_out = download_f32(&d_out_production, h);
    let vulkan_out = download_f32(&d_out_vulkan, h);
    let cross_err = max_abs_error(&production_out, &vulkan_out);

    eprintln!(
        "vulkan_style_q4_0_ffn_down production_ms={:.4} vulkan_ms={:.4} speedup={:.3} cross_err={:.6}",
        production_ms,
        vulkan_ms,
        production_ms / vulkan_ms,
        cross_err
    );

    assert!(
        cross_err <= 1e-5,
        "Vulkan-style GEMV output diverged from production: cross_err={}",
        cross_err
    );
}
