#![cfg(feature = "gpu")]

mod common;

use rocmforge::config::{detect_chat_template, ModelConfig};
use rocmforge::cpu::{
    cache::CpuForwardScratch, forward::cpu_embed_token, ops::rms_norm, weights::CpuModelWeights,
};
use rocmforge::gpu::{self, GpuBuffer, GpuDevice};
use rocmforge::loader::{GgmlType, GgufFile};
use rocmforge::tokenizer::BpeTokenizer;
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
fn test_gpu_vulkan_style_gate_up_interleaved_q4_0() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let _lock = match common::GpuLock::acquire() {
        Ok(lock) => lock,
        Err(err) => {
            eprintln!("Skipping test: {}", err);
            return;
        }
    };

    require_gpu!();
    require_vram!(4);

    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse model config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("CPU weights should load");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("GPU weights should load");
    let layer = gpu_weights.layer(0);

    assert_eq!(
        layer.ffn_gate_meta.wtype,
        GgmlType::Q4_0,
        "expected layer-0 weights to be Q4_0 for this experiment"
    );

    // Prepare input from real model forward pass
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());
    let template =
        detect_chat_template(&config.architecture, file.tokenizer_data().model.as_deref());
    let prompt = template.apply("Hello");
    let prompt_tokens = tok.encode(&prompt, false);

    let h = config.hidden_size;
    let ff_size = config.intermediate_size;
    let mut hidden = vec![0.0f32; h];
    cpu_embed_token(prompt_tokens[0], &cpu_weights, &mut hidden, &config);

    let mut scratch = CpuForwardScratch::new(&config);
    rms_norm(
        &hidden,
        &cpu_weights.layer(0).ffn_norm,
        &mut scratch.normed,
        config.rms_norm_eps,
    );

    let d_input = upload_f32(&scratch.normed);
    let d_out_production = GpuBuffer::alloc(ff_size * 4).expect("alloc production out");
    let d_out_vulkan = GpuBuffer::alloc(ff_size * 4).expect("alloc vulkan out");

    // Heuristic selection for N_WAVES
    let n_waves = if ff_size <= 1024 { 4 } else { 8 };

    // Warmup
    gpu::kernels::quant::gemv_gate_up_swiglu_q4_0_f32_on_stream(
        layer.ffn_gate.as_ptr(),
        layer.ffn_up.as_ptr(),
        d_input.as_ptr() as *const f32,
        d_out_production.as_ptr() as *mut f32,
        h,
        ff_size,
        device.stream(),
    )
    .expect("production warmup");

    gpu::kernels::quant::gemv_gate_up_swiglu_vulkan_q4_0_f32(
        layer.ffn_gate.as_ptr(),
        layer.ffn_up.as_ptr(),
        d_input.as_ptr() as *const f32,
        d_out_vulkan.as_ptr() as *mut f32,
        h,
        ff_size,
        n_waves,
        device.stream(),
    )
    .expect("vulkan warmup");

    device.synchronize().expect("warmup sync");

    // Benchmark Production
    let production_ms = average_kernel_ms(&device, 200, || {
        gpu::kernels::quant::gemv_gate_up_swiglu_q4_0_f32_on_stream(
            layer.ffn_gate.as_ptr(),
            layer.ffn_up.as_ptr(),
            d_input.as_ptr() as *const f32,
            d_out_production.as_ptr() as *mut f32,
            h,
            ff_size,
            device.stream(),
        )
        .expect("production run");
    });

    // Benchmark Vulkan-Style
    let vulkan_ms = average_kernel_ms(&device, 200, || {
        gpu::kernels::quant::gemv_gate_up_swiglu_vulkan_q4_0_f32(
            layer.ffn_gate.as_ptr(),
            layer.ffn_up.as_ptr(),
            d_input.as_ptr() as *const f32,
            d_out_vulkan.as_ptr() as *mut f32,
            h,
            ff_size,
            n_waves,
            device.stream(),
        )
        .expect("vulkan run");
    });

    let production_out = download_f32(&d_out_production, ff_size);
    let vulkan_out = download_f32(&d_out_vulkan, ff_size);
    let cross_err = max_abs_error(&production_out, &vulkan_out);

    eprintln!(
        "vulkan_style_q4_0_gate_up production_ms={:.4} vulkan_ms={:.4} speedup={:.3} cross_err={:.6}",
        production_ms,
        vulkan_ms,
        production_ms / vulkan_ms,
        cross_err
    );

    assert!(
        cross_err <= 1e-5,
        "Vulkan-style output diverged from production: cross_err={}",
        cross_err
    );
}
