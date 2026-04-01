#![cfg(feature = "gpu")]

mod common;

use rocmforge::config::{ModelConfig};
use rocmforge::gpu::{self, GpuBuffer, GpuDevice};
use rocmforge::loader::{GgmlType, GgufFile};
use serial_test::serial;
use std::time::Instant;

const MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/llama3-8b-q4_k_m.gguf";

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

fn upload_u8(data: &[u8]) -> GpuBuffer {
    let mut buffer = GpuBuffer::alloc(data.len()).expect("alloc GPU buffer");
    buffer.copy_from_host(data).expect("upload GPU buffer");
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
fn test_gpu_vulkan_style_q4_k_gemv() {
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
    
    // Find a Q4_K tensor
    let tensor_name = "blk.0.attn_q.weight";
    // file.tensor returns Result<Option<TensorView>, LoadError>
    let tensor_res = file.tensor(tensor_name).expect("Failed to query tensor");
    let tensor_view = tensor_res.expect("Tensor not found in file");
    
    // Note: TensorView in rocmforge uses 'ggml_type' field name, not 'wtype()' method
    assert_eq!(tensor_view.ggml_type, GgmlType::Q4_K);

    let h = config.hidden_size;
    let n_heads = config.num_heads;
    let head_dim = config.head_dim;
    let out_dim = n_heads * head_dim;

    eprintln!("Testing Q4_K on {} ({}x{})", tensor_name, out_dim, h);

    let d_weights = upload_u8(tensor_view.data);
    let mut input = vec![0.0f32; h];
    for i in 0..h { input[i] = (i as f32).sin(); }
    
    let d_input = upload_f32(&input);
    let d_out_production = GpuBuffer::alloc(out_dim * 4).expect("alloc production out");
    let d_out_vulkan = GpuBuffer::alloc(out_dim * 4).expect("alloc vulkan out");

    // Warmup
    gpu::kernels::quant::gemv_q4_k_f32_on_stream(
        d_weights.as_ptr(),
        d_input.as_ptr() as *const f32,
        d_out_production.as_ptr() as *mut f32,
        h,
        out_dim,
        device.stream(),
    ).expect("production warmup");

    gpu::kernels::quant::gemv_q4_k_f32_vulkan_style(
        &device,
        d_weights.as_ptr(),
        d_input.as_ptr() as *const f32,
        d_out_vulkan.as_ptr() as *mut f32,
        h,
        out_dim,
        8, // n_waves
        device.stream(),
    ).expect("vulkan warmup");

    device.synchronize().expect("warmup sync");

    // Benchmark Production
    let production_ms = average_kernel_ms(&device, 200, || {
        gpu::kernels::quant::gemv_q4_k_f32_on_stream(
            d_weights.as_ptr(),
            d_input.as_ptr() as *const f32,
            d_out_production.as_ptr() as *mut f32,
            h,
            out_dim,
            device.stream(),
        ).expect("production run");
    });

    // Benchmark Vulkan-Style
    let vulkan_ms = average_kernel_ms(&device, 200, || {
        gpu::kernels::quant::gemv_q4_k_f32_vulkan_style(
            &device,
            d_weights.as_ptr(),
            d_input.as_ptr() as *const f32,
            d_out_vulkan.as_ptr() as *mut f32,
            h,
            out_dim,
            8,
            device.stream(),
        ).expect("vulkan run");
    });

    let production_out = download_f32(&d_out_production, out_dim);
    let vulkan_out = download_f32(&d_out_vulkan, out_dim);
    let cross_err = max_abs_error(&production_out, &vulkan_out);

    eprintln!(
        "vulkan_style_q4_k_gemv production_ms={:.4} vulkan_ms={:.4} speedup={:.3} cross_err={:.6}",
        production_ms,
        vulkan_ms,
        production_ms / vulkan_ms,
        cross_err
    );
}
