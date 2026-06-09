#![cfg(feature = "gpu")]
//! Test to compare individual FFN down vs. FFN down in full layer forward

use rocmforge::config::ModelConfig;
use rocmforge::cpu::cache::{CpuForwardScratch, CpuKvCache};
use rocmforge::cpu::forward::cpu_layer_forward;

use rocmforge::cpu::weights::CpuModelWeights;
use rocmforge::gpu::ops::gpu_dispatch_gemv_on_stream;
use rocmforge::gpu::{self, gpu_layer_forward_hybrid, GpuDevice, GpuForwardScratch, GpuKvCache};
use rocmforge::loader::GgufFile;
use serial_test::serial;

const MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

fn download_gpu_f32(buf: &gpu::GpuBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    buf.copy_to_host(&mut bytes)
        .expect("GPU buffer should download");
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() }
}

fn upload_gpu_f32(data: &[f32]) -> gpu::GpuBuffer {
    let size_in_bytes = std::mem::size_of_val(data);
    let mut buf = gpu::GpuBuffer::alloc(size_in_bytes).expect("Failed to allocate GPU buffer");
    buf.copy_from_host(unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, size_in_bytes)
    })
    .expect("Failed to upload to GPU");
    buf
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[serial]
fn test_ffn_down_individual_vs_layer_forward() {
    if skip_if_model_missing() {
        return;
    }

    // Load model
    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("Failed to load CPU weights");
    let gpu_weights =
        gpu::GpuModelWeights::load(&file, &config).expect("Failed to load GPU weights");

    // Initialize GPU
    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let h = config.hidden_size;
    let ff_size = config.intermediate_size;
    let layer_idx = 2; // Layer with Q4_0 FFN down

    eprintln!("=== FFN Down Individual vs. Layer Forward ===");

    // Create test input (simulating SwiGLU output)
    let mut swiglu_input = vec![1.0f32; ff_size];
    for (i, val) in swiglu_input.iter_mut().enumerate() {
        *val = (i as f32) * 0.1 - 50.0;
    }

    // Method 1: Individual FFN down call
    eprintln!("\nMethod 1: Individual FFN down call");
    let gpu_input_1 = upload_gpu_f32(&swiglu_input);
    let gpu_output_1 =
        gpu::GpuBuffer::alloc(h * std::mem::size_of::<f32>()).expect("Failed to alloc GPU output");

    gpu_dispatch_gemv_on_stream(
        &device,
        &gpu_weights.layer(layer_idx).ffn_down,
        &gpu_weights.layer(layer_idx).ffn_down_meta,
        gpu_input_1.as_ptr() as *const f32,
        gpu_output_1.as_ptr() as *mut f32,
        h,
        ff_size,
        device.stream(),
    )
    .expect("GPU FFN down should succeed");

    let gpu_result_1 = download_gpu_f32(&gpu_output_1, h);
    eprintln!("  GPU individual[0..3]: {:?}", &gpu_result_1[0..3]);

    // Method 2: FFN down within full layer forward
    eprintln!("\nMethod 2: FFN down within full layer forward");
    let mut cpu_hidden = vec![1.0f32; h];
    let mut cpu_kv = CpuKvCache::new(&config, 1);
    let mut cpu_scratch = CpuForwardScratch::new(&config);

    let mut gpu_kv = GpuKvCache::new(&config, 1).expect("Failed to create GPU KV cache");
    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("Failed to create GPU scratch");

    // Upload initial hidden state
    gpu_scratch
        .hidden
        .copy_from_host(unsafe {
            std::slice::from_raw_parts(
                cpu_hidden.as_ptr() as *const u8,
                h * std::mem::size_of::<f32>(),
            )
        })
        .expect("Failed to upload hidden state");

    // CPU reference
    cpu_layer_forward(
        &mut cpu_hidden,
        cpu_weights.layer(layer_idx),
        &mut cpu_kv,
        &mut cpu_scratch,
        0,
        0,
        &config,
        false,
    )
    .expect("CPU layer forward should succeed");

    // GPU full layer forward
    gpu_layer_forward_hybrid(
        &device,
        gpu_weights.layer(layer_idx),
        Some(cpu_weights.layer(layer_idx)),
        &mut gpu_kv,
        &mut gpu_scratch,
        Some(&mut cpu_scratch),
        0,
        0,
        &config,
    )
    .expect("GPU layer forward should succeed");

    let gpu_result_2 = download_gpu_f32(&gpu_scratch.hidden, h);
    eprintln!("  GPU layer forward[0..3]: {:?}", &gpu_result_2[0..3]);
    eprintln!("  CPU layer forward[0..3]: {:?}", &cpu_hidden[0..3]);

    // Compare
    let error_1_vs_2 = max_abs_error(&gpu_result_1, &gpu_result_2);
    let error_2_vs_cpu = max_abs_error(&gpu_result_2, &cpu_hidden);

    eprintln!("\nComparison:");
    eprintln!("  Individual vs. Layer forward: {:.6}", error_1_vs_2);
    eprintln!("  Layer forward vs. CPU: {:.6}", error_2_vs_cpu);

    // If there's a large difference, it indicates a bug in the layer forward logic
    if error_1_vs_2 > 1.0 {
        eprintln!("  WARNING: Large difference between individual and layer forward!");
        eprintln!("  This suggests a bug in how layer forward calls FFN down");
    }

    assert!(error_2_vs_cpu < 2.0, "GPU layer forward should match CPU");
}
