#![cfg(feature = "gpu")]
//! Test Q4_0 FFN down projection specifically to isolate the bug

use rocmforge::config::ModelConfig;
use rocmforge::cpu::ops::dispatch_gemv;
use rocmforge::cpu::weights::CpuModelWeights;
use rocmforge::gpu::{self, GpuDevice};
use rocmforge::loader::GgufFile;
use serial_test::serial;

const MODEL_PATH: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";

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

fn mean_abs_error(a: &[f32], b: &[f32]) -> f32 {
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f32
}

#[test]
#[serial]
fn test_q4_0_ffn_down_layer2_vs_layer0() {
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

    eprintln!("=== Testing Q4_0 FFN Down Projection ===");
    eprintln!("Hidden size: {}, FFN size: {}", h, ff_size);

    // Test layers with different FFN down quantization types
    for layer_idx in [0, 2] {
        let gpu_layer = gpu_weights.layer(layer_idx);

        eprintln!("\n--- Layer {} ---", layer_idx);
        eprintln!("FFN Down type: {:?}", gpu_layer.ffn_down_meta.wtype);
        eprintln!("FFN Down dims: {:?}", gpu_layer.ffn_down_meta.dims);

        // Create test input
        let mut cpu_input = vec![1.0f32; ff_size];
        let mut cpu_output = vec![0.0f32; h];

        // Fill input with test values
        for (i, val) in cpu_input.iter_mut().enumerate() {
            *val = (i as f32) * 0.1 - 50.0; // Range: -50.0 to +436.4
        }

        // CPU reference
        let cpu_layer = cpu_weights.layer(layer_idx);
        dispatch_gemv(
            &cpu_layer.ffn_down,
            &cpu_layer.ffn_down_meta,
            &cpu_input,
            &mut cpu_output,
            h,
            ff_size,
            None,
        )
        .expect("CPU gemv should succeed");

        // GPU computation
        let gpu_input = upload_gpu_f32(&cpu_input);
        let gpu_output = gpu::GpuBuffer::alloc(h * std::mem::size_of::<f32>())
            .expect("Failed to alloc GPU output");

        gpu::ops::gpu_dispatch_gemv_on_stream(
            &device,
            &gpu_layer.ffn_down,
            &gpu_layer.ffn_down_meta,
            gpu_input.as_ptr() as *const f32,
            gpu_output.as_ptr() as *mut f32,
            h,
            ff_size,
            device.stream(),
        )
        .expect("GPU gemv should succeed");

        // Download and compare
        let gpu_result = download_gpu_f32(&gpu_output, h);
        let max_error = max_abs_error(&cpu_output, &gpu_result);
        let mean_error = mean_abs_error(&cpu_output, &gpu_result);
        let output_scale = cpu_output
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        let rel_error = max_error / output_scale.max(1.0);

        eprintln!(
            "Layer {} FFN Down max error: {:.6}, mean error: {:.6}, rel error: {:.6}",
            layer_idx, max_error, mean_error, rel_error
        );
        eprintln!("  CPU output[0..3]: {:?}", &cpu_output[0..3]);
        eprintln!("  GPU output[0..3]: {:?}", &gpu_result[0..3]);

        // Both checked layers currently use Q4_0 FFN-down weights. Use a combined
        // absolute and relative bound so the test tracks real numerical drift
        // rather than failing on large-magnitude but low-relative-error outputs.
        assert!(
            max_error < 3.0 && rel_error < 0.005,
            "Layer {} FFN Down max_error={} rel_error={} exceeded tolerance",
            layer_idx,
            max_error,
            rel_error
        );
    }

    eprintln!("\nQ4_0 FFN Down test completed");
}
