//! Test to verify Q4_0 kernel behavior with transposed weights

use rocmforge::config::ModelConfig;
use rocmforge::cpu::ops::{dispatch_gemv, gemv_q4_0_transposed};
use rocmforge::cpu::weights::CpuModelWeights;
use rocmforge::gpu::{GpuDevice, GpuModelWeights};
use rocmforge::loader::GgufFile;
use serial_test::serial;

const MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

fn download_gpu_f32(buf: &rocmforge::gpu::GpuBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    buf.copy_to_host(&mut bytes)
        .expect("GPU buffer should download");
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() }
}

fn upload_gpu_f32(data: &[f32]) -> rocmforge::gpu::GpuBuffer {
    let size_in_bytes = data.len() * std::mem::size_of::<f32>();
    let mut buf =
        rocmforge::gpu::GpuBuffer::alloc(size_in_bytes).expect("Failed to allocate GPU buffer");
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
fn test_q4_0_transposed_kernel_behavior() {
    if skip_if_model_missing() {
        return;
    }

    // Load model
    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("Failed to load CPU weights");
    let gpu_weights = GpuModelWeights::load(&file, &config).expect("Failed to load GPU weights");

    // Initialize GPU
    let caps = rocmforge::gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let h = config.hidden_size;
    let ff_size = config.intermediate_size;

    eprintln!("=== Q4_0 Transposed Kernel Verification ===");

    // Test layer 2 (Q4_0 FFN down with needs_transpose=true)
    let layer_idx = 2;
    let cpu_layer = cpu_weights.layer(layer_idx);
    let gpu_layer = gpu_weights.layer(layer_idx);

    eprintln!("Layer {} FFN Down:", layer_idx);
    eprintln!("  Type: {:?}", gpu_layer.ffn_down_meta.wtype);
    eprintln!("  Dims: {:?}", gpu_layer.ffn_down_meta.dims);
    eprintln!(
        "  Needs transpose: {}",
        gpu_layer.ffn_down_meta.needs_transpose
    );

    // Create test input
    let mut cpu_input = vec![1.0f32; ff_size];
    for (i, val) in cpu_input.iter_mut().enumerate() {
        *val = ((i as f32) * 0.1 - 50.0);
    }

    let mut cpu_output_dispatch = vec![0.0f32; h];
    let mut cpu_output_transposed = vec![0.0f32; h];

    // CPU: dispatch_gemv (which should call transposed variant)
    dispatch_gemv(
        &cpu_layer.ffn_down,
        &cpu_layer.ffn_down_meta,
        &cpu_input,
        &mut cpu_output_dispatch,
        h,
        ff_size,
        None,
    )
    .expect("CPU dispatch should succeed");

    // CPU: direct call to transposed function
    gemv_q4_0_transposed(
        &cpu_layer.ffn_down,
        &cpu_input,
        &mut cpu_output_transposed,
        h,
        ff_size,
    );

    // GPU: standard dispatch
    let gpu_input = upload_gpu_f32(&cpu_input);
    let mut gpu_output = rocmforge::gpu::GpuBuffer::alloc(h * std::mem::size_of::<f32>())
        .expect("Failed to alloc GPU output");

    rocmforge::gpu::ops::gpu_dispatch_gemv_on_stream(
        &device,
        &gpu_layer.ffn_down,
        &gpu_layer.ffn_down_meta,
        gpu_input.as_ptr() as *const f32,
        gpu_output.as_ptr() as *mut f32,
        h,
        ff_size,
        device.stream(),
    )
    .expect("GPU dispatch should succeed");

    let gpu_result = download_gpu_f32(&gpu_output, h);

    // Compare results
    let dispatch_vs_transposed = max_abs_error(&cpu_output_dispatch, &cpu_output_transposed);
    let gpu_vs_transposed = max_abs_error(&gpu_result, &cpu_output_transposed);
    let gpu_vs_dispatch = max_abs_error(&gpu_result, &cpu_output_dispatch);

    eprintln!("\nComparison:");
    eprintln!(
        "  CPU dispatch vs CPU transposed: {:.6}",
        dispatch_vs_transposed
    );
    eprintln!("  GPU vs CPU transposed: {:.6}", gpu_vs_transposed);
    eprintln!("  GPU vs CPU dispatch: {:.6}", gpu_vs_dispatch);

    eprintln!("\nFirst few values:");
    eprintln!(
        "  CPU transposed[0..3]:  {:?}",
        &cpu_output_transposed[0..3]
    );
    eprintln!("  CPU dispatch[0..3]:     {:?}", &cpu_output_dispatch[0..3]);
    eprintln!("  GPU[0..3]:               {:?}", &gpu_result[0..3]);

    // CPU dispatch and transposed should match
    assert!(
        dispatch_vs_transposed < 0.001,
        "CPU dispatch should match CPU transposed"
    );

    // GPU should match CPU transposed (the correct implementation)
    assert!(
        gpu_vs_transposed < 1.0,
        "GPU should match CPU transposed implementation"
    );
}
