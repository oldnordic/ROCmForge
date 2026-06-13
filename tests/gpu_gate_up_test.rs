#![cfg(feature = "gpu")]
//! Test to check if gate_up operation is working correctly

use rocmforge::config::ModelConfig;
use rocmforge::cpu::ops::dispatch_gemv;
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
    let size_in_bytes = std::mem::size_of_val(data);
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
fn test_gate_up_operation_correctness() {
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
    let layer_idx = 2; // Layer with Q4_0 operations

    eprintln!("=== Gate_Up Operation Correctness ===");

    // Create test input
    let mut cpu_input = vec![1.0f32; h];
    for (i, val) in cpu_input.iter_mut().enumerate() {
        *val = (i as f32) * 0.1 - 5.0;
    }

    // CPU reference - manual gate_up + swiglu
    let mut cpu_gate = vec![0.0f32; ff_size];
    let mut cpu_up = vec![0.0f32; ff_size];

    let cpu_layer = cpu_weights.layer(layer_idx);

    dispatch_gemv(
        cpu_layer.ffn_gate.as_ref().expect("cpu ffn_gate"),
        cpu_layer.ffn_gate_meta.as_ref().expect("cpu ffn_gate_meta"),
        &cpu_input,
        &mut cpu_gate,
        ff_size,
        h,
        None,
    )
    .expect("CPU gate gemv should succeed");

    dispatch_gemv(
        &cpu_layer.ffn_up,
        &cpu_layer.ffn_up_meta,
        &cpu_input,
        &mut cpu_up,
        ff_size,
        h,
        None,
    )
    .expect("CPU up gemv should succeed");

    // Apply SwiGLU: silu(gate) * up
    let mut cpu_swiglu = vec![0.0f32; ff_size];
    for i in 0..ff_size {
        let silu_val = cpu_gate[i] / (1.0 + (-cpu_gate[i]).exp());
        cpu_swiglu[i] = silu_val * cpu_up[i];
    }

    eprintln!("CPU SwiGLU[0..3]: {:?}", &cpu_swiglu[0..3]);

    // For GPU, we'll just check if gate_up produces reasonable output
    // by testing the individual gate and up operations
    let gpu_input = upload_gpu_f32(&cpu_input);
    let gpu_gate = rocmforge::gpu::GpuBuffer::alloc(ff_size * std::mem::size_of::<f32>())
        .expect("Failed to alloc GPU gate");
    let gpu_up = rocmforge::gpu::GpuBuffer::alloc(ff_size * std::mem::size_of::<f32>())
        .expect("Failed to alloc GPU up");

    rocmforge::gpu::ops::gpu_dispatch_gemv_on_stream(
        &device,
        gpu_weights.layer(layer_idx).ffn_gate.as_ref().expect("ffn_gate"),
        gpu_weights.layer(layer_idx).ffn_gate_meta.as_ref().expect("ffn_gate_meta"),
        gpu_input.as_ptr() as *const f32,
        gpu_gate.as_ptr() as *mut f32,
        ff_size,
        h,
        device.stream(),
    )
    .expect("GPU gate gemv should succeed");

    rocmforge::gpu::ops::gpu_dispatch_gemv_on_stream(
        &device,
        &gpu_weights.layer(layer_idx).ffn_up,
        &gpu_weights.layer(layer_idx).ffn_up_meta,
        gpu_input.as_ptr() as *const f32,
        gpu_up.as_ptr() as *mut f32,
        ff_size,
        h,
        device.stream(),
    )
    .expect("GPU up gemv should succeed");

    let gpu_gate_result = download_gpu_f32(&gpu_gate, ff_size);
    let gpu_up_result = download_gpu_f32(&gpu_up, ff_size);

    eprintln!("GPU Gate[0..3]: {:?}", &gpu_gate_result[0..3]);
    eprintln!("GPU Up[0..3]: {:?}", &gpu_up_result[0..3]);

    // Compare gate and up individually
    let gate_error = max_abs_error(&cpu_gate, &gpu_gate_result);
    let up_error = max_abs_error(&cpu_up, &gpu_up_result);

    eprintln!("Gate error: {:.6}", gate_error);
    eprintln!("Up error: {:.6}", up_error);

    // Check if gate_up operations are correct
    assert!(gate_error < 1.0, "GPU gate operation should match CPU");
    assert!(up_error < 1.0, "GPU up operation should match CPU");

    eprintln!("Gate_up operations are correct!");
}
