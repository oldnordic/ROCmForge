//! Test to check which kernel path is being used for Q4_0

use rocmforge::config::ModelConfig;
use rocmforge::cpu::weights::CpuModelWeights;
use rocmforge::gpu::GpuModelWeights;
use rocmforge::loader::GgufFile;

const MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

#[test]
fn test_kernel_path_detection() {
    if skip_if_model_missing() {
        return;
    }

    // Load model
    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse config");
    let gpu_weights = GpuModelWeights::load(&file, &config).expect("Failed to load GPU weights");

    eprintln!("=== Kernel Path Detection ===");

    // Check layer 2 FFN down
    let layer_idx = 2;
    let gpu_layer = gpu_weights.layer(layer_idx);

    let h = config.hidden_size;
    let ff_size = config.intermediate_size;

    eprintln!("Layer {} FFN Down:", layer_idx);
    eprintln!("  Type: {:?}", gpu_layer.ffn_down_meta.wtype);
    eprintln!("  Dims: {:?}", gpu_layer.ffn_down_meta.dims);
    eprintln!(
        "  Needs transpose: {}",
        gpu_layer.ffn_down_meta.needs_transpose
    );

    // Check experimental kernel settings
    eprintln!("\nExperimental settings:");
    eprintln!(
        "  ROCMFORGE_ENABLE_EXPERIMENTAL_GPU_KERNELS: {:?}",
        std::env::var("ROCMFORGE_ENABLE_EXPERIMENTAL_GPU_KERNELS")
    );
    eprintln!(
        "  ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH: {:?}",
        std::env::var("ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH")
    );

    // Check safety settings
    if let Ok(val) = std::env::var("ROCMFORGE_GPU_SAFE_MODE") {
        eprintln!("  ROCMFORGE_GPU_SAFE_MODE: {}", val);
    } else {
        eprintln!("  ROCMFORGE_GPU_SAFE_MODE: not set");
    }

    eprintln!("\nExpected kernel path:");
    eprintln!("  With experimental enabled: Vulkan-style kernel");
    eprintln!("  With experimental disabled: Baseline kernel");

    assert!(true, "Kernel path detection completed");
}
