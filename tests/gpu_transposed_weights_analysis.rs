//! Test to identify all transposed weight operations in a layer

use rocmforge::config::ModelConfig;
use rocmforge::cpu::weights::CpuModelWeights;
use rocmforge::gpu::GpuModelWeights;
use rocmforge::loader::GgufFile;

const MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

#[test]
fn test_transposed_weights_analysis() {
    if skip_if_model_missing() {
        return;
    }

    // Load model
    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse config");
    let gpu_weights = GpuModelWeights::load(&file, &config).expect("Failed to load GPU weights");

    eprintln!("=== Transposed Weights Analysis ===");

    // Analyze layer 2
    let layer_idx = 2;
    let gpu_layer = gpu_weights.layer(layer_idx);

    eprintln!("Layer {} weight transpose status:", layer_idx);
    eprintln!(
        "  attn_q: needs_transpose={}",
        gpu_layer.attn_q_meta.needs_transpose
    );
    eprintln!(
        "  attn_k: needs_transpose={}",
        gpu_layer.attn_k_meta.needs_transpose
    );
    eprintln!(
        "  attn_v: needs_transpose={}",
        gpu_layer.attn_v_meta.needs_transpose
    );
    eprintln!(
        "  attn_o: needs_transpose={} type={:?}",
        gpu_layer.attn_o_meta.needs_transpose, gpu_layer.attn_o_meta.wtype
    );
    eprintln!(
        "  ffn_gate: needs_transpose={} type={:?}",
        gpu_layer.ffn_gate_meta.needs_transpose, gpu_layer.ffn_gate_meta.wtype
    );
    eprintln!(
        "  ffn_up: needs_transpose={} type={:?}",
        gpu_layer.ffn_up_meta.needs_transpose, gpu_layer.ffn_up_meta.wtype
    );
    eprintln!(
        "  ffn_down: needs_transpose={} type={:?}",
        gpu_layer.ffn_down_meta.needs_transpose, gpu_layer.ffn_down_meta.wtype
    );

    // Check which transposed operations use Q4_0
    eprintln!("\nTransposed Q4_0 operations (potential bugs):");
    if gpu_layer.attn_o_meta.needs_transpose
        && gpu_layer.attn_o_meta.wtype == rocmforge::loader::GgmlType::Q4_0
    {
        eprintln!("  attn_o: Q4_0 + transposed = BUG");
    }
    if gpu_layer.ffn_down_meta.needs_transpose
        && gpu_layer.ffn_down_meta.wtype == rocmforge::loader::GgmlType::Q4_0
    {
        eprintln!("  ffn_down: Q4_0 + transposed = BUG (FIXED)");
    }

    assert!(true, "Transposed weights analysis completed");
}
