#![cfg(feature = "gpu")]
//! Diagnostic test to examine layer 2 weight metadata vs other layers

use rocmforge::config::ModelConfig;
use rocmforge::cpu::weights::CpuModelWeights;
use rocmforge::gpu::GpuModelWeights;
use rocmforge::loader::GgufFile;

const MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

#[test]
fn test_layer2_weight_metadata_diagnostic() {
    if skip_if_model_missing() {
        return;
    }

    // Load model
    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse config");
    let _cpu_weights = CpuModelWeights::load(&file, &config).expect("Failed to load CPU weights");
    let gpu_weights = GpuModelWeights::load(&file, &config).expect("Failed to load GPU weights");

    eprintln!("=== Weight Metadata Diagnostic ===");
    eprintln!("Hidden size: {}", config.hidden_size);
    eprintln!("Intermediate size: {}", config.intermediate_size);
    eprintln!("Number of layers: {}", config.num_layers);

    // Compare metadata across first 5 layers
    for layer_idx in 0..5 {
        let layer = gpu_weights.layer(layer_idx);
        eprintln!("\n--- Layer {} ---", layer_idx);

        // Attention weights
        eprintln!("Attention QKV meta:");
        eprintln!(
            "  Q: type={:?}, dims={:?}",
            layer.attn_q_meta.wtype, layer.attn_q_meta.dims
        );
        eprintln!(
            "  K: type={:?}, dims={:?}",
            layer.attn_k_meta.wtype, layer.attn_k_meta.dims
        );
        eprintln!(
            "  V: type={:?}, dims={:?}",
            layer.attn_v_meta.wtype, layer.attn_v_meta.dims
        );
        eprintln!(
            "  O: type={:?}, dims={:?}",
            layer.attn_o_meta.wtype, layer.attn_o_meta.dims
        );

        // FFN weights
        eprintln!("FFN meta:");
        eprintln!(
            "  Gate: type={:?}, dims={:?}",
            layer
                .ffn_gate_meta
                .as_ref()
                .expect("ffn_gate must be present")
                .wtype,
            layer
                .ffn_gate_meta
                .as_ref()
                .expect("ffn_gate must be present")
                .dims
        );
        eprintln!(
            "  Up: type={:?}, dims={:?}",
            layer.ffn_up_meta.wtype, layer.ffn_up_meta.dims
        );
        eprintln!(
            "  Down: type={:?}, dims={:?}",
            layer.ffn_down_meta.wtype, layer.ffn_down_meta.dims
        );

        // Check for interleaved weights
        eprintln!("Interleaved weights:");
        if let Some(ref gate_up_interleaved) = layer.ffn_gate_up_interleaved {
            eprintln!(
                "  Gate+Up (Q4_0): present, size={} bytes",
                gate_up_interleaved.size()
            );
        } else {
            eprintln!("  Gate+Up (Q4_0): NOT PRESENT");
        }

        if let Some(ref gate_up_tile4) = layer.ffn_gate_up_interleaved_tile4 {
            eprintln!(
                "  Gate+Up (Tile4): present, size={} bytes",
                gate_up_tile4.size()
            );
        } else {
            eprintln!("  Gate+Up (Tile4): NOT PRESENT");
        }
    }

    // Test passes if we can examine metadata without crashing
}
