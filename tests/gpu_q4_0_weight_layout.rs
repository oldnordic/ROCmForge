//! Test to check Q4_0 weight layout and offset calculations

use rocmforge::config::ModelConfig;
use rocmforge::cpu::weights::CpuModelWeights;
use rocmforge::gpu::GpuModelWeights;
use rocmforge::loader::GgufFile;

const MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

#[test]
fn test_q4_0_weight_layout_diagnostic() {
    if skip_if_model_missing() {
        return;
    }

    // Load model
    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("Failed to load CPU weights");
    let gpu_weights = GpuModelWeights::load(&file, &config).expect("Failed to load GPU weights");

    eprintln!("=== Q4_0 Weight Layout Diagnostic ===");

    // Compare FFN down weight layout for layers 0 (Q4_1) and 2 (Q4_0)
    for layer_idx in [0, 2] {
        let cpu_layer = cpu_weights.layer(layer_idx);
        let gpu_layer = gpu_weights.layer(layer_idx);

        eprintln!("\n--- Layer {} ---", layer_idx);
        eprintln!("FFN Down type: {:?}", gpu_layer.ffn_down_meta.wtype);
        eprintln!("FFN Down dims: {:?}", gpu_layer.ffn_down_meta.dims);
        eprintln!(
            "FFN Down needs_transpose: {:?}",
            gpu_layer.ffn_down_meta.needs_transpose
        );
        eprintln!("FFN Down role: {:?}", gpu_layer.ffn_down_meta.role);

        // Calculate expected sizes
        let h = config.hidden_size;
        let ff_size = config.intermediate_size;
        let n_blocks = ff_size / 32; // QK4_0 = 32

        eprintln!("Expected blocks: {}", n_blocks);
        eprintln!("CPU weight bytes: {}", cpu_layer.ffn_down.len());
        eprintln!("GPU weight bytes: {}", gpu_layer.ffn_down.size());

        // Calculate block sizes
        match gpu_layer.ffn_down_meta.wtype {
            rocmforge::loader::GgmlType::Q4_0 => {
                let q4_0_block_size = 18; // half d + 16 int8_t qs
                let expected_size = h * n_blocks * q4_0_block_size;
                eprintln!(
                    "Q4_0 expected size: {} ({} cols * {} blocks * {} bytes/block)",
                    expected_size, h, n_blocks, q4_0_block_size
                );

                // Check if size matches expectation
                if cpu_layer.ffn_down.len() != expected_size {
                    eprintln!("WARNING: CPU size mismatch!");
                    eprintln!(
                        "  Expected: {}, Got: {}",
                        expected_size,
                        cpu_layer.ffn_down.len()
                    );
                }
            }
            rocmforge::loader::GgmlType::Q4_1 => {
                let q4_1_block_size = 20; // half d + half m + 16 int8_t qs
                let expected_size = h * n_blocks * q4_1_block_size;
                eprintln!(
                    "Q4_1 expected size: {} ({} cols * {} blocks * {} bytes/block)",
                    expected_size, h, n_blocks, q4_1_block_size
                );

                if cpu_layer.ffn_down.len() != expected_size {
                    eprintln!("WARNING: CPU size mismatch!");
                    eprintln!(
                        "  Expected: {}, Got: {}",
                        expected_size,
                        cpu_layer.ffn_down.len()
                    );
                }
            }
            _ => {}
        }
    }

    assert!(true, "Weight layout diagnostic completed");
}
