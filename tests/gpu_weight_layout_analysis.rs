//! Test to understand the exact weight storage layout

use rocmforge::config::ModelConfig;
use rocmforge::cpu::weights::CpuModelWeights;
use rocmforge::loader::GgufFile;

const MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

#[test]
fn test_weight_storage_layout_analysis() {
    if skip_if_model_missing() {
        return;
    }

    // Load model
    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("Failed to load CPU weights");

    eprintln!("=== Weight Storage Layout Analysis ===");

    // Analyze layer 2 FFN down weights
    let layer_idx = 2;
    let cpu_layer = cpu_weights.layer(layer_idx);

    let h = config.hidden_size; // 896
    let ff_size = config.intermediate_size; // 4864

    eprintln!("Layer {} FFN Down:", layer_idx);
    eprintln!("  Stored dims: {:?}", cpu_layer.ffn_down_meta.dims); // [4864, 896]
    eprintln!(
        "  GEMV expects: output[{}] = weights[{}, {}] * input[{}]",
        h, h, ff_size, ff_size
    );
    eprintln!(
        "  Storage: [{}, {}]",
        cpu_layer.ffn_down_meta.dims[0], cpu_layer.ffn_down_meta.dims[1]
    );

    // For Q4_0, each block is 32 values stored in 18 bytes
    let Q4_0_BLOCK_ELEMS: usize = 32;
    let Q4_0_BLOCK_BYTES: usize = 18;

    // Calculate expected layout
    let stored_rows = cpu_layer.ffn_down_meta.dims[0] as usize; // 4864
    let stored_cols = cpu_layer.ffn_down_meta.dims[1] as usize; // 896

    let blocks_per_col = stored_rows / Q4_0_BLOCK_ELEMS; // 4864 / 32 = 152
    let col_bytes = blocks_per_col * Q4_0_BLOCK_BYTES; // 152 * 18 = 2736

    let expected_size = stored_cols * col_bytes; // 896 * 2736 = 2451456

    eprintln!("\nBlock structure:");
    eprintln!("  Blocks per column: {}", blocks_per_col);
    eprintln!("  Bytes per column: {}", col_bytes);
    eprintln!("  Expected total size: {}", expected_size);
    eprintln!("  Actual weight bytes: {}", cpu_layer.ffn_down.len());

    // Verify the layout matches
    assert_eq!(
        cpu_layer.ffn_down.len(),
        expected_size,
        "Weight size should match calculated expectation"
    );

    // Simulate how the CPU transposed function accesses the weights
    eprintln!("\nSimulating CPU transposed access for first few columns:");

    for v in 0..3 {
        let col_offset = v * col_bytes;
        eprintln!("  Column {} starts at offset {}", v, col_offset);

        // Check first block in this column
        let first_block_offset = col_offset;
        eprintln!("    First block at offset {}", first_block_offset);

        // The block should contain scale (2 bytes) and 16 quantized bytes
        if first_block_offset + Q4_0_BLOCK_BYTES <= cpu_layer.ffn_down.len() {
            let scale_bytes = &cpu_layer.ffn_down[first_block_offset..first_block_offset + 2];
            let qs_bytes = &cpu_layer.ffn_down[first_block_offset + 2..first_block_offset + 18];
            eprintln!("    Scale bytes: {:?}", scale_bytes);
            eprintln!("    Quantized bytes (first 4): {:?}", &qs_bytes[..4]);
        }
    }

    // Now let's think about how the GPU kernel interprets this
    eprintln!("\nGPU kernel interpretation:");
    eprintln!("  GPU receives: n_rows={}, ncols_dst={}", ff_size, h);
    eprintln!(
        "  GPU calculates: n_blocks_total = n_rows / 32 = {}",
        ff_size / 32
    );
    eprintln!("  For column col, GPU calculates offset: col * n_blocks_total * 18");
    eprintln!("  For column 0: 0 * {} * 18 = {}", ff_size / 32, 0);
    eprintln!(
        "  For column 1: 1 * {} * 18 = {}",
        ff_size / 32,
        (ff_size / 32) * 18
    );

    let gpu_n_blocks_total = ff_size / 32;
    let gpu_col_0_offset = 0 * gpu_n_blocks_total * Q4_0_BLOCK_BYTES;
    let gpu_col_1_offset = 1 * gpu_n_blocks_total * Q4_0_BLOCK_BYTES;

    eprintln!("\nComparing offsets:");
    eprintln!("  Column 0: CPU={}, GPU={}", 0, gpu_col_0_offset);
    eprintln!("  Column 1: CPU={}, GPU={}", col_bytes, gpu_col_1_offset);

    if gpu_col_1_offset != col_bytes {
        eprintln!("  MISMATCH! GPU uses wrong offset calculation for transposed weights");
        eprintln!("  Expected: {}", col_bytes);
        eprintln!("  Got: {}", gpu_col_1_offset);
    } else {
        eprintln!("  Offsets match!");
    }

    assert!(true, "Weight layout analysis completed");
}
