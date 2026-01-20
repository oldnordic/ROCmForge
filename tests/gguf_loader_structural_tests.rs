use rocmforge::loader::gguf::{GgufLoader, GgufMetadata, GgufTensor};
use rocmforge::model::config::ModelConfig;
use std::path::Path;

#[test]
fn test_gguf_header_and_metadata_parsing() {
    let gguf_path = "tests/data/tiny_model.gguf";

    // Skip test if file doesn't exist
    if !Path::new(gguf_path).exists() {
        println!("Skipping test - GGUF file not found at {}", gguf_path);
        return;
    }

    let result = GgufLoader::new(gguf_path);
    assert!(
        result.is_ok(),
        "Failed to load GGUF file: {:?}",
        result.err()
    );

    let loader = result.unwrap();
    let metadata = loader.metadata();

    // Validate essential metadata fields
    assert!(
        !metadata.architecture.is_empty(),
        "Architecture should not be empty"
    );
    assert!(
        metadata.num_layers > 0,
        "Number of layers should be greater than 0"
    );
    assert!(
        metadata.num_heads > 0,
        "Number of heads should be greater than 0"
    );
    assert!(
        metadata.hidden_size > 0,
        "Hidden size should be greater than 0"
    );
    assert!(
        metadata.vocab_size > 0,
        "Vocab size should be greater than 0"
    );
}

#[test]
fn test_gguf_tensor_enumeration() {
    let gguf_path = "tests/data/tiny_model.gguf";

    if !Path::new(gguf_path).exists() {
        println!("Skipping test - GGUF file not found at {}", gguf_path);
        return;
    }

    let result = GgufLoader::new(gguf_path);
    assert!(result.is_ok(), "Failed to load GGUF file");

    let loader = result.unwrap();
    let tensors = loader.load_tensors().unwrap();
    assert!(!tensors.is_empty(), "Should have at least one tensor");

    // Validate each tensor
    for (_name, tensor) in &tensors {
        assert!(!tensor.name.is_empty(), "Tensor name should not be empty");
        assert!(
            !tensor.shape.dims().is_empty(),
            "Tensor should have at least 1 dimension"
        );

        // Check that dtype is known
        match tensor.tensor_type {
            rocmforge::loader::GgufTensorType::F32
            | rocmforge::loader::GgufTensorType::F16
            | rocmforge::loader::GgufTensorType::Q4_0
            | rocmforge::loader::GgufTensorType::Q8_0
            | rocmforge::loader::GgufTensorType::Q2_K
            | rocmforge::loader::GgufTensorType::Q3_K
            | rocmforge::loader::GgufTensorType::Q4_K
            | rocmforge::loader::GgufTensorType::Q5_K
            | rocmforge::loader::GgufTensorType::Q6_K
            | rocmforge::loader::GgufTensorType::Mxfp4
            | rocmforge::loader::GgufTensorType::Mxfp6E2m3
            | rocmforge::loader::GgufTensorType::Mxfp6E3m2 => {
                // Known dtype - OK
            }
        }
    }
}

#[test]
fn test_gguf_core_tensor_shapes() {
    let gguf_path = "tests/data/tiny_model.gguf";

    if !Path::new(gguf_path).exists() {
        println!("Skipping test - GGUF file not found at {}", gguf_path);
        return;
    }

    let result = GgufLoader::new(gguf_path);
    assert!(result.is_ok(), "Failed to load GGUF file");

    let loader = result.unwrap();
    let tensors = loader.load_tensors().unwrap();
    let metadata = loader.metadata();

    // Extract model config from metadata
    let model_config = extract_model_config_from_metadata(metadata);

    // Helper function to find tensor by name pattern
    let find_tensor = |patterns: &[&str]| -> Option<&GgufTensor> {
        tensors
            .iter()
            .find(|(_, t)| patterns.iter().any(|&pattern| t.name.contains(pattern)))
            .map(|(_, t)| t)
    };

    // Test token embedding tensor
    let token_emb_patterns = ["token_emb", "embed_tokens", "word_embeddings"];
    let token_emb = find_tensor(&token_emb_patterns).unwrap_or_else(|| {
        panic!(
            "Token embedding tensor not found. Looked for patterns: {:?}",
            token_emb_patterns
        )
    });

    // Token embedding should be [vocab_size, d_model] or [d_model, vocab_size]
    assert_eq!(
        token_emb.shape.dims().len(),
        2,
        "Token embedding should be 2D"
    );
    let token_emb_dims = token_emb.shape.dims();
    assert!(
        (token_emb_dims[0] == model_config.vocab_size && token_emb_dims[1] == model_config.hidden_size) ||
        (token_emb_dims[0] == model_config.hidden_size && token_emb_dims[1] == model_config.vocab_size),
        "Token embedding dimensions should be [vocab_size, hidden_size] or [hidden_size, vocab_size], got {:?}",
        token_emb_dims
    );

    // Test LM head tensor
    let lm_head_patterns = ["lm_head", "output", "logits"];
    let lm_head = find_tensor(&lm_head_patterns).unwrap_or_else(|| {
        panic!(
            "LM head tensor not found. Looked for patterns: {:?}",
            lm_head_patterns
        )
    });

    // LMhead should be [vocab_size, d_model] or [d_model, vocab_size]
    assert_eq!(lm_head.shape.dims().len(), 2, "LM head should be 2D");
    let lm_head_dims = lm_head.shape.dims();
    assert!(
        (lm_head_dims[0] == model_config.vocab_size && lm_head_dims[1] == model_config.hidden_size) ||
        (lm_head_dims[0] == model_config.hidden_size && lm_head_dims[1] == model_config.vocab_size),
        "LM head dimensions should be [vocab_size, hidden_size] or [hidden_size, vocab_size], got {:?}",
        lm_head_dims
    );

    // Test first layer attention projections
    let layer_0_patterns = ["layers.0", "layer.0", "transformer.h.0"];
    let find_layer_tensor = |suffix: &str| -> Option<&GgufTensor> {
        tensors
            .iter()
            .find(|(_, t)| {
                layer_0_patterns
                    .iter()
                    .any(|&prefix| t.name.starts_with(prefix))
                    && t.name.contains(suffix)
            })
            .map(|(_, t)| t)
    };

    // Q projection
    let q_proj = find_layer_tensor("q_proj")
        .or_else(|| find_layer_tensor("query"))
        .or_else(|| find_layer_tensor("self_attn.q_proj"))
        .unwrap_or_else(|| panic!("Q projection tensor not found in layer 0"));

    // Q projection should be [d_model, num_heads * head_dim] or reverse
    assert_eq!(q_proj.shape.dims().len(), 2, "Q projection should be 2D");
    let q_dims = q_proj.shape.dims();
    let expected_q_size = model_config.num_attention_heads
        * (model_config.hidden_size / model_config.num_attention_heads);
    assert!(
        (q_dims[0] == model_config.hidden_size && q_dims[1] == expected_q_size)
            || (q_dims[0] == expected_q_size && q_dims[1] == model_config.hidden_size),
        "Q projection dimensions should be [hidden_size, num_heads*head_dim] or reversed, got {:?}",
        q_dims
    );

    // K projection
    let k_proj = find_layer_tensor("k_proj")
        .or_else(|| find_layer_tensor("key"))
        .or_else(|| find_layer_tensor("self_attn.k_proj"))
        .unwrap_or_else(|| panic!("K projection tensor not found in layer 0"));

    // K projection should be [d_model, num_heads * head_dim] (assuming MHA for simplicity)
    assert_eq!(k_proj.shape.dims().len(), 2, "K projection should be 2D");
    let k_dims = k_proj.shape.dims();
    let expected_k_size = model_config.num_attention_heads
        * (model_config.hidden_size / model_config.num_attention_heads);
    assert!(
        (k_dims[0] == model_config.hidden_size && k_dims[1] == expected_k_size)
            || (k_dims[0] == expected_k_size && k_dims[1] == model_config.hidden_size),
        "K projection dimensions should be [hidden_size, num_heads*head_dim] or reversed, got {:?}",
        k_dims
    );

    // V projection
    let v_proj = find_layer_tensor("v_proj")
        .or_else(|| find_layer_tensor("value"))
        .or_else(|| find_layer_tensor("self_attn.v_proj"))
        .unwrap_or_else(|| panic!("V projection tensor not found in layer 0"));

    // V projection should be same as K projection
    assert_eq!(v_proj.shape.dims().len(), 2, "V projection should be 2D");
    let v_dims = v_proj.shape.dims();
    assert_eq!(
        v_dims, k_dims,
        "V projection should have same dimensions as K projection"
    );

    // O projection
    let o_proj = find_layer_tensor("o_proj")
        .or_else(|| find_layer_tensor("dense"))
        .or_else(|| find_layer_tensor("self_attn.o_proj"))
        .unwrap_or_else(|| panic!("O projection tensor not found in layer 0"));

    // O projection should be [num_heads * head_dim, d_model] or reverse
    assert_eq!(o_proj.shape.dims().len(), 2, "O projection should be 2D");
    let o_dims = o_proj.shape.dims();
    assert!(
        (o_dims[0] == expected_q_size && o_dims[1] == model_config.hidden_size)
            || (o_dims[0] == model_config.hidden_size && o_dims[1] == expected_q_size),
        "O projection dimensions should be [num_heads*head_dim, hidden_size] or reversed, got {:?}",
        o_dims
    );

    // Test MLP projections
    let mlp_up = find_layer_tensor("mlp.up_proj")
        .or_else(|| find_layer_tensor("mlp.c_fc"))
        .or_else(|| find_layer_tensor("feed_forward.w1"))
        .unwrap_or_else(|| panic!("MLP up projection tensor not found in layer 0"));

    // MLP up should be [d_model, intermediate_size] or reverse
    assert_eq!(
        mlp_up.shape.dims().len(),
        2,
        "MLP up projection should be 2D"
    );
    let mlp_up_dims = mlp_up.shape.dims();
    assert!(
        (mlp_up_dims[0] == model_config.hidden_size && mlp_up_dims[1] == model_config.intermediate_size) ||
        (mlp_up_dims[0] == model_config.intermediate_size && mlp_up_dims[1] == model_config.hidden_size),
        "MLP up projection dimensions should be [hidden_size, intermediate_size] or reversed, got {:?}",
        mlp_up_dims
    );

    let mlp_gate = find_layer_tensor("mlp.gate_proj")
        .or_else(|| find_layer_tensor("mlp.gate"))
        .or_else(|| find_layer_tensor("feed_forward.gate"))
        .unwrap_or_else(|| panic!("MLP gate projection tensor not found in layer 0"));

    // MLP gate should be same as MLP up
    assert_eq!(
        mlp_gate.shape.dims().len(),
        2,
        "MLP gate projection should be 2D"
    );
    let mlp_gate_dims = mlp_gate.shape.dims();
    assert_eq!(
        mlp_gate_dims, mlp_up_dims,
        "MLP gate should have same dimensions as MLP up"
    );

    let mlp_down = find_layer_tensor("mlp.down_proj")
        .or_else(|| find_layer_tensor("mlp.c_proj"))
        .or_else(|| find_layer_tensor("feed_forward.w2"))
        .unwrap_or_else(|| panic!("MLP down projection tensor not found in layer 0"));

    // MLP down should be [intermediate_size, d_model] or reverse
    assert_eq!(
        mlp_down.shape.dims().len(),
        2,
        "MLP down projection should be 2D"
    );
    let mlp_down_dims = mlp_down.shape.dims();
    assert!(
        (mlp_down_dims[0] == model_config.intermediate_size && mlp_down_dims[1] == model_config.hidden_size) ||
        (mlp_down_dims[0] == model_config.hidden_size && mlp_down_dims[1] == model_config.intermediate_size),
        "MLP down projection dimensions should be [intermediate_size, hidden_size] or reversed, got {:?}",
        mlp_down_dims
    );

    // Test LayerNorm weights
    let layernorm_1 = find_layer_tensor("input_layernorm")
        .or_else(|| find_layer_tensor("ln_1"))
        .or_else(|| find_layer_tensor("attention_norm"))
        .unwrap_or_else(|| panic!("First LayerNorm tensor not found in layer 0"));

    // LayerNorm should be 1D with size d_model
    assert_eq!(layernorm_1.shape.dims().len(), 1, "LayerNorm should be 1D");
    assert_eq!(
        layernorm_1.shape.dims()[0],
        model_config.hidden_size,
        "LayerNorm size should match hidden_size"
    );

    let layernorm_2 = find_layer_tensor("post_attention_layernorm")
        .or_else(|| find_layer_tensor("ln_2"))
        .or_else(|| find_layer_tensor("ffn_norm"))
        .unwrap_or_else(|| panic!("Second LayerNorm tensor not found in layer 0"));

    // Second LayerNorm should also be 1D with size d_model
    assert_eq!(
        layernorm_2.shape.dims().len(),
        1,
        "Second LayerNorm should be 1D"
    );
    assert_eq!(
        layernorm_2.shape.dims()[0],
        model_config.hidden_size,
        "Second LayerNorm size should match hidden_size"
    );
}

/// Helper function to extract ModelConfig from GGUF metadata
fn extract_model_config_from_metadata(metadata: &GgufMetadata) -> ModelConfig {
    use rocmforge::model::config::ModelType;

    // Convert architecture string to ModelType
    let model_type = match metadata.architecture.to_lowercase().as_str() {
        "llama" => ModelType::Llama,
        "qwen" => ModelType::Qwen,
        _ => ModelType::Llama, // Default to Llama
    };

    ModelConfig::new(
        metadata.num_layers,
        metadata.num_heads,
        metadata.head_dim,
        metadata.hidden_size,
        metadata.max_position_embeddings,
        metadata.intermediate_size,
        metadata.vocab_size,
        model_type,
    )
}
