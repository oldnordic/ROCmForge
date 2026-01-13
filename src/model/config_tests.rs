//! Tests for ModelConfig MQA/GQA detection
//!
//! These tests verify that ModelConfig correctly detects:
//! - MQA (Multi-Query Attention): num_kv_heads = 1
//! - GQA (Grouped-Query Attention): 1 < num_kv_heads < num_attention_heads
//! - MHA (Multi-Head Attention): num_kv_heads = num_attention_heads or None

#[cfg(test)]
mod config_tests {
    use crate::model::config::{ModelConfig, ModelType};

    /// Helper to create a test config
    fn create_test_config(num_attention_heads: usize, num_kv_heads: Option<usize>) -> ModelConfig {
        ModelConfig {
            num_hidden_layers: 32,
            num_attention_heads,
            num_kv_heads,
            head_dim: 128,
            hidden_size: num_attention_heads * 128,
            max_position_embeddings: 2048,
            intermediate_size: 11008,
            vocab_size: 32000,
            model_type: ModelType::Llama,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
        }
    }

    #[test]
    fn test_config_mqa_detection() {
        // MQA: 32 query heads, 1 KV head
        let config = create_test_config(32, Some(1));

        assert!(config.is_mqa(), "Should detect MQA when num_kv_heads = 1");
        assert!(!config.is_gqa(), "Should not detect GQA for MQA");
        assert!(!config.is_mha(), "Should not detect MHA for MQA");
        assert_eq!(
            config.heads_per_kv(),
            32,
            "MQA should have 32 query heads per KV head"
        );
    }

    #[test]
    fn test_config_gqa_detection() {
        // GQA: 32 query heads, 8 KV heads (4:1 ratio)
        let config = create_test_config(32, Some(8));

        assert!(!config.is_mqa(), "Should not detect MQA for GQA");
        assert!(
            config.is_gqa(),
            "Should detect GQA when 1 < num_kv_heads < num_attention_heads"
        );
        assert!(!config.is_mha(), "Should not detect MHA for GQA");
        assert_eq!(
            config.heads_per_kv(),
            4,
            "GQA should have 4 query heads per KV head"
        );
    }

    #[test]
    fn test_config_mha_explicit() {
        // MHA: 32 query heads, 32 KV heads (standard)
        let config = create_test_config(32, Some(32));

        assert!(!config.is_mqa(), "Should not detect MQA for MHA");
        assert!(!config.is_gqa(), "Should not detect GQA for MHA");
        assert!(
            config.is_mha(),
            "Should detect MHA when num_kv_heads = num_attention_heads"
        );
        assert_eq!(
            config.heads_per_kv(),
            1,
            "MHA should have 1 query head per KV head"
        );
    }

    #[test]
    fn test_config_mha_default() {
        // MHA: num_kv_heads = None (defaults to num_attention_heads)
        let config = create_test_config(32, None);

        assert!(!config.is_mqa(), "Should not detect MQA for default MHA");
        assert!(!config.is_gqa(), "Should not detect GQA for default MHA");
        assert!(
            config.is_mha(),
            "Should detect MHA when num_kv_heads is None"
        );
        assert_eq!(
            config.heads_per_kv(),
            1,
            "Default MHA should have 1 query head per KV head"
        );
    }

    #[test]
    fn test_heads_per_kv_mqa() {
        // Test various MQA configurations
        let config_16_1 = create_test_config(16, Some(1));
        assert_eq!(
            config_16_1.heads_per_kv(),
            16,
            "16:1 MQA should have 16 query heads per KV"
        );

        let config_64_1 = create_test_config(64, Some(1));
        assert_eq!(
            config_64_1.heads_per_kv(),
            64,
            "64:1 MQA should have 64 query heads per KV"
        );
    }

    #[test]
    fn test_heads_per_kv_gqa() {
        // Test various GQA configurations
        let config_32_4 = create_test_config(32, Some(4));
        assert_eq!(
            config_32_4.heads_per_kv(),
            8,
            "32:4 GQA should have 8 query heads per KV"
        );

        let config_40_8 = create_test_config(40, Some(8));
        assert_eq!(
            config_40_8.heads_per_kv(),
            5,
            "40:8 GQA should have 5 query heads per KV"
        );
    }

    #[test]
    fn test_gqa_edge_case_2_kv_heads() {
        // GQA edge case: 32 query heads, 2 KV heads
        let config = create_test_config(32, Some(2));

        assert!(!config.is_mqa(), "Should not be MQA");
        assert!(config.is_gqa(), "Should be GQA with 2 KV heads");
        assert_eq!(
            config.heads_per_kv(),
            16,
            "Should have 16 query heads per KV head"
        );
    }

    #[test]
    fn test_real_world_llama2_7b() {
        // LLaMA 2 7B uses standard MHA (32:32)
        let config = ModelConfig::llama2_7b();

        assert!(config.is_mha(), "LLaMA 2 7B should use MHA");
        assert!(!config.is_mqa(), "LLaMA 2 7B should not be MQA");
        assert!(!config.is_gqa(), "LLaMA 2 7B should not be GQA");
        assert_eq!(
            config.heads_per_kv(),
            1,
            "LLaMA 2 7B should have 1:1 head ratio"
        );
    }

    #[test]
    fn test_validation_rejects_invalid_kv_heads() {
        // num_kv_heads > num_attention_heads should fail validation
        let config = create_test_config(8, Some(16)); // 8 query, 16 KV - INVALID!

        let result = config.validate();
        assert!(
            result.is_err(),
            "Should reject num_kv_heads > num_attention_heads"
        );
    }

    #[test]
    fn test_validation_rejects_zero_kv_heads() {
        let config = create_test_config(32, Some(0)); // 0 KV heads - INVALID!

        let result = config.validate();
        assert!(result.is_err(), "Should reject num_kv_heads = 0");
    }
}
