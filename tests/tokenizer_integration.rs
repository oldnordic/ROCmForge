//! Integration tests for tokenizer with real model files.
//!
//! These tests use the Qwen2.5 model from the Memoria project.

use rocmforge::loader::GgufFile;
use rocmforge::tokenizer::BpeTokenizer;

const MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q8_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

#[test]
fn test_load_qwen25_tokenizer() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let gguf = GgufFile::open(MODEL_PATH).expect("Failed to load GGUF file");

    // Verify architecture
    assert_eq!(gguf.metadata.architecture, "qwen2");

    // Build tokenizer from GGUF
    let tok = BpeTokenizer::from_gguf(gguf.tokenizer_data());

    // Test vocab_size from tokenizer (not GGUF metadata key)
    let vocab_size = tok.vocab_size();
    assert!(vocab_size > 0, "vocab_size should be positive");
    assert_eq!(
        vocab_size,
        gguf.tokenizer_data().tokens.len(),
        "vocab_size must match tokenizer.tokens.len()"
    );
}

#[test]
fn test_encode_decode_roundtrip() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let gguf = GgufFile::open(MODEL_PATH).expect("Failed to load GGUF file");
    let tok = BpeTokenizer::from_gguf(gguf.tokenizer_data());

    // Test round-trip encoding/decoding
    let original = "Hello, world!";

    // Encode with special tokens
    let tokens = tok.encode(original, true);
    assert!(!tokens.is_empty(), "Encoded tokens should not be empty");

    // Decode (skip_special=false to include special tokens)
    let decoded = tok.decode(&tokens, false);
    assert_eq!(
        decoded, original,
        "Round-trip encode/decode should produce same text"
    );
}

#[test]
fn test_encode_decode_complex_text() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let gguf = GgufFile::open(MODEL_PATH).expect("Failed to load GGUF file");
    let tok = BpeTokenizer::from_gguf(gguf.tokenizer_data());

    // Test with more complex text
    let texts = vec![
        "The quick brown fox jumps over the lazy dog.",
        "12345 67890",
        "Hello\nWorld",
        "Special characters: @#$%^&*()",
    ];

    for original in texts {
        let tokens = tok.encode(original, false);
        let decoded = tok.decode(&tokens, false);
        assert_eq!(decoded, original, "Round-trip failed for: {:?}", original);
    }
}

#[test]
fn test_special_token_detection() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let gguf = GgufFile::open(MODEL_PATH).expect("Failed to load GGUF file");
    let tok = BpeTokenizer::from_gguf(gguf.tokenizer_data());

    // Test is_eog with EOS token
    if let Some(eos_id) = tok.eos_id() {
        assert!(tok.is_eog(eos_id), "EOS token should be end-of-generation");
    }

    // Test BOS token exists (Qwen2 uses BOS)
    if let Some(bos_id) = tok.bos_id() {
        assert!(bos_id > 0, "BOS token ID should be positive");
    }
}
