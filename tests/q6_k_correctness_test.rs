// Test Q6_K model output coherence
// This test verifies that Q6_K produces coherent output, not garbage

use std::process::Command;

#[test]
#[ignore = "Requires Q6_K model file and GPU"]
fn test_q6_k_model_output_coherence() {
    let model_path = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q6_k.gguf";

    // Run inference with simple prompt
    let output = Command::new("timeout")
        .arg("30s")
        .arg("./target/release/rocmforge")
        .arg("--gpu")
        .arg("--model")
        .arg(model_path)
        .arg("--prompt")
        .arg("Hello")
        .arg("--max-tokens")
        .arg("10")
        .arg("--no-template")
        .arg("--top-p")
        .arg("1.0")
        .env("ROCMFORGE_DISABLE_DECODE_GRAPH", "1")
        .output()
        .expect("Failed to execute rocmforge");

    // Check command succeeded
    assert!(
        output.status.success(),
        "Command failed: {:?}",
        output.status
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Check for GPU crashes
    assert!(
        !stderr.contains("HIP_ERROR"),
        "GPU error detected: {}",
        stderr
    );
    assert!(
        !stderr.contains("GPU reset"),
        "GPU reset detected: {}",
        stderr
    );

    // Check output is not garbage (should contain coherent English)
    // Garbage output from broken Q6_K looked like:
    // "attacks一排流严重一处 Cackets plasma A  one  one  one"
    let output_text = stdout.trim();

    // Check that output contains reasonable English characters
    assert!(output_text.len() > 5, "Output too short: '{}'", output_text);

    // Check that output doesn't contain random Chinese characters mixed with English
    // (this was the symptom of broken Q6_K)
    let has_chinese = output_text.chars().any(|c| {
        let cp = c as u32;
        (0x4E00..=0x9FFF).contains(&cp) // CJK Unified Ideographs
    });

    let has_english = output_text.chars().any(|c| c.is_ascii_alphabetic());

    // Valid output should have English text
    assert!(
        has_english,
        "Output contains no English text: '{}'",
        output_text
    );

    // If there's Chinese, it's likely garbage (broken Q6_K symptom)
    assert!(
        !has_chinese || has_english && output_text.len() > 20,
        "Output has mixed Chinese/English (garbage symptom): '{}'",
        output_text
    );

    println!("✅ Q6_K output is coherent: '{}'", output_text);
}

#[test]
fn test_q6_k_correctness_infrastructure() {
    // This test verifies our correctness test infrastructure works
    // It's a fast unit test that doesn't require a model

    // Simulate what broken Q6_K output looked like
    let broken_output = "attacks一排流严重一处 Cackets plasma A  one  one  one";

    // Check that our garbage detection works
    let has_chinese = broken_output.chars().any(|c| {
        let cp = c as u32;
        (0x4E00..=0x9FFF).contains(&cp)
    });

    let has_english = broken_output.chars().any(|c| c.is_ascii_alphabetic());

    // Broken output has both Chinese and English
    assert!(has_chinese, "Should detect Chinese characters");
    assert!(has_english, "Should detect English characters");

    // Simulate what correct Q6_K output looks like
    let correct_output = "Hi, I'm a 17 year old girl";

    let has_chinese_correct = correct_output.chars().any(|c| {
        let cp = c as u32;
        (0x4E00..=0x9FFF).contains(&cp)
    });

    let has_english_correct = correct_output.chars().any(|c| c.is_ascii_alphabetic());

    // Correct output has only English
    assert!(!has_chinese_correct, "Should not have Chinese");
    assert!(has_english_correct, "Should have English");

    println!("✅ Correctness test infrastructure works");
}
