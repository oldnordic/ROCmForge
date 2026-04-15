// Test Q8_0 model output coherence
// Q8_0 is used for embeddings and lm_head
use std::process::Command;

#[test]
#[ignore = "Requires Q8_0 model file and GPU"]
fn test_q8_0_model_output_coherence() {
    let model_path = "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q8_0.gguf";

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
        .output()
        .expect("Failed to execute rocmforge");

    // Check command succeeded
    assert!(output.status.success(), "Command failed: {:?}", output.status);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Check for GPU crashes
    assert!(!stderr.contains("HIP_ERROR"), "GPU error detected: {}", stderr);
    assert!(!stderr.contains("GPU reset"), "GPU reset detected: {}", stderr);

    // Check output is coherent English
    let output_text = stdout.trim();

    // Basic sanity checks
    assert!(
        output_text.len() > 5,
        "Output too short: '{}'",
        output_text
    );

    // Should have English characters
    let has_english = output_text.chars().any(|c| c.is_ascii_alphabetic());
    assert!(has_english, "Output contains no English text: '{}'", output_text);

    // Should not have mixed Chinese/English (garbage symptom)
    let has_chinese = output_text.chars().any(|c| {
        let cp = c as u32;
        (0x4E00..=0x9FFF).contains(&cp)
    });

    assert!(
        !has_chinese,
        "Output has mixed Chinese/English (garbage symptom): '{}'",
        output_text
    );

    println!("✅ Q8_0 output is coherent: '{}'", output_text);
}

#[test]
fn test_q8_0_correctness_infrastructure() {
    // Verify our test infrastructure works

    // Simulate broken output (mixed Chinese/English)
    let broken_output = "attacks一排流严重一处 Cackets plasma A";

    let has_chinese = broken_output.chars().any(|c| {
        let cp = c as u32;
        (0x4E00..=0x9FFF).contains(&cp)
    });

    assert!(has_chinese, "Should detect Chinese characters");

    // Simulate correct output
    let correct_output = "Hello, how are you today?";

    let has_chinese_correct = correct_output.chars().any(|c| {
        let cp = c as u32;
        (0x4E00..=0x9FFF).contains(&cp)
    });

    assert!(!has_chinese_correct, "Should not have Chinese in correct output");

    println!("✅ Q8_0 correctness test infrastructure works");
}
