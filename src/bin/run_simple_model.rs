//! Simple model runner binary
//! Demonstrates end-to-end forward pass with a minimal transformer model

use rocmforge::model::{ModelBackend, ModelError, SimpleModel};
use std::env;

fn main() -> Result<(), ModelError> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("ROCmForge Simple Model Runner");
    println!("=============================");

    // Parse command line arguments for backend selection
    let args: Vec<String> = env::args().collect();
    let backend = if args.len() > 1 && args[1] == "gpu" {
        println!("Using GPU backend");
        ModelBackend::Gpu
    } else {
        println!("Using CPU backend");
        ModelBackend::Cpu
    };

    // Model configuration
    let vocab_size = 1000;
    let dim = 64;
    let num_layers = 2;
    let max_seq_len = 8;
    let seed = 42;

    println!("Model configuration:");
    println!("  Vocab size: {}", vocab_size);
    println!("  Hidden dim: {}", dim);
    println!("  Layers: {}", num_layers);
    println!("  Max seq len: {}", max_seq_len);
    println!("  Random seed: {}", seed);
    println!();

    // Create model
    println!("Creating model...");
    let model = SimpleModel::new(vocab_size, dim, num_layers, max_seq_len, backend, seed);
    println!("Model created successfully!");

    // Create dummy input
    let input_tokens = vec![1, 5, 9, 3, 7, 2, 8, 4]; // 8 tokens
    println!("Input tokens: {:?}", input_tokens);
    println!("Input shape: ({},)", input_tokens.len());

    // Run forward pass
    println!("Running forward pass...");
    let start_time = std::time::Instant::now();
    let output = model.forward(&input_tokens)?;
    let duration = start_time.elapsed();

    // Print results
    println!("Forward pass completed in {:?}", duration);
    println!("Output shape: ({},)", output.len());

    // Calculate checksum
    let checksum: f32 = output.iter().sum();
    println!("Output checksum (sum of all elements): {:.6}", checksum);

    // Print some sample output values
    println!("First 10 output values:");
    for (i, &val) in output.iter().take(10).enumerate() {
        println!("  [{}]: {:.6}", i, val);
    }

    // Verify output is finite
    let all_finite = output.iter().all(|x| x.is_finite());
    if all_finite {
        println!("✓ All output values are finite");
    } else {
        println!("✗ Some output values are not finite!");
        return Err(ModelError::ShapeMismatch(
            "Output contains non-finite values".to_string(),
        ));
    }

    // Verify output length
    let expected_len = input_tokens.len() * dim;
    if output.len() == expected_len {
        println!("✓ Output length matches expected: {}", expected_len);
    } else {
        println!(
            "✗ Output length mismatch! Expected {}, got {}",
            expected_len,
            output.len()
        );
        return Err(ModelError::ShapeMismatch(format!(
            "Expected output length {}, got {}",
            expected_len,
            output.len()
        )));
    }

    println!();
    println!("🎉 Model forward pass completed successfully!");
    Ok(())
}
