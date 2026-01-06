//! Test to reproduce InferenceEngine crash

use rocmforge::engine::{EngineConfig, InferenceEngine};

#[test]
fn test_engine_creation_minimal() {
    println!("Starting InferenceEngine::new() test...");
    eprintln!("DEBUG: About to call InferenceEngine::new()");

    let engine = InferenceEngine::new(EngineConfig::default())
        .expect("Engine creation should succeed");

    eprintln!("DEBUG: InferenceEngine created successfully");
    println!("SUCCESS: Engine created");

    std::mem::drop(engine);
    println!("SUCCESS: Engine dropped");
}
