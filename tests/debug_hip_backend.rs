//! Enhanced debugging test for HipBackend initialization
//!
//! Purpose: Isolate and debug the crash that supposedly happens during
//! InferenceEngine::new() initialization, specifically during/after
//! HipBackend::new() returns.

use rocmforge::backend::HipBackend;
use std::io::Write;

#[test]
fn test_hip_backend_detailed() {
    eprintln!("=== DEBUG: Starting test_hip_backend_detailed ===");
    let _ = std::io::stderr().flush();

    eprintln!("DEBUG: About to call HipBackend::new()...");
    let _ = std::io::stderr().flush();

    let backend_result = HipBackend::new();

    eprintln!("DEBUG: HipBackend::new() returned");
    let _ = std::io::stderr().flush();

    match backend_result {
        Ok(backend) => {
            eprintln!("DEBUG: HipBackend::new() returned Ok");
            let _ = std::io::stderr().flush();

            // Test backend methods
            eprintln!("DEBUG: Calling backend.device()...");
            let _ = std::io::stderr().flush();
            let device = backend.device();
            eprintln!("DEBUG: Got device: {}", device.name);
            let _ = std::io::stderr().flush();

            eprintln!("DEBUG: Calling backend.stream()...");
            let _ = std::io::stderr().flush();
            let _stream = backend.stream();
            eprintln!("DEBUG: Got stream successfully");
            let _ = std::io::stderr().flush();

            // Explicit drop to test Drop impl
            eprintln!("DEBUG: About to drop backend...");
            let _ = std::io::stderr().flush();
            drop(backend);
            eprintln!("DEBUG: Backend dropped successfully");
            let _ = std::io::stderr().flush();
        }
        Err(e) => {
            eprintln!("DEBUG: HipBackend::new() returned Err: {}", e);
            let _ = std::io::stderr().flush();
            panic!("HipBackend::new() failed: {}", e);
        }
    }

    eprintln!("=== DEBUG: Test completed successfully ===");
    let _ = std::io::stderr().flush();
}

#[test]
fn test_hip_backend_move() {
    eprintln!("=== DEBUG: Starting test_hip_backend_move ===");
    let _ = std::io::stderr().flush();

    let backend1 = HipBackend::new().expect("Failed to create backend");
    eprintln!("DEBUG: backend1 created");
    let _ = std::io::stderr().flush();

    // Test move
    let backend2 = backend1;
    eprintln!("DEBUG: backend1 moved to backend2");
    let _ = std::io::stderr().flush();

    // Test clone
    let backend3 = backend2.clone();
    eprintln!("DEBUG: backend2 cloned to backend3");
    let _ = std::io::stderr().flush();

    // Drop in reverse order
    drop(backend3);
    eprintln!("DEBUG: backend3 dropped");
    let _ = std::io::stderr().flush();

    drop(backend2);
    eprintln!("DEBUG: backend2 dropped");
    let _ = std::io::stderr().flush();

    eprintln!("=== DEBUG: Test completed successfully ===");
    let _ = std::io::stderr().flush();
}

#[test]
fn test_hip_backend_multiple() {
    eprintln!("=== DEBUG: Starting test_hip_backend_multiple ===");
    let _ = std::io::stderr().flush();

    for i in 0..3 {
        eprintln!("DEBUG: Creating backend {}...", i);
        let _ = std::io::stderr().flush();

        let _backend = HipBackend::new().expect("Failed to create backend");
        eprintln!("DEBUG: Backend {} created and dropped", i);
        let _ = std::io::stderr().flush();
    }

    eprintln!("=== DEBUG: Test completed successfully ===");
    let _ = std::io::stderr().flush();
}

#[test]
fn test_hip_stream_validation() {
    use rocmforge::backend::HipStream;
    use std::ptr;

    eprintln!("=== DEBUG: Starting test_hip_stream_validation ===");
    let _ = std::io::stderr().flush();

    // Test basic stream creation
    eprintln!("DEBUG: Creating HipStream...");
    let _ = std::io::stderr().flush();
    let stream = HipStream::new().expect("Failed to create stream");
    eprintln!("DEBUG: HipStream created");
    let _ = std::io::stderr().flush();

    // Test synchronization
    eprintln!("DEBUG: Synchronizing stream...");
    let _ = std::io::stderr().flush();
    stream.synchronize().expect("Stream sync failed");
    eprintln!("DEBUG: Stream synchronized");
    let _ = std::io::stderr().flush();

    // Test double sync
    eprintln!("DEBUG: Synchronizing stream again...");
    let _ = std::io::stderr().flush();
    stream.synchronize().expect("Second stream sync failed");
    eprintln!("DEBUG: Stream synchronized again");
    let _ = std::io::stderr().flush();

    eprintln!("DEBUG: About to drop stream...");
    let _ = std::io::stderr().flush();
    drop(stream);
    eprintln!("DEBUG: Stream dropped");
    let _ = std::io::stderr().flush();

    eprintln!("=== DEBUG: Test completed successfully ===");
    let _ = std::io::stderr().flush();
}
