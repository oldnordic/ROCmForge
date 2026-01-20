#[cfg(test)]
mod hip_backend_debug_tests {
    use crate::backend::hip_backend::*;
    use serial_test::serial;
    use std::sync::Arc;

    /// Helper: Get GPU backend or skip test if not available (llama.cpp pattern)
    fn get_backend_or_skip() -> Arc<HipBackend> {
        match HipBackend::new_checked() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("\n⚠️  GPU not available for hip_backend_debug_tests: {}", e);
                eprintln!("To enable these tests, ensure:");
                eprintln!("  1. AMD GPU is present");
                eprintln!("  2. ROCm is installed (check with rocm-smi)");
                eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
                eprintln!("\nSkipping test gracefully (llama.cpp pattern).\n");
                panic!("GPU_SKIP");
            }
        }
    }

    #[test]
    #[serial]
    fn test_detect_amd_gpu_step_by_step() {
        println!("Testing detect_amd_gpu step by step...");

        // First initialize HIP
        let _backend = get_backend_or_skip();

        // Step 1: Get device count
        let mut count: i32 = 0;
        let result = unsafe { super::super::backend::hip_backend::hipGetDeviceCount(&mut count) };
        println!("Device count result: {}, count: {}", result, count);
        assert_eq!(result, 0, "hipGetDeviceCount should succeed");
        assert!(count > 0, "Should have at least one device");

        // Step 2: Get properties using proper alignment
        let mut props = std::mem::MaybeUninit::<HipDeviceProp>::uninit();
        let result = unsafe {
            super::super::backend::hip_backend::hipGetDeviceProperties(props.as_mut_ptr(), 0)
        };
        println!("Get properties result: {}", result);
        assert_eq!(result, 0, "hipGetDeviceProperties should succeed");

        // Step 3: Read the properties safely
        let props = unsafe { props.assume_init() };
        println!("Device name: {}", props.name());
        println!("Total memory: {} bytes", props.total_global_mem());
        println!("Compute units: {}", props.multi_processor_count());
    }
}
