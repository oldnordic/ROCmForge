#[cfg(test)]
mod hip_isolation_tests {
    use serial_test::serial;

    #[link(name = "amdhip64")]
    extern "C" {
        fn hipInit(flags: u32) -> i32;
        fn hipGetDeviceCount(count: *mut i32) -> i32;
    }

    const hipSuccess: i32 = 0;

    #[test]
    #[serial]
    fn test_minimal_hip() {
        println!("Testing minimal HIP in test environment...");

        unsafe {
            println!("Calling hipInit(0)...");
            let result = hipInit(0);
            println!("hipInit result: {}", result);

            assert_eq!(result, hipSuccess, "HIP initialization should succeed");

            let mut count: i32 = 0;
            println!("Calling hipGetDeviceCount...");
            let device_result = hipGetDeviceCount(&mut count);
            println!(
                "hipGetDeviceCount result: {}, count: {}",
                device_result, count
            );

            assert_eq!(
                device_result, hipSuccess,
                "Getting device count should succeed"
            );
            assert!(count > 0, "Should have at least one device");
        }
    }
}
