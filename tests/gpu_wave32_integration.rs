#![cfg(feature = "gpu")]

mod common;

use rocmforge::gpu::GpuDevice;
use serial_test::serial;

#[test]
#[serial]
fn test_wave32_device_properties() {
    require_gpu!();

    let device = GpuDevice::init(0).expect("Failed to init GPU");

    // Verify RDNA3 wave32 detection
    assert_eq!(device.warp_size(), 32, "RX 7900 XT should have wave32");

    // Verify VRAM stats work
    let stats = device.vram_stats().expect("Failed to get VRAM stats");
    assert!(stats.total_vram_gb() > 16.0, "Should have >16 GB VRAM");
    assert!(
        stats.safely_allocatable_gb() > 8.0,
        "Should have >8 GB safe VRAM"
    );

    // Explicit cleanup
    drop(device);
}

#[test]
#[serial]
fn test_wave32_kernel_launch_no_vram_leak() {
    require_gpu!();

    let device = GpuDevice::init(0).expect("Failed to init GPU");
    let before = device.vram_stats().expect("Failed to get VRAM before");

    // TODO: Launch a simple wave32 kernel here
    // For now, just verify VRAM tracking works

    drop(device);
    let after = GpuDevice::init(0).expect("Failed to re-init GPU");
    let after_stats = after.vram_stats().expect("Failed to get VRAM after");

    // Allow 10 MB tolerance for driver overhead
    let leaked_mb = (before.used_vram as i64 - after_stats.used_vram as i64).abs() / (1024 * 1024);
    assert!(leaked_mb <= 10, "VRAM leak detected: {} MB", leaked_mb);
}
