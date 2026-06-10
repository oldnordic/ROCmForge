#[cfg(feature = "gpu")]
use rocmforge::gpu;

#[cfg(feature = "gpu")]
pub(crate) struct GpuRuntimeState {
    pub gpu_caps: gpu::GpuCapabilities,
    pub vram_session: gpu::VramSession,
    pub device: &'static gpu::GpuDevice,
}

#[cfg(feature = "gpu")]
pub(crate) fn prepare_gpu_runtime(
    warn_experimental_kernels: bool,
) -> Result<GpuRuntimeState, Box<dyn std::error::Error>> {
    eprint!("Detecting GPU capabilities... ");
    let gpu_caps = gpu::detect().ok_or("GPU requested but no AMD GPU detected")?;
    eprintln!("done");
    eprintln!("  GPU: {}", gpu_caps.device_name);

    let vram_session = gpu::VramSession::new(gpu_caps.device_id)
        .map_err(|e| format!("VRAM query failed: {}", e))?;

    if warn_experimental_kernels && gpu::safety::experimental_gpu_kernels_enabled() {
        eprintln!("  WARNING: ROCMFORGE_ENABLE_EXPERIMENTAL_GPU_KERNELS=1");
        eprintln!("           Sparse CSR / MPO kernels are enabled. These can fault on");
        eprintln!("           display-attached GPUs. Use only for testing.");
    }

    eprint!("Initializing GPU device... ");
    let device =
        gpu::GpuDevice::get_or_init(gpu_caps.device_id).map_err(|e| format!("gpu init: {}", e))?;
    eprintln!("done");

    Ok(GpuRuntimeState {
        gpu_caps,
        vram_session,
        device,
    })
}
