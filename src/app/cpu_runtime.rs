use rocmforge::cpu::SimdKernels;
#[cfg(feature = "gpu")]
use rocmforge::gpu;
use rocmforge::hardware::{CpuCapabilities, detect};

use super::cli::Args;
use super::cpu_debug::print_cpu_hardware_summary;

pub(crate) struct CpuRuntimeState {
    pub caps: CpuCapabilities,
}

fn select_cpu_backend(request_gpu: bool, gpu_available: bool) -> bool {
    request_gpu && gpu_available
}

pub(crate) fn prepare_cpu_runtime(
    args: &Args,
) -> Result<CpuRuntimeState, Box<dyn std::error::Error>> {
    eprint!("Detecting CPU capabilities... ");
    let caps: CpuCapabilities = detect().map_err(|e| format!("hardware detection: {}", e))?;
    let l3_cache_mb = if caps.has_l3_cache() {
        Some(caps.l3_cache_mb())
    } else {
        None
    };

    let simd_kernels = SimdKernels::new(caps.simd.kernel_preference());
    print_cpu_hardware_summary(
        caps.physical_cores,
        caps.logical_cpus,
        caps.simd.description(),
        l3_cache_mb,
        caps.total_memory_gb(),
        simd_kernels.description(),
    );

    #[cfg(feature = "gpu")]
    let gpu_available = {
        eprint!("Detecting GPU capabilities... ");
        let caps = gpu::detect();
        match &caps {
            Some(gpu) => {
                eprintln!("done");
                eprintln!("  GPU: {}", gpu.device_name);
                eprintln!(
                    "  VRAM: {:.1} GB / {:.1} GB",
                    gpu.free_vram_gb(),
                    gpu.total_vram_gb()
                );
            }
            None => {
                eprintln!("none detected");
            }
        }
        caps.is_some()
    };

    #[cfg(not(feature = "gpu"))]
    let gpu_available = false;

    if select_cpu_backend(args.gpu, gpu_available) {
        eprintln!("Device: GPU");
        return Err("GPU inference not implemented yet".into());
    }

    eprintln!("Device: CPU");
    Ok(CpuRuntimeState { caps })
}

#[cfg(test)]
mod tests {
    use super::select_cpu_backend;

    #[test]
    fn select_cpu_backend_requires_request_and_available_gpu() {
        assert!(!select_cpu_backend(false, false));
        assert!(!select_cpu_backend(false, true));
        assert!(!select_cpu_backend(true, false));
        assert!(select_cpu_backend(true, true));
    }
}
