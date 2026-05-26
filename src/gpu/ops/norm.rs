use super::super::error::GpuResult;
use super::super::ffi::hipStream_t;
use super::super::kernels::{rms_norm_on_stream, rms_norm_vulkan_style};
use super::super::safety::experimental_gpu_kernels_enabled;

pub fn gpu_dispatch_rms_norm(
    _device: &super::super::device::GpuDevice,
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    n: usize,
    eps: f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if experimental_gpu_kernels_enabled() {
        if let Ok(()) = rms_norm_vulkan_style(x, weight, out, n, eps, stream) {
            return Ok(());
        }
    }

    rms_norm_on_stream(x, weight, out, n, eps, stream)
}
