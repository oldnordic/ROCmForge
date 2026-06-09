use super::super::error::GpuResult;
use super::super::ffi::hipStream_t;
use super::super::kernels::rms_norm_on_stream;

pub fn gpu_dispatch_rms_norm(
    _device: &super::super::device::GpuDevice,
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    n: usize,
    eps: f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    rms_norm_on_stream(x, weight, out, n, eps, stream)
}
