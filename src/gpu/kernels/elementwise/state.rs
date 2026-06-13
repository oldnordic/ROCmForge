use super::super::super::error::{GpuError, GpuResult};
use super::super::super::ffi::{hipError_t, hipStream_t};
use super::super::super::GpuDevice;
use std::os::raw::c_int;

extern "C" {
    fn gpu_zero_fill(ptr: *mut f32, n: c_int, stream: hipStream_t) -> hipError_t;
    fn gpu_increment_decode_state_on_stream(state: *mut i32, stream: hipStream_t) -> hipError_t;
}

/// Zero-fill GPU memory: ptr[i] = 0.0f for i in 0..n
///
/// Launches asynchronously on device's stream.
/// Caller must call device.synchronize() if sync needed.
pub fn zero_fill(ptr: *mut f32, n: usize, device: &GpuDevice) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "zero_fill: n cannot be zero".to_string(),
        });
    }

    let stream = device.stream();

    let result = unsafe { gpu_zero_fill(ptr, n as c_int, stream) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("zero_fill kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Increment in-device decode state `[pos, seq_len]` by 1 on an explicit stream.
pub fn increment_decode_state_on_stream(state: *mut i32, stream: hipStream_t) -> GpuResult<()> {
    if state.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "increment_decode_state_on_stream: state pointer must be non-null"
                .to_string(),
        });
    }

    let result = unsafe { gpu_increment_decode_state_on_stream(state, stream) };
    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!(
                "increment_decode_state_on_stream kernel failed: {:?}",
                result
            ),
        });
    }

    Ok(())
}
