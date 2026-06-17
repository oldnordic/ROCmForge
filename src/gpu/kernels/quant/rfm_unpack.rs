use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi::{hipError_t, hipStream_t};
use std::ffi::c_void;

unsafe extern "C" {
    fn unpack_q4_split_launch(
        input_split: *const c_void,
        output_q4_0: *mut c_void,
        num_blocks: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    fn unpack_q4_fused_gate_up_launch(
        input_fused: *const c_void,
        output_gate: *mut c_void,
        output_up: *mut c_void,
        output_interleaved: *mut c_void,
        output_interleaved_tile4: *mut c_void,
        intermediate_size: i32,
        hidden_size: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

/// GPU-accelerated unpacking of Q4Split RFM tensor into standard Q4_0 layout.
pub fn gpu_unpack_q4_split(
    input_split: *const u8,
    output_q4_0: *mut u8,
    num_blocks: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let res = unsafe {
        unpack_q4_split_launch(
            input_split as *const c_void,
            output_q4_0 as *mut c_void,
            num_blocks as i32,
            stream,
        )
    };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: format!("unpack_q4_split kernel failed: {:?}", res),
        });
    }
    Ok(())
}

/// GPU-accelerated unpacking of fused FFN gate-up block groups into gate, up, interleaved, and tile4 layouts.
#[allow(clippy::too_many_arguments, reason = "launch ABI")]
pub fn gpu_unpack_q4_fused_gate_up(
    input_fused: *const u8,
    output_gate: *mut u8,
    output_up: *mut u8,
    output_interleaved: *mut u8,
    output_interleaved_tile4: *mut u8,
    intermediate_size: usize,
    hidden_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let res = unsafe {
        unpack_q4_fused_gate_up_launch(
            input_fused as *const c_void,
            output_gate as *mut c_void,
            output_up as *mut c_void,
            output_interleaved as *mut c_void,
            output_interleaved_tile4 as *mut c_void,
            intermediate_size as i32,
            hidden_size as i32,
            stream,
        )
    };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: format!("unpack_q4_fused_gate_up kernel failed: {:?}", res),
        });
    }
    Ok(())
}
