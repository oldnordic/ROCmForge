//! Q8-activation helpers for decode hot paths.
//!
//! These wrappers keep the Q4_0 hot paths separate from the large
//! quantization wrapper module so targeted experiments do not keep growing the
//! generic kernel surface.

use super::super::error::{GpuError, GpuResult};
use super::super::ffi::{hipError_t, hipStream_t};
use std::os::raw::{c_int, c_void};

const QK4_0: usize = 32;
const QK8_0: usize = 32;
const Q8_0_BLOCK_SIZE: usize = 34;

pub fn q8_0_workspace_bytes(n: usize) -> usize {
    n.div_ceil(QK8_0) * Q8_0_BLOCK_SIZE
}

pub fn quantize_q8_0_on_stream(
    input: *const f32,
    output: *mut u8,
    n: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "quantize_q8_0_on_stream: n cannot be zero".to_string(),
        });
    }
    if input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "quantize_q8_0_on_stream: input/output pointers must be non-null"
                .to_string(),
        });
    }

    let result =
        unsafe { quantize_q8_0_on_stream_kernel(input, output as *mut c_void, n as c_int, stream) };
    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("quantize_q8_0_on_stream kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemv_q4_0_q8_0_residual_on_stream(
    weights_q4_0: *const u8,
    input_q8_0: *const u8,
    residual: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_0_q8_0_residual: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }
    if n_rows % QK4_0 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q4_0_q8_0_residual: n_rows must be multiple of {}, got {}",
                QK4_0, n_rows
            ),
        });
    }
    if weights_q4_0.is_null() || input_q8_0.is_null() || residual.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_0_q8_0_residual: kernel pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemv_q4_0_q8_0_residual_launch(
            weights_q4_0 as *const c_void,
            input_q8_0 as *const c_void,
            residual,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_q4_0_q8_0_residual kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemv_q4_0_f32_q8_inline_residual_on_stream(
    weights_q4_0: *const u8,
    input: *const f32,
    residual: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_0_f32_q8_inline_residual: n_rows and ncols_dst cannot be zero"
                .to_string(),
        });
    }
    if n_rows % QK4_0 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q4_0_f32_q8_inline_residual: n_rows must be multiple of {}, got {}",
                QK4_0, n_rows
            ),
        });
    }
    if weights_q4_0.is_null() || input.is_null() || residual.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_0_f32_q8_inline_residual: kernel pointers must be non-null"
                .to_string(),
        });
    }

    let result = unsafe {
        gemv_q4_0_f32_q8_inline_residual_launch(
            weights_q4_0 as *const c_void,
            input,
            residual,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!(
                "gemv_q4_0_f32_q8_inline_residual kernel failed: {:?}",
                result
            ),
        });
    }

    Ok(())
}

pub fn gemv_q4_0_f32_q8_inline_residual_on_stream_variant(
    weights_q4_0: *const u8,
    input: *const f32,
    residual: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    variant: i32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "gemv_q4_0_f32_q8_inline_residual_variant: n_rows and ncols_dst cannot be zero"
                    .to_string(),
        });
    }
    if n_rows % QK4_0 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q4_0_f32_q8_inline_residual_variant: n_rows must be multiple of {}, got {}",
                QK4_0, n_rows
            ),
        });
    }
    if weights_q4_0.is_null() || input.is_null() || residual.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "gemv_q4_0_f32_q8_inline_residual_variant: kernel pointers must be non-null"
                    .to_string(),
        });
    }

    let result = unsafe {
        gemv_q4_0_f32_q8_inline_residual_variant_launch(
            weights_q4_0 as *const c_void,
            input,
            residual,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            variant,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!(
                "gemv_q4_0_f32_q8_inline_residual_variant kernel failed: {:?}",
                result
            ),
        });
    }

    Ok(())
}

pub fn gemv_q4_0_q8_0_on_stream(
    weights_q4_0: *const u8,
    input_q8_0: *const u8,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_0_q8_0: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }
    if n_rows % QK4_0 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q4_0_q8_0: n_rows must be multiple of {}, got {}",
                QK4_0, n_rows
            ),
        });
    }
    if weights_q4_0.is_null() || input_q8_0.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_0_q8_0: kernel pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemv_q4_0_q8_0_launch(
            weights_q4_0 as *const c_void,
            input_q8_0 as *const c_void,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_q4_0_q8_0 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemv_gate_up_q4_0_q8_0_on_stream(
    w_gate_q4_0: *const u8,
    w_up_q4_0: *const u8,
    input_q8_0: *const u8,
    out_gate: *mut f32,
    out_up: *mut f32,
    n_rows: usize,
    n_ff: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || n_ff == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_gate_up_q4_0_q8_0: n_rows and n_ff cannot be zero".to_string(),
        });
    }
    if n_rows % QK4_0 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_gate_up_q4_0_q8_0: n_rows must be multiple of {}, got {}",
                QK4_0, n_rows
            ),
        });
    }
    if w_gate_q4_0.is_null()
        || w_up_q4_0.is_null()
        || input_q8_0.is_null()
        || out_gate.is_null()
        || out_up.is_null()
    {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_gate_up_q4_0_q8_0: kernel pointers must be non-null".to_string(),
        });
    }
    if out_gate == out_up {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_gate_up_q4_0_q8_0: outputs must be distinct buffers".to_string(),
        });
    }

    let result = unsafe {
        gemv_gate_up_q4_0_q8_0_launch(
            w_gate_q4_0 as *const c_void,
            w_up_q4_0 as *const c_void,
            input_q8_0 as *const c_void,
            out_gate,
            out_up,
            n_rows as c_int,
            n_ff as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_gate_up_q4_0_q8_0 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemv_gate_up_swiglu_q4_0_q8_0_on_stream(
    w_gate_q4_0: *const u8,
    w_up_q4_0: *const u8,
    input_q8_0: *const u8,
    out_swiglu: *mut f32,
    n_rows: usize,
    n_ff: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || n_ff == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_gate_up_swiglu_q4_0_q8_0: n_rows and n_ff cannot be zero"
                .to_string(),
        });
    }
    if n_rows % QK4_0 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_gate_up_swiglu_q4_0_q8_0: n_rows must be multiple of {}, got {}",
                QK4_0, n_rows
            ),
        });
    }
    if w_gate_q4_0.is_null() || w_up_q4_0.is_null() || input_q8_0.is_null() || out_swiglu.is_null()
    {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_gate_up_swiglu_q4_0_q8_0: kernel pointers must be non-null"
                .to_string(),
        });
    }

    let result = unsafe {
        gemv_gate_up_swiglu_q4_0_q8_0_launch(
            w_gate_q4_0 as *const c_void,
            w_up_q4_0 as *const c_void,
            input_q8_0 as *const c_void,
            out_swiglu,
            n_rows as c_int,
            n_ff as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_gate_up_swiglu_q4_0_q8_0 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemv_gate_up_swiglu_q4_0_f32_q8_inline_on_stream(
    w_gate_q4_0: *const u8,
    w_up_q4_0: *const u8,
    input: *const f32,
    out_swiglu: *mut f32,
    n_rows: usize,
    n_ff: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || n_ff == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_gate_up_swiglu_q4_0_f32_q8_inline: n_rows and n_ff cannot be zero"
                .to_string(),
        });
    }
    if n_rows % QK4_0 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline: n_rows must be multiple of {}, got {}",
                QK4_0, n_rows
            ),
        });
    }
    if w_gate_q4_0.is_null() || w_up_q4_0.is_null() || input.is_null() || out_swiglu.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_gate_up_swiglu_q4_0_f32_q8_inline: kernel pointers must be non-null"
                .to_string(),
        });
    }

    let result = unsafe {
        gemv_gate_up_swiglu_q4_0_f32_q8_inline_launch(
            w_gate_q4_0 as *const c_void,
            w_up_q4_0 as *const c_void,
            input,
            out_swiglu,
            n_rows as c_int,
            n_ff as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!(
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline kernel failed: {:?}",
                result
            ),
        });
    }

    Ok(())
}

pub fn gemv_gate_up_swiglu_q4_0_f32_q8_inline_on_stream_variant(
    w_gate_q4_0: *const u8,
    w_up_q4_0: *const u8,
    input: *const f32,
    out_swiglu: *mut f32,
    n_rows: usize,
    n_ff: usize,
    variant: i32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || n_ff == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline_variant: n_rows and n_ff cannot be zero"
                    .to_string(),
        });
    }
    if n_rows % QK4_0 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline_variant: n_rows must be multiple of {}, got {}",
                QK4_0, n_rows
            ),
        });
    }
    if w_gate_q4_0.is_null() || w_up_q4_0.is_null() || input.is_null() || out_swiglu.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline_variant: kernel pointers must be non-null"
                    .to_string(),
        });
    }

    let result = unsafe {
        gemv_gate_up_swiglu_q4_0_f32_q8_inline_variant_launch(
            w_gate_q4_0 as *const c_void,
            w_up_q4_0 as *const c_void,
            input,
            out_swiglu,
            n_rows as c_int,
            n_ff as c_int,
            variant as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!(
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline_variant kernel failed: {:?}",
                result
            ),
        });
    }

    Ok(())
}

pub fn gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_on_stream(
    w_gate_up_interleaved_q4_0: *const u8,
    input: *const f32,
    out_swiglu: *mut f32,
    n_rows: usize,
    n_ff: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || n_ff == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved: n_rows and n_ff cannot be zero"
                    .to_string(),
        });
    }
    if n_rows % QK4_0 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved: n_rows must be multiple of {}, got {}",
                QK4_0, n_rows
            ),
        });
    }
    if w_gate_up_interleaved_q4_0.is_null() || input.is_null() || out_swiglu.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved: kernel pointers must be non-null"
                    .to_string(),
        });
    }

    let result = unsafe {
        gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_launch(
            w_gate_up_interleaved_q4_0 as *const c_void,
            input,
            out_swiglu,
            n_rows as c_int,
            n_ff as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!(
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved kernel failed: {:?}",
                result
            ),
        });
    }

    Ok(())
}

pub fn gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_on_stream(
    w_gate_up_interleaved_tile4_q4_0: *const u8,
    input: *const f32,
    out_swiglu: *mut f32,
    n_rows: usize,
    n_ff: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || n_ff == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4: n_rows and n_ff cannot be zero"
                    .to_string(),
        });
    }
    if n_rows % QK4_0 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4: n_rows must be multiple of {}, got {}",
                QK4_0, n_rows
            ),
        });
    }
    if w_gate_up_interleaved_tile4_q4_0.is_null() || input.is_null() || out_swiglu.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4: kernel pointers must be non-null"
                    .to_string(),
        });
    }

    let result = unsafe {
        gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_launch(
            w_gate_up_interleaved_tile4_q4_0 as *const c_void,
            input,
            out_swiglu,
            n_rows as c_int,
            n_ff as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!(
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4 kernel failed: {:?}",
                result
            ),
        });
    }

    Ok(())
}

unsafe extern "C" {
    fn quantize_q8_0_on_stream_kernel(
        input: *const f32,
        output: *mut c_void,
        n: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_q4_0_q8_0_residual_launch(
        weights_q4_0: *const c_void,
        input_q8_0: *const c_void,
        residual: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_q4_0_f32_q8_inline_residual_launch(
        weights_q4_0: *const c_void,
        input: *const f32,
        residual: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_q4_0_f32_q8_inline_residual_variant_launch(
        weights_q4_0: *const c_void,
        input: *const f32,
        residual: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        variant: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_q4_0_q8_0_launch(
        weights_q4_0: *const c_void,
        input_q8_0: *const c_void,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_gate_up_q4_0_q8_0_launch(
        w_gate_q4_0: *const c_void,
        w_up_q4_0: *const c_void,
        input_q8_0: *const c_void,
        out_gate: *mut f32,
        out_up: *mut f32,
        n_rows: c_int,
        n_ff: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_gate_up_swiglu_q4_0_q8_0_launch(
        w_gate_q4_0: *const c_void,
        w_up_q4_0: *const c_void,
        input_q8_0: *const c_void,
        out_swiglu: *mut f32,
        n_rows: c_int,
        n_ff: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_gate_up_swiglu_q4_0_f32_q8_inline_launch(
        w_gate_q4_0: *const c_void,
        w_up_q4_0: *const c_void,
        input: *const f32,
        out_swiglu: *mut f32,
        n_rows: c_int,
        n_ff: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_gate_up_swiglu_q4_0_f32_q8_inline_variant_launch(
        w_gate_q4_0: *const c_void,
        w_up_q4_0: *const c_void,
        input: *const f32,
        out_swiglu: *mut f32,
        n_rows: c_int,
        n_ff: c_int,
        variant: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
    fn gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_launch(
        w_gate_up_interleaved_q4_0: *const c_void,
        input: *const f32,
        out_swiglu: *mut f32,
        n_rows: c_int,
        n_ff: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
    fn gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_launch(
        w_gate_up_interleaved_tile4_q4_0: *const c_void,
        input: *const f32,
        out_swiglu: *mut f32,
        n_rows: c_int,
        n_ff: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}
