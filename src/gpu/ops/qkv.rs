use super::super::device::GpuDevice;
use super::super::error::{GpuError, GpuResult};
use super::super::ffi::hipStream_t;
use super::super::kernels::{
    add_on_stream, fused_qkv_rope_q4_0_gqa_on_stream, gemv_qkv_q4_0_f32_on_stream,
    gemv_qkv_q4_0_f32_on_stream_variant,
};
use super::super::launch_autotune::{lookup_qkv_variant, select_qkv_variant, VariantId};
use super::super::safety::{launch_autotune_enabled, use_dp4a_enabled};
use super::super::weights::{GpuBuffer, SvdCorrection, WeightMeta};
use crate::loader::GgmlType;

use super::gemv::{gpu_dispatch_gemv_on_stream, gpu_dispatch_gemv_svd_on_stream};

pub fn gpu_dispatch_fused_qkv_on_stream(
    device: &GpuDevice,
    w_q: &GpuBuffer,
    q_meta: &WeightMeta,
    q_svd: Option<&SvdCorrection>,
    q_bias: Option<&GpuBuffer>,
    w_k: &GpuBuffer,
    k_meta: &WeightMeta,
    k_svd: Option<&SvdCorrection>,
    k_bias: Option<&GpuBuffer>,
    w_v: &GpuBuffer,
    v_meta: &WeightMeta,
    v_svd: Option<&SvdCorrection>,
    v_bias: Option<&GpuBuffer>,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    q_size: usize,
    kv_size: usize,
    h: usize,
    temp_vector: *mut f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if q_svd.is_some() || k_svd.is_some() || v_svd.is_some() {
        gpu_dispatch_gemv_svd_on_stream(
            device,
            w_q,
            q_meta,
            q_svd,
            input,
            out_q,
            q_size,
            h,
            temp_vector,
            stream,
        )?;
        gpu_dispatch_gemv_svd_on_stream(
            device,
            w_k,
            k_meta,
            k_svd,
            input,
            out_k,
            kv_size,
            h,
            temp_vector,
            stream,
        )?;
        gpu_dispatch_gemv_svd_on_stream(
            device,
            w_v,
            v_meta,
            v_svd,
            input,
            out_v,
            kv_size,
            h,
            temp_vector,
            stream,
        )?;

        if let Some(bias) = q_bias {
            unsafe {
                add_on_stream(out_q, bias.as_ptr() as *const f32, out_q, q_size, stream)?;
            }
        }
        if let Some(bias) = k_bias {
            unsafe {
                add_on_stream(out_k, bias.as_ptr() as *const f32, out_k, kv_size, stream)?;
            }
        }
        if let Some(bias) = v_bias {
            unsafe {
                add_on_stream(out_v, bias.as_ptr() as *const f32, out_v, kv_size, stream)?;
            }
        }
        return Ok(());
    }

    if q_meta.wtype == GgmlType::Q4_0
        && k_meta.wtype == GgmlType::Q4_0
        && v_meta.wtype == GgmlType::Q4_0
    {
        let capture_active = matches!(
            super::super::ffi::hip_stream_is_capturing(stream),
            Err(_)
                | Ok(super::super::ffi::hipStreamCaptureStatus::hipStreamCaptureStatusActive)
                | Ok(super::super::ffi::hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
        );

        if launch_autotune_enabled() {
            let variant = if capture_active {
                lookup_qkv_variant(h, q_size, kv_size).unwrap_or(VariantId::Baseline)
            } else {
                select_qkv_variant(h, q_size, kv_size, |v| {
                    let result = unsafe {
                        gemv_qkv_q4_0_f32_on_stream_variant(
                            w_q.as_ptr() as *const u8,
                            w_k.as_ptr() as *const u8,
                            w_v.as_ptr() as *const u8,
                            q_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                            k_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                            v_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                            input,
                            out_q,
                            out_k,
                            out_v,
                            h,
                            q_size,
                            kv_size,
                            stream,
                            v as i32,
                        )
                    };
                    super::super::ffi::hip_stream_synchronize(stream)?;
                    result
                })
            };

            unsafe {
                gemv_qkv_q4_0_f32_on_stream_variant(
                    w_q.as_ptr() as *const u8,
                    w_k.as_ptr() as *const u8,
                    w_v.as_ptr() as *const u8,
                    q_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    k_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    v_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    input,
                    out_q,
                    out_k,
                    out_v,
                    h,
                    q_size,
                    kv_size,
                    stream,
                    variant as i32,
                )?;
            }
        } else {
            unsafe {
                gemv_qkv_q4_0_f32_on_stream(
                    w_q.as_ptr() as *const u8,
                    w_k.as_ptr() as *const u8,
                    w_v.as_ptr() as *const u8,
                    q_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    k_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    v_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    input,
                    out_q,
                    out_k,
                    out_v,
                    h,
                    q_size,
                    kv_size,
                    stream,
                )?;
            }
        }
        return Ok(());
    }

    gpu_dispatch_gemv_on_stream(device, w_q, q_meta, input, out_q, q_size, h, stream)?;
    gpu_dispatch_gemv_on_stream(device, w_k, k_meta, input, out_k, kv_size, h, stream)?;
    gpu_dispatch_gemv_on_stream(device, w_v, v_meta, input, out_v, kv_size, h, stream)?;

    if let Some(bias) = q_bias {
        unsafe {
            add_on_stream(out_q, bias.as_ptr() as *const f32, out_q, q_size, stream)?;
        }
    }
    if let Some(bias) = k_bias {
        unsafe {
            add_on_stream(out_k, bias.as_ptr() as *const f32, out_k, kv_size, stream)?;
        }
    }
    if let Some(bias) = v_bias {
        unsafe {
            add_on_stream(out_v, bias.as_ptr() as *const f32, out_v, kv_size, stream)?;
        }
    }

    Ok(())
}

pub fn gpu_dispatch_fused_qkv_gqa_on_stream(
    device: &GpuDevice,
    w_q: &GpuBuffer,
    q_meta: &WeightMeta,
    q_bias: Option<&GpuBuffer>,
    w_k: &GpuBuffer,
    k_meta: &WeightMeta,
    k_bias: Option<&GpuBuffer>,
    w_v: &GpuBuffer,
    v_meta: &WeightMeta,
    v_bias: Option<&GpuBuffer>,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    q_size: usize,
    kv_size: usize,
    h: usize,
    pos_ptr: *const i32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if !q_size.is_multiple_of(kv_size) {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "GQA requires q_size divisible by kv_size ({} % {} != 0)",
                q_size, kv_size
            ),
        });
    }

    let n_heads = q_size / h;
    let n_kv_heads = kv_size / h;

    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "GQA requires n_heads divisible by n_kv_heads ({} % {} != 0)",
                n_heads, n_kv_heads
            ),
        });
    }

    let features = super::super::features::GpuFeatures::detect(device)?;

    let rope_theta = 10000.0f32;
    let rope_neox = true;

    let bias_q_ptr = q_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32);
    let bias_k_ptr = k_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32);
    let bias_v_ptr = v_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32);

    if features.has_dp4a && use_dp4a_enabled() {
        // DP4A-optimized path available but not yet wired
    }

    fused_qkv_rope_q4_0_gqa_on_stream(
        device,
        w_q.as_ptr() as *const u8,
        w_k.as_ptr() as *const u8,
        w_v.as_ptr() as *const u8,
        input,
        out_q,
        out_k,
        out_v,
        0,
        n_heads,
        n_kv_heads,
        h,
        rope_theta,
        rope_neox,
        stream,
    )?;

    super::super::kernels::rope::rope_heads_from_state_on_stream(
        out_q, pos_ptr, n_heads, h, rope_theta, rope_neox, stream,
    )?;
    super::super::kernels::rope::rope_heads_from_state_on_stream(
        out_k, pos_ptr, n_kv_heads, h, rope_theta, rope_neox, stream,
    )?;

    Ok(())
}

pub fn gpu_dispatch_fused_qkv(
    device: &GpuDevice,
    w_q: &GpuBuffer,
    q_meta: &WeightMeta,
    q_bias: Option<&GpuBuffer>,
    w_k: &GpuBuffer,
    k_meta: &WeightMeta,
    k_bias: Option<&GpuBuffer>,
    w_v: &GpuBuffer,
    v_meta: &WeightMeta,
    v_bias: Option<&GpuBuffer>,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    q_size: usize,
    kv_size: usize,
    h: usize,
) -> GpuResult<()> {
    gpu_dispatch_fused_qkv_on_stream(
        device,
        w_q,
        q_meta,
        None,
        q_bias,
        w_k,
        k_meta,
        None,
        k_bias,
        w_v,
        v_meta,
        None,
        v_bias,
        input,
        out_q,
        out_k,
        out_v,
        q_size,
        kv_size,
        h,
        std::ptr::null_mut(),
        hipStream_t::null(),
    )
}
