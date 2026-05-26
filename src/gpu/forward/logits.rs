use crate::config::ModelConfig;
use crate::gpu::cache::GpuForwardScratch;
use crate::gpu::decode_graph_keys::gpu_greedy_logits_graph_key;
use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi;
use crate::gpu::graph::HipGraph;
use crate::gpu::kernels::elementwise::{argmax_f32, argmax_f32_on_stream};
use crate::gpu::ops::{gpu_dispatch_gemv_on_stream, gpu_dispatch_rms_norm};
use crate::gpu::weights::GpuModelWeights;

use super::utils::decode_graph_disabled;
use crate::gpu::decode_profile::{
    decode_stage_profiling_enabled, profile_decode_stage, record_tail_invocation, DecodeStage,
};

pub(super) fn gpu_launch_greedy_argmax(
    scratch: &mut GpuForwardScratch,
    vocab_size: usize,
) -> GpuResult<()> {
    argmax_f32(
        scratch.logits_ptr(),
        scratch.argmax_partial_values_mut_ptr(),
        scratch.argmax_partial_indices_mut_ptr(),
        scratch.argmax_result_index_mut_ptr(),
        vocab_size,
    )
}

pub(super) fn gpu_read_greedy_argmax_result(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    _vocab_size: usize,
) -> GpuResult<()> {
    unsafe {
        ffi::hip_memcpy_d2h_async(
            scratch.argmax_result_index.as_ptr(),
            scratch.argmax_result_device.as_ptr(),
            std::mem::size_of::<i32>(),
            device.stream(),
        )?;
    }

    Ok(())
}

pub(super) fn gpu_greedy_argmax_token(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    vocab_size: usize,
) -> GpuResult<u32> {
    gpu_launch_greedy_argmax(scratch, vocab_size)?;
    gpu_read_greedy_argmax_result(device, scratch, vocab_size)?;
    device.synchronize()?;
    let index = scratch.argmax_result_index.as_slice::<i32>()[0];
    Ok(index as u32)
}

pub(super) fn gpu_launch_greedy_logits_tail_with_readback_on_stream(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<()> {
    gpu_launch_greedy_logits_tail_on_stream(device, gpu_weights, scratch, config)?;
    gpu_read_greedy_argmax_result(device, scratch, config.vocab_size)
}

pub(super) fn gpu_launch_greedy_logits_tail_on_stream(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    let v = config.vocab_size;
    if decode_stage_profiling_enabled() {
        record_tail_invocation();
    }
    profile_decode_stage(device, DecodeStage::LogitsNorm, || {
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_weights.output_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            config.rms_norm_eps,
            device.stream(),
        )
    })?;
    profile_decode_stage(device, DecodeStage::LogitsProj, || {
        gpu_dispatch_gemv_on_stream(
            device,
            &gpu_weights.lm_head,
            &gpu_weights.lm_head_meta,
            scratch.normed.as_ptr() as *const f32,
            scratch.logits.as_ptr() as *mut f32,
            v,
            h,
            device.stream(),
        )
    })?;
    profile_decode_stage(device, DecodeStage::Argmax, || {
        argmax_f32_on_stream(
            scratch.logits_ptr(),
            scratch.argmax_partial_values_mut_ptr(),
            scratch.argmax_partial_indices_mut_ptr(),
            scratch.argmax_result_index_mut_ptr(),
            v,
            device.stream(),
        )
    })
}

pub(super) fn gpu_capture_greedy_decode_graph(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<crate::gpu::graph::CapturedDecodeGraph> {
    let key = gpu_greedy_logits_graph_key(device, gpu_weights, config);
    device.begin_capture(ffi::hipStreamCaptureMode::hipStreamCaptureModeGlobal)?;
    let capture_result =
        gpu_launch_greedy_logits_tail_with_readback_on_stream(device, gpu_weights, scratch, config);
    let end_capture_result = device.end_capture();

    match capture_result {
        Ok(()) => {
            let graph = HipGraph::from_raw(end_capture_result?);
            crate::gpu::graph::CapturedDecodeGraph::from_captured_graph(graph, key)
        }
        Err(err) => {
            let _ = end_capture_result;
            Err(err)
        }
    }
}

pub(super) fn gpu_greedy_logits_tail_token(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<u32> {
    gpu_launch_greedy_logits_tail_on_stream(device, gpu_weights, scratch, config)?;
    gpu_read_greedy_argmax_result(device, scratch, config.vocab_size)?;
    device.synchronize()?;
    let index = scratch.argmax_result_index.as_slice::<i32>()[0];
    Ok(index as u32)
}

pub(super) fn gpu_try_greedy_decode_graph(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<u32> {
    if decode_graph_disabled(gpu_weights) {
        return gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config);
    }

    let key = gpu_greedy_logits_graph_key(device, gpu_weights, config);
    if !scratch.has_decode_graph_for(key) {
        scratch.clear_decode_graph();
        let capture_status = match device.stream_capture_status() {
            Ok(status) => status,
            Err(_) => return gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config),
        };
        if capture_status != ffi::hipStreamCaptureStatus::hipStreamCaptureStatusNone {
            return gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config);
        }

        match gpu_capture_greedy_decode_graph(device, gpu_weights, scratch, config) {
            Ok(graph) => {
                scratch.replace_decode_graph(graph);
            }
            Err(err @ GpuError::InvalidWeightLayout { .. })
            | Err(err @ GpuError::UnsupportedWeightType { .. }) => return Err(err),
            Err(_) => return gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config),
        }
    }

    // Check if we have position tracking for decode state updates
    let next_pos = scratch.decode_state_next_pos();

    if next_pos.is_some() {
        let pos = next_pos.unwrap();
        let has_graph = scratch.decode_graph().is_some();

        if has_graph {
            // CRITICAL FIX: Upload decode state before graph launch
            // The graph captures memory pointers but NOT values like position.
            // We must upload updated decode state before each replay to ensure correctness.
            scratch.upload_decode_state(pos, pos + 1, device.stream())?;

            // Get graph again after upload
            if let Some(graph) = scratch.decode_graph() {
                if graph.launch(device.stream()).is_ok() {
                    // Captured graph already includes the fixed argmax D2H readback.
                    // Keep the stream sync before reading pinned host memory.
                    device.synchronize()?;
                    let index = scratch.argmax_result_index.as_slice::<i32>()[0];
                    return Ok(index as u32);
                }
            }
            scratch.clear_decode_graph();
        }
    }

    gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config)
}
