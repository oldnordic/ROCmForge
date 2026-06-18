use crate::config::ModelConfig;
use crate::gpu::cache::{GpuForwardScratch, GpuKvCache};
use crate::gpu::decode_graph_health::{
    record_decode_graph_cache_hit, record_decode_graph_cache_miss,
    record_decode_graph_capture_attempted, record_decode_graph_capture_failed,
    record_decode_graph_capture_succeeded, record_decode_graph_fallback_to_non_graph,
    record_decode_graph_replay_attempted, record_decode_graph_replay_failed,
    record_decode_graph_replay_succeeded, record_decode_graph_update_attempted,
    record_decode_graph_update_succeeded,
};
use crate::gpu::decode_graph_keys::gpu_full_decode_graph_key;
use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi;
use crate::gpu::graph::HipGraph;
use crate::gpu::kernels::q8_0_workspace_bytes;
use crate::gpu::weights::GpuModelWeights;

use super::layer::gpu_layer_forward_from_state_on_stream;
use super::logits::{gpu_launch_greedy_logits_tail_on_stream, gpu_read_greedy_argmax_result};
use super::utils::decode_graph_disabled;

pub(super) fn gpu_launch_full_greedy_decode_on_stream(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<()> {
    for layer_idx in 0..config.num_layers {
        gpu_layer_forward_from_state_on_stream(
            device,
            gpu_weights.layer(layer_idx),
            kv,
            scratch,
            layer_idx,
            config,
        )?;
    }

    gpu_launch_greedy_logits_tail_on_stream(device, gpu_weights, scratch, config)
}

pub(super) fn gpu_launch_full_greedy_decode_with_readback_on_stream(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<()> {
    gpu_launch_full_greedy_decode_on_stream(device, gpu_weights, kv, scratch, config)?;
    gpu_read_greedy_argmax_result(device, scratch, config.vocab_size)
}

pub(super) fn gpu_capture_full_greedy_decode_graph(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<crate::gpu::graph::CapturedDecodeGraph> {
    let key = gpu_full_decode_graph_key(device, gpu_weights, kv, config)?;
    let max_q8_input = config.hidden_size.max(config.intermediate_size);
    device.reserve_q8_workspace(q8_0_workspace_bytes(max_q8_input))?;
    scratch.upload_decode_state(0, 1, device.stream())?;
    scratch.upload_positions(1, 0, config.max_seq_len, device.stream())?;
    device.synchronize()?;
    device.begin_capture(ffi::hipStreamCaptureMode::hipStreamCaptureModeGlobal)?;
    let capture_result = gpu_launch_full_greedy_decode_with_readback_on_stream(
        device,
        gpu_weights,
        kv,
        scratch,
        config,
    );
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

pub(super) fn gpu_try_full_greedy_decode_graph(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    pos: usize,
    config: &ModelConfig,
) -> GpuResult<Option<u32>> {
    if decode_graph_disabled(gpu_weights) {
        return Ok(None);
    }

    let key = gpu_full_decode_graph_key(device, gpu_weights, kv, config)?;
    if !scratch.has_decode_graph_for(key) {
        record_decode_graph_cache_miss();

        // First time: capture and instantiate the graph
        let capture_status = match device.stream_capture_status() {
            Ok(status) => status,
            Err(_) => {
                record_decode_graph_fallback_to_non_graph("stream capture status query failed");
                return Ok(None);
            }
        };
        if capture_status != ffi::hipStreamCaptureStatus::hipStreamCaptureStatusNone {
            record_decode_graph_fallback_to_non_graph("stream already capturing");
            return Ok(None);
        }

        // Capture graph with initial decode state
        record_decode_graph_capture_attempted();
        // Pre-allocate the Q8 activation workspace so that the Q4_0 Q8 fastpath
        // can reuse it during capture; lazy growth inside capture would call
        // hipStreamSynchronize/hipMalloc, which are not capture-safe.
        let max_q8_input = config.hidden_size.max(config.intermediate_size);
        device.reserve_q8_workspace(q8_0_workspace_bytes(max_q8_input))?;
        scratch.upload_decode_state(0, 1, device.stream())?;
        scratch.upload_positions(1, 0, config.max_seq_len, device.stream())?;
        device.begin_capture(ffi::hipStreamCaptureMode::hipStreamCaptureModeGlobal)?;
        let capture_res = gpu_launch_full_greedy_decode_with_readback_on_stream(
            device,
            gpu_weights,
            kv,
            scratch,
            config,
        );
        let raw_graph = device.end_capture()?;
        match capture_res {
            Ok(()) => {
                record_decode_graph_capture_succeeded();
            }
            Err(GpuError::InvalidWeightLayout { .. })
            | Err(GpuError::UnsupportedWeightType { .. })
            | Err(GpuError::UnsupportedOperation { .. }) => {
                record_decode_graph_capture_failed("unsupported weight layout/type/operation");
                return Ok(None);
            }
            Err(e) => {
                record_decode_graph_capture_failed(&format!("{:?}", e));
                return Err(e);
            }
        }

        let new_graph = HipGraph::from_raw(raw_graph);
        if scratch.decode_graph().is_some() {
            record_decode_graph_update_attempted();
            match scratch.try_update_decode_graph(new_graph, key)? {
                Ok(()) => {
                    record_decode_graph_update_succeeded();
                }
                Err(g) => {
                    record_decode_graph_update_succeeded();
                    let captured =
                        crate::gpu::graph::CapturedDecodeGraph::from_captured_graph(g, key)?;
                    scratch.replace_decode_graph(captured);
                }
            }
        } else {
            let captured =
                crate::gpu::graph::CapturedDecodeGraph::from_captured_graph(new_graph, key)?;
            scratch.replace_decode_graph(captured);
        }
    } else {
        record_decode_graph_cache_hit();
    }

    // For each token: upload new decode state and launch the graph
    scratch.upload_decode_state(pos, pos + 1, device.stream())?;
    scratch.upload_positions(pos + 1, 0, config.max_seq_len, device.stream())?;
    if let Some(graph) = scratch.decode_graph() {
        record_decode_graph_replay_attempted();
        if graph.launch(device.stream()).is_ok() {
            // Captured graph already includes the fixed argmax D2H readback.
            // Keep the stream sync before reading pinned host memory.
            device.synchronize()?;
            let token = scratch.argmax_result_index.as_slice::<i32>()[0];
            record_decode_graph_replay_succeeded();
            return Ok(Some(token as u32));
        }
        record_decode_graph_replay_failed("graph launch failed");
        scratch.clear_decode_graph();
    }

    Ok(None)
}
