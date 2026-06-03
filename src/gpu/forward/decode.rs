use crate::config::ModelConfig;
use crate::gpu::cache::{GpuForwardScratch, GpuKvCache};
use crate::gpu::decode_graph_keys::gpu_full_decode_graph_key;
use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi;
use crate::gpu::graph::HipGraph;
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
    scratch.upload_decode_state(0, 1, device.stream())?;
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
        // First time: capture and instantiate the graph
        let capture_status = match device.stream_capture_status() {
            Ok(status) => status,
            Err(_) => return Ok(None),
        };
        if capture_status != ffi::hipStreamCaptureStatus::hipStreamCaptureStatusNone {
            return Ok(None);
        }

        // Capture graph with initial decode state
        scratch.upload_decode_state(0, 1, device.stream())?;
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
            Ok(()) => {}
            Err(GpuError::InvalidWeightLayout { .. })
            | Err(GpuError::UnsupportedWeightType { .. })
            | Err(GpuError::UnsupportedOperation { .. }) => {
                return Ok(None);
            }
            Err(e) => return Err(e),
        }

        let new_graph = HipGraph::from_raw(raw_graph);
        if scratch.decode_graph().is_some() {
            match scratch.try_update_decode_graph(new_graph, key)? {
                Ok(()) => {}
                Err(g) => {
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
    }

    // For each token: upload new decode state and launch the graph
    scratch.upload_decode_state(pos, pos + 1, device.stream())?;
    if let Some(graph) = scratch.decode_graph() {
        if graph.launch(device.stream()).is_ok() {
            // Captured graph already includes the fixed argmax D2H readback.
            // Keep the stream sync before reading pinned host memory.
            device.synchronize()?;
            let token = scratch.argmax_result_index.as_slice::<i32>()[0];
            return Ok(Some(token as u32));
        }
        scratch.clear_decode_graph();
    }

    Ok(None)
}
