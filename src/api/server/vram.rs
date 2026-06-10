use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::api::types::*;
use super::state::ModelManager;
use super::utils::{error_response, bytes_to_gb};

pub(crate) async fn estimate_vram(
    State(state): State<Arc<ModelManager>>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    let entry = match state.get(&req.model).await {
        Some(e) => e,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Model '{}' not found", req.model),
                "invalid_model",
            );
        }
    };
    let config = &entry.config;

    let model_weights = std::fs::metadata(&entry.model_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    // KV cache: [num_layers, 2, max_seq, head_dim, num_heads] as f32.
    // simplified: hidden_size * num_layers * max_seq * 2 * sizeof(f32)
    let max_seq = config.max_seq_len.min(4096);
    let kv_estimate =
        config.hidden_size * config.num_layers * max_seq * 2 * std::mem::size_of::<f32>();

    // Scratch: prefill + decode scratch buffers (upper bound).
    let scratch_estimate =
        config.hidden_size * config.intermediate_size * 4 * std::mem::size_of::<f32>();

    let total = model_weights + kv_estimate + scratch_estimate;

    let (fits, budget_gb) = vram_budget_check(total as u64);

    let response = VramEstimate {
        model_id: req.model.clone(),
        model_weights_gb: bytes_to_gb(model_weights),
        kv_cache_gb: bytes_to_gb(kv_estimate),
        scratch_gb: bytes_to_gb(scratch_estimate),
        total_gb: bytes_to_gb(total),
        fits,
        inference_budget_gb: budget_gb,
    };

    (StatusCode::OK, Json(response)).into_response()
}

pub(crate) async fn list_vram() -> Response {
    let info = collect_vram_info();
    (StatusCode::OK, Json(info)).into_response()
}

pub(crate) fn collect_vram_info() -> Vec<VramInfo> {
    #[cfg(feature = "gpu")]
    {
        let mut result = Vec::new();
        // Probe device 0 only (single-GPU system).
        if let Some(caps) = crate::gpu::detect() {
            match crate::gpu::VramSession::new(caps.device_id) {
                Ok(sess) => {
                    result.push(VramInfo {
                        device_id: sess.device_id,
                        total_gb: bytes_to_gb(sess.total as usize),
                        free_gb: bytes_to_gb(sess.startup_free as usize),
                        used_gb: bytes_to_gb(sess.already_used as usize),
                        desktop_reserved_gb: bytes_to_gb(sess.desktop_reserved as usize),
                        inference_budget_gb: bytes_to_gb(sess.inference_budget as usize),
                    });
                }
                Err(_) => {}
            }
        }
        result
    }
    #[cfg(not(feature = "gpu"))]
    {
        Vec::new()
    }
}

pub(crate) fn vram_budget_check(total_bytes: u64) -> (bool, f64) {
    #[cfg(feature = "gpu")]
    {
        if let Some(caps) = crate::gpu::detect() {
            if let Ok(sess) = crate::gpu::VramSession::new(caps.device_id) {
                let fits = total_bytes <= sess.inference_budget as u64;
                return (fits, bytes_to_gb(sess.inference_budget as usize));
            }
        }
        (false, 0.0)
    }
    #[cfg(not(feature = "gpu"))]
    {
        (true, 0.0)
    }
}
