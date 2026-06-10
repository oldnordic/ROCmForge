use std::sync::Arc;
use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, sse::Sse, IntoResponse, Json, Response},
};
use serde_json::json;
use tokio::sync::Mutex;
use futures::stream::Stream;
use std::convert::Infallible;

use crate::api::types::*;
use super::state::ModelManager;
use super::utils::{error_response, format_chat_messages};
use super::inference::{run_sync_inference, run_stream_inference};

pub(crate) async fn list_models(State(state): State<Arc<ModelManager>>) -> impl IntoResponse {
    let now = chrono::Utc::now().timestamp();
    let data: Vec<_> = state
        .keys()
        .await
        .into_iter()
        .map(|id| ModelInfo {
            id,
            object: "model",
            created: now,
            owned_by: "rocmforge",
        })
        .collect();
    (
        StatusCode::OK,
        Json(ModelList {
            object: "list",
            data,
        }),
    )
}

pub(crate) async fn load_model(
    State(state): State<Arc<ModelManager>>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    let path = req.model;
    if !std::path::Path::new(&path).exists() {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!("Model path does not exist: {}", path),
            "invalid_path",
        );
    }

    match state.try_load(&path, req.draft_model.as_deref()).await {
        Ok(_) => (StatusCode::OK, Json(json!({ "status": "loaded" }))).into_response(),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e, "load_failed"),
    }
}

pub(crate) async fn unload_model(
    State(state): State<Arc<ModelManager>>,
    Json(req): Json<UnloadModelRequest>,
) -> Response {
    state.unload(&req.model).await;
    (StatusCode::OK, Json(json!({ "status": "unloaded" }))).into_response()
}

pub(crate) async fn create_completion(
    State(state): State<Arc<ModelManager>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let entry = match state.get(&req.model).await {
        Some(e) => e,
        None => return error_response(StatusCode::BAD_REQUEST, "Model not loaded", "model_not_found"),
    };

    let prompt = req.prompt.clone();
    let prompt_tokens = entry.tokenizer.encode(&prompt, true, false);
    let prompt_tokens_len = prompt_tokens.len();

    if req.stream.unwrap_or(false) {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let entry_clone = entry.clone();
        let req_clone = req.clone();

        tokio::spawn(async move {
            let _permit = entry_clone.inference_sem.acquire().await.expect("semaphore closed");
            let res = tokio::task::spawn_blocking(move || {
                run_stream_inference(
                    &entry_clone.cpu_weights,
                    #[cfg(feature = "gpu")] &entry_clone.gpu_weights,
                    #[cfg(feature = "gpu")] &entry_clone.speculative_engine,
                    &entry_clone.model_path,
                    &entry_clone.config,
                    &entry_clone.tokenizer,
                    &prompt_tokens,
                    req_clone.max_tokens.unwrap_or(128),
                    req_clone.temperature.unwrap_or(0.7),
                    req_clone.top_p.unwrap_or(0.9),
                    tx,
                )
            }).await;

            if let Err(e) = res {
                eprintln!("Inference task panicked: {:?}", e);
            }
        });

        let stream = async_stream::stream! {
            while let Some(text) = rx.recv().await {
                yield Ok::<Event, Infallible>(Event::default().data(json!({
                    "id": "cmpl-",
                    "object": "text_completion.chunk",
                    "created": chrono::Utc::now().timestamp(),
                    "model": req.model,
                    "choices": [{
                        "text": text,
                        "index": 0,
                        "finish_reason": null
                    }]
                }).to_string()));
            }
            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
        };

        Sse::new(stream).into_response()
    } else {
        let _permit = entry.inference_sem.acquire().await.expect("semaphore closed");
        let entry_clone = entry.clone();
        let res = tokio::task::spawn_blocking(move || {
            run_sync_inference(
                &entry_clone.cpu_weights,
                #[cfg(feature = "gpu")] &entry_clone.gpu_weights,
                #[cfg(feature = "gpu")] &entry_clone.speculative_engine,
                &entry_clone.model_path,
                &entry_clone.config,
                &entry_clone.tokenizer,
                &prompt_tokens,
                req.max_tokens.unwrap_or(128),
                req.temperature.unwrap_or(0.7),
                req.top_p.unwrap_or(0.9),
            )
        }).await.expect("semaphore closed");

        match res {
            Ok((generated, completion_tokens)) => {
                let response = CompletionResponse {
                    id: "cmpl-".to_string(),
                    object: "text_completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: req.model.clone(),
                    choices: vec![CompletionChoice {
                        text: generated,
                        index: 0,
                        finish_reason: "stop".to_string(),
                    }],
                    usage: CompletionUsage {
                        prompt_tokens: prompt_tokens_len,
                        completion_tokens,
                        total_tokens: prompt_tokens_len + completion_tokens,
                    },
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e, "inference_failed"),
        }
    }
}

pub(crate) async fn create_chat_completion(
    State(state): State<Arc<ModelManager>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let entry = match state.get(&req.model).await {
        Some(e) => e,
        None => return error_response(StatusCode::BAD_REQUEST, "Model not loaded", "model_not_found"),
    };

    let prompt = entry.chat_template.apply(&req.messages).unwrap_or_else(|_| format_chat_messages(&req.messages));
    let prompt_tokens = entry.tokenizer.encode(&prompt, true, false);
    let prompt_tokens_len = prompt_tokens.len();

    if req.stream.unwrap_or(false) {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let entry_clone = entry.clone();
        let req_clone = req.clone();

        tokio::spawn(async move {
            let _permit = entry_clone.inference_sem.acquire().await.expect("semaphore closed");
            let res = tokio::task::spawn_blocking(move || {
                run_stream_inference(
                    &entry_clone.cpu_weights,
                    #[cfg(feature = "gpu")] &entry_clone.gpu_weights,
                    #[cfg(feature = "gpu")] &entry_clone.speculative_engine,
                    &entry_clone.model_path,
                    &entry_clone.config,
                    &entry_clone.tokenizer,
                    &prompt_tokens,
                    req_clone.max_tokens.unwrap_or(128),
                    req_clone.temperature.unwrap_or(0.7),
                    req_clone.top_p.unwrap_or(0.9),
                    tx,
                )
            }).await;

            if let Err(e) = res {
                eprintln!("Inference task panicked: {:?}", e);
            }
        });

        let stream = async_stream::stream! {
            while let Some(text) = rx.recv().await {
                yield Ok::<Event, Infallible>(Event::default().data(json!({
                    "id": "chatcmpl-",
                    "object": "chat.completion.chunk",
                    "created": chrono::Utc::now().timestamp(),
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "delta": { "content": text },
                        "finish_reason": null
                    }]
                }).to_string()));
            }
            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
        };

        Sse::new(stream).into_response()
    } else {
        let _permit = entry.inference_sem.acquire().await.expect("semaphore closed");
        let entry_clone = entry.clone();
        let res = tokio::task::spawn_blocking(move || {
            run_sync_inference(
                &entry_clone.cpu_weights,
                #[cfg(feature = "gpu")] &entry_clone.gpu_weights,
                #[cfg(feature = "gpu")] &entry_clone.speculative_engine,
                &entry_clone.model_path,
                &entry_clone.config,
                &entry_clone.tokenizer,
                &prompt_tokens,
                req.max_tokens.unwrap_or(128),
                req.temperature.unwrap_or(0.7),
                req.top_p.unwrap_or(0.9),
            )
        }).await.expect("semaphore closed");

        match res {
            Ok((generated, completion_tokens)) => {
                let response = ChatCompletionResponse {
                    id: "chatcmpl-".to_string(),
                    object: "chat.completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: req.model.clone(),
                    choices: vec![ChatCompletionChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: generated,
                        },
                        finish_reason: "stop".to_string(),
                    }],
                    usage: CompletionUsage {
                        prompt_tokens: prompt_tokens_len,
                        completion_tokens,
                        total_tokens: prompt_tokens_len + completion_tokens,
                    },
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e, "inference_failed"),
        }
    }
}

pub(crate) async fn create_messages(
    State(state): State<Arc<ModelManager>>,
    Json(req): Json<crate::api::types::MessagesRequest>,
) -> Response {
    let entry = match state.get(&req.model).await {
        Some(e) => e,
        None => return error_response(StatusCode::BAD_REQUEST, "Model not loaded", "model_not_found"),
    };

    let prompt = entry.chat_template.apply(&req.messages).unwrap_or_else(|_| format_chat_messages(&req.messages));
    let prompt_tokens = entry.tokenizer.encode(&prompt, true, false);
    let prompt_tokens_len = prompt_tokens.len();

    let _permit = entry.inference_sem.acquire().await.expect("semaphore closed");
    let entry_clone = entry.clone();
    let res = tokio::task::spawn_blocking(move || {
        run_sync_inference(
            &entry_clone.cpu_weights,
            #[cfg(feature = "gpu")] &entry_clone.gpu_weights,
            #[cfg(feature = "gpu")] &entry_clone.speculative_engine,
            &entry_clone.model_path,
            &entry_clone.config,
            &entry_clone.tokenizer,
            &prompt_tokens,
            req.max_tokens.unwrap_or(128),
            req.temperature.unwrap_or(0.7),
            req.top_p.unwrap_or(0.9),
        )
    }).await.expect("semaphore closed");

    match res {
        Ok((generated, completion_tokens)) => {
            let response = crate::api::types::MessagesResponse {
                id: "msg-".to_string(),
                model: req.model.clone(),
                role: "assistant".to_string(),
                content: vec![crate::api::types::MessagesContent {
                    content_type: "text".to_string(),
                    text: generated,
                }],
                usage: crate::api::types::MessagesUsage {
                    input_tokens: prompt_tokens_len,
                    output_tokens: completion_tokens,
                },
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e, "inference_failed"),
    }
}

pub(crate) async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({"status": "ok"})))
}

pub(crate) async fn ready(State(state): State<Arc<ModelManager>>) -> impl IntoResponse {
    let ready = !state.keys().await.is_empty();
    (
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(json!({"status": if ready { "ready" } else { "no_models_loaded" }})),
    )
}
