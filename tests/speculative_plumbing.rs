#![cfg(all(feature = "gpu", feature = "server"))]
#![allow(warnings)]

//! E2E integration test for server-side speculative decoding.

mod common;

use axum::{
    body::Body,
    http::{self, Request, StatusCode},
};
use rocmforge::api::server::{create_router, ModelEntry, ModelManager};
use serial_test::serial;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower::util::ServiceExt;

#[tokio::test]
#[serial]
async fn test_server_speculative_loading_and_inference() {
    require_gpu!();

    let model_path = "/home/feanor/Projects/rocmforge/llama3.2-1b-instruct-q4_0.rfm";
    let init_model_path = "/home/feanor/Projects/rocmforge/llama3.2_svd_smoke.rfm";

    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(init_model_path).exists()
    {
        eprintln!("Skipping speculative server integration test: Required models not found.");
        return;
    }

    // 1. Initialize the server with a baseline model (no draft)
    let entry = Arc::new(
        ModelEntry::load(init_model_path, None)
            .expect("Failed to load initial baseline ModelEntry"),
    );

    let manager = ModelManager::new();
    manager
        .try_load_entry(entry)
        .await
        .expect("Failed to register entry");
    let state = Arc::new(manager);
    let app = create_router(state);

    // 2. Load the speculative target + draft model dynamically via POST /v1/models/load
    let load_req_body = serde_json::json!({
        "model": model_path,
        "draft_model": model_path
    });

    let request = Request::builder()
        .method(http::Method::POST)
        .uri("/v1/models/load")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&load_req_body).unwrap()))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body_bytes = axum::body::to_bytes(response.into_body(), 1000000)
        .await
        .unwrap();
    let res_json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(status, StatusCode::OK, "Load model failed: {:?}", res_json);
    println!("Load model response: {:?}", res_json);
    assert_eq!(res_json["status"], "loaded");

    // 3. Test completions endpoint with speculative model (sync)
    let completion_req_body = serde_json::json!({
        "model": model_path,
        "prompt": "Say hello",
        "max_tokens": 10,
        "temperature": 0.0,
        "top_p": 0.9,
        "stream": false
    });

    let request = Request::builder()
        .method(http::Method::POST)
        .uri("/v1/completions")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::to_vec(&completion_req_body).unwrap(),
        ))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body_bytes = axum::body::to_bytes(response.into_body(), 1000000)
        .await
        .unwrap();
    let res_json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "Sync completions failed: {:?}",
        res_json
    );
    println!("Sync Completions response: {:?}", res_json);
    let text = res_json["choices"][0]["text"].as_str().unwrap();
    assert!(!text.is_empty());

    // 4. Test chat completions endpoint with speculative model (sync)
    let chat_req_body = serde_json::json!({
        "model": model_path,
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 10,
        "temperature": 0.0,
        "top_p": 0.9,
        "stream": false
    });

    let request = Request::builder()
        .method(http::Method::POST)
        .uri("/v1/chat/completions")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&chat_req_body).unwrap()))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = axum::body::to_bytes(response.into_body(), 1000000)
        .await
        .unwrap();
    let res_json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    println!("Sync Chat completions response: {:?}", res_json);
    let text = res_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap();
    assert!(!text.is_empty());

    // 5. Test chat completions endpoint with speculative model (stream)
    let chat_stream_req_body = serde_json::json!({
        "model": model_path,
        "messages": [
            {"role": "user", "content": "Give me a one-word answer."}
        ],
        "max_tokens": 10,
        "temperature": 0.0,
        "top_p": 0.9,
        "stream": true
    });

    let request = Request::builder()
        .method(http::Method::POST)
        .uri("/v1/chat/completions")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::to_vec(&chat_stream_req_body).unwrap(),
        ))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Parse the SSE stream
    let body_bytes = axum::body::to_bytes(response.into_body(), 1000000)
        .await
        .unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();

    let mut chunk_count = 0;
    for line in body_str.lines() {
        if line.starts_with("data: ") {
            let data_val = &line["data: ".len()..];
            if data_val == "[DONE]" {
                break;
            }
            let chunk_json: serde_json::Value = serde_json::from_str(data_val).unwrap();
            let content = chunk_json["choices"][0]["delta"]["content"]
                .as_str()
                .unwrap_or("");
            println!("Stream chunk: {:?}", content);
            chunk_count += 1;
        }
    }

    assert!(chunk_count > 0, "No streaming chunks were received!");
    println!("Stream test succeeded with {} chunks", chunk_count);
}
