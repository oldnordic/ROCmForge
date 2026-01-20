//! HTTP route definitions for ROCmForge server
//!
//! This module sets up the Axum router with all inference server endpoints.
//! It includes CORS middleware and route definitions for:
//! - Text generation (/generate, /generate/stream)
//! - Request management (/status, /cancel)
//! - Model discovery (/models)
//! - Health and readiness checks (/health, /ready)
//! - Observability (/metrics, /traces)

use super::handlers::{
    cancel_handler, generate_handler, generate_stream_handler, health_handler,
    metrics_handler, models_handler, ready_handler, status_handler, traces_handler,
};
use super::server::InferenceServer;
use axum::routing::{get, post};
use axum::Router;
use tower::ServiceBuilder;
use tower_http::cors::{Any, CorsLayer};

/// Create the HTTP router with all inference server endpoints
///
/// This function sets up the complete REST API for ROCmForge with:
/// - CORS support for cross-origin requests
/// - Text generation with and without streaming
/// - Request status and cancellation
/// - Model discovery
/// - Health/ready probes for Kubernetes
/// - Prometheus metrics export
/// - OpenTelemetry traces export
pub fn create_router(server: InferenceServer) -> Router {
    Router::new()
        .route("/generate", post(generate_handler))
        .route("/generate/stream", post(generate_stream_handler))
        .route("/status/:request_id", get(status_handler))
        .route("/cancel/:request_id", post(cancel_handler))
        .route("/models", get(models_handler))
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/metrics", get(metrics_handler))
        .route("/traces", get(traces_handler))
        .layer(ServiceBuilder::new().layer(CorsLayer::new().allow_origin(Any).allow_headers(Any)))
        .with_state(server)
}
