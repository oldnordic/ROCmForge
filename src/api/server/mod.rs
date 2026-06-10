use std::sync::Arc;
use axum::{
    routing::{get, post},
    Router,
};

pub mod handlers;
pub mod inference;
pub mod state;
pub mod utils;
pub mod vram;

pub use state::{ModelEntry, ModelManager};

pub fn create_router(state: Arc<ModelManager>) -> Router {
    Router::new()
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/load", post(handlers::load_model))
        .route("/v1/models/unload", post(handlers::unload_model))
        .route("/v1/models/vram", post(vram::estimate_vram))
        .route("/v1/vram", get(vram::list_vram))
        .route("/v1/completions", post(handlers::create_completion))
        .route("/v1/chat/completions", post(handlers::create_chat_completion))
        .route("/v1/messages", post(handlers::create_messages))
        .route("/health", get(handlers::health))
        .route("/ready", get(handlers::ready))
        .with_state(state)
}
