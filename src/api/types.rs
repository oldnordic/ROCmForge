//! OpenAI-compatible API types for HTTP server inference.

use serde::{Deserialize, Serialize};

// ── /v1/models ────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: &'static str,
}

// ── /v1/completions ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<()>,
    pub finish_reason: String,
}

// ── /v1/chat/completions ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

// ── Shared ──────────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionStreamChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ── Defaults ──────────────────────────────────────────────────────────────────────

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    0.9
}

// ── Errors ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ApiError {
    pub fn new(msg: impl Into<String>, typ: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: msg.into(),
                error_type: typ.into(),
                param: None,
                code: None,
            },
        }
    }
}

// ── Anthropic /v1/messages ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<MessageContent>,
    pub usage: MessagesUsage,
}

#[derive(Debug, Serialize)]
pub struct MessageContent {
    #[serde(rename = "type")]
    pub content_type: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct MessagesUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

// ── Model management ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub model: String,
}

#[derive(Debug, Serialize)]
pub struct LoadModelResponse {
    pub id: String,
    pub object: &'static str,
    pub status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct VramInfo {
    pub device_id: i32,
    pub total_gb: f64,
    pub free_gb: f64,
    pub used_gb: f64,
    pub desktop_reserved_gb: f64,
    pub inference_budget_gb: f64,
}

#[derive(Debug, Serialize)]
pub struct VramEstimate {
    pub model_id: String,
    pub model_weights_gb: f64,
    pub kv_cache_gb: f64,
    pub scratch_gb: f64,
    pub total_gb: f64,
    pub fits: bool,
    pub inference_budget_gb: f64,
}
