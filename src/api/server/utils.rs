use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::json;

pub(crate) fn error_response(code: StatusCode, message: &str) -> Response {
    (code, Json(json!({ "error": message }))).into_response()
}

pub(crate) fn bytes_to_gb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

pub(crate) fn format_chat_messages(messages: &[crate::api::types::ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(&format!("<|{}|>\n{}\n", m.role, m.content));
    }
    out
}
