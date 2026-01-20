//! HTTP handlers for context search endpoint
//!
//! This module provides HTTP endpoints for semantic context search,
//! allowing external clients to query the context store via REST API.
//!
//! # Important Note
//!
//! Due to HNSW index not being Send+Sync (contains `Rc<RefCell<>>` internally),
//! the HTTP context endpoint is not currently usable with Axum's State extractor.
//!
//! The CLI commands provide full context management functionality:
//! - `rocmforge context add <text>` - Add messages
//! - `rocmforge context search <query>` - Semantic search
//! - `rocmforge context list` - List all messages
//! - `rocmforge context clear` - Clear context
//!
//! For HTTP API access, a future implementation would require either:
//! 1. Upstream changes to sqlitegraph to make HNSW Send+Sync
//! 2. A separate single-threaded HTTP service for context
//! 3. Using a thread-safe vector index alternative

use serde::{Deserialize, Serialize};

/// Context search query parameters (for reference only)
#[derive(Debug, Deserialize)]
pub struct ContextQueryParams {
    /// Query text to search for
    pub q: String,
    /// Maximum number of results (default: 5)
    #[serde(default = "default_k")]
    pub k: usize,
    /// Whether to expand results to include conversation context
    #[serde(default)]
    pub expand: bool,
}

fn default_k() -> usize {
    5
}

/// HTTP response wrapper for context search results (for reference only)
#[derive(Debug, Serialize)]
pub struct ContextSearchResponse {
    /// Query that was executed
    pub query: String,
    /// Retrieved messages ordered by similarity
    pub messages: Vec<ContextMessageResponse>,
    /// Total messages in the store
    pub total_messages: usize,
    /// Number of results returned
    pub result_count: usize,
}

/// A context message in HTTP response format (for reference only)
#[derive(Debug, Serialize)]
pub struct ContextMessageResponse {
    /// Message ID
    pub id: u64,
    /// Message text
    pub text: String,
    /// Sequence ID in conversation
    pub seq_id: usize,
    /// Timestamp (Unix seconds)
    pub timestamp: i64,
    /// Similarity score (lower = more similar, since it's distance)
    pub similarity: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_k() {
        assert_eq!(default_k(), 5);
    }

    #[test]
    fn test_context_query_params_default() {
        let params = ContextQueryParams {
            q: "test".to_string(),
            k: 5,
            expand: false,
        };

        assert_eq!(params.q, "test");
        assert_eq!(params.k, 5);
        assert!(!params.expand);
    }

    #[test]
    fn test_context_message_response_serialization() {
        let msg = ContextMessageResponse {
            id: 1,
            text: "Hello world".to_string(),
            seq_id: 0,
            timestamp: 1234567890,
            similarity: Some(0.95),
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("Hello world"));
    }
}
