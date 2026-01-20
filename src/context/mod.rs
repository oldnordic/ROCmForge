//! Context engine for LLM memory augmentation
//!
//! This module provides semantic context storage and retrieval using SQLiteGraph,
//! enabling extended LLM context windows (20k -> 200k "pseudo-context") through
//! HNSW vector search and graph-based message relationships.
//!
//! # Features
//!
//! - **Graph-based message storage**: Messages stored as graph nodes with metadata
//! - **HNSW vector indexing**: Fast approximate nearest neighbor search
//! - **Conversation threading**: Messages connected via "follows" edges
//! - **Semantic retrieval**: Find relevant messages by vector similarity
//! - **Graph expansion**: Include conversation context with retrieved messages
//!
//! # Usage
//!
//! ```rust,no_run
//! use rocmforge::context::GraphContextStore;
//!
//! // Create in-memory store (use open() for persistent storage)
//! let mut store = GraphContextStore::in_memory(384)?;
//!
//! // Add messages to context
//! let msg1 = store.add_message("Hello world", 0, None)?;
//! let msg2 = store.add_message("How are you?", 1, Some(msg1))?;
//!
//! // Retrieve semantically similar messages
//! let results = store.retrieve_context("greeting", 5)?;
//! ```
//!
//! # Feature Flag
//!
//! This module requires the `context` feature flag:
//! ```bash
//! cargo build --features context
//! ```

#[cfg(feature = "context")]
pub mod graph_context;

#[cfg(feature = "context")]
pub use graph_context::{
    ContextError, ContextMessage, ContextResult, ContextSearchParams,
    ContextSearchResult, DummyEmbedding, EmbeddingModel, GraphContextStore,
};

// Re-export sqlitegraph types when context feature is enabled
#[cfg(feature = "context")]
pub use sqlitegraph;
