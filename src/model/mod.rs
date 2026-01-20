//! Model implementations for ROCmForge
//! Provides minimal transformer model implementations

pub mod config;
pub mod execution_plan;
pub mod glm_position;
pub mod kv_cache;
pub mod simple_transformer;

// Include position embedding tests
#[cfg(test)]
pub mod position_embedding_tests;

// Include config tests (MQA/GQA detection)
#[cfg(test)]
pub mod config_tests;

// Include Phase 5: PagedAttention integration tests
#[cfg(test)]
pub mod phase5_paged_tests;

pub use config::*;
pub use execution_plan::*;
pub use glm_position::*;
pub use simple_transformer::*;

// NOTE: We do NOT re-export kv_cache::* here to avoid confusion with the paged KvCache.
// The simple KVCache (model::kv_cache::KVCache) is legacy and should only be used in tests.
// For production, use crate::kv_cache::KvCache (the paged implementation).
