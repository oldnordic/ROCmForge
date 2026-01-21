//! Validation tests for Phase 29 - E2E verification
//!
//! This module validates that all v1.5 fixes work together:
//! - Phase 25: Env var embedding (HSACO paths compile-time)
//! - Phase 26: Transpose kernel fix (block dims 32x32x1)
//! - Phase 27: Device property caching (launch validation)
//! - Phase 28: Debug hygiene (error messages)

mod cache;
mod transpose;
mod model_loading;
mod token_generation;

pub use cache::*;
