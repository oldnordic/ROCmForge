//! Phase 29: Validation & E2E Tests
//!
//! This integration test module validates that all v1.5 fixes work together:
//! - Phase 25: Env var embedding (HSACO paths compile-time)
//! - Phase 26: Transpose kernel fix (block dims 32x32x1)
//! - Phase 27: Device property caching (launch validation)
//! - Phase 28: Debug hygiene (error messages)
//!
//! # Test Organization
//!
//! Tests are organized into validation submodules:
//! - `cache`: Model download/caching utilities
//! - `transpose`: Transpose kernel validation
//! - `model_loading`: E2E model loading tests
//! - `token_generation`: Single token generation tests

mod validation;

// Re-export for use in other tests
pub use validation::*;

// Note: Actual tests are in the submodules
// This file serves as the test target entry point
