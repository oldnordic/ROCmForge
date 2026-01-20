//! Attention kernel implementations
//!
//! Provides CPU and GPU kernels for attention operations:
//! - Causal mask generation
//! - Softmax computation
//! - QK^T matrix multiplication for attention scores
//! - Attention-weighted value aggregation
//! - FlashAttention-style fused operations
