//! Memory pool management for GPU allocations
//!
//! This module implements memory arena patterns to minimize the number of
//! GPU memory allocations, which is critical for RDNA3 architecture stability.
//!
//! # Background
//!
//! Multiple small GPU allocations (200-300 individual hipMalloc calls) can
//! trigger GPU hangs on AMD Radeon RX 7900 XT and other RDNA3 cards.
//! The solution is to allocate once per model load and subdivide internally.
//!
//! # Pattern
//!
//! This follows the llama.cpp ggml_gallocr pattern:
//! 1. Calculate total memory needed for all tensors
//! 2. Allocate single large buffer (or max 16 chunks for very large models)
//! 3. Subdivide internally using best-fit free block allocation
//! 4. Track allocations and free blocks for reuse

pub mod arena;
pub mod calculator;

pub use arena::ModelWeightArena;
pub use calculator::MemoryCalculator;
