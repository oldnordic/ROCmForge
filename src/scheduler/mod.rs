//! Scheduler module for continuous batching
//!
//! This module implements a continuous batching scheduler for efficient GPU utilization.
//! It manages request lifecycles through three queues (pending, processing, completed)
//! and supports both static batching and dynamic continuous batching.

// Module declarations
pub mod scheduler;
pub mod types;
pub mod queue;
pub mod batch;

// Re-export types for backward compatibility
pub use scheduler::Scheduler;
pub use types::{SchedulerError, SchedulerResult, RequestState, GenerationRequest};
pub use queue::QueueStats;
pub use batch::{Batch, IterationBatch, SchedulerConfig};
