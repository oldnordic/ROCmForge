//! Hybrid Scheduler Integration Tests
//!
//! Tests for the hybrid execution scheduler covering:
//! - Backend selection logic
//! - Telemetry recording and reporting
//! - Cost model accuracy
//! - Error handling
//!
//! Run with: cargo test hybrid_scheduler

use rocmforge::ggml::hybrid_scheduler::{
    HybridScheduler, ExecutionStrategy, SelectionReason,
    ExecutionEvent, BackendExecutionSummary,
};
use rocmforge::ggml::hybrid_scheduler::OpType;
use std::time::Instant;

#[test]
fn test_scheduler_creation() {
    let scheduler = HybridScheduler::new(ExecutionStrategy::Automatic);
    let stats = scheduler.backend_usage_stats();
    assert_eq!(stats.total_operations, 0);
}

#[test]
fn test_telemetry_recording() {
    let mut scheduler = HybridScheduler::new(ExecutionStrategy::GpuPreferred);

    let event = ExecutionEvent {
        timestamp: Instant::now(),
        operation: OpType::MatMul,
        backend: "cpu".to_string(),
        reason: SelectionReason::CpuFallback,
        actual_duration_us: Some(100),
    };

    scheduler.record_execution(event);

    let telemetry = scheduler.get_telemetry();
    assert_eq!(telemetry.len(), 1);
    assert_eq!(telemetry[0].backend, "cpu");
}

#[test]
fn test_clear_telemetry() {
    let mut scheduler = HybridScheduler::new(ExecutionStrategy::CpuPreferred);

    let event = ExecutionEvent {
        timestamp: Instant::now(),
        operation: OpType::MatMul,
        backend: "cpu".to_string(),
        reason: SelectionReason::CpuFallback,
        actual_duration_us: Some(100),
    };

    scheduler.record_execution(event);
    assert_eq!(scheduler.get_telemetry().len(), 1);

    scheduler.clear_telemetry();
    assert_eq!(scheduler.get_telemetry().len(), 0);
}

#[test]
fn test_backend_stats() {
    let mut scheduler = HybridScheduler::new(ExecutionStrategy::Automatic);

    // Record some events
    for i in 0..3 {
        scheduler.record_execution(ExecutionEvent {
            timestamp: Instant::now(),
            operation: OpType::MatMul,
            backend: "gpu".to_string(),
            reason: SelectionReason::GpuAvailable,
            actual_duration_us: Some(100 + i * 10),
        });
    }

    for _ in 0..2 {
        scheduler.record_execution(ExecutionEvent {
            timestamp: Instant::now(),
            operation: OpType::Softmax,
            backend: "cpu".to_string(),
            reason: SelectionReason::CpuFallback,
            actual_duration_us: Some(50),
        });
    }

    let stats = scheduler.backend_usage_stats();
    assert_eq!(stats.total_operations, 5);
    assert_eq!(stats.gpu_operations, 3);
    assert_eq!(stats.cpu_operations, 2);
}

#[test]
fn test_execution_summary() {
    let mut scheduler = HybridScheduler::new(ExecutionStrategy::Automatic);

    // Add some test events
    scheduler.record_execution(ExecutionEvent {
        timestamp: Instant::now(),
        operation: OpType::MatMul,
        backend: "gpu".to_string(),
        reason: SelectionReason::GpuAvailable,
        actual_duration_us: Some(100),
    });

    scheduler.record_execution(ExecutionEvent {
        timestamp: Instant::now(),
        operation: OpType::Softmax,
        backend: "cpu".to_string(),
        reason: SelectionReason::CpuFallback,
        actual_duration_us: Some(50),
    });

    let summary = scheduler.execution_summary();
    assert_eq!(summary.total_operations, 2);
    assert_eq!(summary.gpu_operations, 1);
    assert_eq!(summary.cpu_operations, 1);
    assert_eq!(summary.total_time_us, 150);
    assert_eq!(summary.gpu_time_us, 100);
    assert_eq!(summary.cpu_time_us, 50);
}

#[test]
fn test_print_debug_summary() {
    let mut scheduler = HybridScheduler::new(ExecutionStrategy::Automatic);

    scheduler.record_execution(ExecutionEvent {
        timestamp: Instant::now(),
        operation: OpType::MatMul,
        backend: "gpu".to_string(),
        reason: SelectionReason::GpuAvailable,
        actual_duration_us: Some(1000),
    });

    // Should not panic
    scheduler.print_debug_summary();
}

#[test]
fn test_operations_by_type() {
    let mut scheduler = HybridScheduler::new(ExecutionStrategy::Automatic);

    // Add mixed events
    scheduler.record_execution(ExecutionEvent {
        timestamp: Instant::now(),
        operation: OpType::MatMul,
        backend: "gpu".to_string(),
        reason: SelectionReason::GpuAvailable,
        actual_duration_us: Some(100),
    });

    scheduler.record_execution(ExecutionEvent {
        timestamp: Instant::now(),
        operation: OpType::Softmax,
        backend: "cpu".to_string(),
        reason: SelectionReason::CpuFallback,
        actual_duration_us: Some(50),
    });

    scheduler.record_execution(ExecutionEvent {
        timestamp: Instant::now(),
        operation: OpType::MatMul,
        backend: "cpu".to_string(),
        reason: SelectionReason::CpuFallback,
        actual_duration_us: Some(200),
    });

    // Query MatMul operations
    let matmul_ops = scheduler.operations_by_type(OpType::MatMul);
    assert_eq!(matmul_ops.len(), 2);

    // Query Softmax operations
    let softmax_ops = scheduler.operations_by_type(OpType::Softmax);
    assert_eq!(softmax_ops.len(), 1);

    // Query operations that don't exist
    let attention_ops = scheduler.operations_by_type(OpType::Attention);
    assert_eq!(attention_ops.len(), 0);
}

#[test]
fn test_execution_summary_empty() {
    let scheduler = HybridScheduler::new(ExecutionStrategy::Automatic);
    let summary = scheduler.execution_summary();

    assert_eq!(summary.total_operations, 0);
    assert_eq!(summary.gpu_operations, 0);
    assert_eq!(summary.cpu_operations, 0);
    assert_eq!(summary.total_time_us, 0);
    assert_eq!(summary.gpu_time_us, 0);
    assert_eq!(summary.cpu_time_us, 0);
}

#[test]
fn test_execution_summary_with_missing_durations() {
    let mut scheduler = HybridScheduler::new(ExecutionStrategy::Automatic);

    // Event with duration
    scheduler.record_execution(ExecutionEvent {
        timestamp: Instant::now(),
        operation: OpType::MatMul,
        backend: "gpu".to_string(),
        reason: SelectionReason::GpuAvailable,
        actual_duration_us: Some(100),
    });

    // Event without duration
    scheduler.record_execution(ExecutionEvent {
        timestamp: Instant::now(),
        operation: OpType::Softmax,
        backend: "cpu".to_string(),
        reason: SelectionReason::CpuFallback,
        actual_duration_us: None,
    });

    let summary = scheduler.execution_summary();
    assert_eq!(summary.total_operations, 2);
    assert_eq!(summary.total_time_us, 100); // Only counts events with duration
}
