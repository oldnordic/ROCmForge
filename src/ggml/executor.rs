//! Graph executor for ggml IR.

use crate::ggml::{GgmlBackend, GgmlResult, Graph, GraphOptimizer, OptimizerStats};

/// Configuration for graph execution.
#[derive(Debug, Clone, Default)]
pub struct ExecuteConfig {
    /// Enable optimization before execution.
    pub optimize: bool,
    /// Enable deferred synchronization for reduced kernel launch overhead.
    ///
    /// When enabled, synchronization is deferred until the end of graph execution
    /// rather than after each individual kernel launch. This can significantly
    /// reduce latency for graphs with many small operations.
    pub defer_synchronization: bool,
}

impl ExecuteConfig {
    /// Create a new config with optimization enabled.
    pub fn with_optimization() -> Self {
        Self {
            optimize: true,
            defer_synchronization: false,
        }
    }

    /// Create a new config with optimization disabled.
    pub fn without_optimization() -> Self {
        Self {
            optimize: false,
            defer_synchronization: false,
        }
    }

    /// Enable deferred synchronization mode.
    ///
    /// In this mode, the executor will not call `backend.synchronize()` after
    /// each kernel launch. Instead, all kernels are launched asynchronously
    /// and a single synchronization happens at the end.
    ///
    /// This reduces CPU-GPU synchronization overhead and can improve TTFT
    /// (Time To First Token) for inference workloads.
    pub fn with_deferred_sync(mut self) -> Self {
        self.defer_synchronization = true;
        self
    }
}

/// Result of graph execution with optional optimization stats.
#[derive(Debug)]
pub struct ExecuteResult {
    /// Optimization statistics (if optimization was enabled).
    pub optimizer_stats: Option<OptimizerStats>,
    /// Number of nodes executed.
    pub nodes_executed: usize,
}

/// Execute a graph with optional optimization.
///
/// # Arguments
/// - `backend`: The GPU backend to use
/// - `graph`: The graph to execute (will be modified if optimize=true)
/// - `config`: Execution configuration
///
/// # Returns
/// Execution result with optional optimizer statistics
///
/// # Kernel Launch Overhead Optimization
///
/// When `defer_synchronization` is enabled in the config:
/// - Kernels are launched asynchronously without waiting for completion
/// - A single synchronization happens at the end of graph execution
/// - Reduces CPU-GPU synchronization points from N to 1 (where N = number of nodes)
/// - Significantly reduces TTFT for graphs with many small operations
pub fn execute_graph_with_config<B: GgmlBackend>(
    backend: &mut B,
    graph: &mut Graph,
    config: ExecuteConfig,
) -> GgmlResult<ExecuteResult> {
    eprintln!(">>> execute_graph_with_config: ENTRY (defer_sync={})", config.defer_synchronization);
    let mut optimizer_stats = None;

    if config.optimize {
        let optimizer = GraphOptimizer::new();
        optimizer_stats = Some(optimizer.optimize(graph));
    }

    // Allocate buffers for all tensors
    eprintln!(">>> execute_graph_with_config: Allocating buffers for {} tensors", graph.tensors.len());
    for desc in &graph.tensors {
        eprintln!(">>> Checking tensor {:?}: is_view={}, has_buffer={}",
                 desc.id, desc.is_view(), backend.buffer(desc.id).is_some());
        if desc.is_view() {
            eprintln!(">>> SKIP: tensor {:?} is a view", desc.id);
            continue;
        }
        if backend.buffer(desc.id).is_none() {
            eprintln!(">>> execute_graph_with_config: Allocating buffer for tensor {:?} (shape={:?})",
                     desc.id, desc.shape);
            backend.alloc(desc)?;
        } else {
            eprintln!(">>> Tensor {:?} already has buffer", desc.id);
        }
    }
    eprintln!(">>> execute_graph_with_config: Buffer allocation complete");

    // Execute each node
    eprintln!(">>> execute_graph_with_config: Executing {} nodes", graph.nodes.len());
    for (i, node) in graph.nodes.iter().enumerate() {
        eprintln!(">>> execute_graph_with_config: Executing node {} (op={:?})", i, node.op);
        backend.execute_op(&node.op, &node.inputs, &node.outputs)?;

        // Only synchronize per-node if NOT deferring synchronization
        if !config.defer_synchronization {
            backend.synchronize()?;
            eprintln!(">>> execute_graph_with_config: Node {} synchronized", i);
        } else {
            eprintln!(">>> execute_graph_with_config: Node {} launched (async)", i);
        }
    }

    // Single synchronization at the end if deferring
    if config.defer_synchronization {
        eprintln!(">>> execute_graph_with_config: All nodes launched, final synchronization...");
        backend.synchronize()?;
        eprintln!(">>> execute_graph_with_config: Final synchronization complete");
    } else {
        eprintln!(">>> execute_graph_with_config: All nodes executed and synchronized");
    }

    Ok(ExecuteResult {
        optimizer_stats,
        nodes_executed: graph.nodes.len(),
    })
}

/// Execute a graph without optimization (original behavior).
pub fn execute_graph<B: GgmlBackend>(backend: &mut B, graph: &Graph) -> GgmlResult<()> {
    execute_graph_with_config(backend, &mut graph.clone(), ExecuteConfig::default())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // TDD TEST: Optimizer is called when enabled
    #[test]
    fn test_optimizer_called_when_enabled() {
        // Create a simple graph
        let _graph = Graph::new();
        let config = ExecuteConfig::with_optimization();

        // The optimizer should process the graph
        // (We can't test full execution without a GPU backend,
        // but we can verify the config is set correctly)
        assert!(config.optimize, "Optimize should be enabled");
    }

    // TDD TEST: Optimizer is skipped when disabled
    #[test]
    fn test_optimizer_skipped_when_disabled() {
        let config = ExecuteConfig::without_optimization();
        assert!(!config.optimize, "Optimize should be disabled");
    }

    // TDD TEST: Default config does not optimize
    #[test]
    fn test_default_config_no_optimization() {
        let config = ExecuteConfig::default();
        assert!(!config.optimize, "Default should not optimize");
    }

    // TDD TEST: Default config does not defer synchronization
    #[test]
    fn test_default_config_no_deferred_sync() {
        let config = ExecuteConfig::default();
        assert!(!config.defer_synchronization, "Default should not defer sync");
    }

    // TDD TEST: with_deferred_sync enables deferred synchronization
    #[test]
    fn test_with_deferred_sync_enables_flag() {
        let config = ExecuteConfig::default().with_deferred_sync();
        assert!(config.defer_synchronization, "with_deferred_sync should enable flag");
    }

    // TDD TEST: with_optimization preserves defer_synchronization default
    #[test]
    fn test_with_optimization_default_defer_sync() {
        let config = ExecuteConfig::with_optimization();
        assert!(!config.defer_synchronization, "with_optimization should not defer sync by default");
    }
}
