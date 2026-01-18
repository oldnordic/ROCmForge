//! Time to First Token (TTFT) profiling
//!
//! This module provides detailed breakdown of Time to First Token (TTFT) latency
//! for LLM inference. TTFT is a critical metric for user experience in chat applications,
//! representing the time from request submission to the first generated token appearing.
//!
//! # TTFT Components
//!
//! TTFT is broken down into the following components:
//!
//! - **Model Loading**: Time to load model weights into GPU memory
//! - **Tokenization**: Time to convert text prompt to token IDs
//! - **Embedding Lookup**: Time to look up token embeddings
//! - **Prompt Processing**: Time to process the prompt through all layers (prefill phase)
//! - **First Token Generation**: Time to generate the first output token (sampling + LM head)
//! - **Memory Transfers**: CPU-to-GPU and GPU-to-CPU data transfer time
//!
//! # Example
//!
//! ```rust
//! use rocmforge::profiling::ttft::{TtftProfiler, TtftBreakdown};
//!
//! let mut profiler = TtftProfiler::new();
//!
//! // Start TTFT measurement
//! profiler.start_ttft();
//!
//! // ... model loading ...
//! profiler.start_model_loading();
//! // ... load model ...
//! profiler.stop_model_loading();
//!
//! // ... tokenization ...
//! profiler.start_tokenization();
//! // ... tokenize ...
//! profiler.stop_tokenization();
//!
//! // ... prompt processing ...
//! profiler.start_prompt_processing();
//! // ... process prompt ...
//! profiler.stop_prompt_processing();
//!
//! // ... first token generation ...
//! profiler.start_first_token();
//! // ... generate first token ...
//! profiler.stop_first_token();
//!
//! // Get complete TTFT breakdown
//! let breakdown = profiler.finish_ttft();
//! println!("TTFT: {:.2} ms", breakdown.total_ttft_ms);
//! println!("  Model loading: {:.2} ms", breakdown.model_loading_ms);
//! println!("  Tokenization: {:.2} ms", breakdown.tokenization_ms);
//! println!("  Prompt processing: {:.2} ms", breakdown.prompt_processing_ms);
//! println!("  First token generation: {:.2} ms", breakdown.first_token_ms);
//! ```

use std::time::{Duration, Instant};
use std::fmt;

/// Detailed breakdown of Time to First Token (TTFT) latency
///
/// This struct contains all components that contribute to TTFT,
/// allowing for detailed analysis of where time is spent during
/// the initial inference phase.
#[derive(Debug, Clone, Default)]
pub struct TtftBreakdown {
    /// Total TTFT from request to first token (ms)
    pub total_ttft_ms: f64,

    /// Model loading time (ms) - one-time cost
    pub model_loading_ms: f64,

    /// Tokenization time (ms) - text to token IDs
    pub tokenization_ms: f64,

    /// Embedding lookup time (ms) - token IDs to embeddings
    pub embedding_lookup_ms: f64,

    /// Prompt processing time (ms) - processing all prompt tokens through layers
    pub prompt_processing_ms: f64,

    /// First token generation time (ms) - includes LM head and sampling
    pub first_token_ms: f64,

    /// CPU-to-GPU memory transfer time (ms)
    pub h2d_transfer_ms: f64,

    /// GPU-to-CPU memory transfer time (ms)
    pub d2h_transfer_ms: f64,

    /// Number of prompt tokens processed
    pub prompt_token_count: usize,

    /// Quantization format (if detected)
    pub quantization_format: Option<String>,

    /// Individual kernel timings (if measured)
    pub kernel_timings: Vec<KernelTiming>,
}

/// Timing for an individual kernel execution
#[derive(Debug, Clone)]
pub struct KernelTiming {
    /// Kernel name
    pub name: String,

    /// Execution time (ms)
    pub duration_ms: f64,

    /// Number of elements processed
    pub element_count: usize,
}

impl TtftBreakdown {
    /// Create a new empty TTFT breakdown
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the percentage of TTFT spent in model loading
    pub fn model_loading_pct(&self) -> f64 {
        if self.total_ttft_ms > 0.0 {
            (self.model_loading_ms / self.total_ttft_ms) * 100.0
        } else {
            0.0
        }
    }

    /// Get the percentage of TTFT spent in prompt processing
    pub fn prompt_processing_pct(&self) -> f64 {
        if self.total_ttft_ms > 0.0 {
            (self.prompt_processing_ms / self.total_ttft_ms) * 100.0
        } else {
            0.0
        }
    }

    /// Get the percentage of TTFT spent in memory transfers
    pub fn memory_transfer_pct(&self) -> f64 {
        if self.total_ttft_ms > 0.0 {
            ((self.h2d_transfer_ms + self.d2h_transfer_ms) / self.total_ttft_ms) * 100.0
        } else {
            0.0
        }
    }

    /// Get the prompt processing time per token (ms/token)
    pub fn prompt_processing_per_token_ms(&self) -> f64 {
        if self.prompt_token_count > 0 {
            self.prompt_processing_ms / self.prompt_token_count as f64
        } else {
            0.0
        }
    }

    /// Identify the dominant TTFT component
    pub fn dominant_component(&self) -> &'static str {
        let mut max_time = 0.0f64;
        let mut component = "unknown";

        if self.model_loading_ms > max_time {
            max_time = self.model_loading_ms;
            component = "model_loading";
        }
        if self.prompt_processing_ms > max_time {
            max_time = self.prompt_processing_ms;
            component = "prompt_processing";
        }
        if self.first_token_ms > max_time {
            max_time = self.first_token_ms;
            component = "first_token";
        }
        if self.h2d_transfer_ms > max_time {
            max_time = self.h2d_transfer_ms;
            component = "h2d_transfer";
        }
        if self.d2h_transfer_ms > max_time {
            max_time = self.d2h_transfer_ms;
            component = "d2h_transfer";
        }

        component
    }

    /// Check if TTFT meets the target (<200ms for 512 tokens)
    pub fn meets_target(&self) -> bool {
        self.total_ttft_ms < 200.0
    }

    /// Get gap from target (ms) - negative if under target
    pub fn target_gap_ms(&self) -> f64 {
        self.total_ttft_ms - 200.0
    }

    /// Format the breakdown as a table
    pub fn format_table(&self) -> String {
        let mut output = String::new();
        output.push_str("┌─ Time to First Token (TTFT) Breakdown ───────────────────┐\n");
        output.push_str(&format!("│ Total TTFT:           {:>8.2} ms (target: <200ms) │\n", self.total_ttft_ms));

        let target_status = if self.meets_target() { "✓ PASS" } else { "✗ FAIL" };
        output.push_str(&format!("│ Target Status:         {:>8}                  │\n", target_status));

        output.push_str(&format!("│                                                      │\n"));
        output.push_str(&format!("│ Model Loading:         {:>8.2} ms ({:>5.1}%)     │\n",
            self.model_loading_ms, self.model_loading_pct()));
        output.push_str(&format!("│ Tokenization:          {:>8.2} ms                 │\n",
            self.tokenization_ms));
        output.push_str(&format!("│ Embedding Lookup:      {:>8.2} ms                 │\n",
            self.embedding_lookup_ms));
        output.push_str(&format!("│ Prompt Processing:     {:>8.2} ms ({:>5.1}%)     │\n",
            self.prompt_processing_ms, self.prompt_processing_pct()));
        output.push_str(&format!("│   Per Token:           {:>8.2} ms/token          │\n",
            self.prompt_processing_per_token_ms()));
        output.push_str(&format!("│ First Token Gen:       {:>8.2} ms ({:>5.1}%)     │\n",
            self.first_token_ms,
            if self.total_ttft_ms > 0.0 { (self.first_token_ms / self.total_ttft_ms) * 100.0 } else { 0.0 }));
        output.push_str(&format!("│ H2D Transfer:          {:>8.2} ms                 │\n",
            self.h2d_transfer_ms));
        output.push_str(&format!("│ D2H Transfer:          {:>8.2} ms                 │\n",
            self.d2h_transfer_ms));
        output.push_str(&format!("│ Memory Transfers:      {:>8.2} ms ({:>5.1}%)     │\n",
            self.h2d_transfer_ms + self.d2h_transfer_ms, self.memory_transfer_pct()));

        if let Some(format) = &self.quantization_format {
            output.push_str(&format!("│                                                      │\n"));
            output.push_str(&format!("│ Quantization:          {:>8}                     │\n",
                format));
        }

        output.push_str(&format!("│                                                      │\n"));
        output.push_str(&format!("│ Prompt Tokens:         {:>8} tokens               │\n",
            self.prompt_token_count));
        output.push_str(&format!("│ Dominant Component:    {:>8}                     │\n",
            self.dominant_component().replace('_', " ")));

        if !self.kernel_timings.is_empty() {
            output.push_str("│                                                      │\n");
            output.push_str("│ Top Kernel Timings:                                   │\n");
            // Sort by duration and show top 5
            let mut sorted = self.kernel_timings.clone();
            sorted.sort_by(|a, b| b.duration_ms.partial_cmp(&a.duration_ms).unwrap());
            for (i, timing) in sorted.iter().take(5).enumerate() {
                output.push_str(&format!("│   {}. {:<18} {:>7.2} ms        │\n",
                    i + 1, timing.name, timing.duration_ms));
            }
        }

        output.push_str("└──────────────────────────────────────────────────────┘\n");
        output
    }

    /// Create a summary for optimization recommendations
    pub fn optimization_summary(&self) -> String {
        let mut recommendations = Vec::new();

        // Analyze dominant component
        match self.dominant_component() {
            "model_loading" => {
                recommendations.push("Model loading dominates TTFT. Consider:".to_string());
                recommendations.push("  - Keep model resident in memory for multiple requests".to_string());
                recommendations.push("  - Use faster storage (NVMe vs SSD)".to_string());
                recommendations.push("  - Implement model weight caching".to_string());
            }
            "prompt_processing" => {
                recommendations.push("Prompt processing dominates TTFT. Consider:".to_string());
                if self.prompt_processing_per_token_ms() > 1.0 {
                    recommendations.push("  - Optimize attention kernels for batch processing".to_string());
                    recommendations.push("  - Implement flash attention for long prompts".to_string());
                }
                recommendations.push("  - Consider operator fusion (dequant+matmul)".to_string());
                recommendations.push("  - Profile memory bandwidth utilization".to_string());
            }
            "first_token" => {
                recommendations.push("First token generation dominates TTFT. Consider:".to_string());
                recommendations.push("  - Optimize LM head computation".to_string());
                recommendations.push("  - Profile and optimize sampling logic".to_string());
            }
            "h2d_transfer" | "d2h_transfer" => {
                recommendations.push("Memory transfers dominate TTFT. Consider:".to_string());
                recommendations.push("  - Use pinned memory for faster transfers".to_string());
                recommendations.push("  - Overlap transfers with computation".to_string());
                recommendations.push("  - Reduce CPU-GPU synchronization points".to_string());
            }
            _ => {}
        }

        // Memory transfer analysis
        if self.memory_transfer_pct() > 30.0 {
            recommendations.push(
                format!("High memory transfer overhead ({:.1}%). Consider reducing data movement.",
                    self.memory_transfer_pct())
            );
        }

        // Prompt processing efficiency
        if self.prompt_token_count >= 512 && self.prompt_processing_ms > 150.0 {
            recommendations.push(
                format!("Prompt processing slow for {} tokens ({:.2} ms). Target <200ms total TTFT.",
                    self.prompt_token_count, self.prompt_processing_ms)
            );
        }

        if recommendations.is_empty() {
            recommendations.push("TTFT looks reasonable. Continue profiling for optimization opportunities.".to_string());
        }

        recommendations.join("\n")
    }
}

impl fmt::Display for TtftBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_table())
    }
}

/// Profiler for measuring Time to First Token (TTFT)
///
/// `TtftProfiler` provides a simple API for measuring all components of TTFT.
/// It uses CPU-side timing for portability, but can be extended with GPU
/// event timing via the `KernelTimer` module.
///
/// # Example
///
/// ```rust
/// use rocmforge::profiling::ttft::TtftProfiler;
///
/// let mut profiler = TtftProfiler::new();
/// profiler.start_ttft();
///
/// // Measure model loading
/// profiler.start_model_loading();
/// // ... load model ...
/// profiler.stop_model_loading();
///
/// // Measure tokenization
/// profiler.start_tokenization();
/// // ... tokenize ...
/// profiler.stop_tokenization();
///
/// // ... other phases ...
///
/// let breakdown = profiler.finish_ttft();
/// println!("{}", breakdown);
/// ```
pub struct TtftProfiler {
    /// Overall TTFT start time
    ttft_start: Option<Instant>,

    /// Model loading phase
    model_loading_start: Option<Instant>,
    model_loading_duration: Duration,

    /// Tokenization phase
    tokenization_start: Option<Instant>,
    tokenization_duration: Duration,

    /// Embedding lookup phase
    embedding_lookup_start: Option<Instant>,
    embedding_lookup_duration: Duration,

    /// Prompt processing phase
    prompt_processing_start: Option<Instant>,
    prompt_processing_duration: Duration,

    /// First token generation phase
    first_token_start: Option<Instant>,
    first_token_duration: Duration,

    /// CPU-to-GPU transfers
    h2d_transfer_start: Option<Instant>,
    h2d_transfer_duration: Duration,

    /// GPU-to-CPU transfers
    d2h_transfer_start: Option<Instant>,
    d2h_transfer_duration: Duration,

    /// Number of prompt tokens
    prompt_token_count: usize,

    /// Quantization format
    quantization_format: Option<String>,

    /// Individual kernel timings
    kernel_timings: Vec<KernelTiming>,
}

impl TtftProfiler {
    /// Create a new TTFT profiler
    pub fn new() -> Self {
        Self {
            ttft_start: None,
            model_loading_start: None,
            model_loading_duration: Duration::ZERO,
            tokenization_start: None,
            tokenization_duration: Duration::ZERO,
            embedding_lookup_start: None,
            embedding_lookup_duration: Duration::ZERO,
            prompt_processing_start: None,
            prompt_processing_duration: Duration::ZERO,
            first_token_start: None,
            first_token_duration: Duration::ZERO,
            h2d_transfer_start: None,
            h2d_transfer_duration: Duration::ZERO,
            d2h_transfer_start: None,
            d2h_transfer_duration: Duration::ZERO,
            prompt_token_count: 0,
            quantization_format: None,
            kernel_timings: Vec::new(),
        }
    }

    /// Start overall TTFT measurement
    pub fn start_ttft(&mut self) {
        self.ttft_start = Some(Instant::now());
    }

    /// Finish TTFT measurement and return breakdown
    pub fn finish_ttft(&self) -> TtftBreakdown {
        let total_ttft = if let Some(start) = self.ttft_start {
            start.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };

        TtftBreakdown {
            total_ttft_ms: total_ttft,
            model_loading_ms: self.model_loading_duration.as_secs_f64() * 1000.0,
            tokenization_ms: self.tokenization_duration.as_secs_f64() * 1000.0,
            embedding_lookup_ms: self.embedding_lookup_duration.as_secs_f64() * 1000.0,
            prompt_processing_ms: self.prompt_processing_duration.as_secs_f64() * 1000.0,
            first_token_ms: self.first_token_duration.as_secs_f64() * 1000.0,
            h2d_transfer_ms: self.h2d_transfer_duration.as_secs_f64() * 1000.0,
            d2h_transfer_ms: self.d2h_transfer_duration.as_secs_f64() * 1000.0,
            prompt_token_count: self.prompt_token_count,
            quantization_format: self.quantization_format.clone(),
            kernel_timings: self.kernel_timings.clone(),
        }
    }

    /// Start model loading phase
    pub fn start_model_loading(&mut self) {
        self.model_loading_start = Some(Instant::now());
    }

    /// Stop model loading phase
    pub fn stop_model_loading(&mut self) {
        if let Some(start) = self.model_loading_start {
            self.model_loading_duration = start.elapsed();
            self.model_loading_start = None;
        }
    }

    /// Start tokenization phase
    pub fn start_tokenization(&mut self) {
        self.tokenization_start = Some(Instant::now());
    }

    /// Stop tokenization phase
    pub fn stop_tokenization(&mut self) {
        if let Some(start) = self.tokenization_start {
            self.tokenization_duration = start.elapsed();
            self.tokenization_start = None;
        }
    }

    /// Start embedding lookup phase
    pub fn start_embedding_lookup(&mut self) {
        self.embedding_lookup_start = Some(Instant::now());
    }

    /// Stop embedding lookup phase
    pub fn stop_embedding_lookup(&mut self) {
        if let Some(start) = self.embedding_lookup_start {
            self.embedding_lookup_duration = start.elapsed();
            self.embedding_lookup_start = None;
        }
    }

    /// Start prompt processing phase
    pub fn start_prompt_processing(&mut self) {
        self.prompt_processing_start = Some(Instant::now());
    }

    /// Stop prompt processing phase
    pub fn stop_prompt_processing(&mut self) {
        if let Some(start) = self.prompt_processing_start {
            self.prompt_processing_duration = start.elapsed();
            self.prompt_processing_start = None;
        }
    }

    /// Start first token generation phase
    pub fn start_first_token(&mut self) {
        self.first_token_start = Some(Instant::now());
    }

    /// Stop first token generation phase
    pub fn stop_first_token(&mut self) {
        if let Some(start) = self.first_token_start {
            self.first_token_duration = start.elapsed();
            self.first_token_start = None;
        }
    }

    /// Start CPU-to-GPU transfer phase
    pub fn start_h2d_transfer(&mut self) {
        self.h2d_transfer_start = Some(Instant::now());
    }

    /// Stop CPU-to-GPU transfer phase
    pub fn stop_h2d_transfer(&mut self) {
        if let Some(start) = self.h2d_transfer_start {
            self.h2d_transfer_duration += start.elapsed();
            self.h2d_transfer_start = None;
        }
    }

    /// Start GPU-to-CPU transfer phase
    pub fn start_d2h_transfer(&mut self) {
        self.d2h_transfer_start = Some(Instant::now());
    }

    /// Stop GPU-to-CPU transfer phase
    pub fn stop_d2h_transfer(&mut self) {
        if let Some(start) = self.d2h_transfer_start {
            self.d2h_transfer_duration += start.elapsed();
            self.d2h_transfer_start = None;
        }
    }

    /// Set the number of prompt tokens
    pub fn set_prompt_token_count(&mut self, count: usize) {
        self.prompt_token_count = count;
    }

    /// Set the quantization format
    pub fn set_quantization_format(&mut self, format: impl Into<String>) {
        self.quantization_format = Some(format.into());
    }

    /// Record a kernel execution timing
    pub fn record_kernel(&mut self, name: impl Into<String>, duration_ms: f64, element_count: usize) {
        self.kernel_timings.push(KernelTiming {
            name: name.into(),
            duration_ms,
            element_count,
        });
    }

    /// Record a kernel execution timing from a KernelTimer
    #[cfg(feature = "rocm")]
    pub fn record_kernel_from_timer(&mut self, name: impl Into<String>, timer: &crate::profiling::KernelTimer, element_count: usize) {
        if let Some(duration) = timer.elapsed() {
            self.record_kernel(name, duration as f64, element_count);
        }
    }
}

impl Default for TtftProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a TTFT breakdown from component durations
///
/// This is a convenience function for creating TTFT breakdowns
/// when you have timing data from other sources.
pub fn create_ttft_breakdown(
    total_ttft_ms: f64,
    model_loading_ms: f64,
    tokenization_ms: f64,
    embedding_lookup_ms: f64,
    prompt_processing_ms: f64,
    first_token_ms: f64,
    h2d_transfer_ms: f64,
    d2h_transfer_ms: f64,
    prompt_token_count: usize,
) -> TtftBreakdown {
    TtftBreakdown {
        total_ttft_ms,
        model_loading_ms,
        tokenization_ms,
        embedding_lookup_ms,
        prompt_processing_ms,
        first_token_ms,
        h2d_transfer_ms,
        d2h_transfer_ms,
        prompt_token_count,
        quantization_format: None,
        kernel_timings: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ttft_breakdown_new() {
        let breakdown = TtftBreakdown::new();
        assert_eq!(breakdown.total_ttft_ms, 0.0);
        assert_eq!(breakdown.prompt_token_count, 0);
    }

    #[test]
    fn test_ttft_breakdown_percentages() {
        let mut breakdown = TtftBreakdown::new();
        breakdown.total_ttft_ms = 100.0;
        breakdown.model_loading_ms = 20.0;
        breakdown.prompt_processing_ms = 50.0;
        breakdown.h2d_transfer_ms = 10.0;
        breakdown.d2h_transfer_ms = 5.0;

        assert_eq!(breakdown.model_loading_pct(), 20.0);
        assert_eq!(breakdown.prompt_processing_pct(), 50.0);
        assert_eq!(breakdown.memory_transfer_pct(), 15.0);
    }

    #[test]
    fn test_ttft_breakdown_per_token() {
        let mut breakdown = TtftBreakdown::new();
        breakdown.prompt_processing_ms = 100.0;
        breakdown.prompt_token_count = 50;

        assert_eq!(breakdown.prompt_processing_per_token_ms(), 2.0);
    }

    #[test]
    fn test_ttft_breakdown_dominant_component() {
        let mut breakdown = TtftBreakdown::new();
        breakdown.prompt_processing_ms = 100.0;
        breakdown.first_token_ms = 20.0;

        assert_eq!(breakdown.dominant_component(), "prompt_processing");

        breakdown.first_token_ms = 150.0;
        assert_eq!(breakdown.dominant_component(), "first_token");
    }

    #[test]
    fn test_ttft_breakdown_target() {
        let mut breakdown = TtftBreakdown::new();

        breakdown.total_ttft_ms = 150.0;
        assert!(breakdown.meets_target());
        assert_eq!(breakdown.target_gap_ms(), -50.0);

        breakdown.total_ttft_ms = 250.0;
        assert!(!breakdown.meets_target());
        assert_eq!(breakdown.target_gap_ms(), 50.0);
    }

    #[test]
    fn test_ttft_profiler_new() {
        let profiler = TtftProfiler::new();
        assert!(profiler.ttft_start.is_none());
    }

    #[test]
    fn test_ttft_profiler_phases() {
        let mut profiler = TtftProfiler::new();
        profiler.start_ttft();

        // Model loading
        profiler.start_model_loading();
        std::thread::sleep(Duration::from_millis(10));
        profiler.stop_model_loading();

        // Tokenization
        profiler.start_tokenization();
        std::thread::sleep(Duration::from_millis(5));
        profiler.stop_tokenization();

        // Prompt processing
        profiler.start_prompt_processing();
        std::thread::sleep(Duration::from_millis(15));
        profiler.stop_prompt_processing();

        profiler.set_prompt_token_count(128);

        let breakdown = profiler.finish_ttft();

        assert!(breakdown.model_loading_ms >= 10.0);
        assert!(breakdown.tokenization_ms >= 5.0);
        assert!(breakdown.prompt_processing_ms >= 15.0);
        assert_eq!(breakdown.prompt_token_count, 128);
    }

    #[test]
    fn test_ttft_profiler_record_kernel() {
        let mut profiler = TtftProfiler::new();
        profiler.record_kernel("matmul", 5.0, 1000);
        profiler.record_kernel("attention", 10.0, 2000);

        let breakdown = profiler.finish_ttft();
        assert_eq!(breakdown.kernel_timings.len(), 2);
        assert_eq!(breakdown.kernel_timings[0].name, "matmul");
        assert_eq!(breakdown.kernel_timings[1].name, "attention");
    }

    #[test]
    fn test_create_ttft_breakdown() {
        let breakdown = create_ttft_breakdown(
            200.0,  // total
            50.0,   // model loading
            5.0,    // tokenization
            10.0,   // embedding lookup
            100.0,  // prompt processing
            20.0,   // first token
            5.0,    // h2d transfer
            5.0,    // d2h transfer
            512,    // prompt tokens
        );

        assert_eq!(breakdown.total_ttft_ms, 200.0);
        assert_eq!(breakdown.prompt_processing_ms, 100.0);
        assert_eq!(breakdown.prompt_token_count, 512);
        // Target is strictly less than 200ms, so exactly 200ms does NOT meet target
        assert!(!breakdown.meets_target());
        assert_eq!(breakdown.target_gap_ms(), 0.0);
    }

    #[test]
    fn test_optimization_summary() {
        let mut breakdown = TtftBreakdown::new();
        breakdown.total_ttft_ms = 100.0;
        breakdown.prompt_processing_ms = 60.0;

        let summary = breakdown.optimization_summary();
        assert!(summary.contains("Prompt processing dominates TTFT"));
    }
}
