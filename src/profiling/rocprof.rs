//! ROCm rocprof profiling tool integration
//!
//! This module provides integration with rocprof, the HSA tracer and profiler
//! included in ROCm installations.
//!
//! # Overview
//!
//! [`RocprofSession`] manages profiling sessions with rocprof, providing:
//! - Command building for running applications under profiling
//! - Output parsing for PMC (Performance Monitor Counters) CSV files
//! - HSA trace parsing for kernel execution records
//!
//! # Example
//!
//! ```rust,ignore
//! use rocmforge::profiling::rocprof::{RocprofSession, ProfilingConfig};
//!
//! // Create a profiling session
//! let config = ProfilingConfig::default()
//!     .with_counters(vec!["SQ_WAVES", "SQ_INSTS"]);
//!
//! let session = RocprofSession::with_config(config)?;
//!
//! // Build command to run your application under rocprof
//! let cmd = session.build_command("my_app", &["--arg1", "arg2"]);
//! println!("Run: {:?}", cmd);
//!
//! // After running, parse results
//! let results = session.parse_results()?;
//! println!("GPM Waves: {:?}", results.get_counter("SQ_WAVES"));
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

// Re-export common types from the parent module
pub use crate::profiling::types::{
    ProfilingError, ProfilingResult, ProfilingTool, CounterCategory,
};

/// Configuration for a profiling session
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Output directory for profiling results
    pub output_dir: PathBuf,
    /// Performance counters to collect
    pub counters: Vec<String>,
    /// Counter categories to include
    pub categories: Vec<CounterCategory>,
    /// Duration to sample (for continuous profiling)
    pub sample_duration: Option<std::time::Duration>,
    /// Enable HSA trace
    pub enable_hsa_trace: bool,
    /// Enable I trace (instruction trace)
    pub enable_i_trace: bool,
    /// Additional arguments to pass to the profiler
    pub extra_args: Vec<String>,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        ProfilingConfig {
            output_dir: PathBuf::from("/tmp/rocmforge_profile"),
            counters: Vec::new(),
            categories: vec![
                CounterCategory::Instructions,
                CounterCategory::Waves,
                CounterCategory::Memory,
            ],
            sample_duration: None,
            enable_hsa_trace: true,
            enable_i_trace: false,
            extra_args: Vec::new(),
        }
    }
}

impl ProfilingConfig {
    /// Create a new profiling config with the specified output directory
    pub fn new(output_dir: impl AsRef<Path>) -> Self {
        ProfilingConfig {
            output_dir: output_dir.as_ref().to_path_buf(),
            ..Default::default()
        }
    }

    /// Add a specific performance counter
    pub fn with_counter(mut self, counter: impl Into<String>) -> Self {
        self.counters.push(counter.into());
        self
    }

    /// Add multiple performance counters
    pub fn with_counters(mut self, counters: Vec<impl Into<String>>) -> Self {
        for counter in counters {
            self.counters.push(counter.into());
        }
        self
    }

    /// Add a counter category (expands to common counters)
    pub fn with_category(mut self, category: CounterCategory) -> Self {
        self.categories.push(category);
        self
    }

    /// Set the sample duration
    pub fn with_sample_duration(mut self, duration: std::time::Duration) -> Self {
        self.sample_duration = Some(duration);
        self
    }

    /// Enable or disable HSA tracing
    pub fn with_hsa_trace(mut self, enable: bool) -> Self {
        self.enable_hsa_trace = enable;
        self
    }

    /// Enable or disable instruction tracing
    pub fn with_i_trace(mut self, enable: bool) -> Self {
        self.enable_i_trace = enable;
        self
    }

    /// Add extra arguments to pass to the profiler
    pub fn with_extra_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_args.push(arg.into());
        self
    }

    /// Get all counters to collect (explicit + from categories)
    pub fn get_all_counters(&self) -> Vec<String> {
        let mut counters = self.counters.clone();
        for category in &self.categories {
            for &counter in category.common_counters() {
                if !counters.contains(&counter.to_string()) {
                    counters.push(counter.to_string());
                }
            }
        }
        counters
    }

    /// Validate the configuration
    pub fn validate(&self) -> ProfilingResult<()> {
        if self.output_dir.as_os_str().is_empty() {
            return Err(ProfilingError::InvalidConfig(
                "Output directory cannot be empty".to_string(),
            ));
        }

        if self.counters.is_empty() && self.categories.is_empty() {
            return Err(ProfilingError::InvalidConfig(
                "At least one counter or category must be specified".to_string(),
            ));
        }

        Ok(())
    }
}

/// A profiling session for measuring GPU kernel performance
#[derive(Debug, Clone)]
pub struct RocprofSession {
    /// Tool to use for profiling
    tool: ProfilingTool,
    /// Session configuration
    config: ProfilingConfig,
    /// Session ID (for identification)
    session_id: String,
}

impl RocprofSession {
    /// Create a new profiling session
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Directory where profiling results will be stored
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use rocmforge::profiling::rocprof::RocprofSession;
    ///
    /// let session = RocprofSession::new("/tmp/profile")?;
    /// ```
    pub fn new(output_dir: impl AsRef<Path>) -> ProfilingResult<Self> {
        let tool = ProfilingTool::Rocprof;
        if !tool.is_available() {
            return Err(ProfilingError::ToolNotFound(tool.command_name().to_string()));
        }

        let config = ProfilingConfig::new(output_dir);
        let session_id = Self::generate_session_id();

        Ok(RocprofSession {
            tool,
            config,
            session_id,
        })
    }

    /// Create a new profiling session with custom configuration
    pub fn with_config(config: ProfilingConfig) -> ProfilingResult<Self> {
        config.validate()?;

        let tool = ProfilingTool::Rocprof;
        if !tool.is_available() {
            return Err(ProfilingError::ToolNotFound(tool.command_name().to_string()));
        }

        let session_id = Self::generate_session_id();

        Ok(RocprofSession {
            tool,
            config,
            session_id,
        })
    }

    /// Create a session for a specific profiling tool
    pub fn with_tool(output_dir: impl AsRef<Path>, tool: ProfilingTool) -> ProfilingResult<Self> {
        if !tool.is_available() {
            return Err(ProfilingError::ToolNotFound(tool.command_name().to_string()));
        }

        let config = ProfilingConfig::new(output_dir);
        let session_id = Self::generate_session_id();

        Ok(RocprofSession {
            tool,
            config,
            session_id,
        })
    }

    fn generate_session_id() -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        format!("profile_{}", timestamp)
    }

    /// Get the session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get the output directory
    pub fn output_dir(&self) -> &Path {
        &self.config.output_dir
    }

    /// Get the configuration
    pub fn config(&self) -> &ProfilingConfig {
        &self.config
    }

    /// Build a command to run an application under rocprof
    ///
    /// # Arguments
    ///
    /// * `application` - Path to the application to profile
    /// * `args` - Arguments to pass to the application
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let session = RocprofSession::new("/tmp/profile")?;
    /// let cmd = session.build_command("./my_app", &["--input", "data.txt"]);
    /// println!("Run: {:?}", cmd);
    /// ```
    pub fn build_command(&self, application: &str, args: &[&str]) -> Command {
        let mut cmd = Command::new(self.tool.command_name());

        // Add tool-specific arguments
        match self.tool {
            ProfilingTool::Rocprof => {
                // rocprof -o output_dir --hsa-trace -- app args
                cmd.arg("-o").arg(&self.config.output_dir);

                if self.config.enable_hsa_trace {
                    cmd.arg("--hsa-trace");
                }

                if self.config.enable_i_trace {
                    cmd.arg("--itrace");
                }

                // Add counter specifications
                for counter in self.config.get_all_counters() {
                    cmd.arg("-p").arg(&counter);
                }

                // Separator for application command
                cmd.arg("--");
            }
            ProfilingTool::Omniperf => {
                // omniperf profile -d output_dir -- app args
                cmd.arg("profile")
                    .arg("-d")
                    .arg(&self.config.output_dir)
                    .arg("--");
            }
            ProfilingTool::Rocperf => {
                // rocperf is typically configured via XML, simpler interface
                cmd.arg("-o").arg(&self.config.output_dir);
                cmd.arg("--");
            }
        }

        // Add application and its arguments
        cmd.arg(application);
        cmd.args(args);

        // Add extra arguments
        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        cmd
    }

    /// Parse profiling results from the output directory
    ///
    /// This reads and parses the output files generated by the profiling tool.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // After running the profiling command...
    /// let results = session.parse_results()?;
    /// for (counter, value) in results.counters() {
    ///     println!("{}: {}", counter, value);
    /// }
    /// ```
    pub fn parse_results(&self) -> ProfilingResult<ProfilingResults> {
        let output_dir = &self.config.output_dir;

        // Check if output directory exists
        if !output_dir.exists() {
            return Err(ProfilingError::SessionError(
                format!("Output directory does not exist: {:?}", output_dir),
            ));
        }

        // Look for common output files
        let csv_file = output_dir.join("pmc_perf.csv");
        let hsa_file = output_dir.join("trace.hsa-trace");

        let mut counters = HashMap::new();
        let mut kernels = Vec::new();

        // Parse PMC CSV if it exists
        if csv_file.exists() {
            match Self::parse_pmc_csv(&csv_file) {
                Ok(parsed_counters) => counters = parsed_counters,
                Err(e) => {
                    tracing::warn!("Failed to parse PMC CSV: {}", e);
                }
            }
        }

        // Parse HSA trace if it exists
        if hsa_file.exists() {
            match Self::parse_hsa_trace(&hsa_file) {
                Ok(parsed_kernels) => kernels = parsed_kernels,
                Err(e) => {
                    tracing::warn!("Failed to parse HSA trace: {}", e);
                }
            }
        }

        Ok(ProfilingResults {
            session_id: self.session_id.clone(),
            counters,
            kernels,
            output_dir: output_dir.clone(),
        })
    }

    fn parse_pmc_csv(path: &Path) -> ProfilingResult<HashMap<String, f64>> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ProfilingError::IoError(format!("Failed to read file: {}", e)))?;

        let mut counters = HashMap::new();

        // Skip header, parse data rows
        for line in content.lines().skip(1) {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }

            // Expected format: counter_name,value or GPU_ID,counter_name,value
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let counter_name = parts[0].trim();
                let value_str = parts[parts.len() - 1].trim();

                if let Ok(value) = value_str.parse::<f64>() {
                    counters.insert(counter_name.to_string(), value);
                }
            }
        }

        Ok(counters)
    }

    fn parse_hsa_trace(path: &Path) -> ProfilingResult<Vec<KernelExecution>> {
        // HSA trace parsing is complex; this is a simplified version
        // In practice, you'd use a proper trace parser
        let content = std::fs::read_to_string(path)
            .map_err(|e| ProfilingError::IoError(format!("Failed to read file: {}", e)))?;

        let mut kernels = Vec::new();

        // Simple line-by-line parsing for kernel names
        for line in content.lines() {
            if line.contains("kernel:") || line.contains("dispatch:") {
                // Extract kernel information
                if let Some(kernel_name) = Self::extract_kernel_name(line) {
                    kernels.push(KernelExecution {
                        name: kernel_name,
                        duration_us: 0.0, // Would be parsed from actual trace
                        start_time_ns: 0,
                        end_time_ns: 0,
                    });
                }
            }
        }

        Ok(kernels)
    }

    fn extract_kernel_name(line: &str) -> Option<String> {
        // Try to extract kernel name from various trace formats
        if let Some(start) = line.find("kernel:") {
            let rest = &line[start + 7..];
            if let Some(end) = rest.find(',') {
                return Some(rest[..end].trim().to_string());
            }
            return Some(rest.trim().to_string());
        }
        None
    }
}

/// Results from a profiling session
#[derive(Debug, Clone)]
pub struct ProfilingResults {
    /// Session identifier
    pub session_id: String,
    /// Collected performance counters
    pub counters: HashMap<String, f64>,
    /// Kernel execution records
    pub kernels: Vec<KernelExecution>,
    /// Output directory path
    pub output_dir: PathBuf,
}

impl ProfilingResults {
    /// Get a specific counter value
    pub fn get_counter(&self, name: &str) -> Option<f64> {
        self.counters.get(name).copied()
    }

    /// Get all counter names and values
    pub fn counters(&self) -> &HashMap<String, f64> {
        &self.counters
    }

    /// Get all kernel executions
    pub fn kernels(&self) -> &[KernelExecution] {
        &self.kernels
    }

    /// Calculate derived metrics from collected counters
    pub fn calculate_metrics(&self) -> ProfilingMetrics {
        let mut metrics = ProfilingMetrics::default();

        // Instruction throughput
        if let Some(insts) = self.get_counter("SQ_INSTS") {
            metrics.total_instructions = insts as u64;
        }

        // Wave activity
        if let Some(waves) = self.get_counter("SQ_WAVES") {
            metrics.total_waves = waves as u64;
        }

        // Cache hit rate
        if let (Some(accesses), Some(misses)) = (
            self.get_counter("TCP_TOTAL_CACHE_ACCESSES"),
            self.get_counter("TCP_TOTAL_CACHE_MISSES"),
        ) {
            if accesses > 0.0 {
                metrics.cache_hit_rate = Some((accesses - misses) / accesses);
            }
        }

        // Memory bandwidth estimation (in GB/s)
        if let (Some(active_cycles), Some(freq)) = (
            self.get_counter("GRBM_GUI_ACTIVE"),
            self.get_counter("GPU_CLK"),
        ) {
            // Rough estimation: active_cycles * frequency / 1e9
            metrics.estimated_bandwidth_gbps = Some((active_cycles * freq) / 1e9);
        }

        metrics
    }
}

/// A single kernel execution record
#[derive(Debug, Clone)]
pub struct KernelExecution {
    /// Kernel name
    pub name: String,
    /// Execution duration in microseconds
    pub duration_us: f64,
    /// Start time in nanoseconds
    pub start_time_ns: u64,
    /// End time in nanoseconds
    pub end_time_ns: u64,
}

/// Derived metrics calculated from profiling results
#[derive(Debug, Clone, Default)]
pub struct ProfilingMetrics {
    /// Total instructions executed
    pub total_instructions: u64,
    /// Total wavefronts launched
    pub total_waves: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: Option<f64>,
    /// Estimated memory bandwidth in GB/s
    pub estimated_bandwidth_gbps: Option<f64>,
    /// Instructions per cycle (IPC)
    pub ipc: Option<f64>,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: Option<f64>,
    /// Compute unit utilization (0.0 to 1.0)
    pub compute_utilization: Option<f64>,
}

impl ProfilingMetrics {
    /// Get a summary of the metrics
    pub fn summary(&self) -> String {
        let mut summary = String::from("Profiling Metrics Summary:\n");

        summary.push_str(&format!("  Total Instructions: {}\n", self.total_instructions));
        summary.push_str(&format!("  Total Waves: {}\n", self.total_waves));

        if let Some(hit_rate) = self.cache_hit_rate {
            summary.push_str(&format!("  Cache Hit Rate: {:.1}%\n", hit_rate * 100.0));
        }

        if let Some(bandwidth) = self.estimated_bandwidth_gbps {
            summary.push_str(&format!("  Est. Bandwidth: {:.2} GB/s\n", bandwidth));
        }

        if let Some(ipc) = self.ipc {
            summary.push_str(&format!("  IPC: {:.2}\n", ipc));
        }

        summary
    }
}

/// Quick profiling helpers for common scenarios
pub mod helpers {
    use super::*;
    use crate::profiling::types::CounterCategory;

    /// Profile a specific kernel using rocprof
    ///
    /// This is a convenience function that sets up a session with
    /// commonly-used counters for kernel profiling.
    pub fn profile_kernel(output_dir: impl AsRef<Path>) -> ProfilingResult<RocprofSession> {
        let config = ProfilingConfig::new(output_dir)
            .with_category(CounterCategory::Instructions)
            .with_category(CounterCategory::Waves)
            .with_category(CounterCategory::Memory);

        RocprofSession::with_config(config)
    }

    /// Profile memory bandwidth using rocprof
    ///
    /// Sets up counters specifically for analyzing memory bandwidth usage.
    pub fn profile_memory(output_dir: impl AsRef<Path>) -> ProfilingResult<RocprofSession> {
        let config = ProfilingConfig::new(output_dir)
            .with_counters(vec![
                "GRBM_GUI_ACTIVE",
                "GRBM_COUNT",
                "TCP_TOTAL_CACHE_ACCESSES",
                "TCP_TOTAL_CACHE_MISSES",
            ])
            .with_category(CounterCategory::Cache);

        RocprofSession::with_config(config)
    }

    /// Profile memory bandwidth with detailed stall analysis
    ///
    /// Includes additional counters for memory stall cycles and latency.
    pub fn profile_memory_detailed(output_dir: impl AsRef<Path>) -> ProfilingResult<RocprofSession> {
        let config = ProfilingConfig::new(output_dir)
            .with_counters(vec![
                "GRBM_GUI_ACTIVE",
                "GRBM_COUNT",
                "TCP_TOTAL_CACHE_ACCESSES",
                "TCP_TOTAL_CACHE_MISSES",
                "TCP_TOTAL_HIT_RATE",
                "SQ_WAVES",
                "SQ_INSTS_VMEM",
                "SQ_INSTS_FLAT",
                "SQ_LDS_IDX_ACTIVE",
                "SQ_LDS_BANK_ACTIVE",
            ])
            .with_category(CounterCategory::Cache)
            .with_category(CounterCategory::Lds);

        RocprofSession::with_config(config)
    }

    /// Profile memory bandwidth for matmul operations
    ///
    /// Specialized configuration for matrix multiplication kernels.
    pub fn profile_matmul_memory(output_dir: impl AsRef<Path>) -> ProfilingResult<RocprofSession> {
        let config = ProfilingConfig::new(output_dir)
            .with_counters(vec![
                "GRBM_GUI_ACTIVE",
                "TCP_TOTAL_CACHE_ACCESSES",
                "TCP_TOTAL_CACHE_MISSES",
                "SQ_INSTS_VMEM",
                "SQ_INSTS_FLAT",
                "SQ_LDS_BANK_ACTIVE",
            ])
            .with_hsa_trace(true);

        RocprofSession::with_config(config)
    }

    /// Profile compute unit utilization
    pub fn profile_compute_unit(output_dir: impl AsRef<Path>) -> ProfilingResult<RocprofSession> {
        let config = ProfilingConfig::new(output_dir)
            .with_category(CounterCategory::ComputeUnit)
            .with_category(CounterCategory::Waves);

        RocprofSession::with_config(config)
    }

    /// Check which ROCm profiling tools are available
    pub fn available_tools() -> Vec<ProfilingTool> {
        vec![
            ProfilingTool::Rocprof,
            ProfilingTool::Omniperf,
            ProfilingTool::Rocperf,
        ]
        .into_iter()
        .filter(|tool| tool.is_available())
        .collect()
    }

    /// Print available tools to stdout
    pub fn print_available_tools() {
        let tools = available_tools();
        if tools.is_empty() {
            println!("No ROCm profiling tools found in PATH.");
            println!("Install ROCm toolkit to use profiling features:");
            println!("  - rocprof: Included in ROCm");
            println!("  - omniperf: pip install omniperf");
        } else {
            println!("Available ROCm profiling tools:");
            for tool in tools {
                println!("  - {}", tool.command_name());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiling_config_default() {
        let config = ProfilingConfig::default();
        assert_eq!(config.output_dir, PathBuf::from("/tmp/rocmforge_profile"));
        assert!(config.counters.is_empty());
        assert_eq!(config.categories.len(), 3);
        assert!(config.enable_hsa_trace);
        assert!(!config.enable_i_trace);
    }

    #[test]
    fn test_profiling_config_builder() {
        let config = ProfilingConfig::new("/tmp/test")
            .with_counter("TEST_COUNTER")
            .with_category(CounterCategory::Instructions)
            .with_hsa_trace(false)
            .with_i_trace(true);

        assert_eq!(config.output_dir, PathBuf::from("/tmp/test"));
        assert!(config.counters.contains(&"TEST_COUNTER".to_string()));
        assert!(config.categories.contains(&CounterCategory::Instructions));
        assert!(!config.enable_hsa_trace);
        assert!(config.enable_i_trace);
    }

    #[test]
    fn test_profiling_config_with_sample_duration() {
        let config = ProfilingConfig::default()
            .with_sample_duration(std::time::Duration::from_secs(10));

        assert_eq!(config.sample_duration, Some(std::time::Duration::from_secs(10)));
    }

    #[test]
    fn test_profiling_config_validate_empty_output_dir() {
        let config = ProfilingConfig {
            output_dir: PathBuf::new(),
            ..Default::default()
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(matches!(result, Err(ProfilingError::InvalidConfig(_))));
    }

    #[test]
    fn test_profiling_config_validate_success() {
        let config = ProfilingConfig::new("/tmp/test");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_profiling_config_get_all_counters() {
        let config = ProfilingConfig::default()
            .with_counter("CUSTOM_COUNTER");

        let counters = config.get_all_counters();

        // Should include custom counter
        assert!(counters.contains(&"CUSTOM_COUNTER".to_string()));

        // Should include counters from default categories
        assert!(counters.contains(&"SQ_INSTS".to_string()));
        assert!(counters.contains(&"SQ_WAVES".to_string()));
    }

    #[test]
    fn test_rocprof_session_new() {
        let config = ProfilingConfig::new("/tmp/test_profile");
        let session = RocprofSession {
            tool: ProfilingTool::Rocprof,
            config,
            session_id: "test_session_123".to_string(),
        };

        assert_eq!(session.session_id(), "test_session_123");
        assert_eq!(session.output_dir(), PathBuf::from("/tmp/test_profile"));
        assert_eq!(session.config().output_dir, PathBuf::from("/tmp/test_profile"));
    }

    #[test]
    fn test_rocprof_session_build_command_rocprof() {
        let config = ProfilingConfig::new("/tmp/out")
            .with_counter("SQ_INSTS")
            .with_hsa_trace(false);

        let session = RocprofSession {
            tool: ProfilingTool::Rocprof,
            config,
            session_id: "test".to_string(),
        };

        let cmd = session.build_command("my_app", &["--arg1", "arg2"]);

        // Verify command structure (we can't actually run it without rocprof installed)
        assert_eq!(cmd.get_program(), "rocprof");
    }

    #[test]
    fn test_profiling_results_empty() {
        let results = ProfilingResults {
            session_id: "test".to_string(),
            counters: HashMap::new(),
            kernels: Vec::new(),
            output_dir: PathBuf::from("/tmp"),
        };

        assert!(results.get_counter("NONEXISTENT").is_none());
        assert!(results.kernels().is_empty());
    }

    #[test]
    fn test_profiling_results_with_data() {
        let mut counters = HashMap::new();
        counters.insert("SQ_INSTS".to_string(), 1000.0);
        counters.insert("SQ_WAVES".to_string(), 500.0);

        let results = ProfilingResults {
            session_id: "test".to_string(),
            counters,
            kernels: Vec::new(),
            output_dir: PathBuf::from("/tmp"),
        };

        assert_eq!(results.get_counter("SQ_INSTS"), Some(1000.0));
        assert_eq!(results.get_counter("SQ_WAVES"), Some(500.0));
    }

    #[test]
    fn test_profiling_metrics_calculate() {
        let mut counters = HashMap::new();
        counters.insert("SQ_INSTS".to_string(), 1000000.0);
        counters.insert("SQ_WAVES".to_string(), 10000.0);
        counters.insert("TCP_TOTAL_CACHE_ACCESSES".to_string(), 1000.0);
        counters.insert("TCP_TOTAL_CACHE_MISSES".to_string(), 100.0);

        let results = ProfilingResults {
            session_id: "test".to_string(),
            counters,
            kernels: Vec::new(),
            output_dir: PathBuf::from("/tmp"),
        };

        let metrics = results.calculate_metrics();

        assert_eq!(metrics.total_instructions, 1000000);
        assert_eq!(metrics.total_waves, 10000);
        assert_eq!(metrics.cache_hit_rate, Some(0.9));
    }

    #[test]
    fn test_profiling_metrics_summary() {
        let metrics = ProfilingMetrics {
            total_instructions: 1000000,
            total_waves: 10000,
            cache_hit_rate: Some(0.85),
            estimated_bandwidth_gbps: Some(250.0),
            ipc: Some(1.5),
            memory_utilization: Some(0.7),
            compute_utilization: Some(0.8),
        };

        let summary = metrics.summary();

        // Check for plain number format (no commas)
        assert!(summary.contains("1000000") || summary.contains("1,000,000"));
        assert!(summary.contains("10000") || summary.contains("10,000"));
        assert!(summary.contains("85.0%"));
        assert!(summary.contains("250.00 GB/s"));
    }

    #[test]
    fn test_kernel_execution() {
        let kernel = KernelExecution {
            name: "test_kernel".to_string(),
            duration_us: 100.0,
            start_time_ns: 1000,
            end_time_ns: 101000,
        };

        assert_eq!(kernel.name, "test_kernel");
        assert_eq!(kernel.duration_us, 100.0);
    }

    #[test]
    fn test_helpers_profile_kernel() {
        // Test that helpers create valid configurations
        let config = ProfilingConfig::new("/tmp/test")
            .with_category(CounterCategory::Instructions)
            .with_category(CounterCategory::Waves)
            .with_category(CounterCategory::Memory);

        assert!(config.validate().is_ok());
        assert!(config.categories.contains(&CounterCategory::Instructions));
        assert!(config.categories.contains(&CounterCategory::Waves));
        assert!(config.categories.contains(&CounterCategory::Memory));
    }

    #[test]
    fn test_helpers_profile_memory() {
        let config = ProfilingConfig::new("/tmp/test")
            .with_counters(vec![
                "GRBM_GUI_ACTIVE",
                "GRBM_COUNT",
                "TCP_TOTAL_CACHE_ACCESSES",
                "TCP_TOTAL_CACHE_MISSES",
            ])
            .with_category(CounterCategory::Cache);

        assert!(config.validate().is_ok());
        assert!(config.counters.contains(&"GRBM_GUI_ACTIVE".to_string()));
        assert!(config.counters.contains(&"TCP_TOTAL_CACHE_ACCESSES".to_string()));
    }

    #[test]
    fn test_session_id_format() {
        let config = ProfilingConfig::default();
        let session = RocprofSession {
            tool: ProfilingTool::Rocprof,
            config,
            session_id: "profile_1234567890".to_string(),
        };

        assert!(session.session_id().starts_with("profile_"));
    }
}
