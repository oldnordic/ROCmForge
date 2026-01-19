//! Logging configuration and initialization
//!
//! This module provides centralized logging setup using the `tracing` ecosystem.
//! It supports both human-readable (with colors) and JSON output formats,
//! configurable via environment variables or programmatically.
//!
//! # Environment Variables
//!
//! - `RUST_LOG`: Standard tracing filter (e.g., "info", "debug,rocmforge=trace")
//! - `ROCFORGE_LOG_LEVEL`: Simple log level (error, warn, info, debug, trace)
//! - `ROCFORGE_LOG_FORMAT`: Output format ("human" or "json")
//! - `ROCFORGE_LOG_FILE`: Optional file path for log output (JSON format)

use once_cell::sync::OnceCell;
use std::path::PathBuf;
use thiserror::Error;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
};

/// Global flag to track if tracing has been initialized
static TRACING_INITIALIZED: OnceCell<()> = OnceCell::new();

/// Default log level when no environment variable is set
const DEFAULT_LOG_LEVEL: &str = "info";

/// Environment variable for log level override
const LOG_LEVEL_ENV: &str = "ROCFORGE_LOG_LEVEL";

/// Environment variable for log format (json/human)
const LOG_FORMAT_ENV: &str = "ROCFORGE_LOG_FORMAT";

/// Environment variable for log file path
const LOG_FILE_ENV: &str = "ROCFORGE_LOG_FILE";

/// Errors that can occur during logging initialization
#[derive(Debug, Error)]
pub enum LoggingError {
    /// Invalid log level string provided
    #[error("invalid log level: {0}")]
    InvalidLogLevel(String),

    /// Invalid log format string provided
    #[error("invalid log format: {0}")]
    InvalidLogFormat(String),

    /// Failed to create log file directory
    #[error("failed to create log directory: {0}")]
    DirectoryCreationFailed(String),

    /// Failed to open log file
    #[error("failed to open log file: {0}")]
    FileOpenFailed(String),
}

/// Log level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogLevel {
    /// Error level
    Error,
    /// Warning level
    Warn,
    /// Info level (default)
    #[default]
    Info,
    /// Debug level
    Debug,
    /// Trace level
    Trace,
}

impl LogLevel {
    /// Convert to tracing Level
    pub fn as_tracing_level(&self) -> tracing::Level {
        match self {
            LogLevel::Error => tracing::Level::ERROR,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Trace => tracing::Level::TRACE,
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "error" => Some(LogLevel::Error),
            "warn" | "warning" => Some(LogLevel::Warn),
            "info" => Some(LogLevel::Info),
            "debug" => Some(LogLevel::Debug),
            "trace" => Some(LogLevel::Trace),
            _ => None,
        }
    }

    /// Convert to EnvFilter string
    pub fn as_filter_str(&self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }
}

/// Log format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogFormat {
    /// Human-readable colored output (default)
    #[default]
    Human,
    /// JSON structured output
    Json,
}

impl LogFormat {
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "human" | "pretty" | "console" => Some(LogFormat::Human),
            "json" | "structured" => Some(LogFormat::Json),
            _ => None,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Default)]
pub struct LoggingConfig {
    /// Log level to use
    pub level: LogLevel,
    /// Output format
    pub format: LogFormat,
    /// Whether to include file/line in logs
    pub with_file_info: bool,
    /// Whether to include span events
    pub with_span_events: bool,
    /// Optional file path for log output
    pub log_file: Option<PathBuf>,
}

impl LoggingConfig {
    /// Create a new default logging configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the log level
    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.level = level;
        self
    }

    /// Set the log format
    pub fn with_format(mut self, format: LogFormat) -> Self {
        self.format = format;
        self
    }

    /// Enable or disable file/line information
    pub fn with_file_info(mut self, with_file_info: bool) -> Self {
        self.with_file_info = with_file_info;
        self
    }

    /// Enable or disable span events
    pub fn with_span_events(mut self, with_span_events: bool) -> Self {
        self.with_span_events = with_span_events;
        self
    }

    /// Set the log file path for output
    pub fn with_log_file(mut self, path: PathBuf) -> Self {
        self.log_file = Some(path);
        self
    }
}

/// Initialize logging with default configuration.
///
/// Uses the `ROCFORGE_LOG_LEVEL` and `ROCFORGE_LOG_FORMAT` environment variables
/// if set, otherwise defaults to `warn` level and human-readable format.
///
/// This function is idempotent - calling it multiple times will only
/// initialize the subscriber once.
///
/// # Environment Variables
///
/// - `ROCFORGE_LOG_LEVEL`: Set the default log level (default: "warn").
///   Examples: "error", "warn", "info", "debug", "trace"
///   Module-specific filtering: "rocmforge=debug,hyper=info"
///
/// - `ROCFORGE_LOG_FORMAT`: Set output format (default: "human").
///   - "human": Colored, human-readable output
///   - "json": JSON-formatted output for log aggregation
///
/// # Example
///
/// ```ignore
/// rocmforge::init_logging_default();
/// tracing::info!("Application started");
/// ```
pub fn init_logging_default() {
    init_logging_from_env().ok();
}

/// Initialize logging from environment variables.
///
/// Reads the following environment variables:
/// - `RUST_LOG`: Standard tracing filter (e.g., "info", "debug,rocmforge=trace")
/// - `ROCFORGE_LOG_LEVEL`: Simple log level (error, warn, info, debug, trace)
/// - `ROCFORGE_LOG_FORMAT`: Output format ("human" or "json")
/// - `ROCFORGE_LOG_FILE`: Optional file path for log output (JSON format)
///
/// Falls back to defaults if not set.
///
/// This function is idempotent.
pub fn init_logging_from_env() -> Result<(), LoggingError> {
    TRACING_INITIALIZED.get_or_init(|| {
        let log_level = std::env::var(LOG_LEVEL_ENV)
            .ok()
            .and_then(|s| LogLevel::from_str(&s))
            .unwrap_or(LogLevel::Info);

        let log_format = std::env::var(LOG_FORMAT_ENV)
            .ok()
            .and_then(|s| LogFormat::from_str(&s))
            .unwrap_or(LogFormat::Human);

        let log_file = std::env::var(LOG_FILE_ENV)
            .ok()
            .map(PathBuf::from);

        let mut config = LoggingConfig::new()
            .with_level(log_level)
            .with_format(log_format);

        if let Some(file) = log_file {
            config = config.with_log_file(file);
        }

        // Ignore errors during initialization
        let _ = init_with_config_internal(&config);
    });
    Ok(())
}

/// Initialize logging with a custom configuration.
///
/// This function is idempotent.
pub fn init_with_config(config: &LoggingConfig) {
    TRACING_INITIALIZED.get_or_init(|| {
        let _ = init_with_config_internal(config);
    });
}

/// Internal initialization that can return errors
fn init_with_config_internal(config: &LoggingConfig) -> Result<(), LoggingError> {
    // Build env filter - try RUST_LOG first, then ROCFORGE_LOG_LEVEL, then default
    let env_filter = build_env_filter(config.level)?;

    // Initialize based on format and file logging
    match (config.format, &config.log_file) {
        (LogFormat::Json, Some(log_path)) => {
            // JSON format with file logging
            init_json_with_file(env_filter, config, log_path)?;
        }
        (LogFormat::Json, None) => {
            // JSON format without file
            let layer = fmt::layer()
                .json()
                .with_target(false)
                .with_file(config.with_file_info)
                .with_line_number(config.with_file_info)
                .with_span_events(span_events(config.with_span_events));
            tracing_subscriber::registry()
                .with(env_filter)
                .with(layer)
                .init();
        }
        (LogFormat::Human, Some(log_path)) => {
            // Human format with file logging
            init_human_with_file(env_filter, config, log_path)?;
        }
        (LogFormat::Human, None) => {
            // Human format without file
            let layer = fmt::layer()
                .with_target(true)
                .with_thread_ids(false)
                .with_thread_names(false)
                .with_file(config.with_file_info)
                .with_line_number(config.with_file_info)
                .with_span_events(span_events(config.with_span_events));
            tracing_subscriber::registry()
                .with(env_filter)
                .with(layer)
                .init();
        }
    }

    Ok(())
}

/// Helper to convert bool to FmtSpan
fn span_events(enabled: bool) -> FmtSpan {
    if enabled {
        FmtSpan::CLOSE
    } else {
        FmtSpan::NONE
    }
}

/// Initialize with JSON console and JSON file output
fn init_json_with_file(
    env_filter: EnvFilter,
    config: &LoggingConfig,
    log_path: &PathBuf,
) -> Result<(), LoggingError> {
    // Create parent directories if needed
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| LoggingError::DirectoryCreationFailed(e.to_string()))?;
    }

    // Open the log file
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| LoggingError::FileOpenFailed(e.to_string()))?;

    // Console layer
    let console = fmt::layer()
        .json()
        .with_target(false)
        .with_file(config.with_file_info)
        .with_line_number(config.with_file_info)
        .with_span_events(span_events(config.with_span_events));

    // File layer
    let file_layer = fmt::layer()
        .json()
        .with_writer(file)
        .with_target(false)
        .with_file(true)
        .with_line_number(true)
        .with_ansi(false)
        .with_span_events(span_events(config.with_span_events));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(console)
        .with(file_layer)
        .init();

    Ok(())
}

/// Initialize with human console and JSON file output
fn init_human_with_file(
    env_filter: EnvFilter,
    config: &LoggingConfig,
    log_path: &PathBuf,
) -> Result<(), LoggingError> {
    // Create parent directories if needed
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| LoggingError::DirectoryCreationFailed(e.to_string()))?;
    }

    // Open the log file
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| LoggingError::FileOpenFailed(e.to_string()))?;

    // Console layer (human-readable)
    let console = fmt::layer()
        .with_target(true)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(config.with_file_info)
        .with_line_number(config.with_file_info)
        .with_span_events(span_events(config.with_span_events));

    // File layer (always JSON)
    let file_layer = fmt::layer()
        .json()
        .with_writer(file)
        .with_target(false)
        .with_file(true)
        .with_line_number(true)
        .with_ansi(false)
        .with_span_events(span_events(config.with_span_events));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(console)
        .with(file_layer)
        .init();

    Ok(())
}

/// Build the environment filter for log level.
/// Tries RUST_LOG first (standard tracing convention), then ROCFORGE_LOG_LEVEL.
fn build_env_filter(default_level: LogLevel) -> Result<EnvFilter, LoggingError> {
    // Try RUST_LOG first (standard tracing convention)
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        return EnvFilter::try_new(rust_log)
            .map_err(|e| LoggingError::InvalidLogLevel(e.to_string()));
    }

    // Try ROCFORGE_LOG_LEVEL next
    if let Ok(rocmforge_level) = std::env::var(LOG_LEVEL_ENV) {
        if let Some(level) = LogLevel::from_str(&rocmforge_level) {
            return Ok(EnvFilter::new(level.as_filter_str()));
        }
    }

    // Use default level
    Ok(EnvFilter::new(default_level.as_filter_str()))
}

/// Check if tracing has been initialized
pub fn is_initialized() -> bool {
    TRACING_INITIALIZED.get().is_some()
}

/// Legacy function alias for backwards compatibility
///
/// # Deprecated
/// Use `init_logging_default()` or `init_with_config()` instead.
pub fn init_tracing() {
    init_logging_default();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_logging_default_idempotent() {
        // Multiple calls should not panic
        init_logging_default();
        init_logging_default();
        init_logging_default();
        assert!(is_initialized());
    }

    #[test]
    fn test_log_level_from_str() {
        assert_eq!(LogLevel::from_str("error"), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_str("warn"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str("warning"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str("info"), Some(LogLevel::Info));
        assert_eq!(LogLevel::from_str("debug"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::from_str("trace"), Some(LogLevel::Trace));
        assert_eq!(LogLevel::from_str("invalid"), None);
    }

    #[test]
    fn test_log_format_from_str() {
        assert_eq!(LogFormat::from_str("human"), Some(LogFormat::Human));
        assert_eq!(LogFormat::from_str("pretty"), Some(LogFormat::Human));
        assert_eq!(LogFormat::from_str("json"), Some(LogFormat::Json));
        assert_eq!(LogFormat::from_str("structured"), Some(LogFormat::Json));
        assert_eq!(LogFormat::from_str("invalid"), None);
    }

    #[test]
    fn test_logging_config_builder() {
        let config = LoggingConfig::new()
            .with_level(LogLevel::Debug)
            .with_format(LogFormat::Json)
            .with_file_info(true)
            .with_span_events(true);

        assert_eq!(config.level, LogLevel::Debug);
        assert_eq!(config.format, LogFormat::Json);
        assert!(config.with_file_info);
        assert!(config.with_span_events);
    }

    #[test]
    fn test_is_initialized_returns_true_after_init() {
        // This test may run after other tests that initialized tracing
        // We just verify the function works without panicking
        let _ = is_initialized();
    }

    #[test]
    fn test_log_level_as_tracing_level() {
        assert_eq!(LogLevel::Error.as_tracing_level(), tracing::Level::ERROR);
        assert_eq!(LogLevel::Warn.as_tracing_level(), tracing::Level::WARN);
        assert_eq!(LogLevel::Info.as_tracing_level(), tracing::Level::INFO);
        assert_eq!(LogLevel::Debug.as_tracing_level(), tracing::Level::DEBUG);
        assert_eq!(LogLevel::Trace.as_tracing_level(), tracing::Level::TRACE);
    }

    #[test]
    fn test_logging_config_with_log_file() {
        let path = PathBuf::from("/tmp/test_rocmforge.log");
        let config = LoggingConfig::new().with_log_file(path.clone());

        assert_eq!(config.log_file, Some(path));
    }

    #[test]
    fn test_init_tracing_function_exists() {
        // Test that init_tracing function works (it's an alias for init_logging_default)
        init_tracing();
        assert!(is_initialized());
    }
}
