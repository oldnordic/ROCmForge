//! Kernel timer for measuring GPU kernel execution time
//!
//! This module provides timing infrastructure for profiling GPU kernels.
//! It supports both HIP event-based timing (for GPU kernels) and CPU-side timing
//! (for fallback paths).
//!
//! # Example
//!
//! ```rust
//! use rocmforge::profiling::KernelTimer;
//!
//! // Create a new timer for a kernel
//! let mut timer = KernelTimer::for_kernel("my_kernel");
//!
//! // Start timing
//! timer.start()?;
//!
//! // ... execute kernel ...
//!
//! // Stop timing and get elapsed time
//! timer.stop()?;
//! let elapsed = timer.elapsed();
//!
//! println!("Kernel '{}' took {:.2} ms", timer.name(), elapsed);
//! ```

#[cfg(feature = "rocm")]
use crate::backend::{HipError, HipResult, HipStream, HipEvent};
use std::time::Instant;

/// Timer for measuring kernel execution time
///
/// `KernelTimer` provides a unified interface for timing GPU kernels and CPU operations.
/// When the `rocm` feature is enabled, it uses HIP events for accurate GPU timing.
/// Otherwise, it falls back to CPU-side timing.
///
/// # GPU Timing (with `rocm` feature)
///
/// GPU timing uses HIP events to measure the actual time spent executing kernels on the GPU.
/// This is more accurate than CPU timing because it doesn't include driver overhead.
///
/// # CPU Timing (fallback)
///
/// When HIP is not available, `KernelTimer` falls back to CPU-side timing using `std::time::Instant`.
/// This measures wall-clock time and may include driver overhead.
#[derive(Debug)]
pub struct KernelTimer {
    /// Name of the kernel being timed
    name: String,
    /// Start event/time
    start: Option<TimerStart>,
    /// Stop event/time
    stop: Option<TimerStop>,
    /// Elapsed time in milliseconds (after stop() is called)
    elapsed_ms: Option<f32>,
}

/// Internal representation of timer start state
#[derive(Debug)]
enum TimerStart {
    /// GPU timing using HIP events
    #[cfg(feature = "rocm")]
    Gpu {
        event: HipEvent,
    },
    /// CPU timing using Instant
    Cpu {
        instant: Instant,
    },
}

/// Internal representation of timer stop state
#[derive(Debug)]
enum TimerStop {
    /// GPU timing using HIP events
    #[cfg(feature = "rocm")]
    Gpu {
        event: HipEvent,
    },
    /// CPU timing using Instant
    Cpu {
        instant: Instant,
    },
}

impl KernelTimer {
    /// Create a new timer for the specified kernel
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the kernel to time
    ///
    /// # Example
    ///
    /// ```rust
    /// use rocmforge::profiling::KernelTimer;
    ///
    /// let timer = KernelTimer::for_kernel("matmul");
    /// ```
    pub fn for_kernel(name: impl Into<String>) -> Self {
        KernelTimer {
            name: name.into(),
            start: None,
            stop: None,
            elapsed_ms: None,
        }
    }

    /// Get the name of the kernel being timed
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Start timing for GPU kernel execution
    ///
    /// This method records a HIP event in the given stream, marking the start
    /// of the timed region. Subsequent GPU operations will be timed until `stop()` is called.
    ///
    /// # Arguments
    ///
    /// * `stream` - HIP stream to record the start event in
    ///
    /// # Errors
    ///
    /// Returns `HipError` if the event recording fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use rocmforge::profiling::KernelTimer;
    ///
    /// let mut timer = KernelTimer::for_kernel("matmul");
    /// timer.start(&stream)?;
    ///
    /// // ... execute kernel ...
    ///
    /// timer.stop(&stream)?;
    /// ```
    #[cfg(feature = "rocm")]
    pub fn start(&mut self, stream: &HipStream) -> HipResult<()> {
        // Create start event
        let start_event = HipEvent::new()?;

        // Record the event in the stream
        start_event.record(stream)?;

        self.start = Some(TimerStart::Gpu {
            event: start_event,
        });
        self.stop = None;
        self.elapsed_ms = None;

        Ok(())
    }

    /// Start timing for CPU operations (fallback path)
    ///
    /// This method starts a CPU-side timer using `std::time::Instant`.
    /// Use this for timing CPU operations or when HIP is not available.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rocmforge::profiling::KernelTimer;
    ///
    /// let mut timer = KernelTimer::for_kernel("cpu_matmul");
    /// timer.start_cpu();
    ///
    /// // ... execute CPU operation ...
    ///
    /// timer.stop_cpu();
    /// ```
    pub fn start_cpu(&mut self) {
        self.start = Some(TimerStart::Cpu {
            instant: Instant::now(),
        });
        self.stop = None;
        self.elapsed_ms = None;
    }

    /// Stop timing for GPU kernel execution
    ///
    /// This method records a HIP event in the given stream, marking the end
    /// of the timed region. It also synchronizes the stop event to ensure
    /// the elapsed time is available.
    ///
    /// # Arguments
    ///
    /// * `stream` - HIP stream to record the stop event in
    ///
    /// # Errors
    ///
    /// Returns `HipError` if event creation or recording fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use rocmforge::profiling::KernelTimer;
    ///
    /// timer.start(&stream)?;
    /// // ... execute kernel ...
    /// timer.stop(&stream)?;
    /// ```
    #[cfg(feature = "rocm")]
    pub fn stop(&mut self, stream: &HipStream) -> HipResult<()> {
        let start_event = match &self.start {
            Some(TimerStart::Gpu { event }) => event,
            Some(TimerStart::Cpu { .. }) => {
                return Err(HipError::GenericError(
                    "Cannot call stop() on a CPU-started timer. Use stop_cpu() instead.".to_string(),
                ))
            }
            None => {
                return Err(HipError::GenericError(
                    "Timer must be started before stopping".to_string(),
                ))
            }
        };

        // Create stop event
        let stop_event = HipEvent::new()?;

        // Record the event in the stream
        stop_event.record(stream)?;

        // Synchronize the stop event to ensure timing is available
        // This is necessary for accurate elapsed time calculation
        stop_event.synchronize()?;

        // Calculate elapsed time before moving stop_event
        let elapsed = start_event.elapsed_time(&stop_event)?;

        self.stop = Some(TimerStop::Gpu {
            event: stop_event,
        });

        self.elapsed_ms = Some(elapsed);

        Ok(())
    }

    /// Stop timing for CPU operations (fallback path)
    ///
    /// This method stops a CPU-side timer and calculates the elapsed time.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rocmforge::profiling::KernelTimer;
    ///
    /// timer.start_cpu();
    /// // ... execute CPU operation ...
    /// timer.stop_cpu();
    /// ```
    pub fn stop_cpu(&mut self) {
        let stop_instant = Instant::now();

        self.stop = Some(TimerStop::Cpu {
            instant: stop_instant,
        });

        // Calculate elapsed time
        if let Some(TimerStart::Cpu { instant: start }) = &self.start {
            let duration = stop_instant.duration_since(*start);
            self.elapsed_ms = Some(duration.as_secs_f64() as f32 * 1000.0);
        }
    }

    /// Get the elapsed time in milliseconds
    ///
    /// Returns `None` if the timer hasn't been stopped yet.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rocmforge::profiling::KernelTimer;
    ///
    /// let mut timer = KernelTimer::for_kernel("matmul");
    /// timer.start_cpu();
    /// // ... execute operation ...
    /// timer.stop_cpu();
    ///
    /// if let Some(elapsed) = timer.elapsed() {
    ///     println!("Elapsed: {:.2} ms", elapsed);
    /// }
    /// ```
    pub fn elapsed(&self) -> Option<f32> {
        self.elapsed_ms
    }

    /// Get the elapsed time in milliseconds, or panic if not available
    ///
    /// This is a convenience method for situations where you know the timer
    /// has been stopped and want to unwrap the result directly.
    ///
    /// # Panics
    ///
    /// Panics if the timer hasn't been stopped yet.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rocmforge::profiling::KernelTimer;
    ///
    /// let mut timer = KernelTimer::for_kernel("matmul");
    /// timer.start_cpu();
    /// // ... execute operation ...
    /// timer.stop_cpu();
    ///
    /// let elapsed = timer.elapsed_unwrap(); // f32
    /// ```
    pub fn elapsed_unwrap(&self) -> f32 {
        self.elapsed_ms.expect(
            "Timer must be stopped before calling elapsed_unwrap()",
        )
    }

    /// Check if the timer has been started
    pub fn is_started(&self) -> bool {
        self.start.is_some()
    }

    /// Check if the timer has been stopped
    pub fn is_stopped(&self) -> bool {
        self.stop.is_some()
    }
}

/// Scoped kernel timer that automatically stops when dropped
///
/// `ScopedTimer` measures the time from creation until it's dropped.
/// This is useful for automatically timing blocks of code.
///
/// # Example
///
/// ```rust
/// use rocmforge::profiling::ScopedTimer;
///
/// {
///     let _timer = ScopedTimer::new("my_operation");
///     // ... code to time ...
/// } // Timer stops here and logs elapsed time
/// ```
#[derive(Debug)]
pub struct ScopedTimer {
    name: String,
    start: Instant,
}

impl ScopedTimer {
    /// Create a new scoped timer with the given name
    pub fn new(name: impl Into<String>) -> Self {
        ScopedTimer {
            name: name.into(),
            start: Instant::now(),
        }
    }

    /// Get the elapsed time in milliseconds
    pub fn elapsed(&self) -> f32 {
        let duration = self.start.elapsed();
        duration.as_secs_f64() as f32 * 1000.0
    }

    /// Get the name of the timer
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let elapsed = self.elapsed();
        tracing::debug!("ScopedTimer '{}': {:.3} ms", self.name, elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_timer_creation() {
        let timer = KernelTimer::for_kernel("test_kernel");
        assert_eq!(timer.name(), "test_kernel");
        assert!(!timer.is_started());
        assert!(!timer.is_stopped());
        assert!(timer.elapsed().is_none());
    }

    #[test]
    fn test_kernel_timer_cpu_timing() {
        let mut timer = KernelTimer::for_kernel("cpu_kernel");
        assert!(!timer.is_started());

        timer.start_cpu();
        assert!(timer.is_started());
        assert!(!timer.is_stopped());
        assert!(timer.elapsed().is_none());

        // Simulate some work
        std::thread::sleep(std::time::Duration::from_millis(10));

        timer.stop_cpu();
        assert!(timer.is_stopped());

        let elapsed = timer.elapsed().expect("elapsed should be Some after stop");
        // Should be at least 10ms (plus overhead)
        assert!(elapsed >= 10.0, "Expected at least 10ms, got {:.2} ms", elapsed);
        // Should be reasonable (less than 1 second)
        assert!(elapsed < 1000.0, "Expected less than 1000ms, got {:.2} ms", elapsed);
    }

    #[test]
    fn test_kernel_timer_elapsed_unwrap() {
        let mut timer = KernelTimer::for_kernel("unwrap_test");
        timer.start_cpu();
        timer.stop_cpu();

        let elapsed = timer.elapsed_unwrap();
        assert!(elapsed >= 0.0);
    }

    #[test]
    #[should_panic(expected = "Timer must be stopped")]
    fn test_kernel_timer_elapsed_unwrap_panics_if_not_stopped() {
        let mut timer = KernelTimer::for_kernel("panic_test");
        timer.start_cpu();
        timer.elapsed_unwrap(); // Should panic
    }

    #[test]
    fn test_scoped_timer() {
        // ScopedTimer automatically logs on drop
        // Just verify it doesn't panic and compiles correctly
        let _timer = ScopedTimer::new("scoped_test");

        // Simulate some work
        std::thread::sleep(std::time::Duration::from_millis(5));

        // Timer drops here and logs the elapsed time
    }

    #[test]
    fn test_multiple_timers() {
        let mut timer1 = KernelTimer::for_kernel("kernel1");
        let mut timer2 = KernelTimer::for_kernel("kernel2");

        timer1.start_cpu();
        timer2.start_cpu();

        std::thread::sleep(std::time::Duration::from_millis(10));

        timer1.stop_cpu();
        timer2.stop_cpu();

        let elapsed1 = timer1.elapsed().unwrap();
        let elapsed2 = timer2.elapsed().unwrap();

        // Both should be at least 10ms
        assert!(elapsed1 >= 10.0);
        assert!(elapsed2 >= 10.0);
    }

    #[test]
    fn test_timer_reuse() {
        let mut timer = KernelTimer::for_kernel("reusable");

        // First use
        timer.start_cpu();
        std::thread::sleep(std::time::Duration::from_millis(5));
        timer.stop_cpu();
        let elapsed1 = timer.elapsed().unwrap();

        // Second use
        timer.start_cpu();
        std::thread::sleep(std::time::Duration::from_millis(10));
        timer.stop_cpu();
        let elapsed2 = timer.elapsed().unwrap();

        // Second timing should be longer
        assert!(elapsed2 > elapsed1);
    }

    #[test]
    fn test_scoped_timer_accuracy() {
        // Test timing accuracy is within reasonable bounds
        let timer = ScopedTimer::new("accuracy_test");
        std::thread::sleep(std::time::Duration::from_millis(20));
        let elapsed = timer.elapsed();

        // Should be close to 20ms (with tolerance for scheduling overhead)
        assert!(elapsed >= 20.0, "Expected at least 20ms, got {:.2} ms", elapsed);
        assert!(elapsed < 100.0, "Expected less than 100ms, got {:.2} ms", elapsed);
    }
}
