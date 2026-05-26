//! Performance profiling for GPU kernels.
//!
//! Tracks kernel execution time and memory bandwidth utilization
//! to measure optimization impact.

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Kernel execution timing record.
#[derive(Debug, Clone)]
pub struct KernelTiming {
    pub name: String,
    pub avg_ns: u64,
    pub calls: u64,
    pub total_ns: u64,
}

impl KernelTiming {
    /// Average time in milliseconds.
    pub fn avg_ms(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.avg_ns as f64 / 1_000_000.0
        }
    }

    /// Total time in seconds.
    pub fn total_s(&self) -> f64 {
        self.total_ns as f64 / 1_000_000_000.0
    }
}

/// Performance profiler singleton.
pub struct Profiler {
    timings: Vec<Mutex<KernelTiming>>,
}

impl Profiler {
    /// Get the global profiler instance.
    pub fn global() -> &'static Mutex<Self> {
        static INSTANCE: OnceLock<Mutex<Profiler>> = OnceLock::new();
        INSTANCE.get_or_init(|| {
            Mutex::new(Profiler {
                timings: Vec::new(),
            })
        })
    }

    /// Record a kernel execution timing.
    pub fn record(kernel_name: &str, elapsed_ns: u64) {
        let profiler = Self::global().lock().unwrap();

        // Find existing timing record
        for timing in profiler.timings.iter() {
            let mut t = timing.lock().unwrap();
            if t.name == kernel_name {
                t.calls += 1;
                t.total_ns += elapsed_ns;
                t.avg_ns = t.total_ns / t.calls;
                return;
            }
        }

        // Create new timing record
        // Need to drop the lock before creating new timing
        drop(profiler);
        let mut profiler = Self::global().lock().unwrap();
        profiler.timings.push(Mutex::new(KernelTiming {
            name: kernel_name.to_string(),
            avg_ns: elapsed_ns,
            calls: 1,
            total_ns: elapsed_ns,
        }));
    }

    /// Get all timing records.
    pub fn get_timings(&self) -> Vec<KernelTiming> {
        self.timings
            .iter()
            .map(|t| t.lock().unwrap().clone())
            .collect()
    }

    /// Print timing summary.
    pub fn print_summary(&self) {
        let timings = self.get_timings();

        println!("\n=== GPU Kernel Performance ===");
        println!(
            "{:<30} {:>10} {:>10} {:>12}",
            "Kernel", "Calls", "Avg (ms)", "Total (s)"
        );
        println!("{}", "-".repeat(74));

        for timing in timings.iter() {
            println!(
                "{:<30} {:>10} {:>10.2} {:>12.4}",
                timing.name,
                timing.calls,
                timing.avg_ms(),
                timing.total_s()
            );
        }
    }
}

/// RAII timer for kernel execution.
pub struct KernelTimer {
    name: String,
    start: Instant,
}

impl KernelTimer {
    /// Start timing a kernel.
    pub fn start(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start: Instant::now(),
        }
    }
}

impl Drop for KernelTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_nanos() as u64;
        Profiler::record(&self.name, elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_record() {
        Profiler::record("test_kernel_1", 1_000_000); // 1ms
        Profiler::record("test_kernel_1", 2_000_000); // 2ms

        let profiler = Profiler::global().lock().unwrap();
        let timings = profiler.get_timings();

        let timing = timings
            .iter()
            .find(|t| t.name == "test_kernel_1")
            .expect("Expected to find test_kernel_1 in timings");
        assert_eq!(timing.calls, 2);
        assert_eq!(timing.avg_ns, 1_500_000); // average of 1ms and 2ms
    }

    #[test]
    fn test_kernel_timer() {
        {
            let _timer = KernelTimer::start("test_timer_kernel");
            std::thread::sleep(std::time::Duration::from_millis(10));
            // Timer records on drop
        }

        let profiler = Profiler::global().lock().unwrap();
        let timings = profiler.get_timings();

        let timing = timings
            .iter()
            .find(|t| t.name == "test_timer_kernel")
            .expect("Expected to find test_timer_kernel in timings");
        assert!(timing.avg_ns >= 10_000_000); // At least 10ms
    }
}
