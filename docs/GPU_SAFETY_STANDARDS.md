# GPU Safety Standards for ROCmForge

**Status:** ENFORCEABLE - All GPU work MUST follow these standards

**Last Updated:** 2026-04-16

**Root Cause Analysis:**
- **Parallel GPU tests** → GPU memory corruption → Desktop crash
- **VRAM exhaustion** → Out of memory → System hang
- **Resource leaks** → VRAM not released → Performance degradation
- **Graph capture misuse** → HIP errors → Silent failures

---

## 1. PARALLEL TESTING PROTOCOL

### THE IRON LAW
```
NEVER run multiple GPU operations simultaneously without explicit coordination
```

### 1.1 Test Thread Safety

**MANDATORY:**
```rust
#[test]
#[ignore = "Requires GPU - run with --test-threads=1"]
fn test_gpu_operation() {
    // GPU test code here
}
```

**TEST EXECUTION:**
```bash
# ❌ WRONG - Will crash desktop
cargo test --features gpu

# ✅ CORRECT - Single-threaded execution
cargo test --features gpu -- --test-threads=1

# ✅ CORRECT - Single specific test
cargo test --features gpu test_specific_gpu_function -- --test-threads=1
```

### 1.2 GPU Mutex Pattern

For any code that might use GPU concurrently:

```rust
use std::sync::Mutex;

// Global GPU lock
static GPU_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_with_gpu_lock() {
    let _lock = GPU_LOCK.lock().unwrap();
    // Safe GPU operations here
}
```

### 1.3 Detection and Prevention

**Add to CI/CD:**
```yaml
# .github/workflows/gpu-tests.yml
- name: Run GPU tests
  run: cargo test --features gpu -- --test-threads=1 --test-threads=1
```

**Pre-commit hook:**
```bash
# Check for GPU tests without proper guards
if grep -r "cargo test.*--features gpu" .github/ 2>/dev/null | grep -v "test-threads=1"; then
    echo "❌ GPU tests must use --test-threads=1"
    exit 1
fi
```

---

## 2. VRAM MANAGEMENT STANDARDS

### 2.1 Allocation Limits

**MAX VRAM USAGE:** 80% of available VRAM per operation

```rust
// Example: Check VRAM before allocation
fn check_vram_availability(required_bytes: usize) -> Result<(), String> {
    let available = get_available_vram(); // Implement this
    let max_allowed = (available * 8) / 10; // 80% limit

    if required_bytes > max_allowed {
        return Err(format!(
            "Insufficient VRAM: need {} MB, max allowed {} MB",
            required_bytes / 1024 / 1024,
            max_allowed / 1024 / 1024
        ));
    }

    Ok(())
}
```

### 2.2 Mandatory Cleanup

**RULE: Every GPU allocation MUST have a corresponding cleanup**

```rust
struct GpuBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl GpuBuffer {
    fn new(size: usize) -> Result<Self, String> {
        check_vram_availability(size)?;
        
        let ptr = unsafe { hip_malloc(size) };
        if ptr.is_null() {
            return Err("GPU allocation failed".to_string());
        }

        Ok(Self { ptr, size })
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hip_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
```

### 2.3 VRAM Leak Detection

**Add to test suite:**
```rust
#[test]
fn test_vram_no_leaks() {
    let vram_before = get_total_allocated_vram();
    
    {
        let _buffer = GpuBuffer::new(1024 * 1024).unwrap();
        // Use buffer
    } // Buffer dropped here
    
    let vram_after = get_total_allocated_vram();
    
    assert_eq!(vram_before, vram_after, "VRAM leak detected!");
}
```

### 2.4 Batch Size Limits

**Dynamic batch sizing based on VRAM:**
```rust
fn calculate_safe_batch_size(base_size: usize) -> usize {
    let available_vram = get_available_vram();
    let vram_per_element = estimate_vram_per_element();
    
    let max_elements = (available_vram * 8) / (10 * vram_per_element); // 80% limit
    
    std::cmp::min(base_size, max_elements)
}
```

---

## 3. HIP GRAPH DECODE STANDARDS

### 3.1 When to Use Graph Capture

**USE GRAPH for:**
- ✅ Decode operations (repeated kernel launches)
- ✅ Stable, validated kernel sequences
- ✅ Performance-critical paths

**DO NOT USE GRAPH for:**
- ❌ Single-shot operations
- ❌ Experimental kernels
- ❌ Debug/testing code

### 3.2 Graph Safety Protocol

```rust
struct GpuGraph {
    graph: hipGraph_t,
    exec: hipGraphExec_t,
    initialized: bool,
}

impl GpuGraph {
    fn new() -> Self {
        Self {
            graph: std::ptr::null_mut(),
            exec: std::ptr::null_mut(),
            initialized: false,
        }
    }

    fn capture<F>(&mut self, f: F) -> Result<(), String>
    where
        F: FnOnce() -> Result<(), String>,
    {
        unsafe {
            // Begin capture
            hipGraphCreate(&mut self.graph, 0)?;
            
            hipStreamBeginCapture(hipStreamPerThread, hipStreamCaptureModeGlobal);
            
            // Run kernels
            f()?;
            
            // End capture
            self.exec = hipGraphEndCapture();
            if self.exec.is_null() {
                return Err("Graph capture failed".to_string());
            }
            
            // Validate
            let result = hipGraphInstantiate(&mut self.exec, self.graph, 0, 0);
            if result != hipSuccess {
                return Err(format!("Graph instantiation failed: {}", result));
            }
            
            self.initialized = true;
            Ok(())
        }
    }

    fn launch(&self) -> Result<(), String> {
        if !self.initialized {
            return Err("Graph not initialized".to_string());
        }
        
        unsafe {
            let result = hipGraphLaunch(self.exec, hipStreamPerThread);
            if result != hipSuccess {
                return Err(format!("Graph launch failed: {}", result));
            }
        }
        
        Ok(())
    }
}

impl Drop for GpuGraph {
    fn drop(&mut self) {
        if self.initialized {
            unsafe {
                if !self.exec.is_null() {
                    hipGraphExecDestroy(self.exec);
                }
                if !self.graph.is_null() {
                    hipGraphDestroy(self.graph);
                }
            }
        }
    }
}
```

### 3.3 Graph Error Handling

```rust
// NEVER ignore graph errors
match graph.capture(|| run_kernels()) {
    Ok(_) => {},
    Err(e) => {
        eprintln!("Graph capture failed, falling back to linear: {}", e);
        // Fallback to non-graph execution
        run_kernels_linear()?;
    }
}
```

### 3.4 Environment Override

```rust
// Respect environment variable
fn should_use_graph() -> bool {
    std::env::var("ROCMFORGE_ENABLE_GRAPH")
        .unwrap_or_else(|_| "1".to_string())
        .parse::<bool>()
        .unwrap_or(true)
}
```

---

## 4. ERROR HANDLING STANDARDS

### 4.1 HIP Error Checks

**MANDATORY: Check every HIP call**
```rust
#[macro_export]
macro_rules! hip_check {
    ($expr:expr) => {
        let result = unsafe { $expr };
        if result != hipSuccess {
            return Err(format!(
                "HIP error at {}:{}: {} ({})",
                file!(),
                line!(),
                result,
                get_hip_error_string(result)
            ));
        }
    };
}

// Usage
hip_check!(hipMalloc(&mut ptr, size));
```

### 4.2 GPU Reset Detection

```rust
fn check_gpu_health() -> Result<(), String> {
    unsafe {
        let mut device = 0;
        hip_check!(hipGetDevice(&device));
        
        let mut props = std::mem::zeroed();
        hip_check!(hipGetDeviceProperties(&mut props, device));
        
        // Check for common failure modes
        if props.major == 0 && props.minor == 0 {
            return Err("GPU appears to be in reset state".to_string());
        }
        
        Ok(())
    }
}
```

### 4.3 Timeout Protection

```rust
use std::time::Duration;

fn run_with_timeout<F, T>(f: F, timeout: Duration) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String>,
{
    let handle = std::thread::spawn(f);
    
    match handle.join_timeout(timeout) {
        Ok(result) => result,
        Err(_) => Err("GPU operation timeout".to_string()),
    }
}
```

---

## 5. TESTING STANDARDS

### 5.1 GPU Test Structure

```rust
#[test]
#[ignore = "Requires GPU - run with --test-threads=1"]
fn test_gpu_feature() {
    // 1. Check GPU availability
    let gpu_available = check_gpu_available().unwrap_or(false);
    if !gpu_available {
        println!("Skipping test - no GPU available");
        return;
    }
    
    // 2. Check VRAM before test
    let vram_before = get_total_allocated_vram();
    
    // 3. Run test
    let result = std::panic::catch_unwind(|| {
        // GPU test code here
    });
    
    // 4. Verify VRAM cleanup
    let vram_after = get_total_allocated_vram();
    assert_eq!(vram_before, vram_after, "VRAM leak detected");
    
    // 5. Propagate test result
    result.unwrap();
}
```

### 5.2 Integration Test Protocol

```bash
# SAFE integration testing
cargo test --features gpu --test integration_test -- --test-threads=1 --ignored

# NEVER run this in parallel:
cargo test --features gpu --test integration_test
```

### 5.3 Performance Testing

```rust
#[bench]
#[ignore = "Requires GPU - run with --test-threads=1"]
fn bench_gpu_operation(b: &mut Bencher) {
    // Warmup
    for _ in 0..10 {
        gpu_operation();
    }
    
    // Benchmark
    b.iter(|| {
        gpu_operation();
    });
}
```

---

## 6. CODE REVIEW CHECKLIST

Before merging GPU code, verify:

- [ ] All GPU tests use `#[ignore]` + `--test-threads=1`
- [ ] Every allocation has corresponding cleanup
- [ ] HIP errors are checked and handled
- [ ] Graph capture has fallback to linear execution
- [ ] VRAM usage is within 80% limit
- [ ] No parallel GPU operations without coordination
- [ ] Tests verify VRAM cleanup
- [ ] Timeout protection for long operations
- [ ] GPU health checks before critical operations

---

## 7. EMERGENCY RECOVERY

### 7.1 GPU Crash Recovery

```bash
# If GPU crashes during development:
pkill -9 rocmforge  # Kill stuck processes
rocm-smi --showmeminfo vram  # Check VRAM state
sudo reboot  # Last resort if GPU is stuck
```

### 7.2 VRAM Leak Recovery

```bash
# Check for VRAM leaks
watch -n 1 rocm-smi --showmeminfo vram

# If leak detected, restart process
killall rocmforge
cargo build --release --features gpu  # Rebuild
```

### 7.3 Development Safety

```bash
# Add to .bashrc for development
alias gpu-test='cargo test --features gpu -- --test-threads=1'
alias gpu-bench='cargo bench --features gpu -- --test-threads=1'
```

---

## 8. MONITORING AND OBSERVABILITY

### 8.1 VRAM Monitoring

```rust
pub fn log_vram_usage(context: &str) {
    let total = get_total_vram();
    let used = get_used_vram();
    let free = total - used;
    
    log::info!(
        "VRAM [{}]: {} MB used / {} MB total ({} MB free)",
        context,
        used / 1024 / 1024,
        total / 1024 / 1024,
        free / 1024 / 1024
    );
}
```

### 8.2 GPU Health Monitoring

```rust
pub fn check_gpu_healthy() -> bool {
    match check_gpu_health() {
        Ok(_) => true,
        Err(e) => {
            log::error!("GPU health check failed: {}", e);
            false
        }
    }
}
```

---

## 9. COMPLIANCE ENFORCEMENT

### 9.1 Pre-commit Hooks

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Check for GPU tests without proper guards
if git diff --cached --name-only | grep -E "test.*\.rs$"; then
    echo "Checking GPU test safety..."
    
    # Look for GPU tests without #[ignore] or proper thread limits
    if git diff --cached | grep -E "fn test.*gpu" | grep -v "ignore"; then
        echo "❌ GPU tests must use #[ignore] and --test-threads=1"
        exit 1
    fi
fi

echo "✅ GPU safety checks passed"
```

### 9.2 CI/CD Integration

```yaml
# .github/workflows/gpu-safety.yml
name: GPU Safety Checks

on: [pull_request]

jobs:
  gpu-safety:
    runs-on: [self-hosted, gpu]
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Check GPU test guards
        run: |
          if grep -r "cargo test.*gpu" .github/ 2>/dev/null | grep -v "test-threads=1"; then
            echo "::error::GPU tests must use --test-threads=1"
            exit 1
          fi
      
      - name: Run single-threaded GPU tests
        run: cargo test --features gpu -- --test-threads=1
```

---

## 10. SUMMARY: THE GOLDEN RULES

1. **ONE GPU OPERATION AT A TIME** - No parallel GPU tests
2. **CLEANUP WHAT YOU ALLOCATE** - Every malloc needs a free
3. **CHECK EVERY ERROR** - HIP calls can fail
4. **LIMIT VRAM USAGE** - Stay under 80%
5. **USE GRAPHS WISELY** - Only for stable, repeated operations
6. **TEST SAFELY** - Use `#[ignore]` + `--test-threads=1`
7. **MONITOR HEALTH** - Check VRAM and GPU state
8. **HAVE FALLBACKS** - Graph → Linear, GPU → CPU

**Violating these rules WILL crash your desktop.**

**Following these rules ensures stable, safe GPU computing.**

---

## Appendix A: Quick Reference

**SAFE GPU TEST TEMPLATE:**
```rust
#[test]
#[ignore = "Requires GPU - run with --test-threads=1"]
fn test_gpu_feature() {
    // 1. VRAM check
    // 2. GPU operation
    // 3. VRAM cleanup verification
}
```

**SAFE VRAM ALLOCATION:**
```rust
let vram_before = get_used_vram();
let buffer = GpuBuffer::new(size).unwrap();
// Use buffer
drop(buffer); // Explicit cleanup
let vram_after = get_used_vram();
assert_eq!(vram_before, vram_after);
```

**SAFE GRAPH USAGE:**
```rust
let mut graph = GpuGraph::new();
match graph.capture(|| run_kernels()) {
    Ok(_) => graph.launch(),
    Err(e) => run_kernels_linear(), // Fallback
}
```

---

**These standards are ENFORCEABLE. All GPU code must comply.**

**Questions? Ask before breaking desktop.**