// GPU Safety Policy and Reference
//
// This file documents the mandatory GPU safety harness that must be used
// for all real-model GPU testing. It is NOT a test file - it serves as
// policy documentation and integration reference.
//
// ================================================================
// REQUIRED SAFETY HARNESS FOR GPU WORK
// ================================================================
//
// Before ANY real-model GPU execution, the following staged safety
// checks must be performed in order:
//
// 1. ACQUIRE GPU LOCK (cross-process mutex)
//    Use: scripts/gpu_lock.sh acquire
//    Purpose: Prevent concurrent GPU access that can cause deadlocks
//
// 2. RUN GPU PREFLIGHT (staged checks)
//    Use: scripts/gpu_preflight.sh
//    Purpose: Verify driver, ROCm runtime, memory, and kernel launch
//
// 3. EXECUTE WITH TIMEOUT WRAPPER
//    Use: scripts/gpu_safe_run.sh --timeout <seconds> --max-tokens <n> <cmd>
//    Purpose: Enforce timeout and max token limits for safety
//
// 4. RELEASE GPU LOCK
//    Use: scripts/gpu_lock.sh release
//    Purpose: Allow other processes to use GPU
//
// ================================================================
// FOR DIRECT CLI TESTING (MANUAL GPU RUNS)
// ================================================================
//
// When manually testing the GPU CLI with real models, ALWAYS use the
// safe runner wrapper:
//
//   ./scripts/gpu_safe_run.sh ./target/release/rocmforge --gpu \
//     --model <path> --prompt "test"
//
// The wrapper will:
// - Acquire GPU lock (with timeout)
// - Run preflight checks
// - Enforce timeout and max-tokens limits
// - Release lock on completion or failure
//
// NEVER run:
//   ./target/release/rocmforge --gpu --model <path>
//
// ALWAYS run:
//   ./scripts/gpu_safe_run.sh ./target/release/rocmforge --gpu --model <path>
//
// ================================================================
// FOR AUTOMATED TESTS (REAL GPU QA)
// ================================================================
//
// Real GPU tests should:
// 1. Be gated by ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1
// 2. Use subprocess isolation (not direct in-process GPU calls)
// 3. Use the safe runner wrapper (gpu_safe_run.sh)
// 4. Run serially (--test-threads=1)
//
// See: tests/gpu_cli_qa.rs for reference implementation
//
// ================================================================
// ENVIRONMENT VARIABLES
// ================================================================
//
// ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS
//   Set to "1" to enable real-model GPU QA tests
//   Default: unset (tests are skipped)
//
// ROCMFORGE_GPU_LOCK_TIMEOUT
//   GPU lock acquisition timeout in seconds
//   Default: 30
//
// ROCMFORGE_DEFAULT_TIMEOUT
//   Default command execution timeout in seconds
//   Default: 120
//
// ROCMFORGE_DEFAULT_MAX_TOKENS
//   Default max tokens for decode runs
//   Default: 50
//
// ================================================================
// HANDLING LOCK ISSUES
// ================================================================
//
// If GPU lock is stuck:
//
//   # Check lock status
//   ./scripts/gpu_lock.sh status
//
//   # If stale lock (process not running), remove manually
//   rm -rf /tmp/rocmforge_gpu_lock
//
//   # Kill stuck processes
//   pkill -9 rocmforge
//
// ================================================================
// SCRIPT EXIT CODES
// ================================================================
//
// gpu_lock.sh:
//   0 - Success
//   1 - Lock acquisition timeout
//   2 - Lock not held
//   3 - Lock file corrupted
//
// gpu_preflight.sh:
//   0 - All checks passed
//   1 - Render node check failed
//   2 - ROCm visibility check failed
//   3 - Memory round-trip failed
//   4 - Trivial kernel launch failed
//
// gpu_safe_run.sh:
//   0 - Success
//   1-4 - Preflight check failed
//   10 - Lock acquisition timeout
//   11 - Command timeout
//   12 - Command execution failed
//   255 - Usage error
//
// ================================================================
// VERIFICATION CHECKLIST
// ================================================================
//
// Before merging new GPU code:
// [ ] Lock acquisition works (concurrent processes wait)
// [ ] Preflight checks pass on target hardware
// [ ] Timeout enforcement works (kill runaway processes)
// [ ] Max tokens enforcement works (prevent long runs)
// [ ] Lock release on success/failure/timeout
// [ ] Tests use env gating (ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS)
// [ ] Tests use subprocess isolation
// [ ] Tests run serially (--test-threads=1)
//
// ================================================================
// RATIONALE
// ================================================================
//
// This harness exists because:
// - Concurrent GPU access can cause MES queue teardown failures
// - Unbounded GPU runs can cause desktop freezes and GPU resets
// - Real-model testing carries risk of VRAM exhaustion and page faults
// - Previous attempts at prefill integration caused amdgpu page faults
//
// The staged approach ensures that problems are caught early
// (preflight) and contained (timeout/max-tokens/lock).
//
