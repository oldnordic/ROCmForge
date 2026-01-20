# Codebase Concerns

**Analysis Date:** 2026-01-20

## Tech Debt

**MoE (Mixture of Experts) Metadata Support:**
- Issue: `src/loader/metadata.rs:170` and `:174` contain TODOs for `num_local_experts` and `experts_per_token` fields. These are parsed but not stored in `GgufMetadata`.
- Files: `src/loader/metadata.rs`
- Impact: MoE models (Mixtral, etc.) cannot be fully supported because critical routing information is discarded during parsing.
- Fix approach: Add `num_local_experts: Option<usize>` and `experts_per_token: Option<usize>` fields to `GgufMetadata` struct, store parsed values, update `to_model_config()` conversion.

**Copy-on-Write Block Copying:**
- Issue: `src/kv_cache/kv_cache.rs:1266-1276` - `copy_block()` is a stub that allocates fresh blocks instead of copying GPU memory.
- Files: `src/kv_cache/kv_cache.rs`
- Impact: Cannot efficiently share KV cache prefixes between diverging sequences. Forces full recomputation.
- Fix approach: Implement `HipBackend::memcpy_device_to_device()` and update `copy_block()` to perform GPU-to-GPU copies.

**MXFP Quantization Types:**
- Issue: `src/loader/gguf.rs:894-898` - MXFP4/MXFP6 dequantization returns "not yet implemented" error.
- Files: `src/loader/gguf.rs`, `src/loader/mxfp.rs`
- Impact: Models using MXFP quantization cannot be loaded.
- Fix approach: Implement GPU kernels for MXFP block decoding or add CPU fallback path.

**K-Quant Types Q2_K, Q3_K, Q5_K:**
- Issue: `src/loader/gguf.rs:887-892` - Returns error for Q2_K, Q3_K, Q5_K quantization types.
- Files: `src/loader/gguf.rs`
- Impact: Many GGUF models use these quantization formats; they cannot be loaded.
- Fix approach: Add dequantization implementations for missing K-quant formats.

**Legacy Simple KV Cache:**
- Issue: `src/model/kv_cache/` (simple KV cache) is legacy but still maintained alongside the paged implementation.
- Files: `src/model/kv_cache/`, `src/model/mod.rs:28-30`
- Impact: Code duplication, maintenance burden. The simple cache should be removed once paged cache is fully validated.
- Fix approach: Deprecate and remove `model::kv_cache::KVCache`, migrate all tests to use `crate::kv_cache::KvCache`.

## Known Bugs

**Scheduler Token Stale Clone Bug:**
- Symptoms: `src/scheduler/scheduler.rs:1043` test comment notes "BUG: This fails because update_iteration_batch overwrites with stale clone!"
- Files: `src/scheduler/scheduler.rs:1005-1046`, specifically `test_stale_batch_clone_does_not_overwrite_scheduler`
- Trigger: When engine passes stale batch clones to `update_iteration_batch()` instead of fresh state from `snapshot_request()`.
- Workaround: The test verifies current (buggy) behavior. The workaround is to ensure engine always calls `snapshot_request()` before `update_iteration_batch()`.
- Fix approach: Make `update_iteration_batch()` compare token counts and only update if batch has more tokens (preserves freshest data).

**Unreachable! in GPU Dequantization:**
- Symptoms: `src/loader/gguf.rs:756` uses `unreachable!()` as catch-all for unsupported tensor types in GPU path.
- Files: `src/loader/gguf.rs:756`
- Trigger: Loading a tensor with type not matched in GPU dequantization switch.
- Workaround: CPU fallback path exists for most types.
- Fix approach: Replace `unreachable!()` with proper error return: `Err(anyhow!("GPU dequantization not implemented for {:?}", tensor_type))`.

**Panic in GPU Unavailable Tests:**
- Symptoms: Multiple test files use `panic!("GPU_SKIP")` to skip tests when GPU is unavailable.
- Files: `src/attention/causal_mask_tests.rs:30`, `src/attention/flash_attention_tests.rs:28`, `src/attention/flash_causal_tests.rs:31`, `src/mlp/rms_norm_tests.rs:31`, and many others.
- Trigger: Running tests without ROCm GPU available.
- Workaround: Tests abort with panic; CI may need special handling.
- Fix approach: Use `#[ignore]` attribute or Result-based skip instead of panic.

## Security Considerations

**Input Validation in Metadata Parsing:**
- Risk: `src/loader/metadata.rs` parses string values directly into integers without bounds checking.
- Files: `src/loader/metadata.rs`
- Current mitigation: `.parse().unwrap_or(0)` provides default values on parse failure.
- Recommendations: Add validation for reasonable ranges (e.g., `num_layers` should be 1-200, `hidden_size` should be power of 2 in typical range).

**Memory Allocation without Limits:**
- Risk: GPU buffer allocation in `src/backend/hip_backend/backend.rs` does not enforce per-request limits.
- Files: `src/backend/hip_backend/backend.rs`
- Current mitigation: HipBackend checks total GPU memory but not per-allocation quotas.
- Recommendations: Add quota system for per-request GPU allocation to prevent OOM via malicious model files.

**FFI Safety:**
- Risk: Raw pointer usage in HIP FFI bindings (`src/backend/hip_backend/backend.rs:16-74`).
- Files: `src/backend/hip_backend/backend.rs`
- Current mitigation: Wrappers like `HipBuffer`, `HipStream` use RAII for cleanup.
- Recommendations: Continue auditing FFI code; consider using safer abstractions where possible.

## Performance Bottlenecks

**CPU Dequantization Before GPU Upload:**
- Problem: `src/loader/gguf.rs:796-900` - Many tensor types are dequantized on CPU before uploading to GPU.
- Files: `src/loader/gguf.rs`
- Cause: Missing GPU kernels for some quantization types (Q4_1, Q5_0, Q5_1, etc.).
- Improvement path: Implement GPU-side dequantization kernels for all quantization types to avoid CPU-GPU round-trip.

**Lock Contention in KV Cache:**
- Problem: `src/kv_cache/kv_cache.rs` uses multiple `RwLock`-protected structures that must be acquired sequentially.
- Files: `src/kv_cache/kv_cache.rs`
- Cause: Granular locking (block_pool, block_table, page_table, sequences) requires multiple lock acquisitions per operation.
- Improvement path: Consider a single RwLock on a struct containing all cache state, or use lock-free data structures for read-heavy paths.

**Transpose on CPU for Fused QKV:**
- Problem: `src/loader/gguf.rs:924` - Fused QKV weights are transposed on CPU before GPU upload.
- Files: `src/loader/gguf.rs:915-935`
- Cause: Matmul kernels expect transposed layout but GGUF stores original orientation.
- Improvement path: Implement GPU-side transpose kernel or update matmul to handle both layouts.

**Cache Compaction Overhead:**
- Problem: `src/kv_cache/kv_cache.rs:1445-1475` - Cache compaction is expensive and locks the entire cache.
- Files: `src/kv_cache/kv_cache.rs`
- Cause: Requires traversing all sequences, reclaiming blocks, reorganizing free lists.
- Improvement path: Incremental compaction, background thread, or generational approach.

## Fragile Areas

**Lock Poisoning Handling:**
- Files: `src/kv_cache/kv_cache.rs:878-892`, `:1284-1338`, `:1536-1554`
- Why fragile: Uses `.expect()` extensively for lock poisoning; if any lock poisons, entire cache panics.
- Safe modification: Always check lock results; cache should return errors, not panic.
- Test coverage: No tests for lock poisoning scenarios; difficult to test deterministically.

**Global Backend Singleton:**
- Files: `src/backend/hip_backend/backend.rs:1290-1315` (GLOBAL_BACKEND)
- Why fragile: Global mutable state with `Arc<RwLock<Option<Arc<HipBackend>>>>`. Can cause issues with multiple GPU devices.
- Safe modification: Thread through explicit backend references instead of global singleton.
- Test coverage: Tests exist but may not cover concurrent initialization scenarios.

**Kernel Module Loading:**
- Files: `src/sampler/gpu.rs:71-200` (GLOBAL_SAMPLING_CACHE), `src/backend/hip_backend/backend.rs`
- Why fragile: Kernel loading depends on environment variables and file paths. Missing kernels cause runtime fallback to CPU.
- Safe modification: Validate kernel availability at startup, provide clear error messages.
- Test coverage: Some tests check for GPU_SKIP but kernel loading is not comprehensively tested.

**FFI Struct Layout:**
- Files: `src/backend/hip_backend/backend.rs:82-153` (HipDeviceProp)
- Why fragile: Uses opaque byte buffer with hardcoded offsets. Breaks if ROCm version changes struct layout.
- Safe modification: Generate offsets from C code at build time, add runtime sanity checks.
- Test coverage: Basic tests exist but cannot detect ROCm version incompatibilities.

**Scheduler Batch Update State Machine:**
- Files: `src/scheduler/scheduler.rs:603-648` (update_iteration_batch)
- Why fragile: Complex logic to prevent stale overwrites; comment indicates previous bugs here.
- Safe modification: Add state machine diagram to comments, add property-based tests.
- Test coverage: `test_stale_batch_clone_does_not_overwrite_scheduler` exists but tests buggy behavior.

## Scaling Limits

**KV Cache Memory:**
- Current capacity: Configurable via `CacheConfig::max_pages`, defaulting to model-specific presets (Small/Medium/Large).
- Limit: GPU memory size - KV cache can use ~50% of VRAM by design.
- Scaling path: Multi-GPU KV cache sharding (not implemented), CPU fallback for overflow (not implemented).

**Batch Size:**
- Current capacity: Default 32, configurable via `SchedulerConfig::max_batch_size`.
- Limit: No explicit limit; practical limit based on GPU memory for activations.
- Scaling path: Pipeline parallelism, tensor parallelism (not implemented).

**Sequence Length:**
- Current capacity: Configurable, defaults to 4096 tokens.
- Limit: Bounded by KV cache capacity and model architecture.
- Scaling path: PagedAttention already supports long contexts; need larger page pools.

**Request Queue:**
- Current capacity: 1000 pending requests (`SchedulerConfig::max_queue_size`).
- Limit: Fixed-size `VecDeque`; memory scales with pending request count.
- Scaling path: Unbounded queue with backpressure, or distributed queue across workers.

## Dependencies at Risk

**ROCm/HIP Driver:**
- Risk: Code tightly coupled to specific ROCm API. Driver updates may break FFI compatibility.
- Impact: Cannot run without ROCm GPU and compatible driver.
- Migration plan: Add CPU fallback for all operations, abstract HIP behind a trait.

**GGUF Format Specification:**
- Risk: GGUF is external format; changes upstream could break loader.
- Impact: Cannot load models if GGUF format changes significantly.
- Migration plan: Version detection in loader, support multiple format versions.

**Kernel HSACO Files:**
- Risk: Kernels compiled separately and loaded at runtime via environment variables (`SAMPLING_UTILS_HSACO`, etc.).
- Impact: If kernel files missing, runtime degrades to CPU fallback.
- Migration plan: Embed kernels as bytes in binary, or compile kernels at build time.

**Tokenizers:**
- Risk: `tokenizers` crate dependency for tokenizer loading.
- Impact: If tokenizer format changes, may not load models.
- Migration plan: Support multiple tokenizer formats, add fallback tokenizers.

## Missing Critical Features

**Speculative Decoding:**
- Problem: Not implemented.
- Blocks: Faster inference for draft models.
- Status: Architecture supports multiple models simultaneously; scheduler can handle speculation.

**Multi-GPU Support:**
- Problem: Single GPU only.
- Blocks: Scaling to larger models, higher throughput.
- Status: HipBackend abstracts device selection but only one device initialized.

**Streaming Output:**
- Problem: Engine returns complete response only.
- Blocks: Real-time chat applications, early token visibility.
- Status: HTTP server already supports SSE; engine integration needed.

**LoRA Adapter Support:**
- Problem: Cannot load LoRA adapters.
- Blocks: Model fine-tuning without full retraining.
- Status: No implementation started.

## Test Coverage Gaps

**Lock Poisoning Recovery:**
- What's not tested: Behavior when RwLocks become poisoned (thread panic while holding lock).
- Files: `src/kv_cache/kv_cache.rs`, `src/backend/hip_backend/backend.rs`, `src/sampler/gpu.rs`
- Risk: Lock poisoning causes panic/expect, not graceful degradation.
- Priority: Medium (poisoning indicates bugs, but should be handled).

**Concurrent Model Loading:**
- What's not tested: Multiple threads loading models simultaneously.
- Files: `src/loader/gguf.rs`, `src/engine.rs`
- Risk: Race conditions in global backend initialization, GPU cache concurrent writes.
- Priority: High (production use case).

**GPU OOM Conditions:**
- What's not tested: Behavior when GPU memory is exhausted.
- Files: `src/backend/hip_backend/backend.rs`, `src/kv_cache/kv_cache.rs`
- Risk: Undefined behavior, crashes instead of clean errors.
- Priority: High (multi-tenant environments).

**Kernel Fallback Paths:**
- What's not tested: CPU fallback when GPU kernels missing.
- Files: `src/sampler/gpu.rs`, `src/loader/gguf.rs`
- Risk: Silent failures or incorrect results when kernels unavailable.
- Priority: Medium (CI may run without GPU).

**Scheduler State Transitions:**
- What's not tested: All possible state machine transitions in continuous batching.
- Files: `src/scheduler/scheduler.rs`
- Risk: Invalid states causing request hangs or token loss.
- Priority: High (core functionality).

**Long-Running Inference:**
- What's not tested: Inference spanning millions of tokens, cache compaction, LRU eviction.
- Files: `src/kv_cache/kv_cache.rs`, `src/scheduler/scheduler.rs`
- Risk: Memory leaks, fragmentation in long-running services.
- Priority: Medium (production deployment concern).

---

*Concerns audit: 2026-01-20*
