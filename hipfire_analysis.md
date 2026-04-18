# hipfire Analysis - Findings for ROCmForge

**Date:** 2026-04-18
**Repository:** https://github.com/Kaden-Schutt/hipfire/tree/dflash
**Purpose:** Learn from another Rust+HIP LLM inference engine

## Key Insights

### 1. WMMA on RDNA3 (Wave Matrix Multiply-Accumulate)

hipfire uses AMD WMMA intrinsics for matrix multiplication on RDNA3:

```cpp
// File: kernels/src/gemm_hfq4g256_residual_wmma.hip
acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
```

- **Intrinsic:** `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
- **Tile size:** 16×16×16 (similar to NVIDIA tensor cores)
- **Wave size:** 32 threads per wave
- **Architectures:** gfx1100+ (RDNA3 only: RX 7900 XTX, etc.)
- **Use case:** Prefill GEMM acceleration

**Performance impact (from README):**
- 9B model prefill: 1663 tok/s
- Effective bandwidth: 654 GiB/s (68% of 960 GB/s peak)
- 27B model prefill: 478 tok/s

**Relevance to ROCmForge:**
- We don't use WMMA yet - potential performance gain for RDNA3 users
- Requires detecting gfx1100+ at runtime
- Needs separate kernel code path for WMMA vs non-WMMA

### 2. Custom Quantization Format: HFQ

hipfire defines their own quantization format instead of using GGUF:

**HFQ4-G256 format:**
- 4-bit weights
- Group size: 256
- FP16 scale and zero-point per group
- Block size: 136 bytes (2 + 2 + 128 bytes for scale/zp/weights)

**Advantages over GGUF Q4_0:**
- Larger groups (256 vs 32) = less scale overhead
- FP16 scales (not FP32) = smaller memory footprint
- Zero-point per group (better accuracy)
- Optimized for WMMA dequantization pattern

**Format spec (from hfq.rs):**
```rust
pub struct HfqTensorInfo {
    pub quant_type: u8, // 0=Q4F16G64, 1=F16, 2=F32
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data_offset: usize,
    pub data_size: usize,
}
```

**Relevance to ROCmForge:**
- Current: GGUF Q4_0 (32-value blocks, FP32 scales)
- Could add HFQ4-G256 support for better WMMA utilization
- Trade-off: New format vs GGUF ecosystem compatibility

### 3. DFlash Speculative Decoding

The `dflash` branch implements a draft model for speculative decoding:

**Architecture:**
- 5-layer Qwen3 decoder (draft model)
- Non-causal cross-attention over target model hidden states
- No persistent KV cache (recomputes from target hidden states)
- Native Rust+HIP implementation (no Python)

**Performance benefit:**
- Speculative decoding can accelerate generation 2-3×
- Draft model is tiny, runs very fast
- Target model verifies multiple tokens at once

**Implementation:**
- File: `crates/engine/src/dflash.rs`
- Kernel: `kernels/src/attention_dflash.hip`
- Format: `.hfq` files with `arch_id = 20` for draft models

**Relevance to ROCmForge:**
- We don't have speculative decoding yet
- Could implement similar draft model approach
- Requires training draft models or converting existing ones

### 4. hipGraph Decode

From README: "WMMA prefill + hipGraph decode"

hipfire uses HIP graphs for the decode path, similar to our efforts.

**Difference:**
- They use hipGraph for decode (not prefill)
- We're experimenting with hipGraph for both

**Learning:**
- hipGraph is stable for decode in production
- WMMA used for prefill, not decode (smaller batches)
- Clear separation between prefill (WMMA) and decode (hipGraph) strategies

### 5. Performance vs Ollama

**Results on RX 7900 XTX (same models, same hardware):**

| Model | hipfire decode | ollama decode | speedup |
|-------|---------------|---------------|---------|
| Qwen 3.5 0.8B | 353 tok/s | 168 tok/s | **2.10×** |
| Qwen 3.5 4B | 165 tok/s | 93 tok/s | **1.78×** |
| Qwen 3.5 9B | 122 tok/s | 71 tok/s | **1.71×** |

**Why hipfire wins:**
1. Custom kernels optimized for RDNA
2. Better quantization format (HFQ vs GGUF Q4_K_M)
3. Native HIP (no hipfy CUDA overhead)
4. WMMA for prefill
5. hipGraph for decode

**Relevance to ROCmForge:**
- Our speed: ~154 tok/s (0.5B model)
- hipfire: 353 tok/s (0.8B model)
- Gap suggests we have significant optimization headroom

### 6. File Format Design

hipfire uses custom `.hfq` format instead of GGUF:

**HFQ format structure:**
```
[Header: 32 bytes]
  - Magic: "HFQM"
  - Version
  - Architecture ID
  - Tensor count
  - Metadata offset
  - Data offset

[Metadata JSON]
  - Config (hidden size, heads, etc.)
  - Model architecture details

[Tensor Index]
  - Name, type, shape for each tensor
  - Data offset and size

[Tensor Data]
  - Quantized weights
```

**Advantages:**
- Simpler parsing (no complex GGUF nested structures)
- Supports custom quantization formats
- JSON metadata is easy to extend
- Direct mmap loading

**Trade-offs:**
- No ecosystem compatibility (can't use existing GGUF models)
- Need conversion tool
- Fragmented ecosystem (every project has own format)

**Relevance to ROCmForge:**
- Current: GGUF format (ecosystem compatibility)
- Could offer both: GGUF for compatibility, native format for performance

### 7. Code Architecture

**Crate structure:**
- `hip-bridge`: FFI layer to HIP runtime
- `hsa-bridge`: FFI layer to HSA runtime
- `rdna-compute`: GPU abstraction and kernel dispatch
- `engine`: Model inference logic
- `redline`: Performance monitoring

**Similar to our structure:**
- `src/gpu/device.rs` ≈ `hip-bridge`
- `src/gpu/ops.rs` ≈ `rdna-compute`
- `src/gpu/forward.rs` ≈ `engine`

**Key difference:**
- hipfire splits concerns more finely (separate crates)
- We're monolithic (all in `src/gpu/`)

**Learning:**
- Splitting concerns might help with maintenance
- Separate FFI layers make hipfire more portable

### 8. Kernel Naming Conventions

hipfire uses descriptive kernel names:

```
attention_flash_asym3_tile_batched.hip
gemm_gate_up_hfq4g256_dot2.hip
gemm_hfq4g256_residual_wmma.hip
```

**Pattern:** `<operation>_<format>_<optimization>.hip`

- `asym3`: Asymptotic 3 (quantization variant)
- `tile`: Tiled memory layout
- `batched`: Batch processing
- `dot2`: DOT product 2-way optimization
- `wmma`: Wave matrix multiply
- `residual`: Fused residual add

**Our naming:**
```
q4_0_fused_norm_qkv_rope.hip
q4_k_gemm.hip
```

**Learning:**
- More descriptive names help with kernel selection
- Encoding optimization in filename makes dispatch clearer

### 9. Testing Infrastructure

hipfire has extensive testing:

```
tests/
  batch_attn.rs
  quantization.rs
  wmma.rs
  spec_decode.rs
```

And benchmarking examples:
```
bench_hfq6_vs_hfq4.rs
bench_wide.rs
debug_dequant.rs
```

**Relevance to ROCmForge:**
- We have `tests/gpu_forward_regression.rs` but could expand
- Should add quantization correctness tests
- Need kernel-level unit tests

### 10. Performance Monitoring

hipfire has a `redline` crate for performance monitoring.

Features (from name):
- FPS tracking
- Memory bandwidth monitoring
- Kernel timing
- Bottleneck identification

**Relevance to ROCmForge:**
- We have basic profiling but no dedicated monitoring
- Could help identify bottlenecks in fusion kernels

## Recommendations for ROCmForge

### Immediate Wins

1. **Add WMMA support for RDNA3**
   - Detect gfx1100+ at runtime
   - Add separate WMMA kernel path for prefill
   - Expected gain: 2-3× prefill speedup

2. **Optimize dequantization pattern**
   - hipfire loads 16 values with efficient bit extraction
   - Our current fix is correct but may not be optimal
   - Study their `DQ` macro pattern

3. **Split quantization from projection**
   - hipfire does dequant → GEMM in separate kernels
   - We fuse everything (may limit optimization opportunities)

### Medium-Term

4. **Consider hybrid format support**
   - Keep GGUF for ecosystem
   - Add native format for optimized kernels
   - Conversion tool for existing models

5. **Add speculative decoding**
   - Start with small draft model
   - Focus on decode acceleration (where users notice speed)
   - Use DFlash as reference implementation

6. **Better kernel organization**
   - Adopt descriptive naming convention
   - Separate kernel variants by filename
   - Makes dispatch logic clearer

### Long-Term

7. **Code restructuring**
   - Split GPU code into separate crates
   - Better FFI abstractions
   - Easier to maintain and test

8. **Comprehensive testing**
   - Add quantization correctness tests
   - Kernel-level unit tests
   - Performance regression tests

## Files to Study

### High Priority
- `kernels/src/gemm_hfq4g256_residual_wmma.hip` - WMMA usage
- `kernels/src/gemm_hfq4g256_residual.hip` - Non-WMMA baseline
- `crates/engine/src/hfq.rs` - HFQ format parsing
- `crates/engine/src/dflash.rs` - Speculative decoding

### Medium Priority
- `kernels/src/attention_flash_asym3_tile.hip` - Attention optimization
- `crates/hip-bridge/src/lib.rs` - HIP FFI patterns
- `docs/DFLASH_ARCHITECTURE.md` - Draft model design

### Low Priority
- `harness.sh` - Testing methodology
- `CHANGELOG.md` - Feature evolution history
- `CONTRIBUTING.md` - Development workflow

## Conclusion

hipfire validates that Rust+HIP is a viable approach for LLM inference on AMD GPUs. Their performance numbers (1.7-2.1× faster than ollama) show that native HIP kernels can outperform hipfied CUDA.

**Key takeaways:**
1. WMMA is critical for prefill performance on RDNA3
2. Custom quantization formats can beat GGUF
3. Speculative decoding provides major user-visible gains
4. Clear separation between prefill (WMMA) and decode (hipGraph) strategies

**Our path forward:**
- Short term: Fix correctness bugs (DONE)
- Medium term: Add WMMA for RDNA3, optimize kernels
- Long term: Speculative decoding, custom format support

The fusion kernel bug we fixed was about correctness. hipfire shows that performance comes from architecture-level optimizations (WMMA, better formats, speculative decoding), not just fusing operations.
