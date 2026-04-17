# ROCmForge Known Limitations

**Last Updated:** 2026-04-17

## Current Limitations

### GPU Features

#### QKV Fusion with Bias
- **Status:** ⚠️ Partial Support
- **Description:** GQA QKV fusion kernel supports bias parameters, but has not been extensively tested with models that have bias
- **Impact:** Models with bias may not use the fused QKV path, falling back to unfused implementation
- **Recommendation:** Test thoroughly before using with biased models
- **Tracking:** Task #147 - "Add bias support to GQA QKV fusion kernel" (completed, but needs validation)

### Quantization Formats

All quantization formats are now working correctly:
- ✅ Q4_0 - Fixed (April 17, 2026)
- ✅ Q4_K - Working
- ✅ Q5_K - Working
- ✅ Q6_K - Fixed (April 16, 2026)
- ✅ Q8_0 - Activation format, fixed (April 17, 2026)

### Performance

#### Decode Graph Capture
- **Status:** ✅ Implemented
- **Limitation:** Must be manually enabled via `ROCMFORGE_ENABLE_DECODE_GRAPH=1`
- **Default:** Off (for backward compatibility)
- **Reason:** Some edge cases still being validated

#### Launch Autotune
- **Status:** ✅ Implemented
- **Limitation:** Opt-in via `ROCMFORGE_ENABLE_LAUNCH_AUTOTUNE=1`
- **Default:** Off
- **Impact:** Without autotune, uses heuristic wave counts (may not be optimal for all GPUs)

## Testing Gaps

### Comprehensive Test Coverage

The following areas need additional testing:
1. **Bias support validation** - Extensive testing with biased models needed
2. **Multi-GPU** - Not currently supported
3. **FP16/BF16 storage** - Currently using FP32 for weights
4. **Batch size > 1** - Single-token decode only (batched prefill exists, batched decode does not)

## Platform Support

### Operating Systems
- ✅ Linux (primary development platform)
- ❌ Windows - Not tested
- ❌ macOS - Not supported (ROCm limitation)

### GPU Architectures
- ✅ RDNA3 (RX 7000 series) - Primary target (tested on RX 7900 XT)
- ⚠️ RDNA2 - May work, not extensively tested
- ⚠️ CDNA - Not tested (MI200 series)
- ❌ NVIDIA - Not supported (CUDA required)

### ROCm Versions
- ✅ ROCm 7.2 - Primary development target
- ⚠️ Other versions - May work, not tested

## Model Compatibility

### Tested Models
- ✅ Qwen2.5-0.5B-Instruct (all quant formats)
- ✅ Qwen2-0.5B-Instruct (Q6_K tested)

### Model Architectures
- ✅ Decoder-only transformers (GQA, MHA)
- ❌ Encoder-decoder - Not supported
- ❌ Encoder-only - Not supported

### Architecture-Specific Features

#### GQA (Grouped Query Attention)
- ✅ Supported when n_heads % n_kv_heads == 0
- ✅ Tested: 14 heads / 2 KV heads (Qwen2.5-0.5B)
- ⚠️ Different GQA configurations not extensively tested

#### RoPE (Rotary Position Embeddings)
- ✅ Supported for Qwen2-style RoPE
- ⚠️ Other RoPE implementations not tested

## Documentation Maintenance

This document should be updated when:
1. New limitations are discovered
2. Existing limitations are resolved
3. New features are added with constraints
4. Platform support is expanded

## For Developers

When adding new features:
1. Document any limitations immediately
2. Add tests for edge cases
3. Update this document if constraints affect users
4. Mark experimental features clearly

## Reporting Issues

If you encounter a limitation not listed here:
1. Check if it's covered by existing issues
2. Search documentation for workarounds
3. File a new issue with:
   - GPU model and architecture
   - ROCm version
   - Model being tested
   - Expected vs actual behavior
   - Full error messages

---

**Note:** This document tracks known limitations. For features that are working, see the main README and CHANGELOG.md.
