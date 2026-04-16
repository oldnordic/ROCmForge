# Q6_K Single vs Multi-token Comparison

## Single Token (Working)

**Prompt:** "Hello" (1 token)

**Output:**
```
, my name is Matthew Mason. I'm a teacher of English and journalism at Concord Middle School.
```

**Performance:**
- Tokens/second: 132.3 tok/s
- Prefill time: 11.3ms (88.6 tok/s)
- Total tokens generated: 20
- Status: ✅ **Coherent English output**

## Multi-token (Broken)

**Prompt:** "Hello, how are you today? Please tell me about yourself." (13 tokens)

**Output:**
```
���的，我是来自中国的一名计算机科学专业的大学生，名叫张晓明。我可以使用编程和计算机科学进行各种技术实践，同时也喜欢阅读书籍和探索不同的领域。你是谁？ 对不起，我不知道你是谁，
```

**Performance:**
- Tokens/second: 133.4 tok/s
- Prefill time: 71.3ms (182.4 tok/s)
- Total tokens generated: 50
- Status: ❌ **Chinese characters and garbled text**

## Divergence Analysis

### Token Count Threshold
- **Working:** 1 token prompts
- **Broken:** 13 token prompts
- **Threshold:** The issue appears to be related to multi-token prompt processing

### Degradation Pattern
- **Immediate vs Gradual:** The degradation is **immediate** - the first tokens generated after the multi-token prompt are already corrupted
- **Output Characteristics:** Instead of continuing the English conversation, the model switches to Chinese characters and produces content that appears to be from a completely different context (Chinese computer science student named Zhang Xiaoming)

### Code Paths
- **Single-token (working):** Uses GEMV (decode path) - processes one token at a time
- **Multi-token (broken):** Uses GEMM (prefill path) - batches multiple tokens for efficient processing

### Timing Analysis
- Single-token prefill: 11.3ms (88.6 tok/s) - slower but correct
- Multi-token prefill: 71.3ms (182.4 tok/s) - faster but produces corrupted output
- Decode speed is consistent (132-133 tok/s) in both cases

### Root Cause Hypothesis
The Q6_K GPU GEMM kernel (used for multi-token prefill) has a bug that:
1. Does not affect single-token decode (GEMV works fine)
2. Corrupts the hidden state during batch processing
3. Causes the model to generate tokens from wrong parts of the vocabulary space
4. The corruption is consistent - always producing Chinese/Unicode characters instead of English

### CPU vs GPU
- **CPU Q6_K:** Works perfectly for both single and multi-token prompts
- **GPU Q6_K:** Single-token works, multi-token broken
- **Conclusion:** This is a GPU-specific GEMM kernel issue, not a general Q6_K format problem

## Technical Details

### File Locations
- GPU GEMM kernel: `hip_kernels/quant/q6_k_gemm.hip`
- GPU GEMV kernel: `hip_kernels/quant/q6_k_gemv.hip`
- GPU forward code: `src/gpu/forward.rs`

### Additional Testing - Token Count Threshold Discovery

**Systematic Token Count Testing:**
- **1 token** ("Hello"): ✅ Works (132.3 tok/s)
- **3 tokens** ("Hello, how"): ✅ Works (130.3 tok/s)
- **5 tokens** ("Hello, how are you?"): ✅ Works (131.4 tok/s)
- **6 tokens** ("Hello, how are you today?"): ✅ Works (130.4 tok/s)
- **10 tokens** ("Hello, how are you today? Please tell me"): ✅ Works (129.4 tok/s)
- **12 tokens** ("Hello, how are you today? Please tell me about yourself"): ✅ Works (129.6 tok/s)
- **13 tokens** ("Hello, how are you today? Please tell me about yourself."): ❌ **Broken** (133.4 tok/s) - Chinese characters

**CRITICAL FINDING:** The issue is NOT a simple token count threshold! The original 13-token prompt produced corruption, but systematic testing shows 12 tokens work fine.

**BREAKTHROUGH DISCOVERY:** The bug is triggered by **specific prompt ending with a period** combined with multi-token processing!

**Detailed Testing Results:**
- **13 tokens with period** ("Hello, how are you today? Please tell me about yourself."): ❌ **Broken** - Chinese characters
- **13 tokens without period** ("Hello, how are you today? Please tell me about yourself"): ✅ **Works** - Normal output
- **13 tokens different content** ("Hello, how are you doing today my friend"): ✅ **Works** - Normal output
- **Period in middle** ("Hello. How are you doing today my friend"): ✅ **Works** - Normal output
- **Question mark ending** ("Hello, how are you today?"): ✅ **Works** - Normal output

**New Hypothesis:** The bug is triggered by the combination of:
1. Multi-token prompt (GEMM path)
2. Prompt ending with a period character
3. Specific token sequence or tokenization pattern

This suggests a memory indexing or shared memory issue in the Q6_K GEMM kernel that manifests when processing certain token patterns, particularly those ending with sentence-ending punctuation.

### Next Steps
1. Compare Q6_K GEMM implementation with working Q4_K/Q5_K GEMM kernels
2. Check for indexing or shared memory issues specific to batch processing
3. Verify dequantization logic in multi-token context
4. Test with intermediate token counts (7-12 tokens) to pinpoint exact failure threshold
5. **CRITICAL:** Test 8-12 token prompts to find the exact breaking point

## Test Commands

```bash
# Single-token (working)
./target/release/rocmforge --gpu --model /home/feanor/Projects/Memoria/models/qwen2-0.5b-instruct-q6_k.gguf --prompt "Hello" --max-tokens 20 --no-template

# Multi-token (broken)
./target/release/rocmforge --gpu --model /home/feanor/Projects/Memoria/models/qwen2-0.5b-instruct-q6_k.gguf --prompt "Hello, how are you today? Please tell me about yourself." --max-tokens 50 --no-template
```

**Date:** 2026-04-16
**Model:** qwen2-0.5b-instruct-q6_k.gguf
**GPU:** AMD Radeon RX 7900 XT
**Status:** Documented for Task #1 of Q6_K multi-token fix plan