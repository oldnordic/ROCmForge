---
phase: 01-foundation
plan: 04
subsystem: config
tags: [config, model-traits, chat-template, metadata-driven]
requires: [01-01, 01-02, 01-03]
provides: [ModelConfig, ModelTraits, RopeStyle, AttentionLayout, ChatTemplate, ConfigError]
affects: [inference, tokenizer]
key_decisions:
  - vocab_size from tokenizer tokens length, not GGUF metadata key
  - intermediate_size inferred from tensor shape when metadata missing
  - Unknown architectures fall back to LLaMA defaults
  - Qwen2 uses NeoX RoPE style with 1M default theta
tech_stack:
  added: [thiserror-pattern for ConfigError, OnceLock registry]
  patterns: [registry pattern, metadata-driven config, trait-based architecture dispatch]
key_files:
  created: []
  modified:
    - path: src/config.rs
      lines_added: 572
      exports: [ModelConfig, ModelTraits, RopeStyle, AttentionLayout, ChatTemplate, ConfigError, detect_chat_template]
duration: PT5M
completed: 2026-03-24T11:26:32Z
---

# Phase 1 Plan 4: ModelConfig and ModelTraits Summary

## One-liner

Model configuration module with architecture-specific traits registry, GGUF-derived ModelConfig, and ChatTemplate detection for Qwen2.5 and other model families.

## What Was Done

### Task 1: Enums and ModelTraits Registry

- Implemented `RopeStyle` enum (Normal/NeoX) for RoPE rotation patterns
- Implemented `AttentionLayout` enum (SplitQkv/FusedQkv) for tensor layout
- Implemented `ModelTraits` struct with architecture-specific behavioral traits
- Created registry with entries for:
  - **Qwen2 family**: NeoX RoPE, attention bias, 1M rope theta, 1e-6 norm eps
  - **LLaMA family**: Normal RoPE, no bias, 10K rope theta
  - **Phi3**: FusedQkv layout, Normal RoPE
  - **Gemma family**: 1e-6 norm eps
  - **GLM**: NeoX RoPE, FusedQkv
- `ModelTraits::for_arch()` falls back to LLaMA defaults for unknown architectures

### Task 2: ModelConfig::from_gguf and Validation

- Implemented `ModelConfig` struct with all inference hyperparameters
- `from_gguf()` derives all values from GGUF metadata + traits registry
- **Critical**: `vocab_size` from `tokenizer_data.tokens.len()` (not GGUF key - Qwen2.5 reports 0)
- `intermediate_size`: tries metadata first, then infers from tensor shape
- `validate()` checks all required fields are non-zero and GQA validity
- Implemented `ConfigError` enum with Display and Error traits

### Task 3: ChatTemplate Enum and Detection

- Implemented `ChatTemplate` enum (None, ChatML, LLaMA3, LLaMA2, Phi3, Gemma)
- `apply()` method formats user text in appropriate prompt format
- `detect_chat_template()` uses architecture string + tokenizer model type
- LLaMA2 vs LLaMA3 distinction based on BPE vs SentencePiece tokenizer
- `name()` method for human-readable logging

## Deviations from Plan

None - plan executed exactly as written. All implementation followed the Memoria reference with clean port to rocmforge patterns.

## Files Modified

| File | Lines | Description |
|------|-------|-------------|
| src/config.rs | +572 | Full config module implementation |

## Test Results

All 15 config tests pass:

```
test config::tests::chatml_apply_wraps_correctly ... ok
test config::tests::llama2_apply ... ok
test config::tests::none_apply_passthrough ... ok
test config::tests::template_gemma ... ok
test config::tests::template_llama2_detected_by_spm ... ok
test config::tests::template_llama3_detected_by_bpe ... ok
test config::tests::template_phi3 ... ok
test config::tests::template_qwen2_is_chatml ... ok
test config::tests::template_unknown_arch_is_none ... ok
test config::tests::traits_llama_normal ... ok
test config::tests::traits_phi3_fused_qkv ... ok
test config::tests::traits_qwen2_neox ... ok
test config::tests::traits_unknown_falls_back_to_llama ... ok
test config::tests::validation_rejects_bad_gqa ... ok
test config::tests::validation_rejects_zero_layers ... ok
```

## Key Decisions

1. **vocab_size from tokens**: Uses `file.tokenizer_data().tokens.len()` because Qwen2.5 GGUF reports 0 in the `vocab_size` key (per D-05 from CONTEXT.md)

2. **intermediate_size inference**: When `feed_forward_length` metadata is 0, inspects `blk.0.ffn_gate.weight` tensor shape to infer the dimension

3. **Registry fallback**: Unknown architectures fall back to LLaMA defaults rather than failing, enabling graceful handling of new model families

4. **Qwen2 traits**: High rope theta (1,000,000), NeoX style, attention bias enabled - matching Qwen2.5 requirements

## Integration Points

- `ModelConfig::from_gguf(&GgufFile)` - bridges loader module to config
- `file.tokenizer_data()` - provides vocab for size calculation
- `file.tensor()` - provides tensor shapes for intermediate_size inference

## Commit

```
e5493df feat(01-04): implement ModelConfig, ModelTraits registry, and ChatTemplate
```

## Self-Check: PASSED

- config.rs exists and contains 572 lines
- Commit e5493df exists in git history
- SUMMARY.md exists at expected path
- All 33 library tests pass
