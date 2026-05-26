# Batched Preflight Validation Plan

## Objective
Revalidate the corrected batched prefill path using the staged safety harness, starting from least risky checks and only advancing if each stage passes.

## Hard Constraints
- MUST use the safety harness
- Do NOT run GPU code directly except through approved wrappers/scripts
- Prefer the smallest-risk validation that can answer the question
- Stop immediately if any stage suggests instability, page fault risk, or suspicious behavior
- Keep a factual record in CHANGELOG.md only if validation actually occurs
- No overclaiming

## Required Sequence

### Stage 0: File Inspection
- [x] Inspect safety scripts (gpu_lock.sh, gpu_preflight.sh, gpu_safe_run.sh)
- [x] Check available GPU CLI QA tests (tests/gpu_cli_qa.rs)
- [x] Confirm validation model exists (qwen2.5-0.5b-instruct-q4_0.gguf, 352MB)
- [x] Check git status for batched prefill changes

### Stage 1: Safety Infrastructure Check
- [ ] Run: `./scripts/gpu_lock.sh status`
- [ ] Verify lock is available or can be acquired

### Stage 2: GPU Preflight
- [ ] Run: `./scripts/gpu_preflight.sh`
- [ ] Verify all 4 checks pass:
  - [ ] Render node presence
  - [ ] ROCm/HIP runtime device visibility
  - [ ] Memory round-trip
  - [ ] Trivial kernel launch

### Stage 3: Minimal CLI QA (Subprocess-Isolated)
- [ ] Run: `./scripts/gpu_safe_run.sh --timeout 30 --max-tokens 5 ./target/release/rocmforge --gpu --model <path> --prompt "Hi" --no-template`
- [ ] Uses smallest available model (qwen2.5-0.5b-instruct-q4_0.gguf)
- [ ] Verify: No crashes, no page faults, clean exit
- [ ] Check output is non-empty and looks coherent

### Stage 4: Slightly Larger Prompt (If Stage 3 Passes)
- [ ] Run: `./scripts/gpu_safe_run.sh --timeout 60 --max-tokens 10 ./target/release/rocmforge --gpu --model <path> --prompt "Hello world" --no-template`
- [ ] Verify: Still stable, output looks reasonable

### Stage 5: Confirm Batched Prefill Exercised
- [ ] Check if the test actually triggered batched prefill path
- [ ] Look for "batched prefill" in debug output or logs
- [ ] Verify VRAM headroom gate is working (5 GiB reserved)

### Stage 6: Throughput Observation (If All Stages Pass)
- [ ] Measure rough tok/s from Stage 4 output
- [ ] Check output coherence (does it make sense?)
- [ ] Document findings

### Stage 7: CHANGELOG Update (If Validation Occurred)
- [ ] Only update if validation actually occurred
- [ ] Record factual observations
- [ ] No speculation or overclaiming

## Files Changed (To Be Determined)
- Check git status to see what files were modified for batched prefill correction

## Validation Priorities
1. **First question**: Does corrected batched prefill run without immediate instability?
2. **Second question**: Does the output look coherent enough to show the path is not obviously corrupt?
3. **Third question**: What rough tok/s do we observe under the safe wrapper?

## Success Criteria
- All safety checks pass
- No crashes, page faults, or GPU resets
- Output is coherent and non-empty
- Batched prefill path was actually exercised
- CHANGELOG updated factually with observed results

## Abort Conditions
- Any safety check fails
- GPU lock timeout or stale lock detected
- Preflight checks fail
- CLI execution crashes or times out
- Output is empty or obviously corrupted
- GPU instability detected (artifacts, resets, etc.)
