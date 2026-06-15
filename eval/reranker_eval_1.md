# Reranker Token-Level Accuracy Evaluation (Option 1)

Model: Qwen2.5-0.5B-Instruct (Q4_0)

Value head: trained value head loaded from `/home/feanor/Projects/rocmforge/target/trivia_value_head.bin`

Dataset: `eval/rerank_trivia.jsonl`

## Summary

| config | first-token accuracy | continuation-prefix accuracy |
|--------|----------------------|-------------------------------|
| baseline | 20.0% | 20.0% |
| rerank-d1 | 20.0% | 20.0% |
| rerank-d2 | 20.0% | 20.0% |
| beam-w2-d1 | 30.0% | 30.0% |
| beam-w2-d2 | 30.0% | 30.0% |
| beam-w2-d1-lp0.5 | 30.0% | 30.0% |

## Per-sample results

| prompt | expected | baseline | rerank-d1 | rerank-d2 | beam-w2-d1 | beam-w2-d2 | beam-w2-d1-lp0.5 |
|--------|----------|----------|----------|----------|----------|----------|----------|
| The capital of France is | ` Paris` | `( ), which is the` ✗ | `( ), which is the` ✗ | `( ), which is the` ✗ | `Paris and it is located` ✓ | `Paris and its official name` ✓ | `Paris and it is located` ✓ |
| The color of the sky on a clear day is | ` blue` | `produced by which of the` ✗ | `produced by which of the` ✗ | `produced by which of the` ✗ | `usually blue or blueish` ✗ | `often blue or blueish` ✗ | `usually blue or blueish` ✗ |
| 2 + 2 equals | ` 4` | `the sum of the squares` ✗ | `the sum of the squares` ✗ | `the sum of the squares` ✗ | `what number when rounded up` ✗ | `1000` ✗ | `what number when rounded up` ✗ |
| The first month of the year is | ` January` | `:` ✗ | `:` ✗ | `:` ✗ | `divided evenly, each part` ✗ | `a busy one as it` ✗ | `divided evenly, each part` ✗ |
| The opposite of hot is | ` cold` | `:` ✗ | `:` ✗ | `:` ✗ | `______.` ✗ | `______.` ✗ | `______.` ✗ |
| A circle has 360 | ` degrees` | `street blocks and is divided` ✗ | `street blocks and is divided` ✗ | `street blocks and is divided` ✗ | `degrees, divided evenly to` ✓ | `degrees of circumference, which` ✓ | `degrees, divided evenly to` ✓ |
| Count: one two three four five six seven eight nine | ` ten` | `high: six seven eight` ✗ | `high: six seven eight` ✗ | `high: six seven eight` ✗ | `one zero One Two Thirty` ✗ | `one zero One Two Thirty` ✗ | `one zero One Two Thirty` ✗ |
| The largest planet in our solar system is | ` Jupiter` | `half the size of Earth` ✗ | `half the size of Earth` ✗ | `half the size of Earth` ✗ | `approximately what fraction distance across` ✗ | `known by what common word` ✗ | `approximately what fraction distance across` ✗ |
| The chemical symbol for water is H2 | `O` | `O. If the number` ✓ | `O. If the number` ✓ | `O. If the number` ✓ | `O and for sodium oxide` ✓ | `SO4 and for iron` ✗ | `O and for sodium oxide` ✓ |
| The freezing point of water is 0 degrees | ` Celsius` | `Celsius. The freezing point` ✓ | `Celsius. The freezing point` ✓ | `Celsius. The freezing point` ✓ | `Fahrenheit or Celsius, which` ✗ | `Celsius (0 degrees Kelvin` ✓ | `Fahrenheit or Celsius, which` ✗ |

## Observations

- This run uses a value head trained on temperature-sampled completions from the same trivia dataset, labeled by exact-match correctness.
- Beam search now receives a real quality signal, and the accuracy change versus greedy is a meaningful measurement of the reranker's utility.
