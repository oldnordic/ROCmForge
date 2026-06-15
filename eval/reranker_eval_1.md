# Reranker Token-Level Accuracy Evaluation (Option 1)

Model: Qwen2.5-0.5B-Instruct (Q4_0)

Value head: trained value head loaded from `/home/feanor/Projects/rocmforge/target/trivia_prm_head.bin`

Dataset: `eval/rerank_trivia.jsonl`

## Summary

| config | first-token accuracy | continuation-prefix accuracy |
|--------|----------------------|-------------------------------|
| baseline | 20.0% | 20.0% |
| rerank-d1 | 20.0% | 20.0% |
| rerank-d2 | 20.0% | 20.0% |
| beam-w2-d1 | 20.0% | 20.0% |
| beam-w2-d2 | 20.0% | 20.0% |
| beam-w2-d1-lp0.5 | 20.0% | 20.0% |

## Per-sample results

| prompt | expected | baseline | rerank-d1 | rerank-d2 | beam-w2-d1 | beam-w2-d2 | beam-w2-d1-lp0.5 |
|--------|----------|----------|----------|----------|----------|----------|----------|
| The capital of France is | ` Paris` | `( ), which is the` ✗ | `( ), which is the` ✗ | `( ), which is the` ✗ | `the city called Paris.` ✗ | `the city called Paris.` ✗ | `the city called Paris.` ✗ |
| The color of the sky on a clear day is | ` blue` | `produced by which of the` ✗ | `produced by which of the` ✗ | `produced by which of the` ✗ | `always blue and never a` ✗ | `the same color on a` ✗ | `always blue and never a` ✗ |
| 2 + 2 equals | ` 4` | `the sum of the squares` ✗ | `the sum of the squares` ✗ | `the sum of the squares` ✗ | `? Let me think out` ✗ | `? Let me think out` ✗ | `? Let me think out` ✗ |
| The first month of the year is | ` January` | `:` ✗ | `:` ✗ | `:` ✗ | `the start year. What` ✗ | `a leap one. What` ✗ | `the start year. What` ✗ |
| The opposite of hot is | ` cold` | `:` ✗ | `:` ✗ | `:` ✗ | `____.` ✗ | `______.` ✗ | `____.` ✗ |
| A circle has 360 | ` degrees` | `street blocks and is divided` ✗ | `street blocks and is divided` ✗ | `street blocks and is divided` ✗ | `degree in its degree and` ✗ | `° in its radian` ✗ | `degree in its degree and` ✗ |
| Count: one two three four five six seven eight nine | ` ten` | `high: six seven eight` ✗ | `high: six seven eight` ✗ | `high: six seven eight` ✗ | `ten one twenty thirty fourteen` ✓ | `ten` ✓ | `ten one twenty thirty fourteen` ✓ |
| The largest planet in our solar system is | ` Jupiter` | `half the size of Earth` ✗ | `half the size of Earth` ✗ | `half the size of Earth` ✗ | `approximately what fraction larger or` ✗ | `known by what scientific classification` ✗ | `approximately what fraction larger or` ✗ |
| The chemical symbol for water is H2 | `O` | `O. If the number` ✓ | `O. If the number` ✓ | `O. If the number` ✓ | `O and represents the compound` ✓ | `O and the formula is` ✓ | `O and represents the compound` ✓ |
| The freezing point of water is 0 degrees | ` Celsius` | `Celsius. The freezing point` ✓ | `Celsius. The freezing point` ✓ | `Celsius. The freezing point` ✓ | `Fahrenheit and the boiling point` ✗ | `and boiling of mercury at` ✗ | `Fahrenheit and the boiling point` ✗ |

## Observations

- This run uses a value head trained on temperature-sampled completions from the same trivia dataset, labeled by exact-match correctness.
- Beam search now receives a real quality signal, and the accuracy change versus greedy is a meaningful measurement of the reranker's utility.
