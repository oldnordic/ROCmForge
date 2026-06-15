# Reranker Token-Level Accuracy Evaluation (Option 1)

Model: Qwen2.5-7B-Instruct (Q4_0)

Value head: trained process-reward head loaded from `/home/feanor/Projects/rocmforge/target/trivia_prm_7b_head.bin`

Dataset: `eval/rerank_trivia.jsonl`

## Summary

| config | first-token accuracy | continuation-prefix accuracy |
|--------|----------------------|-------------------------------|
| baseline | 30.0% | 30.0% |
| rerank-d1 | 30.0% | 30.0% |
| rerank-d2 | 30.0% | 30.0% |
| beam-w2-d1 | 70.0% | 70.0% |
| beam-w2-d2 | 60.0% | 60.0% |
| beam-w2-d1-lp0.5 | 70.0% | 70.0% |

## Per-sample results

| prompt | expected | baseline | rerank-d1 | rerank-d2 | beam-w2-d1 | beam-w2-d2 | beam-w2-d1-lp0.5 |
|--------|----------|----------|----------|----------|----------|----------|----------|
| The capital of France is | ` Paris` | `called Paris, and the` ✗ | `called Paris, and the` ✗ | `called Paris, and the` ✗ | `a city with a population` ✗ | `a city that is located` ✗ | `a city with a population` ✗ |
| The color of the sky on a clear day is | ` blue` | `best described as:` ✗ | `best described as:` ✗ | `best described as:` ✗ | `blue.` ✓ | `a common topic that people` ✗ | `blue.` ✓ |
| 2 + 2 equals | ` 4` | `$4. Therefore,` ✗ | `$4. Therefore,` ✗ | `$4. Therefore,` ✗ | `2222` ✗ | `2222` ✗ | `2222` ✗ |
| The first month of the year is | ` January` | `named after a Roman god` ✗ | `named after a Roman god` ✗ | `named after a Roman god` ✗ | `January, the second January` ✓ | `January, the last month` ✓ | `January, the second January` ✓ |
| The opposite of hot is | ` cold` | `A. Freezing B` ✗ | `A. Freezing B` ✗ | `A. Freezing B` ✗ | `cold．（改为一般` ✓ | `cold．（改为一般` ✓ | `cold．（改为一般` ✓ |
| A circle has 360 | ` degrees` | `°. If an arc` ✗ | `°. If an arc` ✗ | `°. If an arc` ✗ | `° and 1/` ✗ | `° in total, so` ✗ | `° and 1/` ✗ |
| Count: one two three four five six seven eight nine | ` ten` | `ten eleven twelve thirteen fourteen` ✓ | `ten eleven twelve thirteen fourteen` ✓ | `ten eleven twelve thirteen fourteen` ✓ | `ten eleven twelve thirteen fourteen` ✓ | `ten eleven twelve thirteen fourteen` ✓ | `ten eleven twelve thirteen fourteen` ✓ |
| The largest planet in our solar system is | ` Jupiter` | `certainly a grand sight to` ✗ | `certainly a grand sight to` ✗ | `certainly a grand sight to` ✗ | `Jupiter with a mass approximately` ✓ | `Jupiter with a mass approximately` ✓ | `Jupiter with a mass approximately` ✓ |
| The chemical symbol for water is H2 | `O` | `O. True or False` ✓ | `O. True or False` ✓ | `O. True or False` ✓ | `O, with H representing` ✓ | `O, with H representing` ✓ | `O, with H representing` ✓ |
| The freezing point of water is 0 degrees | ` Celsius` | `Celsius, which is` ✓ | `Celsius, which is` ✓ | `Celsius, which is` ✓ | `Celsius, and this temperature` ✓ | `Celsius, and this temperature` ✓ | `Celsius, and this temperature` ✓ |

## Observations

- This run uses a value head trained on temperature-sampled completions from the same trivia dataset, labeled by exact-match correctness.
- Beam search now receives a real quality signal, and the accuracy change versus greedy is a meaningful measurement of the reranker's utility.
