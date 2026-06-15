# Reranker Token-Level Accuracy Evaluation (Option 1)

Model: Qwen2.5-0.5B-Instruct (Q4_0)

Value head: deterministic `score = hidden[0]`

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
| The capital of France is | ` Paris` | `( ), which is the` ✗ | `( ), which is the` ✗ | `( ), which is the` ✗ | `Paris.` ✓ | `Paris.` ✓ | `Paris.` ✓ |
| The color of the sky on a clear day is | ` blue` | `produced by which of the` ✗ | `produced by which of the` ✗ | `produced by which of the` ✗ | `usually ____ than on an` ✗ | `a colorless light source` ✗ | `usually ____ than on an` ✗ |
| 2 + 2 equals | ` 4` | `the sum of the squares` ✗ | `the sum of the squares` ✗ | `the sum of the squares` ✗ | `?` ✗ | `14.` ✗ | `?` ✗ |
| The first month of the year is | ` January` | `:` ✗ | `:` ✗ | `:` ✗ | `divided evenly among seven days` ✗ | `considered "good," while` ✗ | `divided evenly among seven days` ✗ |
| The opposite of hot is | ` cold` | `:` ✗ | `:` ✗ | `:` ✗ | `____.` ✗ | `____.` ✗ | `____.` ✗ |
| A circle has 360 | ` degrees` | `street blocks and is divided` ✗ | `street blocks and is divided` ✗ | `street blocks and is divided` ✗ | `pieces of candies and` ✗ | `pieces of candies and` ✗ | `pieces of candies and` ✗ |
| Count: one two three four five six seven eight nine | ` ten` | `high: six seven eight` ✗ | `high: six seven eight` ✗ | `high: six seven eight` ✗ | `ten and ten ten twenty` ✓ | `ten one ten two ten` ✓ | `ten and ten ten twenty` ✓ |
| The largest planet in our solar system is | ` Jupiter` | `half the size of Earth` ✗ | `half the size of Earth` ✗ | `half the size of Earth` ✗ | `known for its high surface` ✗ | `known for producing more greenhouse` ✗ | `known for its high surface` ✗ |
| The chemical symbol for water is H2 | `O` | `O. If the number` ✓ | `O. If the number` ✓ | `O. If the number` ✓ | `. Is the following equation` ✗ | `0．（） �` ✗ | `. Is the following equation` ✗ |
| The freezing point of water is 0 degrees | ` Celsius` | `Celsius. The freezing point` ✓ | `Celsius. The freezing point` ✓ | `Celsius. The freezing point` ✓ | `and boiling of the fluid` ✗ | `and boiling point of the` ✗ | `and boiling of the fluid` ✗ |

## Observations

- This is a synthetic value head, not trained on quality signals, so the results measure the *mechanical effect* of the reranker/beam, not a real quality improvement.
- A real evaluation requires either a trained value head or a model-based judge.
