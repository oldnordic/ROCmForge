# GPU Sequential Token Corruption - ROOT CAUSE FOUND

**Date:** 2026-04-17
**Status:** 🔴 ROOT CAUSE IDENTIFIED - K values corrupted during KV-cache write

## Smoking Gun: K Values Corrupted During Cache Write

### Diagnostic Data: Before vs After Cache Write

**K Values BEFORE Cache Write (pos=1, local K buffer):**
```
[-8.804068, -3.4432926, -6.238553, 0.6017397, -0.14521708, 9.352216, 8.023085, -1.3291036, -0.12585078, -0.24078582]
```

**K Values AFTER Cache Write (pos=1, read from cache):**
```
[-10.270655, -8.263332, -7.3779364, 2.4629862, -0.12994537, 9.456434, 8.115107, -1.7641096, -0.12956794, -0.24023396]
```

### Value-by-Value Comparison:
| Index | Before Cache | After Cache | Difference |
|-------|--------------|-------------|------------|
| 0 | -8.804068 | -10.270655 | -1.466587 ❌ |
| 1 | -3.4432926 | -8.263332 | -4.820039 ❌ |
| 2 | -6.238553 | -7.3779364 | -1.139383 ❌ |
| 3 | 0.6017397 | 2.4629862 | 1.8612465 ❌ |

**V Values:** Unchanged (before/after cache match) ✅

## Root Cause

**K values are corrupted during KV-cache write operation.**

This explains why:
- Token 1 (pos=1) produces correct output: Uses local K buffer, not cached K
- Token 2+ (pos=2+) produces corrupted output: Uses corrupted K from cache

## Next Step

Examine `kv_write_rope_from_state_on_stream` implementation to identify why K cache writes are corrupting values but V writes are not.
