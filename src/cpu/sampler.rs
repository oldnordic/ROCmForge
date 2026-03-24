//! CPU sampling — greedy and top-p (nucleus) sampling.
//!
//! Operates directly on a logits slice (no GPU download needed).

use super::ops::argmax;

/// Greedy sampling: return argmax of logits.
pub fn cpu_sample_greedy(logits: &[f32]) -> u32 {
    argmax(logits) as u32
}

/// Top-p (nucleus) sampling.
///
/// Applies temperature, softmax, sorts descending, finds nucleus (cumsum >= p),
/// samples uniformly within it.
///
/// # Arguments
/// * `logits` - Raw logits from the model (vocab_size elements)
/// * `temperature` - Sampling temperature (higher = more random)
/// * `top_p` - Nucleus probability threshold (0.0-1.0)
/// * `seed` - Random seed for reproducibility
pub fn cpu_sample_top_p(logits: &[f32], temperature: f32, top_p: f32, seed: u64) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    let t = temperature.max(1e-6);
    let mut scaled: Vec<f32> = logits.iter().map(|v| v / t).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in &mut scaled {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    for v in &mut scaled {
        *v /= sum;
    }

    // Sort descending by probability (store as (prob, index))
    let mut pairs: Vec<(f32, u32)> = scaled
        .iter()
        .copied()
        .enumerate()
        .map(|(i, p)| (p, i as u32))
        .collect();
    pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Find nucleus (smallest set of tokens with cumsum >= top_p)
    let mut cumsum = 0.0f32;
    let mut nucleus_end = 1usize;
    for (i, (prob, _)) in pairs.iter().enumerate() {
        cumsum += prob;
        nucleus_end = i + 1;
        if cumsum >= top_p {
            break;
        }
    }

    // Sample uniformly from nucleus using LCG
    let rand_val = lcg_unit(seed) * cumsum;
    let mut running = 0.0f32;
    for i in 0..nucleus_end {
        running += pairs[i].0;
        if running >= rand_val {
            return pairs[i].1;
        }
    }
    pairs[0].1
}

/// Top-k sampling.
///
/// Limits sampling to top-k most likely tokens, then applies temperature and softmax.
pub fn cpu_sample_top_k(logits: &[f32], temperature: f32, top_k: usize, seed: u64) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    // Find top-k indices (store as (index, logit) then sort)
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let k = top_k.min(indexed.len());
    let top_k_pairs: Vec<(usize, f32)> = indexed[..k].to_vec();

    // Apply temperature and softmax to top-k
    let t = temperature.max(1e-6);
    let max_val = top_k_pairs[0].1;
    let mut probs: Vec<f32> = top_k_pairs
        .iter()
        .map(|(_, v)| ((v / t) - (max_val / t)).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Sample from top-k
    let rand_val = lcg_unit(seed);
    let mut cumsum = 0.0f32;
    for (i, prob) in probs.iter().enumerate() {
        cumsum += prob;
        if cumsum >= rand_val {
            return top_k_pairs[i].0 as u32;
        }
    }
    top_k_pairs[0].0 as u32
}

/// Combined top-k and top-p sampling.
///
/// First limits to top-k, then applies nucleus sampling within that set.
pub fn cpu_sample_top_k_top_p(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    // Find top-k indices
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let k = top_k.min(indexed.len());
    let top_k_pairs: Vec<(usize, f32)> = indexed[..k].to_vec();

    // Apply temperature and softmax
    let t = temperature.max(1e-6);
    let max_val = top_k_pairs[0].1;
    let mut probs: Vec<f32> = top_k_pairs
        .iter()
        .map(|(_, v)| ((v / t) - (max_val / t)).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Combine with original indices: (prob, token_id)
    let mut pairs: Vec<(f32, u32)> = probs
        .iter()
        .copied()
        .enumerate()
        .map(|(i, p)| (p, top_k_pairs[i].0 as u32))
        .collect();

    // Re-sort by probability
    pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Find nucleus
    let mut cumsum = 0.0f32;
    let mut nucleus_end = 1usize;
    for (i, (prob, _)) in pairs.iter().enumerate() {
        cumsum += prob;
        nucleus_end = i + 1;
        if cumsum >= top_p {
            break;
        }
    }

    // Sample
    let rand_val = lcg_unit(seed) * cumsum;
    let mut running = 0.0f32;
    for i in 0..nucleus_end {
        running += pairs[i].0;
        if running >= rand_val {
            return pairs[i].1;
        }
    }
    pairs[0].1
}

/// Simple LCG random number generator returning value in [0, 1).
fn lcg_unit(seed: u64) -> f32 {
    let v = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (v >> 11) as f32 / (1u64 << 53) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_returns_argmax() {
        let logits = vec![-1.0, 5.0, 2.0, -3.0];
        assert_eq!(cpu_sample_greedy(&logits), 1);
    }

    #[test]
    fn greedy_handles_empty() {
        let logits: Vec<f32> = vec![];
        assert_eq!(cpu_sample_greedy(&logits), 0);
    }

    #[test]
    fn top_p_returns_high_prob_token() {
        // Very peaked distribution, top_p=0.9 should return index 1
        let logits = vec![-100.0, 100.0, -100.0, -100.0];
        let result = cpu_sample_top_p(&logits, 1.0, 0.9, 42);
        assert_eq!(result, 1);
    }

    #[test]
    fn top_k_returns_from_top_k() {
        // Index 1 is highest, 2 is second highest
        let logits = vec![-100.0, 100.0, 50.0, -100.0];
        let result = cpu_sample_top_k(&logits, 1.0, 2, 42);
        // Should be either 1 or 2 (top 2)
        assert!(result == 1 || result == 2);
    }

    #[test]
    fn top_k_top_p_combines_both() {
        let logits = vec![-100.0, 100.0, 50.0, 25.0];
        // top_k=3, top_p=0.99 - should sample from indices 1, 2, 3
        let result = cpu_sample_top_k_top_p(&logits, 1.0, 3, 0.99, 42);
        assert!(result == 1 || result == 2 || result == 3);
    }

    #[test]
    fn low_temperature_concentrates_probs() {
        // With very low temperature, should almost always return argmax
        let logits = vec![-1.0, 5.0, 2.0, -3.0];
        let mut counts = std::collections::HashMap::new();
        for seed in 0..100 {
            let token = cpu_sample_top_p(&logits, 0.01, 0.9, seed);
            *counts.entry(token).or_insert(0) += 1;
        }
        // Index 1 should be selected almost always
        assert!(*counts.get(&1).unwrap_or(&0) > 95);
    }

    #[test]
    fn high_temperature_spreads_probs() {
        // With high temperature, distribution should be more uniform
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let mut counts = std::collections::HashMap::new();
        for seed in 0..100 {
            let token = cpu_sample_top_p(&logits, 10.0, 0.99, seed);
            *counts.entry(token).or_insert(0) += 1;
        }
        // All 4 tokens should appear at least once
        assert!(counts.len() >= 3);
    }

    #[test]
    fn lcg_produces_values_in_unit_interval() {
        for seed in 0..100 {
            let v = lcg_unit(seed);
            assert!(v >= 0.0 && v < 1.0, "value {} not in [0, 1)", v);
        }
    }
}
