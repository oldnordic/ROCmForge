use rocmforge::tokenizer::Tokenizer;

#[derive(Debug, Clone, PartialEq)]
struct LogitHealth {
    nan_count: usize,
    inf_count: usize,
    min: f32,
    max: f32,
    mean: f32,
}

fn inspect_logits(logits: &[f32]) -> Result<(), LogitHealth> {
    let nan_count = logits.iter().filter(|l| l.is_nan()).count();
    let inf_count = logits.iter().filter(|l| l.is_infinite()).count();
    if nan_count > 0 || inf_count > 0 {
        return Err(LogitHealth {
            nan_count,
            inf_count,
            min: logits.iter().cloned().fold(f32::INFINITY, f32::min),
            max: logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            mean: logits.iter().sum::<f32>() / logits.len() as f32,
        });
    }
    Ok(())
}

fn top_k_probabilities(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .expect("invariant: partial_cmp failed (NaN in debug logic)")
    });
    indexed.truncate(k.min(indexed.len()));
    indexed
}

pub(crate) fn print_top_k_tokens(logits: &[f32], tok: &dyn Tokenizer, k: usize) {
    if let Err(health) = inspect_logits(logits) {
        eprintln!(
            "ERROR: logits contain {} NaN and {} Inf values",
            health.nan_count, health.inf_count
        );
        eprintln!(
            "  Stats: min={:.4}, max={:.4}, mean={:.4}",
            health.min, health.max, health.mean
        );
        return;
    }

    let indexed = top_k_probabilities(logits, k);
    eprintln!("Top-{} tokens:", indexed.len());
    for (i, (id, prob)) in indexed.iter().enumerate() {
        let token = tok.decode_token(*id as u32);
        let token_display = if token.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
            token.clone()
        } else {
            format!("{:?}", token)
        };
        eprintln!("  {:2}. {:8} ({:.4}) id={}", i + 1, token_display, prob, id);
    }
}

#[cfg(test)]
mod tests {
    use super::{inspect_logits, top_k_probabilities};

    #[test]
    fn top_k_probabilities_orders_descending() {
        let top = top_k_probabilities(&[1.0, 3.0, 2.0, -1.0], 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[1].0, 2);
        assert_eq!(top[2].0, 0);
        assert!(top[0].1 >= top[1].1);
        assert!(top[1].1 >= top[2].1);
    }

    #[test]
    fn inspect_logits_reports_nonfinite_values() {
        let err = inspect_logits(&[1.0, f32::NAN, f32::INFINITY]).unwrap_err();
        assert_eq!(err.nan_count, 1);
        assert_eq!(err.inf_count, 1);
        assert_eq!(err.min, 1.0);
        assert_eq!(err.max, f32::INFINITY);
        assert!(err.mean.is_nan());
    }
}
