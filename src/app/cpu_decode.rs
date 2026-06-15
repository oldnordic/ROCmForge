use std::io::Write;
use std::time::Instant;

use rocmforge::config::ModelConfig;
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward},
    sampler::{cpu_sample_greedy, cpu_sample_top_p},
    weights::CpuModelWeights,
    CpuError,
};
use rocmforge::tokenizer::BpeTokenizer;

#[cfg(feature = "cpu-graph")]
use rocmforge::cpu::{
    forward::cpu_full_forward_with_ctx,
    graph::{BranchValueHead, CaptureContext, ScoreMetric},
};

use super::cli::Args;
use super::cpu_debug::{
    print_decode_token_debug, print_eos_stats, print_generation_stats, print_hidden_stats,
    print_logits_stats, print_top_logits_debug,
};

fn sample_next_token(
    logits: &[f32],
    use_greedy: bool,
    temperature: f32,
    top_p: f32,
    seed: &mut u64,
) -> u32 {
    if use_greedy {
        cpu_sample_greedy(logits)
    } else {
        *seed = seed.wrapping_add(1);
        cpu_sample_top_p(logits, temperature, top_p, *seed)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "CLI orchestration passes through many params"
)]
pub(crate) fn run_cpu_decode_loop(
    args: &Args,
    config: &ModelConfig,
    tok: &BpeTokenizer,
    weights: &CpuModelWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    use_greedy: bool,
    n_prompt: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut pos = n_prompt;
    let mut n_generated = 0usize;
    let t_gen = Instant::now();
    let mut seed = 0xdeadbeef_u64;
    let mut next_token = sample_next_token(
        &scratch.logits,
        use_greedy,
        args.temperature,
        args.top_p,
        &mut seed,
    );
    let mut hidden = vec![0.0f32; config.hidden_size];

    println!();

    loop {
        if tok.is_eog(next_token) || n_generated >= args.max_tokens {
            break;
        }

        let text = tok.decode_token(next_token);
        if args.debug {
            print_decode_token_debug(next_token, &text);
        }
        print!("{}", text);
        std::io::stdout().flush().ok();
        n_generated += 1;

        cpu_embed_token(next_token, weights, &mut hidden, config);

        if args.debug && n_generated <= 3 {
            print_hidden_stats(n_generated, next_token, &hidden);
        }

        cpu_full_forward(&mut hidden, weights, kv, scratch, pos, config)
            .map_err(|e: CpuError| format!("decode: {}", e))?;
        pos += 1;

        if args.debug && n_generated <= 3 {
            print_logits_stats(n_generated, &scratch.logits);
            print_top_logits_debug(n_generated, &scratch.logits, tok, 5);
        }

        next_token = sample_next_token(
            &scratch.logits,
            use_greedy,
            args.temperature,
            args.top_p,
            &mut seed,
        );
    }

    println!();

    if n_generated > 0 {
        let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        print_generation_stats(n_generated, gen_ms);
    } else {
        print_eos_stats();
    }

    Ok(())
}

/// Return the indices of the `k` largest logits, highest first.
#[cfg(feature = "cpu-graph")]
fn top_k_token_ids(logits: &[f32], k: usize) -> Vec<u32> {
    let k = k.min(logits.len());
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i as u32).collect()
}

/// Score a single candidate next token using `head`.
///
/// This runs a speculative decode step: it embeds `candidate`, runs one forward
/// pass from a clone of the current KV cache, and returns the value-head score
/// for the resulting hidden state.  The original `kv` and `hidden` are left
/// untouched.
#[cfg(feature = "cpu-graph")]
fn score_candidate_value(
    candidate: u32,
    hidden: &[f32],
    kv: &CpuKvCache,
    scratch: &mut CpuForwardScratch,
    weights: &CpuModelWeights,
    config: &ModelConfig,
    pos: usize,
    head: &BranchValueHead,
) -> f32 {
    let mut hidden_candidate = hidden.to_vec();
    cpu_embed_token(candidate, weights, &mut hidden_candidate, config);
    let mut kv_candidate = kv.clone();
    if cpu_full_forward(
        &mut hidden_candidate,
        weights,
        &mut kv_candidate,
        scratch,
        pos,
        config,
    )
    .is_err()
    {
        return 0.0f32;
    }
    head.predict(&scratch.normed)
}

#[cfg(feature = "cpu-graph")]
#[expect(
    clippy::too_many_arguments,
    reason = "CLI orchestration passes through many params"
)]
pub(crate) fn run_cpu_decode_loop_with_ctx(
    args: &Args,
    config: &ModelConfig,
    tok: &BpeTokenizer,
    weights: &CpuModelWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    use_greedy: bool,
    n_prompt: usize,
    ctx: &mut CaptureContext,
    score_metric: ScoreMetric,
    value_head: Option<&BranchValueHead>,
    rerank_top_k: usize,
    rerank_scale: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut pos = n_prompt;
    let mut n_generated = 0usize;
    let t_gen = Instant::now();
    let mut seed = 0xdeadbeef_u64;
    let mut next_token = sample_next_token(
        &scratch.logits,
        use_greedy,
        args.temperature,
        args.top_p,
        &mut seed,
    );
    let mut hidden = vec![0.0f32; config.hidden_size];

    println!();

    loop {
        if tok.is_eog(next_token) || n_generated >= args.max_tokens {
            break;
        }

        let text = tok.decode_token(next_token);
        if args.debug {
            print_decode_token_debug(next_token, &text);
        }
        print!("{}", text);
        std::io::stdout().flush().ok();
        n_generated += 1;

        cpu_embed_token(next_token, weights, &mut hidden, config);

        if args.debug && n_generated <= 3 {
            print_hidden_stats(n_generated, next_token, &hidden);
        }

        // Each generated token becomes its own branch/timestamp in the graph.
        ctx.timestamp = n_generated as u64;
        cpu_full_forward_with_ctx(ctx, &mut hidden, weights, kv, scratch, pos, config)
            .map_err(|e: CpuError| format!("decode: {}", e))?;

        // Record a scalar score for this branch from the output distribution.
        ctx.score_against(&scratch.logits, None, score_metric);

        pos += 1;

        if args.debug && n_generated <= 3 {
            print_logits_stats(n_generated, &scratch.logits);
            print_top_logits_debug(n_generated, &scratch.logits, tok, 5);
        }

        next_token = if let Some(head) = value_head {
            let candidates = top_k_token_ids(&scratch.logits[..config.vocab_size], rerank_top_k);
            // The speculative candidate forward would write at `pos`, so do not
            // rerank if the KV cache is already at its last valid position.
            if candidates.is_empty() || pos >= kv.max_seq_len {
                sample_next_token(
                    &scratch.logits,
                    use_greedy,
                    args.temperature,
                    args.top_p,
                    &mut seed,
                )
            } else {
                // Evaluate each candidate with the value head and bias the
                // original logits by the speculative score.
                let mut rerank_scratch = CpuForwardScratch::new(config);
                let mut biased_logits = scratch.logits[..config.vocab_size].to_vec();
                let mut candidate_scores: Vec<(u32, f32)> = Vec::with_capacity(candidates.len());
                for &candidate in &candidates {
                    let score = score_candidate_value(
                        candidate,
                        &hidden,
                        kv,
                        &mut rerank_scratch,
                        weights,
                        config,
                        pos,
                        head,
                    );
                    biased_logits[candidate as usize] += rerank_scale * score;
                    candidate_scores.push((candidate, score));
                }
                let chosen = sample_next_token(
                    &biased_logits,
                    use_greedy,
                    args.temperature,
                    args.top_p,
                    &mut seed,
                );
                if args.debug {
                    let scores_str = candidate_scores
                        .iter()
                        .map(|(id, s)| format!("{}:{:.4}", id, s))
                        .collect::<Vec<_>>()
                        .join(" ");
                    eprintln!(
                        "[Rerank] step={} candidates=[{}] chosen={}",
                        n_generated, scores_str, chosen
                    );
                }
                chosen
            }
        } else {
            sample_next_token(
                &scratch.logits,
                use_greedy,
                args.temperature,
                args.top_p,
                &mut seed,
            )
        };
    }

    println!();

    if n_generated > 0 {
        let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        print_generation_stats(n_generated, gen_ms);
    } else {
        print_eos_stats();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::sample_next_token;

    #[test]
    fn sample_next_token_greedy_uses_argmax_without_mutating_seed() {
        let mut seed = 41_u64;
        let token = sample_next_token(&[0.1, 0.9, 0.3], true, 1.0, 0.9, &mut seed);
        assert_eq!(token, 1);
        assert_eq!(seed, 41);
    }
}
