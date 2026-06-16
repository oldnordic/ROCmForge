use std::io::Write;
use std::time::Instant;

#[cfg(feature = "cpu-graph")]
use std::collections::HashMap;

use rocmforge::config::ModelConfig;
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward},
    sampler::{cpu_sample_greedy, cpu_sample_top_p},
    weights::CpuModelWeights,
    CpuError,
};
use rocmforge::tokenizer::Tokenizer;

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
    tok: &dyn Tokenizer,
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

        cpu_embed_token(next_token, weights, &mut hidden, config, Some(scratch));

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

/// Delta of a single speculative forward step: the K/V slices written at the
/// candidate position plus any conv-state changes.  This is far smaller than a
/// full KV clone and is enough to reconstruct the chosen candidate's state.
#[cfg(feature = "cpu-graph")]
struct KvCacheDelta {
    k_slices: Vec<Vec<f32>>,
    v_slices: Vec<Vec<f32>>,
    conv_state: Vec<Vec<f32>>,
}

#[cfg(feature = "cpu-graph")]
fn capture_kv_delta(child_kv: &CpuKvCache, pos: usize) -> KvCacheDelta {
    let mut k_slices = Vec::with_capacity(child_kv.num_layers);
    let mut v_slices = Vec::with_capacity(child_kv.num_layers);
    for layer in 0..child_kv.num_layers {
        k_slices.push(child_kv.k_at(layer, pos).to_vec());
        v_slices.push(child_kv.v_at(layer, pos).to_vec());
    }
    let conv_state = if child_kv.shortconv_l_cache > 1 {
        child_kv.conv_state.iter().map(|v| v.to_vec()).collect()
    } else {
        Vec::new()
    };
    KvCacheDelta {
        k_slices,
        v_slices,
        conv_state,
    }
}

#[cfg(feature = "cpu-graph")]
fn apply_kv_delta(kv: &mut CpuKvCache, delta: &KvCacheDelta, pos: usize) {
    for (layer, (k, v)) in delta.k_slices.iter().zip(delta.v_slices.iter()).enumerate() {
        kv.write_k(layer, pos, k);
        kv.write_v(layer, pos, v);
    }
    if kv.shortconv_l_cache > 1 {
        for (layer, state) in delta.conv_state.iter().enumerate() {
            kv.conv_state[layer].copy_from_slice(state);
        }
    }
}

/// Reset the speculative KV scratch to the parent state at the candidate
/// position so the next candidate starts from an identical parent.
#[cfg(feature = "cpu-graph")]
fn reset_kv_to_parent(child_kv: &mut CpuKvCache, parent_kv: &CpuKvCache, pos: usize) {
    for layer in 0..parent_kv.num_layers {
        child_kv.write_k(layer, pos, parent_kv.k_at(layer, pos));
        child_kv.write_v(layer, pos, parent_kv.v_at(layer, pos));
    }
    if parent_kv.shortconv_l_cache > 1 {
        for layer in 0..parent_kv.num_layers {
            child_kv.conv_state[layer].copy_from_slice(&parent_kv.conv_state[layer]);
        }
    }
}

#[cfg(feature = "cpu-graph")]
struct CandidateState {
    token_id: u32,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    kv_delta: KvCacheDelta,
}

/// State after a chosen candidate token has already been forwarded.  Applying
/// this on the next iteration skips the embed + main forward for that token.
#[cfg(feature = "cpu-graph")]
struct ReusableState {
    token_id: u32,
    hidden: Vec<f32>,
    kv: CpuKvCache,
    logits: Vec<f32>,
    pos_after: usize,
}

/// Normalized beam score used for pruning and final selection.
#[cfg(feature = "cpu-graph")]
fn beam_normalized_score(total_score: f32, len: usize, alpha: f32) -> f32 {
    if len == 0 {
        f32::NEG_INFINITY
    } else {
        total_score / (len as f32).powf(alpha)
    }
}

/// Evaluate a single candidate token plus a short greedy continuation.
///
/// Returns the cumulative value-head score over `beam_depth` tokens and the
/// first-token state (post-token hidden vector, output logits, KV delta) that
/// can be reused if this candidate is selected.  The shared `kv_scratch` is
/// reset to `kv_parent` before returning so the next candidate starts clean.
#[cfg(feature = "cpu-graph")]
fn evaluate_candidate_beam(
    candidate: u32,
    hidden: &[f32],
    kv_parent: &CpuKvCache,
    kv_scratch: &mut CpuKvCache,
    rerank_scratch: &mut CpuForwardScratch,
    tok: &dyn Tokenizer,
    weights: &CpuModelWeights,
    config: &ModelConfig,
    pos: usize,
    head: &BranchValueHead,
    beam_depth: usize,
) -> Option<(f32, CandidateState)> {
    let mut hidden_candidate = hidden.to_vec();
    cpu_embed_token(
        candidate,
        weights,
        &mut hidden_candidate,
        config,
        Some(rerank_scratch),
    );
    cpu_full_forward(
        &mut hidden_candidate,
        weights,
        kv_scratch,
        rerank_scratch,
        pos,
        config,
    )
    .ok()?;

    let first_score = head.predict(&rerank_scratch.normed);
    let hidden_first = hidden_candidate.clone();
    let first_logits = rerank_scratch.logits[..config.vocab_size].to_vec();
    let first_kv_delta = capture_kv_delta(kv_scratch, pos);

    let mut beam_score = first_score;
    let mut cont_pos = pos + 1;
    for _ in 1..beam_depth {
        if cont_pos >= kv_scratch.max_seq_len {
            break;
        }
        let cont_token = cpu_sample_greedy(&rerank_scratch.logits[..config.vocab_size]);
        if tok.is_eog(cont_token) {
            break;
        }
        cpu_embed_token(
            cont_token,
            weights,
            &mut hidden_candidate,
            config,
            Some(rerank_scratch),
        );
        if cpu_full_forward(
            &mut hidden_candidate,
            weights,
            kv_scratch,
            rerank_scratch,
            cont_pos,
            config,
        )
        .is_err()
        {
            break;
        }
        beam_score += head.predict(&rerank_scratch.normed);
        cont_pos += 1;
    }

    reset_kv_to_parent(kv_scratch, kv_parent, pos);
    Some((
        beam_score,
        CandidateState {
            token_id: candidate,
            hidden: hidden_first,
            logits: first_logits,
            kv_delta: first_kv_delta,
        },
    ))
}

#[cfg(feature = "cpu-graph")]
#[expect(
    clippy::too_many_arguments,
    reason = "CLI orchestration passes through many params"
)]
pub(crate) fn run_cpu_decode_loop_with_ctx(
    args: &Args,
    config: &ModelConfig,
    tok: &dyn Tokenizer,
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
    rerank_beam_depth: usize,
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
    let mut pending_state: Option<ReusableState> = None;

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

        // Each generated token becomes its own branch/timestamp in the graph.
        ctx.timestamp = n_generated as u64;

        // Advance hidden/KV/logits for this token.  If the previous iteration
        // already ran the speculative forward for this token via the reranker,
        // reuse that state instead of recomputing it.
        if let Some(state) = pending_state.take() {
            assert_eq!(
                state.token_id, next_token,
                "pending reusable state must match the token about to be emitted"
            );
            hidden = state.hidden;
            *kv = state.kv;
            scratch.logits.copy_from_slice(&state.logits);
            pos = state.pos_after;

            if args.debug && n_generated <= 3 {
                print_logits_stats(n_generated, &scratch.logits);
                print_top_logits_debug(n_generated, &scratch.logits, tok, 5);
            }
        } else {
            cpu_embed_token(next_token, weights, &mut hidden, config, Some(scratch));

            if args.debug && n_generated <= 3 {
                print_hidden_stats(n_generated, next_token, &hidden);
            }

            cpu_full_forward_with_ctx(ctx, &mut hidden, weights, kv, scratch, pos, config)
                .map_err(|e: CpuError| format!("decode: {}", e))?;

            // Record a scalar score for this branch from the output distribution.
            ctx.score_against(&scratch.logits, None, score_metric);

            pos += 1;

            if args.debug && n_generated <= 3 {
                print_logits_stats(n_generated, &scratch.logits);
                print_top_logits_debug(n_generated, &scratch.logits, tok, 5);
            }
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
                // Evaluate each candidate with the value head, capture the full
                // state of each candidate, and bias the original logits by the
                // speculative score.  Only the chosen candidate's state is kept.
                let mut rerank_scratch = CpuForwardScratch::new(config);
                let mut kv_scratch = kv.clone();
                let mut biased_logits = scratch.logits[..config.vocab_size].to_vec();
                let mut candidate_states: Vec<CandidateState> =
                    Vec::with_capacity(candidates.len());
                let mut candidate_entries: Vec<rocmforge::cpu::graph::CandidateBranch> =
                    Vec::with_capacity(candidates.len());
                for &candidate in &candidates {
                    let Some((score, state)) = evaluate_candidate_beam(
                        candidate,
                        &hidden,
                        kv,
                        &mut kv_scratch,
                        &mut rerank_scratch,
                        tok,
                        weights,
                        config,
                        pos,
                        head,
                        rerank_beam_depth,
                    ) else {
                        let original_logit = scratch.logits[candidate as usize];
                        if args.debug {
                            eprintln!(
                                "[Rerank] step={} candidate={} forward_failed logit_before={:.4}",
                                n_generated, candidate, original_logit
                            );
                        }
                        continue;
                    };
                    let original_logit = scratch.logits[candidate as usize];
                    biased_logits[candidate as usize] += rerank_scale * score;
                    candidate_states.push(state);
                    candidate_entries.push(rocmforge::cpu::graph::CandidateBranch {
                        parent_timestamp: ctx.timestamp,
                        token_id: candidate,
                        value_score: score,
                        biased_logit: biased_logits[candidate as usize],
                        chosen: false,
                    });
                    if args.debug {
                        eprintln!(
                            "[Rerank] step={} candidate={} value_score={:.4} logit_before={:.4} logit_after={:.4}",
                            n_generated,
                            candidate,
                            score,
                            original_logit,
                            biased_logits[candidate as usize]
                        );
                    }
                }
                let chosen = sample_next_token(
                    &biased_logits,
                    use_greedy,
                    args.temperature,
                    args.top_p,
                    &mut seed,
                );
                for entry in &mut candidate_entries {
                    if entry.token_id == chosen {
                        entry.chosen = true;
                    }
                }
                ctx.candidate_branches.extend(candidate_entries);

                // Reconstruct the chosen candidate's full KV and stash the post-
                // token state so the next iteration can skip the main forward.
                let chosen_state = candidate_states
                    .into_iter()
                    .find(|s| s.token_id == chosen)
                    .ok_or_else(|| {
                        format!(
                            "internal error: chosen candidate {} not found in candidate_states",
                            chosen
                        )
                    })?;
                let mut kv_chosen = kv.clone();
                apply_kv_delta(&mut kv_chosen, &chosen_state.kv_delta, pos);
                pending_state = Some(ReusableState {
                    token_id: chosen,
                    hidden: chosen_state.hidden,
                    kv: kv_chosen,
                    logits: chosen_state.logits,
                    pos_after: pos + 1,
                });
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

#[cfg(feature = "cpu-graph")]
struct BeamState {
    hidden: Vec<f32>,
    kv: CpuKvCache,
    logits: Vec<f32>,
    pos_after: usize,
}

#[cfg(feature = "cpu-graph")]
struct BeamHypothesis {
    tokens: Vec<u32>,
    total_score: f32,
    state: BeamState,
}

#[cfg(feature = "cpu-graph")]
struct BeamExpansion {
    tokens: Vec<u32>,
    total_score: f32,
    candidate_state: CandidateState,
    pos_after: usize,
    parent_idx: usize,
    entry_index: usize,
    is_eos: bool,
}

/// Run a value-head beam search that keeps `beam_width` hypotheses alive.
///
/// Each active hypothesis is expanded by its top-k next tokens.  Every
/// expansion is scored over a short greedy continuation (`rerank_beam_depth`)
/// and the top `beam_width` expansions survive to the next step.
#[cfg(feature = "cpu-graph")]
#[expect(
    clippy::too_many_arguments,
    reason = "CLI orchestration passes through many params"
)]
pub(crate) fn run_cpu_decode_beam_loop_with_ctx(
    args: &Args,
    config: &ModelConfig,
    tok: &dyn Tokenizer,
    weights: &CpuModelWeights,
    kv: &CpuKvCache,
    scratch: &CpuForwardScratch,
    n_prompt: usize,
    ctx: &mut CaptureContext,
    value_head: &BranchValueHead,
    rerank_top_k: usize,
    rerank_scale: f32,
    rerank_beam_depth: usize,
    rerank_beam_width: usize,
    rerank_beam_length_penalty: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let initial_state = BeamState {
        hidden: vec![0.0f32; config.hidden_size],
        kv: kv.clone(),
        logits: scratch.logits[..config.vocab_size].to_vec(),
        pos_after: n_prompt,
    };
    let mut active: Vec<BeamHypothesis> = vec![BeamHypothesis {
        tokens: Vec::new(),
        total_score: 0.0f32,
        state: initial_state,
    }];
    let mut finished: Vec<BeamHypothesis> = Vec::new();

    let t_gen = Instant::now();
    let mut rerank_scratch = CpuForwardScratch::new(config);

    for step in 1..=args.max_tokens {
        let mut expansions: Vec<BeamExpansion> = Vec::with_capacity(active.len() * rerank_top_k);
        let mut candidate_entries: Vec<rocmforge::cpu::graph::CandidateBranch> = Vec::new();

        for (parent_idx, parent) in active.iter().enumerate() {
            let candidates =
                top_k_token_ids(&parent.state.logits[..config.vocab_size], rerank_top_k);
            if candidates.is_empty() || parent.state.pos_after >= parent.state.kv.max_seq_len {
                continue;
            }

            let mut kv_scratch = parent.state.kv.clone();
            for &candidate in &candidates {
                let Some((beam_score, candidate_state)) = evaluate_candidate_beam(
                    candidate,
                    &parent.state.hidden,
                    &parent.state.kv,
                    &mut kv_scratch,
                    &mut rerank_scratch,
                    tok,
                    weights,
                    config,
                    parent.state.pos_after,
                    value_head,
                    rerank_beam_depth,
                ) else {
                    continue;
                };

                let original_logit = parent.state.logits[candidate as usize];
                let biased_logit = original_logit + rerank_scale * beam_score;
                if args.debug {
                    eprintln!(
                        "[Rerank] step={} candidate={} value_score={:.4} logit_before={:.4} logit_after={:.4}",
                        step, candidate, beam_score, original_logit, biased_logit
                    );
                }
                let entry_index = candidate_entries.len();
                candidate_entries.push(rocmforge::cpu::graph::CandidateBranch {
                    parent_timestamp: step as u64,
                    token_id: candidate,
                    value_score: beam_score,
                    biased_logit,
                    chosen: false,
                });

                let mut tokens = parent.tokens.clone();
                tokens.push(candidate);
                expansions.push(BeamExpansion {
                    tokens,
                    total_score: parent.total_score + beam_score,
                    candidate_state,
                    pos_after: parent.state.pos_after + 1,
                    parent_idx,
                    entry_index,
                    is_eos: tok.is_eog(candidate),
                });
            }
        }

        if expansions.is_empty() {
            break;
        }

        // Recombine hypotheses that end with the same token, keeping the best
        // normalized score for each group.
        let mut best_by_last: HashMap<u32, BeamExpansion> = HashMap::new();
        for expansion in expansions {
            let last = *expansion
                .tokens
                .last()
                .expect("beam expansion must contain at least one token");
            let score = beam_normalized_score(
                expansion.total_score,
                expansion.tokens.len(),
                rerank_beam_length_penalty,
            );
            let keep = best_by_last
                .get(&last)
                .map(|existing| {
                    score
                        > beam_normalized_score(
                            existing.total_score,
                            existing.tokens.len(),
                            rerank_beam_length_penalty,
                        )
                })
                .unwrap_or(true);
            if keep {
                best_by_last.insert(last, expansion);
            }
        }
        let mut expansions: Vec<BeamExpansion> = best_by_last.into_values().collect();

        expansions.sort_by(|a, b| {
            let a_score =
                beam_normalized_score(a.total_score, a.tokens.len(), rerank_beam_length_penalty);
            let b_score =
                beam_normalized_score(b.total_score, b.tokens.len(), rerank_beam_length_penalty);
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let survivors: Vec<BeamExpansion> =
            expansions.into_iter().take(rerank_beam_width).collect();

        for survivor in &survivors {
            if let Some(entry) = candidate_entries.get_mut(survivor.entry_index) {
                entry.chosen = true;
            }
        }
        ctx.candidate_branches.extend(candidate_entries);

        let mut next_active: Vec<BeamHypothesis> = Vec::with_capacity(rerank_beam_width);
        for survivor in survivors {
            if survivor.is_eos {
                finished.push(BeamHypothesis {
                    tokens: survivor.tokens,
                    total_score: survivor.total_score,
                    state: BeamState {
                        hidden: survivor.candidate_state.hidden,
                        kv: {
                            let mut kv = active[survivor.parent_idx].state.kv.clone();
                            apply_kv_delta(
                                &mut kv,
                                &survivor.candidate_state.kv_delta,
                                active[survivor.parent_idx].state.pos_after,
                            );
                            kv
                        },
                        logits: survivor.candidate_state.logits,
                        pos_after: survivor.pos_after,
                    },
                });
                continue;
            }

            let mut kv = active[survivor.parent_idx].state.kv.clone();
            apply_kv_delta(
                &mut kv,
                &survivor.candidate_state.kv_delta,
                active[survivor.parent_idx].state.pos_after,
            );
            next_active.push(BeamHypothesis {
                tokens: survivor.tokens,
                total_score: survivor.total_score,
                state: BeamState {
                    hidden: survivor.candidate_state.hidden,
                    kv,
                    logits: survivor.candidate_state.logits,
                    pos_after: survivor.pos_after,
                },
            });
        }
        active = next_active;
    }

    let best = if !finished.is_empty() {
        finished
            .into_iter()
            .max_by(|a, b| {
                let a_score = beam_normalized_score(
                    a.total_score,
                    a.tokens.len(),
                    rerank_beam_length_penalty,
                );
                let b_score = beam_normalized_score(
                    b.total_score,
                    b.tokens.len(),
                    rerank_beam_length_penalty,
                );
                a_score
                    .partial_cmp(&b_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or("no finished beam hypothesis")?
    } else {
        active
            .into_iter()
            .max_by(|a, b| {
                let a_score = beam_normalized_score(
                    a.total_score,
                    a.tokens.len(),
                    rerank_beam_length_penalty,
                );
                let b_score = beam_normalized_score(
                    b.total_score,
                    b.tokens.len(),
                    rerank_beam_length_penalty,
                );
                a_score
                    .partial_cmp(&b_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or("no beam hypothesis survived")?
    };

    print!("\n");
    for &token in &best.tokens {
        let text = tok.decode_token(token);
        print!("{}", text);
    }
    std::io::stdout().flush().ok();
    println!();

    if !best.tokens.is_empty() {
        let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        print_generation_stats(best.tokens.len(), gen_ms);
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
