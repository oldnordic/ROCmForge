use std::io::Write;
use std::time::Instant;

use rocmforge::config::ModelConfig;
use rocmforge::cpu::{
    CpuError,
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward},
    sampler::{cpu_sample_greedy, cpu_sample_top_p},
    weights::CpuModelWeights,
};
use rocmforge::tokenizer::{Tokenizer, TokenizerHandle};

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
