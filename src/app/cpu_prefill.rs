use std::time::Instant;

use rocmforge::config::ModelConfig;
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::cpu_embed_token,
    prefill::cpu_prefill_forward_parallel,
    weights::CpuModelWeights,
    CpuError,
};
use rocmforge::hardware::BatchConfig;
use rocmforge::tokenizer::BpeTokenizer;

use super::cli::Args;
use super::cpu_debug::{print_prefill_debug, print_prefill_stats, print_top_logits_debug};

fn first_prompt_token(prompt_tokens: &[u32]) -> Option<u32> {
    prompt_tokens.first().copied()
}

#[expect(
    clippy::too_many_arguments,
    reason = "CLI orchestration passes through many params"
)]
pub(crate) fn run_cpu_prefill(
    args: &Args,
    config: &ModelConfig,
    tok: &BpeTokenizer,
    weights: &CpuModelWeights,
    batch_config: &BatchConfig,
    prompt_tokens: &[u32],
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
) -> Result<(), Box<dyn std::error::Error>> {
    let t_prefill = Instant::now();
    let n_prompt = prompt_tokens.len();

    if args.debug {
        if let Some(first_tok) = first_prompt_token(prompt_tokens) {
            let mut test_hidden = vec![0.0f32; config.hidden_size];
            cpu_embed_token(first_tok, weights, &mut test_hidden, config);
            print_prefill_debug(first_tok, &test_hidden);
        }
    }

    cpu_prefill_forward_parallel(prompt_tokens, weights, kv, scratch, 0, config, batch_config)
        .map_err(|e: CpuError| format!("prefill: {}", e))?;

    if args.debug {
        print_top_logits_debug(0, &scratch.logits, tok, 10);
    }

    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    print_prefill_stats(prefill_ms, n_prompt);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::first_prompt_token;

    #[test]
    fn first_prompt_token_handles_empty_and_nonempty_inputs() {
        assert_eq!(first_prompt_token(&[]), None);
        assert_eq!(first_prompt_token(&[42, 7]), Some(42));
    }
}
