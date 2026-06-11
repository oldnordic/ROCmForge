use rocmforge::config::ModelConfig;
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    weights::CpuModelWeights,
};
use rocmforge::hardware::{derive_batch_config, BatchConfig, CpuCapabilities};
use rocmforge::loader::ModelFile;
use rocmforge::tokenizer::BpeTokenizer;

use super::cli::Args;
use super::cpu_debug::{print_batch_config, print_prompt_summary};

pub(crate) struct CpuRunState {
    pub config: ModelConfig,
    pub tok: BpeTokenizer,
    pub weights: CpuModelWeights,
    pub batch_config: BatchConfig,
    pub prompt_tokens: Vec<u32>,
    pub kv: CpuKvCache,
    pub scratch: CpuForwardScratch,
    pub use_greedy: bool,
}

fn compute_max_seq(
    ctx_size: Option<usize>,
    prompt_len: usize,
    max_tokens: usize,
    model_max_seq_len: usize,
) -> usize {
    ctx_size.unwrap_or_else(|| (prompt_len + max_tokens).min(model_max_seq_len))
}

pub(crate) fn prepare_cpu_inference_state(
    args: &Args,
    caps: &CpuCapabilities,
) -> Result<CpuRunState, Box<dyn std::error::Error>> {
    let file = ModelFile::open(&args.model)?;
    eprintln!("[Args] model path ({}): {}", file.format_name(), args.model);

    let config = file.config()?;
    let tok = file.tokenizer();
    eprintln!(
        "[Tokenizer] bos_id={:?} eos_id={:?} add_bos={} add_eos={}",
        tok.bos_id(),
        tok.eos_id(),
        tok.add_bos(),
        tok.add_eos()
    );
    eprintln!(
        "Model: {} layers, {} vocab, {} hidden",
        config.num_layers, config.vocab_size, config.hidden_size
    );

    eprint!("Loading weights... ");
    let t_load = std::time::Instant::now();
    let weights = file
        .load_cpu_weights(&config)
        .map_err(|e| format!("weight load: {}", e))?;
    eprintln!("done in {:.1}s", t_load.elapsed().as_secs_f64());

    let template = file.chat_template(&config, args.no_template);
    let mut batch_config: BatchConfig = derive_batch_config(caps, &config);
    if let Some(t) = args.threads {
        batch_config.num_cores = t;
    }
    print_batch_config(batch_config.max_tokens_per_batch, batch_config.num_cores);

    let prompted = template.apply(&args.prompt);
    let prompt_tokens = tok.encode(&prompted, false);
    if prompt_tokens.is_empty() {
        return Err("Prompt tokenized to zero tokens".into());
    }
    print_prompt_summary(template.name(), prompt_tokens.len());

    let max_seq = compute_max_seq(
        args.ctx_size,
        prompt_tokens.len(),
        args.max_tokens,
        config.max_seq_len,
    );
    let kv = CpuKvCache::new(&config, max_seq);
    let scratch = CpuForwardScratch::new(&config);
    let use_greedy = args.top_p >= 1.0;

    Ok(CpuRunState {
        config,
        tok,
        weights,
        batch_config,
        prompt_tokens,
        kv,
        scratch,
        use_greedy,
    })
}

#[cfg(test)]
mod tests {
    use super::compute_max_seq;

    #[test]
    fn compute_max_seq_uses_override_when_present() {
        assert_eq!(compute_max_seq(Some(4096), 100, 200, 2048), 4096);
    }

    #[test]
    fn compute_max_seq_clamps_to_model_default() {
        assert_eq!(compute_max_seq(None, 300, 400, 512), 512);
        assert_eq!(compute_max_seq(None, 100, 50, 512), 150);
    }
}
