#[cfg(feature = "gpu")]
use rocmforge::config::ModelConfig;
#[cfg(feature = "gpu")]
use rocmforge::loader::ModelFile;
#[cfg(feature = "gpu")]
use rocmforge::tokenizer::TokenizerHandle;

#[cfg(feature = "gpu")]
use super::cli::Args;

#[cfg(feature = "gpu")]
pub(crate) struct GpuPromptSetup {
    pub tok: TokenizerHandle,
    pub prompt_tokens: Vec<u32>,
    pub max_seq: usize,
}

#[cfg(any(feature = "gpu", test))]
fn compute_gpu_max_seq(
    ctx_size: Option<usize>,
    prompt_len: usize,
    max_tokens: usize,
    model_max_seq_len: usize,
) -> usize {
    ctx_size.unwrap_or_else(|| (prompt_len + max_tokens).min(model_max_seq_len))
}

#[cfg(feature = "gpu")]
pub(crate) fn prepare_gpu_prompt(
    file: &ModelFile,
    config: &ModelConfig,
    args: &Args,
) -> Result<GpuPromptSetup, Box<dyn std::error::Error>> {
    eprintln!("[Args] model path ({}): {}", file.format_name(), args.model);
    let tok = file.tokenizer();
    let template = file.chat_template(config, args.no_template);
    let prompted = template.apply(&args.prompt);
    eprintln!("Chat template: {}", template.name());

    let prompt_tokens = tok.encode(&prompted, false);
    if prompt_tokens.is_empty() {
        return Err("Prompt tokenized to zero tokens".into());
    }
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    let max_seq = compute_gpu_max_seq(
        args.ctx_size,
        prompt_tokens.len(),
        args.max_tokens,
        config.max_seq_len,
    );

    Ok(GpuPromptSetup {
        tok,
        prompt_tokens,
        max_seq,
    })
}

#[cfg(test)]
mod tests {
    use super::compute_gpu_max_seq;

    #[test]
    fn compute_gpu_max_seq_prefers_override_and_clamps_default() {
        assert_eq!(compute_gpu_max_seq(Some(4096), 100, 200, 2048), 4096);
        assert_eq!(compute_gpu_max_seq(None, 100, 50, 512), 150);
        assert_eq!(compute_gpu_max_seq(None, 300, 400, 512), 512);
    }
}
