#[cfg(feature = "gpu")]
use rocmforge::config::ModelConfig;
#[cfg(feature = "gpu")]
use rocmforge::cpu::{cache::CpuForwardScratch, weights::CpuModelWeights};
#[cfg(feature = "gpu")]
use rocmforge::gpu;
#[cfg(feature = "gpu")]
use rocmforge::gpu::weights::GpuModelWeights;

#[cfg(any(feature = "gpu", test))]
fn is_last_prompt_pos(pos: usize, prompt_len: usize) -> bool {
    pos + 1 == prompt_len
}

#[cfg(feature = "gpu")]
fn logits_mode_for_prompt_pos(
    pos: usize,
    prompt_len: usize,
    final_prompt_logits_mode: gpu::GpuLogitsMode,
) -> gpu::GpuLogitsMode {
    if is_last_prompt_pos(pos, prompt_len) {
        final_prompt_logits_mode
    } else {
        gpu::GpuLogitsMode::Skip
    }
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments, reason = "CLI shape")]
pub(crate) fn run_decode_style_prompt_path(
    device: &gpu::GpuDevice,
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    kv: &mut gpu::GpuKvCache,
    gpu_scratch: &mut gpu::GpuForwardScratch,
    host_scratch: &mut CpuForwardScratch,
    prompt_tokens: &[u32],
    config: &ModelConfig,
    final_prompt_logits_mode: gpu::GpuLogitsMode,
) -> Result<Option<u32>, Box<dyn std::error::Error>> {
    let mut prompt_next_token = None;
    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        gpu::gpu_embed_token_hybrid(
            device,
            token_id,
            gpu_weights,
            cpu_weights,
            gpu_scratch,
            host_scratch,
            config,
        )
        .map_err(|e| format!("gpu embed: {}", e))?;
        let logits_mode =
            logits_mode_for_prompt_pos(pos, prompt_tokens.len(), final_prompt_logits_mode);
        prompt_next_token = gpu::gpu_full_forward_hybrid(
            device,
            gpu_weights,
            cpu_weights,
            kv,
            gpu_scratch,
            host_scratch,
            pos,
            config,
            logits_mode,
            token_id,
        )
        .map_err(|e| format!("gpu prefill/decode: {}", e))?;
    }
    Ok(prompt_next_token)
}

#[cfg(test)]
mod tests {
    use super::is_last_prompt_pos;

    #[test]
    fn is_last_prompt_pos_detects_final_token() {
        assert!(!is_last_prompt_pos(0, 3));
        assert!(!is_last_prompt_pos(1, 3));
        assert!(is_last_prompt_pos(2, 3));
    }
}
