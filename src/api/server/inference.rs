use crate::config::ModelConfig;
use crate::cpu::weights::CpuModelWeights;
use crate::tokenizer::BpeTokenizer;
use bytes::Bytes;
use std::sync::Arc;
use tokio::sync::Mutex;

pub(crate) fn run_sync_inference(
    cpu_weights: &Arc<CpuModelWeights>,
    #[cfg(feature = "gpu")] gpu_weights: &Option<Arc<crate::gpu::GpuModelWeights>>,
    #[cfg(feature = "gpu")] speculative_engine: &Option<Arc<Mutex<crate::gpu::SpeculativeEngine>>>,
    model_path: &str,
    config: &ModelConfig,
    tok: &BpeTokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> crate::error::RocmForgeResult<(String, usize)> {
    #[cfg(feature = "gpu")]
    {
        if let Some(spec_engine_arc) = speculative_engine {
            let gpu_caps = crate::gpu::detect().ok_or("GPU requested but no AMD GPU detected")?;
            let device = crate::gpu::GpuDevice::get_or_init(gpu_caps.device_id)
                .map_err(|e| format!("gpu init: {}", e))?;
            let mut engine = spec_engine_arc.blocking_lock();
            let orchestrator =
                crate::gpu::SpeculativeOrchestrator::new(4).map_err(|e| format!("{:?}", e))?;
            return orchestrator.generate(&device, &mut engine, tok, prompt_tokens, max_tokens);
        }
        if let Some(gw) = gpu_weights {
            return crate::api::gpu_inference::run_gpu_sync_inference(
                gw,
                cpu_weights,
                config,
                tok,
                prompt_tokens,
                max_tokens,
                temperature,
                top_p,
            );
        }
    }

    run_cpu_sync_inference(
        cpu_weights,
        model_path,
        config,
        tok,
        prompt_tokens,
        max_tokens,
        temperature,
        top_p,
    )
}

pub(crate) fn run_stream_inference(
    cpu_weights: &Arc<CpuModelWeights>,
    #[cfg(feature = "gpu")] gpu_weights: &Option<Arc<crate::gpu::GpuModelWeights>>,
    #[cfg(feature = "gpu")] speculative_engine: &Option<Arc<Mutex<crate::gpu::SpeculativeEngine>>>,
    model_path: &str,
    config: &ModelConfig,
    tok: &BpeTokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    tx: tokio::sync::mpsc::UnboundedSender<Bytes>,
) -> crate::error::RocmForgeResult<()> {
    #[cfg(feature = "gpu")]
    {
        if speculative_engine.is_some() {
            return Err("Streaming is not yet supported for speculative decoding".into());
        }
        if let Some(gw) = gpu_weights {
            return crate::api::gpu_inference::run_gpu_stream_inference(
                gw,
                cpu_weights,
                config,
                tok,
                prompt_tokens,
                max_tokens,
                temperature,
                top_p,
                tx,
            );
        }
    }

    run_cpu_stream_inference(
        cpu_weights,
        model_path,
        config,
        tok,
        prompt_tokens,
        max_tokens,
        temperature,
        top_p,
        tx,
    )
}

pub(crate) fn run_cpu_sync_inference(
    weights: &CpuModelWeights,
    _model_path: &str,
    config: &ModelConfig,
    tok: &BpeTokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> crate::error::RocmForgeResult<(String, usize)> {
    use crate::cpu::cache::{CpuForwardScratch, CpuKvCache};
    use crate::cpu::forward::{cpu_embed_token, cpu_full_forward, cpu_prefill};
    use crate::cpu::sampler::{cpu_sample_greedy, cpu_sample_top_p};

    let mut kv = CpuKvCache::new(config, prompt_tokens.len() + max_tokens);
    let mut scratch = CpuForwardScratch::new(config);
    let mut hidden = vec![0.0f32; config.hidden_size];
    let use_greedy = temperature <= 0.0;
    let mut seed = 42u64;

    cpu_prefill(
        &mut hidden,
        weights,
        &mut kv,
        &mut scratch,
        prompt_tokens,
        config,
    )?;

    let mut next_token = if use_greedy {
        cpu_sample_greedy(&scratch.logits)
    } else {
        cpu_sample_top_p(&scratch.logits, temperature, top_p, seed)
    };

    let mut output_tokens = Vec::new();
    let mut pos = prompt_tokens.len();

    for _ in 0..max_tokens {
        if tok.is_eog(next_token) {
            break;
        }

        output_tokens.push(next_token);
        cpu_embed_token(next_token, weights, &mut hidden, config, Some(&mut scratch));
        cpu_full_forward(&mut hidden, weights, &mut kv, &mut scratch, pos, config)?;

        next_token = if use_greedy {
            cpu_sample_greedy(&scratch.logits)
        } else {
            seed = seed.wrapping_add(1);
            cpu_sample_top_p(&scratch.logits, temperature, top_p, seed)
        };
        pos += 1;
    }

    Ok((tok.decode(&output_tokens, true), output_tokens.len()))
}

pub(crate) fn run_cpu_stream_inference(
    weights: &CpuModelWeights,
    _model_path: &str,
    config: &ModelConfig,
    tok: &BpeTokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    tx: tokio::sync::mpsc::UnboundedSender<Bytes>,
) -> crate::error::RocmForgeResult<()> {
    use crate::cpu::cache::{CpuForwardScratch, CpuKvCache};
    use crate::cpu::forward::{cpu_embed_token, cpu_full_forward, cpu_prefill};
    use crate::cpu::sampler::{cpu_sample_greedy, cpu_sample_top_p};

    let mut kv = CpuKvCache::new(config, prompt_tokens.len() + max_tokens);
    let mut scratch = CpuForwardScratch::new(config);
    let mut hidden = vec![0.0f32; config.hidden_size];
    let use_greedy = temperature <= 0.0;
    let mut seed = 42u64;

    cpu_prefill(
        &mut hidden,
        weights,
        &mut kv,
        &mut scratch,
        prompt_tokens,
        config,
    )?;

    let mut next_token = if use_greedy {
        cpu_sample_greedy(&scratch.logits)
    } else {
        cpu_sample_top_p(&scratch.logits, temperature, top_p, seed)
    };

    let mut output_tokens = Vec::new();
    let mut pos = prompt_tokens.len();
    let mut previous_text = String::new();

    for _ in 0..max_tokens {
        if tok.is_eog(next_token) {
            break;
        }

        output_tokens.push(next_token);

        // Decode incremental text for this batch of tokens
        let text = tok.decode(&output_tokens, false);
        let new_chars = &text[previous_text.len().min(text.len())..];
        if !new_chars.is_empty() {
            let _ = tx.send(Bytes::from(new_chars.to_string()));
        }
        previous_text = text;

        cpu_embed_token(next_token, weights, &mut hidden, config, Some(&mut scratch));
        cpu_full_forward(&mut hidden, weights, &mut kv, &mut scratch, pos, config)?;

        next_token = if use_greedy {
            cpu_sample_greedy(&scratch.logits)
        } else {
            seed = seed.wrapping_add(1);
            cpu_sample_top_p(&scratch.logits, temperature, top_p, seed)
        };
        pos += 1;
    }

    Ok(())
}
