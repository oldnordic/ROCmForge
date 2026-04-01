#![cfg(feature = "gpu")]

mod common;

use rocmforge::config::{detect_chat_template, ModelConfig};
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward, cpu_layer_forward},
    ops::dispatch_gemv as cpu_dispatch_gemv,
    sampler::cpu_sample_greedy,
    weights::CpuModelWeights,
};
use rocmforge::gpu::{
    self, graph::DecodeGraphScope, GpuBuffer, GpuDevice, GpuForwardScratch, GpuKvCache,
};
use rocmforge::loader::GgufFile;
use rocmforge::tokenizer::BpeTokenizer;
use serial_test::serial;

const MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

fn run_cpu_prompt_reference(
    prompt_tokens: &[u32],
    weights: &CpuModelWeights,
    config: &ModelConfig,
) -> Vec<f32> {
    let mut kv = CpuKvCache::new(config, prompt_tokens.len().max(1));
    let mut scratch = CpuForwardScratch::new(config);
    let mut hidden = vec![0.0f32; config.hidden_size];

    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        cpu_embed_token(token_id, weights, &mut hidden, config);
        cpu_full_forward(&mut hidden, weights, &mut kv, &mut scratch, pos, config)
            .expect("CPU decode should succeed");
    }

    scratch.logits
}

fn build_cpu_prompt_embeddings(
    prompt_tokens: &[u32],
    weights: &CpuModelWeights,
    config: &ModelConfig,
) -> Vec<f32> {
    let mut hidden = vec![0.0f32; prompt_tokens.len() * config.hidden_size];
    for (row, &token_id) in prompt_tokens.iter().enumerate() {
        cpu_embed_token(
            token_id,
            weights,
            &mut hidden[row * config.hidden_size..(row + 1) * config.hidden_size],
            config,
        );
    }
    hidden
}

fn download_gpu_f32(buf: &rocmforge::gpu::GpuBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    buf.copy_to_host(&mut bytes)
        .expect("GPU buffer should download");
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() }
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64], avg: f64) -> f64 {
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - avg;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

#[test]
#[serial]
fn test_gpu_embed_real_model_matches_cpu_hidden() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let _lock = match common::GpuLock::acquire() {
        Ok(lock) => lock,
        Err(err) => {
            eprintln!("Skipping test: {}", err);
            return;
        }
    };

    require_gpu!();
    require_vram!(4);

    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse model config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("CPU weights should load");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("GPU weights should load");
    assert_eq!(
        gpu_weights.token_emb_meta.wtype,
        rocmforge::loader::GgmlType::Q8_0,
        "expected Q8_0 token embeddings for this GPU embedding regression"
    );

    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());
    let template =
        detect_chat_template(&config.architecture, file.tokenizer_data().model.as_deref());
    let prompt = template.apply("Hello");
    let prompt_tokens = tok.encode(&prompt, false);
    assert!(!prompt_tokens.is_empty(), "prompt should tokenize");

    let first_token = prompt_tokens[0];
    let mut cpu_hidden = vec![0.0f32; config.hidden_size];
    cpu_embed_token(first_token, &cpu_weights, &mut cpu_hidden, &config);

    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("GPU scratch should allocate");
    let mut host_scratch = CpuForwardScratch::new(&config);
    gpu::gpu_embed_token_hybrid(
        &device,
        first_token,
        &gpu_weights,
        &cpu_weights,
        &mut gpu_scratch,
        &mut host_scratch,
        &config,
    )
    .expect("GPU embedding should succeed");

    let gpu_hidden = download_gpu_f32(&gpu_scratch.hidden, config.hidden_size);
    let max_err = max_abs_error(&cpu_hidden, &gpu_hidden);
    assert!(
        max_err <= 1e-6,
        "GPU token embedding mismatch: max_abs_error={}",
        max_err
    );
}

#[test]
#[serial]
fn test_gpu_decode_real_model_matches_cpu_greedy_token() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let _lock = match common::GpuLock::acquire() {
        Ok(lock) => lock,
        Err(err) => {
            eprintln!("Skipping test: {}", err);
            return;
        }
    };

    require_gpu!();
    require_vram!(4);

    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse model config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("CPU weights should load");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("GPU weights should load");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());

    let template =
        detect_chat_template(&config.architecture, file.tokenizer_data().model.as_deref());
    let prompt = template.apply("Hello");
    let prompt_tokens = tok.encode(&prompt, false);
    assert!(!prompt_tokens.is_empty(), "prompt should tokenize");

    let cpu_logits = run_cpu_prompt_reference(&prompt_tokens, &cpu_weights, &config);
    let cpu_next = cpu_sample_greedy(&cpu_logits);

    let mut kv =
        GpuKvCache::new(&config, prompt_tokens.len().max(1)).expect("GPU KV should allocate");
    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("GPU scratch should allocate");
    let mut host_scratch = CpuForwardScratch::new(&config);

    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        gpu::gpu_embed_token_hybrid(
            &device,
            token_id,
            &gpu_weights,
            &cpu_weights,
            &mut gpu_scratch,
            &mut host_scratch,
            &config,
        )
        .expect("GPU embed should succeed");
        gpu::gpu_full_forward_hybrid(
            &device,
            &gpu_weights,
            &cpu_weights,
            &mut kv,
            &mut gpu_scratch,
            &mut host_scratch,
            pos,
            &config,
            gpu::GpuLogitsMode::DownloadToHost,
        )
        .expect("GPU decode should succeed");
    }

    assert!(
        host_scratch.logits.iter().all(|x| x.is_finite()),
        "GPU logits should be finite"
    );

    let gpu_next = cpu_sample_greedy(&host_scratch.logits);

    if gpu_next != cpu_next {
        let first_token = prompt_tokens[0];

        let mut cpu_hidden_l0 = vec![0.0f32; config.hidden_size];
        cpu_embed_token(first_token, &cpu_weights, &mut cpu_hidden_l0, &config);
        let mut cpu_kv_l0 = CpuKvCache::new(&config, 1);
        let mut cpu_scratch_l0 = CpuForwardScratch::new(&config);
        cpu_layer_forward(
            &mut cpu_hidden_l0,
            cpu_weights.layer(0),
            &mut cpu_kv_l0,
            &mut cpu_scratch_l0,
            0,
            0,
            &config,
            false,
        )
        .expect("CPU layer 0 should succeed");

        let mut gpu_kv_l0 = GpuKvCache::new(&config, 1).expect("GPU layer-0 KV should allocate");
        let mut gpu_scratch_l0 =
            GpuForwardScratch::new(&config).expect("GPU layer-0 scratch should allocate");
        let mut gpu_host_scratch_l0 = CpuForwardScratch::new(&config);
        gpu::gpu_embed_token_hybrid(
            &device,
            first_token,
            &gpu_weights,
            &cpu_weights,
            &mut gpu_scratch_l0,
            &mut gpu_host_scratch_l0,
            &config,
        )
        .expect("GPU layer-0 embed should succeed");
        gpu::gpu_layer_forward_hybrid(
            &device,
            gpu_weights.layer(0),
            &mut gpu_kv_l0,
            &mut gpu_scratch_l0,
            0,
            0,
            &config,
        )
        .expect("GPU layer 0 should succeed");

        let gpu_hidden_l0 = download_gpu_f32(&gpu_scratch_l0.hidden, config.hidden_size);
        let layer0_max_abs_error = max_abs_error(&cpu_hidden_l0, &gpu_hidden_l0);
        let q_size = config.num_heads * config.head_dim;
        let kv_size = config.num_kv_heads * config.head_dim;
        let gpu_q_l0 = download_gpu_f32(&gpu_scratch_l0.q, q_size);
        let gpu_k_l0 = download_gpu_f32(&gpu_scratch_l0.k, kv_size);
        let gpu_v_l0 = download_gpu_f32(&gpu_scratch_l0.v, kv_size);
        let gpu_attn_out_l0 = download_gpu_f32(&gpu_scratch_l0.attn_out, q_size);
        let gpu_normed_l0 = download_gpu_f32(&gpu_scratch_l0.normed, config.hidden_size);
        let gpu_gate_l0 = download_gpu_f32(&gpu_scratch_l0.gate, config.intermediate_size);
        let gpu_swiglu_l0 = download_gpu_f32(&gpu_scratch_l0.swiglu, config.intermediate_size);
        let gpu_layer_out_l0 = download_gpu_f32(&gpu_scratch_l0.layer_out, config.hidden_size);
        let q_max_abs_error = max_abs_error(&cpu_scratch_l0.q, &gpu_q_l0);
        let k_max_abs_error = max_abs_error(&cpu_scratch_l0.k, &gpu_k_l0);
        let v_max_abs_error = max_abs_error(&cpu_scratch_l0.v, &gpu_v_l0);
        let attn_out_max_abs_error = max_abs_error(&cpu_scratch_l0.attn_out, &gpu_attn_out_l0);
        let normed_max_abs_error = max_abs_error(&cpu_scratch_l0.normed, &gpu_normed_l0);
        let gate_max_abs_error = max_abs_error(&cpu_scratch_l0.gate, &gpu_gate_l0);
        let swiglu_max_abs_error = max_abs_error(&cpu_scratch_l0.swiglu, &gpu_swiglu_l0);
        let layer_out_max_abs_error = max_abs_error(&cpu_scratch_l0.layer_out, &gpu_layer_out_l0);
        let mut cpu_layer_out_from_gpu_swiglu = vec![0.0f32; config.hidden_size];
        let mut gpu_swiglu_q8_scratch = vec![0u8; host_scratch.q8_scratch.len()];
        cpu_dispatch_gemv(
            &cpu_weights.layer(0).ffn_down,
            &cpu_weights.layer(0).ffn_down_meta,
            &gpu_swiglu_l0,
            &mut cpu_layer_out_from_gpu_swiglu,
            config.hidden_size,
            config.intermediate_size,
            Some(&mut gpu_swiglu_q8_scratch),
        )
        .expect("CPU ffn_down on GPU swiglu should succeed");
        let gpu_layer_out_vs_cpu_on_gpu_swiglu =
            max_abs_error(&cpu_layer_out_from_gpu_swiglu, &gpu_layer_out_l0);

        let mut cpu_hidden_diag = vec![0.0f32; config.hidden_size];
        let mut cpu_kv_diag = CpuKvCache::new(&config, prompt_tokens.len().max(1));
        let mut cpu_scratch_diag = CpuForwardScratch::new(&config);
        let mut gpu_kv_diag =
            GpuKvCache::new(&config, prompt_tokens.len().max(1)).expect("diag GPU KV");
        let mut gpu_scratch_diag = GpuForwardScratch::new(&config).expect("diag GPU scratch");
        let mut gpu_host_diag = CpuForwardScratch::new(&config);
        let mut worst_ffn_down_err = 0.0f32;
        let mut worst_ffn_down_layer = 0usize;
        let mut worst_ffn_down_pos = 0usize;

        for (diag_pos, &diag_token_id) in prompt_tokens.iter().enumerate() {
            cpu_embed_token(diag_token_id, &cpu_weights, &mut cpu_hidden_diag, &config);
            gpu::gpu_embed_token_hybrid(
                &device,
                diag_token_id,
                &gpu_weights,
                &cpu_weights,
                &mut gpu_scratch_diag,
                &mut gpu_host_diag,
                &config,
            )
            .expect("diag GPU embed should succeed");

            for layer_idx in 0..config.num_layers {
                cpu_layer_forward(
                    &mut cpu_hidden_diag,
                    cpu_weights.layer(layer_idx),
                    &mut cpu_kv_diag,
                    &mut cpu_scratch_diag,
                    layer_idx,
                    diag_pos,
                    &config,
                    false,
                )
                .expect("diag CPU layer should succeed");
                gpu::gpu_layer_forward_hybrid(
                    &device,
                    gpu_weights.layer(layer_idx),
                    &mut gpu_kv_diag,
                    &mut gpu_scratch_diag,
                    layer_idx,
                    diag_pos,
                    &config,
                )
                .expect("diag GPU layer should succeed");

                let diag_gpu_swiglu =
                    download_gpu_f32(&gpu_scratch_diag.swiglu, config.intermediate_size);
                let diag_gpu_layer_out =
                    download_gpu_f32(&gpu_scratch_diag.layer_out, config.hidden_size);
                let mut cpu_layer_out_from_diag_gpu_swiglu = vec![0.0f32; config.hidden_size];
                let mut diag_q8_scratch = vec![0u8; cpu_scratch_diag.q8_scratch.len()];
                cpu_dispatch_gemv(
                    &cpu_weights.layer(layer_idx).ffn_down,
                    &cpu_weights.layer(layer_idx).ffn_down_meta,
                    &diag_gpu_swiglu,
                    &mut cpu_layer_out_from_diag_gpu_swiglu,
                    config.hidden_size,
                    config.intermediate_size,
                    Some(&mut diag_q8_scratch),
                )
                .expect("diag CPU ffn_down should succeed");

                let err = max_abs_error(&cpu_layer_out_from_diag_gpu_swiglu, &diag_gpu_layer_out);
                if err > worst_ffn_down_err {
                    worst_ffn_down_err = err;
                    worst_ffn_down_layer = layer_idx;
                    worst_ffn_down_pos = diag_pos;
                }
            }
        }

        let gpu_normed = download_gpu_f32(&gpu_scratch.normed, config.hidden_size);
        let mut cpu_logits_from_gpu_normed = vec![0.0f32; config.vocab_size];
        let mut q8_scratch = vec![0u8; host_scratch.q8_scratch.len()];
        cpu_dispatch_gemv(
            &cpu_weights.lm_head,
            &cpu_weights.lm_head_meta,
            &gpu_normed,
            &mut cpu_logits_from_gpu_normed,
            config.vocab_size,
            config.hidden_size,
            Some(&mut q8_scratch),
        )
        .expect("CPU LM head on GPU normed state should succeed");
        let gpu_normed_cpu_lm_head_next = cpu_sample_greedy(&cpu_logits_from_gpu_normed);

        eprintln!(
            "CPU next={} GPU next={} GPU-hidden/CPU-lm_head next={} layer0_hidden={:.6} q={:.6} k={:.6} v={:.6} attn_out={:.6} normed={:.6} gate={:.6} swiglu={:.6} layer_out={:.6} layer0_ffn_down_gpu_input={:.6} worst_ffn_down={:.6}@layer{}:pos{}:{:?}",
            cpu_next,
            gpu_next,
            gpu_normed_cpu_lm_head_next,
            layer0_max_abs_error,
            q_max_abs_error,
            k_max_abs_error,
            v_max_abs_error,
            attn_out_max_abs_error,
            normed_max_abs_error,
            gate_max_abs_error,
            swiglu_max_abs_error,
            layer_out_max_abs_error,
            gpu_layer_out_vs_cpu_on_gpu_swiglu,
            worst_ffn_down_err,
            worst_ffn_down_layer,
            worst_ffn_down_pos,
            gpu_weights.layer(worst_ffn_down_layer).ffn_down_meta.wtype
        );
    }

    assert_eq!(
        gpu_next, cpu_next,
        "GPU and CPU greedy next-token should match"
    );
}

#[test]
#[serial]
fn test_gpu_greedy_decode_populates_cached_graph() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let _lock = match common::GpuLock::acquire() {
        Ok(lock) => lock,
        Err(err) => {
            eprintln!("Skipping test: {}", err);
            return;
        }
    };

    require_gpu!();
    require_vram!(4);

    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse model config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("CPU weights should load");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("GPU weights should load");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());

    let template =
        detect_chat_template(&config.architecture, file.tokenizer_data().model.as_deref());
    let prompt = template.apply("Hello");
    let prompt_tokens = tok.encode(&prompt, false);
    assert!(!prompt_tokens.is_empty(), "prompt should tokenize");

    let mut kv =
        GpuKvCache::new(&config, prompt_tokens.len().max(1)).expect("GPU KV should allocate");
    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("GPU scratch should allocate");
    let mut host_scratch = CpuForwardScratch::new(&config);

    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        gpu::gpu_embed_token_hybrid(
            &device,
            token_id,
            &gpu_weights,
            &cpu_weights,
            &mut gpu_scratch,
            &mut host_scratch,
            &config,
        )
        .expect("GPU embed should succeed");
        gpu::gpu_full_forward_hybrid(
            &device,
            &gpu_weights,
            &cpu_weights,
            &mut kv,
            &mut gpu_scratch,
            &mut host_scratch,
            pos,
            &config,
            gpu::GpuLogitsMode::GreedyArgmax,
        )
        .expect("GPU decode should succeed");
    }

    let decode_graph = gpu_scratch
        .decode_graph()
        .expect("greedy GPU decode should cache a reusable decode graph");
    assert_eq!(
        decode_graph.key().scope(),
        DecodeGraphScope::FullGreedyDecode,
        "greedy GPU decode should cache the full-token replay graph"
    );
}

#[test]
#[ignore = "manual profiling entry point for rocprofv3 and decode throughput checks"]
#[serial]
fn test_gpu_greedy_decode_profile_real_model() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let _lock = match common::GpuLock::acquire() {
        Ok(lock) => lock,
        Err(err) => {
            eprintln!("Skipping test: {}", err);
            return;
        }
    };

    require_gpu!();
    require_vram!(4);

    gpu::reset_decode_stage_profile();
    std::env::set_var("ROCMFORGE_PROFILE_DECODE_STAGES", "1");

    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse model config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("CPU weights should load");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("GPU weights should load");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());

    let prompt_tokens = tok.encode("Hello", false);
    assert!(!prompt_tokens.is_empty(), "prompt should tokenize");

    let decode_tokens = 64usize;
    let mut kv = GpuKvCache::new(&config, prompt_tokens.len() + decode_tokens)
        .expect("GPU KV should allocate");
    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("GPU scratch should allocate");
    let mut host_scratch = CpuForwardScratch::new(&config);

    let prefill_start = std::time::Instant::now();
    let mut next_token = None;
    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        gpu::gpu_embed_token_hybrid(
            &device,
            token_id,
            &gpu_weights,
            &cpu_weights,
            &mut gpu_scratch,
            &mut host_scratch,
            &config,
        )
        .expect("GPU embed should succeed");
        next_token = gpu::gpu_full_forward_hybrid(
            &device,
            &gpu_weights,
            &cpu_weights,
            &mut kv,
            &mut gpu_scratch,
            &mut host_scratch,
            pos,
            &config,
            gpu::GpuLogitsMode::GreedyArgmax,
        )
        .expect("GPU prompt decode should succeed");
    }
    let prefill_elapsed = prefill_start.elapsed();

    let mut token = next_token.expect("prompt decode should produce a greedy token");
    let decode_start = std::time::Instant::now();
    for step in 0..decode_tokens {
        gpu::gpu_embed_token_hybrid(
            &device,
            token,
            &gpu_weights,
            &cpu_weights,
            &mut gpu_scratch,
            &mut host_scratch,
            &config,
        )
        .expect("GPU embed should succeed");
        token = gpu::gpu_full_forward_hybrid(
            &device,
            &gpu_weights,
            &cpu_weights,
            &mut kv,
            &mut gpu_scratch,
            &mut host_scratch,
            prompt_tokens.len() + step,
            &config,
            gpu::GpuLogitsMode::GreedyArgmax,
        )
        .expect("GPU decode should succeed")
        .expect("decode step should produce a greedy token");
    }
    let decode_elapsed = decode_start.elapsed();

    let prefill_tok_s = prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64();
    let decode_tok_s = decode_tokens as f64 / decode_elapsed.as_secs_f64();
    eprintln!(
        "PROFILE gpu_greedy_decode_real_model prompt_tokens={} decode_tokens={} prefill_ms={:.2} prefill_tok_s={:.1} decode_ms={:.2} decode_tok_s={:.1}",
        prompt_tokens.len(),
        decode_tokens,
        prefill_elapsed.as_secs_f64() * 1000.0,
        prefill_tok_s,
        decode_elapsed.as_secs_f64() * 1000.0,
        decode_tok_s,
    );

    let stage_profile = gpu::decode_stage_profile_snapshot();
    eprintln!(
        "PROFILE decode_stage_counts layers={} tails={}",
        stage_profile.layer_invocations, stage_profile.tail_invocations
    );
    for (name, nanos) in [
        ("attn_norm", stage_profile.attn_norm_ns),
        ("qkv", stage_profile.qkv_ns),
        ("q_rope", stage_profile.q_rope_ns),
        ("k_rope", stage_profile.k_rope_ns),
        ("kv_write", stage_profile.kv_write_ns),
        ("attention", stage_profile.attention_ns),
        ("attn_proj", stage_profile.attn_proj_ns),
        ("attn_residual", stage_profile.attn_residual_ns),
        ("ffn_norm", stage_profile.ffn_norm_ns),
        ("gate_up", stage_profile.gate_up_ns),
        ("ffn_down", stage_profile.ffn_down_ns),
        ("ffn_residual", stage_profile.ffn_residual_ns),
        ("logits_norm", stage_profile.logits_norm_ns),
        ("logits_proj", stage_profile.logits_proj_ns),
        ("argmax", stage_profile.argmax_ns),
    ] {
        eprintln!(
            "PROFILE decode_stage name={} ms={:.3}",
            name,
            nanos as f64 / 1_000_000.0
        );
    }

    assert_eq!(
        stage_profile.layer_invocations,
        (prompt_tokens.len() + decode_tokens) as u64 * config.num_layers as u64,
        "stage profiler should see one layer profile entry per decode layer invocation"
    );
    assert_eq!(
        stage_profile.tail_invocations,
        (prompt_tokens.len() + decode_tokens) as u64,
        "stage profiler should see one logits tail per decode step"
    );
    assert!(
        gpu_scratch.decode_graph().is_none(),
        "stage profiling disables decode graph replay so direct-path timings are meaningful"
    );

    std::env::remove_var("ROCMFORGE_PROFILE_DECODE_STAGES");
}

#[test]
#[serial]
#[ignore]
fn test_gpu_greedy_decode_benchmark_real_model_multi_run() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let _lock = match common::GpuLock::acquire() {
        Ok(lock) => lock,
        Err(err) => {
            eprintln!("Skipping test: {}", err);
            return;
        }
    };

    require_gpu!();
    require_vram!(4);

    let runs = std::env::var("ROCMFORGE_BENCH_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(5);
    let warmup_runs = std::env::var("ROCMFORGE_BENCH_WARMUP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let decode_tokens = std::env::var("ROCMFORGE_BENCH_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(64);

    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse model config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("CPU weights should load");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("GPU weights should load");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());

    let prompt_tokens = tok.encode("Hello", false);
    assert!(!prompt_tokens.is_empty(), "prompt should tokenize");

    let mut prefill_tok_s_samples = Vec::with_capacity(runs);
    let mut decode_tok_s_samples = Vec::with_capacity(runs);

    for run_idx in 0..(warmup_runs + runs) {
        let mut kv = GpuKvCache::new(&config, prompt_tokens.len() + decode_tokens)
            .expect("GPU KV should allocate");
        let mut gpu_scratch = GpuForwardScratch::new(&config).expect("GPU scratch should allocate");
        let mut host_scratch = CpuForwardScratch::new(&config);

        let prefill_start = std::time::Instant::now();
        let mut next_token = None;
        for (pos, &token_id) in prompt_tokens.iter().enumerate() {
            gpu::gpu_embed_token_hybrid(
                &device,
                token_id,
                &gpu_weights,
                &cpu_weights,
                &mut gpu_scratch,
                &mut host_scratch,
                &config,
            )
            .expect("GPU embed should succeed");
            next_token = gpu::gpu_full_forward_hybrid(
                &device,
                &gpu_weights,
                &cpu_weights,
                &mut kv,
                &mut gpu_scratch,
                &mut host_scratch,
                pos,
                &config,
                gpu::GpuLogitsMode::GreedyArgmax,
            )
            .expect("GPU prompt decode should succeed");
        }
        let prefill_elapsed = prefill_start.elapsed();

        let mut token = next_token.expect("prompt decode should produce a greedy token");
        let decode_start = std::time::Instant::now();
        for step in 0..decode_tokens {
            gpu::gpu_embed_token_hybrid(
                &device,
                token,
                &gpu_weights,
                &cpu_weights,
                &mut gpu_scratch,
                &mut host_scratch,
                &config,
            )
            .expect("GPU embed should succeed");
            token = gpu::gpu_full_forward_hybrid(
                &device,
                &gpu_weights,
                &cpu_weights,
                &mut kv,
                &mut gpu_scratch,
                &mut host_scratch,
                prompt_tokens.len() + step,
                &config,
                gpu::GpuLogitsMode::GreedyArgmax,
            )
            .expect("GPU decode should succeed")
            .expect("decode step should produce a greedy token");
        }
        let decode_elapsed = decode_start.elapsed();

        let prefill_tok_s = prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64();
        let decode_tok_s = decode_tokens as f64 / decode_elapsed.as_secs_f64();

        if run_idx < warmup_runs {
            eprintln!(
                "BENCH gpu_greedy_decode_real_model warmup_run={} prefill_tok_s={:.1} decode_tok_s={:.1}",
                run_idx + 1,
                prefill_tok_s,
                decode_tok_s,
            );
            continue;
        }

        let sample_idx = run_idx - warmup_runs + 1;
        eprintln!(
            "BENCH gpu_greedy_decode_real_model run={} prefill_tok_s={:.1} decode_tok_s={:.1}",
            sample_idx, prefill_tok_s, decode_tok_s,
        );
        prefill_tok_s_samples.push(prefill_tok_s);
        decode_tok_s_samples.push(decode_tok_s);
    }

    let prefill_avg = mean(&prefill_tok_s_samples);
    let prefill_stddev = stddev(&prefill_tok_s_samples, prefill_avg);
    let decode_avg = mean(&decode_tok_s_samples);
    let decode_stddev = stddev(&decode_tok_s_samples, decode_avg);
    let decode_min = decode_tok_s_samples
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let decode_max = decode_tok_s_samples
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    eprintln!(
        "BENCH gpu_greedy_decode_real_model summary runs={} warmup_runs={} prompt_tokens={} decode_tokens={} prefill_avg_tok_s={:.1} prefill_stddev={:.1} decode_avg_tok_s={:.1} decode_stddev={:.1} decode_min_tok_s={:.1} decode_max_tok_s={:.1}",
        runs,
        warmup_runs,
        prompt_tokens.len(),
        decode_tokens,
        prefill_avg,
        prefill_stddev,
        decode_avg,
        decode_stddev,
        decode_min,
        decode_max,
    );
}

#[test]
#[serial]
fn test_gpu_prefill_real_model_matches_cpu_greedy_token() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let _lock = match common::GpuLock::acquire() {
        Ok(lock) => lock,
        Err(err) => {
            eprintln!("Skipping test: {}", err);
            return;
        }
    };

    require_gpu!();
    require_vram!(4);

    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse model config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("CPU weights should load");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("GPU weights should load");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());

    let template =
        detect_chat_template(&config.architecture, file.tokenizer_data().model.as_deref());
    let prompt = template.apply("Hello");
    let prompt_tokens = tok.encode(&prompt, false);
    assert!(!prompt_tokens.is_empty(), "prompt should tokenize");

    let cpu_logits = run_cpu_prompt_reference(&prompt_tokens, &cpu_weights, &config);
    let cpu_next = cpu_sample_greedy(&cpu_logits);

    let mut kv =
        GpuKvCache::new(&config, prompt_tokens.len().max(1)).expect("GPU KV should allocate");
    let mut prefill =
        gpu::GpuPrefillScratch::new(&config, prompt_tokens.len()).expect("GPU prefill scratch");
    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("GPU scratch should allocate");
    let mut host_scratch = CpuForwardScratch::new(&config);

    gpu::gpu_prefill_forward_hybrid(
        &device,
        &gpu_weights,
        &cpu_weights,
        &mut kv,
        &mut prefill,
        &mut gpu_scratch,
        &mut host_scratch,
        &prompt_tokens,
        0,
        &config,
        gpu::GpuLogitsMode::DownloadToHost,
    )
    .expect("GPU batched prefill should succeed");

    let gpu_next = cpu_sample_greedy(&host_scratch.logits);
    if gpu_next != cpu_next {
        let logits_max_abs_error = max_abs_error(&cpu_logits, &host_scratch.logits);
        let gpu_prefill_hidden =
            download_gpu_f32(&prefill.hidden, prompt_tokens.len() * config.hidden_size);
        let gpu_prefill_last_hidden =
            &gpu_prefill_hidden[(prompt_tokens.len() - 1) * config.hidden_size..];

        let mut kv_decode = GpuKvCache::new(&config, prompt_tokens.len()).expect("GPU decode KV");
        let mut scratch_decode = GpuForwardScratch::new(&config).expect("GPU decode scratch");
        let mut host_decode = CpuForwardScratch::new(&config);
        for (pos, &token_id) in prompt_tokens.iter().enumerate() {
            gpu::gpu_embed_token_hybrid(
                &device,
                token_id,
                &gpu_weights,
                &cpu_weights,
                &mut scratch_decode,
                &mut host_decode,
                &config,
            )
            .expect("GPU decode embed should succeed");
            gpu::gpu_full_forward_hybrid(
                &device,
                &gpu_weights,
                &cpu_weights,
                &mut kv_decode,
                &mut scratch_decode,
                &mut host_decode,
                pos,
                &config,
                gpu::GpuLogitsMode::Skip,
            )
            .expect("GPU decode prompt step should succeed");
        }
        let gpu_decode_last_hidden = download_gpu_f32(&scratch_decode.hidden, config.hidden_size);
        let decode_vs_prefill_hidden_err =
            max_abs_error(&gpu_decode_last_hidden, gpu_prefill_last_hidden);

        let cpu_prompt_embeddings =
            build_cpu_prompt_embeddings(&prompt_tokens, &cpu_weights, &config);
        let mut decode_embeddings = vec![0.0f32; prompt_tokens.len() * config.hidden_size];
        for (row, &token_id) in prompt_tokens.iter().enumerate() {
            gpu::gpu_embed_token_hybrid(
                &device,
                token_id,
                &gpu_weights,
                &cpu_weights,
                &mut scratch_decode,
                &mut host_decode,
                &config,
            )
            .expect("GPU decode embedding should succeed");
            let row_hidden = download_gpu_f32(&scratch_decode.hidden, config.hidden_size);
            decode_embeddings[row * config.hidden_size..(row + 1) * config.hidden_size]
                .copy_from_slice(&row_hidden);
        }

        let token_ids_i32: Vec<i32> = prompt_tokens.iter().map(|&id| id as i32).collect();
        let mut d_token_ids =
            GpuBuffer::alloc(token_ids_i32.len() * std::mem::size_of::<i32>()).expect("alloc ids");
        d_token_ids
            .copy_from_host(unsafe {
                std::slice::from_raw_parts(
                    token_ids_i32.as_ptr() as *const u8,
                    token_ids_i32.len() * std::mem::size_of::<i32>(),
                )
            })
            .expect("upload ids");
        let d_embed_batch =
            GpuBuffer::alloc(prompt_tokens.len() * config.hidden_size * std::mem::size_of::<f32>())
                .expect("alloc embed batch");
        gpu::embed_q8_0_batch(
            gpu_weights.token_emb.as_ptr(),
            d_token_ids.as_ptr() as *const i32,
            d_embed_batch.as_ptr() as *mut f32,
            config.hidden_size,
            config.vocab_size,
            prompt_tokens.len(),
        )
        .expect("real-model batch embedding should succeed");
        let gpu_batch_embeddings =
            download_gpu_f32(&d_embed_batch, prompt_tokens.len() * config.hidden_size);
        let embed_batch_vs_decode_err = max_abs_error(&gpu_batch_embeddings, &decode_embeddings);
        let mut kv_prefill_l0 =
            GpuKvCache::new(&config, prompt_tokens.len()).expect("GPU prefill L0 KV");
        let mut prefill_l0 = gpu::GpuPrefillScratch::new(&config, prompt_tokens.len())
            .expect("GPU prefill L0 scratch");
        prefill_l0
            .hidden
            .copy_from_host(unsafe {
                std::slice::from_raw_parts(
                    gpu_batch_embeddings.as_ptr() as *const u8,
                    gpu_batch_embeddings.len() * std::mem::size_of::<f32>(),
                )
            })
            .expect("Upload prompt embeddings");
        gpu::gpu_prefill_layer_forward_hybrid(
            &device,
            gpu_weights.layer(0),
            &mut kv_prefill_l0,
            &mut prefill_l0,
            0,
            0,
            &config,
        )
        .expect("GPU batched layer0 should succeed");
        let gpu_prefill_l0_hidden =
            download_gpu_f32(&prefill_l0.hidden, prompt_tokens.len() * config.hidden_size);

        let mut kv_decode_l0 =
            GpuKvCache::new(&config, prompt_tokens.len()).expect("GPU decode L0 KV");
        let mut scratch_decode_l0 = GpuForwardScratch::new(&config).expect("GPU decode L0 scratch");
        let mut host_decode_l0 = CpuForwardScratch::new(&config);
        let mut gpu_decode_l0_hidden = vec![0.0f32; prompt_tokens.len() * config.hidden_size];
        let q_size = config.num_heads * config.head_dim;
        let kv_size = config.num_kv_heads * config.head_dim;
        let ff_size = config.intermediate_size;
        let mut gpu_decode_l0_normed = vec![0.0f32; prompt_tokens.len() * config.hidden_size];
        let mut gpu_decode_l0_q = vec![0.0f32; prompt_tokens.len() * q_size];
        let mut gpu_decode_l0_k = vec![0.0f32; prompt_tokens.len() * kv_size];
        let mut gpu_decode_l0_v = vec![0.0f32; prompt_tokens.len() * kv_size];
        let mut gpu_decode_l0_attn_out = vec![0.0f32; prompt_tokens.len() * q_size];
        let mut gpu_decode_l0_layer_out = vec![0.0f32; prompt_tokens.len() * config.hidden_size];
        let mut gpu_decode_l0_gate = vec![0.0f32; prompt_tokens.len() * ff_size];
        let mut gpu_decode_l0_swiglu = vec![0.0f32; prompt_tokens.len() * ff_size];
        for (pos, &token_id) in prompt_tokens.iter().enumerate() {
            gpu::gpu_embed_token_hybrid(
                &device,
                token_id,
                &gpu_weights,
                &cpu_weights,
                &mut scratch_decode_l0,
                &mut host_decode_l0,
                &config,
            )
            .expect("GPU decode L0 embed should succeed");
            gpu::gpu_layer_forward_hybrid(
                &device,
                gpu_weights.layer(0),
                &mut kv_decode_l0,
                &mut scratch_decode_l0,
                0,
                pos,
                &config,
            )
            .expect("GPU decode L0 step should succeed");
            let row_hidden = download_gpu_f32(&scratch_decode_l0.hidden, config.hidden_size);
            gpu_decode_l0_hidden[pos * config.hidden_size..(pos + 1) * config.hidden_size]
                .copy_from_slice(&row_hidden);
            let row_normed = download_gpu_f32(&scratch_decode_l0.normed, config.hidden_size);
            gpu_decode_l0_normed[pos * config.hidden_size..(pos + 1) * config.hidden_size]
                .copy_from_slice(&row_normed);
            let row_q = download_gpu_f32(&scratch_decode_l0.q, q_size);
            gpu_decode_l0_q[pos * q_size..(pos + 1) * q_size].copy_from_slice(&row_q);
            let row_k = download_gpu_f32(&scratch_decode_l0.k, kv_size);
            gpu_decode_l0_k[pos * kv_size..(pos + 1) * kv_size].copy_from_slice(&row_k);
            let row_v = download_gpu_f32(&scratch_decode_l0.v, kv_size);
            gpu_decode_l0_v[pos * kv_size..(pos + 1) * kv_size].copy_from_slice(&row_v);
            let row_attn_out = download_gpu_f32(&scratch_decode_l0.attn_out, q_size);
            gpu_decode_l0_attn_out[pos * q_size..(pos + 1) * q_size].copy_from_slice(&row_attn_out);
            let row_layer_out = download_gpu_f32(&scratch_decode_l0.layer_out, config.hidden_size);
            gpu_decode_l0_layer_out[pos * config.hidden_size..(pos + 1) * config.hidden_size]
                .copy_from_slice(&row_layer_out);
            let row_gate = download_gpu_f32(&scratch_decode_l0.gate, ff_size);
            gpu_decode_l0_gate[pos * ff_size..(pos + 1) * ff_size].copy_from_slice(&row_gate);
            let row_swiglu = download_gpu_f32(&scratch_decode_l0.swiglu, ff_size);
            gpu_decode_l0_swiglu[pos * ff_size..(pos + 1) * ff_size].copy_from_slice(&row_swiglu);
        }
        let layer0_prefill_vs_decode_err =
            max_abs_error(&gpu_decode_l0_hidden, &gpu_prefill_l0_hidden);
        let gpu_prefill_l0_normed =
            download_gpu_f32(&prefill_l0.normed, prompt_tokens.len() * config.hidden_size);
        let gpu_prefill_l0_q = download_gpu_f32(&prefill_l0.q, prompt_tokens.len() * q_size);
        let gpu_prefill_l0_k = download_gpu_f32(&prefill_l0.k, prompt_tokens.len() * kv_size);
        let gpu_prefill_l0_v = download_gpu_f32(&prefill_l0.v, prompt_tokens.len() * kv_size);
        let gpu_prefill_l0_attn_out =
            download_gpu_f32(&prefill_l0.attn_out, prompt_tokens.len() * q_size);
        let gpu_prefill_l0_layer_out = download_gpu_f32(
            &prefill_l0.layer_out,
            prompt_tokens.len() * config.hidden_size,
        );
        let gpu_prefill_l0_gate = download_gpu_f32(&prefill_l0.gate, prompt_tokens.len() * ff_size);
        let gpu_prefill_l0_swiglu =
            download_gpu_f32(&prefill_l0.swiglu, prompt_tokens.len() * ff_size);
        let layer0_normed_err = max_abs_error(&gpu_decode_l0_normed, &gpu_prefill_l0_normed);
        let layer0_q_err = max_abs_error(&gpu_decode_l0_q, &gpu_prefill_l0_q);
        let layer0_k_err = max_abs_error(&gpu_decode_l0_k, &gpu_prefill_l0_k);
        let layer0_v_err = max_abs_error(&gpu_decode_l0_v, &gpu_prefill_l0_v);
        let layer0_attn_out_err = max_abs_error(&gpu_decode_l0_attn_out, &gpu_prefill_l0_attn_out);
        let layer0_layer_out_err =
            max_abs_error(&gpu_decode_l0_layer_out, &gpu_prefill_l0_layer_out);
        let layer0_gate_err = max_abs_error(&gpu_decode_l0_gate, &gpu_prefill_l0_gate);
        let layer0_swiglu_err = max_abs_error(&gpu_decode_l0_swiglu, &gpu_prefill_l0_swiglu);

        eprintln!(
            "GPU batched prefill mismatch: cpu_next={} gpu_next={} logits_max_abs_error={:.6} decode_vs_prefill_hidden={:.6} embed_batch_vs_decode={:.6} layer0_hidden={:.6} normed={:.6} q={:.6} k={:.6} v={:.6} attn_out={:.6} layer_out={:.6} gate={:.6} swiglu={:.6}",
            cpu_next,
            gpu_next,
            logits_max_abs_error,
            decode_vs_prefill_hidden_err,
            embed_batch_vs_decode_err,
            layer0_prefill_vs_decode_err,
            layer0_normed_err,
            layer0_q_err,
            layer0_k_err,
            layer0_v_err,
            layer0_attn_out_err,
            layer0_layer_out_err,
            layer0_gate_err,
            layer0_swiglu_err
        );
    }

    assert_eq!(
        gpu_next, cpu_next,
        "GPU batched prefill and CPU greedy next-token should match"
    );
}

#[test]
#[serial]
fn test_gpu_ffn_down_real_model_matches_cpu_layer0_projection() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let _lock = match common::GpuLock::acquire() {
        Ok(lock) => lock,
        Err(err) => {
            eprintln!("Skipping test: {}", err);
            return;
        }
    };

    require_gpu!();
    require_vram!(4);

    let caps = gpu::detect().expect("GPU should be detected");
    let _device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse model config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("CPU weights should load");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("GPU weights should load");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());

    let template =
        detect_chat_template(&config.architecture, file.tokenizer_data().model.as_deref());
    let prompt = template.apply("Hello");
    let prompt_tokens = tok.encode(&prompt, false);
    assert!(!prompt_tokens.is_empty(), "prompt should tokenize");

    let first_token = prompt_tokens[0];
    let mut cpu_hidden = vec![0.0f32; config.hidden_size];
    cpu_embed_token(first_token, &cpu_weights, &mut cpu_hidden, &config);

    let mut cpu_kv = CpuKvCache::new(&config, 1);
    let mut cpu_scratch = CpuForwardScratch::new(&config);
    cpu_layer_forward(
        &mut cpu_hidden,
        cpu_weights.layer(0),
        &mut cpu_kv,
        &mut cpu_scratch,
        0,
        0,
        &config,
        false,
    )
    .expect("CPU layer 0 should succeed");

    let layer = gpu_weights.layer(0);
    assert_eq!(
        layer.ffn_down_meta.wtype,
        rocmforge::loader::GgmlType::Q4_1,
        "expected layer-0 ffn_down to be Q4_1 for this regression"
    );

    let ff_size = config.intermediate_size;
    let hidden_size = config.hidden_size;

    let mut d_input =
        GpuBuffer::alloc(ff_size * std::mem::size_of::<f32>()).expect("alloc ffn input");
    let d_output =
        GpuBuffer::alloc(hidden_size * std::mem::size_of::<f32>()).expect("alloc ffn output");

    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            cpu_scratch.swiglu.as_ptr() as *const u8,
            ff_size * std::mem::size_of::<f32>(),
        )
    };
    d_input
        .copy_from_host(input_bytes)
        .expect("upload swiglu input");

    gpu::gemv_q4_1_f32(
        layer.ffn_down.as_ptr(),
        d_input.as_ptr() as *const f32,
        d_output.as_ptr() as *mut f32,
        ff_size,
        hidden_size,
    )
    .expect("GPU ffn_down GEMV should succeed");

    let mut output_bytes = vec![0u8; hidden_size * std::mem::size_of::<f32>()];
    d_output
        .copy_to_host(&mut output_bytes)
        .expect("download ffn output");
    let gpu_output: Vec<f32> = unsafe {
        std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, hidden_size).to_vec()
    };

    let max_err = max_abs_error(&cpu_scratch.layer_out, &gpu_output);
    assert!(
        max_err <= 1e-3,
        "real-model ffn_down projection mismatch: max_abs_error={}",
        max_err
    );
}
