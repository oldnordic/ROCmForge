use std::io::Write;
use std::time::Instant;

use rocmforge::cpu::forward_graph_trace::{ForwardGraphRecorder, TraceComponent};
use rocmforge::cpu::sampler::{cpu_sample_greedy, cpu_sample_top_p};
use rocmforge::gpu;

use super::cli::Args;
use super::debug::print_top_k_tokens;
use super::gpu_inference_setup::prepare_gpu_inference_state;
use super::gpu_prompt_decode::run_decode_style_prompt_path;
use super::gpu_runtime::prepare_gpu_runtime;
use super::gpu_setup::prepare_gpu_prompt;

#[cfg(feature = "cpu-graph")]
use rocmforge::cpu::graph::{compute_score, GpuTraceEntry, GraphMap, ScoreMetric};

/// Estimate bytes that will actually reside on the GPU for this model.
pub(crate) fn estimate_gpu_resident_model_bytes(
    model_path: &str,
    file: &rocmforge::loader::ModelFile,
    config: &rocmforge::config::ModelConfig,
) -> usize {
    use rocmforge::config::TensorNamingScheme;
    // Non-MoE: every weight lives on the GPU; fall back to file size.
    if !matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
        return std::fs::metadata(model_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
    }

    let tensor_bytes = |name: &str| -> usize { file.tensor_byte_size(name) };

    let mut total = 0usize;

    // Fixed weights always on the GPU.
    for name in &["token_emb.weight", "output_norm.weight", "output.weight"] {
        total += tensor_bytes(name);
    }

    for layer in 0..config.num_layers {
        // Attention + layer norms (always GPU-resident).
        for suffix in &[
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_qkv.weight", // fused QKV variant
            "attn_norm.weight",
            "ffn_norm.weight",
            "post_attention_norm.weight",
        ] {
            total += tensor_bytes(&format!("blk.{}.{}", layer, suffix));
        }
        // MoE router (tiny, GPU-resident).
        total += tensor_bytes(&format!("blk.{}.ffn_gate_inp.weight", layer));
        // Shared expert (small, GPU-resident — not part of the expert pool).
        for suffix in &[
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
            "ffn_gate_inp_shexp.weight",
        ] {
            total += tensor_bytes(&format!("blk.{}.{}", layer, suffix));
        }
        // NOTE: ffn_gate_exps / ffn_up_exps / ffn_down_exps are CPU-resident
        // (CpuCompressedExperts) and must NOT be counted here.
    }

    total
}

pub(crate) fn run_gpu_inference(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "cpu-graph"))]
    if args.graph_map_dir.is_some() || args.load_graph_map_dir.is_some() {
        return Err(
            "--graph-map-dir and --load-graph-map-dir require the cpu-graph feature".into(),
        );
    }

    let runtime = prepare_gpu_runtime(true)?;
    let gpu_caps = runtime.gpu_caps;
    let vram_session = runtime.vram_session;
    let device = runtime.device;

    // ── VRAM pre-flight ───────────────────────────────────────────────────────────
    let file = rocmforge::loader::ModelFile::open(&args.model)?;
    let config = file.config()?;

    let model_file_bytes = estimate_gpu_resident_model_bytes(&args.model, &file, &config);
    let max_seq_estimate = args
        .ctx_size
        .unwrap_or_else(|| config.max_seq_len.min(args.max_tokens + 2048));
    let kv_estimate = gpu::GpuKvCache::estimate_bytes(&config, max_seq_estimate);
    let scratch_estimate = gpu::GpuForwardScratch::estimate_bytes(&config);

    vram_session.print_startup_report(model_file_bytes, kv_estimate, scratch_estimate);
    vram_session
        .check_fits(model_file_bytes, kv_estimate, scratch_estimate)
        .map_err(|e| format!("Insufficient VRAM: {}", e))?;
    // ── end VRAM pre-flight ───────────────────────────────────────────────────────

    let setup = prepare_gpu_prompt(&file, &config, args)?;
    let tok = setup.tok;
    let prompt_tokens = setup.prompt_tokens;
    let max_seq = setup.max_seq;

    let mut recorder = args
        .forward_graph_trace
        .as_ref()
        .map(|_| ForwardGraphRecorder::new(&prompt_tokens));
    if let (Some(recorder), Some(json)) = (recorder.as_mut(), args.expected_attention.as_ref()) {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| format!("--expected-attention is not valid JSON: {}", e))?;
        recorder.set_expected_attention(value);
    }

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

    let setup_state =
        prepare_gpu_inference_state(&file, &config, args, gpu_caps.device_id, max_seq)?;
    let cpu_weights = setup_state.cpu_weights;
    let gpu_weights = setup_state.gpu_weights;
    let mut kv = setup_state.kv;
    let mut gpu_scratch = setup_state.gpu_scratch;
    let mut host_scratch = setup_state.host_scratch;
    let use_greedy = setup_state.use_greedy;
    let use_gpu_greedy_fastpath = setup_state.use_gpu_greedy_fastpath;
    let t_prefill = Instant::now();

    #[cfg(feature = "cpu-graph")]
    if let Some(load_dir) = &args.load_graph_map_dir {
        let map = GraphMap::load(std::path::Path::new(load_dir))?;
        eprintln!("Loaded GraphMap from {}", load_dir);
        eprintln!("  branches: {}", map.branch_scores().len());
        eprintln!("  gpu trace tokens: {}", map.gpu_trace().len());
    }

    // When capturing a GraphMap we need logits on the host, so disable the GPU
    // greedy fastpath for the decode loop.  Prefill may still use it.
    #[cfg(feature = "cpu-graph")]
    let use_gpu_greedy_fastpath = use_gpu_greedy_fastpath && args.graph_map_dir.is_none();

    #[cfg(feature = "cpu-graph")]
    let score_metric = ScoreMetric::from_name(&args.graph_score_metric);
    #[cfg(feature = "cpu-graph")]
    let mut gpu_trace: Vec<GpuTraceEntry> = Vec::new();
    #[cfg(feature = "cpu-graph")]
    let mut gpu_score_log: Vec<(u64, ScoreMetric, f32)> = Vec::new();

    // Forward-graph tracing requires the standard decode path (host logits and
    // the instrumented attention kernel), so disable fastpaths when tracing.
    let use_gpu_greedy_fastpath = use_gpu_greedy_fastpath && recorder.is_none();
    let final_prompt_logits_mode = if use_gpu_greedy_fastpath {
        gpu::GpuLogitsMode::GreedyArgmax
    } else {
        gpu::GpuLogitsMode::DownloadToHost
    };

    // ── Hotpath Router ────────────────────────────────────────────────────────────
    let profile = gpu::ModelProfile::from_weights(&gpu_weights, &config);
    let path = gpu::select_path(&profile, prompt_tokens.len(), &vram_session);
    let path = if recorder.is_some() {
        gpu::InferencePath::DecodeStyle
    } else {
        path
    };
    eprintln!("[Router] Model profile: {}", profile.summary());
    eprintln!("[Router] Selected path: {}", path);

    if let Err(e) = gpu::check_path_vram(&path, &config, prompt_tokens.len(), &vram_session) {
        eprintln!(
            "[Router] Path VRAM check failed ({}), falling back to DecodeStyle",
            e
        );
    }

    let prompt_next_token = match path {
        gpu::InferencePath::BatchedPrefill { .. } => {
            match gpu::GpuPrefillScratch::new(&config, prompt_tokens.len()) {
                Ok(mut prefill_scratch) => {
                    eprintln!(
                        "Using batched GPU prefill for Q4_0 model ({} tokens)",
                        prompt_tokens.len()
                    );
                    match gpu::gpu_batched_prefill_forward(
                        device,
                        &gpu_weights,
                        &cpu_weights,
                        &mut kv,
                        &mut prefill_scratch,
                        &mut host_scratch,
                        &prompt_tokens,
                        0,
                        &config,
                        final_prompt_logits_mode,
                    ) {
                        Ok(token) => token,
                        Err(err) => {
                            eprintln!(
                                "Batched GPU prefill failed ({}), falling back to decode-style prompt path",
                                err
                            );
                            run_decode_style_prompt_path(
                                device,
                                &gpu_weights,
                                &cpu_weights,
                                &mut kv,
                                &mut gpu_scratch,
                                &mut host_scratch,
                                &prompt_tokens,
                                &config,
                                final_prompt_logits_mode,
                            )?
                        }
                    }
                }
                Err(err) => {
                    eprintln!(
                        "Batched GPU prefill scratch allocation failed ({}), falling back to decode-style prompt path",
                        err
                    );
                    run_decode_style_prompt_path(
                        device,
                        &gpu_weights,
                        &cpu_weights,
                        &mut kv,
                        &mut gpu_scratch,
                        &mut host_scratch,
                        &prompt_tokens,
                        &config,
                        final_prompt_logits_mode,
                    )?
                }
            }
        }
        gpu::InferencePath::SvdOptimized => {
            eprintln!("Using SVD-optimized decode-style path");
            run_decode_style_prompt_path(
                device,
                &gpu_weights,
                &cpu_weights,
                &mut kv,
                &mut gpu_scratch,
                &mut host_scratch,
                &prompt_tokens,
                &config,
                final_prompt_logits_mode,
            )?
        }
        gpu::InferencePath::DecodeStyle | gpu::InferencePath::CpuFallback { .. } => {
            if !gpu_weights.uses_q4_0_quantization() {
                eprintln!(
                    "Batched GPU prefill only available for Q4_0 models, using decode-style prompt path"
                );
            } else if prompt_tokens.len() == 1 {
                eprintln!("Single token prompt, using decode-style path");
            } else {
                eprintln!(
                    "Prompt too long for batched prefill ({}), using decode-style path",
                    prompt_tokens.len()
                );
            }
            run_decode_style_prompt_path(
                device,
                &gpu_weights,
                &cpu_weights,
                &mut kv,
                &mut gpu_scratch,
                &mut host_scratch,
                &prompt_tokens,
                &config,
                final_prompt_logits_mode,
            )?
        }
    };

    if args.debug {
        print_top_k_tokens(&host_scratch.logits, &tok, 10);
    }

    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "Prefill: {:.1}ms ({:.1} tok/s)",
        prefill_ms,
        prompt_tokens.len() as f64 / prefill_ms * 1000.0
    );

    if let Some(ref dump_path) = args.kv_dump {
        eprint!("Dumping KV cache to {}... ", dump_path);
        kv.dump_to_file(
            std::path::Path::new(dump_path),
            prompt_tokens.len(),
            config.num_kv_heads,
            config.head_dim,
        )?;
        eprintln!(
            "done ({} layers × {} tokens × {} head_dim)",
            config.num_layers,
            prompt_tokens.len(),
            config.head_dim
        );
    }

    if args.prefill_only_validate {
        let logits = &host_scratch.logits;
        let has_nan = logits.iter().any(|l| l.is_nan());
        let has_inf = logits.iter().any(|l| l.is_infinite());
        let has_finite = logits.iter().any(|l| l.is_finite());

        if has_nan || has_inf {
            eprintln!("PREFILL_ONLY_VALIDATE: FAILED - logits contain NaN or Inf");
            std::process::exit(1);
        }

        if !has_finite {
            eprintln!("PREFILL_ONLY_VALIDATE: FAILED - no finite logits");
            std::process::exit(1);
        }

        let used_batched_prefill = gpu_weights.uses_q4_0_quantization()
            && prompt_tokens.len() > 1
            && prompt_tokens.len() <= 512;

        if used_batched_prefill {
            eprintln!("PREFILL_ONLY_VALIDATE: PASSED");
            eprintln!("  Batched prefill: exercised");
            eprintln!("  Prompt tokens: {}", prompt_tokens.len());
            eprintln!("  Prefill time: {:.1}ms", prefill_ms);
            eprintln!(
                "  Throughput: {:.1} tok/s",
                prompt_tokens.len() as f64 / prefill_ms * 1000.0
            );
        } else {
            eprintln!("PREFILL_ONLY_VALIDATE: PASSED (decode-style path)");
            eprintln!("  Batched prefill: not exercised (non-Q4_0 or single token)");
            eprintln!("  Prompt tokens: {}", prompt_tokens.len());
            eprintln!("  Prefill time: {:.1}ms", prefill_ms);
        }

        std::process::exit(0);
    }

    let mut pos = prompt_tokens.len();
    let mut n_generated = 0usize;
    let t_gen = Instant::now();
    let mut seed = 0xdeadbeef_u64;

    let mut next_token = if use_greedy {
        if use_gpu_greedy_fastpath {
            prompt_next_token.expect("greedy GPU prompt pass should return next token")
        } else {
            cpu_sample_greedy(&host_scratch.logits)
        }
    } else {
        seed = seed.wrapping_add(1);
        cpu_sample_top_p(&host_scratch.logits, args.temperature, args.top_p, seed)
    };

    println!();

    gpu_scratch.set_forward_graph_recorder(recorder.as_mut());

    loop {
        if tok.is_eog(next_token) || n_generated >= args.max_tokens || pos >= max_seq {
            break;
        }

        #[cfg(feature = "cpu-graph")]
        let trace_input_token = next_token;
        #[cfg(feature = "cpu-graph")]
        let trace_pos = pos;

        let text = tok.decode_token(next_token);
        if args.debug {
            eprintln!("[Generated] token_id={} text={:?}", next_token, text);
        }
        print!("{}", text);
        std::io::stdout().flush().ok();
        n_generated += 1;

        gpu::gpu_embed_token_hybrid(
            device,
            next_token,
            &gpu_weights,
            &cpu_weights,
            &mut gpu_scratch,
            &mut host_scratch,
            &config,
        )
        .map_err(|e| format!("gpu embed: {}", e))?;

        if let Some(recorder) = recorder.as_mut() {
            let hidden = gpu_scratch
                .hidden
                .copy_to_host_vec()
                .map_err(|e| format!("download hidden for trace: {}", e))?;
            recorder.record_node(
                TraceComponent::InputEmbedding,
                0,
                Some(pos),
                &hidden[..config.hidden_size],
            );
        }

        let logits_mode = if use_gpu_greedy_fastpath {
            gpu::GpuLogitsMode::GreedyArgmax
        } else {
            gpu::GpuLogitsMode::DownloadToHost
        };
        let decode_next_token = gpu::gpu_full_forward_hybrid(
            device,
            &gpu_weights,
            &cpu_weights,
            &mut kv,
            &mut gpu_scratch,
            &mut host_scratch,
            pos,
            &config,
            logits_mode,
            next_token,
        )
        .map_err(|e| format!("gpu decode: {}", e))?;

        #[cfg(feature = "cpu-graph")]
        if args.graph_map_dir.is_some() {
            let logits = &host_scratch.logits[..config.vocab_size];
            let score = compute_score(score_metric, logits, None);
            gpu_score_log.push((n_generated as u64, score_metric, score));
        }

        if args.debug && n_generated <= 3 {
            eprintln!("\n[Token {} logits]", n_generated);
            print_top_k_tokens(&host_scratch.logits, &tok, 5);
        }

        let sampled_token = if let Some(token) = decode_next_token {
            token
        } else {
            device
                .synchronize()
                .map_err(|e| format!("gpu sync: {}", e))?;

            if use_greedy {
                if use_gpu_greedy_fastpath {
                    let token = gpu_scratch.argmax_result_index.as_slice::<i32>()[0];
                    if token < 0 || (token as usize) >= config.vocab_size {
                        return Err(
                            format!("gpu argmax returned out-of-range index {}", token).into()
                        );
                    }
                    token as u32
                } else {
                    cpu_sample_greedy(&host_scratch.logits)
                }
            } else {
                seed = seed.wrapping_add(1);
                cpu_sample_top_p(&host_scratch.logits, args.temperature, args.top_p, seed)
            }
        };

        if let Some(recorder) = recorder.as_mut() {
            recorder.record_confidence(
                pos,
                sampled_token,
                &host_scratch.logits[..config.vocab_size],
            );
            recorder.push_token(sampled_token);
        }
        pos += 1;

        #[cfg(feature = "cpu-graph")]
        if args.graph_map_dir.is_some() {
            let score = gpu_score_log.last().map(|(_, _, s)| *s).unwrap_or(0.0);
            gpu_trace.push(GpuTraceEntry {
                timestamp: n_generated as u64,
                pos: trace_pos,
                input_token_id: trace_input_token,
                sampled_token_id: sampled_token,
                score,
            });
        }

        next_token = sampled_token;
    }

    println!();

    gpu_scratch.clear_forward_graph_recorder();
    if let (Some(path), Some(recorder)) = (args.forward_graph_trace.as_ref(), recorder) {
        recorder.write_jsonl(path)?;
        eprintln!("Wrote forward graph trace to {}", path);
    }

    #[cfg(feature = "cpu-graph")]
    if let Some(save_dir) = &args.graph_map_dir {
        let map = GraphMap::from_gpu_trace(gpu_score_log, gpu_trace);
        map.save(std::path::Path::new(save_dir))?;
        eprintln!("Saved GraphMap to {}", save_dir);
        eprintln!("  gpu trace tokens: {}", map.gpu_trace().len());
        eprintln!("  branch scores: {}", map.branch_scores().len());
    }

    if n_generated > 0 {
        let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "\n{} tokens in {:.1}ms = {:.1} tok/s",
            n_generated,
            gen_ms,
            n_generated as f64 / gen_ms * 1000.0
        );
    } else {
        eprintln!("\n[EOS on first token]");
    }

    Ok(())
}

pub(crate) fn run_gpu_speculative_inference(
    args: &Args,
    draft_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "cpu-graph"))]
    if args.graph_map_dir.is_some() || args.load_graph_map_dir.is_some() {
        return Err(
            "--graph-map-dir and --load-graph-map-dir require the cpu-graph feature".into(),
        );
    }

    let runtime = prepare_gpu_runtime(false)?;
    let vram_session = runtime.vram_session;
    let device = runtime.device;

    // ── VRAM pre-flight for speculative co-execution ─────────────────────────────
    let file = rocmforge::loader::ModelFile::open(&args.model)?;
    let target_config = file.config()?;

    let draft_file = rocmforge::loader::ModelFile::open(draft_path)?;
    let draft_config = draft_file.config()?;

    let target_file_bytes = std::fs::metadata(&args.model)
        .map(|m| m.len() as usize)
        .unwrap_or(0);
    let draft_file_bytes = std::fs::metadata(draft_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    let max_seq_estimate = args
        .ctx_size
        .unwrap_or_else(|| target_config.max_seq_len.min(args.max_tokens + 2048));
    let target_kv_estimate = gpu::GpuKvCache::estimate_bytes(&target_config, max_seq_estimate);
    let draft_kv_estimate = gpu::GpuKvCache::estimate_bytes(&draft_config, max_seq_estimate);

    let target_scratch_estimate = gpu::GpuForwardScratch::estimate_bytes(&target_config)
        + gpu::GpuPrefillScratch::estimate_total_bytes(&target_config, max_seq_estimate);
    let draft_scratch_estimate = gpu::GpuForwardScratch::estimate_bytes(&draft_config)
        + gpu::GpuPrefillScratch::estimate_total_bytes(&draft_config, max_seq_estimate);

    let total_weights = target_file_bytes + draft_file_bytes;
    let total_kv = target_kv_estimate + draft_kv_estimate;
    let total_scratch = target_scratch_estimate + draft_scratch_estimate;

    vram_session.print_startup_report(total_weights, total_kv, total_scratch);
    vram_session
        .check_fits(total_weights, total_kv, total_scratch)
        .map_err(|e| format!("Insufficient VRAM for speculative decoding: {}", e))?;
    // ── end VRAM pre-flight ───────────────────────────────────────────────────────

    let setup = prepare_gpu_prompt(&file, &target_config, args)?;
    let tok = setup.tok;
    let prompt_tokens = setup.prompt_tokens;
    let max_seq = setup.max_seq;

    eprintln!("Co-loading models into GPU VRAM (Speculative Engine)...");
    let t_load = Instant::now();
    let mut engine = gpu::SpeculativeEngine::new(
        device,
        &args.model,
        draft_path,
        max_seq,
        prompt_tokens.len(),
    )
    .map_err(|e| format!("speculative engine construct: {}", e))?;
    eprintln!(
        "Speculative Engine initialized in {:.1}s",
        t_load.elapsed().as_secs_f64()
    );

    let t_prefill = Instant::now();
    let final_prompt_logits_mode = gpu::GpuLogitsMode::DownloadToHost;

    // Prefill Target Model
    let mut target_prompt_next_token = None;
    let can_use_batched_prefill_target = engine.target_model.uses_q4_0_quantization()
        && prompt_tokens.len() > 1
        && prompt_tokens.len() <= 512;

    if can_use_batched_prefill_target {
        eprintln!(
            "Using batched GPU prefill for Target model ({} tokens)",
            prompt_tokens.len()
        );
        match gpu::gpu_batched_prefill_forward(
            device,
            &engine.target_model,
            &engine.target_cpu_weights,
            &mut engine.target_kv,
            &mut engine.target_prefill_scratch,
            &mut engine.target_host_scratch,
            &prompt_tokens,
            0,
            &engine.target_config,
            final_prompt_logits_mode,
        ) {
            Ok(token) => target_prompt_next_token = token,
            Err(err) => {
                eprintln!(
                    "Batched GPU prefill failed for target ({}), falling back to decode-style path",
                    err
                );
            }
        }
    }

    if target_prompt_next_token.is_none() {
        for (pos, &token_id) in prompt_tokens.iter().enumerate() {
            gpu::gpu_embed_token_hybrid(
                device,
                token_id,
                &engine.target_model,
                &engine.target_cpu_weights,
                &mut engine.target_scratch,
                &mut engine.target_host_scratch,
                &engine.target_config,
            )
            .map_err(|e| format!("gpu embed target: {}", e))?;

            let logits_mode = if pos + 1 == prompt_tokens.len() {
                final_prompt_logits_mode
            } else {
                gpu::GpuLogitsMode::Skip
            };

            target_prompt_next_token = gpu::gpu_full_forward_hybrid(
                device,
                &engine.target_model,
                &engine.target_cpu_weights,
                &mut engine.target_kv,
                &mut engine.target_scratch,
                &mut engine.target_host_scratch,
                pos,
                &engine.target_config,
                logits_mode,
                token_id,
            )
            .map_err(|e| format!("gpu prefill target: {}", e))?;
        }
    }

    // Prefill Draft Model
    let mut draft_prefilled = false;
    let can_use_batched_prefill_draft = engine.draft_model.uses_q4_0_quantization()
        && prompt_tokens.len() > 1
        && prompt_tokens.len() <= 512;

    if can_use_batched_prefill_draft {
        eprintln!(
            "Using batched GPU prefill for Draft model ({} tokens)",
            prompt_tokens.len()
        );
        match gpu::gpu_batched_prefill_forward(
            device,
            &engine.draft_model,
            &engine.draft_cpu_weights,
            &mut engine.draft_kv,
            &mut engine.draft_prefill_scratch,
            &mut engine.draft_host_scratch,
            &prompt_tokens,
            0,
            &engine.draft_config,
            gpu::GpuLogitsMode::Skip,
        ) {
            Ok(_) => draft_prefilled = true,
            Err(err) => {
                eprintln!(
                    "Batched GPU prefill failed for draft ({}), falling back to decode-style path",
                    err
                );
            }
        }
    }

    if !draft_prefilled {
        for (pos, &token_id) in prompt_tokens.iter().enumerate() {
            gpu::gpu_embed_token_hybrid(
                device,
                token_id,
                &engine.draft_model,
                &engine.draft_cpu_weights,
                &mut engine.draft_scratch,
                &mut engine.draft_host_scratch,
                &engine.draft_config,
            )
            .map_err(|e| format!("gpu embed draft: {}", e))?;

            gpu::gpu_full_forward_hybrid(
                device,
                &engine.draft_model,
                &engine.draft_cpu_weights,
                &mut engine.draft_kv,
                &mut engine.draft_scratch,
                &mut engine.draft_host_scratch,
                pos,
                &engine.draft_config,
                gpu::GpuLogitsMode::Skip,
                token_id,
            )
            .map_err(|e| format!("gpu prefill draft: {}", e))?;
        }
    }

    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "Prefill: {:.1}ms ({:.1} tok/s)",
        prefill_ms,
        prompt_tokens.len() as f64 / prefill_ms * 1000.0
    );

    if args.prefill_only_validate {
        let logits = &engine.target_host_scratch.logits;
        let has_nan = logits.iter().any(|l| l.is_nan());
        let has_inf = logits.iter().any(|l| l.is_infinite());
        let has_finite = logits.iter().any(|l| l.is_finite());
        if has_nan || has_inf || !has_finite {
            eprintln!(
                "PREFILL_ONLY_VALIDATE: FAILED - target logits contain NaN/Inf or no finite values"
            );
            std::process::exit(1);
        }
        eprintln!("PREFILL_ONLY_VALIDATE: PASSED");
        std::process::exit(0);
    }

    let mut pos = prompt_tokens.len();
    let mut n_generated = 0usize;
    let mut n_drafted_total = 0usize;
    let mut n_accepted_total = 0usize;
    let t_gen = Instant::now();

    let mut next_token = target_prompt_next_token.unwrap_or_else(|| {
        cpu_sample_greedy(&engine.target_host_scratch.logits[..engine.target_config.vocab_size])
    });

    println!();

    loop {
        if tok.is_eog(next_token) || n_generated >= args.max_tokens || pos >= max_seq {
            break;
        }

        let text = tok.decode_token(next_token);
        if args.debug {
            eprintln!("[Generated] token_id={} text={:?}", next_token, text);
        }
        print!("{}", text);
        std::io::stdout().flush().ok();
        n_generated += 1;

        if tok.is_eog(next_token) || n_generated >= args.max_tokens || pos >= max_seq {
            break;
        }

        let spec_count = args
            .speculative_tokens
            .min(args.max_tokens - n_generated)
            .min(max_seq - pos - 1);

        if spec_count > 0 {
            let draft_tokens = engine
                .draft_tokens(device, pos, spec_count, next_token)
                .map_err(|e| format!("draft tokens: {}", e))?;

            let (accepted_tokens, num_accepted) = engine
                .verify_tokens(device, pos, &draft_tokens, next_token)
                .map_err(|e| format!("verify tokens: {}", e))?;

            for &token in &accepted_tokens[..num_accepted] {
                let text = tok.decode_token(token);
                if args.debug {
                    eprintln!("[Speculative Accepted] token_id={} text={:?}", token, text);
                } else {
                    print!("{}", text);
                }
                n_generated += 1;
            }
            std::io::stdout().flush().ok();

            n_drafted_total += spec_count;
            n_accepted_total += num_accepted;

            next_token = accepted_tokens[num_accepted];
            pos += num_accepted + 1;
        } else {
            gpu::gpu_embed_token_hybrid(
                device,
                next_token,
                &engine.target_model,
                &engine.target_cpu_weights,
                &mut engine.target_scratch,
                &mut engine.target_host_scratch,
                &engine.target_config,
            )
            .map_err(|e| format!("gpu embed target step: {}", e))?;

            let opt_token = gpu::gpu_full_forward_hybrid(
                device,
                &engine.target_model,
                &engine.target_cpu_weights,
                &mut engine.target_kv,
                &mut engine.target_scratch,
                &mut engine.target_host_scratch,
                pos,
                &engine.target_config,
                gpu::GpuLogitsMode::GreedyArgmax,
                next_token,
            )
            .map_err(|e| format!("gpu decode target step: {}", e))?;

            next_token = opt_token.unwrap_or_else(|| {
                cpu_sample_greedy(
                    &engine.target_host_scratch.logits[..engine.target_config.vocab_size],
                )
            });
            pos += 1;
        }
    }

    println!();

    if n_generated > 0 {
        let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        let acceptance_rate = if n_drafted_total > 0 {
            (n_accepted_total as f64 / n_drafted_total as f64) * 100.0
        } else {
            0.0
        };
        eprintln!("\n[Speculative Generation Statistics]");
        eprintln!("  Tokens generated: {} tokens", n_generated);
        eprintln!("  Total time: {:.1}ms", gen_ms);
        eprintln!(
            "  Generation speed: {:.1} tok/s",
            n_generated as f64 / gen_ms * 1000.0
        );
        eprintln!("  Draft tokens speculated: {}", n_drafted_total);
        eprintln!("  Draft tokens accepted: {}", n_accepted_total);
        eprintln!("  Draft acceptance rate: {:.2}%", acceptance_rate);
    } else {
        eprintln!("\n[EOS on first token]");
    }

    Ok(())
}
