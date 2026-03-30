//! rocmforge CLI - AMD-first LLM inference engine.
//!
//! Supports Qwen2.5 family models via GGUF format.
//! CPU execution path (GPU via HIP coming later).

use std::io::Write;
use std::time::Instant;

use rocmforge::config::{detect_chat_template, ModelConfig};
use rocmforge::cpu::SimdKernels;
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward},
    prefill::cpu_prefill_forward_parallel,
    sampler::{cpu_sample_greedy, cpu_sample_top_p},
    weights::CpuModelWeights,
    CpuError,
};
use rocmforge::hardware::{derive_batch_config, detect, BatchConfig, CpuCapabilities};
use rocmforge::loader::GgufFile;
use rocmforge::tokenizer::BpeTokenizer;

#[cfg(feature = "gpu")]
use rocmforge::gpu;

// ── CLI Args ─────────────────────────────────────────────────────────────────────

struct Args {
    model: String,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    no_template: bool,
    list_tensors: bool,
    debug: bool,
    gpu: bool,
}

fn usage() -> ! {
    eprintln!("rocmforge - AMD-first LLM inference engine");
    eprintln!();
    eprintln!("Usage: rocmforge --model <path> --prompt <text> [OPTIONS]");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --model <path>         Path to GGUF model file");
    eprintln!("  --prompt <text>        Input prompt");
    eprintln!();
    eprintln!("Optional:");
    eprintln!("  --max-tokens N         Maximum tokens to generate [default: 256]");
    eprintln!("  --temperature F        Sampling temperature [default: 1.0]");
    eprintln!("  --top-p F              Nucleus sampling threshold [default: 0.9]");
    eprintln!("  --no-template          Disable chat template");
    eprintln!("  --list-tensors         List tensors in model file and exit");
    eprintln!("  --debug                Show debug info (top logits, etc.)");
    eprintln!("  --gpu                  Use GPU backend (requires ROCm/HIP)");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  rocmforge --model qwen2.5-7b.gguf --prompt \"Hello, world!\"");
    eprintln!("  rocmforge -m model.gguf -p \"Write a poem\" --temp 0.7 --top-p 0.95");
    std::process::exit(1);
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1);
    let mut model = None;
    let mut prompt = None;
    let mut max_tokens = 256usize;
    let mut temperature = 1.0f32;
    let mut top_p = 0.9f32;
    let mut no_template = false;
    let mut list_tensors = false;
    let mut debug = false;
    let mut gpu = false;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "-m" | "--model" => model = Some(args.next().unwrap_or_else(|| usage())),
            "-p" | "--prompt" => prompt = Some(args.next().unwrap_or_else(|| usage())),
            "--max-tokens" => {
                max_tokens = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "-t" | "--temp" | "--temperature" => {
                temperature = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--top-p" => {
                top_p = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--no-template" => no_template = true,
            "--list-tensors" => list_tensors = true,
            "--debug" => debug = true,
            "--gpu" => gpu = true,
            "-h" | "--help" => usage(),
            other => {
                eprintln!("Unknown flag: {}", other);
                usage();
            }
        }
    }

    Args {
        model: model.unwrap_or_else(|| usage()),
        prompt: prompt.unwrap_or_default(),
        max_tokens,
        temperature,
        top_p,
        no_template,
        list_tensors,
        debug,
        gpu,
    }
}

// ── Tensor listing ───────────────────────────────────────────────────────────────

fn list_tensors(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = GgufFile::open(path)?;
    let mut names: Vec<&str> = file.tensor_names().collect();
    names.sort_unstable();

    println!("{:<45} {:<20} {}", "NAME", "TYPE", "SHAPE");
    println!("{}", "-".repeat(80));
    for n in &names {
        if let Ok(Some(t)) = file.tensor(n) {
            println!("{:<45} {::<20} {:?}", n, t.ggml_type, t.dims);
        }
    }
    println!("\nTotal: {} tensors", names.len());
    Ok(())
}

// ── Debug helpers ────────────────────────────────────────────────────────────────

/// Print top-k tokens with their probabilities.
fn print_top_k_tokens(logits: &[f32], tok: &BpeTokenizer, k: usize) {
    // Check for NaN/Inf in logits
    let nan_count = logits.iter().filter(|l| l.is_nan()).count();
    let inf_count = logits.iter().filter(|l| l.is_infinite()).count();
    if nan_count > 0 || inf_count > 0 {
        eprintln!(
            "ERROR: logits contain {} NaN and {} Inf values",
            nan_count, inf_count
        );
        eprintln!(
            "  Stats: min={:.4}, max={:.4}, mean={:.4}",
            logits.iter().cloned().fold(f32::INFINITY, f32::min),
            logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            logits.iter().sum::<f32>() / logits.len() as f32
        );
        return;
    }

    // Softmax to get probabilities
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Get top-k indices
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("Top-{} tokens:", k.min(indexed.len()));
    for (i, (id, prob)) in indexed.iter().take(k).enumerate() {
        let token = tok.decode_token(*id as u32);
        let token_display = if token.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
            token.clone()
        } else {
            format!("{:?}", token)
        };
        eprintln!("  {:2}. {:8} ({:.4}) id={}", i + 1, token_display, prob, id);
    }
}

// ── CPU Inference ────────────────────────────────────────────────────────────────

fn run_cpu_inference(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Detect CPU hardware capabilities
    eprint!("Detecting CPU capabilities... ");
    let caps: CpuCapabilities = detect().map_err(|e| format!("hardware detection: {}", e))?;
    eprintln!("done");
    eprintln!("  Physical cores: {}", caps.physical_cores);
    eprintln!("  Logical CPUs: {}", caps.logical_cpus);
    eprintln!("  SIMD features: {}", caps.simd.description());
    if caps.has_l3_cache() {
        eprintln!("  L3 cache: {:.1} MB", caps.l3_cache_mb());
    } else {
        eprintln!("  L3 cache: undetectable (using fallback)");
    }
    eprintln!("  Total memory: {:.1} GB", caps.total_memory_gb());

    // Initialize SIMD kernels
    let _simd_kernels = SimdKernels::new(caps.simd.kernel_preference());
    eprintln!("  Kernel preference: {}", _simd_kernels.description());

    // 2. Detect GPU (if gpu feature enabled)
    #[cfg(feature = "gpu")]
    let gpu_caps = {
        eprint!("Detecting GPU capabilities... ");
        let caps = gpu::detect();
        match &caps {
            Some(gpu) => {
                eprintln!("done");
                eprintln!("  GPU: {}", gpu.device_name);
                eprintln!(
                    "  VRAM: {:.1} GB / {:.1} GB",
                    gpu.free_vram_gb(),
                    gpu.total_vram_gb()
                );
            }
            None => {
                eprintln!("none detected");
            }
        }
        caps
    };

    // Load GGUF file

    // Load GGUF file
    let file = GgufFile::open(&args.model)?;
    eprintln!("[Args] model path: {}", args.model);
    let config = ModelConfig::from_gguf(&file)?;
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());
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

    // 3. Choose backend based on GPU availability and preference
    #[cfg(feature = "gpu")]
    let use_gpu = args.gpu && gpu_caps.is_some();

    #[cfg(not(feature = "gpu"))]
    let use_gpu = false;

    if use_gpu {
        eprintln!("Device: GPU");
        return Err("GPU inference not implemented yet".into());
    } else {
        eprintln!("Device: CPU");
    }

    // 2. Derive batch config from hardware + model
    let batch_config: BatchConfig = derive_batch_config(&caps, &config);
    eprintln!(
        "Batch config: max {} tokens/batch, use {} cores",
        batch_config.max_tokens_per_batch, batch_config.num_cores
    );

    // Load weights
    eprint!("Loading weights... ");
    let t_load = Instant::now();
    let weights =
        CpuModelWeights::load(&file, &config).map_err(|e| format!("weight load: {}", e))?;
    eprintln!("done in {:.1}s", t_load.elapsed().as_secs_f64());

    // Apply chat template
    let template = if args.no_template {
        rocmforge::config::ChatTemplate::None
    } else {
        detect_chat_template(&config.architecture, file.tokenizer_data().model.as_deref())
    };
    let prompted = template.apply(&args.prompt);
    eprintln!("Chat template: {}", template.name());

    // Tokenize prompt
    let prompt_tokens = tok.encode(&prompted, false);
    if prompt_tokens.is_empty() {
        return Err("Prompt tokenized to zero tokens".into());
    }
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    // Allocate KV cache and scratch buffers
    let max_seq = (prompt_tokens.len() + args.max_tokens).min(config.max_seq_len);
    let mut kv = CpuKvCache::new(&config, max_seq);
    let mut scratch = CpuForwardScratch::new(&config);
    let use_greedy = args.top_p >= 1.0;

    // ── Prefill ───────────────────────────────────────────────────────────────────
    let t_prefill = Instant::now();
    let n_prompt = prompt_tokens.len();

    // Debug: show first prompt token embedding
    if args.debug && n_prompt > 0 {
        let first_tok = prompt_tokens[0];
        let mut test_hidden = vec![0.0f32; config.hidden_size];
        cpu_embed_token(first_tok, &weights, &mut test_hidden, &config);
        let mean: f32 = test_hidden.iter().copied().sum::<f32>() / test_hidden.len() as f32;
        let std: f32 = ((test_hidden.iter().map(|x| x * x).sum::<f32>()
            / test_hidden.len() as f32)
            - mean * mean)
            .sqrt();
        eprintln!(
            "[Prefill] first token {} embedding: mean={:.4} std={:.4}",
            first_tok, mean, std
        );
        eprintln!("  hidden[0..5]: {:?}", &test_hidden[0..5]);
    }

    cpu_prefill_forward_parallel(
        &prompt_tokens,
        &weights,
        &mut kv,
        &mut scratch,
        0,
        &config,
        &batch_config,
    )
    .map_err(|e: CpuError| format!("prefill: {}", e))?;

    // Debug: show top tokens after prefill
    if args.debug {
        print_top_k_tokens(&scratch.logits, &tok, 10);
    }

    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "Prefill: {:.1}ms ({:.1} tok/s)",
        prefill_ms,
        n_prompt as f64 / prefill_ms * 1000.0
    );

    // ── Decode loop ───────────────────────────────────────────────────────────────
    let mut pos = n_prompt;
    let mut n_generated = 0usize;
    let t_gen = Instant::now();
    let mut seed = 0xdeadbeef_u64;
    let mut generated_ids = Vec::with_capacity(args.max_tokens);

    // Sample first token from prefill output
    let mut next_token = if use_greedy {
        cpu_sample_greedy(&scratch.logits)
    } else {
        seed = seed.wrapping_add(1);
        cpu_sample_top_p(&scratch.logits, args.temperature, args.top_p, seed)
    };

    // Allocate hidden state buffer (reused for each token)
    let mut hidden = vec![0.0f32; config.hidden_size];

    print!("\n");

    loop {
        // Check termination conditions
        if tok.is_eog(next_token) {
            break;
        }
        if n_generated >= args.max_tokens {
            break;
        }

        // Decode and print token
        let text = tok.decode_token(next_token);
        if args.debug {
            eprintln!("[Generated] token_id={} text={:?}", next_token, text);
        }
        print!("{}", text);
        std::io::stdout().flush().ok();
        n_generated += 1;
        generated_ids.push(next_token);

        // Embed token
        cpu_embed_token(next_token, &weights, &mut hidden, &config);

        // Debug: show hidden state statistics
        if args.debug && n_generated <= 3 {
            let mean: f32 = hidden.iter().copied().sum::<f32>() / hidden.len() as f32;
            let std: f32 = ((hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len() as f32)
                - mean * mean)
                .sqrt();
            let min: f32 = hidden.iter().cloned().fold(f32::INFINITY, f32::min);
            let max: f32 = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "\n[Token {} embed] id={} mean={:.4} std={:.4} range=[{:.4}, {:.4}]",
                n_generated, next_token, mean, std, min, max
            );
            // Show first 5 hidden values
            eprintln!("  hidden[0..5]: {:?}", &hidden[0..5]);
        }

        // Forward pass
        cpu_full_forward(&mut hidden, &weights, &mut kv, &mut scratch, pos, &config)
            .map_err(|e: CpuError| format!("decode: {}", e))?;
        pos += 1;

        // Debug: show logits statistics
        if args.debug && n_generated <= 3 {
            let logits = &scratch.logits;
            let mean: f32 = logits.iter().copied().sum::<f32>() / logits.len() as f32;
            let std: f32 = ((logits.iter().map(|x| x * x).sum::<f32>() / logits.len() as f32)
                - mean * mean)
                .sqrt();
            let min: f32 = logits.iter().cloned().fold(f32::INFINITY, f32::min);
            let max: f32 = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[Token {} logits] mean={:.4} std={:.4} range=[{:.4}, {:.4}]",
                n_generated, mean, std, min, max
            );
        }

        // Debug: show top tokens
        if args.debug && n_generated <= 3 {
            eprintln!("\n[Token {} logits]", n_generated);
            print_top_k_tokens(&scratch.logits, &tok, 5);
        }

        // Sample next token
        next_token = if use_greedy {
            cpu_sample_greedy(&scratch.logits)
        } else {
            seed = seed.wrapping_add(1);
            cpu_sample_top_p(&scratch.logits, args.temperature, args.top_p, seed)
        };
    }

    println!();

    // ── Stats ─────────────────────────────────────────────────────────────────────
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

// ── GPU Inference ────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn run_gpu_inference(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    eprint!("Detecting GPU capabilities... ");
    let gpu_caps = gpu::detect().ok_or("GPU requested but no AMD GPU detected")?;
    eprintln!("done");
    eprintln!("  GPU: {}", gpu_caps.device_name);
    eprintln!(
        "  VRAM: {:.1} GB / {:.1} GB",
        gpu_caps.free_vram_gb(),
        gpu_caps.total_vram_gb()
    );

    let file = GgufFile::open(&args.model)?;
    eprintln!("[Args] model path: {}", args.model);
    let config = ModelConfig::from_gguf(&file)?;
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());
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

    if let Ok(meta) = std::fs::metadata(&args.model) {
        let model_size = meta.len() as usize;
        if !gpu_caps.can_fit_model(model_size) {
            eprintln!(
                "Warning: model file is {:.1} GB and may exceed safe free VRAM headroom",
                model_size as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
    }

    eprint!("Initializing GPU device... ");
    let device =
        gpu::GpuDevice::init(gpu_caps.device_id).map_err(|e| format!("gpu init: {}", e))?;
    eprintln!("done");

    eprint!("Loading CPU weights... ");
    let t_cpu_load = Instant::now();
    let cpu_weights =
        CpuModelWeights::load(&file, &config).map_err(|e| format!("cpu weight load: {}", e))?;
    eprintln!("done in {:.1}s", t_cpu_load.elapsed().as_secs_f64());

    eprint!("Loading GPU weights... ");
    let t_gpu_load = Instant::now();
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config)
        .map_err(|e| format!("gpu weight load: {}", e))?;
    eprintln!("done in {:.1}s", t_gpu_load.elapsed().as_secs_f64());

    let template = if args.no_template {
        rocmforge::config::ChatTemplate::None
    } else {
        detect_chat_template(&config.architecture, file.tokenizer_data().model.as_deref())
    };
    let prompted = template.apply(&args.prompt);
    eprintln!("Chat template: {}", template.name());

    let prompt_tokens = tok.encode(&prompted, false);
    if prompt_tokens.is_empty() {
        return Err("Prompt tokenized to zero tokens".into());
    }
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    let max_seq = (prompt_tokens.len() + args.max_tokens).min(config.max_seq_len);
    let mut kv = gpu::GpuKvCache::new(&config, max_seq).map_err(|e| format!("gpu kv: {}", e))?;
    let mut gpu_scratch =
        gpu::GpuForwardScratch::new(&config).map_err(|e| format!("gpu scratch: {}", e))?;
    let mut host_scratch = CpuForwardScratch::new(&config);
    let use_greedy = args.top_p >= 1.0;
    let use_gpu_greedy_fastpath = use_greedy && !args.debug;

    let t_prefill = Instant::now();
    let final_prompt_logits_mode = if use_gpu_greedy_fastpath {
        gpu::GpuLogitsMode::GreedyArgmax
    } else {
        gpu::GpuLogitsMode::DownloadToHost
    };
    let mut prompt_next_token = match gpu::GpuPrefillScratch::new(&config, prompt_tokens.len()) {
        Ok(mut prefill_scratch) => match gpu::gpu_prefill_forward_hybrid(
            &device,
            &gpu_weights,
            &cpu_weights,
            &mut kv,
            &mut prefill_scratch,
            &mut gpu_scratch,
            &mut host_scratch,
            &prompt_tokens,
            0,
            &config,
            final_prompt_logits_mode,
        ) {
            Ok(token) => token,
            Err(
                err @ (gpu::GpuError::UnsupportedWeightType { .. }
                | gpu::GpuError::InvalidWeightLayout { .. }),
            ) => {
                eprintln!(
                    "Batched GPU prefill unavailable ({}), falling back to decode-style prompt path",
                    err
                );
                let mut prompt_next_token = None;
                for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                    gpu::gpu_embed_token_hybrid(&device, 
                        token_id,
                        &gpu_weights,
                        &cpu_weights,
                        &mut gpu_scratch,
                        &mut host_scratch,
                        &config,
                    )
                    .map_err(|e| format!("gpu embed: {}", e))?;
                    let logits_mode = if pos + 1 == prompt_tokens.len() {
                        final_prompt_logits_mode
                    } else {
                        gpu::GpuLogitsMode::Skip
                    };
                    prompt_next_token = gpu::gpu_full_forward_hybrid(
                        &device,
                        &gpu_weights,
                        &cpu_weights,
                        &mut kv,
                        &mut gpu_scratch,
                        &mut host_scratch,
                        pos,
                        &config,
                        logits_mode,
                    )
                    .map_err(|e| format!("gpu prefill/decode: {}", e))?;
                }
                prompt_next_token
            }
            Err(err) => return Err(format!("gpu batched prefill: {}", err).into()),
        },
        Err(err) => {
            eprintln!(
                "Batched GPU prefill scratch unavailable ({}), falling back to decode-style prompt path",
                err
            );
            let mut prompt_next_token = None;
            for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                gpu::gpu_embed_token_hybrid(&device, 
                    token_id,
                    &gpu_weights,
                    &cpu_weights,
                    &mut gpu_scratch,
                    &mut host_scratch,
                    &config,
                )
                .map_err(|e| format!("gpu embed: {}", e))?;
                let logits_mode = if pos + 1 == prompt_tokens.len() {
                    final_prompt_logits_mode
                } else {
                    gpu::GpuLogitsMode::Skip
                };
                prompt_next_token = gpu::gpu_full_forward_hybrid(
                    &device,
                    &gpu_weights,
                    &cpu_weights,
                    &mut kv,
                    &mut gpu_scratch,
                    &mut host_scratch,
                    pos,
                    &config,
                    logits_mode,
                )
                .map_err(|e| format!("gpu prefill/decode: {}", e))?;
            }
            prompt_next_token
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

    print!("\n");

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

        gpu::gpu_embed_token_hybrid(&device, 
            next_token,
            &gpu_weights,
            &cpu_weights,
            &mut gpu_scratch,
            &mut host_scratch,
            &config,
        )
        .map_err(|e| format!("gpu embed: {}", e))?;
        let logits_mode = if use_gpu_greedy_fastpath {
            gpu::GpuLogitsMode::GreedyArgmax
        } else {
            gpu::GpuLogitsMode::DownloadToHost
        };
        gpu::gpu_full_forward_hybrid(
            &device,
            &gpu_weights,
            &cpu_weights,
            &mut kv,
            &mut gpu_scratch,
            &mut host_scratch,
            pos,
            &config,
            logits_mode,
        )
        .map_err(|e| format!("gpu decode: {}", e))?;
        pos += 1;

        if args.debug && n_generated <= 3 {
            eprintln!("\n[Token {} logits]", n_generated);
            print_top_k_tokens(&host_scratch.logits, &tok, 5);
        }

        // SYNC POINT: Wait for GPU to finish forward + argmax download
        device.synchronize().map_err(|e| format!("gpu sync: {}", e))?;

        next_token = if use_greedy {
            if use_gpu_greedy_fastpath {
                let token = gpu_scratch.argmax_result_index.as_slice::<i32>()[0];
                if token < 0 || (token as usize) >= config.vocab_size {
                    return Err(format!("gpu argmax returned out-of-range index {}", token));
                }
                token as u32
            } else {
                cpu_sample_greedy(&host_scratch.logits)
            }
        } else {
            seed = seed.wrapping_add(1);
            cpu_sample_top_p(&host_scratch.logits, args.temperature, args.top_p, seed)
        };
    }

    println!();

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

// ── Entry point ───────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    // Handle --list-tensors
    if args.list_tensors {
        if let Err(e) = list_tensors(&args.model) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    #[cfg(feature = "gpu")]
    if args.gpu {
        if let Err(e) = run_gpu_inference(&args) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    #[cfg(not(feature = "gpu"))]
    if args.gpu {
        eprintln!("Error: GPU backend requires building with --features gpu");
        std::process::exit(1);
    }

    if let Err(e) = run_cpu_inference(&args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
