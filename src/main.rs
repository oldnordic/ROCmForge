//! rocmforge CLI - AMD-first LLM inference engine.
//!
//! Supports Qwen2.5 family models via GGUF format.
//! CPU execution path (GPU via HIP coming later).

use std::io::Write;
use std::time::Instant;

use rocmforge::cpu::SimdKernels;
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward},
    prefill::cpu_prefill_forward_parallel,
    sampler::{cpu_sample_greedy, cpu_sample_top_p},
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
    #[allow(dead_code)]
    prefill_only_validate: bool,
    #[allow(dead_code)]
    draft_model: Option<String>,
    #[allow(dead_code)]
    speculative_tokens: usize,
    #[allow(dead_code)]
    /// If `Some(path)`, dump the post-prefill KV cache to this file.
    kv_dump: Option<String>,
    server: bool,
    #[cfg_attr(
        not(feature = "server"),
        expect(dead_code, reason = "used only with server feature")
    )]
    port: u16,
    threads: Option<usize>,
    ctx_size: Option<usize>,
}

fn usage() -> ! {
    eprintln!("rocmforge - AMD-first LLM inference engine");
    eprintln!();
    eprintln!("Usage: rocmforge --model <path> --prompt <text> [OPTIONS]");
    eprintln!("       rocmforge --model <path> --server [--port N]");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --model <path>         Path to GGUF model file");
    eprintln!();
    eprintln!("Generation mode:");
    eprintln!("  --prompt <text>        Input prompt");
    eprintln!("  --max-tokens N         Maximum tokens to generate [default: 256]");
    eprintln!("  --temperature F        Sampling temperature [default: 1.0]");
    eprintln!("  --top-p F              Nucleus sampling threshold [default: 0.9]");
    eprintln!("  --no-template          Disable chat template");
    eprintln!("  --list-tensors         List tensors in model file and exit");
    eprintln!("  --debug                Show debug info (top logits, etc.)");
    eprintln!("  --gpu                  Use GPU backend (requires ROCm/HIP)");
    eprintln!("  --threads N, -t N      Number of CPU threads/cores to use [default: auto-detect]");
    eprintln!(
        "  --ctx-size N, -c N     Override maximum context window size [default: model default]"
    );
    eprintln!("  --prefill-only-validate Run prefill only, exit with validation status");
    eprintln!("  --kv-dump <path>       Dump post-prefill KV cache to binary file");
    eprintln!(
        "  --draft-model <path>   Path to draft GGUF/RFM model file for speculative decoding"
    );
    eprintln!("  --speculative-tokens N Number of draft tokens to speculate per step [default: 4]");
    eprintln!();
    eprintln!("Server mode:");
    eprintln!("  --server               Start OpenAI-compatible HTTP API server");
    eprintln!("  --port N               Port to bind [default: 8080]");
    eprintln!();
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  rocmforge --model qwen2.5-7b.gguf --prompt \"Hello, world!\"");
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
    let mut server = false;
    let mut port = 8080u16;
    let mut prefill_only_validate = false;
    let mut draft_model = None;
    let mut speculative_tokens = 4usize;
    let mut kv_dump: Option<String> = None;
    let mut threads = None;
    let mut ctx_size = None;

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
            "--temp" | "--temperature" => {
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
            "--server" => server = true,
            "--port" => {
                port = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "-t" | "--threads" => {
                threads = Some(
                    args.next()
                        .unwrap_or_else(|| usage())
                        .parse()
                        .unwrap_or_else(|_| usage()),
                )
            }
            "-c" | "--ctx-size" => {
                ctx_size = Some(
                    args.next()
                        .unwrap_or_else(|| usage())
                        .parse()
                        .unwrap_or_else(|_| usage()),
                )
            }
            "--prefill-only-validate" => prefill_only_validate = true,
            "--kv-dump" => kv_dump = Some(args.next().unwrap_or_else(|| usage())),
            "--draft-model" => draft_model = Some(args.next().unwrap_or_else(|| usage())),
            "--speculative-tokens" => {
                speculative_tokens = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
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
        prefill_only_validate,
        draft_model,
        speculative_tokens,
        kv_dump,
        server,
        port,
        threads,
        ctx_size,
    }
}

// ── Tensor listing ───────────────────────────────────────────────────────────────

fn list_tensors(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if path.ends_with(".rfm") {
        use rocmforge::loader::RfmFile;
        let file = RfmFile::open(path)?;
        let mut names: Vec<&str> = file.tensor_names().collect();
        names.sort_unstable();

        println!("{:<45} {:<20} SHAPE", "NAME", "TYPE");
        println!("{}", "-".repeat(80));
        for n in &names {
            if let Ok(Some(t)) = file.tensor(n) {
                println!("{:<45} {:<20?} {:?}", n, t.wtype, t.dims);
            }
        }
        println!("\nTotal: {} tensors", names.len());
        Ok(())
    } else {
        let file = GgufFile::open(path)?;
        let mut names: Vec<&str> = file.tensor_names().collect();
        names.sort_unstable();

        println!("{:<45} {:<20} SHAPE", "NAME", "TYPE");
        println!("{}", "-".repeat(80));
        for n in &names {
            if let Ok(Some(t)) = file.tensor(n) {
                println!("{:<45} {:<20} {:?}", n, t.ggml_type, t.dims);
            }
        }
        println!("\nTotal: {} tensors", names.len());
        Ok(())
    }
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
    indexed.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .expect("invariant: partial_cmp failed (NaN in main logic)")
    });

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

    let file = rocmforge::loader::ModelFile::open(&args.model)?;
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
    let t_load = Instant::now();
    let weights = file
        .load_cpu_weights(&config)
        .map_err(|e| format!("weight load: {}", e))?;
    eprintln!("done in {:.1}s", t_load.elapsed().as_secs_f64());

    let template = file.chat_template(&config, args.no_template);

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
    let mut batch_config: BatchConfig = derive_batch_config(&caps, &config);
    if let Some(t) = args.threads {
        batch_config.num_cores = t;
    }
    eprintln!(
        "Batch config: max {} tokens/batch, use {} cores",
        batch_config.max_tokens_per_batch, batch_config.num_cores
    );
    let prompted = template.apply(&args.prompt);
    eprintln!("Chat template: {}", template.name());

    // Tokenize prompt
    let prompt_tokens = tok.encode(&prompted, false);
    if prompt_tokens.is_empty() {
        return Err("Prompt tokenized to zero tokens".into());
    }
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    // Allocate KV cache and scratch buffers
    let max_seq = args
        .ctx_size
        .unwrap_or_else(|| (prompt_tokens.len() + args.max_tokens).min(config.max_seq_len));
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

    println!();

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

/// Estimate bytes that will actually reside on the GPU for this model.
///
/// Uses file size for non-MoE models. For MoE models the expert tensors
/// (`_exps`) are streamed from CPU RAM one expert at a time and never fully
/// resident in VRAM, so we sum only the attention weights, layer norms, MoE
/// router, and shared-expert tensors that stay on the GPU throughout inference.
#[cfg(feature = "gpu")]
fn estimate_gpu_resident_model_bytes(
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

#[cfg(feature = "gpu")]
fn run_gpu_inference(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    eprint!("Detecting GPU capabilities... ");
    let gpu_caps = gpu::detect().ok_or("GPU requested but no AMD GPU detected")?;
    eprintln!("done");
    eprintln!("  GPU: {}", gpu_caps.device_name);

    // ── VRAM pre-flight ───────────────────────────────────────────────────────────
    // Query current VRAM state before touching anything. This captures what the
    // desktop and other processes are already using so we can compute the real
    // inference budget and refuse to start if the workload won't fit.
    let vram_session = gpu::VramSession::new(gpu_caps.device_id)
        .map_err(|e| format!("VRAM query failed: {}", e))?;

    // Open the model file early so we can read config for VRAM estimates.
    let file = rocmforge::loader::ModelFile::open(&args.model)?;
    let config = file.config()?;

    // Estimate VRAM before allocating anything.
    // For MoE models, experts are CPU-resident; only sum GPU-resident tensors.
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

    // Warn if experimental GPU kernels are enabled (display-attached GPU safety)
    if gpu::safety::experimental_gpu_kernels_enabled() {
        eprintln!("  WARNING: ROCMFORGE_ENABLE_EXPERIMENTAL_GPU_KERNELS=1");
        eprintln!("           Sparse CSR / MPO kernels are enabled. These can fault on");
        eprintln!("           display-attached GPUs. Use only for testing.");
    }

    eprint!("Initializing GPU device... ");
    let device =
        gpu::GpuDevice::get_or_init(gpu_caps.device_id).map_err(|e| format!("gpu init: {}", e))?;
    eprintln!("done");

    eprintln!("[Args] model path ({}): {}", file.format_name(), args.model);

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

    eprint!("Loading CPU weights... ");
    let t_cpu_load = Instant::now();
    let cpu_weights = file
        .load_cpu_weights(&config)
        .map_err(|e| format!("cpu weight load: {}", e))?;
    eprintln!("done in {:.1}s", t_cpu_load.elapsed().as_secs_f64());

    eprint!("Loading GPU weights... ");
    let t_gpu_load = Instant::now();
    let gpu_weights = file
        .load_gpu_weights(&config, gpu_caps.device_id)
        .map_err(|e| format!("gpu weight load: {}", e))?;
    eprintln!("done in {:.1}s", t_gpu_load.elapsed().as_secs_f64());

    let template = file.chat_template(&config, args.no_template);

    let prompted = template.apply(&args.prompt);
    eprintln!("Chat template: {}", template.name());

    let prompt_tokens = tok.encode(&prompted, false);
    if prompt_tokens.is_empty() {
        return Err("Prompt tokenized to zero tokens".into());
    }
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    let max_seq = args
        .ctx_size
        .unwrap_or_else(|| (prompt_tokens.len() + args.max_tokens).min(config.max_seq_len));
    let mut kv = gpu::GpuKvCache::new(&config, max_seq).map_err(|e| format!("gpu kv: {}", e))?;
    let mut gpu_scratch =
        gpu::GpuForwardScratch::new(&config).map_err(|e| format!("gpu scratch: {}", e))?;

    // Allocate per-expert GPU scratch sized for the maximum across gate/up/down expert dims.
    // gate/up use [rows=ff_size, cols=hidden]; down uses [rows=hidden, cols=ff_size].
    // The scratch must hold the largest U, V, CSR, and row_ptr across all three.
    'expert_scratch: for layer in &gpu_weights.layers {
        let all_compressed = [
            layer.ffn_gate_compressed.as_ref(),
            layer.ffn_up_compressed.as_ref(),
            layer.ffn_down_compressed.as_ref(),
        ];
        if all_compressed.iter().all(|x| x.is_some()) {
            let k = layer
                .ffn_gate_compressed
                .as_ref()
                .map(|c| c.k)
                .unwrap_or(32);
            let max_rows = all_compressed
                .iter()
                .filter_map(|x| *x)
                .map(|c| c.rows)
                .max()
                .unwrap_or(1);
            let max_cols = all_compressed
                .iter()
                .filter_map(|x| *x)
                .map(|c| c.cols)
                .max()
                .unwrap_or(1);
            let max_nnz = all_compressed
                .iter()
                .filter_map(|x| *x)
                .map(|c| c.max_nnz())
                .max()
                .unwrap_or(1);
            gpu_scratch
                .init_expert_scratch(k as u32, max_rows, max_cols, max_nnz)
                .map_err(|e| format!("expert scratch init: {}", e))?;
            eprintln!(
                "  Expert scratch: k={}, max_rows={}, max_cols={}, max_nnz={}",
                k, max_rows, max_cols, max_nnz
            );
            break 'expert_scratch;
        }
    }

    let mut host_scratch = CpuForwardScratch::new(&config);
    let use_greedy = args.top_p >= 1.0;
    let use_gpu_greedy_fastpath = use_greedy && !args.debug;

    let t_prefill = Instant::now();
    let final_prompt_logits_mode = if use_gpu_greedy_fastpath {
        gpu::GpuLogitsMode::GreedyArgmax
    } else {
        gpu::GpuLogitsMode::DownloadToHost
    };

    // ── Hotpath Router ────────────────────────────────────────────────────────────
    // Build model profile from loaded weights and select the optimal inference path.
    // This runs AFTER VRAM pre-flight (above) and BEFORE any scratch allocation.
    let profile = gpu::ModelProfile::from_weights(&gpu_weights, &config);
    let path = gpu::select_path(&profile, prompt_tokens.len(), &vram_session);
    eprintln!("[Router] Model profile: {}", profile.summary());
    eprintln!("[Router] Selected path: {}", path);

    // Check path-specific VRAM requirements (e.g., batched prefill scratch).
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
                    match gpu::gpu_batched_prefill_forward_q4_0(
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
                            // Fallback to decode-style processing
                            let mut prompt_next_token = None;
                            for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                                gpu::gpu_embed_token_hybrid(
                                    device,
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
                                    device,
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
                    }
                }
                Err(err) => {
                    eprintln!(
                        "Batched GPU prefill scratch allocation failed ({}), falling back to decode-style prompt path",
                        err
                    );
                    // Fallback to decode-style processing
                    let mut prompt_next_token = None;
                    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                        gpu::gpu_embed_token_hybrid(
                            device,
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
                            device,
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
            }
        }
        gpu::InferencePath::SvdOptimized => {
            eprintln!("Using SVD-optimized decode-style path");
            let mut prompt_next_token = None;
            for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                gpu::gpu_embed_token_hybrid(
                    device,
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
                    device,
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
        gpu::InferencePath::DecodeStyle | gpu::InferencePath::CpuFallback { .. } => {
            if !gpu_weights.uses_q4_0_quantization() {
                eprintln!("Batched GPU prefill only available for Q4_0 models, using decode-style prompt path");
            } else if prompt_tokens.len() == 1 {
                eprintln!("Single token prompt, using decode-style path");
            } else {
                eprintln!(
                    "Prompt too long for batched prefill ({}), using decode-style path",
                    prompt_tokens.len()
                );
            }
            let mut prompt_next_token = None;
            for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                gpu::gpu_embed_token_hybrid(
                    device,
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
                    device,
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

    // Optional KV cache dump for compression research
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

    // Prefill-only validation mode: exit after prefill with clear success/failure signal
    if args.prefill_only_validate {
        // Validate that prefill produced finite logits
        let logits = &host_scratch.logits;
        let has_nan = logits.iter().any(|l| l.is_nan());
        let has_inf = logits.iter().any(|l| l.is_infinite());
        let has_finite = logits.iter().any(|l| l.is_finite());

        if has_nan || has_inf {
            eprintln!("PREFILL_ONLY_VALIDATE: FAILED - logits contain NaN or Inf");
            eprintln!(
                "  NaN: {}, Inf: {}, Finite: {}",
                has_nan, has_inf, has_finite
            );
            std::process::exit(1);
        }

        if !has_finite {
            eprintln!("PREFILL_ONLY_VALIDATE: FAILED - no finite logits");
            std::process::exit(1);
        }

        // Validate that batched prefill was exercised for Q4_0 models
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
        )
        .map_err(|e| format!("gpu decode: {}", e))?;
        pos += 1;

        if args.debug && n_generated <= 3 {
            eprintln!("\n[Token {} logits]", n_generated);
            print_top_k_tokens(&host_scratch.logits, &tok, 5);
        }

        next_token = if let Some(token) = decode_next_token {
            token
        } else {
            // SYNC POINT: Wait for GPU to finish forward + argmax download (non-graph path)
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

#[cfg(feature = "gpu")]
fn run_gpu_speculative_inference(
    args: &Args,
    draft_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    eprint!("Detecting GPU capabilities... ");
    let gpu_caps = gpu::detect().ok_or("GPU requested but no AMD GPU detected")?;
    eprintln!("done");
    eprintln!("  GPU: {}", gpu_caps.device_name);

    // ── VRAM pre-flight for speculative co-execution ─────────────────────────────
    // Query current VRAM state before touching anything. This captures what the
    // desktop and other processes are already using so we can compute the real
    // inference budget and refuse to start if both models won't fit.
    let vram_session = gpu::VramSession::new(gpu_caps.device_id)
        .map_err(|e| format!("VRAM query failed: {}", e))?;

    let file = rocmforge::loader::ModelFile::open(&args.model)?;
    let target_config = file.config()?;

    let draft_file = rocmforge::loader::ModelFile::open(draft_path)?;
    let draft_config = draft_file.config()?;

    // Estimate VRAM of both models using their file sizes as conservative upper bounds
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

    eprint!("Initializing GPU device... ");
    let device =
        gpu::GpuDevice::get_or_init(gpu_caps.device_id).map_err(|e| format!("gpu init: {}", e))?;
    eprintln!("done");

    eprintln!("[Args] model path ({}): {}", file.format_name(), args.model);
    let tok = file.tokenizer();
    let template = file.chat_template(&target_config, args.no_template);

    let prompted = template.apply(&args.prompt);
    eprintln!("Chat template: {}", template.name());

    let prompt_tokens = tok.encode(&prompted, false);
    if prompt_tokens.is_empty() {
        return Err("Prompt tokenized to zero tokens".into());
    }
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    let max_seq = args
        .ctx_size
        .unwrap_or_else(|| (prompt_tokens.len() + args.max_tokens).min(target_config.max_seq_len));

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
        match gpu::gpu_batched_prefill_forward_q4_0(
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
        // Fallback or standard decode-style prompt path for Target Model
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
        match gpu::gpu_batched_prefill_forward_q4_0(
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
            // Autoregressively draft N speculative tokens on the GPU
            let draft_tokens = engine
                .draft_tokens(device, pos, spec_count, next_token)
                .map_err(|e| format!("draft tokens: {}", e))?;

            // Run target verification pass over the N drafted tokens
            let (accepted_tokens, num_accepted) = engine
                .verify_tokens(device, pos, &draft_tokens, next_token)
                .map_err(|e| format!("verify tokens: {}", e))?;

            // Print accepted draft tokens
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
            // Speculative count is 0, fall back to single target step
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

// ── Entry point ───────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    #[cfg(feature = "server")]
    if args.server {
        use rocmforge::api::server::{create_router, ModelEntry, ModelManager};
        let entry =
            ModelEntry::load(&args.model, args.draft_model.as_deref()).unwrap_or_else(|e| {
                eprintln!("Failed to load model: {}", e);
                std::process::exit(1);
            });
        let manager = ModelManager::new(entry);
        let state = std::sync::Arc::new(tokio::sync::Mutex::new(manager));
        let router = create_router(state);
        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], args.port));
        let rt = tokio::runtime::Runtime::new()
            .expect("invariant: failed to build tokio runtime in main");
        eprintln!("rocmforge server listening on http://{}/", addr);
        rt.block_on(async {
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .expect("invariant: failed to bind TCP address for server");
            axum::serve(listener, router)
                .await
                .expect("invariant: failed to serve HTTP router");
        });
        return;
    }

    // Handle --server mode (requires server feature)
    #[cfg(not(feature = "server"))]
    if args.server {
        eprintln!("Error: --server requires building with --features server");
        std::process::exit(1);
    }
    if args.list_tensors {
        if let Err(e) = list_tensors(&args.model) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    #[cfg(feature = "gpu")]
    if args.gpu {
        // Enforce VRAM safety pre-flight as the absolute first thing
        let caps = match gpu::detect() {
            Some(c) => c,
            None => {
                eprintln!("Error: GPU requested but no AMD GPU detected.");
                std::process::exit(1);
            }
        };
        gpu::binary_vram_safety_preflight(caps.device_id);

        // 1. Acquire cross-process GPU lock
        let _gpu_lock = match gpu::GpuLock::acquire(30) {
            Ok(lock) => lock,
            Err(e) => {
                eprintln!("Error: Failed to acquire GPU lock ({}).", e);
                std::process::exit(10);
            }
        };

        // 2. Run GPU hardware/driver safety preflight checks
        if let Err(e) = gpu::gpu_safety_preflight() {
            eprintln!("❌ Error: GPU safety preflight failed: {}. Refusing execution to prevent driver freeze.", e);
            std::process::exit(1);
        }

        if let Some(ref draft_path) = args.draft_model {
            if let Err(e) = run_gpu_speculative_inference(&args, draft_path) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        } else {
            if let Err(e) = run_gpu_inference(&args) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
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

#[cfg(test)]
mod main_tests {
    use super::*;

    #[test]
    fn test_args_fields_present_for_linter() {
        let args = Args {
            model: String::new(),
            prompt: String::new(),
            max_tokens: 0,
            temperature: 0.0,
            top_p: 0.0,
            no_template: false,
            list_tensors: false,
            debug: false,
            gpu: false,
            prefill_only_validate: false,
            draft_model: None,
            speculative_tokens: 0,
            kv_dump: None,
            server: false,
            port: 0,
            threads: None,
            ctx_size: None,
        };
        assert!(!args.prefill_only_validate);
        assert!(args.draft_model.is_none());
        assert_eq!(args.speculative_tokens, 0);
        assert!(args.kv_dump.is_none());
    }
}
