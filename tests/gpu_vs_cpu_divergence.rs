#![cfg(feature = "gpu")]

//! CPU-vs-GPU divergence harness.
//!
//! Runs ONE token through Qwen2.5-0.5B Q4_0 on both CPU and GPU, captures
//! intermediate tensors at each stage of layer 0, and reports the first stage
//! where max_abs_err > threshold. This isolates WHICH step in the generic
//! decode arm introduces the numerical divergence.
//!
//! Run:
//!   ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 cargo test --release --features gpu \
//!     --test gpu_vs_cpu_divergence -- --ignored --nocapture --test-threads=1

mod common;

use rocmforge::config::ModelConfig;
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward, cpu_layer_forward},
    ops::{argmax, dispatch_gemv, rms_norm},
    quant::{load_f16_scale, Q4_BLOCK_BYTES, Q4_BLOCK_ELEMS},
    weights::CpuModelWeights,
};
use rocmforge::gpu::{self, GpuBuffer, GpuDevice, GpuForwardScratch, GpuKvCache};
use rocmforge::loader::{GgmlType, GgufFile};
use rocmforge::tokenizer::BpeTokenizer;
use serial_test::serial;

const MODEL_PATH: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";

/// Divergence threshold on raw (un-normalized) intermediate values.
/// Q4_0 dequant introduces ~0.01–0.03 error vs f32. Anything above this
/// means the stage is broken, not just noisy.
const THRESHOLD: f32 = 0.1;

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut max_err = 0.0f32;
    let mut idx = 0;
    for i in 0..n {
        let err = (a[i] - b[i]).abs();
        if err > max_err {
            max_err = err;
            idx = i;
        }
    }
    let _ = idx; // tracked for reporting
    max_err
}

fn has_nan_or_inf(v: &[f32]) -> bool {
    v.iter().any(|x| x.is_nan() || x.is_infinite())
}

fn report(stage: &str, cpu: &[f32], gpu: &[f32]) -> bool {
    let n = cpu.len().min(gpu.len());
    if cpu.len() != gpu.len() {
        eprintln!(
            "  [{}] SIZE MISMATCH: cpu={} gpu={}",
            stage,
            cpu.len(),
            gpu.len()
        );
        return true;
    }
    if has_nan_or_inf(cpu) {
        eprintln!("  [{}] CPU has NaN/Inf!", stage);
        return true;
    }
    if has_nan_or_inf(gpu) {
        eprintln!("  [{}] GPU has NaN/Inf!", stage);
        return true;
    }
    let err = max_abs_err(cpu, gpu);
    let cpu_abs = cpu.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let gpu_abs = gpu.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let flag = if err > THRESHOLD {
        " *** BROKEN ***"
    } else {
        ""
    };
    eprintln!(
        "  [{:>20}] max_abs_err={:.6}  cpu_max_abs={:.4}  gpu_max_abs={:.4}  n={}{}",
        stage, err, cpu_abs, gpu_abs, n, flag
    );
    // Print first 8 elements of each for visual comparison
    let show = n.min(8);
    eprintln!("    cpu[0..{}]: {:?}", show, &cpu[..show]);
    eprintln!("    gpu[0..{}]: {:?}", show, &gpu[..show]);
    err > THRESHOLD
}

#[test]
#[serial]
#[ignore]
fn test_cpu_vs_gpu_divergence_qwen() {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }
    if !common::gpu_available() {
        eprintln!("Skipping: no GPU");
        return;
    }

    let file = GgufFile::open(MODEL_PATH).expect("open GGUF");
    let config = ModelConfig::from_gguf(&file).expect("parse config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("load CPU weights");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());

    eprintln!(
        "=== Model: arch={} hidden={} layers={} heads={} kv_heads={} head_dim={}",
        config.architecture,
        config.hidden_size,
        config.num_layers,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim
    );
    eprintln!(
        "    rope_neox={} rope_theta={} norm_eps={}",
        config.rope_neox, config.rope_theta, config.rms_norm_eps
    );

    // ── Token: use "Hello" → first token id ──
    let prompt_tokens = tok.encode("Hello", false);
    let token_id = prompt_tokens[0];
    eprintln!("    token_id={}", token_id);

    // ══════════════════════════════════════
    // STEP 1: EMBEDDING (CPU)
    // ══════════════════════════════════════
    let h = config.hidden_size;
    let mut cpu_hidden = vec![0.0f32; h];
    cpu_embed_token(token_id, &cpu_weights, &mut cpu_hidden, &config, None);
    eprintln!("\n--- CPU embedding: first 8 scales/values ---");
    eprintln!("    cpu_hidden[0..8]: {:?}", &cpu_hidden[..8]);

    // ══════════════════════════════════════
    // STEP 2: GPU EMBEDDING
    // ══════════════════════════════════════
    let caps = gpu::detect().expect("GPU detect");
    let device = GpuDevice::init(caps.device_id).expect("GPU init");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("load GPU weights");

    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("GPU scratch");
    let mut cpu_scratch = CpuForwardScratch::new(&config);
    let mut kv_gpu = GpuKvCache::new(&config, 4).expect("GPU KV");
    let mut kv_cpu = CpuKvCache::new(&config, 4);

    gpu::gpu_embed_token_hybrid(
        &device,
        token_id,
        &gpu_weights,
        &cpu_weights,
        &mut gpu_scratch,
        &mut cpu_scratch,
        &config,
    )
    .expect("GPU embed");
    device.synchronize().expect("sync");

    let gpu_hidden = download_f32(&gpu_scratch.hidden, h, &device);

    eprintln!("\n═══ STAGE 0: EMBEDDING ═══");
    let broken = report("embedding", &cpu_hidden, &gpu_hidden);
    if broken {
        eprintln!(">>> DIVERGENCE STARTS AT EMBEDDING. Q4_0 dequant is broken.");
        eprintln!(">>> Checking Q4_0 block scales separately...");
        check_q4_0_scales(token_id, &cpu_weights, &config);
        return;
    }

    // ══════════════════════════════════════
    // STEP 3-4: Run full forward on both CPU and GPU (all layers).
    // Compare the FINAL hidden state after all layers. If it diverges here,
    // we enable ROCMFORGE_DUMP_LAYER_INTERMEDIATES to bisect per-stage.
    // ══════════════════════════════════════

    // --- CPU: full forward, all layers ---
    let mut cpu_h = cpu_hidden.clone();
    let mut kv_c = CpuKvCache::new(&config, 4);
    let mut cpu_scr = CpuForwardScratch::new(&config);
    use rocmforge::cpu::forward::cpu_full_forward;
    cpu_full_forward(
        &mut cpu_h,
        &cpu_weights,
        &mut kv_c,
        &mut cpu_scr,
        0,
        &config,
    )
    .expect("CPU full forward");

    // --- GPU: full forward, all layers ---
    unsafe {
        std::env::set_var("ROCMFORGE_DISABLE_DECODE_GRAPH", "1");
    }
    rocmforge::gpu::refresh_runtime_env_flags();

    // Re-embed on GPU (scratch.hidden was overwritten by embedding already)
    let mut gpu_scr = GpuForwardScratch::new(&config).expect("GPU scratch");
    let mut cpu_scr_gpu = CpuForwardScratch::new(&config);
    let mut kv_g = GpuKvCache::new(&config, 4).expect("GPU KV");
    gpu::gpu_embed_token_hybrid(
        &device,
        token_id,
        &gpu_weights,
        &cpu_weights,
        &mut gpu_scr,
        &mut cpu_scr_gpu,
        &config,
    )
    .expect("GPU embed");
    gpu::gpu_full_forward_hybrid(
        &device,
        &gpu_weights,
        &cpu_weights,
        &mut kv_g,
        &mut gpu_scr,
        &mut cpu_scr_gpu,
        0,
        &config,
        gpu::GpuLogitsMode::GreedyArgmax,
        token_id,
    )
    .expect("GPU full forward");
    device.synchronize().expect("sync");

    let gpu_h = download_f32(&gpu_scr.hidden, h, &device);

    // ══════════════════════════════════════
    // STEP 4.5: Check gate weights at layer 0
    // ══════════════════════════════════════
    eprintln!("\n═══ GATE WEIGHT VERIFICATION (layer 0) ═══");
    check_gate_weights_layer_0(&cpu_weights, &gpu_weights, &device, &config);

    // ══════════════════════════════════════
    // STEP 5: Compare final hidden state after all layers
    // ══════════════════════════════════════
    eprintln!(
        "\n═══ FINAL HIDDEN STATE COMPARISON (after all {} layers) ═══",
        config.num_layers
    );
    eprintln!(
        "    (threshold = {}, Q4_0 dequant noise expected ~0.01-0.03)\n",
        THRESHOLD
    );

    let broken = report("hidden_after_all_layers", &cpu_h, &gpu_h);
    if broken {
        eprintln!("\n>>> Hidden state diverges after all layers.");
        eprintln!(
            ">>> Re-run with ROCMFORGE_DUMP_LAYER_INTERMEDIATES=1 to see per-stage CPU/GPU dumps."
        );
        eprintln!(">>> Checking Q4_0 dequant scales...");
        check_q4_0_scales(token_id, &cpu_weights, &config);
    }

    // Restore graph default
    unsafe {
        std::env::remove_var("ROCMFORGE_DISABLE_DECODE_GRAPH");
    }
    rocmforge::gpu::refresh_runtime_env_flags();
}

/// Download n f32 elements from a GpuBuffer to host.
fn download_f32(buf: &GpuBuffer, n: usize, device: &GpuDevice) -> Vec<f32> {
    let v = buf.copy_to_host_vec().expect("download");
    let n = n.min(v.len());
    v[..n].to_vec()
}

/// Dump the first few Q4_0 block scales from the token embedding to verify
/// the d * (val - 8) dequant convention. If scales differ between what the
/// loader stored and what the kernel reads, that's the bug.
fn check_q4_0_scales(token_id: u32, weights: &CpuModelWeights, config: &ModelConfig) {
    let h = config.hidden_size;
    let block_elems = 32usize;
    let block_bytes = 18usize; // Q4_0: 2-byte f16 scale + 16-byte quantized
    let num_blocks = h / block_elems;
    let row_offset = token_id as usize * num_blocks * block_bytes;

    eprintln!(
        "\n--- Q4_0 block scales (token {}, first 8 blocks) ---",
        token_id
    );
    eprintln!("    block: scale(f16)  qs[0..4 hex]  dequant[0..4]");
    for b in 0..num_blocks.min(8) {
        let off = row_offset + b * block_bytes;
        let block = &weights.token_emb[off..off + block_bytes];
        let scale_bytes = [block[0], block[1]];
        let scale_f16 = half::f16::from_le_bytes(scale_bytes);
        let scale = scale_f16.to_f32();
        let qs = &block[2..18];
        let dequant: Vec<f32> = (0..4)
            .flat_map(|i| {
                let lo = ((qs[i] & 0x0F) as i32 - 8) as f32 * scale;
                let hi = ((qs[i] >> 4) as i32 - 8) as f32 * scale;
                [lo, hi]
            })
            .take(4)
            .collect();
        eprintln!(
            "    [{}]: scale={:.6}  qs=[{:02x}{:02x}{:02x}{:02x}]  dequant={:?}",
            b, scale, qs[0], qs[1], qs[2], qs[3], dequant
        );
    }
}

#[test]
#[serial]
#[ignore]
fn test_cpu_vs_gpu_layer0_stage_divergence_qwen() {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }
    if !common::gpu_available() {
        eprintln!("Skipping: no GPU");
        return;
    }

    let file = GgufFile::open(MODEL_PATH).expect("open GGUF");
    let config = ModelConfig::from_gguf(&file).expect("parse config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("load CPU weights");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());
    let token_id = tok.encode("Hello", false)[0];

    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;

    // CPU reference: embedding then layer 0
    let mut cpu_hidden = vec![0.0f32; h];
    cpu_embed_token(token_id, &cpu_weights, &mut cpu_hidden, &config, None);
    let cpu_input = cpu_hidden.clone();

    let mut cpu_kv = CpuKvCache::new(&config, 4);
    let mut cpu_scratch = CpuForwardScratch::new(&config);
    let half = config.head_dim / 2;
    for i in 0..half {
        let angle = 0.0f32 * config.rope_freq[i];
        let (s, c) = angle.sin_cos();
        cpu_scratch.rope_sin[i] = s;
        cpu_scratch.rope_cos[i] = c;
    }
    let rope_sin = unsafe { std::slice::from_raw_parts(cpu_scratch.rope_sin.as_ptr(), half) };
    let rope_cos = unsafe { std::slice::from_raw_parts(cpu_scratch.rope_cos.as_ptr(), half) };

    cpu_layer_forward(
        &mut cpu_hidden,
        cpu_weights.layer(0),
        &mut cpu_kv,
        &mut cpu_scratch,
        0,
        0,
        rope_sin,
        rope_cos,
        &config,
        false,
    )
    .expect("CPU layer 0");

    // GPU: embedding then layer 0 (disable graph capture for visibility)
    unsafe {
        std::env::set_var("ROCMFORGE_ENABLE_DECODE_GRAPH", "0");
    }
    rocmforge::gpu::refresh_runtime_env_flags();

    let caps = gpu::detect().expect("GPU detect");
    let device = GpuDevice::init(caps.device_id).expect("GPU init");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("load GPU weights");
    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("GPU scratch");
    let mut cpu_scratch_gpu = CpuForwardScratch::new(&config);
    let mut gpu_kv = GpuKvCache::new(&config, 4).expect("GPU KV");

    gpu::gpu_embed_token_hybrid(
        &device,
        token_id,
        &gpu_weights,
        &cpu_weights,
        &mut gpu_scratch,
        &mut cpu_scratch_gpu,
        &config,
    )
    .expect("GPU embed");
    // overwrite with the same CPU input so the comparison is apples-to-apples
    gpu_scratch
        .hidden
        .copy_from_host(unsafe {
            std::slice::from_raw_parts(
                cpu_input.as_ptr() as *const u8,
                h * std::mem::size_of::<f32>(),
            )
        })
        .expect("upload CPU input");

    gpu::gpu_layer_forward_hybrid(
        &device,
        gpu_weights.layer(0),
        Some(cpu_weights.layer(0)),
        &mut gpu_kv,
        &mut gpu_scratch,
        Some(&mut cpu_scratch_gpu),
        0,
        0,
        0, // token_id (dummy value)
        &config,
        None, // shared_ple_token_emb
        None, // shared_ple_model_proj
        None, // shared_ple_proj_norm
    )
    .expect("GPU layer 0");
    device.synchronize().expect("sync");

    // Download GPU intermediates and compare with CPU scratch
    eprintln!(
        "\n═══ LAYER 0 PER-STAGE CPU vs GPU (sizes h={} q={} kv={} ff={}) ═══",
        h, q_size, kv_size, ff_size
    );
    compare_stage(
        "hidden_in",
        &cpu_input,
        &download_f32(&gpu_scratch.hidden, h, &device),
    );
    compare_stage(
        "attn_normed",
        &cpu_scratch.normed[..h],
        &download_f32(&gpu_scratch.normed, h, &device),
    );
    compare_stage(
        "q",
        &cpu_scratch.q[..q_size],
        &download_f32(&gpu_scratch.q, q_size, &device),
    );
    compare_stage(
        "k",
        &cpu_scratch.k[..kv_size],
        &download_f32(&gpu_scratch.k, kv_size, &device),
    );
    compare_stage(
        "v",
        &cpu_scratch.v[..kv_size],
        &download_f32(&gpu_scratch.v, kv_size, &device),
    );
    compare_stage(
        "attn_out",
        &cpu_scratch.attn_out[..q_size],
        &download_f32(&gpu_scratch.attn_out, q_size, &device),
    );
    compare_stage(
        "attn_layer_out",
        &cpu_scratch.layer_out[..h],
        &download_f32(&gpu_scratch.layer_out, h, &device),
    );
    compare_stage(
        "after_attn_resid",
        &cpu_hidden[..h],
        &download_f32(&gpu_scratch.hidden, h, &device),
    );

    // Re-run CPU layer on a fresh hidden state to capture FFN intermediates
    // (the CPU scratch fields were overwritten by ffn_down output before residual).
    // The CPU scratch after cpu_layer_forward has:
    //   - normed: FFN RMSNorm output
    //   - gate: FFN down projection output (before residual)
    //   - swiglu: SwiGLU output
    //   - hidden: after FFN residual
    compare_stage(
        "ffn_normed",
        &cpu_scratch.normed[..h],
        &download_f32(&gpu_scratch.normed, h, &device),
    );
    compare_stage(
        "ffn_swiglu",
        &cpu_scratch.swiglu[..ff_size],
        &download_f32(&gpu_scratch.swiglu, ff_size, &device),
    );
    compare_stage(
        "ffn_down_out",
        &cpu_scratch.gate[..h],
        &download_f32(&gpu_scratch.gate, h, &device),
    );
    compare_stage(
        "after_ffn_resid",
        &cpu_hidden[..h],
        &download_f32(&gpu_scratch.hidden, h, &device),
    );

    unsafe {
        std::env::remove_var("ROCMFORGE_ENABLE_DECODE_GRAPH");
    }
    rocmforge::gpu::refresh_runtime_env_flags();
}

fn compare_stage(name: &str, cpu: &[f32], gpu: &[f32]) {
    let n = cpu.len().min(gpu.len());
    let mut max_err = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut idx = 0usize;
    for i in 0..n {
        let err = (cpu[i] - gpu[i]).abs();
        let denom = cpu[i].abs().max(1e-6);
        let rel = err / denom;
        if err > max_err {
            max_err = err;
            idx = i;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    eprintln!(
        "  [{:>20}] len={} max_abs_err={:.6} max_rel_err={:.6} @ idx={}",
        name, n, max_err, max_rel, idx
    );
    if max_err > 0.001 {
        let show = 4usize;
        eprintln!(
            "    cpu[{:?}]: {:?}",
            idx.saturating_sub(show)..(idx + show).min(n),
            &cpu[idx.saturating_sub(show)..(idx + show).min(n)]
        );
        eprintln!(
            "    gpu[{:?}]: {:?}",
            idx.saturating_sub(show)..(idx + show).min(n),
            &gpu[idx.saturating_sub(show)..(idx + show).min(n)]
        );
    }
}

fn upload_f32(data: &[f32]) -> rocmforge::gpu::GpuResult<GpuBuffer> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    let mut buf = GpuBuffer::alloc(bytes.len())?;
    buf.copy_from_host(bytes)?;
    Ok(buf)
}

fn upload_u8(data: &[u8]) -> rocmforge::gpu::GpuResult<GpuBuffer> {
    let mut buf = GpuBuffer::alloc(data.len())?;
    buf.copy_from_host(data)?;
    Ok(buf)
}

fn download_f32_from_gpu(buf: &GpuBuffer, n: usize) -> rocmforge::gpu::GpuResult<Vec<f32>> {
    let mut dst = vec![0.0f32; n];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, n * std::mem::size_of::<f32>())
    };
    buf.copy_to_host(bytes)?;
    Ok(dst)
}

fn download_u8_from_gpu(buf: &GpuBuffer, n: usize) -> rocmforge::gpu::GpuResult<Vec<u8>> {
    let mut dst = vec![0u8; n];
    buf.copy_to_host(&mut dst)?;
    Ok(dst)
}

/// Download `n` f32 elements from a raw device pointer on `device`'s stream.
fn download_device_f32(
    ptr: *const f32,
    n: usize,
    device: &GpuDevice,
) -> rocmforge::gpu::GpuResult<Vec<f32>> {
    let mut dst = vec![0.0f32; n];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, n * std::mem::size_of::<f32>())
    };
    rocmforge::gpu::ffi::hip_memcpy_d2h_async(
        bytes.as_mut_ptr(),
        ptr as *const u8,
        bytes.len(),
        device.stream(),
    )?;
    device.synchronize()?;
    Ok(dst)
}

/// Reference int32 per-block accumulation for Q4_0 weights x Q8_0 activations.
/// Mirrors `rocmforge::cpu::ops::gemv::gemv_q4_0_q8_0` and the llama.cpp
/// `ggml_vec_dot_q4_0_q8_0_generic` reference.
fn q4_0_q8_0_cpu_oracle(
    weights: &[u8],
    input_q8: &[u8],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q4_BLOCK_BYTES;
    assert_eq!(input_q8.len(), num_blocks * 34);

    let mut out = vec![0.0f32; out_dim];
    for (col, val) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        let col_offset = col * col_bytes;
        for block_idx in 0..num_blocks {
            let w_block = &weights[col_offset + block_idx * Q4_BLOCK_BYTES
                ..col_offset + (block_idx + 1) * Q4_BLOCK_BYTES];
            let x_block = &input_q8[block_idx * 34..(block_idx + 1) * 34];
            let w_scale = load_f16_scale(&w_block[..2]);
            let x_scale = load_f16_scale(&x_block[..2]);
            let scale = w_scale * x_scale;
            let qs = &w_block[2..18];
            let x_qs = &x_block[2..];

            let mut block_sum = 0i32;
            for i in 0..16 {
                let packed = qs[i];
                block_sum += (((packed & 0x0F) as i32) - 8) * ((x_qs[i] as i8) as i32);
                block_sum += (((packed >> 4) as i32) - 8) * ((x_qs[i + 16] as i8) as i32);
            }

            acc += scale * block_sum as f32;
        }
        *val = acc;
    }
    out
}

fn quantize_q8_0_cpu(x: &[f32]) -> Vec<u8> {
    let n = x.len();
    let num_blocks = n.div_ceil(32);
    let mut out = vec![0u8; num_blocks * 34];
    for b in 0..num_blocks {
        let start = b * 32;
        let end = ((b + 1) * 32).min(n);
        let mut amax = 0.0f32;
        for i in start..end {
            amax = amax.max(x[i].abs());
        }
        let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let scale_f16 = half::f16::from_f32(scale);
        let bytes = scale_f16.to_le_bytes();
        out[b * 34] = bytes[0];
        out[b * 34 + 1] = bytes[1];
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for i in start..end {
            let q = (x[i] * inv).round().clamp(-127.0, 127.0) as i8;
            out[b * 34 + 2 + (i - start)] = q as u8;
        }
    }
    out
}

fn to_gpu_meta(meta: &rocmforge::cpu::weights::WeightMeta) -> rocmforge::gpu::WeightMeta {
    rocmforge::gpu::WeightMeta {
        wtype: meta.wtype,
        dims: meta.dims.clone(),
        needs_transpose: meta.needs_transpose,
        role: meta.role,
        svd_k: meta.svd_k,
    }
}

#[test]
#[serial]
#[ignore]
fn test_rms_norm_parity_real_activations() {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }
    if !common::gpu_available() {
        eprintln!("Skipping: no GPU");
        return;
    }

    unsafe {
        std::env::set_var("ROCMFORGE_ENABLE_DECODE_GRAPH", "0");
    }
    rocmforge::gpu::refresh_runtime_env_flags();

    let file = GgufFile::open(MODEL_PATH).expect("open GGUF");
    let config = ModelConfig::from_gguf(&file).expect("parse config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("load CPU weights");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());
    let token_id = tok.encode("Hello", false)[0];

    let h = config.hidden_size;
    let mut cpu_hidden = vec![0.0f32; h];
    cpu_embed_token(token_id, &cpu_weights, &mut cpu_hidden, &config, None);

    let layer0 = cpu_weights.layer(0);
    let mut cpu_attn_normed = vec![0.0f32; h];
    rms_norm(
        &cpu_hidden,
        &layer0.attn_norm,
        &mut cpu_attn_normed,
        config.rms_norm_eps,
    );

    let caps = gpu::detect().expect("GPU detect");
    let device = GpuDevice::init(caps.device_id).expect("GPU init");
    let d_x = upload_f32(&cpu_hidden).expect("upload hidden");
    let d_weight = upload_f32(&layer0.attn_norm).expect("upload attn_norm");
    let d_out = GpuBuffer::alloc(h * std::mem::size_of::<f32>()).expect("alloc norm out");

    gpu::ops::gpu_dispatch_rms_norm(
        &device,
        d_x.as_ptr() as *const f32,
        d_weight.as_ptr() as *const f32,
        d_out.as_ptr() as *mut f32,
        h,
        config.rms_norm_eps,
        device.stream(),
    )
    .expect("GPU RMS norm");
    device.synchronize().expect("sync");

    let gpu_attn_normed = download_f32_from_gpu(&d_out, h).expect("download gpu norm");

    eprintln!("\n═══ RMS norm parity (attn, h={}) ═══", h);
    report("attn_normed", &cpu_attn_normed, &gpu_attn_normed);

    let mut cpu_ffn_normed = vec![0.0f32; h];
    rms_norm(
        &cpu_hidden,
        &layer0.ffn_norm,
        &mut cpu_ffn_normed,
        config.rms_norm_eps,
    );
    let d_ffn_weight = upload_f32(&layer0.ffn_norm).expect("upload ffn_norm");
    gpu::ops::gpu_dispatch_rms_norm(
        &device,
        d_x.as_ptr() as *const f32,
        d_ffn_weight.as_ptr() as *const f32,
        d_out.as_ptr() as *mut f32,
        h,
        config.rms_norm_eps,
        device.stream(),
    )
    .expect("GPU FFN RMS norm");
    device.synchronize().expect("sync ffn norm");
    let gpu_ffn_normed = download_f32_from_gpu(&d_out, h).expect("download gpu ffn norm");
    eprintln!("\n═══ RMS norm parity (ffn, h={}) ═══", h);
    report("ffn_normed", &cpu_ffn_normed, &gpu_ffn_normed);

    unsafe {
        std::env::remove_var("ROCMFORGE_ENABLE_DECODE_GRAPH");
    }
    rocmforge::gpu::refresh_runtime_env_flags();
}

#[test]
#[serial]
#[ignore]
fn test_q4_0_fused_qkv_parity_real_activations() {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }
    if !common::gpu_available() {
        eprintln!("Skipping: no GPU");
        return;
    }

    unsafe {
        std::env::set_var("ROCMFORGE_ENABLE_DECODE_GRAPH", "0");
    }
    rocmforge::gpu::refresh_runtime_env_flags();

    let file = GgufFile::open(MODEL_PATH).expect("open GGUF");
    let config = ModelConfig::from_gguf(&file).expect("parse config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("load CPU weights");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());
    let token_id = tok.encode("Hello", false)[0];

    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;

    let mut cpu_hidden = vec![0.0f32; h];
    cpu_embed_token(token_id, &cpu_weights, &mut cpu_hidden, &config, None);

    let layer0 = cpu_weights.layer(0);
    let mut cpu_attn_normed = vec![0.0f32; h];
    rms_norm(
        &cpu_hidden,
        &layer0.attn_norm,
        &mut cpu_attn_normed,
        config.rms_norm_eps,
    );

    // CPU reference for Q/K/V using the same Q8 GEMV path the CPU decode uses.
    let mut cpu_q = vec![0.0f32; q_size];
    let mut cpu_k = vec![0.0f32; kv_size];
    let mut cpu_v = vec![0.0f32; kv_size];
    let mut q8_scratch = vec![0u8; rocmforge::gpu::kernels::q8_0_workspace_bytes(h)];
    dispatch_gemv(
        &layer0.attn_q,
        &layer0.attn_q_meta,
        &cpu_attn_normed,
        &mut cpu_q,
        q_size,
        h,
        Some(&mut q8_scratch),
    )
    .expect("CPU Q GEMV");
    dispatch_gemv(
        &layer0.attn_k,
        &layer0.attn_k_meta,
        &cpu_attn_normed,
        &mut cpu_k,
        kv_size,
        h,
        Some(&mut q8_scratch),
    )
    .expect("CPU K GEMV");
    dispatch_gemv(
        &layer0.attn_v,
        &layer0.attn_v_meta,
        &cpu_attn_normed,
        &mut cpu_v,
        kv_size,
        h,
        Some(&mut q8_scratch),
    )
    .expect("CPU V GEMV");

    // CPU f32-per-element fallback reference (the path used when no Q8 scratch is provided).
    let mut cpu_q_f32 = vec![0.0f32; q_size];
    let mut cpu_k_f32 = vec![0.0f32; kv_size];
    let mut cpu_v_f32 = vec![0.0f32; kv_size];
    dispatch_gemv(
        &layer0.attn_q,
        &layer0.attn_q_meta,
        &cpu_attn_normed,
        &mut cpu_q_f32,
        q_size,
        h,
        None,
    )
    .expect("CPU Q f32");
    dispatch_gemv(
        &layer0.attn_k,
        &layer0.attn_k_meta,
        &cpu_attn_normed,
        &mut cpu_k_f32,
        kv_size,
        h,
        None,
    )
    .expect("CPU K f32");
    dispatch_gemv(
        &layer0.attn_v,
        &layer0.attn_v_meta,
        &cpu_attn_normed,
        &mut cpu_v_f32,
        kv_size,
        h,
        None,
    )
    .expect("CPU V f32");

    // GPU fused QKV dispatch: this is the exact path used by gpu_layer_forward_hybrid.
    let caps = gpu::detect().expect("GPU detect");
    let device = GpuDevice::init(caps.device_id).expect("GPU init");
    let d_input = upload_f32(&cpu_attn_normed).expect("upload normed");
    let d_wq = upload_u8(&layer0.attn_q).expect("upload q weights");
    let d_wk = upload_u8(&layer0.attn_k).expect("upload k weights");
    let d_wv = upload_u8(&layer0.attn_v).expect("upload v weights");
    let d_q = GpuBuffer::alloc(q_size * std::mem::size_of::<f32>()).expect("alloc q");
    let d_k = GpuBuffer::alloc(kv_size * std::mem::size_of::<f32>()).expect("alloc k");
    let d_v = GpuBuffer::alloc(kv_size * std::mem::size_of::<f32>()).expect("alloc v");

    gpu::ops::gpu_dispatch_fused_qkv_on_stream(
        &device,
        &d_wq,
        &to_gpu_meta(&layer0.attn_q_meta),
        None,
        None,
        &d_wk,
        &to_gpu_meta(&layer0.attn_k_meta),
        None,
        None,
        &d_wv,
        &to_gpu_meta(&layer0.attn_v_meta),
        None,
        None,
        d_input.as_ptr() as *const f32,
        d_q.as_ptr() as *mut f32,
        d_k.as_ptr() as *mut f32,
        d_v.as_ptr() as *mut f32,
        q_size,
        kv_size,
        h,
        std::ptr::null_mut(),
        device.stream(),
    )
    .expect("GPU fused QKV");
    device.synchronize().expect("sync fused QKV");

    let gpu_q = download_f32_from_gpu(&d_q, q_size).expect("download q");
    let gpu_k = download_f32_from_gpu(&d_k, kv_size).expect("download k");
    let gpu_v = download_f32_from_gpu(&d_v, kv_size).expect("download v");

    eprintln!("\n═══ Q4_0 fused QKV dispatch parity (real attn_normed) ═══");
    eprintln!("--- vs CPU Q8 per-block reference (CPU decode path) ---");
    report("fused_qkv_q", &cpu_q, &gpu_q);
    report("fused_qkv_k", &cpu_k, &gpu_k);
    report("fused_qkv_v", &cpu_v, &gpu_v);
    eprintln!("--- vs CPU f32-per-element fallback ---");
    report("fused_qkv_q_f32_ref", &cpu_q_f32, &gpu_q);
    report("fused_qkv_k_f32_ref", &cpu_k_f32, &gpu_k);
    report("fused_qkv_v_f32_ref", &cpu_v_f32, &gpu_v);

    unsafe {
        std::env::remove_var("ROCMFORGE_ENABLE_DECODE_GRAPH");
    }
    rocmforge::gpu::refresh_runtime_env_flags();
}

#[test]
#[serial]
#[ignore]
fn test_q4_0_gemv_parity_real_activations() {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }
    if !common::gpu_available() {
        eprintln!("Skipping: no GPU");
        return;
    }

    unsafe {
        std::env::set_var("ROCMFORGE_ENABLE_DECODE_GRAPH", "0");
    }
    rocmforge::gpu::refresh_runtime_env_flags();

    let file = GgufFile::open(MODEL_PATH).expect("open GGUF");
    let config = ModelConfig::from_gguf(&file).expect("parse config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("load CPU weights");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());
    let token_id = tok.encode("Hello", false)[0];

    let h = config.hidden_size;
    let mut cpu_hidden = vec![0.0f32; h];
    cpu_embed_token(token_id, &cpu_weights, &mut cpu_hidden, &config, None);

    // CPU reference RMSNorm for the attention path.
    let mut cpu_attn_normed = vec![0.0f32; h];
    rms_norm(
        &cpu_hidden,
        &cpu_weights.layer(0).attn_norm,
        &mut cpu_attn_normed,
        config.rms_norm_eps,
    );

    let cpu_q8_attn = quantize_q8_0_cpu(&cpu_attn_normed);

    // GPU: upload the same CPU-normalized activation and quantize on device.
    let caps = gpu::detect().expect("GPU detect");
    let device = GpuDevice::init(caps.device_id).expect("GPU init");
    let d_normed_f32 = upload_f32(&cpu_attn_normed).expect("upload normed f32");
    let d_normed_q8 =
        GpuBuffer::alloc(rocmforge::gpu::kernels::q8_0_workspace_bytes(h)).expect("alloc q8");
    rocmforge::gpu::kernels::quantize_q8_0_on_stream(
        d_normed_f32.as_ptr() as *const f32,
        d_normed_q8.as_ptr(),
        h,
        device.stream(),
    )
    .expect("GPU quantize Q8");
    device.synchronize().expect("sync after quantize");
    let gpu_q8_attn =
        download_u8_from_gpu(&d_normed_q8, cpu_q8_attn.len()).expect("download gpu q8");

    // Compare Q8 activations byte-by-byte (ignore tail padding blocks).
    let q8_blocks = h.div_ceil(32);
    let mut q8_byte_diffs = 0usize;
    let mut q8_scale_diffs = 0usize;
    for b in 0..q8_blocks {
        let off = b * 34;
        if cpu_q8_attn[off] != gpu_q8_attn[off] || cpu_q8_attn[off + 1] != gpu_q8_attn[off + 1] {
            q8_scale_diffs += 1;
        }
        for i in 2..34 {
            if cpu_q8_attn[off + i] != gpu_q8_attn[off + i] {
                q8_byte_diffs += 1;
            }
        }
    }
    eprintln!("\n═══ Q8_0 activation quantization parity (attn_normed) ═══");
    eprintln!(
        "  blocks={} scale_diff_blocks={} quant_byte_diffs={}",
        q8_blocks, q8_scale_diffs, q8_byte_diffs
    );

    // Projections to test.  Order: (name, weights, meta, out_dim, in_dim).
    let layer0 = cpu_weights.layer(0);
    let mut projections: Vec<(
        &str,
        &[u8],
        &rocmforge::cpu::weights::WeightMeta,
        usize,
        usize,
    )> = Vec::new();
    if let Some(ref w) = layer0.attn_qkv {
        let meta = layer0
            .attn_qkv_meta
            .as_ref()
            .expect("attn_qkv weight exists but meta missing");
        projections.push((
            "attn_qkv",
            w.as_slice(),
            meta,
            config.num_heads * config.head_dim + 2 * config.num_kv_heads * config.head_dim,
            h,
        ));
    } else {
        projections.push((
            "attn_q",
            layer0.attn_q.as_slice(),
            &layer0.attn_q_meta,
            h,
            h,
        ));
        projections.push((
            "attn_k",
            layer0.attn_k.as_slice(),
            &layer0.attn_k_meta,
            config.num_kv_heads * config.head_dim,
            h,
        ));
        projections.push((
            "attn_v",
            layer0.attn_v.as_slice(),
            &layer0.attn_v_meta,
            config.num_kv_heads * config.head_dim,
            h,
        ));
    }
    projections.push((
        "attn_o",
        layer0.attn_o.as_slice(),
        &layer0.attn_o_meta,
        h,
        h,
    ));
    if let Some(ref w) = layer0.attn_gate {
        let meta = layer0
            .attn_gate_meta
            .as_ref()
            .expect("attn_gate weight exists but meta missing");
        projections.push((
            "attn_gate",
            w.as_slice(),
            meta,
            w.len() / (h * Q4_BLOCK_ELEMS / Q4_BLOCK_BYTES),
            h,
        ));
    }
    if let Some(ref w) = layer0.ffn_gate {
        let meta = layer0
            .ffn_gate_meta
            .as_ref()
            .expect("ffn_gate weight exists but meta missing");
        projections.push(("ffn_gate", w.as_slice(), meta, config.intermediate_size, h));
    }
    projections.push((
        "ffn_up",
        layer0.ffn_up.as_slice(),
        &layer0.ffn_up_meta,
        config.intermediate_size,
        h,
    ));

    eprintln!("\n═══ Q4_0 GEMV CPU vs GPU (same f32 input, each backend quantizes its own Q8) ═══");
    for (name, cpu_w, cpu_meta, out_dim, in_dim) in projections {
        if cpu_meta.wtype != GgmlType::Q4_0 {
            eprintln!("  [{:>12}] skipped (wtype={:?})", name, cpu_meta.wtype);
            continue;
        }
        if in_dim % Q4_BLOCK_ELEMS != 0 {
            eprintln!("  [{:>12}] skipped (in_dim {} not aligned)", name, in_dim);
            continue;
        }

        // CPU reference GEMV (quantizes input with CPU quantizer).
        let mut cpu_y = vec![0.0f32; out_dim];
        let mut q8_scratch = vec![0u8; rocmforge::gpu::kernels::q8_0_workspace_bytes(in_dim)];
        dispatch_gemv(
            cpu_w,
            cpu_meta,
            &cpu_attn_normed,
            &mut cpu_y,
            out_dim,
            in_dim,
            Some(&mut q8_scratch),
        )
        .expect("CPU GEMV");

        // GPU dispatch GEMV (quantizes input with GPU quantizer).
        let d_w = upload_u8(cpu_w).expect("upload weights");
        let gpu_meta = to_gpu_meta(cpu_meta);
        let d_y = GpuBuffer::alloc(out_dim * std::mem::size_of::<f32>()).expect("alloc gpu output");
        gpu::ops::gpu_dispatch_gemv_on_stream(
            &device,
            &d_w,
            &gpu_meta,
            d_normed_f32.as_ptr() as *const f32,
            d_y.as_ptr() as *mut f32,
            out_dim,
            in_dim,
            device.stream(),
        )
        .expect("GPU GEMV");
        device.synchronize().expect("sync GPU GEMV");
        let gpu_y = download_f32_from_gpu(&d_y, out_dim).expect("download gpu output");

        // CPU oracle using the *GPU-quantized* input isolates kernel arithmetic.
        let gpu_y_expected = q4_0_q8_0_cpu_oracle(cpu_w, &gpu_q8_attn, out_dim, in_dim);

        let full_err = max_abs_err(&cpu_y, &gpu_y);
        let kernel_err = max_abs_err(&gpu_y_expected, &gpu_y);
        eprintln!(
            "  [{:>12}] out={} in={}  full_err={:.6}  kernel_err={:.6}  quantizer_contrib={:.6}",
            name,
            out_dim,
            in_dim,
            full_err,
            kernel_err,
            (full_err - kernel_err).abs()
        );
        if full_err > 0.1 {
            let idx = cpu_y
                .iter()
                .zip(&gpu_y)
                .enumerate()
                .map(|(i, (a, b))| (i, (a - b).abs()))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .expect("non-empty iterator");
            eprintln!(
                "    cpu[{}]={:.6} gpu[{}]={:.6}",
                idx, cpu_y[idx], idx, gpu_y[idx]
            );
        }
    }

    unsafe {
        std::env::remove_var("ROCMFORGE_ENABLE_DECODE_GRAPH");
    }
    rocmforge::gpu::refresh_runtime_env_flags();
}

#[test]
#[serial]
#[ignore]
fn test_qwen_gpu_greedy_token_matches_cpu() {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }
    if !common::gpu_available() {
        eprintln!("Skipping: no GPU");
        return;
    }

    unsafe {
        std::env::set_var("ROCMFORGE_ENABLE_DECODE_GRAPH", "0");
    }
    rocmforge::gpu::refresh_runtime_env_flags();

    let file = GgufFile::open(MODEL_PATH).expect("open GGUF");
    let config = ModelConfig::from_gguf(&file).expect("parse config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("load CPU weights");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());
    let token_id = tok.encode("Hello", false)[0];

    let h = config.hidden_size;
    let half = config.head_dim / 2;
    let rope_sin = vec![0.0f32; half];
    let rope_cos = vec![0.0f32; half];

    // CPU reference: embed, then all layers.
    let mut cpu_hidden = vec![0.0f32; h];
    cpu_embed_token(token_id, &cpu_weights, &mut cpu_hidden, &config, None);
    let mut cpu_kv = CpuKvCache::new(&config, 4);
    let mut cpu_scratch = CpuForwardScratch::new(&config);
    for layer in 0..config.num_layers {
        cpu_layer_forward(
            &mut cpu_hidden,
            cpu_weights.layer(layer),
            &mut cpu_kv,
            &mut cpu_scratch,
            layer,
            0,
            &rope_sin,
            &rope_cos,
            &config,
            false,
        )
        .expect("CPU layer");
    }

    // CPU lm_head logits.
    let vocab_size =
        cpu_weights.lm_head_meta.dims[0].max(cpu_weights.lm_head_meta.dims[1]) as usize;
    let mut cpu_logits = vec![0.0f32; vocab_size];
    let mut q8_scratch = vec![0u8; rocmforge::gpu::kernels::q8_0_workspace_bytes(h)];
    dispatch_gemv(
        &cpu_weights.lm_head,
        &cpu_weights.lm_head_meta,
        &cpu_hidden,
        &mut cpu_logits,
        vocab_size,
        h,
        Some(&mut q8_scratch),
    )
    .expect("CPU lm_head");
    let cpu_next = argmax(&cpu_logits) as u32;

    // GPU: embed, then all layers (no graph capture).
    let caps = gpu::detect().expect("GPU detect");
    let device = GpuDevice::init(caps.device_id).expect("GPU init");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("load GPU weights");
    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("GPU scratch");
    let mut cpu_scratch_gpu = CpuForwardScratch::new(&config);
    let mut gpu_kv = GpuKvCache::new(&config, 4).expect("GPU KV");

    gpu::gpu_embed_token_hybrid(
        &device,
        token_id,
        &gpu_weights,
        &cpu_weights,
        &mut gpu_scratch,
        &mut cpu_scratch_gpu,
        &config,
    )
    .expect("GPU embed");

    for layer in 0..config.num_layers {
        gpu::gpu_layer_forward_hybrid(
            &device,
            gpu_weights.layer(layer),
            Some(cpu_weights.layer(layer)),
            &mut gpu_kv,
            &mut gpu_scratch,
            Some(&mut cpu_scratch_gpu),
            layer,
            0,
            0, // token_id (dummy value)
            &config,
            None, // shared_ple_token_emb
            None, // shared_ple_model_proj
            None, // shared_ple_proj_norm
        )
        .expect("GPU layer");
    }
    device.synchronize().expect("sync after layers");

    let gpu_hidden = download_f32_from_gpu(&gpu_scratch.hidden, h).expect("download gpu hidden");

    // GPU lm_head logits.
    let d_input = upload_f32(&gpu_hidden).expect("upload gpu hidden");
    let d_lm_head = gpu_weights.lm_head.as_dense().expect("lm_head dense");
    let mut d_logits =
        GpuBuffer::alloc(vocab_size * std::mem::size_of::<f32>()).expect("alloc logits");
    gpu::ops::gpu_dispatch_gemv_on_stream(
        &device,
        d_lm_head,
        &gpu_weights.lm_head_meta,
        d_input.as_ptr() as *const f32,
        d_logits.as_ptr() as *mut f32,
        vocab_size,
        h,
        device.stream(),
    )
    .expect("GPU lm_head");
    device.synchronize().expect("sync lm_head");
    let gpu_logits = download_f32_from_gpu(&d_logits, vocab_size).expect("download gpu logits");
    let gpu_next = argmax(&gpu_logits) as u32;

    eprintln!("\n═══ Qwen greedy token (CPU vs GPU) ═══");
    eprintln!(
        "  CPU next token: {}  GPU next token: {}",
        cpu_next, gpu_next
    );
    eprintln!(
        "  CPU logits max={:.4}  GPU logits max={:.4}",
        cpu_logits.iter().fold(0.0f32, |m, v| m.max(*v)),
        gpu_logits.iter().fold(0.0f32, |m, v| m.max(*v))
    );
    assert_eq!(cpu_next, gpu_next, "GPU greedy token diverged from CPU");

    unsafe {
        std::env::remove_var("ROCMFORGE_ENABLE_DECODE_GRAPH");
    }
    rocmforge::gpu::refresh_runtime_env_flags();
}

/// Regression: batched GPU prefill must match CPU prefill for multi-token prompts.
///
/// This exercises `gpu_batched_prefill_forward` with `seq_len > 1` and verifies
/// both the greedy next-token and the final logits agree with the CPU batched
/// prefill reference. It catches pointer-arithmetic or row-layout bugs that only
/// appear on token positions > 0.
#[test]
#[serial]
#[ignore]
fn test_gpu_batched_prefill_matches_cpu_multi_token_qwen() {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }
    if !common::gpu_available() {
        eprintln!("Skipping: no GPU");
        return;
    }

    let file = GgufFile::open(MODEL_PATH).expect("open GGUF");
    let config = ModelConfig::from_gguf(&file).expect("parse config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("load CPU weights");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());

    let prompt = "Hello, world";
    let tokens = tok.encode(prompt, false);
    assert!(
        tokens.len() > 1,
        "regression prompt must span multiple tokens"
    );

    // CPU reference: per-token decode loop (same path that produced the trusted
    // greedy tokens in `test_qwen_gpu_greedy_token_matches_cpu`). Batched CPU
    // prefill uses a different attention accumulation, so we compare against
    // the decode reference rather than the batched CPU prefill.
    let mut cpu_hidden = vec![0.0f32; config.hidden_size];
    let mut kv_cpu = CpuKvCache::new(&config, tokens.len().max(1));
    let mut cpu_scratch = CpuForwardScratch::new(&config);
    for (pos, &token) in tokens.iter().enumerate() {
        cpu_embed_token(token, &cpu_weights, &mut cpu_hidden, &config, None);
        cpu_full_forward(
            &mut cpu_hidden,
            &cpu_weights,
            &mut kv_cpu,
            &mut cpu_scratch,
            pos,
            &config,
        )
        .expect("CPU decode step");
    }

    // CPU final logits for the last prompt position.
    let mut cpu_normed = vec![0.0f32; config.hidden_size];
    rms_norm(
        &cpu_hidden,
        &cpu_weights.output_norm,
        &mut cpu_normed,
        config.rms_norm_eps,
    );
    let mut cpu_logits = vec![0.0f32; config.vocab_size];
    let mut q8_scratch =
        vec![0u8; rocmforge::gpu::kernels::q8_0_workspace_bytes(config.hidden_size)];
    dispatch_gemv(
        &cpu_weights.lm_head,
        &cpu_weights.lm_head_meta,
        &cpu_normed,
        &mut cpu_logits,
        config.vocab_size,
        config.hidden_size,
        Some(&mut q8_scratch),
    )
    .expect("CPU lm_head");
    let cpu_token = argmax(&cpu_logits) as u32;

    // GPU batched prefill (the path under test).
    let caps = gpu::detect().expect("GPU detect");
    let device = GpuDevice::init(caps.device_id).expect("GPU init");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("load GPU weights");

    let mut kv_gpu = GpuKvCache::new(&config, tokens.len().max(1)).expect("GPU KV");
    let mut gpu_prefill_scratch =
        rocmforge::gpu::GpuPrefillScratch::new(&config, tokens.len()).expect("GPU prefill scratch");
    let mut host_scratch = CpuForwardScratch::new(&config);

    gpu::gpu_batched_prefill_forward(
        &device,
        &gpu_weights,
        &cpu_weights,
        &mut kv_gpu,
        &mut gpu_prefill_scratch,
        &mut host_scratch,
        &tokens,
        0,
        &config,
        gpu::GpuLogitsMode::DownloadToHost,
    )
    .expect("GPU batched prefill");

    let gpu_token =
        rocmforge::cpu::sampler::cpu_sample_greedy(&host_scratch.logits[..config.vocab_size]);
    assert_eq!(
        Some(gpu_token),
        Some(cpu_token),
        "GPU batched prefill greedy token diverged from CPU"
    );

    // Compare final hidden state (last prompt position) — this is where the
    // byte-vs-element pointer bug corrupted rows > 0, so a tight tolerance
    // would have failed catastrophically. 0.5 leaves room for Q4_0 noise
    // across 24 layers while still catching the regression.
    let last_pos = tokens.len() - 1;
    let gpu_hidden = download_device_f32(
        gpu_prefill_scratch.hidden_row_ptr(last_pos, config.hidden_size),
        config.hidden_size,
        &device,
    )
    .expect("download last hidden row");
    let hidden_err = max_abs_err(&cpu_hidden, &gpu_hidden);

    // Also check raw logits, since the pointer bug corrupted rows > 0 silently.
    let logits_err = max_abs_err(
        &cpu_logits[..config.vocab_size],
        &host_scratch.logits[..config.vocab_size],
    );
    eprintln!("═══ batched prefill parity (seq_len={}) ═══", tokens.len());
    eprintln!(
        "  CPU token={}  GPU token={}  hidden_err={:.6}  logits_err={:.6}",
        cpu_token, gpu_token, hidden_err, logits_err
    );
    assert!(
        hidden_err < 1.0,
        "batched prefill final hidden max_abs_err={:.6} >= 1.0",
        hidden_err
    );
    assert!(
        logits_err < 0.5,
        "batched prefill logits max_abs_err={:.6} >= 0.5",
        logits_err
    );
}

/// Check if CPU and GPU gate weights match at layer 0.
fn check_gate_weights_layer_0(
    cpu_weights: &CpuModelWeights,
    gpu_weights: &gpu::GpuModelWeights,
    device: &GpuDevice,
    config: &ModelConfig,
) {
    let layer0_cpu = cpu_weights.layer(0);
    let layer0_gpu = gpu_weights.layer(0);

    // Check if gate weights exist
    let (cpu_gate_data, cpu_gate_meta) = match (
        layer0_cpu.ffn_gate.as_ref(),
        layer0_cpu.ffn_gate_meta.as_ref(),
    ) {
        (Some(gate), Some(meta)) => (gate, meta),
        _ => {
            eprintln!("  No gate weights found (non-SwiGLU architecture?)");
            return;
        }
    };

    let (gpu_gate_buf, gpu_gate_meta) = match (
        layer0_gpu.ffn_gate.as_ref(),
        layer0_gpu.ffn_gate_meta.as_ref(),
    ) {
        (Some(gate), Some(meta)) => (gate, meta),
        _ => {
            eprintln!("  ERROR: GPU gate weights missing but CPU has them!");
            return;
        }
    };

    let ff_size = config.intermediate_size;
    let h = config.hidden_size;

    // Verify metadata matches
    eprintln!("  Metadata check:");
    eprintln!(
        "    CPU wtype={:?} dims={:?}",
        cpu_gate_meta.wtype, cpu_gate_meta.dims
    );
    eprintln!(
        "    GPU wtype={:?} dims={:?}",
        gpu_gate_meta.wtype, gpu_gate_meta.dims
    );

    if cpu_gate_meta.wtype != gpu_gate_meta.wtype {
        eprintln!("  ERROR: Weight type mismatch!");
        return;
    }

    if cpu_gate_meta.dims != gpu_gate_meta.dims {
        eprintln!("  ERROR: Dimension mismatch!");
        return;
    }

    // Download GPU gate weights
    let gpu_gate_bytes = match download_u8_from_gpu(gpu_gate_buf, cpu_gate_data.len()) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("  ERROR: Failed to download GPU gate weights: {:?}", e);
            return;
        }
    };

    // Compare first 100 bytes
    let compare_len = cpu_gate_data.len().min(100).min(gpu_gate_bytes.len());
    let mut diff_count = 0usize;
    for i in 0..compare_len {
        if cpu_gate_data[i] != gpu_gate_bytes[i] {
            diff_count += 1;
            if diff_count <= 10 {
                eprintln!(
                    "  Byte {}: CPU={:02x} GPU={:02x}",
                    i, cpu_gate_data[i], gpu_gate_bytes[i]
                );
            }
        }
    }

    eprintln!(
        "  Byte comparison (first {} bytes): {} differences",
        compare_len, diff_count
    );

    // Check if GPU weights are all zero
    let gpu_zero_count = gpu_gate_bytes.iter().filter(|&&b| b == 0).count();
    let cpu_zero_count = cpu_gate_data.iter().filter(|&&b| b == 0).count();

    eprintln!(
        "  Zero bytes: CPU={} / {} ({:.1}%), GPU={} / {} ({:.1}%)",
        cpu_zero_count,
        cpu_gate_data.len(),
        100.0 * cpu_zero_count as f32 / cpu_gate_data.len() as f32,
        gpu_zero_count,
        gpu_gate_bytes.len(),
        100.0 * gpu_zero_count as f32 / gpu_gate_bytes.len() as f32
    );

    // Verdict
    if gpu_zero_count == gpu_gate_bytes.len() {
        eprintln!("  *** VERDICT: GPU GATE WEIGHTS ARE ALL ZERO - LOADER BUG ***");
    } else if diff_count == 0 {
        eprintln!("  *** VERDICT: GPU gate weights match CPU exactly - OK ***");
    } else {
        eprintln!(
            "  *** VERDICT: GPU gate weights differ from CPU - {} differences ***",
            diff_count
        );
    }
}
