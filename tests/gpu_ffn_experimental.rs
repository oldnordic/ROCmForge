#![cfg(feature = "gpu")]

mod common;

use rocmforge::config::{detect_chat_template, ModelConfig};
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward, cpu_layer_forward},
    ops::{
        add_bias, dispatch_gemv as cpu_dispatch_gemv, flash_attn_decode, residual_add, rms_norm,
        rope, silu_fuse,
    },
    weights::CpuModelWeights,
};
use rocmforge::gpu::{self, GpuBuffer, GpuDevice};
use rocmforge::loader::{GgmlType, GgufFile};
use rocmforge::tokenizer::BpeTokenizer;
use serial_test::serial;
use std::time::Instant;

const MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max)
}

fn upload_f32(data: &[f32]) -> GpuBuffer {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    let mut buffer = GpuBuffer::alloc(std::mem::size_of_val(data)).expect("alloc GPU buffer");
    buffer.copy_from_host(bytes).expect("upload GPU buffer");
    buffer
}

fn download_f32(buf: &GpuBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    buf.copy_to_host(&mut bytes).expect("download GPU buffer");
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() }
}

fn average_kernel_ms<F>(device: &GpuDevice, iters: usize, mut launch: F) -> f64
where
    F: FnMut(),
{
    let start = Instant::now();
    for _ in 0..iters {
        launch();
        device.synchronize().expect("stream synchronize");
    }
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn prepare_layer0_ffn_inputs(
    cpu_weights: &CpuModelWeights,
    config: &ModelConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let file = GgufFile::open(MODEL_PATH).expect("Failed to reopen GGUF file");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());
    let template =
        detect_chat_template(&config.architecture, file.tokenizer_data().model.as_deref());
    let prompt = template.apply("Hello");
    let prompt_tokens = tok.encode(&prompt, false);
    assert!(!prompt_tokens.is_empty(), "prompt should tokenize");

    let layer = cpu_weights.layer(0);
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;

    let mut hidden = vec![0.0f32; h];
    cpu_embed_token(prompt_tokens[0], cpu_weights, &mut hidden, config);

    let mut kv = CpuKvCache::new(config, 1);
    let mut scratch = CpuForwardScratch::new(config);

    rms_norm(
        &hidden,
        &layer.attn_norm,
        &mut scratch.normed,
        config.rms_norm_eps,
    );
    cpu_dispatch_gemv(
        &layer.attn_q,
        &layer.attn_q_meta,
        &scratch.normed,
        &mut scratch.q,
        q_size,
        h,
        Some(&mut scratch.q8_scratch),
    )
    .expect("CPU q projection");
    cpu_dispatch_gemv(
        &layer.attn_k,
        &layer.attn_k_meta,
        &scratch.normed,
        &mut scratch.k,
        kv_size,
        h,
        Some(&mut scratch.q8_scratch),
    )
    .expect("CPU k projection");
    cpu_dispatch_gemv(
        &layer.attn_v,
        &layer.attn_v_meta,
        &scratch.normed,
        &mut scratch.v,
        kv_size,
        h,
        Some(&mut scratch.q8_scratch),
    )
    .expect("CPU v projection");
    if let Some(bq) = &layer.attn_q_bias {
        add_bias(&mut scratch.q, bq);
    }
    if let Some(bk) = &layer.attn_k_bias {
        add_bias(&mut scratch.k, bk);
    }
    if let Some(bv) = &layer.attn_v_bias {
        add_bias(&mut scratch.v, bv);
    }
    rope(
        &mut scratch.q,
        config.num_heads,
        config.head_dim,
        0,
        config.rope_theta,
        config.rope_neox,
    );
    rope(
        &mut scratch.k,
        config.num_kv_heads,
        config.head_dim,
        0,
        config.rope_theta,
        config.rope_neox,
    );
    kv.write_k(0, 0, &scratch.k);
    kv.write_v(0, 0, &scratch.v);
    flash_attn_decode(
        &scratch.q,
        kv.k_buf(0),
        kv.v_buf(0),
        &mut scratch.attn_out,
        1,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
    );
    cpu_dispatch_gemv(
        &layer.attn_o,
        &layer.attn_o_meta,
        &scratch.attn_out,
        &mut scratch.layer_out,
        h,
        q_size,
        Some(&mut scratch.q8_scratch),
    )
    .expect("CPU attn_o projection");
    residual_add(&mut hidden, &scratch.layer_out);

    rms_norm(
        &hidden,
        &layer.ffn_norm,
        &mut scratch.normed,
        config.rms_norm_eps,
    );
    cpu_dispatch_gemv(
        &layer.ffn_gate,
        &layer.ffn_gate_meta,
        &scratch.normed,
        &mut scratch.gate,
        ff_size,
        h,
        Some(&mut scratch.q8_scratch),
    )
    .expect("CPU gate projection");
    cpu_dispatch_gemv(
        &layer.ffn_up,
        &layer.ffn_up_meta,
        &scratch.normed,
        &mut scratch.swiglu,
        ff_size,
        h,
        Some(&mut scratch.q8_scratch),
    )
    .expect("CPU up projection");

    let normed = scratch.normed.clone();
    let gate = scratch.gate.clone();
    let up = scratch.swiglu.clone();
    silu_fuse(&scratch.gate, &mut scratch.swiglu);

    let mut reference = vec![0.0f32; h];
    cpu_dispatch_gemv(
        &layer.ffn_down,
        &layer.ffn_down_meta,
        &scratch.swiglu,
        &mut reference,
        h,
        ff_size,
        Some(&mut scratch.q8_scratch),
    )
    .expect("CPU ffn_down projection");

    (normed, gate, up, scratch.swiglu, reference)
}

#[test]
#[serial]
fn test_gpu_experimental_gate_up_q4_0_layer0_real_model_matches_cpu() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    require_experimental_gpu_tests!();
    require_gpu!();
    require_vram!(4);

    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse model config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("CPU weights should load");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("GPU weights should load");
    let layer = gpu_weights.layer(0);

    assert_eq!(
        layer.ffn_gate_meta.wtype,
        GgmlType::Q4_0,
        "expected layer-0 ffn_gate to be Q4_0 for this experiment"
    );
    assert_eq!(
        layer.ffn_up_meta.wtype,
        GgmlType::Q4_0,
        "expected layer-0 ffn_up to be Q4_0 for this experiment"
    );

    let (normed, cpu_gate, cpu_up, _, _) = prepare_layer0_ffn_inputs(&cpu_weights, &config);
    let ff_size = config.intermediate_size;
    let hidden_size = config.hidden_size;

    let d_normed = upload_f32(&normed);
    let d_gate = GpuBuffer::alloc(ff_size * std::mem::size_of::<f32>()).expect("alloc gate");
    let d_up = GpuBuffer::alloc(ff_size * std::mem::size_of::<f32>()).expect("alloc up");

    gpu::kernels::quant::gemv_gate_up_q4_0_f32_on_stream(
        layer.ffn_gate.as_ptr(),
        layer.ffn_up.as_ptr(),
        d_normed.as_ptr() as *const f32,
        d_gate.as_ptr() as *mut f32,
        d_up.as_ptr() as *mut f32,
        hidden_size,
        ff_size,
        device.stream(),
    )
    .expect("raw gate/up kernel");
    device.synchronize().expect("gate/up synchronize");

    let gpu_gate = download_f32(&d_gate, ff_size);
    let gpu_up = download_f32(&d_up, ff_size);
    let gate_err = max_abs_error(&cpu_gate, &gpu_gate);
    let up_err = max_abs_error(&cpu_up, &gpu_up);

    assert!(
        gate_err <= 1e-3,
        "layer-0 raw gate mismatch: max_abs_error={}",
        gate_err
    );
    assert!(
        up_err <= 1e-3,
        "layer-0 raw up mismatch: max_abs_error={}",
        up_err
    );
}

#[test]
#[ignore = "experimental FFN microbenchmark against the real model"]
#[serial]
fn test_gpu_experimental_ffn_down_swiglu_q4_1_layer0_real_model() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    require_experimental_gpu_tests!();
    require_gpu!();
    require_vram!(4);

    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse model config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("CPU weights should load");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("GPU weights should load");
    let layer = gpu_weights.layer(0);

    assert_eq!(
        layer.ffn_down_meta.wtype,
        GgmlType::Q4_1,
        "expected layer-0 ffn_down to be Q4_1 for this experiment"
    );

    let (_, gate, up, swiglu, cpu_reference) = prepare_layer0_ffn_inputs(&cpu_weights, &config);
    let ff_size = config.intermediate_size;
    let hidden_size = config.hidden_size;

    let d_gate = upload_f32(&gate);
    let d_up = upload_f32(&up);
    let d_swiglu = upload_f32(&swiglu);
    let d_baseline =
        GpuBuffer::alloc(hidden_size * std::mem::size_of::<f32>()).expect("alloc baseline out");
    let d_experimental =
        GpuBuffer::alloc(hidden_size * std::mem::size_of::<f32>()).expect("alloc exp out");

    gpu::kernels::quant::gemv_q4_1_f32_on_stream(
        layer.ffn_down.as_ptr(),
        d_swiglu.as_ptr() as *const f32,
        d_baseline.as_ptr() as *mut f32,
        ff_size,
        hidden_size,
        device.stream(),
    )
    .expect("warmup baseline ffn_down");
    gpu::kernels::quant::gemv_ffn_down_swiglu_q4_1_f32_experimental_on_stream(
        layer.ffn_down.as_ptr(),
        d_gate.as_ptr() as *const f32,
        d_up.as_ptr() as *const f32,
        d_experimental.as_ptr() as *mut f32,
        ff_size,
        hidden_size,
        device.stream(),
    )
    .expect("warmup experimental ffn");
    device.synchronize().expect("warmup synchronize");

    let baseline_ms = average_kernel_ms(&device, 200, || {
        gpu::kernels::quant::gemv_q4_1_f32_on_stream(
            layer.ffn_down.as_ptr(),
            d_swiglu.as_ptr() as *const f32,
            d_baseline.as_ptr() as *mut f32,
            ff_size,
            hidden_size,
            device.stream(),
        )
        .expect("baseline ffn_down");
    });

    let experimental_ms = average_kernel_ms(&device, 200, || {
        gpu::kernels::quant::gemv_ffn_down_swiglu_q4_1_f32_experimental_on_stream(
            layer.ffn_down.as_ptr(),
            d_gate.as_ptr() as *const f32,
            d_up.as_ptr() as *const f32,
            d_experimental.as_ptr() as *mut f32,
            ff_size,
            hidden_size,
            device.stream(),
        )
        .expect("experimental ffn_down");
    });

    let baseline = download_f32(&d_baseline, hidden_size);
    let experimental = download_f32(&d_experimental, hidden_size);
    let baseline_err = max_abs_error(&cpu_reference, &baseline);
    let experimental_err = max_abs_error(&cpu_reference, &experimental);
    let cross_err = max_abs_error(&baseline, &experimental);

    eprintln!(
        "experimental_ffn_layer0_q4_1 baseline_ms={:.4} experimental_ms={:.4} speedup={:.3} baseline_err={:.6} experimental_err={:.6} cross_err={:.6}",
        baseline_ms,
        experimental_ms,
        baseline_ms / experimental_ms,
        baseline_err,
        experimental_err,
        cross_err
    );

    assert!(
        baseline_err <= 1e-3,
        "baseline layer-0 Q4_1 FFN-down mismatch: max_abs_error={}",
        baseline_err
    );
    assert!(
        experimental_err <= 1e-3,
        "experimental layer-0 Q4_1 FFN-down mismatch: max_abs_error={}",
        experimental_err
    );
    assert!(
        cross_err <= 1e-4,
        "baseline and experimental FFN paths diverged: max_abs_error={}",
        cross_err
    );
}

#[test]
#[serial]
fn test_gpu_experimental_full_ffn_block_q4_1_layer0_real_model_matches_cpu() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    require_experimental_gpu_tests!();
    require_gpu!();
    require_vram!(4);

    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse model config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("CPU weights should load");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("GPU weights should load");
    let layer = gpu_weights.layer(0);

    assert_eq!(
        layer.ffn_gate_meta.wtype,
        GgmlType::Q4_0,
        "expected layer-0 ffn_gate to be Q4_0 for this experiment"
    );
    assert_eq!(
        layer.ffn_up_meta.wtype,
        GgmlType::Q4_0,
        "expected layer-0 ffn_up to be Q4_0 for this experiment"
    );
    assert_eq!(
        layer.ffn_down_meta.wtype,
        GgmlType::Q4_1,
        "expected layer-0 ffn_down to be Q4_1 for this experiment"
    );

    let (normed, _, _, _, cpu_reference) = prepare_layer0_ffn_inputs(&cpu_weights, &config);
    let ff_size = config.intermediate_size;
    let hidden_size = config.hidden_size;

    let d_normed = upload_f32(&normed);
    let d_gate = GpuBuffer::alloc(ff_size * std::mem::size_of::<f32>()).expect("alloc gate");
    let d_up = GpuBuffer::alloc(ff_size * std::mem::size_of::<f32>()).expect("alloc up");
    let d_output =
        GpuBuffer::alloc(hidden_size * std::mem::size_of::<f32>()).expect("alloc output");

    gpu::kernels::quant::gemv_gate_up_q4_0_f32_on_stream(
        layer.ffn_gate.as_ptr(),
        layer.ffn_up.as_ptr(),
        d_normed.as_ptr() as *const f32,
        d_gate.as_ptr() as *mut f32,
        d_up.as_ptr() as *mut f32,
        hidden_size,
        ff_size,
        device.stream(),
    )
    .expect("raw gate/up kernel");
    gpu::kernels::quant::gemv_ffn_down_swiglu_q4_1_f32_experimental_on_stream(
        layer.ffn_down.as_ptr(),
        d_gate.as_ptr() as *const f32,
        d_up.as_ptr() as *const f32,
        d_output.as_ptr() as *mut f32,
        ff_size,
        hidden_size,
        device.stream(),
    )
    .expect("experimental full FFN block");
    device.synchronize().expect("full FFN synchronize");

    let experimental = download_f32(&d_output, hidden_size);
    let experimental_err = max_abs_error(&cpu_reference, &experimental);

    assert!(
        experimental_err <= 1e-3,
        "experimental full layer-0 FFN mismatch: max_abs_error={}",
        experimental_err
    );
}

#[test]
#[serial]
fn test_gpu_experimental_full_ffn_block_prompt_tail_matches_cpu_across_eligible_layers() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    require_experimental_gpu_tests!();
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

    let target_pos = prompt_tokens.len() - 1;
    let ff_size = config.intermediate_size;
    let hidden_size = config.hidden_size;

    let mut cpu_kv = CpuKvCache::new(&config, prompt_tokens.len());
    let mut cpu_scratch = CpuForwardScratch::new(&config);
    let mut hidden = vec![0.0f32; hidden_size];

    for (pos, &token_id) in prompt_tokens[..target_pos].iter().enumerate() {
        cpu_embed_token(token_id, &cpu_weights, &mut hidden, &config);
        cpu_full_forward(
            &mut hidden,
            &cpu_weights,
            &mut cpu_kv,
            &mut cpu_scratch,
            pos,
            &config,
        )
        .expect("CPU prefill should succeed");
    }

    cpu_embed_token(
        prompt_tokens[target_pos],
        &cpu_weights,
        &mut hidden,
        &config,
    );

    let mut checked_layers = 0usize;
    for layer_idx in 0..config.num_layers {
        cpu_layer_forward(
            &mut hidden,
            cpu_weights.layer(layer_idx),
            &mut cpu_kv,
            &mut cpu_scratch,
            layer_idx,
            target_pos,
            &config,
            false,
        )
        .expect("CPU layer forward should succeed");

        let gpu_layer = gpu_weights.layer(layer_idx);
        if gpu_layer.ffn_gate_meta.wtype != GgmlType::Q4_0
            || gpu_layer.ffn_up_meta.wtype != GgmlType::Q4_0
            || gpu_layer.ffn_down_meta.wtype != GgmlType::Q4_1
        {
            continue;
        }

        checked_layers += 1;
        let d_normed = upload_f32(&cpu_scratch.normed);
        let d_gate = GpuBuffer::alloc(ff_size * std::mem::size_of::<f32>()).expect("alloc gate");
        let d_up = GpuBuffer::alloc(ff_size * std::mem::size_of::<f32>()).expect("alloc up");
        let d_output =
            GpuBuffer::alloc(hidden_size * std::mem::size_of::<f32>()).expect("alloc output");

        gpu::kernels::quant::gemv_gate_up_q4_0_f32_on_stream(
            gpu_layer.ffn_gate.as_ptr(),
            gpu_layer.ffn_up.as_ptr(),
            d_normed.as_ptr() as *const f32,
            d_gate.as_ptr() as *mut f32,
            d_up.as_ptr() as *mut f32,
            hidden_size,
            ff_size,
            device.stream(),
        )
        .expect("raw gate/up kernel");
        gpu::kernels::quant::gemv_ffn_down_swiglu_q4_1_f32_experimental_on_stream(
            gpu_layer.ffn_down.as_ptr(),
            d_gate.as_ptr() as *const f32,
            d_up.as_ptr() as *const f32,
            d_output.as_ptr() as *mut f32,
            ff_size,
            hidden_size,
            device.stream(),
        )
        .expect("experimental full FFN block");
        device.synchronize().expect("layer synchronize");

        let actual = download_f32(&d_output, hidden_size);
        let err = max_abs_error(&cpu_scratch.layer_out, &actual);
        assert!(
            err <= 1e-3,
            "experimental FFN mismatch at layer {} pos {}: max_abs_error={}",
            layer_idx,
            target_pos,
            err
        );
    }

    assert!(
        checked_layers > 0,
        "expected at least one eligible Q4_0/Q4_0/Q4_1 FFN layer"
    );
}
