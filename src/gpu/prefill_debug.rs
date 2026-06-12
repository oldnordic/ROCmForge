//! CPU reference activations for prefill layer 0 debug validation.

use crate::config::ModelConfig;
use crate::cpu::ops::dispatch_gemv as cpu_dispatch_gemv;
use crate::cpu::weights::CpuModelWeights;

pub(crate) struct CpuLayer0Activations {
    pub hidden_in: Vec<f32>,
    pub normed_attn: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub q_rope: Vec<f32>,
    pub k_rope: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub layer_out_attn: Vec<f32>,
    pub hidden_after_attn: Vec<f32>,
    pub normed_ffn: Vec<f32>,
    pub gate: Vec<f32>,
    pub swiglu: Vec<f32>,
    pub layer_out_ffn: Vec<f32>,
    pub hidden_out: Vec<f32>,
}

pub(crate) fn download_gpu_buffer(buf: &crate::gpu::GpuBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    buf.copy_to_host(&mut bytes).expect("copy_to_host failed");
    let mut out = vec![0.0f32; len];
    for i in 0..len {
        out[i] = f32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
    }
    out
}

pub(crate) fn max_abs_error_slice(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

pub(crate) fn compute_layer0_cpu_reference(
    token_ids: &[u32],
    cpu_weights: &CpuModelWeights,
    config: &ModelConfig,
) -> CpuLayer0Activations {
    use crate::cpu::forward::cpu_embed_token;

    let seq_len = token_ids.len();
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;

    println!("DEBUG PREFILL: Computing CPU reference activations for layer 0...");
    let mut cpu_kv = crate::cpu::cache::CpuKvCache::new(config, seq_len.max(1));
    let mut cpu_scratch = crate::cpu::cache::CpuForwardScratch::new(config);
    let mut cpu_hidden = vec![0.0f32; h];

    let mut hidden_in = vec![0.0f32; seq_len * h];
    let mut normed_attn = vec![0.0f32; seq_len * h];
    let mut q = vec![0.0f32; seq_len * q_size];
    let mut k = vec![0.0f32; seq_len * kv_size];
    let mut v = vec![0.0f32; seq_len * kv_size];
    let mut q_rope = vec![0.0f32; seq_len * q_size];
    let mut k_rope = vec![0.0f32; seq_len * kv_size];
    let mut attn_out = vec![0.0f32; seq_len * q_size];
    let mut layer_out_attn = vec![0.0f32; seq_len * h];
    let mut hidden_after_attn = vec![0.0f32; seq_len * h];
    let mut normed_ffn = vec![0.0f32; seq_len * h];
    let mut gate = vec![0.0f32; seq_len * ff_size];
    let mut swiglu = vec![0.0f32; seq_len * ff_size];
    let mut layer_out_ffn = vec![0.0f32; seq_len * h];
    let mut hidden_out = vec![0.0f32; seq_len * h];

    for (pos, &token_id) in token_ids.iter().enumerate() {
        cpu_embed_token(token_id, cpu_weights, &mut cpu_hidden, config);
        hidden_in[pos * h..(pos + 1) * h].copy_from_slice(&cpu_hidden);

        let layer_idx = 0;
        let layer_weights = cpu_weights.layer(layer_idx);

        // 1. Attn Norm
        let mut t_normed_attn = vec![0.0f32; h];
        crate::cpu::ops::rms_norm(
            &cpu_hidden,
            &layer_weights.attn_norm,
            &mut t_normed_attn,
            config.rms_norm_eps,
        );
        normed_attn[pos * h..(pos + 1) * h].copy_from_slice(&t_normed_attn);

        // 2. QKV projection
        let mut t_q = vec![0.0f32; q_size];
        let mut t_k = vec![0.0f32; kv_size];
        let mut t_v = vec![0.0f32; kv_size];
        let mut q8_scratch = vec![0u8; cpu_scratch.q8_scratch.len()];
        cpu_dispatch_gemv(
            &layer_weights.attn_q,
            &layer_weights.attn_q_meta,
            &t_normed_attn,
            &mut t_q,
            q_size,
            h,
            Some(&mut q8_scratch),
        )
        .expect("M-ALLOW: cpu_dispatch_gemv infallible for valid weight dims in validation path");
        cpu_dispatch_gemv(
            &layer_weights.attn_k,
            &layer_weights.attn_k_meta,
            &t_normed_attn,
            &mut t_k,
            kv_size,
            h,
            Some(&mut q8_scratch),
        )
        .expect("M-ALLOW: cpu_dispatch_gemv infallible for valid weight dims in validation path");
        cpu_dispatch_gemv(
            &layer_weights.attn_v,
            &layer_weights.attn_v_meta,
            &t_normed_attn,
            &mut t_v,
            kv_size,
            h,
            Some(&mut q8_scratch),
        )
        .expect("M-ALLOW: cpu_dispatch_gemv infallible for valid weight dims in validation path");

        if let Some(bq) = &layer_weights.attn_q_bias {
            crate::cpu::ops::add_bias(&mut t_q, bq);
        }
        if let Some(bk) = &layer_weights.attn_k_bias {
            crate::cpu::ops::add_bias(&mut t_k, bk);
        }
        if let Some(bv) = &layer_weights.attn_v_bias {
            crate::cpu::ops::add_bias(&mut t_v, bv);
        }
        q[pos * q_size..(pos + 1) * q_size].copy_from_slice(&t_q);
        k[pos * kv_size..(pos + 1) * kv_size].copy_from_slice(&t_k);
        v[pos * kv_size..(pos + 1) * kv_size].copy_from_slice(&t_v);

        // 3. RoPE
        let half = config.head_dim / 2;
        let freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / config.rope_theta.powf((2 * i) as f32 / config.head_dim as f32))
            .collect();
        let mut t_q_rope = t_q.clone();
        let mut t_k_rope = t_k.clone();
        crate::cpu::ops::rope_with_pos(
            &mut t_q_rope,
            config.num_heads,
            config.head_dim,
            pos,
            &freq,
            config.rope_neox,
        );
        crate::cpu::ops::rope_with_pos(
            &mut t_k_rope,
            config.num_kv_heads,
            config.head_dim,
            pos,
            &freq,
            config.rope_neox,
        );
        q_rope[pos * q_size..(pos + 1) * q_size].copy_from_slice(&t_q_rope);
        k_rope[pos * kv_size..(pos + 1) * kv_size].copy_from_slice(&t_k_rope);

        // Write to cache
        cpu_kv.write_k(layer_idx, pos, &t_k_rope);
        cpu_kv.write_v(layer_idx, pos, &t_v);

        // 4. Attention
        let mut t_attn_out = vec![0.0f32; q_size];
        crate::cpu::ops::flash_attn_decode(
            &t_q_rope,
            cpu_kv.k_buf(layer_idx),
            cpu_kv.v_buf(layer_idx),
            &mut t_attn_out,
            pos + 1,
            config.num_heads,
            config.num_kv_heads,
            config.head_dim,
        );
        attn_out[pos * q_size..(pos + 1) * q_size].copy_from_slice(&t_attn_out);

        // 5. Attn Out Projection
        let mut t_layer_out_attn = vec![0.0f32; h];
        cpu_dispatch_gemv(
            &layer_weights.attn_o,
            &layer_weights.attn_o_meta,
            &t_attn_out,
            &mut t_layer_out_attn,
            h,
            q_size,
            Some(&mut q8_scratch),
        )
        .expect("M-ALLOW: cpu_dispatch_gemv infallible for valid weight dims in validation path");
        layer_out_attn[pos * h..(pos + 1) * h].copy_from_slice(&t_layer_out_attn);

        // 6. Attn Residual
        let mut t_hidden_after_attn = cpu_hidden.clone();
        crate::cpu::ops::residual_add(&mut t_hidden_after_attn, &t_layer_out_attn);
        hidden_after_attn[pos * h..(pos + 1) * h].copy_from_slice(&t_hidden_after_attn);

        // 7. FFN Norm
        let mut t_normed_ffn = vec![0.0f32; h];
        crate::cpu::ops::rms_norm(
            &t_hidden_after_attn,
            &layer_weights.ffn_norm,
            &mut t_normed_ffn,
            config.rms_norm_eps,
        );
        normed_ffn[pos * h..(pos + 1) * h].copy_from_slice(&t_normed_ffn);

        // 8. FFN Gate + Up
        let mut t_gate = vec![0.0f32; ff_size];
        let mut t_swiglu = vec![0.0f32; ff_size];
        cpu_dispatch_gemv(
            &layer_weights.ffn_gate,
            &layer_weights.ffn_gate_meta,
            &t_normed_ffn,
            &mut t_gate,
            ff_size,
            h,
            Some(&mut q8_scratch),
        )
        .expect("M-ALLOW: cpu_dispatch_gemv infallible for valid weight dims in validation path");
        cpu_dispatch_gemv(
            &layer_weights.ffn_up,
            &layer_weights.ffn_up_meta,
            &t_normed_ffn,
            &mut t_swiglu,
            ff_size,
            h,
            Some(&mut q8_scratch),
        )
        .expect("M-ALLOW: cpu_dispatch_gemv infallible for valid weight dims in validation path");
        gate[pos * ff_size..(pos + 1) * ff_size].copy_from_slice(&t_gate);

        crate::cpu::ops::silu_fuse(&t_gate, &mut t_swiglu);
        swiglu[pos * ff_size..(pos + 1) * ff_size].copy_from_slice(&t_swiglu);

        // 9. FFN Down Projection
        let mut t_layer_out_ffn = vec![0.0f32; h];
        cpu_dispatch_gemv(
            &layer_weights.ffn_down,
            &layer_weights.ffn_down_meta,
            &t_swiglu,
            &mut t_layer_out_ffn,
            h,
            ff_size,
            Some(&mut q8_scratch),
        )
        .expect("M-ALLOW: cpu_dispatch_gemv infallible for valid weight dims in validation path");
        layer_out_ffn[pos * h..(pos + 1) * h].copy_from_slice(&t_layer_out_ffn);

        // 10. FFN Residual
        crate::cpu::ops::residual_add(&mut cpu_hidden, &t_layer_out_ffn);
        hidden_out[pos * h..(pos + 1) * h].copy_from_slice(&cpu_hidden);
    }

    CpuLayer0Activations {
        hidden_in,
        normed_attn,
        q,
        k,
        v,
        q_rope,
        k_rope,
        attn_out,
        layer_out_attn,
        hidden_after_attn,
        normed_ffn,
        gate,
        swiglu,
        layer_out_ffn,
        hidden_out,
    }
}
