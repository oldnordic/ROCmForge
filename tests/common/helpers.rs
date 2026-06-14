use half::f16;
use rocmforge::config::{
    AttentionLayout, FfnLayout, ModelConfig, TensorNameRegistry, TensorNamingScheme, TensorRole,
};
use rocmforge::gpu::error::GpuResult;
use rocmforge::gpu::weights::{GpuLayerType, GpuShortconvWeights};
use rocmforge::gpu::{GpuBuffer, GpuLayerWeights, WeightMeta};
use rocmforge::loader::GgmlType;

pub fn mock_model_config() -> ModelConfig {
    ModelConfig {
        num_layers: 2,
        num_kv_heads: 4,
        head_dim: 64,
        max_seq_len: 128,
        hidden_size: 256,
        num_heads: 4,
        intermediate_size: 768,
        vocab_size: 1000,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_neox: false,
        use_attention_bias: false,
        attention_layout: AttentionLayout::SplitQkv,
        ffn_layout: FfnLayout::SwiGLU,
        architecture: "test".to_string(),
        tensor_registry: TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf),
        shortconv_l_cache: Some(3),
        num_dense_layers: None,
        num_experts_per_tok: None,
        use_expert_bias: false,
        expert_weights_scale: 1.0,
        rope_freq: Vec::new(),
        kv_lora_dim: None,
        kv_frame_codec_enabled: None,
        adastate_anchors_enabled: None,
        kv_quant_bits: None,
        turboquant_centroids: None,
        qjl_scale: None,
    }
}

pub fn mock_gpu_meta(rows: usize, cols: usize, wtype: GgmlType, role: TensorRole) -> WeightMeta {
    WeightMeta {
        wtype,
        dims: vec![rows as u64, cols as u64],
        needs_transpose: false,
        role,
        svd_k: None,
    }
}

pub fn mock_cpu_meta(
    rows: usize,
    cols: usize,
    wtype: GgmlType,
    role: TensorRole,
) -> rocmforge::cpu::weights::WeightMeta {
    rocmforge::cpu::weights::WeightMeta {
        wtype,
        dims: vec![rows as u64, cols as u64],
        needs_transpose: false,
        role,
        svd_k: None,
    }
}

pub fn upload_f32(device_id: i32, data: &[f32]) -> GpuBuffer {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();
    let mut buf =
        GpuBuffer::alloc_for_device(bytes.len(), device_id).expect("helper: alloc_for_device");
    buf.copy_from_host(&bytes).expect("helper: copy_from_host");
    buf
}

pub fn download_f32(buf: &GpuBuffer, count: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; count * 4];
    buf.copy_to_host(&mut bytes).expect("helper: copy_to_host");
    let mut out = vec![0.0f32; count];
    for i in 0..count {
        out[i] = f32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
    }
    out
}

pub fn upload_raw(device_id: i32, bytes: &[u8]) -> GpuBuffer {
    let mut buf =
        GpuBuffer::alloc_for_device(bytes.len(), device_id).expect("helper: upload_raw alloc");
    buf.copy_from_host(bytes).expect("helper: upload_raw copy");
    buf
}

pub fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    let n = data.len();
    assert!(n % 32 == 0);
    let num_blocks = n / 32;
    let mut out = vec![0u8; num_blocks * 18];
    for b in 0..num_blocks {
        let block_data = &data[b * 32..(b + 1) * 32];
        let mut amax = 0.0f32;
        for &x in block_data {
            amax = amax.max(x.abs());
        }
        let d = amax / 7.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };

        // Scale as f16
        let d_f16 = f16::from_f32(d);
        let d_bytes = d_f16.to_le_bytes();
        out[b * 18] = d_bytes[0];
        out[b * 18 + 1] = d_bytes[1];

        for i in 0..16 {
            let x0 = block_data[i];
            let x1 = block_data[i + 16];
            let q0 = ((x0 * id + 8.5) as i32).clamp(0, 15) as u8;
            let q1 = ((x1 * id + 8.5) as i32).clamp(0, 15) as u8;
            out[b * 18 + 2 + i] = q0 | (q1 << 4);
        }
    }
    out
}

pub fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f32::max)
}
