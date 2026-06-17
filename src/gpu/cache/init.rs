use super::{
    binding::compute_kv_binding_tag, BlockAllocator, BlockTable, GpuBuffer, GpuError, GpuKvCache,
    GpuResult, TURBOQUANT_POS_ALIGN, TURBOQUANT_POS_ALIGN_MASK, TURBOQUANT_RMS_SCALE_BYTES,
};
use crate::config::{FfnLayout, ModelConfig};

const PAGED_BLOCK_SIZE_TOKENS: usize = 16;

pub(super) struct CacheLayout {
    pub kv_size: usize,
    pub layer_bytes: usize,
    pub total_cache_bytes: usize,
}

pub(super) struct HybridState {
    pub ssm_state: Option<Vec<GpuBuffer>>,
    pub ssm_conv_state: Option<Vec<GpuBuffer>>,
}

pub(super) struct ProjectionWeights {
    pub w_down_k: Option<Vec<GpuBuffer>>,
    pub w_down_v: Option<Vec<GpuBuffer>>,
    pub w_up_k: Option<Vec<GpuBuffer>>,
    pub w_up_v: Option<Vec<GpuBuffer>>,
}

pub(super) struct CacheStorage {
    pub k: Vec<GpuBuffer>,
    pub v: Vec<GpuBuffer>,
    pub decode_binding_tag: u64,
}

fn identity_projection_rows(rows: usize, cols: usize, row_major_stride: usize) -> Vec<f32> {
    let mut host = vec![0.0f32; rows * cols];
    for j in 0..rows {
        if j < row_major_stride {
            host[j * cols + j] = 1.0f32;
        }
    }
    host
}

fn f32s_as_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

fn layer_allocation_error(kind: &str, layer: usize, err: impl std::fmt::Display) -> GpuError {
    GpuError::CacheAllocationFailed {
        reason: format!("{kind} layer {layer} allocation failed: {err}"),
    }
}

pub(super) fn compute_layout(config: &ModelConfig, max_seq_len: usize) -> CacheLayout {
    let kv_size = config.num_kv_heads * config.head_dim;
    let effective_kv = config.kv_lora_dim.unwrap_or(kv_size);
    let layer_bytes = if let Some(bits) = config.kv_quant_bits {
        let pack_bytes = (effective_kv * bits).div_ceil(8);
        let qjl_bytes = effective_kv.div_ceil(8);
        // V cache stores RMS scales (not QJL signs) at pos_v_base + pack_bytes,
        // so the per-position stride must accommodate the larger of:
        //   K: pack_bytes + qjl_bytes  (indices + signs)
        //   V: pack_bytes + TURBOQUANT_RMS_SCALE_BYTES  (indices + scales)
        let content_bytes = pack_bytes + qjl_bytes.max(TURBOQUANT_RMS_SCALE_BYTES);
        let aligned_pos_bytes =
            (content_bytes + TURBOQUANT_POS_ALIGN_MASK) & !TURBOQUANT_POS_ALIGN_MASK;
        max_seq_len * aligned_pos_bytes
    } else {
        max_seq_len * effective_kv * std::mem::size_of::<f32>()
    };

    let mut total_cache_bytes = 2 * config.num_layers * layer_bytes;
    if let Some(dc) = config.kv_lora_dim {
        total_cache_bytes += 4 * config.num_layers * (dc * kv_size * std::mem::size_of::<f32>());
    }
    if let Some(ref centroids) = config.turboquant_centroids {
        total_cache_bytes += centroids.len() * std::mem::size_of::<f32>();
    }

    CacheLayout {
        kv_size,
        layer_bytes,
        total_cache_bytes,
    }
}

pub(super) fn enforce_vram_budget(
    config: &ModelConfig,
    layer_bytes: usize,
    total_cache_bytes: usize,
) -> GpuResult<()> {
    let budget = super::super::vram_budget::query_vram_budget(
        super::super::vram_budget::active_or_default_device_id(),
    )?;
    if total_cache_bytes > budget.safe_allocation_size {
        return Err(GpuError::CacheAllocationFailed {
            reason: format!(
                "KV cache requires {} MB but only {} MB safely allocatable ({} MB free, 2 * {} layers * {} MB/layer)",
                total_cache_bytes / (1024 * 1024),
                budget.safe_allocation_size / (1024 * 1024),
                budget.free_vram / (1024 * 1024),
                config.num_layers,
                layer_bytes / (1024 * 1024),
            ),
        });
    }
    Ok(())
}

pub(super) fn allocate_cache_storage(
    config: &ModelConfig,
    layer_bytes: usize,
) -> GpuResult<CacheStorage> {
    let mut k = Vec::with_capacity(config.num_layers);
    for layer in 0..config.num_layers {
        let buf = GpuBuffer::alloc(layer_bytes)
            .map_err(|e| layer_allocation_error("K cache", layer, e))?;
        k.push(buf);
    }

    let mut v = Vec::with_capacity(config.num_layers);
    for layer in 0..config.num_layers {
        let buf = GpuBuffer::alloc(layer_bytes)
            .map_err(|e| layer_allocation_error("V cache", layer, e))?;
        v.push(buf);
    }

    let decode_binding_tag = compute_kv_binding_tag(&k, &v);
    Ok(CacheStorage {
        k,
        v,
        decode_binding_tag,
    })
}

pub(super) fn allocate_hybrid_state(config: &ModelConfig) -> GpuResult<HybridState> {
    if !config.architecture.contains("qwen35") {
        return Ok(HybridState {
            ssm_state: None,
            ssm_conv_state: None,
        });
    }

    let mut states = Vec::with_capacity(config.num_layers);
    let mut conv_states = Vec::with_capacity(config.num_layers);

    let ssm_heads = std::cmp::max(config.num_heads * 2, 32);
    let ssm_state_bytes = ssm_heads * 128 * 128 * std::mem::size_of::<f32>();
    let qkv_dim = std::cmp::max(config.num_kv_heads * 128 * 2 + config.num_heads * 128, 8192);
    let ssm_conv_bytes = qkv_dim * 3 * std::mem::size_of::<f32>();

    for layer in 0..config.num_layers {
        let s_buf = GpuBuffer::alloc(ssm_state_bytes)
            .map_err(|e| layer_allocation_error("SSM state", layer, e))?;
        super::super::ffi::hip_memset(s_buf.as_ptr(), 0, ssm_state_bytes)?;
        states.push(s_buf);

        let c_buf = GpuBuffer::alloc(ssm_conv_bytes)
            .map_err(|e| layer_allocation_error("SSM conv state", layer, e))?;
        super::super::ffi::hip_memset(c_buf.as_ptr(), 0, ssm_conv_bytes)?;
        conv_states.push(c_buf);
    }

    Ok(HybridState {
        ssm_state: Some(states),
        ssm_conv_state: Some(conv_states),
    })
}

pub(super) fn allocate_projection_weights(
    config: &ModelConfig,
    kv_size: usize,
) -> GpuResult<ProjectionWeights> {
    let Some(dc) = config.kv_lora_dim else {
        return Ok(ProjectionWeights {
            w_down_k: None,
            w_down_v: None,
            w_up_k: None,
            w_up_v: None,
        });
    };

    let mut down_k = Vec::with_capacity(config.num_layers);
    let mut down_v = Vec::with_capacity(config.num_layers);
    let mut up_k = Vec::with_capacity(config.num_layers);
    let mut up_v = Vec::with_capacity(config.num_layers);

    let proj_bytes = dc * kv_size * std::mem::size_of::<f32>();
    let host_down = identity_projection_rows(dc, kv_size, kv_size);
    let host_up = identity_projection_rows(kv_size, dc, dc);
    let down_bytes = f32s_as_bytes(&host_down);
    let up_bytes = f32s_as_bytes(&host_up);

    for layer in 0..config.num_layers {
        let mut d_k = GpuBuffer::alloc(proj_bytes)
            .map_err(|e| layer_allocation_error("W_down_k", layer, e))?;
        d_k.copy_from_host(down_bytes)?;
        down_k.push(d_k);

        let mut d_v = GpuBuffer::alloc(proj_bytes)
            .map_err(|e| layer_allocation_error("W_down_v", layer, e))?;
        d_v.copy_from_host(down_bytes)?;
        down_v.push(d_v);

        let mut u_k =
            GpuBuffer::alloc(proj_bytes).map_err(|e| layer_allocation_error("W_up_k", layer, e))?;
        u_k.copy_from_host(up_bytes)?;
        up_k.push(u_k);

        let mut u_v =
            GpuBuffer::alloc(proj_bytes).map_err(|e| layer_allocation_error("W_up_v", layer, e))?;
        u_v.copy_from_host(up_bytes)?;
        up_v.push(u_v);
    }

    Ok(ProjectionWeights {
        w_down_k: Some(down_k),
        w_down_v: Some(down_v),
        w_up_k: Some(up_k),
        w_up_v: Some(up_v),
    })
}

pub(super) fn upload_centroids(config: &ModelConfig) -> GpuResult<Option<GpuBuffer>> {
    let Some(ref host_centroids) = config.turboquant_centroids else {
        return Ok(None);
    };

    let mut buf =
        GpuBuffer::alloc(host_centroids.len() * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("TurboQuant centroids allocation failed: {e}"),
            }
        })?;
    buf.copy_from_host(f32s_as_bytes(host_centroids))?;
    Ok(Some(buf))
}

pub(super) fn init_paged_state(
    num_layers: usize,
    layer_bytes: usize,
    max_seq_len: usize,
) -> (
    usize,
    usize,
    BlockAllocator,
    BlockTable,
    Vec<Vec<Option<std::sync::Arc<GpuBuffer>>>>,
    Vec<Vec<Option<std::sync::Arc<GpuBuffer>>>>,
) {
    let pos_bytes = layer_bytes / max_seq_len;
    let block_size_tokens = PAGED_BLOCK_SIZE_TOKENS;
    let block_allocator = BlockAllocator::new(block_size_tokens);
    let block_table = BlockTable {
        block_ids: Vec::new(),
    };
    let paged_k = vec![Vec::new(); num_layers];
    let paged_v = vec![Vec::new(); num_layers];
    (
        pos_bytes,
        block_size_tokens,
        block_allocator,
        block_table,
        paged_k,
        paged_v,
    )
}

impl GpuKvCache {
    pub(super) fn build_from_config(config: &ModelConfig, max_seq_len: usize) -> GpuResult<Self> {
        if let Some(bits) = config.kv_quant_bits {
            if !(1..=4).contains(&bits) {
                return Err(GpuError::UnsupportedOperation {
                    operation: "TurboQuant cache init".to_string(),
                    reason: format!("kv_quant_bits must be in {{1,2,3,4}}, got {}", bits),
                });
            }
        }
        let layout = compute_layout(config, max_seq_len);
        enforce_vram_budget(config, layout.layer_bytes, layout.total_cache_bytes)?;

        let storage = allocate_cache_storage(config, layout.layer_bytes)?;
        let hybrid = allocate_hybrid_state(config)?;
        let projections = allocate_projection_weights(config, layout.kv_size)?;
        let centroids = upload_centroids(config)?;
        let (pos_bytes, block_size_tokens, block_allocator, block_table, paged_k, paged_v) =
            init_paged_state(config.num_layers, layout.layer_bytes, max_seq_len);

        let conv_state = if config.shortconv_l_cache.is_some() || config.architecture == "lfm2moe" {
            let l_cache = config.shortconv_l_cache.unwrap_or(3);
            let conv_bytes = l_cache * config.hidden_size * std::mem::size_of::<f32>();
            let mut states = Vec::with_capacity(config.num_layers);
            for layer in 0..config.num_layers {
                let buf = GpuBuffer::alloc(conv_bytes)
                    .map_err(|e| layer_allocation_error("shortconv conv state", layer, e))?;
                super::super::ffi::hip_memset(buf.as_ptr(), 0, conv_bytes)?;
                states.push(buf);
            }
            Some(states)
        } else {
            None
        };

        Ok(Self {
            k: storage.k,
            v: storage.v,
            ssm_state: hybrid.ssm_state,
            ssm_conv_state: hybrid.ssm_conv_state,
            conv_state,
            max_seq_len,
            kv_size: layout.kv_size,
            num_layers: config.num_layers,
            decode_binding_tag: storage.decode_binding_tag,
            kv_lora_dim: config.kv_lora_dim,
            adastate_anchors_enabled: config.adastate_anchors_enabled.unwrap_or(false),
            kv_frame_codec_enabled: config.kv_frame_codec_enabled.unwrap_or(false),
            w_down_k: projections.w_down_k,
            w_down_v: projections.w_down_v,
            w_up_k: projections.w_up_k,
            w_up_v: projections.w_up_v,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            kv_quant_bits: config.kv_quant_bits,
            centroids,
            qjl_scale: config.qjl_scale.unwrap_or(0.0f32),
            block_size_tokens,
            pos_bytes,
            block_allocator,
            block_table,
            paged_k,
            paged_v,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{compute_layout, identity_projection_rows, FfnLayout, PAGED_BLOCK_SIZE_TOKENS};

    fn make_test_config() -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 128,
            max_seq_len: 512,
            hidden_size: 1024,
            num_heads: 8,
            intermediate_size: 2048,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_freq: (0..64)
                .map(|i| 1.0 / 10000.0f32.powf((2 * i) as f32 / 128.0f32))
                .collect(),
            rope_neox: false,
            use_attention_bias: false,
            attention_layout: crate::config::AttentionLayout::SplitQkv,
            ffn_layout: FfnLayout::SwiGLU,
            architecture: "test".to_string(),
            tensor_registry: crate::config::TensorNameRegistry::from_scheme(
                &crate::config::TensorNamingScheme::Gguf,
            ),
            shortconv_l_cache: None,
            num_dense_layers: None,
            num_experts_per_tok: None,
            use_expert_bias: false,
            expert_weights_scale: 1.0,
            kv_lora_dim: None,
            kv_frame_codec_enabled: None,
            adastate_anchors_enabled: None,
            kv_quant_bits: None,
            turboquant_centroids: None,
            qjl_scale: None,
            ..Default::default()
        }
    }

    #[test]
    fn identity_projection_rows_writes_diagonal_only() {
        let rows = identity_projection_rows(3, 5, 5);
        assert_eq!(rows[0], 1.0);
        assert_eq!(rows[6], 1.0);
        assert_eq!(rows[12], 1.0);
        assert_eq!(rows.iter().filter(|&&v| v == 1.0).count(), 3);
    }

    #[test]
    fn compute_layout_matches_unquantized_expectation() {
        let config = make_test_config();
        let layout = compute_layout(&config, 256);
        assert_eq!(layout.kv_size, 512);
        assert_eq!(layout.layer_bytes, 256 * 512 * std::mem::size_of::<f32>());
        assert_eq!(
            layout.total_cache_bytes,
            2 * config.num_layers * layout.layer_bytes
        );
        assert_eq!(PAGED_BLOCK_SIZE_TOKENS, 16);
    }
}
