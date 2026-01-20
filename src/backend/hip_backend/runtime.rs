//! Model runtime for managing device buffers and weights

use std::sync::Arc;

use crate::backend::hip_backend::backend::{DeviceTensor, HipBackend};
use crate::backend::hip_backend::error::HipResult;
use crate::backend::hip_backend::HipError;

/// Model runtime for managing device buffers and weights
#[derive(Debug)]
pub struct ModelRuntime {
    backend: Arc<HipBackend>,
    execution_plan: Option<crate::model::execution_plan::ExecutionPlan>,
    weight_buffers: Vec<usize>,
    scratch: crate::backend::scratch::ScratchBufferManager,
    kv_cache: crate::model::kv_cache::KVCache,
}

impl ModelRuntime {
    /// Create new model runtime with minimal overhead
    pub fn new() -> HipResult<Self> {
        tracing::debug!("ModelRuntime::new() called");
        let backend_ref = HipBackend::new()?;
        let backend = backend_ref.clone();
        tracing::debug!("ModelRuntime::new() backend created, creating scratch buffer...");
        let scratch = crate::backend::scratch::ScratchBufferManager::new(
            &backend,
            32,   // num_heads
            4096, // hidden_size
            128,  // head_dim
            2048, // max_seq_len
        )
        .map_err(|e| HipError::GenericError(format!("Scratch buffer creation failed: {}", e)))?;
        tracing::debug!("ModelRuntime::new() scratch buffer created, creating KV cache...");

        let kv_cache = crate::model::kv_cache::KVCache::new(
            &backend,
            32,   // num_layers
            32,   // num_heads
            128,  // head_dim
            2048, // max_seq_len
        )
        .map_err(|e| HipError::GenericError(format!("KV cache creation failed: {}", e)))?;
        tracing::debug!("ModelRuntime::new() KV cache created, returning ModelRuntime");

        Ok(ModelRuntime {
            backend,
            execution_plan: None,
            weight_buffers: Vec::new(),
            scratch,
            kv_cache,
        })
    }

    /// Load model directly from GGUF file
    pub fn load_from_gguf(path: &str) -> HipResult<Self> {
        Self::load_from_gguf_with_config(path, None)
    }

    /// Load model from GGUF with optional custom config
    pub fn load_from_gguf_with_config(path: &str, custom_config: Option<crate::model::config::ModelConfig>) -> HipResult<Self> {
        tracing::debug!("load_from_gguf: Loading GGUF from path: {}", path);

        let loader = crate::loader::gguf::GgufLoader::new(path)
            .map_err(|e| HipError::GenericError(format!("Failed to load GGUF: {}", e)))?;
        tracing::debug!("load_from_gguf: GgufLoader created successfully");

        let mut config = loader
            .to_model_config()
            .map_err(|e| HipError::GenericError(format!("Failed to create config: {}", e)))?;
        tracing::debug!(
            "load_from_gguf: Config created - layers={}, heads={}, hidden={}",
            config.num_hidden_layers,
            config.num_attention_heads,
            config.hidden_size
        );

        if let Some(custom) = custom_config {
            config.max_position_embeddings = custom.max_position_embeddings;
        }

        let backend = HipBackend::new()?;

        tracing::debug!("load_from_gguf: Creating scratch buffer manager...");
        let scratch = crate::backend::scratch::ScratchBufferManager::new(
            &backend,
            config.num_attention_heads,
            config.hidden_size,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("Scratch buffer creation failed: {}", e)))?;
        eprintln!("load_from_gguf: Scratch buffer manager created");

        tracing::debug!("load_from_gguf: Creating KV cache...");
        eprintln!("load_from_gguf: Creating KV cache...");
        let kv_cache = crate::model::kv_cache::KVCache::new(
            &backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("KV cache creation failed: {}", e)))?;
        tracing::debug!("load_from_gguf: KV cache created");
        eprintln!("load_from_gguf: KV cache created");

        tracing::debug!("load_from_gguf: Creating execution plan from GGUF...");
        eprintln!("load_from_gguf: Creating execution plan from GGUF...");
        let execution_plan =
            crate::model::execution_plan::ExecutionPlan::from_gguf(&backend, &loader)?;
        tracing::debug!("load_from_gguf: Execution plan created successfully");
        eprintln!("load_from_gguf: Execution plan created successfully");

        tracing::debug!("load_from_gguf: Preloading all model weights to GPU...");
        eprintln!("load_from_gguf: Preloading all model weights to GPU...");
        let _preload_start = std::time::Instant::now();
        loader.load_to_gpu_async(&backend)
            .map_err(|e| HipError::GenericError(format!("Failed to preload weights: {}", e)))?;
        let preload_time = _preload_start.elapsed();
        tracing::debug!("load_from_gguf: Weight preload complete in {:?}", preload_time);
        eprintln!("load_from_gguf: All weights preloaded in {:.2}s", preload_time.as_secs_f64());

        tracing::debug!("load_from_gguf: ModelRuntime created successfully");
        eprintln!("load_from_gguf: ModelRuntime created successfully, returning...");
        Ok(ModelRuntime {
            backend,
            execution_plan: Some(execution_plan),
            weight_buffers: Vec::new(),
            scratch,
            kv_cache,
        })
    }

    /// Load from GGUF with pre-parsed loader
    pub fn load_from_gguf_with_loader(
        loader: Arc<crate::loader::gguf::GgufLoader>,
        custom_config: Option<crate::model::config::ModelConfig>,
    ) -> HipResult<Self> {
        tracing::debug!("load_from_gguf_with_loader: Using pre-parsed GGUF loader");

        let mut config = loader
            .to_model_config()
            .map_err(|e| HipError::GenericError(format!("Failed to create config: {}", e)))?;

        if let Some(custom) = custom_config {
            config.max_position_embeddings = custom.max_position_embeddings;
        }

        let backend = HipBackend::new()?;

        let scratch = crate::backend::scratch::ScratchBufferManager::new(
            &backend,
            config.num_attention_heads,
            config.hidden_size,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("Scratch buffer creation failed: {}", e)))?;

        let kv_cache = crate::model::kv_cache::KVCache::new(
            &backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("KV cache creation failed: {}", e)))?;

        let execution_plan =
            crate::model::execution_plan::ExecutionPlan::from_gguf(&backend, &loader)?;

        loader.load_to_gpu_async(&backend)
            .map_err(|e| HipError::GenericError(format!("Failed to preload weights: {}", e)))?;

        Ok(ModelRuntime {
            backend,
            execution_plan: Some(execution_plan),
            weight_buffers: Vec::new(),
            scratch,
            kv_cache,
        })
    }

    /// Create with config
    pub fn new_with_config(config: crate::model::config::ModelConfig) -> HipResult<Self> {
        let backend = HipBackend::new()?;
        let scratch = crate::backend::scratch::ScratchBufferManager::new(
            &backend,
            config.num_attention_heads,
            config.hidden_size,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("Scratch buffer creation failed: {}", e)))?;

        let kv_cache = crate::model::kv_cache::KVCache::new(
            &backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("KV cache creation failed: {}", e)))?;

        Ok(ModelRuntime {
            backend,
            execution_plan: None,
            weight_buffers: Vec::new(),
            scratch,
            kv_cache,
        })
    }

    /// Get reference to backend
    pub fn backend(&self) -> &HipBackend {
        &self.backend
    }

    /// Get reference to KV cache
    pub fn kv_cache(&self) -> &crate::model::kv_cache::KVCache {
        &self.kv_cache
    }

    /// Get mutable reference to KV cache
    pub fn kv_cache_mut(&mut self) -> &mut crate::model::kv_cache::KVCache {
        &mut self.kv_cache
    }

    /// Set execution plan
    pub fn set_execution_plan(&mut self, plan: crate::model::execution_plan::ExecutionPlan) {
        self.execution_plan = Some(plan);
    }

    /// Allocate weight buffer
    pub fn allocate_weight_buffer(&mut self, size: usize) -> HipResult<usize> {
        self.weight_buffers.push(size);
        Ok(self.weight_buffers.len() - 1)
    }

    /// Get mutable reference to scratch buffers
    pub fn scratch_buffers(&mut self) -> &mut crate::backend::scratch::ScratchBufferManager {
        &mut self.scratch
    }

    /// Get number of weight buffers
    pub fn weight_buffer_count(&self) -> usize {
        self.weight_buffers.len()
    }

    /// Get total weight memory
    pub fn total_weight_memory(&self) -> usize {
        self.weight_buffers.iter().sum()
    }

    /// Decode step
    pub fn decode_step(&mut self, input: &DeviceTensor) -> HipResult<DeviceTensor> {
        eprintln!(">>> decode_step: ENTRY");
        let start = std::time::Instant::now();
        tracing::debug!(
            "decode_step() called, input shape: {:?}",
            input.shape().dims()
        );
        let execution_plan = self.execution_plan.as_ref().ok_or_else(|| {
            HipError::GenericError("decode_step called without execution plan".to_string())
        })?;
        eprintln!(
            ">>> decode_step: Got execution plan with {} layers (took {:?})",
            execution_plan.layers().len(),
            start.elapsed()
        );

        if execution_plan.layers().is_empty() {
            return Err(HipError::GenericError(
                "Execution plan contains no layers".to_string(),
            ));
        }

        let hidden_size = execution_plan.config().hidden_size;

        let input_dims = input.shape().dims();
        let mut hidden_states = if input_dims.len() == 1 {
            if input_dims[0] != hidden_size {
                return Err(HipError::GenericError(format!(
                    "Input hidden size {} does not match model hidden size {}",
                    input_dims[0], hidden_size
                )));
            }
            let reshaped = crate::loader::mmap_loader::TensorShape::from_dims(&[1, hidden_size]);
            let mut tensor = DeviceTensor::empty(&self.backend, reshaped)?;
            tensor.copy_from_device_buffer(input.buffer())?;
            tensor
        } else if input_dims.len() == 2 {
            if input_dims[1] != hidden_size {
                return Err(HipError::GenericError(format!(
                    "Input hidden size {} does not match model hidden size {}",
                    input_dims[1], hidden_size
                )));
            }
            input.clone()
        } else {
            return Err(HipError::GenericError(format!(
                "decode_step input must be 1D or 2D, got shape {:?}",
                input_dims
            )));
        };

        for (layer_idx, layer_plan) in execution_plan.layers().iter().enumerate() {
            eprintln!(
                ">>> decode_step: Layer {}/{} starting...",
                layer_idx + 1,
                execution_plan.layers().len()
            );

            let seq_len = hidden_states.shape().dims()[0];
            if seq_len == 1 {
                hidden_states = execution_plan.forward_layer_ggml_decode(
                    &self.backend,
                    &hidden_states,
                    &mut self.kv_cache,
                    layer_idx,
                )?;
            } else {
                hidden_states = execution_plan.forward_layer(
                    &self.backend,
                    &hidden_states,
                    layer_plan,
                    Some(&mut self.kv_cache),
                    layer_idx,
                )?;
            }
        }

        let logits = execution_plan.apply_lm_head(&self.backend, &hidden_states)?;
        let logits_dims = logits.shape().dims();
        let output = if logits_dims.len() == 2 && logits_dims[0] == 1 {
            let mut tensor =
                DeviceTensor::empty(&self.backend, crate::loader::mmap_loader::TensorShape::from_dims(&[logits_dims[1]]))?;
            tensor.copy_from_device_slice(&logits, 0)?;
            tensor
        } else {
            logits
        };

        Ok(output)
    }

    /// Load model from GGUF file
    pub fn load_model(&self, path: &str) -> HipResult<Self> {
        tracing::debug!("load_model: Loading GGUF from path: {}", path);

        let loader = crate::loader::gguf::GgufLoader::new(path)
            .map_err(|e| HipError::GenericError(format!("Failed to load GGUF: {}", e)))?;

        let config = loader
            .to_model_config()
            .map_err(|e| HipError::GenericError(format!("Failed to create config: {}", e)))?;

        let scratch = crate::backend::scratch::ScratchBufferManager::new(
            &self.backend,
            config.num_attention_heads,
            config.hidden_size,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("Scratch buffer creation failed: {}", e)))?;

        let kv_cache = crate::model::kv_cache::KVCache::new(
            &self.backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("KV cache creation failed: {}", e)))?;

        let execution_plan =
            crate::model::execution_plan::ExecutionPlan::from_gguf(&self.backend, &loader)?;

        loader.load_to_gpu_async(&self.backend)
            .map_err(|e| HipError::GenericError(format!("Failed to preload weights: {}", e)))?;

        Ok(ModelRuntime {
            backend: self.backend.clone(),
            execution_plan: Some(execution_plan),
            weight_buffers: Vec::new(),
            scratch,
            kv_cache,
        })
    }

    /// Create from execution plan
    pub fn from_execution_plan(
        &self,
        execution_plan: crate::model::execution_plan::ExecutionPlan,
    ) -> HipResult<Self> {
        Self::from_execution_plan_with_backend(execution_plan)
    }

    /// Create from execution plan with backend
    pub fn from_execution_plan_with_backend(
        execution_plan: crate::model::execution_plan::ExecutionPlan,
    ) -> HipResult<Self> {
        eprintln!(">>> from_execution_plan_with_backend ENTRY");
        let backend = HipBackend::new()?;
        let config = execution_plan.config();

        let scratch = crate::backend::scratch::ScratchBufferManager::new(
            &backend,
            config.num_attention_heads,
            config.hidden_size,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("Scratch buffer creation failed: {}", e)))?;

        let kv_cache = crate::model::kv_cache::KVCache::new(
            &backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("KV cache recreation failed: {}", e)))?;

        Ok(ModelRuntime {
            backend,
            execution_plan: Some(execution_plan),
            weight_buffers: Vec::new(),
            scratch,
            kv_cache,
        })
    }

    /// Get execution plan
    pub fn execution_plan(&self) -> Option<&crate::model::execution_plan::ExecutionPlan> {
        self.execution_plan.as_ref()
    }

    /// Reset state
    pub fn reset_state(&mut self) -> HipResult<()> {
        if self.execution_plan.is_none() {
            return Err(HipError::GenericError(
                "reset_state called without execution plan".to_string(),
            ));
        }
        self.kv_cache.reset();
        Ok(())
    }

    /// Recreate KV cache
    pub fn recreate_kv_cache(&mut self, max_seq_len: usize) -> HipResult<()> {
        let config = self.execution_plan.as_ref()
            .ok_or_else(|| HipError::GenericError("No execution plan".to_string()))?
            .config();

        self.kv_cache = crate::model::kv_cache::KVCache::new(
            &self.backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            max_seq_len,
        )
        .map_err(|e| HipError::GenericError(format!("KV cache recreation failed: {}", e)))?;

        Ok(())
    }
}

impl DeviceTensor {
    pub fn copy_from_device_buffer(&mut self, src: &super::memory::HipBuffer) -> HipResult<()> {
        self.buffer.copy_from_buffer(src)
    }

    pub fn copy_from_device_slice(
        &mut self,
        src: &DeviceTensor,
        src_offset_elements: usize,
    ) -> HipResult<()> {
        let byte_len = self.len() * std::mem::size_of::<f32>();
        let byte_offset = src_offset_elements * std::mem::size_of::<f32>();
        self.buffer
            .copy_from_buffer_with_offset(src.buffer(), byte_offset, byte_len)
    }
}
