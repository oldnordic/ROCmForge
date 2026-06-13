use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::config::{ChatTemplate, ModelConfig};
use crate::cpu::weights::CpuModelWeights;
use crate::loader::ModelFile;
use crate::tokenizer::BpeTokenizer;

// ── Per-model state ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct ModelEntry {
    pub model_path: String,
    pub config: ModelConfig,
    pub tokenizer: BpeTokenizer,
    pub chat_template: ChatTemplate,
    pub created_at: i64,
    /// One permit per model: serializes inference so concurrent requests
    /// do not race on weights or KV cache.  Arc so acquire_owned() yields
    /// an OwnedSemaphorePermit that is Send for spawn_blocking.
    pub inference_sem: Arc<tokio::sync::Semaphore>,
    /// Cached CPU weights — loaded once at model load time and shared across
    /// all inference requests via Arc so cloning into spawn_blocking is cheap.
    pub cpu_weights: Arc<CpuModelWeights>,
    #[cfg(feature = "gpu")]
    pub gpu_weights: Option<Arc<crate::gpu::GpuModelWeights>>,
    #[cfg(feature = "gpu")]
    pub speculative_engine: Option<Arc<Mutex<crate::gpu::SpeculativeEngine>>>,
}

impl ModelEntry {
    pub fn load(
        model_path: &str,
        draft_path: Option<&str>,
    ) -> crate::error::RocmForgeResult<Self> {
        let file = ModelFile::open(model_path)?;
        let config = file.config()?;
        let tokenizer = file.tokenizer();
        let chat_template = file.chat_template(&config, false); // enable template by default
        let cpu_weights = Arc::new(file.load_cpu_weights(&config)?);

        #[cfg(feature = "gpu")]
        let gpu_weights = {
            let gpu_caps = crate::gpu::detect();
            if let Some(caps) = gpu_caps {
                crate::gpu::GpuDevice::get_or_init(caps.device_id)?;
                let w = file.load_gpu_weights(&config, caps.device_id)?;
                Some(Arc::new(w))
            } else {
                None
            }
        };

        #[cfg(feature = "gpu")]
        let speculative_engine = if let Some(dp) = draft_path {
            let gpu_caps = crate::gpu::detect()
                .ok_or("GPU requested for speculative decoding but no AMD GPU detected")?;
            let device = crate::gpu::GpuDevice::get_or_init(gpu_caps.device_id)?;
            let engine = crate::gpu::SpeculativeEngine::new(
                &device,
                model_path,
                dp,
                config.max_seq_len.min(2048),
                256,
            )?;
            Some(Arc::new(Mutex::new(engine)))
        } else {
            None
        };

        Ok(ModelEntry {
            model_path: model_path.to_string(),
            config,
            tokenizer,
            chat_template,
            created_at: chrono::Utc::now().timestamp(),
            inference_sem: Arc::new(tokio::sync::Semaphore::new(1)),
            cpu_weights,
            #[cfg(feature = "gpu")]
            gpu_weights,
            #[cfg(feature = "gpu")]
            speculative_engine,
        })
    }
}

// ── Global server state ─────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct ModelManager {
    /// Active models indexed by their internal ID (usually the path or alias).
    /// Currently rocmforge only supports loading one model at a time due to
    /// exclusive VRAM usage, but this is future-proofed for multi-GPU.
    pub active_models: Mutex<HashMap<String, Arc<ModelEntry>>>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            active_models: Mutex::new(HashMap::new()),
        }
    }

    pub async fn try_load(
        &self,
        model_path: &str,
        draft_path: Option<&str>,
    ) -> Result<Arc<ModelEntry>, String> {
        let entry = Arc::new(ModelEntry::load(model_path, draft_path).map_err(|e| e.to_string())?);
        self.try_load_entry(entry.clone()).await?;
        Ok(entry)
    }

    pub async fn try_load_entry(&self, entry: Arc<ModelEntry>) -> Result<(), String> {
        self.active_models
            .lock()
            .await
            .insert(entry.model_path.clone(), entry);
        Ok(())
    }

    pub async fn unload(&self, model_path: &str) {
        self.active_models.lock().await.remove(model_path);
    }

    pub async fn get(&self, model_path: &str) -> Option<Arc<ModelEntry>> {
        self.active_models.lock().await.get(model_path).cloned()
    }

    pub async fn keys(&self) -> Vec<String> {
        self.active_models.lock().await.keys().cloned().collect()
    }
}
