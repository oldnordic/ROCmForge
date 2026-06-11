//! Unified model file abstraction over GGUF and RFM formats.
//!
//! Eliminates the repeated `if path.ends_with(".rfm")` branching throughout
//! the codebase by providing a single enum that dispatches format-specific logic.

use crate::config::{detect_chat_template, ChatTemplate, ModelConfig};
use crate::cpu::weights::CpuModelWeights;
use crate::loader::{GgufFile, RfmFile};
use crate::tokenizer::BpeTokenizer;

pub enum ModelFile {
    Gguf(GgufFile),
    Rfm(RfmFile),
}

impl ModelFile {
    pub fn open(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        if path.ends_with(".rfm") {
            Ok(Self::Rfm(RfmFile::open(path)?))
        } else {
            Ok(Self::Gguf(GgufFile::open(path)?))
        }
    }

    pub fn format_name(&self) -> &'static str {
        match self {
            Self::Gguf(_) => "GGUF",
            Self::Rfm(_) => "RFM",
        }
    }

    pub fn config(&self) -> Result<ModelConfig, Box<dyn std::error::Error>> {
        match self {
            Self::Gguf(f) => Ok(ModelConfig::from_gguf(f)?),
            Self::Rfm(f) => Ok(ModelConfig::from_rfm(&f.metadata)?),
        }
    }

    pub fn tokenizer(&self) -> BpeTokenizer {
        match self {
            Self::Gguf(f) => BpeTokenizer::from_gguf(f.tokenizer_data()),
            Self::Rfm(f) => BpeTokenizer::from_rfm(&f.metadata),
        }
    }

    pub fn chat_template(&self, config: &ModelConfig, no_template: bool) -> ChatTemplate {
        if no_template {
            return ChatTemplate::None;
        }
        match self {
            Self::Gguf(f) => {
                detect_chat_template(&config.architecture, f.tokenizer_data().model.as_deref())
            }
            Self::Rfm(f) => {
                detect_chat_template(&config.architecture, f.metadata.tokenizer_model.as_deref())
            }
        }
    }

    pub fn load_cpu_weights(
        &self,
        config: &ModelConfig,
    ) -> Result<CpuModelWeights, Box<dyn std::error::Error>> {
        match self {
            Self::Gguf(f) => Ok(CpuModelWeights::load(f, config)?),
            Self::Rfm(f) => Ok(CpuModelWeights::load_rfm(f, config)?),
        }
    }

    #[cfg(feature = "gpu")]
    pub fn load_gpu_weights(
        &self,
        config: &ModelConfig,
        device_id: i32,
    ) -> Result<crate::gpu::GpuModelWeights, Box<dyn std::error::Error>> {
        match self {
            Self::Gguf(f) => Ok(crate::gpu::GpuModelWeights::load_for_device(
                f, config, device_id,
            )?),
            Self::Rfm(f) => Ok(crate::gpu::GpuModelWeights::load_rfm_for_device(
                f, config, device_id,
            )?),
        }
    }

    pub fn as_rfm(&self) -> Option<&RfmFile> {
        match self {
            Self::Rfm(f) => Some(f),
            _ => None,
        }
    }

    pub fn as_gguf(&self) -> Option<&GgufFile> {
        match self {
            Self::Gguf(f) => Some(f),
            _ => None,
        }
    }

    /// Returns the byte size of a named tensor, or 0 if not present.
    /// Used for VRAM pre-flight estimation without triggering a full load.
    pub fn tensor_byte_size(&self, name: &str) -> usize {
        match self {
            Self::Gguf(f) => f
                .tensor(name)
                .ok()
                .flatten()
                .map(|t| t.data.len())
                .unwrap_or(0),
            Self::Rfm(f) => f
                .tensor(name)
                .ok()
                .flatten()
                .map(|t| t.data.len())
                .unwrap_or(0),
        }
    }
}
