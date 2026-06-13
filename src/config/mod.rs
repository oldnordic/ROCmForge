//! Model configuration - all hyperparameters needed for inference.
//!
//! Two concerns are strictly separated:
//!   - **Values** (sizes, dimensions, epsilons): always read from GGUF metadata.
//!   - **Behaviors** (RoPE style, attention layout): hardcoded per architecture
//!     in the `ModelTraits` registry below.
//!
//! `ModelConfig::from_gguf()` combines both into one validated struct.

mod chat_template;
mod model_config;
mod tensor_names;
mod tensor_role;
mod traits;

pub use chat_template::{detect_chat_template, ChatTemplate};
pub use model_config::{ConfigError, ModelConfig};
pub use tensor_names::{TensorName, TensorNameRegistry, TensorNamingScheme};
pub use tensor_role::TensorRole;
pub use traits::{AttentionLayout, FfnLayout, ModelTraits, RopeStyle};
