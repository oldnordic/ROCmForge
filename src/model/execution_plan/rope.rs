//! RoPE (Rotary Position Embedding) caching
//!
//! Handles caching of RoPE cos/sin tables for position embeddings.

use crate::backend::{DeviceTensor, HipError, HipResult};
use crate::loader::TensorShape;
use crate::model::execution_plan::ggml_plan::RopeCache;

use super::types::ExecutionPlan;

/// Get RoPE cache (cached cos/sin tables on GPU)
pub fn rope_cache(plan: &ExecutionPlan) -> HipResult<Option<&'static RopeCache>> {
    use once_cell::sync::OnceCell;

    let Some(ref position_handler) = plan.position_handler() else {
        return Ok(None);
    };
    let Some(rope) = position_handler.rope() else {
        return Ok(None);
    };

    // SAFETY: We extend the lifetime to 'static because the RoPE cache
    // is stored in a OnceCell within the ExecutionPlan, which owns the data.
    // The ExecutionPlan outlives any borrow of the cache.
    let cache = unsafe {
        std::mem::transmute::<
            &OnceCell<RopeCache>,
            &'static OnceCell<RopeCache>,
        >(plan.rope_cache())
    };

    let result = cache.get_or_try_init(|| {
        let half_dim = rope.config().head_dim / 2;
        let max_seq_len = rope.config().max_seq_len;
        let cos_shape = TensorShape::from_dims(&[max_seq_len, half_dim]);
        let sin_shape = TensorShape::from_dims(&[max_seq_len, half_dim]);
        let cos_tensor =
            DeviceTensor::from_host_vec(plan.backend(), rope.cos().to_vec(), cos_shape)?;
        let sin_tensor =
            DeviceTensor::from_host_vec(plan.backend(), rope.sin().to_vec(), sin_shape)?;
        Ok::<RopeCache, HipError>(RopeCache {
            cos: cos_tensor,
            sin: sin_tensor,
            half_dim,
            max_seq_len,
        })
    })?;

    Ok(Some(result))
}
