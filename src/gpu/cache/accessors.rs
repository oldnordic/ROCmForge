use super::{GpuBuffer, GpuError, GpuKvCache, GpuResult};
use crate::gpu::ffi::hip_memcpy_d2h;

fn layer_bounds_error(layer: usize, num_layers: usize) -> GpuError {
    GpuError::HipApiError {
        code: -1,
        description: format!("Layer {layer} exceeds num_layers {num_layers}"),
    }
}

fn layer_ptr(bufs: &[GpuBuffer], layer: usize, num_layers: usize) -> GpuResult<*mut f32> {
    if layer >= num_layers {
        return Err(layer_bounds_error(layer, num_layers));
    }
    Ok(bufs[layer].as_ptr() as *mut f32)
}

fn optional_layer_ptr(
    bufs: Option<&Vec<GpuBuffer>>,
    layer: usize,
    num_layers: usize,
) -> GpuResult<Option<*mut f32>> {
    if layer >= num_layers {
        return Err(layer_bounds_error(layer, num_layers));
    }
    Ok(bufs.map(|buffers| buffers[layer].as_ptr() as *mut f32))
}

impl GpuKvCache {
    /// Get pointer to TurboQuant centroids in GPU VRAM.
    pub fn centroids_ptr(&self) -> GpuResult<*const f32> {
        Ok(self
            .centroids
            .as_ref()
            .map(|buf| buf.as_ptr() as *const f32)
            .unwrap_or(std::ptr::null()))
    }

    /// Get GPU pointer to SSM state cache for a layer.
    pub fn ssm_state_ptr(&self, layer: usize) -> GpuResult<Option<*mut f32>> {
        optional_layer_ptr(self.ssm_state.as_ref(), layer, self.num_layers)
    }

    /// Get GPU pointer to SSM conv state cache for a layer.
    pub fn ssm_conv_state_ptr(&self, layer: usize) -> GpuResult<Option<*mut f32>> {
        optional_layer_ptr(self.ssm_conv_state.as_ref(), layer, self.num_layers)
    }

    /// Get GPU pointer to K cache for a layer.
    ///
    /// Returns pointer suitable for kernel arguments.
    pub fn k_ptr(&self, layer: usize) -> GpuResult<*mut f32> {
        layer_ptr(&self.k, layer, self.num_layers)
    }

    /// Get GPU pointer to V cache for a layer.
    pub fn v_ptr(&self, layer: usize) -> GpuResult<*mut f32> {
        layer_ptr(&self.v, layer, self.num_layers)
    }

    /// Get GPU pointer to shortconv conv state for a layer.
    pub fn conv_state_ptr(&self, layer: usize) -> GpuResult<Option<*mut f32>> {
        if let Some(ref states) = self.conv_state {
            layer_ptr(states, layer, self.num_layers).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Copy the populated prefix of K/V cache for a layer back to host memory.
    ///
    /// `dst_k` and `dst_v` must each have length `seq_len * effective_kv`, where
    /// `effective_kv` is `kv_lora_dim` when LoRA compression is enabled and
    /// `kv_size` otherwise.
    pub fn copy_kv_prefix_to_host(
        &self,
        layer: usize,
        seq_len: usize,
        dst_k: &mut [f32],
        dst_v: &mut [f32],
    ) -> GpuResult<()> {
        if layer >= self.num_layers {
            return Err(layer_bounds_error(layer, self.num_layers));
        }
        let effective_kv = self.kv_lora_dim.unwrap_or(self.kv_size);
        let expected = seq_len.saturating_mul(effective_kv);
        if dst_k.len() != expected || dst_v.len() != expected {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "kv prefix size mismatch: expected {} floats, got k={} v={}",
                    expected,
                    dst_k.len(),
                    dst_v.len()
                ),
            });
        }
        if expected == 0 {
            return Ok(());
        }
        let bytes = expected * std::mem::size_of::<f32>();
        let k_ptr = self.k[layer].as_ptr();
        let v_ptr = self.v[layer].as_ptr();
        hip_memcpy_d2h(dst_k.as_mut_ptr() as *mut u8, k_ptr, bytes)?;
        hip_memcpy_d2h(dst_v.as_mut_ptr() as *mut u8, v_ptr, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::layer_bounds_error;

    #[test]
    fn layer_bounds_error_mentions_requested_and_limit() {
        let err = layer_bounds_error(5, 2);
        let msg = err.to_string();
        assert!(msg.contains("Layer 5 exceeds num_layers 2"));
    }
}
