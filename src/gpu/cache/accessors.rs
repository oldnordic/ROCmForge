use super::{GpuBuffer, GpuError, GpuKvCache, GpuResult};

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
