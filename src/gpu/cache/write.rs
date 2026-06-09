use super::{GpuBuffer, GpuError, GpuKvCache, GpuResult};
use crate::gpu::ffi::hipStream_t;
use crate::gpu::kernels::{
    kv_write_batched, kv_write_batched_compressed, kv_write_compressed, kv_write_on_stream,
    kv_write_turboquant, reconstruct_kv_cache_prefix_sum,
};

fn i32_arg_buffer(value: i32) -> GpuResult<GpuBuffer> {
    let mut buf = GpuBuffer::alloc(std::mem::size_of::<i32>())?;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            &value as *const i32 as *const u8,
            std::mem::size_of::<i32>(),
        )
    };
    buf.copy_from_host(bytes)?;
    Ok(buf)
}

fn optional_layer_weights(weights: Option<&Vec<GpuBuffer>>, layer: usize) -> *const f32 {
    weights
        .map(|per_layer| per_layer[layer].as_ptr() as *const f32)
        .unwrap_or(std::ptr::null())
}

impl GpuKvCache {
    /// Write K/V vectors to cache.
    pub fn write(
        &mut self,
        layer: usize,
        pos: usize,
        k_gpu: *const f32,
        v_gpu: *const f32,
    ) -> GpuResult<()> {
        self.write_on_stream(layer, pos, k_gpu, v_gpu, hipStream_t::null())
    }

    /// Write K/V vectors to cache on an explicit HIP stream.
    pub fn write_on_stream(
        &mut self,
        layer: usize,
        pos: usize,
        k_gpu: *const f32,
        v_gpu: *const f32,
        stream: hipStream_t,
    ) -> GpuResult<()> {
        self.write_on_stream_impl(layer, pos, k_gpu, v_gpu, 10000.0, false, stream)
    }

    fn write_on_stream_impl(
        &mut self,
        layer: usize,
        pos: usize,
        k_gpu: *const f32,
        v_gpu: *const f32,
        theta_base: f32,
        neox: bool,
        stream: hipStream_t,
    ) -> GpuResult<()> {
        let k_cache = self.k_ptr(layer)?;
        let v_cache = self.v_ptr(layer)?;

        if self.kv_quant_bits.is_some() {
            let dc = self.kv_lora_dim.ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: "TurboQuant requires kv_lora_dim".to_string(),
            })?;
            let d_pos = i32_arg_buffer(pos as i32)?;
            let w_down_k = optional_layer_weights(self.w_down_k.as_ref(), layer);
            let w_down_v = optional_layer_weights(self.w_down_v.as_ref(), layer);
            let centroids = self.centroids_ptr()?;

            kv_write_turboquant(
                k_cache as *mut u8,
                v_cache as *mut u8,
                k_gpu,
                v_gpu,
                d_pos.as_ptr() as *const i32,
                self.num_kv_heads,
                self.head_dim,
                theta_base,
                neox,
                dc,
                centroids,
                self.qjl_scale,
                w_down_k,
                w_down_v,
                stream,
            )?;
            super::super::ffi::hip_stream_synchronize(stream)?;
            self.scatter_to_paged(layer, pos, 1)?;
            return Ok(());
        }

        if let Some(dc) = self.kv_lora_dim {
            let d_pos = i32_arg_buffer(pos as i32)?;
            let w_down_k = optional_layer_weights(self.w_down_k.as_ref(), layer);
            let w_down_v = optional_layer_weights(self.w_down_v.as_ref(), layer);

            kv_write_compressed(
                k_cache,
                v_cache,
                k_gpu,
                v_gpu,
                d_pos.as_ptr() as *const i32,
                self.num_kv_heads,
                self.head_dim,
                theta_base,
                neox,
                dc,
                self.kv_frame_codec_enabled,
                w_down_k,
                w_down_v,
                stream,
            )?;
            super::super::ffi::hip_stream_synchronize(stream)?;
            self.scatter_to_paged(layer, pos, 1)?;
            return Ok(());
        }

        kv_write_on_stream(
            k_cache,
            v_cache,
            k_gpu,
            v_gpu,
            pos,
            self.kv_size,
            self.max_seq_len,
            stream,
        )?;
        super::super::ffi::hip_stream_synchronize(stream)?;
        self.scatter_to_paged(layer, pos, 1)?;
        Ok(())
    }

    /// Batch write K/V for prefill (multiple positions).
    ///
    /// # Arguments
    /// * `start_pos` - Starting position
    /// * `seq_len` - Number of positions to write
    /// * `k_gpu` - GPU pointer to batched key vectors [seq_len * kv_size]
    /// * `v_gpu` - GPU pointer to batched value vectors
    pub fn write_batched(
        &mut self,
        layer: usize,
        start_pos: usize,
        seq_len: usize,
        k_gpu: *const f32,
        v_gpu: *const f32,
    ) -> GpuResult<()> {
        let k_cache = self.k_ptr(layer)?;
        let v_cache = self.v_ptr(layer)?;

        let res = if self.kv_quant_bits.is_some() {
            for s in 0..seq_len {
                let pos = start_pos + s;
                let k_ptr = unsafe { k_gpu.add(s * self.kv_size) };
                let v_ptr = unsafe { v_gpu.add(s * self.kv_size) };
                self.write_on_stream_impl(
                    layer,
                    pos,
                    k_ptr,
                    v_ptr,
                    0.0f32,
                    false,
                    hipStream_t::null(),
                )?;
            }
            Ok(())
        } else if let Some(dc) = self.kv_lora_dim {
            let w_down_k = optional_layer_weights(self.w_down_k.as_ref(), layer);
            let w_down_v = optional_layer_weights(self.w_down_v.as_ref(), layer);

            kv_write_batched_compressed(
                k_cache,
                v_cache,
                k_gpu,
                v_gpu,
                start_pos,
                self.num_kv_heads,
                self.head_dim,
                seq_len,
                dc,
                self.kv_frame_codec_enabled,
                w_down_k,
                w_down_v,
                hipStream_t::null(),
            )?;

            if self.kv_frame_codec_enabled {
                reconstruct_kv_cache_prefix_sum(
                    k_cache,
                    v_cache,
                    start_pos,
                    seq_len,
                    dc,
                    hipStream_t::null(),
                )?;
            }
            Ok(())
        } else {
            kv_write_batched(
                k_cache,
                v_cache,
                k_gpu,
                v_gpu,
                start_pos,
                self.kv_size,
                self.max_seq_len,
                seq_len,
            )
        };

        if res.is_ok() {
            super::super::ffi::hip_stream_synchronize(hipStream_t::null())?;
            self.scatter_to_paged(layer, start_pos, seq_len)?;
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::optional_layer_weights;
    use crate::gpu::weights::GpuBuffer;

    #[test]
    fn optional_layer_weights_returns_null_for_missing_weights() {
        assert!(optional_layer_weights(None, 0).is_null());
    }

    #[test]
    fn optional_layer_weights_returns_layer_pointer() {
        let buf = GpuBuffer::alloc(16);
        if let Ok(buffer) = buf {
            let weights = vec![buffer];
            let ptr = optional_layer_weights(Some(&weights), 0);
            assert!(!ptr.is_null());
        }
    }
}
