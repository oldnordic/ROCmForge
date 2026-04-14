//! GPU weight storage in VRAM.
//!
//! Safety-first design:
//! - All hipMalloc/hipMemcpy calls wrapped with error checking
//! - RAII cleanup on Drop prevents VRAM leaks
//! - Never panic, always return GpuError

use super::error::{GpuError, GpuResult};
use super::ffi;
use super::ffi::hipStream_t;
use crate::config::{ModelConfig, TensorName, TensorNamingScheme};
use crate::cpu::transpose::compute_transpose_flag;
use crate::loader::{GgmlType, GgufFile, TensorDesc};
use std::ptr::NonNull;

// ── VRAM Safety Constants ────────────────────────────────────────────────────────

/// VRAM reserved for desktop/compositor (multi-monitor setups).
/// This prevents allocations from stealing memory needed for display.
/// 4 GB is typical for multi-monitor 4K setups with desktop compositors.
const DESKTOP_VRAM_RESERVATION_BYTES: usize = 4 * 1024 * 1024 * 1024; // 4 GB

/// Safety margin for VRAM allocations (10% of free VRAM).
/// This prevents allocating 100% of available VRAM which could cause issues.
const VRAM_SAFETY_MARGIN_RATIO: f64 = 0.1;

/// Additional guardrail for full model loads.
///
/// Model loading performs many allocations back-to-back, so keep a larger
/// buffer than the one-off allocation guard.
const MODEL_LOAD_SAFE_RATIO: f64 = 0.7;

#[derive(Clone, Copy, Debug)]
struct VramBudget {
    device_id: i32,
    free_vram: usize,
    total_vram: usize,
    safe_allocation_size: usize,
    safe_model_load_limit: usize,
}

fn query_vram_budget(device_id: i32) -> GpuResult<VramBudget> {
    let (free_vram, total_vram) =
        ffi::hip_get_mem_info(device_id).map_err(|e| GpuError::HipApiError {
            code: -1,
            description: format!(
                "VRAM safety query failed for device {}: {}. Refusing unsafe GPU allocation.",
                device_id, e
            ),
        })?;
    let usable_vram = free_vram.saturating_sub(DESKTOP_VRAM_RESERVATION_BYTES);
    Ok(VramBudget {
        device_id,
        free_vram,
        total_vram,
        safe_allocation_size: (usable_vram as f64 * (1.0 - VRAM_SAFETY_MARGIN_RATIO)) as usize,
        safe_model_load_limit: (usable_vram as f64 * MODEL_LOAD_SAFE_RATIO) as usize,
    })
}

fn active_or_default_device_id() -> i32 {
    ffi::hip_get_device().unwrap_or(0)
}

fn check_model_load_headroom(
    budget: VramBudget,
    current_usage: usize,
    next_allocation: usize,
) -> GpuResult<()> {
    let projected = current_usage.saturating_add(next_allocation);
    if projected > budget.safe_model_load_limit {
        return Err(GpuError::ModelTooLarge {
            required: projected,
            available: budget.safe_model_load_limit,
            hint: format!(
                "Projected GPU weight load on device {} would use {} MB, exceeding the guarded load budget of {} MB ({} MB free, {} MB reserved for desktop, {} MB total VRAM).",
                budget.device_id,
                projected / (1024 * 1024),
                budget.safe_model_load_limit / (1024 * 1024),
                budget.free_vram / (1024 * 1024),
                DESKTOP_VRAM_RESERVATION_BYTES / (1024 * 1024),
                budget.total_vram / (1024 * 1024)
            ),
        });
    }
    Ok(())
}

// ── Weight Metadata ────────────────────────────────────────────────────────────

/// Metadata for a weight tensor on GPU.
///
/// Same as CPU WeightMeta - quantization type and dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorRole {
    Generic,
    LmHead,
    TiedLmHead,
}

#[derive(Clone, Debug)]
pub struct WeightMeta {
    /// Quantization type (F32, Q4_0, Q4_1, Q8_0, etc.)
    pub wtype: GgmlType,
    /// Dimensions from GGUF (innermost first)
    pub dims: Vec<u64>,
    /// Whether this weight tensor needs transposed access
    pub needs_transpose: bool,
    /// Semantic role derived from GGUF/model metadata.
    ///
    /// This allows dispatch to specialize important tensors without
    /// hardcoding model names or architecture-specific assumptions.
    pub role: TensorRole,
}

impl WeightMeta {
    /// Create metadata from a GGUF tensor descriptor.
    pub fn from_desc(desc: &TensorDesc, needs_transpose: bool) -> Self {
        Self {
            wtype: desc.ggml_type,
            dims: desc.dims.clone(),
            needs_transpose,
            role: TensorRole::Generic,
        }
    }

    /// Total size in bytes for this weight tensor.
    pub fn byte_size(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Number of elements in this tensor.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_desc_works() {
        let desc = TensorDesc {
            name: "test.weight".to_string(),
            ggml_type: GgmlType::Q4_0,
            dims: vec![1024, 768],
            offset: 0,
        };
        let meta = WeightMeta::from_desc(&desc, false);
        assert_eq!(meta.wtype, GgmlType::Q4_0);
        assert_eq!(meta.dims, vec![1024, 768]);
        assert_eq!(meta.byte_size(), 1024 * 768);
        assert_eq!(meta.num_elements(), 1024 * 768);
    }

    #[test]
    fn byte_size_calculates_correctly() {
        let meta = WeightMeta {
            wtype: GgmlType::F32,
            dims: vec![100, 200],
            needs_transpose: false,
            role: TensorRole::Generic,
        };
        assert_eq!(meta.byte_size(), 100 * 200);
    }
}

// ── GPU Buffer (RAII) ─────────────────────────────────────────────────────────────

/// RAII wrapper for GPU memory allocation.
///
/// Ensures memory is freed when dropped.
/// Never leaks VRAM, even on panic.
pub struct GpuBuffer {
    /// Pointer to GPU memory (null if empty)
    ptr: Option<NonNull<u8>>,
    /// Size in bytes
    size: usize,
}

impl GpuBuffer {
    /// Allocate GPU memory with safety checking.
    ///
    /// Returns error if allocation fails (OutOfMemory).
    /// Checks available VRAM before allocation to prevent stealing memory from desktop.
    pub fn alloc(size: usize) -> GpuResult<Self> {
        Self::alloc_for_device(size, active_or_default_device_id())
    }

    /// Allocate GPU memory on a specific device with safety checking.
    pub fn alloc_for_device(size: usize, device_id: i32) -> GpuResult<Self> {
        if size == 0 {
            return Ok(Self { ptr: None, size: 0 });
        }

        ffi::hip_set_device(device_id)?;
        let budget = query_vram_budget(device_id)?;
        if size > budget.safe_allocation_size {
            return Err(GpuError::OutOfMemory {
                requested: size,
                available: budget.safe_allocation_size,
                hint: format!(
                    "Device {} only has {} MB safely allocatable ({} MB free, {} MB reserved for desktop, {}% safety margin, {} MB total VRAM).",
                    device_id,
                    budget.safe_allocation_size / (1024 * 1024),
                    budget.free_vram / (1024 * 1024),
                    DESKTOP_VRAM_RESERVATION_BYTES / (1024 * 1024),
                    (VRAM_SAFETY_MARGIN_RATIO * 100.0) as u32,
                    budget.total_vram / (1024 * 1024)
                ),
            });
        }

        let ptr = ffi::hip_malloc(size)?;

        // Verify allocation succeeded (pointer not null)
        let nn = NonNull::new(ptr).ok_or_else(|| GpuError::OutOfMemory {
            requested: size,
            available: 0,
            hint: "hipMalloc returned null pointer".to_string(),
        })?;

        Ok(Self {
            ptr: Some(nn),
            size,
        })
    }

    /// Create empty buffer (no allocation).
    pub fn empty() -> Self {
        Self { ptr: None, size: 0 }
    }

    /// Get pointer to GPU memory.
    ///
    /// Returns None if buffer is empty.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
            .map(|nn| nn.as_ptr())
            .unwrap_or(std::ptr::null_mut())
    }

    /// Get size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Set size in bytes (only for empty or manually managed buffers).
    pub fn set_size(&mut self, size: usize) {
        self.size = size;
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Copy data from CPU to this GPU buffer.
    pub fn copy_from_host(&mut self, src: &[u8]) -> GpuResult<()> {
        if src.len() != self.size {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "size mismatch: got {} bytes, expected {}",
                    src.len(),
                    self.size
                ),
            });
        }
        if self.size == 0 {
            return Ok(());
        }
        ffi::hip_memcpy_h2d(self.as_ptr(), src.as_ptr(), self.size)
    }

    /// Copy data from CPU to this GPU buffer on an explicit HIP stream.
    pub fn copy_from_host_on_stream(&mut self, src: &[u8], stream: hipStream_t) -> GpuResult<()> {
        if src.len() != self.size {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "size mismatch: got {} bytes, expected {}",
                    src.len(),
                    self.size
                ),
            });
        }
        if self.size == 0 {
            return Ok(());
        }
        ffi::hip_memcpy_h2d_async(self.as_ptr(), src.as_ptr(), self.size, stream)
    }

    /// Copy data from GPU buffer to CPU.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> GpuResult<()> {
        if dst.len() != self.size {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "size mismatch: got {} bytes, expected {}",
                    dst.len(),
                    self.size
                ),
            });
        }
        if self.size == 0 {
            return Ok(());
        }
        ffi::hip_memcpy_d2h(dst.as_mut_ptr(), self.as_ptr(), self.size)
    }
}

// ── GPU Pinned Buffer (RAII) ──────────────────────────────────────────────────────

/// RAII wrapper for Pinned (Page-locked) Host memory.
///
/// Allows for high-speed DMA transfers and zero-copy access from GPU.
pub struct GpuPinnedBuffer {
    ptr: Option<NonNull<u8>>,
    size: usize,
}

impl GpuPinnedBuffer {
    pub fn alloc(size: usize) -> GpuResult<Self> {
        if size == 0 {
            return Ok(Self { ptr: None, size: 0 });
        }

        let ptr = ffi::hip_host_malloc(size)?;
        let nn = NonNull::new(ptr).ok_or_else(|| GpuError::OutOfMemory {
            requested: size,
            available: 0,
            hint: "hipHostMalloc returned null pointer".to_string(),
        })?;

        Ok(Self {
            ptr: Some(nn),
            size,
        })
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
            .map(|nn| nn.as_ptr())
            .unwrap_or(std::ptr::null_mut())
    }

    pub fn as_slice<T>(&self) -> &[T] {
        if self.size == 0 {
            return &[];
        }
        unsafe {
            std::slice::from_raw_parts(
                self.as_ptr() as *const T,
                self.size / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn as_slice_mut<T>(&mut self) -> &mut [T] {
        if self.size == 0 {
            return &mut [];
        }
        unsafe {
            std::slice::from_raw_parts_mut(
                self.as_ptr() as *mut T,
                self.size / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Drop for GpuPinnedBuffer {
    fn drop(&mut self) {
        if let Some(nn) = self.ptr {
            let _ = ffi::hip_host_free(nn.as_ptr());
        }
    }
}

unsafe impl Send for GpuPinnedBuffer {}
unsafe impl Sync for GpuPinnedBuffer {}

// SAFETY: Send/Sync are safe because this represents owned GPU memory
// Access is only through &mut self for copy operations
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if let Some(nn) = self.ptr {
            ffi::hip_free(nn.as_ptr());
            // Ignore errors in Drop - can't panic here
            self.ptr = None;
        }
    }
}

#[cfg(test)]
mod buffer_tests {
    use super::*;

    #[test]
    fn empty_buffer_has_no_allocation() {
        let buf = GpuBuffer::empty();
        assert!(buf.is_empty());
        assert_eq!(buf.size(), 0);
        assert_eq!(buf.as_ptr(), std::ptr::null_mut());
    }

    #[test]
    fn alloc_zero_size_returns_empty() {
        let buf = GpuBuffer::alloc(0).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn copy_from_host_rejects_size_mismatch() {
        let mut buf = GpuBuffer::alloc(100).unwrap();
        let data = vec![1u8; 50]; // Wrong size
        let result = buf.copy_from_host(&data);
        assert!(result.is_err());
    }
}

fn supports_gpu_matrix_type(wtype: GgmlType) -> bool {
    matches!(
        wtype,
        GgmlType::F32
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_0
    )
}

fn derive_tensor_role(is_lm_head: bool, is_tied: bool) -> TensorRole {
    match (is_lm_head, is_tied) {
        (true, true) => TensorRole::TiedLmHead,
        (true, false) => TensorRole::LmHead,
        (false, _) => TensorRole::Generic,
    }
}

fn build_matrix_meta(
    weight_name: &str,
    dims: &[u64],
    wtype: GgmlType,
    config: &ModelConfig,
    is_lm_head: bool,
    is_tied: bool,
) -> GpuResult<WeightMeta> {
    if dims.len() < 2 {
        return Err(GpuError::InvalidWeightLayout {
            tensor: weight_name.to_string(),
            dims: dims.to_vec(),
            reason: "matrix weights must have at least 2 dimensions".to_string(),
        });
    }

    if !supports_gpu_matrix_type(wtype) {
        return Err(GpuError::UnsupportedWeightType {
            tensor: weight_name.to_string(),
            wtype,
        });
    }

    Ok(WeightMeta {
        wtype,
        dims: dims.to_vec(),
        needs_transpose: compute_transpose_flag(
            weight_name,
            dims,
            wtype,
            config,
            is_lm_head,
            is_tied,
        ),
        role: derive_tensor_role(is_lm_head, is_tied),
    })
}

fn upload_tensor_bytes(data: &[u8]) -> GpuResult<GpuBuffer> {
    let mut buf = GpuBuffer::alloc(data.len())?;
    buf.copy_from_host(data)?;
    Ok(buf)
}

fn upload_tensor_bytes_for_device(data: &[u8], device_id: i32) -> GpuResult<GpuBuffer> {
    let mut buf = GpuBuffer::alloc_for_device(data.len(), device_id)?;
    buf.copy_from_host(data)?;
    Ok(buf)
}

fn try_build_q4_0_gate_up_interleaved(
    gate_data: &[u8],
    gate_meta: &WeightMeta,
    up_data: &[u8],
    up_meta: &WeightMeta,
) -> Option<Vec<u8>> {
    const QK4_0: usize = 32;
    const Q4_0_BLOCK_SIZE: usize = 18;

    if gate_meta.wtype != GgmlType::Q4_0 || up_meta.wtype != GgmlType::Q4_0 {
        return None;
    }
    if gate_meta.dims != up_meta.dims || gate_meta.dims.len() < 2 {
        return None;
    }

    let n_rows = gate_meta.dims[0] as usize;
    let n_ff = gate_meta.dims[1] as usize;
    if n_rows == 0 || n_ff == 0 || n_rows % QK4_0 != 0 {
        return None;
    }

    let n_blocks_total = n_rows / QK4_0;
    let expected_len = n_ff
        .checked_mul(n_blocks_total)?
        .checked_mul(Q4_0_BLOCK_SIZE)?;
    if gate_data.len() != expected_len || up_data.len() != expected_len {
        return None;
    }

    let mut interleaved = Vec::with_capacity(expected_len * 2);
    for ff_idx in 0..n_ff {
        for block_idx in 0..n_blocks_total {
            let offset = (ff_idx * n_blocks_total + block_idx) * Q4_0_BLOCK_SIZE;
            interleaved.extend_from_slice(&gate_data[offset..offset + Q4_0_BLOCK_SIZE]);
            interleaved.extend_from_slice(&up_data[offset..offset + Q4_0_BLOCK_SIZE]);
        }
    }

    Some(interleaved)
}

fn try_build_q4_0_gate_up_interleaved_tile4(
    gate_data: &[u8],
    gate_meta: &WeightMeta,
    up_data: &[u8],
    up_meta: &WeightMeta,
) -> Option<Vec<u8>> {
    const QK4_0: usize = 32;
    const Q4_0_BLOCK_SIZE: usize = 18;
    const TILE_FF: usize = 4;

    if gate_meta.wtype != GgmlType::Q4_0 || up_meta.wtype != GgmlType::Q4_0 {
        return None;
    }
    if gate_meta.dims != up_meta.dims || gate_meta.dims.len() < 2 {
        return None;
    }

    let n_rows = gate_meta.dims[0] as usize;
    let n_ff = gate_meta.dims[1] as usize;
    if n_rows == 0 || n_ff == 0 || n_rows % QK4_0 != 0 || n_ff % TILE_FF != 0 {
        return None;
    }

    let n_blocks_total = n_rows / QK4_0;
    let expected_len = n_ff
        .checked_mul(n_blocks_total)?
        .checked_mul(Q4_0_BLOCK_SIZE)?;
    if gate_data.len() != expected_len || up_data.len() != expected_len {
        return None;
    }

    let mut interleaved = Vec::with_capacity(expected_len * 2);
    for ff_base in (0..n_ff).step_by(TILE_FF) {
        for block_idx in 0..n_blocks_total {
            for tile_ff in 0..TILE_FF {
                let ff_idx = ff_base + tile_ff;
                let offset = (ff_idx * n_blocks_total + block_idx) * Q4_0_BLOCK_SIZE;
                interleaved.extend_from_slice(&gate_data[offset..offset + Q4_0_BLOCK_SIZE]);
                interleaved.extend_from_slice(&up_data[offset..offset + Q4_0_BLOCK_SIZE]);
            }
        }
    }

    Some(interleaved)
}

// ── GPU Layer Weights ─────────────────────────────────────────────────────────────

/// Weights for a single transformer layer, stored in VRAM.
///
/// All weight tensors are stored in their native quantized format.
/// GPU kernels dequantize during inference.
pub struct GpuLayerWeights {
    /// RMS norm weights for attention (always F32)
    pub attn_norm: GpuBuffer,
    /// Query projection weights (quantized)
    pub attn_q: GpuBuffer,
    pub attn_q_meta: WeightMeta,
    /// Query bias (optional, always F32 if present)
    pub attn_q_bias: Option<GpuBuffer>,
    /// Key projection weights (quantized)
    pub attn_k: GpuBuffer,
    pub attn_k_meta: WeightMeta,
    /// Key bias (optional)
    pub attn_k_bias: Option<GpuBuffer>,
    /// Value projection weights (quantized)
    pub attn_v: GpuBuffer,
    pub attn_v_meta: WeightMeta,
    /// Value bias (optional)
    pub attn_v_bias: Option<GpuBuffer>,
    /// Attention output projection (quantized)
    pub attn_o: GpuBuffer,
    pub attn_o_meta: WeightMeta,
    /// RMS norm weights for FFN (always F32)
    pub ffn_norm: GpuBuffer,
    /// FFN gate projection (SwiGLU gate) (quantized)
    pub ffn_gate: GpuBuffer,
    pub ffn_gate_meta: WeightMeta,
    /// FFN up projection (quantized)
    pub ffn_up: GpuBuffer,
    pub ffn_up_meta: WeightMeta,
    /// Optional decode-friendly interleaved Q4_0 layout for fused gate/up kernels.
    pub ffn_gate_up_interleaved: Option<GpuBuffer>,
    /// Optional decode-friendly 4-column tiled Q4_0 layout for fused gate/up kernels.
    pub ffn_gate_up_interleaved_tile4: Option<GpuBuffer>,
    /// FFN down projection (quantized)
    pub ffn_down: GpuBuffer,
    pub ffn_down_meta: WeightMeta,
}

impl GpuLayerWeights {
    /// Load a single layer's weights from GGUF file into GPU memory.
    ///
    /// Returns error if any allocation or transfer fails.
    /// On error, all allocated memory is freed via Drop.
    pub fn load(file: &GgufFile, layer: usize, config: &ModelConfig) -> GpuResult<Self> {
        Self::load_for_device(file, layer, config, active_or_default_device_id())
    }

    pub fn load_for_device(
        file: &GgufFile,
        layer: usize,
        config: &ModelConfig,
        device_id: i32,
    ) -> GpuResult<Self> {
        // Helper to load weight into GPU buffer with metadata
        let load_weight = |name: &str| -> GpuResult<(GpuBuffer, WeightMeta)> {
            let t = file
                .tensor(name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor lookup failed: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", name),
                })?;

            let meta = build_matrix_meta(name, t.dims, t.ggml_type, config, false, false)?;
            let buf = upload_tensor_bytes_for_device(t.data, device_id)?;

            Ok((buf, meta))
        };

        // Helper to load weight with fallback names (for MoE models)
        let load_weight_fallback = |names: &[&str]| -> GpuResult<(GpuBuffer, WeightMeta)> {
            for name in names {
                match file.tensor(name) {
                    Ok(Some(t)) => {
                        let meta =
                            build_matrix_meta(name, t.dims, t.ggml_type, config, false, false)?;
                        let buf = upload_tensor_bytes_for_device(t.data, device_id)?;
                        return Ok((buf, meta));
                    }
                    Ok(None) => {}
                    Err(e) => {
                        return Err(GpuError::HipApiError {
                            code: -1,
                            description: format!("tensor lookup failed: {}", e),
                        });
                    }
                }
            }
            Err(GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: tried {:?}", names),
            })
        };

        // Helper to load F32 weight
        let load_f32 = |name: &str| -> GpuResult<GpuBuffer> {
            let t = file
                .tensor(name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor lookup failed: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", name),
                })?;

            let data = t.data;
            let mut buf = GpuBuffer::alloc_for_device(data.len(), device_id)?;
            buf.copy_from_host(data)?;
            Ok(buf)
        };

        // Helper to load optional F32 weight
        let load_f32_opt = |name: &str| -> GpuResult<Option<GpuBuffer>> {
            match file.tensor(name) {
                Ok(Some(t)) => {
                    let mut buf = GpuBuffer::alloc_for_device(t.data.len(), device_id)?;
                    buf.copy_from_host(t.data)?;
                    Ok(Some(buf))
                }
                Ok(None) => Ok(None),
                Err(_) => Ok(None), // Missing tensor is OK for optional weights
            }
        };

        // Load all weights - if any fail, this entire struct is dropped (RAII cleanup)
        let attn_norm = load_f32(&config.tensor_registry.resolve(TensorName::AttnNorm, layer))?;
        let (attn_q, attn_q_meta) =
            load_weight(&config.tensor_registry.resolve(TensorName::AttnQ, layer))?;
        let attn_q_bias = load_f32_opt(
            &config
                .tensor_registry
                .resolve_optional(TensorName::AttnQBias, layer)
                .unwrap_or_default(),
        )?;
        let (attn_k, attn_k_meta) =
            load_weight(&config.tensor_registry.resolve(TensorName::AttnK, layer))?;
        let attn_k_bias = load_f32_opt(
            &config
                .tensor_registry
                .resolve_optional(TensorName::AttnKBias, layer)
                .unwrap_or_default(),
        )?;
        let (attn_v, attn_v_meta) =
            load_weight(&config.tensor_registry.resolve(TensorName::AttnV, layer))?;
        let attn_v_bias = load_f32_opt(
            &config
                .tensor_registry
                .resolve_optional(TensorName::AttnVBias, layer)
                .unwrap_or_default(),
        )?;
        let (attn_o, attn_o_meta) = load_weight(
            &config
                .tensor_registry
                .resolve(TensorName::AttnOutput, layer),
        )?;
        let ffn_norm = load_f32(&config.tensor_registry.resolve(TensorName::FfnNorm, layer))?;

        // For MoE models, try _exps tensors first, then fall back to standard names
        let ffn_gate_name = config.tensor_registry.resolve(TensorName::FfnGate, layer);
        let ffn_up_name = config.tensor_registry.resolve(TensorName::FfnUp, layer);
        let ffn_down_name = config.tensor_registry.resolve(TensorName::FfnDown, layer);

        let (ffn_gate_name_used, ffn_gate, ffn_gate_meta) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let ffn_gate_exps_name = config
                    .tensor_registry
                    .resolve(TensorName::FfnGateExps, layer);
                let (buf, meta) = load_weight_fallback(&[&ffn_gate_exps_name, &ffn_gate_name])?;
                let chosen = if file.has_tensor(&ffn_gate_exps_name) {
                    ffn_gate_exps_name
                } else {
                    ffn_gate_name.clone()
                };
                (chosen, buf, meta)
            } else {
                let (buf, meta) = load_weight(&ffn_gate_name)?;
                (ffn_gate_name.clone(), buf, meta)
            };

        let (ffn_up_name_used, ffn_up, ffn_up_meta) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let ffn_up_exps_name = config.tensor_registry.resolve(TensorName::FfnUpExps, layer);
                let (buf, meta) = load_weight_fallback(&[&ffn_up_exps_name, &ffn_up_name])?;
                let chosen = if file.has_tensor(&ffn_up_exps_name) {
                    ffn_up_exps_name
                } else {
                    ffn_up_name.clone()
                };
                (chosen, buf, meta)
            } else {
                let (buf, meta) = load_weight(&ffn_up_name)?;
                (ffn_up_name.clone(), buf, meta)
            };

        let ffn_gate_up_interleaved = match (
            file.tensor(&ffn_gate_name_used).ok().and_then(|t| t),
            file.tensor(&ffn_up_name_used).ok().and_then(|t| t),
        ) {
            (Some(gate_t), Some(up_t)) => try_build_q4_0_gate_up_interleaved(
                gate_t.data,
                &ffn_gate_meta,
                up_t.data,
                &ffn_up_meta,
            )
            .map(|bytes| upload_tensor_bytes_for_device(&bytes, device_id))
            .transpose()?,
            _ => None,
        };
        let ffn_gate_up_interleaved_tile4 = match (
            file.tensor(&ffn_gate_name_used).ok().and_then(|t| t),
            file.tensor(&ffn_up_name_used).ok().and_then(|t| t),
        ) {
            (Some(gate_t), Some(up_t)) => try_build_q4_0_gate_up_interleaved_tile4(
                gate_t.data,
                &ffn_gate_meta,
                up_t.data,
                &ffn_up_meta,
            )
            .map(|bytes| upload_tensor_bytes_for_device(&bytes, device_id))
            .transpose()?,
            _ => None,
        };

        let (ffn_down, ffn_down_meta) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let ffn_down_exps_name = config
                    .tensor_registry
                    .resolve(TensorName::FfnDownExps, layer);
                load_weight_fallback(&[&ffn_down_exps_name, &ffn_down_name])?
            } else {
                load_weight(&ffn_down_name)?
            };

        Ok(Self {
            attn_norm,
            attn_q,
            attn_q_meta,
            attn_q_bias,
            attn_k,
            attn_k_meta,
            attn_k_bias,
            attn_v,
            attn_v_meta,
            attn_v_bias,
            attn_o,
            attn_o_meta,
            ffn_norm,
            ffn_gate,
            ffn_gate_meta,
            ffn_up,
            ffn_up_meta,
            ffn_gate_up_interleaved,
            ffn_gate_up_interleaved_tile4,
            ffn_down,
            ffn_down_meta,
        })
    }

    fn estimate_vram_usage_from_file(
        file: &GgufFile,
        layer: usize,
        config: &ModelConfig,
    ) -> GpuResult<usize> {
        let tensor_bytes = |name: &str| -> GpuResult<usize> {
            let t = file
                .tensor(name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor lookup failed: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", name),
                })?;
            Ok(t.data.len())
        };
        let tensor_bytes_optional = |name: &str| -> GpuResult<usize> {
            if name.is_empty() {
                return Ok(0);
            }
            Ok(file
                .tensor(name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor lookup failed: {}", e),
                })?
                .map(|t| t.data.len())
                .unwrap_or(0))
        };
        let choose_ffn_tensor = |primary: &str, fallback: &str| -> GpuResult<(String, usize)> {
            if file.has_tensor(primary) {
                Ok((primary.to_string(), tensor_bytes(primary)?))
            } else {
                Ok((fallback.to_string(), tensor_bytes(fallback)?))
            }
        };

        let attn_norm_name = config.tensor_registry.resolve(TensorName::AttnNorm, layer);
        let attn_q_name = config.tensor_registry.resolve(TensorName::AttnQ, layer);
        let attn_k_name = config.tensor_registry.resolve(TensorName::AttnK, layer);
        let attn_v_name = config.tensor_registry.resolve(TensorName::AttnV, layer);
        let attn_o_name = config
            .tensor_registry
            .resolve(TensorName::AttnOutput, layer);
        let ffn_norm_name = config.tensor_registry.resolve(TensorName::FfnNorm, layer);
        let ffn_gate_name = config.tensor_registry.resolve(TensorName::FfnGate, layer);
        let ffn_up_name = config.tensor_registry.resolve(TensorName::FfnUp, layer);
        let ffn_down_name = config.tensor_registry.resolve(TensorName::FfnDown, layer);

        let (ffn_gate_name_used, ffn_gate_bytes) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let primary = config
                    .tensor_registry
                    .resolve(TensorName::FfnGateExps, layer);
                choose_ffn_tensor(&primary, &ffn_gate_name)?
            } else {
                (ffn_gate_name.clone(), tensor_bytes(&ffn_gate_name)?)
            };
        let (ffn_up_name_used, ffn_up_bytes) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let primary = config.tensor_registry.resolve(TensorName::FfnUpExps, layer);
                choose_ffn_tensor(&primary, &ffn_up_name)?
            } else {
                (ffn_up_name.clone(), tensor_bytes(&ffn_up_name)?)
            };
        let ffn_down_bytes = if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE)
        {
            let primary = config
                .tensor_registry
                .resolve(TensorName::FfnDownExps, layer);
            choose_ffn_tensor(&primary, &ffn_down_name)?.1
        } else {
            tensor_bytes(&ffn_down_name)?
        };

        let gate_tensor = file
            .tensor(&ffn_gate_name_used)
            .map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("tensor lookup failed: {}", e),
            })?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: {}", ffn_gate_name_used),
            })?;
        let up_tensor = file
            .tensor(&ffn_up_name_used)
            .map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("tensor lookup failed: {}", e),
            })?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: {}", ffn_up_name_used),
            })?;
        let ffn_gate_meta = build_matrix_meta(
            &ffn_gate_name_used,
            gate_tensor.dims,
            gate_tensor.ggml_type,
            config,
            false,
            false,
        )?;
        let ffn_up_meta = build_matrix_meta(
            &ffn_up_name_used,
            up_tensor.dims,
            up_tensor.ggml_type,
            config,
            false,
            false,
        )?;
        let interleaved_bytes = try_build_q4_0_gate_up_interleaved(
            gate_tensor.data,
            &ffn_gate_meta,
            up_tensor.data,
            &ffn_up_meta,
        )
        .map_or(0, |bytes| bytes.len());
        let interleaved_tile4_bytes = try_build_q4_0_gate_up_interleaved_tile4(
            gate_tensor.data,
            &ffn_gate_meta,
            up_tensor.data,
            &ffn_up_meta,
        )
        .map_or(0, |bytes| bytes.len());

        Ok(tensor_bytes(&attn_norm_name)?
            + tensor_bytes(&attn_q_name)?
            + tensor_bytes_optional(
                &config
                    .tensor_registry
                    .resolve_optional(TensorName::AttnQBias, layer)
                    .unwrap_or_default(),
            )?
            + tensor_bytes(&attn_k_name)?
            + tensor_bytes_optional(
                &config
                    .tensor_registry
                    .resolve_optional(TensorName::AttnKBias, layer)
                    .unwrap_or_default(),
            )?
            + tensor_bytes(&attn_v_name)?
            + tensor_bytes_optional(
                &config
                    .tensor_registry
                    .resolve_optional(TensorName::AttnVBias, layer)
                    .unwrap_or_default(),
            )?
            + tensor_bytes(&attn_o_name)?
            + tensor_bytes(&ffn_norm_name)?
            + ffn_gate_bytes
            + ffn_up_bytes
            + interleaved_bytes
            + interleaved_tile4_bytes
            + ffn_down_bytes)
    }

    /// Estimate total VRAM usage for this layer in bytes.
    ///
    /// This is a conservative estimate that sums all buffer sizes.
    pub fn estimate_vram_usage(&self) -> usize {
        let mut total = 0;

        // Mandatory buffers
        total += self.attn_norm.size();
        total += self.attn_q.size();
        total += self.attn_k.size();
        total += self.attn_v.size();
        total += self.attn_o.size();
        total += self.ffn_norm.size();
        total += self.ffn_gate.size();
        total += self.ffn_up.size();
        total += self.ffn_down.size();

        // Optional buffers
        if let Some(ref buf) = self.attn_q_bias {
            total += buf.size();
        }
        if let Some(ref buf) = self.attn_k_bias {
            total += buf.size();
        }
        if let Some(ref buf) = self.attn_v_bias {
            total += buf.size();
        }
        if let Some(ref buf) = self.ffn_gate_up_interleaved {
            total += buf.size();
        }
        if let Some(ref buf) = self.ffn_gate_up_interleaved_tile4 {
            total += buf.size();
        }

        total
    }
}

// ── GPU Model Weights ─────────────────────────────────────────────────────────────

/// All weights for a transformer model, stored in VRAM.
///
/// Holds token embeddings, all layer weights, output norm, and LM head.
pub struct GpuModelWeights {
    /// Per-layer weights (all in VRAM)
    pub layers: Vec<GpuLayerWeights>,
    /// Token embedding matrix (quantized, in VRAM)
    pub token_emb: GpuBuffer,
    pub token_emb_meta: WeightMeta,
    /// Final RMS norm weights (F32, in VRAM)
    pub output_norm: GpuBuffer,
    /// Language model head / output projection (quantized, in VRAM)
    pub lm_head: GpuBuffer,
    pub lm_head_meta: WeightMeta,
    /// Whether LM head is tied to token embeddings
    pub lm_head_tied: bool,
    /// Cached pointer-mix used by decode-graph key construction.
    decode_binding_tag: u64,
}

impl GpuModelWeights {
    /// Load all weights from GGUF file into GPU memory.
    ///
    /// Returns error if any allocation or transfer fails.
    /// On error, all allocated memory is freed via Drop.
    /// Includes cumulative VRAM tracking to prevent model from exceeding safe limits.
    pub fn load(file: &GgufFile, config: &ModelConfig) -> GpuResult<Self> {
        Self::load_for_device(file, config, active_or_default_device_id())
    }

    pub fn load_for_device(
        file: &GgufFile,
        config: &ModelConfig,
        device_id: i32,
    ) -> GpuResult<Self> {
        let n = config.num_layers;
        ffi::hip_set_device(device_id)?;
        let budget = query_vram_budget(device_id)?;

        // Helper to load tensor into GPU buffer without VRAM tracking (done separately)
        fn load_tensor_no_track(
            file: &GgufFile,
            name: &str,
            config: &ModelConfig,
            is_lm_head: bool,
            is_tied: bool,
            device_id: i32,
        ) -> GpuResult<(GpuBuffer, WeightMeta)> {
            let t = file
                .tensor(name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor lookup failed: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", name),
                })?;

            let meta = build_matrix_meta(name, t.dims, t.ggml_type, config, is_lm_head, is_tied)?;
            let buf = upload_tensor_bytes_for_device(t.data, device_id)?;

            Ok((buf, meta))
        }

        let mut estimated_vram_used = 0usize;

        // Load token embeddings using registry
        let token_emb_name = config.tensor_registry.resolve(TensorName::TokenEmb, 0);
        let token_emb_view = file
            .tensor(&token_emb_name)
            .map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("tensor lookup failed: {}", e),
            })?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: {}", token_emb_name),
            })?;
        check_model_load_headroom(budget, estimated_vram_used, token_emb_view.data.len())?;
        let (token_emb, token_emb_meta) =
            load_tensor_no_track(file, &token_emb_name, config, false, false, device_id)?;
        estimated_vram_used += token_emb.size();

        // Load output norm using registry
        let output_norm_name = config.tensor_registry.resolve(TensorName::OutputNorm, 0);
        let output_norm_view = file
            .tensor(&output_norm_name)
            .map_err(|_| GpuError::WeightTransferFailed { layer: 0 })?
            .ok_or_else(|| GpuError::WeightTransferFailed { layer: 0 })?;

        check_model_load_headroom(budget, estimated_vram_used, output_norm_view.data.len())?;
        let mut output_norm = GpuBuffer::alloc_for_device(output_norm_view.data.len(), device_id)?;
        output_norm.copy_from_host(output_norm_view.data)?;
        estimated_vram_used += output_norm.size();

        // LM head: use lm_head.weight if present, otherwise tie to embeddings
        let lm_head_name = config.tensor_registry.resolve(TensorName::LmHead, 0);
        let (lm_head, lm_head_meta, lm_head_tied) = if file.has_tensor(&lm_head_name) {
            let lm_head_view = file
                .tensor(&lm_head_name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor lookup failed: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", lm_head_name),
                })?;
            check_model_load_headroom(budget, estimated_vram_used, lm_head_view.data.len())?;
            let (buf, meta) =
                load_tensor_no_track(file, &lm_head_name, config, true, false, device_id)?;
            estimated_vram_used += buf.size();
            (buf, meta, false)
        } else {
            // Materialize a second GPU buffer for the tied head.
            // This keeps the decode path simple while preserving explicit tied metadata.
            check_model_load_headroom(budget, estimated_vram_used, token_emb_view.data.len())?;
            let (buf, _) =
                load_tensor_no_track(file, &token_emb_name, config, false, false, device_id)?;
            estimated_vram_used += buf.size();

            let tied_meta = build_matrix_meta(
                &lm_head_name,
                &token_emb_meta.dims,
                token_emb_meta.wtype,
                config,
                true,
                true,
            )?;
            (buf, tied_meta, true)
        };

        // Load all layers
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            eprintln!("[GPU weights] Loading layer {}/{}", i + 1, n);
            let layer_vram = GpuLayerWeights::estimate_vram_usage_from_file(file, i, config)?;
            check_model_load_headroom(budget, estimated_vram_used, layer_vram)?;
            let layer = GpuLayerWeights::load_for_device(file, i, config, device_id)?;
            estimated_vram_used += layer_vram;

            layers.push(layer);
        }

        let decode_binding_tag = compute_model_binding_tag(&layers, &output_norm, &lm_head);

        eprintln!(
            "[GPU weights] Total estimated VRAM usage: {} MB",
            estimated_vram_used / (1024 * 1024)
        );

        Ok(Self {
            layers,
            token_emb,
            token_emb_meta,
            output_norm,
            lm_head,
            lm_head_meta,
            lm_head_tied,
            decode_binding_tag,
        })
    }

    /// Get weights for a specific layer.
    pub fn layer(&self, i: usize) -> &GpuLayerWeights {
        &self.layers[i]
    }

    /// Cached pointer-mix used by decode-graph key construction.
    #[inline]
    pub fn binding_tag(&self) -> u64 {
        self.decode_binding_tag
    }

    /// Check if any weights use Q6_K quantization (incompatible with HIP graph capture)
    pub fn uses_q6_k_quantization(&self) -> bool {
        // Check token embedding
        if self.token_emb_meta.wtype == GgmlType::Q6_K {
            return true;
        }

        // Check output layer
        if self.lm_head_meta.wtype == GgmlType::Q6_K {
            return true;
        }

        // Check all layers
        for layer in &self.layers {
            if layer.attn_q_meta.wtype == GgmlType::Q6_K
                || layer.attn_k_meta.wtype == GgmlType::Q6_K
                || layer.attn_v_meta.wtype == GgmlType::Q6_K
                || layer.attn_o_meta.wtype == GgmlType::Q6_K
                || layer.ffn_gate_meta.wtype == GgmlType::Q6_K
                || layer.ffn_up_meta.wtype == GgmlType::Q6_K
                || layer.ffn_down_meta.wtype == GgmlType::Q6_K
            {
                return true;
            }
        }

        false
    }
}

#[inline]
fn mix_binding_tag(tag: u64, ptr: usize) -> u64 {
    tag.rotate_left(13) ^ (ptr as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn gpu_layer_weights_binding_tag(layer: &GpuLayerWeights) -> u64 {
    let mut tag = 0u64;
    tag = mix_binding_tag(tag, layer.attn_norm.as_ptr() as usize);
    tag = mix_binding_tag(tag, layer.attn_q.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .attn_q_bias
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(tag, layer.attn_k.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .attn_k_bias
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(tag, layer.attn_v.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .attn_v_bias
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(tag, layer.attn_o.as_ptr() as usize);
    tag = mix_binding_tag(tag, layer.ffn_norm.as_ptr() as usize);
    tag = mix_binding_tag(tag, layer.ffn_gate.as_ptr() as usize);
    tag = mix_binding_tag(tag, layer.ffn_up.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .ffn_gate_up_interleaved
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(
        tag,
        layer
            .ffn_gate_up_interleaved_tile4
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    mix_binding_tag(tag, layer.ffn_down.as_ptr() as usize)
}

fn compute_model_binding_tag(
    layers: &[GpuLayerWeights],
    output_norm: &GpuBuffer,
    lm_head: &GpuBuffer,
) -> u64 {
    let mut tag = 0u64;
    tag = mix_binding_tag(tag, output_norm.as_ptr() as usize);
    tag = mix_binding_tag(tag, lm_head.as_ptr() as usize);
    for layer in layers {
        tag ^= gpu_layer_weights_binding_tag(layer);
    }
    tag
}

#[cfg(test)]
mod matrix_meta_tests {
    use super::*;
    use crate::config::{AttentionLayout, TensorNameRegistry, TensorNamingScheme};

    fn make_test_config() -> ModelConfig {
        ModelConfig {
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
            rope_neox: false,
            use_attention_bias: false,
            attention_layout: AttentionLayout::SplitQkv,
            architecture: "test".to_string(),
            tensor_registry: TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf),
        }
    }

    #[test]
    fn explicit_lm_head_matches_cpu_transpose_rule() {
        let config = make_test_config();
        let meta = build_matrix_meta(
            "output.weight",
            &[32000, 1024],
            GgmlType::Q4_0,
            &config,
            true,
            false,
        )
        .unwrap();

        assert!(!meta.needs_transpose);
        assert_eq!(meta.role, TensorRole::LmHead);
    }

    #[test]
    fn tied_lm_head_is_marked_transposed() {
        let config = make_test_config();
        let meta = build_matrix_meta(
            "output.weight",
            &[32000, 1024],
            GgmlType::Q4_0,
            &config,
            true,
            true,
        )
        .unwrap();

        assert!(meta.needs_transpose);
        assert_eq!(meta.role, TensorRole::TiedLmHead);
    }

    #[test]
    fn unsupported_matrix_type_is_rejected() {
        let config = make_test_config();
        let err = build_matrix_meta(
            "blk.0.attn_q.weight",
            &[1024, 1024],
            GgmlType::Q6_K,
            &config,
            false,
            false,
        )
        .unwrap_err();

        assert!(matches!(err, GpuError::UnsupportedWeightType { .. }));
    }

    #[test]
    fn matrix_weights_require_two_dims() {
        let config = make_test_config();
        let err = build_matrix_meta(
            "output.weight",
            &[32000],
            GgmlType::Q4_0,
            &config,
            true,
            false,
        )
        .unwrap_err();

        assert!(matches!(err, GpuError::InvalidWeightLayout { .. }));
    }
}
