//! GGUF GPU Upload and Caching
//!
//! This module handles GPU tensor upload operations:
//! - Single tensor on-demand loading
/// - Async batch loading with parallel dequantization
/// - Arena-based memory allocation for stability
/// - GPU cache management

use crate::backend::hip_backend::{AsyncLoader, DeviceTensor, HipBackend};
use crate::loader::gguf::tensor_data::{dequantize_q4_0, dequantize_q4_k, dequantize_q6_k, dequantize_q8_0};
use crate::loader::{GgufLoader, GgufTensor, GgufTensorType, TensorShape};
use crate::memory::{MemoryCalculator, ModelWeightArena};
use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, RwLock};

/// Load a single tensor to GPU with dequantization
///
/// This is the implementation for `GgufLoader::load_tensor_to_gpu`
pub fn load_tensor_to_gpu_impl(
    loader: &GgufLoader,
    name: &str,
    backend: &HipBackend,
    offset: u64,
    size: usize,
    shape: &TensorShape,
    tensor_type: GgufTensorType,
) -> Result<Arc<DeviceTensor>> {
    // Read tensor data from memory-mapped file (zero-copy)
    let mmap = loader.mmap.as_ref().ok_or_else(|| {
        anyhow!("Memory mapping not available - loader not initialized correctly")
    })?;

    let tensor_bytes = mmap
        .get_slice(offset, size)
        .map_err(|e| anyhow!("Failed to read tensor '{}' from mmap: {}", name, e))?;

    // Check if this is a Q4_0, Q4_K, or Q6_K tensor that can use GPU dequantization
    // Upload quantized bytes directly, dequantize on GPU
    let needs_transpose = GgufLoader::is_fused_qkv_weight(name);
    if (tensor_type == GgufTensorType::Q4_0
        || tensor_type == GgufTensorType::Q4_K
        || tensor_type == GgufTensorType::Q6_K) && !needs_transpose {
        // GPU dequantization path for Q4_0/Q4_K/Q6_K
        return gpu_dequantize_upload(backend, name, tensor_bytes, shape, tensor_type, loader);
    }

    // CPU dequantization path (for types without GPU dequantization or with transpose requirement)
    cpu_dequantize_upload(
        loader, name, backend, tensor_bytes, shape, tensor_type, offset, size,
    )
}

/// GPU dequantization path for Q4_0/Q4_K/Q6_K
#[cfg(feature = "rocm")]
fn gpu_dequantize_upload(
    backend: &HipBackend,
    name: &str,
    tensor_bytes: &[u8],
    shape: &TensorShape,
    tensor_type: GgufTensorType,
    loader: &GgufLoader,
) -> Result<Arc<DeviceTensor>> {
    use crate::ggml::hip_backend::ops::q4_0_dequant::dequantize_q4_0_kernel_cached;
    use crate::ggml::hip_backend::ops::q4_k_dequant::dequantize_q4_k_gpu_kernel;
    use crate::ggml::hip_backend::ops::q6_k_dequant::dequantize_q6_k_gpu_kernel;

    // Allocate output buffer for FP32 data
    let num_elements = shape.total_elements();
    let output_buffer = backend
        .allocate_buffer(num_elements * 4)
        .map_err(|e| anyhow!("Failed to allocate output buffer for '{}': {}", name, e))?;

    // Call GPU dequantization
    match tensor_type {
        GgufTensorType::Q4_0 => {
            dequantize_q4_0_kernel_cached(backend, tensor_bytes, &output_buffer, num_elements)
                .map_err(|e| anyhow!("GPU dequantization failed for '{}': {}. If GPU is unavailable, use CPU tensors instead.", name, e))?;
        }
        GgufTensorType::Q4_K => {
            dequantize_q4_k_gpu_kernel(backend, tensor_bytes, &output_buffer, num_elements)
                .map_err(|e| anyhow!("GPU dequantization failed for '{}': {}. If GPU is unavailable, use CPU tensors instead.", name, e))?;
        }
        GgufTensorType::Q6_K => {
            dequantize_q6_k_gpu_kernel(backend, tensor_bytes, &output_buffer, num_elements)
                .map_err(|e| anyhow!("GPU dequantization failed for '{}': {}. If GPU is unavailable, use CPU tensors instead.", name, e))?;
        }
        _ => unreachable!(),
    }

    eprintln!(
        ">>> load_tensor_to_gpu: '{}' GPU dequantization complete, {} f32 values ({} MB)",
        name,
        num_elements,
        num_elements * 4 / 1024 / 1024
    );

    // Wrap buffer in DeviceTensor
    let device_tensor = DeviceTensor::from_buffer(backend, output_buffer, shape.clone())
        .map_err(|e| anyhow!("Failed to create DeviceTensor for '{}': {}", name, e))?;

    let device_tensor_arc = Arc::new(device_tensor);

    eprintln!(
        ">>> load_tensor_to_gpu: '{}' uploaded to GPU successfully",
        name
    );

    // Cache the result
    {
        let mut cache = loader
            .gpu_cache
            .write()
            .map_err(|e| anyhow!("GPU cache write lock poisoned: {}", e))?;
        cache.insert(name.to_string(), device_tensor_arc.clone());
    }

    tracing::debug!(
        "Loaded tensor '{}' to GPU and cached ({} bytes)",
        name,
        tensor_bytes.len()
    );
    Ok(device_tensor_arc)
}

/// GPU dequantization path (no ROCm feature - should not be called)
#[cfg(not(feature = "rocm"))]
fn gpu_dequantize_upload(
    _backend: &HipBackend,
    _name: &str,
    _tensor_bytes: &[u8],
    _shape: &TensorShape,
    _tensor_type: GgufTensorType,
    _loader: &GgufLoader,
) -> Result<Arc<DeviceTensor>> {
    Err(anyhow!("GPU dequantization not available (rocm feature disabled)"))
}

/// CPU dequantization path for other tensor types
fn cpu_dequantize_upload(
    loader: &GgufLoader,
    name: &str,
    backend: &HipBackend,
    tensor_bytes: &[u8],
    shape: &TensorShape,
    tensor_type: GgufTensorType,
    offset: u64,
    size: usize,
) -> Result<Arc<DeviceTensor>> {
    // Dequantize based on tensor type
    let mut f32_data: Vec<f32> = match tensor_type {
        GgufTensorType::F32 => tensor_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        GgufTensorType::F16 => tensor_bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect(),
        GgufTensorType::Q8_0 => {
            let temp_tensor = GgufTensor {
                name: name.to_string(),
                shape: shape.clone(),
                tensor_type,
                quant_type: tensor_type.to_string().to_string(),
                offset,
                data: tensor_bytes.to_vec(),
            };
            dequantize_q8_0(&temp_tensor)?
        }
        GgufTensorType::Q4_0 => {
            let temp_tensor = GgufTensor {
                name: name.to_string(),
                shape: shape.clone(),
                tensor_type,
                quant_type: tensor_type.to_string().to_string(),
                offset,
                data: tensor_bytes.to_vec(),
            };
            dequantize_q4_0(&temp_tensor)?
        }
        GgufTensorType::Q4_K => {
            let temp_tensor = GgufTensor {
                name: name.to_string(),
                shape: shape.clone(),
                tensor_type,
                quant_type: tensor_type.to_string().to_string(),
                offset,
                data: tensor_bytes.to_vec(),
            };
            dequantize_q4_k(&temp_tensor)?
        }
        GgufTensorType::Q6_K => {
            let temp_tensor = GgufTensor {
                name: name.to_string(),
                shape: shape.clone(),
                tensor_type,
                quant_type: tensor_type.to_string().to_string(),
                offset,
                data: tensor_bytes.to_vec(),
            };
            dequantize_q6_k(&temp_tensor)?
        }
        GgufTensorType::Q2_K | GgufTensorType::Q3_K | GgufTensorType::Q5_K => {
            return Err(anyhow!(
                "K-quant type {:?} not yet implemented for tensor '{}'",
                tensor_type,
                name
            ));
        }
        GgufTensorType::Mxfp4 | GgufTensorType::Mxfp6E2m3 | GgufTensorType::Mxfp6E3m2 => {
            return Err(anyhow!(
                "MXFP dequantization not yet implemented for tensor '{}'",
                name
            ));
        }
    };

    eprintln!(
        ">>> load_tensor_to_gpu: '{}' dequantization complete, {} f32 values ({} MB)",
        name,
        f32_data.len(),
        f32_data.len() * 4 / 1024 / 1024
    );

    let mut shape = shape.clone();

    // Embedding weights stay in GGUF layout; GetRows handles layout at execution time.

    // Normalize fused QKV weight orientation: some models store [3*hidden, hidden]
    // while the matmul path expects [hidden, 3*hidden].
    if GgufLoader::is_fused_qkv_weight(name) {
        let dims = shape.dims();
        if dims.len() == 2 {
            let (rows, cols) = (dims[0], dims[1]);
            if rows == cols * 3 {
                eprintln!(
                    ">>> load_tensor_to_gpu: Transposing fused QKV weight '{}' from [{}, {}] to [{}, {}]",
                    name, rows, cols, cols, rows
                );
                f32_data = transpose_f32_matrix(&f32_data, rows, cols);
                shape = TensorShape::from_dims(&[cols, rows]);
            } else if cols != rows * 3 {
                tracing::warn!(
                    "Unexpected fused QKV shape [{} x {}] for '{}'; expected one dimension to be 3x the other",
                    rows,
                    cols,
                    name
                );
            }
        }
    }

    // Upload to GPU
    let device_tensor = DeviceTensor::from_host_vec(backend, f32_data, shape)
        .map_err(|e| anyhow!("Failed to upload tensor '{}' to GPU: {}", name, e))?;

    let device_tensor_arc = Arc::new(device_tensor);

    eprintln!(
        ">>> load_tensor_to_gpu: '{}' uploaded to GPU successfully",
        name
    );

    // Cache the result
    {
        let mut cache = loader
            .gpu_cache
            .write()
            .map_err(|e| anyhow!("GPU cache write lock poisoned: {}", e))?;
        cache.insert(name.to_string(), device_tensor_arc.clone());
    }

    tracing::debug!(
        "Loaded tensor '{}' to GPU and cached ({} bytes)",
        name,
        size
    );
    Ok(device_tensor_arc)
}

fn transpose_f32_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; data.len()];
    for r in 0..rows {
        let row_offset = r * cols;
        for c in 0..cols {
            transposed[c * rows + r] = data[row_offset + c];
        }
    }
    transposed
}

/// Parallel dequantization phase for async loading
///
/// Phase A: Parallel Dequantization (Rayon)
/// All tensors dequantized in parallel on CPU
pub fn parallel_dequantization_phase(
    loader: &GgufLoader,
    tensor_names: &[String],
    dequantized_data: &Mutex<BTreeMap<String, Vec<f32>>>,
    tensor_shapes: &Mutex<BTreeMap<String, Vec<usize>>>,
) -> Result<()> {
    tracing::info!(
        "load_to_gpu_async: Phase A - Parallel dequantization of {} tensors",
        tensor_names.len()
    );

    // Process all tensors in parallel (Rayon)
    tensor_names.par_iter().for_each(|name| {
        // Check GPU cache first
        {
            let cache = match loader.gpu_cache.read() {
                Ok(guard) => guard,
                Err(_) => return, // Skip if cache is poisoned
            };
            if cache.contains_key(name) {
                // Already loaded, skip
                return;
            }
        }

        // Get lazy tensor metadata
        let lazy = match loader.lazy_tensors.get(name) {
            Some(l) => l,
            None => return,
        };

        let (offset, size, shape, tensor_type) = match lazy {
            crate::loader::lazy_tensor::LazyTensor::Unloaded {
                offset,
                size,
                shape,
                tensor_type,
                ..
            } => (*offset, *size, shape.clone(), *tensor_type),
            crate::loader::lazy_tensor::LazyTensor::Gpu { .. } => return,
        };

        // Read tensor data from memory-mapped file
        let mmap = match &loader.mmap {
            Some(m) => m,
            None => return,
        };

        let tensor_bytes = match mmap.get_slice(offset, size) {
            Ok(slice) => slice,
            Err(_) => return,
        };

        // Dequantize based on tensor type (using Rayon-parallelized methods)
        let f32_data: Vec<f32> = match tensor_type {
            GgufTensorType::F32 => tensor_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
            GgufTensorType::F16 => tensor_bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect(),
            GgufTensorType::Q8_0 => {
                let temp_tensor = GgufTensor {
                    name: name.clone(),
                    shape: TensorShape::from_dims(&shape),
                    tensor_type,
                    quant_type: tensor_type.to_string().to_string(),
                    offset,
                    data: tensor_bytes.to_vec(),
                };
                match dequantize_q8_0(&temp_tensor) {
                    Ok(data) => data,
                    Err(_) => return,
                }
            }
            GgufTensorType::Q4_0 => {
                let temp_tensor = GgufTensor {
                    name: name.clone(),
                    shape: TensorShape::from_dims(&shape),
                    tensor_type,
                    quant_type: tensor_type.to_string().to_string(),
                    offset,
                    data: tensor_bytes.to_vec(),
                };
                match dequantize_q4_0(&temp_tensor) {
                    Ok(data) => data,
                    Err(_) => return,
                }
            }
            _ => {
                // For other types, skip or return empty
                return;
            }
        };

        // Store dequantized data and shape
        {
            let mut data_guard = dequantized_data.lock().unwrap_or_else(|e| {
                panic!(
                    "Failed to lock dequantized_data for tensor '{}': {}",
                    name, e
                )
            });
            let mut shape_guard = tensor_shapes.lock().unwrap_or_else(|e| {
                panic!("Failed to lock tensor_shapes for tensor '{}': {}", name, e)
            });
            data_guard.insert(name.clone(), f32_data);
            shape_guard.insert(name.clone(), shape);
        }
    });

    Ok(())
}

/// Arena-based upload phase for async loading
///
/// Phase A-1: Memory Requirements Calculation
/// Phase B-1: Create memory arena
/// Phase B-2: Upload to arena
/// Phase C: Update GPU cache
pub fn arena_upload_phase(
    backend: &HipBackend,
    async_loader: &AsyncLoader,
    dequantized_data: &Mutex<BTreeMap<String, Vec<f32>>>,
    tensor_shapes: &Mutex<BTreeMap<String, Vec<usize>>>,
    gpu_cache: &Arc<RwLock<HashMap<String, Arc<DeviceTensor>>>>,
) -> Result<HashMap<String, DeviceTensor>> {
    use std::sync::Mutex;

    // Phase A-1: Memory Requirements Calculation (Phase 22-02)
    tracing::info!("load_to_gpu_async: Phase A-1 - Memory requirements calculation");

    let shapes = tensor_shapes
        .lock()
        .map_err(|e| anyhow!("Failed to lock tensor_shapes for memory calculation: {}", e))?;

    let mut memory_calc = MemoryCalculator::new();
    for (name, shape) in shapes.iter() {
        let elements: usize = shape.iter().product();
        memory_calc.add_tensor(name.clone(), elements, std::mem::size_of::<f32>());
    }

    let total_needed_bytes = memory_calc.total_bytes();
    let tensor_count = memory_calc.tensor_count();
    tracing::info!(
        "Memory calculation: {} tensors, {} MB needed (base)",
        tensor_count,
        total_needed_bytes / 1024 / 1024
    );

    // Query GPU memory and check if sufficient
    let (has_enough, free_mb, needed_mb) = backend
        .check_memory_for_model(total_needed_bytes)
        .map_err(|e| anyhow!("Failed to query GPU memory: {}", e))?;

    if !has_enough {
        bail!(
            "Insufficient GPU memory for model loading\n\
             Required: {} MB (including 10%% safety margin)\n\
             Available: {} MB\n\
             Deficit: {} MB\n\
             \n\
             Suggestions:\n\
             - Use a smaller model\n\
             - Reduce context length\n\
             - Close other GPU-intensive applications",
            needed_mb, free_mb, needed_mb.saturating_sub(free_mb)
        );
    }

    tracing::info!(
        "GPU memory check passed: {} MB available for {} MB needed (safety margin included)",
        free_mb, needed_mb
    );

    // Release the shapes lock before Phase B
    drop(shapes);

    // Phase B-1: Create memory arena for all tensor weights
    tracing::info!("load_to_gpu_async: Phase B-1 - Creating memory arena");
    let mut arena = ModelWeightArena::new(total_needed_bytes, backend)
        .map_err(|e| anyhow!("Failed to create memory arena: {}", e))?;

    tracing::info!(
        "Created arena: {} MB capacity, alignment {} bytes",
        arena.capacity() / 1024 / 1024,
        ModelWeightArena::DEFAULT_ALIGNMENT
    );

    // Phase B-2: Upload all tensors to arena using offset-based allocation
    tracing::info!("load_to_gpu_async: Phase B-2 - Arena-based concurrent uploads");

    let dequantized = dequantized_data
        .lock()
        .map_err(|e| anyhow!("Failed to lock dequantized_data: {}", e))?;
    let shapes = tensor_shapes
        .lock()
        .map_err(|e| anyhow!("Failed to lock tensor_shapes: {}", e))?;
    let gpu_buffers: Mutex<HashMap<String, Arc<DeviceTensor>>> = Mutex::new(HashMap::new());

    // Upload tensors concurrently using multiple streams
    let mut stream_idx = 0;
    for (name, data) in dequantized.iter() {
        let shape = shapes
            .get(name)
            .ok_or_else(|| anyhow!("Shape not found for tensor '{}'", name))?;

        // Calculate tensor size
        let total_elements: usize = shape.iter().product();
        let bytes_needed = total_elements * std::mem::size_of::<f32>();

        // Allocate from arena (single large buffer subdivided)
        let offset = arena.allocate_named(name.clone(), bytes_needed)
            .map_err(|e| anyhow!("Failed to allocate arena space for '{}': {}", name, e))?;

        // Convert f32 data to bytes for upload
        let data_bytes: Vec<u8> = data
            .iter()
            .flat_map(|&f| f.to_le_bytes().to_vec())
            .collect();

        // Upload to arena buffer at offset using round-robin stream selection
        let selected_stream = stream_idx % 4;
        async_loader
            .upload_to_buffer_offset(
                arena.buffer(),
                offset,
                &data_bytes,
                selected_stream
            )
            .map_err(|e| anyhow!("Failed to upload tensor '{}': {}", name, e))?;

        // Create DeviceTensor from arena slice
        let device_tensor = Arc::new(DeviceTensor::from_arena_slice(
            backend,
            arena.buffer(),
            offset,
            bytes_needed,
            TensorShape::from_dims(shape),
        )?);

        // Store result
        gpu_buffers
            .lock()
            .map_err(|e| anyhow!("Failed to lock gpu_buffers for insert: {}", e))?
            .insert(name.clone(), device_tensor);

        stream_idx += 1;
    }

    // Synchronize all uploads
    async_loader
        .synchronize()
        .map_err(|e| anyhow!("Failed to synchronize async loader: {}", e))?;

    tracing::info!(
        "load_to_gpu_async: Phase B complete - {} tensors uploaded to GPU via arena",
        gpu_buffers
            .lock()
            .map_err(|e| anyhow!("Failed to lock gpu_buffers for logging: {}", e))?
            .len()
    );

    tracing::info!(
        "Arena upload complete: {} MB used, {} MB free, {:.1}% fragmentation, {} fragments",
        arena.allocated_bytes() / 1024 / 1024,
        arena.remaining_capacity() / 1024 / 1024,
        arena.fragmentation() * 100.0,
        arena.fragment_count()
    );

    // Phase C: Update GPU Cache
    tracing::info!("load_to_gpu_async: Phase C - Updating GPU cache");

    let mut cache = gpu_cache
        .write()
        .map_err(|e| anyhow!("GPU cache write lock poisoned: {}", e))?;

    let buffers = gpu_buffers
        .lock()
        .map_err(|e| anyhow!("Failed to lock gpu_buffers for cache update: {}", e))?;
    for (name, tensor) in buffers.iter() {
        cache.insert(name.clone(), tensor.clone());
    }

    // Debug: Log cache size after preload
    let cache_size = cache.len();
    tracing::info!("load_to_gpu_async: GPU cache now has {} tensors", cache_size);
    eprintln!(">>> load_to_gpu_async: GPU cache now has {} tensors", cache_size);

    // Convert Arc<DeviceTensor> to DeviceTensor for backward compatibility
    let result: HashMap<String, DeviceTensor> = buffers
        .iter()
        .map(|(name, tensor)| (name.clone(), DeviceTensor::clone(tensor)))
        .collect();

    tracing::info!(
        "load_to_gpu_async: Complete - Loaded {} tensors",
        result.len()
    );
    Ok(result)
}

/// Load metadata-only from file (no tensor data)
pub fn metadata_from_file(path: &str) -> Result<crate::loader::metadata::GgufMetadata> {
    use std::path::Path;

    let mmap = crate::loader::mmap::MmapGguf::open(Path::new(path))
        .map_err(|e| anyhow!("Failed to memory-map GGUF file '{}': {}", path, e))?;

    // Create a temporary loader for metadata extraction
    let mut loader = crate::loader::gguf::types::GgufLoader {
        path: path.to_string(),
        metadata: crate::loader::metadata::GgufMetadata::default(),
        tensors: std::collections::HashMap::new(),
        mmap: Some(std::sync::Arc::new(mmap)),
        lazy_tensors: std::collections::HashMap::new(),
        gpu_cache: std::sync::Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
    };

    // Load metadata only
    crate::loader::gguf::loader_impl::load_from_disk_impl(&mut loader, false)?;
    Ok(loader.metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_f32_matrix() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let transposed = transpose_f32_matrix(&data, 2, 3);
        assert_eq!(transposed.len(), 6);
        assert_eq!(transposed[0], 1.0);
        assert_eq!(transposed[1], 4.0);
        assert_eq!(transposed[2], 2.0);
        assert_eq!(transposed[3], 5.0);
        assert_eq!(transposed[4], 3.0);
        assert_eq!(transposed[5], 6.0);
    }
}
