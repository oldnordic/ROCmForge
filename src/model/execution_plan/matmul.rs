//! Matrix multiplication and tensor transformation helpers
//!
//! Handles matmul operations, tensor reshaping, and transformations.

use crate::backend::{DeviceTensor, HipBackend, HipError, HipResult};
use crate::backend::hip_blas::HipBlasHandle;
use crate::tensor::matmul::matmul_f32;

use super::types::ExecutionPlan;

/// Matrix multiplication with optional bias
pub fn matmul(
    plan: &ExecutionPlan,
    backend: &HipBackend,
    input: &DeviceTensor,
    weight: &DeviceTensor,
    bias: Option<&DeviceTensor>,
) -> HipResult<DeviceTensor> {
    eprintln!(">>>       matmul: ENTRY - getting shapes...");

    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();
    let _batch_size = input_shape[0];
    let _input_dim = input_shape[1];
    let output_dim = weight_shape[1];

    eprintln!(
        ">>>       matmul: input_shape={:?}, weight_shape={:?}, expecting output_dim={}",
        input_shape, weight_shape, output_dim
    );

    let batch_size = input_shape[0];
    let input_dim = input_shape[1];
    let output_dim = weight_shape[1];

    // Validate shapes
    if weight_shape[0] != input_dim {
        return Err(HipError::GenericError(format!(
            "Weight shape {:?} incompatible with input shape {:?}",
            weight_shape, input_shape
        )));
    }

    eprintln!(">>>       matmul: input_shape={:?}, weight_shape={:?}, expecting output_dim=3*hidden={}",
             input.shape().dims(), weight.shape().dims(), 3 * input_dim);
    let matmul_start = std::time::Instant::now();

    // Create hipBLAS handle for matrix operations
    let blas_handle = HipBlasHandle::new().map_err(|e| {
        HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
    })?;
    eprintln!(">>>       matmul: hipBLAS handle created");

    // CRITICAL: Associate hipBLAS handle with our HIP stream
    // Without this, hipBLAS uses the default stream while our kernels use a custom stream,
    // causing synchronization issues and hangs.
    blas_handle
        .set_stream(backend.stream().as_ptr())
        .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;
    eprintln!(">>>       matmul: hipBLAS stream set");

    // Perform matrix multiplication: input @ weight -> output
    // input: [batch_size, input_dim], weight: [input_dim, output_dim] -> output: [batch_size, output_dim]
    eprintln!(">>>       matmul: calling matmul_f32...",);
    let matmul_call_start = std::time::Instant::now();
    let output_buffer = matmul_f32(
        backend,
        &blas_handle,
        input.buffer(),
        weight.buffer(),
        batch_size as i32,
        output_dim as i32,
        input_dim as i32,
    )
    .map_err(|e| HipError::GenericError(format!("Matrix multiplication failed: {}", e)))?;
    eprintln!(
        ">>>       matmul: matmul_f32 done in {:?}",
        matmul_call_start.elapsed()
    );

    // CRITICAL: Synchronize after matmul_f32 to ensure GPU completes the operation
    // before we try to copy the data. Without this, the memcpy blocks indefinitely.
    eprintln!(">>>       matmul: synchronizing GPU...",);
    let sync_start = std::time::Instant::now();
    backend
        .synchronize()
        .map_err(|e| HipError::GenericError(format!("GPU sync failed: {}", e)))?;
    eprintln!(
        ">>>       matmul: GPU synchronized in {:?}",
        sync_start.elapsed()
    );

    let output_shape = crate::loader::TensorShape::from_dims(&[batch_size, output_dim]);
    let mut output_tensor = DeviceTensor::empty(backend, output_shape)?;
    eprintln!(">>>       matmul: copying to output tensor...",);
    output_tensor.copy_from_device_buffer(&output_buffer)?;
    // Synchronize again after copy to ensure data is actually transferred
    backend
        .synchronize()
        .map_err(|e| HipError::GenericError(format!("Post-copy sync failed: {}", e)))?;
    eprintln!(">>>       matmul: copy done",);

    if let Some(bias_tensor) = bias {
        eprintln!(">>>       matmul: adding bias...",);
        backend.add_row_bias(&mut output_tensor, bias_tensor)?;
        eprintln!(">>>       matmul: bias added",);
    }

    eprintln!(">>>       matmul: COMPLETE in {:?}", matmul_start.elapsed());
    Ok(output_tensor)
}

/// Transpose a 2D tensor on the host and upload it back to the device
pub fn transpose_2d_tensor(backend: &HipBackend, tensor: &DeviceTensor) -> HipResult<DeviceTensor> {
    let shape = tensor.shape().dims();
    if shape.len() != 2 {
        return Err(HipError::GenericError(format!(
            "Expected 2D tensor for transpose, got {}D",
            shape.len()
        )));
    }

    let rows = shape[0];
    let cols = shape[1];
    tracing::debug!(
        "transpose_2d_tensor: shape=[{}, {}], size={} bytes",
        rows,
        cols,
        tensor.len() * std::mem::size_of::<f32>()
    );
    let host = tensor.to_host_vec()?;
    let mut transposed = vec![0.0f32; host.len()];

    for r in 0..rows {
        for c in 0..cols {
            let src_idx = r * cols + c;
            let dst_idx = c * rows + r;
            transposed[dst_idx] = host[src_idx];
        }
    }

    let new_shape = crate::loader::TensorShape::from_dims(&[cols, rows]);
    DeviceTensor::from_host_vec(backend, transposed, new_shape)
}

/// Reshape a projected tensor for multi-head attention
///
/// Takes a 2D tensor [seq_len, dim] and reshapes it to 3D [seq_len, num_heads, head_dim]
/// where dim = num_heads * head_dim. This is a simple reshape that changes the stride
/// interpretation - no data movement is required.
pub fn reshape_for_attention(
    backend: &HipBackend,
    proj: &DeviceTensor,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> HipResult<DeviceTensor> {
    let expected_dim = num_heads * head_dim;
    let proj_shape = proj.shape().dims();

    eprintln!(">>>         reshape_for_attention: proj shape={:?}, expected=[{}, {}], target=[{}, {}, {}]",
             proj_shape, seq_len, expected_dim, seq_len, num_heads, head_dim);

    if proj_shape[0] != seq_len || proj_shape[1] != expected_dim {
        return Err(HipError::GenericError(format!(
            "reshape_for_attention: proj shape {:?} doesn't match expected [{}, {}] where {} = {} * {}",
            proj_shape, seq_len, expected_dim, expected_dim, num_heads, head_dim
        )));
    }

    // For tensors that are already contiguous with the right total size,
    // we can create a new DeviceTensor with a different shape interpretation.
    // This is a view/reshape operation that doesn't copy data.

    // Read the data and create a new tensor with the 3D shape
    let proj_data = proj.to_host_vec().map_err(|e| {
        HipError::GenericError(format!(
            "reshape_for_attention: failed to download tensor: {}",
            e
        ))
    })?;

    let new_shape = crate::loader::TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
    DeviceTensor::from_host_vec(backend, proj_data, new_shape).map_err(|e| {
        HipError::GenericError(format!(
            "reshape_for_attention: failed to upload tensor: {}",
            e
        ))
    })
}

/// Extract Q, K, V tensors from fused QKV projection
pub fn extract_qkv_tensors(
    backend: &HipBackend,
    qkv_proj: &DeviceTensor,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> HipResult<(DeviceTensor, DeviceTensor, DeviceTensor)> {
    let hidden_size = num_heads * head_dim;
    let chunk_elements = seq_len * hidden_size;

    fn copy_chunk(
        backend: &HipBackend,
        src: &DeviceTensor,
        offset_elements: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        name: &str,
    ) -> HipResult<DeviceTensor> {
        eprintln!(
            ">>>         copy_chunk({}): creating tensor [{},{},{}], offset={}",
            name, seq_len, num_heads, head_dim, offset_elements
        );
        let step_start = std::time::Instant::now();
        let shape = crate::loader::TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        eprintln!(
            ">>>         copy_chunk({}): calling DeviceTensor::empty...",
            name
        );
        let mut tensor = DeviceTensor::empty(backend, shape)?;
        eprintln!(
            ">>>         copy_chunk({}): empty done, calling copy_from_device_slice...",
            name
        );
        tensor.copy_from_device_slice(src, offset_elements)?;
        eprintln!(
            ">>>         copy_chunk({}): COMPLETE ({:?})",
            name,
            step_start.elapsed()
        );
        Ok(tensor)
    }

    eprintln!(">>>       extract_qkv_tensors: Starting Q extraction...");
    let q = copy_chunk(backend, qkv_proj, 0, seq_len, num_heads, head_dim, "Q")?;
    eprintln!(">>>       extract_qkv_tensors: Q extracted, starting K extraction...");
    let k = copy_chunk(
        backend,
        qkv_proj,
        chunk_elements,
        seq_len,
        num_heads,
        head_dim,
        "K",
    )?;
    eprintln!(">>>       extract_qkv_tensors: K extracted, starting V extraction...");
    let v = copy_chunk(
        backend,
        qkv_proj,
        chunk_elements * 2,
        seq_len,
        num_heads,
        head_dim,
        "V",
    )?;
    eprintln!(">>>       extract_qkv_tensors: All Q,K,V extracted successfully");

    Ok((q, k, v))
}

/// Flatten attention output from 3D to 2D
pub fn flatten_attention_output(
    backend: &HipBackend,
    attention_output: &DeviceTensor,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> HipResult<DeviceTensor> {
    let hidden_size = num_heads * head_dim;
    let shape = crate::loader::TensorShape::from_dims(&[seq_len, hidden_size]);
    let mut tensor = DeviceTensor::empty(backend, shape)?;
    tensor.copy_from_device_slice(attention_output, 0)?;
    Ok(tensor)
}

/// Add residual connection using optimized GPU operations
pub fn add_residual(
    backend: &HipBackend,
    input: &DeviceTensor,
    residual: &DeviceTensor,
) -> HipResult<DeviceTensor> {
    // Validate shapes match
    if input.shape().dims() != residual.shape().dims() {
        return Err(HipError::GenericError(
            "Input and residual tensors must have the same shape".to_string(),
        ));
    }

    let output_shape = input.shape().clone();
    let mut output = DeviceTensor::empty(backend, output_shape)?;

    output.buffer().copy_from_buffer(residual.buffer())?;
    backend.add_inplace(input, &mut output)?;

    Ok(output)
}
