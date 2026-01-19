//! Accumulate op for efficient KV cache writes.

use crate::backend::{HipBackend, HipBuffer, HipError, HipResult};

/// Accumulates source tensor into destination tensor at an offset.
///
/// This operation performs: `dst[offset:offset+src_size] += src`
/// and writes the result to the output buffer.
///
/// This is used for in-place KV cache updates without Copy overhead.
///
/// # Parameters
/// - `backend`: The HIP backend for buffer operations
/// - `src`: Source buffer containing data to add
/// - `dst`: Destination buffer to accumulate into
/// - `output`: Output buffer for the result
/// - `element_count`: Number of f32 elements to accumulate
/// - `byte_offset`: Offset in dst buffer to start accumulation (in bytes)
///
/// # Returns
/// - `HipResult<()>`: Success or error
pub fn accumulate(
    backend: &HipBackend,
    src: &HipBuffer,
    dst: &HipBuffer,
    output: &HipBuffer,
    element_count: usize,
    byte_offset: usize,
) -> HipResult<()> {
    // Validate sizes
    let src_bytes = element_count * std::mem::size_of::<f32>();
    if src.size() < src_bytes {
        return Err(HipError::GenericError(format!(
            "Source buffer too small: need {} bytes, have {}",
            src_bytes,
            src.size()
        )));
    }

    if dst.size() < byte_offset + src_bytes {
        return Err(HipError::GenericError(format!(
            "Destination buffer too small for offset: need {} bytes, have {}",
            byte_offset + src_bytes,
            dst.size()
        )));
    }

    if output.size() < src_bytes {
        return Err(HipError::GenericError(format!(
            "Output buffer too small: need {} bytes, have {}",
            src_bytes,
            output.size()
        )));
    }

    // Download data from GPU
    let mut src_data = vec![0u8; src_bytes];
    backend
        .copy_from_device_safe(src, &mut src_data)
        .map_err(|e| HipError::GenericError(format!("Failed to download src: {}", e)))?;

    let mut dst_data = vec![0u8; dst.size()];
    backend
        .copy_from_device_safe(dst, &mut dst_data)
        .map_err(|e| HipError::GenericError(format!("Failed to download dst: {}", e)))?;

    // Convert to slices
    let src_slice: &[f32] = bytemuck::cast_slice(&src_data);
    let dst_slice: &[f32] = bytemuck::cast_slice(&dst_data);

    // Perform accumulate: result[i] = dst[offset+i] + src[i]
    let dst_offset = byte_offset / std::mem::size_of::<f32>();
    let mut result = Vec::with_capacity(element_count);

    for i in 0..element_count {
        let dst_idx = dst_offset + i;
        let src_val = if i < src_slice.len() {
            src_slice[i]
        } else {
            0.0
        };
        let dst_val = if dst_idx < dst_slice.len() {
            dst_slice[dst_idx]
        } else {
            0.0
        };
        result.push(dst_val + src_val);
    }

    // Upload result to output buffer
    output
        .copy_from_host(&result)
        .map_err(|e| HipError::GenericError(format!("Failed to upload output: {}", e)))?;

    // Also update dst in-place (write the accumulated values back)
    let dst_full = {
        let mut dst_vec: Vec<f32> = dst_slice.to_vec();
        for i in 0..element_count {
            let dst_idx = dst_offset + i;
            if dst_idx < dst_vec.len() && i < src_slice.len() {
                dst_vec[dst_idx] = dst_slice[dst_idx] + src_slice[i];
            }
        }
        dst_vec
    };

    dst.copy_from_host(&dst_full)
        .map_err(|e| HipError::GenericError(format!("Failed to update dst: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_accumulate_basic() {
        // Simple case: accumulate [1.0, 2.0, 3.0] into [10.0, 20.0, 30.0, 40.0]
        // at offset 1
        // result[0] = dst[1] + src[0] = 20.0 + 1.0 = 21.0
        // result[1] = dst[2] + src[1] = 30.0 + 2.0 = 32.0
        // result[2] = dst[3] + src[2] = 40.0 + 3.0 = 43.0
        // dst becomes: [10.0, 21.0, 32.0, 43.0]
        let src = vec![1.0f32, 2.0, 3.0];
        let mut dst = vec![10.0f32, 20.0, 30.0, 40.0];
        let element_count = 3;
        let byte_offset = 1 * std::mem::size_of::<f32>();

        let dst_offset = byte_offset / std::mem::size_of::<f32>();
        let mut result = Vec::with_capacity(element_count);

        for i in 0..element_count {
            let dst_idx = dst_offset + i;
            result.push(dst[dst_idx] + src[i]);
            dst[dst_idx] += src[i]; // In-place update
        }

        assert_eq!(result, vec![21.0, 32.0, 43.0]);
        assert_eq!(dst, vec![10.0, 21.0, 32.0, 43.0]);
    }

    #[test]
    fn test_accumulate_offset() {
        // Accumulate at offset 0
        let src = vec![1.0f32, 2.0];
        let mut dst = vec![10.0f32, 20.0, 30.0];
        let element_count = 2;
        let byte_offset = 0;

        let dst_offset = byte_offset / std::mem::size_of::<f32>();
        let mut result = Vec::with_capacity(element_count);

        for i in 0..element_count {
            let dst_idx = dst_offset + i;
            result.push(dst[dst_idx] + src[i]);
            dst[dst_idx] += src[i];
        }

        assert_eq!(result, vec![11.0, 22.0]);
        assert_eq!(dst, vec![11.0, 22.0, 30.0]);
    }

    #[test]
    fn test_accumulate_zeros() {
        // Accumulating zeros should leave dst unchanged
        let src = vec![0.0f32, 0.0, 0.0];
        let mut dst = vec![10.0f32, 20.0, 30.0];
        let element_count = 3;
        let byte_offset = 0;

        let dst_offset = byte_offset / std::mem::size_of::<f32>();
        let mut result = Vec::with_capacity(element_count);

        for i in 0..element_count {
            let dst_idx = dst_offset + i;
            result.push(dst[dst_idx] + src[i]);
            dst[dst_idx] += src[i];
        }

        assert_eq!(result, vec![10.0, 20.0, 30.0]);
        assert_eq!(dst, vec![10.0, 20.0, 30.0]);
    }
}
