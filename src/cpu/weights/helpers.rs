use super::meta::{WeightError, WeightMeta};
use crate::loader::{GgmlType, GgufFile, RfmType, TensorView};

/// Copy tensor bytes from the mmap into a Vec<u8>.
pub(crate) fn copy_tensor(file: &GgufFile, name: &str) -> Result<Vec<u8>, WeightError> {
    let t = file
        .tensor(name)
        .map_err(WeightError::Load)?
        .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
    Ok(t.data.to_vec())
}

pub(crate) fn copy_tensor_optional(
    file: &GgufFile,
    name: &str,
) -> Result<Option<Vec<u8>>, WeightError> {
    match file.tensor(name).map_err(WeightError::Load)? {
        None => Ok(None),
        Some(t) => Ok(Some(t.data.to_vec())),
    }
}

/// Copy an always-F32 tensor as Vec<f32>.
pub(crate) fn copy_f32(file: &GgufFile, name: &str) -> Result<Vec<f32>, WeightError> {
    let bytes = copy_tensor(file, name)?;
    let n = bytes.len() / 4;
    let mut out = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const f32, out.as_mut_ptr(), n);
    }
    Ok(out)
}

pub(crate) fn copy_f32_from_bytes(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 4;
    let mut out = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const f32, out.as_mut_ptr(), n);
    }
    out
}

pub(crate) fn optional_f32(file: &GgufFile, name: &str) -> Result<Option<Vec<f32>>, WeightError> {
    if file.tensor(name).map_err(WeightError::Load)?.is_some() {
        copy_f32(file, name).map(Some)
    } else {
        Ok(None)
    }
}

pub(crate) fn copy_tensor_with_meta(
    file: &GgufFile,
    name: &str,
    needs_transpose: bool,
) -> Result<(Vec<u8>, WeightMeta), WeightError> {
    let t = file
        .tensor(name)
        .map_err(WeightError::Load)?
        .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
    Ok((t.data.to_vec(), WeightMeta::from_view(&t, needs_transpose)))
}

pub(crate) fn rfm_type_to_ggml(rfm: &RfmType) -> GgmlType {
    match rfm {
        RfmType::F32 => GgmlType::F32,
        RfmType::Mq4 => GgmlType::Q4_0,
        RfmType::Mq6 => GgmlType::Q6_K,
        RfmType::Q4Split => GgmlType::Q4_0,
        RfmType::Q4SvdQuant { .. } => GgmlType::Q4_0,
        RfmType::GgufPassthrough(t) => GgmlType::from_u32(*t).unwrap_or(GgmlType::F32),
        RfmType::MoeExpertSvdSparse { .. } | RfmType::MoeExpertSvdFwhtSparse { .. } => {
            GgmlType::F32
        }
        _ => GgmlType::F32,
    }
}

pub(crate) fn rfm_weight_meta(
    t: &crate::loader::RfmTensorView<'_>,
    needs_transpose: bool,
) -> WeightMeta {
    let mut meta = WeightMeta {
        wtype: rfm_type_to_ggml(&t.wtype),
        dims: t.dims.to_vec(),
        needs_transpose,
        svd_k: None,
    };
    if let RfmType::Q4SvdQuant { k, .. } = t.wtype {
        meta.svd_k = Some(k);
    }
    meta
}

/// Unpack Q4_0 data stored in RFM's "split" format: [scales (f32) | blocks (u8)].
pub(crate) fn unpack_q4_split(data: &[u8], element_count: usize) -> Vec<u8> {
    let n_blocks = element_count / 32;
    let mut out = vec![0u8; n_blocks * 18];
    let rfm_scales_ptr = data.as_ptr() as *const f32;
    let rfm_quants_ptr = unsafe { data.as_ptr().add(n_blocks * 4) };

    for i in 0..n_blocks {
        let scale_f32 = unsafe { *rfm_scales_ptr.add(i) };
        let scale_f16 = half::f16::from_f32(scale_f32).to_bits();
        let block_out = unsafe { out.as_mut_ptr().add(i * 18) };

        unsafe {
            std::ptr::copy_nonoverlapping(&scale_f16 as *const u16 as *const u8, block_out, 2);
            std::ptr::copy_nonoverlapping(rfm_quants_ptr.add(i * 16), block_out.add(2), 16);
        }
    }
    out
}

pub(crate) fn sparse_csr_to_dense_f32_bytes(
    rows: usize,
    cols: usize,
    values: &[f32],
    col_indices: &[i32],
    row_offsets: &[i32],
) -> Vec<u8> {
    let mut dense = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let start = row_offsets[r] as usize;
        let end = row_offsets[r + 1] as usize;
        for i in start..end {
            let c = col_indices[i] as usize;
            dense[r * cols + c] = values[i];
        }
    }
    let mut bytes = vec![0u8; dense.len() * 4];
    unsafe {
        std::ptr::copy_nonoverlapping(dense.as_ptr() as *const u8, bytes.as_mut_ptr(), bytes.len());
    }
    bytes
}

pub(crate) fn unpack_q4_fused_gate_up(data: &[u8], element_count: usize) -> (Vec<u8>, Vec<u8>) {
    let total_blocks = element_count / 32;
    let blocks_per_tensor = total_blocks / 2;
    let mut gate_out = vec![0u8; blocks_per_tensor * 18];
    let mut up_out = vec![0u8; blocks_per_tensor * 18];

    let gate_scales_ptr = data.as_ptr() as *const f32;
    let up_scales_ptr = unsafe { data.as_ptr().add(blocks_per_tensor * 4) } as *const f32;
    let gate_quants_ptr = unsafe { data.as_ptr().add(total_blocks * 4) };
    let up_quants_ptr = unsafe { data.as_ptr().add(total_blocks * 4 + blocks_per_tensor * 16) };

    for i in 0..blocks_per_tensor {
        let g_scale = half::f16::from_f32(unsafe { *gate_scales_ptr.add(i) }).to_bits();
        unsafe {
            let p = gate_out.as_mut_ptr().add(i * 18);
            std::ptr::copy_nonoverlapping(&g_scale as *const u16 as *const u8, p, 2);
            std::ptr::copy_nonoverlapping(gate_quants_ptr.add(i * 16), p.add(2), 16);
        }
        let u_scale = half::f16::from_f32(unsafe { *up_scales_ptr.add(i) }).to_bits();
        unsafe {
            let p = up_out.as_mut_ptr().add(i * 18);
            std::ptr::copy_nonoverlapping(&u_scale as *const u16 as *const u8, p, 2);
            std::ptr::copy_nonoverlapping(up_quants_ptr.add(i * 16), p.add(2), 16);
        }
    }
    (gate_out, up_out)
}
