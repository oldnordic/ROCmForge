use super::meta::{WeightError, WeightMeta};
use crate::config::TensorRole;
use crate::loader::{GgmlType, GgufFile, RfmType};

/// Copy tensor bytes from the mmap into a Vec<u8>.
pub(crate) fn copy_tensor(file: &GgufFile, name: &str) -> Result<Vec<u8>, WeightError> {
    let t = file
        .tensor(name)
        .map_err(WeightError::Load)?
        .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
    Ok(t.data.to_vec())
}

/// Copy an always-F32 tensor as Vec<f32>.
/// Uses byte-wise copy to avoid alignment requirements on the source buffer.
pub(crate) fn copy_f32(file: &GgufFile, name: &str) -> Result<Vec<f32>, WeightError> {
    let bytes = copy_tensor(file, name)?;
    Ok(copy_f32_from_bytes(&bytes))
}

/// Convert a little-endian byte slice to Vec<f32> without alignment assumptions.
pub(crate) fn copy_f32_from_bytes(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 4;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let b = &bytes[i * 4..i * 4 + 4];
        out[i] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    out
}

/// Attempt to safely view a `&[u8]` as `&[f32]`.
///
/// Returns `Some` only if the byte slice pointer is 4-byte aligned and the length
/// is a multiple of 4. On x86_64 Linux, `Vec<u8>` heap allocations are typically
/// 16-byte aligned, so this succeeds in practice.
///
/// # Safety
/// When this returns `Some`, the underlying bytes are reinterpreted as `f32`
/// in native endianness (GGUF/RFM are little-endian, and this code only runs
/// on little-endian platforms).
pub(crate) fn try_as_f32_slice(bytes: &[u8]) -> Option<&[f32]> {
    let ptr = bytes.as_ptr() as *const f32;
    if ptr.align_offset(std::mem::align_of::<f32>()) == 0 && bytes.len().is_multiple_of(4) {
        Some(unsafe { std::slice::from_raw_parts(ptr, bytes.len() / 4) })
    } else {
        None
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
    let role = TensorRole::from_name(name, false, false);
    Ok((
        t.data.to_vec(),
        WeightMeta::from_view_with_role(&t, needs_transpose, role),
    ))
}

pub(crate) fn rfm_type_to_ggml(rfm: &RfmType) -> GgmlType {
    match rfm {
        RfmType::F32 => GgmlType::F32,
        RfmType::Mq4 => GgmlType::Q4_0,
        RfmType::Mq6 => GgmlType::Q6_K,
        RfmType::Q4Split => GgmlType::Q4_0,
        RfmType::Q4SvdQuant { .. } => GgmlType::Q4_0,
        RfmType::GgufPassthrough(t) => GgmlType::from_u32(*t).unwrap_or(GgmlType::F32),
        RfmType::MoeExpertSvdSparse { .. }
        | RfmType::MoeExpertSvdFwhtSparse { .. }
        | RfmType::MoeExpertMpo { .. } => GgmlType::F32,
        _ => GgmlType::F32,
    }
}

pub(crate) fn rfm_weight_meta(
    t: &crate::loader::RfmTensorView<'_>,
    needs_transpose: bool,
) -> WeightMeta {
    let role = TensorRole::from_name(t.name, false, false);
    let mut meta = WeightMeta {
        wtype: rfm_type_to_ggml(&t.wtype),
        dims: t.dims.to_vec(),
        needs_transpose,
        role,
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
    let rfm_quants_ptr = unsafe { data.as_ptr().add(n_blocks * 4) };

    for i in 0..n_blocks {
        let scale_f32 = unsafe { std::ptr::read_unaligned(data.as_ptr().add(i * 4) as *const f32) };
        let scale_f16 = half::f16::from_f32(scale_f32).to_bits();
        let block_out = unsafe { out.as_mut_ptr().add(i * 18) };

        unsafe {
            std::ptr::copy_nonoverlapping(&scale_f16 as *const u16 as *const u8, block_out, 2);
            std::ptr::copy_nonoverlapping(rfm_quants_ptr.add(i * 16), block_out.add(2), 16);
        }
    }
    out
}
