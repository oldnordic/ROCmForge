use super::super::super::error::{GpuError, GpuResult};
use super::super::super::ffi::hipStream_t;
use super::super::buffer::GpuBuffer;
use super::super::metadata::{TensorRole, WeightMeta};
use super::super::upload::{rfm_type_to_ggml, upload_tensor_bytes_for_device};
use crate::config::ModelConfig;
use crate::cpu::transpose::compute_transpose_flag;
use crate::gpu::kernels::quant;
use crate::loader::{GgmlType, GgufFile, RfmFile, RfmType};

/// SVD low-rank outlier correction matrices stored in VRAM.
pub struct SvdCorrection {
    /// Left singular vectors (N_out x k) scaled by singular values
    pub u: GpuBuffer,
    /// Right singular vectors (k x N_in)
    pub v: GpuBuffer,
    /// SVD rank k
    pub k: u32,
}

/// Sparse CSR weight representation for GPU execution.
#[derive(Debug)]
pub struct GpuSparseCsrWeights {
    pub values: GpuBuffer,
    pub col_idx: GpuBuffer,
    pub row_ptr: GpuBuffer,
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
}

/// MPO (Matrix Product Operator) weight representation for GPU execution.
#[derive(Debug)]
pub struct GpuMpoWeights {
    pub site_data: GpuBuffer,
    pub site_dims: GpuBuffer,
    pub n_sites: u32,
}

/// CPU-resident MPO-compressed expert weights for one expert tensor type (gate, up, or down).
///
/// Loaded from `MoeExpertMpo` RFM tensors. Stays in CPU RAM;
/// uploaded one expert at a time to `GpuExpertScratch` during decode.
#[derive(Debug)]
pub struct CpuMpoExperts {
    pub n_experts: usize,
    pub chi_max: usize,
    pub rows: usize,
    pub cols: usize,
    /// Packed site data: `[n_experts, rows*chi_max + chi_max*cols]` F32
    pub site_data: Vec<f32>,
}

impl CpuMpoExperts {
    /// Byte-slice of site data for expert `i`.
    pub fn site_bytes(&self, i: usize) -> &[u8] {
        let stride = self.rows * self.chi_max + self.chi_max * self.cols;
        let slice = &self.site_data[i * stride..(i + 1) * stride];
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, stride * 4) }
    }
}

/// CPU-resident compressed expert weights for one expert tensor type (gate, up, or down).
///
/// Loaded from `MoeExpertSvdSparse` RFM tensors. Stays in CPU RAM;
/// uploaded one expert at a time to `GpuExpertScratch` during decode.
///
/// Expert `i`'s matrix is `[rows, cols]` row-major — rows = out_dim, cols = in_dim.
pub struct CpuCompressedExperts {
    pub n_experts: usize,
    pub k: usize,
    pub rows: usize,
    pub cols: usize,
    /// Packed U: `[n_experts, rows, k]` F32
    pub u_data: Vec<f32>,
    /// Packed V: `[n_experts, k, cols]` F32
    pub v_data: Vec<f32>,
    /// CSR row pointers: `[n_experts, rows+1]` u32 (experts concatenated)
    pub csr_row_ptr: Vec<u32>,
    /// CSR col indices: `[total_nnz]` u32
    pub csr_col_idx: Vec<u32>,
    /// CSR values: `[total_nnz]` F32
    pub csr_values: Vec<f32>,
    /// NNZ per expert — indexes into col_idx / values.
    pub expert_nnz: Vec<usize>,
    /// Flag indicating whether Fast Walsh-Hadamard Transform is needed for inputs before SVD GEMV
    pub needs_fwht_input: bool,
}

impl CpuCompressedExperts {
    /// Byte-slice of U for expert `i` (rows × k F32).
    pub fn u_bytes(&self, i: usize) -> &[u8] {
        let stride = self.rows * self.k;
        let slice = &self.u_data[i * stride..(i + 1) * stride];
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, stride * 4) }
    }

    /// Byte-slice of V for expert `i` (k × cols F32).
    pub fn v_bytes(&self, i: usize) -> &[u8] {
        let stride = self.k * self.cols;
        let slice = &self.v_data[i * stride..(i + 1) * stride];
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, stride * 4) }
    }

    /// (row_ptr_bytes, col_idx_bytes, val_bytes, nnz) for expert `i`.
    pub fn csr_bytes(&self, i: usize) -> (&[u8], &[u8], &[u8], usize) {
        let rp_stride = self.rows + 1;
        let row_ptr = &self.csr_row_ptr[i * rp_stride..(i + 1) * rp_stride];
        let nnz_start: usize = self.expert_nnz[..i].iter().sum();
        let nnz = self.expert_nnz[i];
        let col_idx = &self.csr_col_idx[nnz_start..nnz_start + nnz];
        let values = &self.csr_values[nnz_start..nnz_start + nnz];
        unsafe {
            let rp_bytes =
                std::slice::from_raw_parts(row_ptr.as_ptr() as *const u8, (self.rows + 1) * 4);
            let ci_bytes = std::slice::from_raw_parts(col_idx.as_ptr() as *const u8, nnz * 4);
            let val_bytes = std::slice::from_raw_parts(values.as_ptr() as *const u8, nnz * 4);
            (rp_bytes, ci_bytes, val_bytes, nnz)
        }
    }

    /// Maximum nnz across all experts — used to size `GpuExpertScratch`.
    pub fn max_nnz(&self) -> usize {
        self.expert_nnz.iter().copied().max().unwrap_or(0)
    }
}

/// Shortconv (depthwise causal conv1d) weights for LFM2 layers, resident in VRAM.
pub struct GpuShortconvWeights {
    pub in_proj: GpuBuffer,
    pub in_proj_meta: WeightMeta,
    pub conv: GpuBuffer,
    pub conv_meta: WeightMeta,
    pub out_proj: GpuBuffer,
    pub out_proj_meta: WeightMeta,
}

/// Mixture-of-Experts side weights for Qwen-style MoE layers.
pub struct GpuMoeWeights {
    pub router: GpuBuffer,
    pub router_meta: WeightMeta,
    pub router_svd: Option<SvdCorrection>,
    pub router_bias: Option<GpuBuffer>,
    pub shared_gate: Option<GpuBuffer>,
    pub shared_gate_meta: Option<WeightMeta>,
    pub shared_gate_svd: Option<SvdCorrection>,
    pub shared_up: Option<GpuBuffer>,
    pub shared_up_meta: Option<WeightMeta>,
    pub shared_up_svd: Option<SvdCorrection>,
    pub shared_down: Option<GpuBuffer>,
    pub shared_down_meta: Option<WeightMeta>,
    pub shared_down_svd: Option<SvdCorrection>,
    pub shared_gate_inp: Option<GpuBuffer>,
    pub shared_gate_inp_meta: Option<WeightMeta>,
}

/// Native Qwen35 SSM tensors for one layer, resident in VRAM.
pub struct GpuSsmWeights {
    pub a: GpuBuffer,
    pub dt: GpuBuffer,
    pub norm: GpuBuffer,
    pub conv1d: GpuBuffer,
    pub alpha: GpuBuffer,
    pub alpha_meta: WeightMeta,
    pub alpha_svd: Option<SvdCorrection>,
    pub beta: GpuBuffer,
    pub beta_meta: WeightMeta,
    pub beta_svd: Option<SvdCorrection>,
    pub out: GpuBuffer,
    pub out_meta: WeightMeta,
    pub out_svd: Option<SvdCorrection>,
}

fn qwen35_ssm_meta(name: &str, dims: &[u64], wtype: GgmlType, config: &ModelConfig) -> WeightMeta {
    WeightMeta {
        wtype,
        dims: dims.to_vec(),
        needs_transpose: compute_transpose_flag(name, dims, wtype, config, false, false),
        role: TensorRole::Generic,
        svd_k: None,
    }
}

pub(super) fn qwen35_post_attention_norm_name(
    config: &ModelConfig,
    layer: usize,
) -> Option<String> {
    if config.architecture.contains("qwen35") {
        Some(format!("blk.{}.post_attention_norm.weight", layer))
    } else {
        None
    }
}

pub(super) fn load_qwen35_ssm_gguf(
    file: &GgufFile,
    layer: usize,
    config: &ModelConfig,
    device_id: i32,
) -> GpuResult<GpuSsmWeights> {
    let load_f32 = |suffix: &str| -> GpuResult<GpuBuffer> {
        let name = format!("blk.{}.{}", layer, suffix);
        let tensor = file
            .tensor(&name)
            .map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("tensor lookup failed: {}", e),
            })?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: {}", name),
            })?;
        upload_tensor_bytes_for_device(tensor.data, device_id)
    };
    let load_weight = |suffix: &str| -> GpuResult<(GpuBuffer, WeightMeta)> {
        let name = format!("blk.{}.{}", layer, suffix);
        let tensor = file
            .tensor(&name)
            .map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("tensor lookup failed: {}", e),
            })?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: {}", name),
            })?;
        let meta = qwen35_ssm_meta(&name, tensor.dims, tensor.ggml_type, config);
        let buffer = upload_tensor_bytes_for_device(tensor.data, device_id)?;
        Ok((buffer, meta))
    };

    let (alpha, alpha_meta) = load_weight("ssm_alpha.weight")?;
    let (beta, beta_meta) = load_weight("ssm_beta.weight")?;
    let (out, out_meta) = load_weight("ssm_out.weight")?;

    Ok(GpuSsmWeights {
        a: load_f32("ssm_a")?,
        dt: load_f32("ssm_dt")?,
        norm: load_f32("ssm_norm.weight")?,
        conv1d: load_f32("ssm_conv1d.weight")?,
        alpha,
        alpha_meta,
        alpha_svd: None,
        beta,
        beta_meta,
        beta_svd: None,
        out,
        out_meta,
        out_svd: None,
    })
}

pub(super) fn load_qwen35_ssm_rfm(
    file: &RfmFile,
    layer: usize,
    config: &ModelConfig,
    device_id: i32,
) -> GpuResult<GpuSsmWeights> {
    let load_tensor = |name: &str| -> GpuResult<crate::loader::RfmTensorView<'_>> {
        file.tensor(name)
            .map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("tensor lookup failed: {}", e),
            })?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: {}", name),
            })
    };
    let load_f32 = |suffix: &str| -> GpuResult<GpuBuffer> {
        let name = format!("blk.{}.{}", layer, suffix);
        let tensor = load_tensor(&name)?;
        upload_tensor_bytes_for_device(tensor.data, device_id)
    };
    let load_weight_svd =
        |suffix: &str| -> GpuResult<(GpuBuffer, WeightMeta, Option<SvdCorrection>)> {
            let name = format!("blk.{}.{}", layer, suffix);
            let tensor = load_tensor(&name)?;
            let wtype = rfm_type_to_ggml(&tensor.wtype);
            let mut meta = qwen35_ssm_meta(&name, tensor.dims, wtype, config);
            let svd_k = match tensor.wtype {
                RfmType::Q4SvdQuant { k } | RfmType::SvdSparseCsr { k, .. } => Some(k),
                _ => None,
            };
            meta.svd_k = svd_k;
            let buffer = upload_tensor_bytes_for_device(tensor.data, device_id)?;
            let svd_corr = match tensor.wtype {
                RfmType::Q4SvdQuant { k } | RfmType::SvdSparseCsr { k, .. } => {
                    let u_name = format!("{}.svd_u", name);
                    let v_name = format!("{}.svd_v", name);
                    let u_t = file
                        .tensor(&u_name)
                        .map_err(|e| GpuError::HipApiError {
                            code: -1,
                            description: format!("SVD U lookup failed for {}: {}", name, e),
                        })?
                        .ok_or_else(|| GpuError::HipApiError {
                            code: -1,
                            description: format!("SVD U tensor not found: {}", u_name),
                        })?;
                    let v_t = file
                        .tensor(&v_name)
                        .map_err(|e| GpuError::HipApiError {
                            code: -1,
                            description: format!("SVD V lookup failed for {}: {}", name, e),
                        })?
                        .ok_or_else(|| GpuError::HipApiError {
                            code: -1,
                            description: format!("SVD V tensor not found: {}", v_name),
                        })?;
                    let u_buf = upload_tensor_bytes_for_device(u_t.data, device_id)?;
                    let v_buf = upload_tensor_bytes_for_device(v_t.data, device_id)?;
                    Some(SvdCorrection {
                        u: u_buf,
                        v: v_buf,
                        k,
                    })
                }
                _ => None,
            };
            Ok((buffer, meta, svd_corr))
        };

    let (alpha, alpha_meta, alpha_svd) = load_weight_svd("ssm_alpha.weight")?;
    let (beta, beta_meta, beta_svd) = load_weight_svd("ssm_beta.weight")?;
    let (out, out_meta, out_svd) = load_weight_svd("ssm_out.weight")?;

    Ok(GpuSsmWeights {
        a: load_f32("ssm_a")?,
        dt: load_f32("ssm_dt")?,
        norm: load_f32("ssm_norm.weight")?,
        conv1d: load_f32("ssm_conv1d.weight")?,
        alpha,
        alpha_meta,
        alpha_svd,
        beta,
        beta_meta,
        beta_svd,
        out,
        out_meta,
        out_svd,
    })
}

pub(super) fn try_load_sparse_csr(
    file: &RfmFile,
    name: &str,
    device_id: i32,
) -> GpuResult<Option<GpuSparseCsrWeights>> {
    let t = match file.tensor(name) {
        Ok(Some(t)) => t,
        _ => return Ok(None),
    };
    let csr = match t.as_sparse_csr() {
        Some(csr) => csr,
        None => return Ok(None),
    };

    let row_ptr_buf = upload_tensor_bytes_for_device(csr.row_offsets, device_id)?;
    let col_idx_buf = upload_tensor_bytes_for_device(csr.col_indices, device_id)?;
    let values_buf = upload_tensor_bytes_for_device(csr.values, device_id)?;

    Ok(Some(GpuSparseCsrWeights {
        values: values_buf,
        col_idx: col_idx_buf,
        row_ptr: row_ptr_buf,
        rows: csr.rows,
        cols: csr.cols,
        nnz: csr.nnz,
    }))
}

pub(super) fn try_load_mpo(
    file: &RfmFile,
    name: &str,
    device_id: i32,
) -> GpuResult<Option<GpuMpoWeights>> {
    let t = match file.tensor(name) {
        Ok(Some(t)) => t,
        _ => return Ok(None),
    };
    let mpo = match t.as_mpo() {
        Some(mpo) => mpo,
        None => return Ok(None),
    };

    let site_data = upload_tensor_bytes_for_device(mpo.data, device_id)?;
    let site_dims_host: Vec<u32> = mpo.site_dims.iter().map(|d| *d as u32).collect();
    let mut site_dims = GpuBuffer::alloc(site_dims_host.len() * std::mem::size_of::<u32>())?;
    let site_dims_bytes = unsafe {
        std::slice::from_raw_parts(
            site_dims_host.as_ptr() as *const u8,
            site_dims_host.len() * std::mem::size_of::<u32>(),
        )
    };
    site_dims.copy_from_host(site_dims_bytes)?;

    Ok(Some(GpuMpoWeights {
        site_data,
        site_dims,
        n_sites: mpo.n_sites as u32,
    }))
}

/// Parse a `MoeExpertMpo` RFM tensor into CPU-resident `CpuMpoExperts`.
/// Returns `Ok(None)` if the tensor is absent or not the right type.
pub(super) fn try_load_moe_expert_mpo(
    file: &RfmFile,
    name: &str,
) -> GpuResult<Option<CpuMpoExperts>> {
    let t = match file.tensor(name) {
        Ok(Some(t)) => t,
        _ => return Ok(None),
    };
    let mpo = match t.as_moe_expert_mpo() {
        Some(mpo) => mpo,
        None => return Ok(None),
    };

    let n_experts = mpo.n_experts;
    let chi_max = mpo.chi_max;
    let rows = mpo.rows;
    let cols = mpo.cols;
    let expert_elements = rows * chi_max + chi_max * cols;
    let total_elements = n_experts * expert_elements;

    if mpo.site_data.len() != total_elements * 4 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "MoeExpertMpo '{}': site_data size {} bytes != expected {}",
                name,
                mpo.site_data.len(),
                total_elements * 4
            ),
        });
    }

    let site_data: Vec<f32> = mpo
        .site_data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok(Some(CpuMpoExperts {
        n_experts,
        chi_max,
        rows,
        cols,
        site_data,
    }))
}

/// Parse a `MoeExpertSvdSparse` or `MoeExpertSvdFwhtSparse` RFM tensor into CPU-resident `CpuCompressedExperts`.
/// Returns `Ok(None)` if the tensor is absent or not the right type.
pub(super) fn try_load_compressed_experts(
    file: &RfmFile,
    name: &str,
) -> GpuResult<Option<CpuCompressedExperts>> {
    let t = match file.tensor(name) {
        Ok(Some(t)) => t,
        _ => return Ok(None),
    };
    let (n_experts, k, rows, cols, total_nnz, index_bits, needs_fwht_input) = match t.wtype {
        RfmType::MoeExpertSvdSparse {
            n_experts,
            k,
            rows,
            cols,
            total_nnz,
            index_bits,
            ..
        } => (
            n_experts as usize,
            k as usize,
            rows as usize,
            cols as usize,
            total_nnz as usize,
            index_bits,
            false,
        ),
        RfmType::MoeExpertSvdFwhtSparse {
            n_experts,
            k,
            rows,
            cols,
            total_nnz,
            index_bits,
            ..
        } => (
            n_experts as usize,
            k as usize,
            rows as usize,
            cols as usize,
            total_nnz as usize,
            index_bits,
            true,
        ),
        _ => return Ok(None),
    };

    if index_bits != 32 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "MoeExpertSvdSparse/MoeExpertSvdFwhtSparse tensor '{}' uses unsupported index_bits={}",
                name, index_bits
            ),
        });
    }

    let u_count = n_experts * rows * k;
    let v_count = n_experts * k * cols;
    let rp_count = n_experts * (rows + 1);
    let expected = (u_count + v_count + total_nnz + total_nnz) * 4 + rp_count * 4 + n_experts * 4;

    if t.data.len() < expected {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "MoeExpertSvdSparse/MoeExpertSvdFwhtSparse '{}': payload {} bytes < expected {}",
                name,
                t.data.len(),
                expected
            ),
        });
    }

    let read_f32 = |data: &[u8], count: usize| -> Vec<f32> {
        data[..count * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };
    let read_u32 = |data: &[u8], count: usize| -> Vec<u32> {
        data[..count * 4]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };

    let mut off = 0usize;
    let u_data = read_f32(&t.data[off..], u_count);
    off += u_count * 4;
    let v_data = read_f32(&t.data[off..], v_count);
    off += v_count * 4;
    let csr_row_ptr = read_u32(&t.data[off..], rp_count);
    off += rp_count * 4;
    let csr_col_idx = read_u32(&t.data[off..], total_nnz);
    off += total_nnz * 4;
    let csr_values = read_f32(&t.data[off..], total_nnz);
    off += total_nnz * 4;
    let nnz_raw = read_u32(&t.data[off..], n_experts);
    let expert_nnz: Vec<usize> = nnz_raw.iter().map(|&x| x as usize).collect();

    Ok(Some(CpuCompressedExperts {
        n_experts,
        k,
        rows,
        cols,
        u_data,
        v_data,
        csr_row_ptr,
        csr_col_idx,
        csr_values,
        expert_nnz,
        needs_fwht_input,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressed_experts_max_nnz_defaults_to_zero() {
        let experts = CpuCompressedExperts {
            n_experts: 0,
            k: 0,
            rows: 0,
            cols: 0,
            u_data: Vec::new(),
            v_data: Vec::new(),
            csr_row_ptr: Vec::new(),
            csr_col_idx: Vec::new(),
            csr_values: Vec::new(),
            expert_nnz: Vec::new(),
            needs_fwht_input: false,
        };

        assert_eq!(experts.max_nnz(), 0);
    }
}
