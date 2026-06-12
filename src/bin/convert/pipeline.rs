use std::fs::File;
use std::io::Write;

use rayon::prelude::*;
use rocmforge::loader::{GgmlType, GgufFile, TensorView};
use rocmforge::loader::{RfmTensorEntry, RfmType};

use super::cli::ConvertOptions;

use super::layout::{pack_tensor, rfm_type_for_tensor};
use super::math::{fwht_inplace, matmul, svd_batch_experts, svd_decompose};
use super::quant::{
    bytes_to_f32, dequantize_f16_to_f32, dequantize_q4_0_to_f32, dequantize_q6_k_to_f32,
    dequantize_q8_0_to_f32, quantize_matrix_q4_0,
};

pub(super) fn should_compress_tensor(name: &str, tensor: &TensorView) -> bool {
    if tensor.dims.len() != 2 {
        return false;
    }
    if tensor.dims.iter().any(|&d| d < 64) {
        return false;
    }
    matches!(
        tensor.ggml_type,
        GgmlType::F32 | GgmlType::Q4_0 | GgmlType::Q4_K | GgmlType::Q6_K
    ) && name.ends_with(".weight")
        && (name.contains("ffn_gate") || name.contains("ffn_up") || name.contains("ffn_down"))
}

pub(super) fn estimate_nnz_ratio(tensor: &TensorView) -> f32 {
    let count = tensor.element_count();
    let sample_size = count.min(4096);
    let step = if count > sample_size {
        count / sample_size
    } else {
        1
    };

    let w_f32 = match tensor.ggml_type {
        GgmlType::Q4_0 => dequantize_q4_0_to_f32(tensor.data, count),
        GgmlType::Q4_K => {
            let mut out = vec![0.0f32; count];
            rocmforge::cpu::quant::embed_q4_k(0, tensor.data, &mut out, count);
            out
        }
        GgmlType::Q6_K => dequantize_q6_k_to_f32(tensor.data, count),
        GgmlType::F32 => bytes_to_f32(tensor.data),
        _ => return 1.0f32,
    };

    let mut nnz = 0usize;
    for i in 0..sample_size {
        let idx = i * step;
        if idx < w_f32.len() && w_f32[idx].abs() > 1e-6 {
            nnz += 1;
        }
    }

    (nnz as f32) / (sample_size as f32)
}

pub(super) fn convert_sparse_csr_tensor(
    tensor: &TensorView,
    base_name: &str,
    writer: &mut dyn Write,
    current_offset: &mut u64,
    entries: &mut Vec<RfmTensorEntry>,
    align_offset: &impl Fn(&mut dyn Write, &mut u64) -> Result<(), std::io::Error>,
) -> Result<(), Box<dyn std::error::Error>> {
    let rows = tensor.dims[0] as usize;
    let cols = tensor.dims[1] as usize;

    let w_f32 = match tensor.ggml_type {
        GgmlType::Q4_0 => dequantize_q4_0_to_f32(tensor.data, tensor.element_count()),
        GgmlType::Q4_K => {
            let mut out = vec![0.0f32; tensor.element_count()];
            rocmforge::cpu::quant::embed_q4_k(0, tensor.data, &mut out, tensor.element_count());
            out
        }
        GgmlType::Q6_K => dequantize_q6_k_to_f32(tensor.data, tensor.element_count()),
        GgmlType::F32 => bytes_to_f32(tensor.data),
        other => {
            return Err(format!(
                "Unsupported source type for sparse CSR conversion: {:?}",
                other
            )
            .into())
        }
    };

    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_offsets = vec![0u32; rows + 1];

    for i in 0..rows {
        for j in 0..cols {
            let v = w_f32[i * cols + j];
            if v.abs() > 1e-6 {
                values.push(v);
                col_indices.push(j as u32);
            }
        }
        row_offsets[i + 1] = values.len() as u32;
    }

    let nnz = values.len();
    align_offset(writer, current_offset)?;
    let payload_offset = *current_offset;

    for &off in &row_offsets {
        writer.write_all(&off.to_le_bytes())?;
    }
    for &col in &col_indices {
        writer.write_all(&col.to_le_bytes())?;
    }
    for &val in &values {
        writer.write_all(&val.to_le_bytes())?;
    }

    let payload_size = (row_offsets.len() + col_indices.len()) * 4 + values.len() * 4;
    *current_offset += payload_size as u64;

    entries.push(RfmTensorEntry {
        name: base_name.to_string(),
        dims: tensor.dims.to_vec(),
        wtype: RfmType::SparseCsr {
            rows: rows as u64,
            cols: cols as u64,
            nnz: nnz as u64,
            index_bits: 32,
            value_type: 0,
        },
        offset: payload_offset,
        size: payload_size as u64,
    });

    Ok(())
}

pub(super) fn convert_mpo_tensor(
    tensor: &TensorView,
    chi_max: u32,
    use_gpu: bool,
    base_name: &str,
    writer: &mut dyn Write,
    current_offset: &mut u64,
    entries: &mut Vec<RfmTensorEntry>,
    align_offset: &impl Fn(&mut dyn Write, &mut u64) -> Result<(), std::io::Error>,
) -> Result<(), Box<dyn std::error::Error>> {
    let rows = tensor.dims[0] as usize;
    let cols = tensor.dims[1] as usize;

    let w_f32 = match tensor.ggml_type {
        GgmlType::Q4_0 => dequantize_q4_0_to_f32(tensor.data, tensor.element_count()),
        GgmlType::Q4_K => {
            let mut out = vec![0.0f32; tensor.element_count()];
            rocmforge::cpu::quant::embed_q4_k(0, tensor.data, &mut out, tensor.element_count());
            out
        }
        GgmlType::Q6_K => dequantize_q6_k_to_f32(tensor.data, tensor.element_count()),
        GgmlType::F32 => bytes_to_f32(tensor.data),
        other => {
            return Err(format!("Unsupported source type for MPO conversion: {:?}", other).into())
        }
    };

    let chi = (chi_max as usize).min(rows.min(cols));
    let (u_sigma, vt) = svd_decompose(&w_f32, rows, cols, chi, base_name, use_gpu)?;
    let site_dims: Vec<u64> = vec![1, rows as u64, chi as u64, 1, chi as u64, cols as u64, 1, 1];

    let mut site_data = Vec::with_capacity(u_sigma.len() + vt.len());
    site_data.extend_from_slice(&u_sigma);
    site_data.extend_from_slice(&vt);

    align_offset(writer, current_offset)?;
    let payload_offset = *current_offset;

    for &val in &site_data {
        writer.write_all(&val.to_le_bytes())?;
    }

    let payload_size = site_data.len() * 4;
    *current_offset += payload_size as u64;

    entries.push(RfmTensorEntry {
        name: base_name.to_string(),
        dims: site_dims,
        wtype: RfmType::Mpo {
            n_sites: 2,
            chi_max,
            value_type: 0,
        },
        offset: payload_offset,
        size: payload_size as u64,
    });

    Ok(())
}

pub(super) fn convert_svd_sparse_tensor(
    tensor: &TensorView,
    k_rank: u32,
    use_gpu: bool,
    sparse_threshold: f32,
    residual_prune_threshold: Option<f32>,
    base_name: &str,
    writer: &mut dyn Write,
    current_offset: &mut u64,
    entries: &mut Vec<RfmTensorEntry>,
    align_offset: &impl Fn(&mut dyn Write, &mut u64) -> Result<(), std::io::Error>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let in_dim = tensor.dims[0] as usize;
    let out_dim = tensor.dims[1] as usize;

    let w_f32 = match tensor.ggml_type {
        GgmlType::Q4_0 => dequantize_q4_0_to_f32(tensor.data, tensor.element_count()),
        GgmlType::Q4_K => {
            let mut out = vec![0.0f32; tensor.element_count()];
            rocmforge::cpu::quant::embed_q4_k(0, tensor.data, &mut out, tensor.element_count());
            out
        }
        GgmlType::Q6_K => dequantize_q6_k_to_f32(tensor.data, tensor.element_count()),
        GgmlType::F32 => bytes_to_f32(tensor.data),
        other => {
            return Err(format!(
                "Unsupported source type for SVD+sparse conversion: {:?}",
                other
            )
            .into())
        }
    };

    let min_mn = out_dim.min(in_dim);
    let k = (k_rank as usize).min(min_mn);
    let (u_sigma, vt) = svd_decompose(&w_f32, out_dim, in_dim, k, base_name, use_gpu)?;
    let low_rank_approx = matmul(&u_sigma, &vt, out_dim, k, in_dim);

    let mut residual: Vec<f32> = w_f32
        .iter()
        .zip(low_rank_approx.iter())
        .map(|(w, l)| w - l)
        .collect();

    if let Some(prune_mag) = residual_prune_threshold {
        let mut zeroed = 0usize;
        for r in &mut residual {
            if r.abs() < prune_mag {
                *r = 0.0;
                zeroed += 1;
            }
        }
        println!(
            "    magnitude pruned {}/{} residual elements (|r| < {:.4})",
            zeroed,
            residual.len(),
            prune_mag
        );
    }

    let count = residual.len();
    let sample_size = count.min(4096);
    let step = if count > sample_size {
        count / sample_size
    } else {
        1
    };
    let nnz_sample = (0..sample_size)
        .filter(|&i| {
            let idx = i * step;
            idx < residual.len() && residual[idx].abs() > 1e-6
        })
        .count();
    let nnz_ratio = nnz_sample as f32 / sample_size as f32;

    if nnz_ratio >= sparse_threshold {
        println!(
            "    residual nnz {:.2}% >= threshold {:.2}% → Q4 fallback",
            nnz_ratio * 100.0,
            sparse_threshold * 100.0
        );
        convert_svd_quant_tensor(
            tensor,
            k_rank,
            use_gpu,
            base_name,
            writer,
            current_offset,
            entries,
            align_offset,
        )?;
        return Ok(false);
    }

    let rows = out_dim;
    let cols = in_dim;
    let mut values: Vec<f32> = Vec::new();
    let mut col_indices: Vec<u32> = Vec::new();
    let mut row_offsets: Vec<u32> = vec![0u32; rows + 1];

    for i in 0..rows {
        for j in 0..cols {
            let v = residual[i * cols + j];
            if v.abs() > 1e-6 {
                values.push(v);
                col_indices.push(j as u32);
            }
        }
        row_offsets[i + 1] = values.len() as u32;
    }
    let nnz = values.len();

    align_offset(writer, current_offset)?;
    let base_offset = *current_offset;
    for &off in &row_offsets {
        writer.write_all(&off.to_le_bytes())?;
    }
    for &col in &col_indices {
        writer.write_all(&col.to_le_bytes())?;
    }
    for &val in &values {
        writer.write_all(&val.to_le_bytes())?;
    }
    let base_size = ((rows + 1 + nnz) * 4 + nnz * 4) as u64;
    *current_offset += base_size;

    entries.push(RfmTensorEntry {
        name: base_name.to_string(),
        dims: tensor.dims.to_vec(),
        wtype: RfmType::SvdSparseCsr {
            k: k_rank,
            rows: rows as u64,
            cols: cols as u64,
            nnz: nnz as u64,
            index_bits: 32,
            value_type: 0,
        },
        offset: base_offset,
        size: base_size,
    });

    align_offset(writer, current_offset)?;
    let u_offset = *current_offset;
    for &x in &u_sigma {
        writer.write_all(&x.to_le_bytes())?;
    }
    let u_size = (u_sigma.len() * 4) as u64;
    *current_offset += u_size;
    entries.push(RfmTensorEntry {
        name: format!("{}.svd_u", base_name),
        dims: vec![k_rank as u64, out_dim as u64],
        wtype: RfmType::F32,
        offset: u_offset,
        size: u_size,
    });

    align_offset(writer, current_offset)?;
    let v_offset = *current_offset;
    for &x in &vt {
        writer.write_all(&x.to_le_bytes())?;
    }
    let v_size = (vt.len() * 4) as u64;
    *current_offset += v_size;
    entries.push(RfmTensorEntry {
        name: format!("{}.svd_v", base_name),
        dims: vec![in_dim as u64, k_rank as u64],
        wtype: RfmType::F32,
        offset: v_offset,
        size: v_size,
    });

    println!(
        "    residual nnz {:.2}% ({}/{} elements), sparse CSR {} nnz",
        nnz_ratio * 100.0,
        nnz,
        rows * cols,
        nnz
    );

    Ok(true)
}

/// Convert a 3D MoE expert tensor to per-expert MPO (2-site tensor network) format.
///
/// Each expert is approximated by SVD rank-`chi_max` factors stored as MPO sites.
/// No sparse residual is retained; the approximation error is pure truncation loss.
pub(super) fn convert_moe_expert_mpo(
    tensor: &TensorView,
    chi_max: u32,
    use_gpu: bool,
    base_name: &str,
    writer: &mut dyn Write,
    current_offset: &mut u64,
    entries: &mut Vec<RfmTensorEntry>,
    align_offset: &impl Fn(&mut dyn Write, &mut u64) -> Result<(), std::io::Error>,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        tensor.dims.len(),
        3,
        "convert_moe_expert_mpo requires 3D tensor"
    );
    let cols = tensor.dims[0] as usize;
    let rows = tensor.dims[1] as usize;
    let n_experts = tensor.dims[2] as usize;
    let chi = (chi_max as usize).min(rows.min(cols));
    let total_elements = cols * rows * n_experts;

    let w_f32 = match tensor.ggml_type {
        GgmlType::Q4_0 => dequantize_q4_0_to_f32(tensor.data, total_elements),
        GgmlType::Q4_K => {
            let mut out = vec![0.0f32; total_elements];
            rocmforge::cpu::quant::embed_q4_k(0, tensor.data, &mut out, total_elements);
            out
        }
        GgmlType::Q6_K => dequantize_q6_k_to_f32(tensor.data, total_elements),
        GgmlType::F32 => bytes_to_f32(tensor.data),
        other => return Err(format!("unsupported type for MoE MPO: {:?}", other).into()),
    };

    println!(
        "    {} experts, rows={}, cols={}, chi_max={}",
        n_experts, rows, cols, chi
    );

    let (all_u, all_v) = svd_batch_experts(&w_f32, rows, cols, chi, n_experts, base_name, use_gpu)?;

    // Site dims for 2-site MPO: [1, rows, chi, 1, chi, cols, 1, 1]
    let site_dims: Vec<u32> = vec![1, rows as u32, chi as u32, 1, chi as u32, cols as u32, 1, 1];

    align_offset(writer, current_offset)?;
    let base_offset = *current_offset;

    // Write site_dims first (8 u32s)
    for &d in &site_dims {
        writer.write_all(&d.to_le_bytes())?;
    }

    // Write all expert site data: U_sigma followed by V^T for each expert
    for e in 0..n_experts {
        let u_offset = e * rows * chi;
        let v_offset = e * chi * cols;
        for &val in &all_u[u_offset..u_offset + rows * chi] {
            writer.write_all(&val.to_le_bytes())?;
        }
        for &val in &all_v[v_offset..v_offset + chi * cols] {
            writer.write_all(&val.to_le_bytes())?;
        }
    }

    let site_data_size = n_experts * (rows * chi + chi * cols);
    let payload_size = site_dims.len() * 4 + site_data_size * 4;
    *current_offset += payload_size as u64;

    entries.push(RfmTensorEntry {
        name: base_name.to_string(),
        dims: tensor.dims.to_vec(),
        wtype: RfmType::MoeExpertMpo {
            n_experts: n_experts as u32,
            n_sites: 2,
            chi_max: chi as u32,
            rows: rows as u64,
            cols: cols as u64,
            value_type: 0,
        },
        offset: base_offset,
        size: payload_size as u64,
    });

    println!(
        "    MoE expert MPO: {} experts, chi_max={}, payload={} bytes",
        n_experts, chi, payload_size
    );

    Ok(())
}

pub(super) fn convert_moe_expert_svd_sparse(
    tensor: &TensorView,
    k_rank: u32,
    use_gpu: bool,
    sparse_threshold: Option<f32>,
    residual_prune_threshold: Option<f32>,
    use_fwht: bool,
    mpo_chi_max: Option<u32>,
    base_name: &str,
    writer: &mut dyn Write,
    current_offset: &mut u64,
    entries: &mut Vec<RfmTensorEntry>,
    align_offset: &impl Fn(&mut dyn Write, &mut u64) -> Result<(), std::io::Error>,
) -> Result<bool, Box<dyn std::error::Error>> {
    assert_eq!(
        tensor.dims.len(),
        3,
        "convert_moe_expert_svd_sparse requires 3D tensor"
    );
    let cols = tensor.dims[0] as usize;
    let rows = tensor.dims[1] as usize;
    let n_experts = tensor.dims[2] as usize;
    let k = (k_rank as usize).min(rows.min(cols));
    let total_elements = cols * rows * n_experts;

    let mut w_f32 = match tensor.ggml_type {
        GgmlType::Q4_0 => dequantize_q4_0_to_f32(tensor.data, total_elements),
        GgmlType::Q4_K => {
            let mut out = vec![0.0f32; total_elements];
            rocmforge::cpu::quant::embed_q4_k(0, tensor.data, &mut out, total_elements);
            out
        }
        GgmlType::Q6_K => dequantize_q6_k_to_f32(tensor.data, total_elements),
        GgmlType::F32 => bytes_to_f32(tensor.data),
        other => return Err(format!("unsupported type for MoE SVD+sparse: {:?}", other).into()),
    };

    if use_fwht {
        println!("    [FWHT] Rotating MoE expert weights before SVD...");
        let scale = 1.0 / (cols as f32).sqrt();
        for e in 0..n_experts {
            let offset = e * rows * cols;
            let expert_w = &mut w_f32[offset..offset + rows * cols];
            expert_w.par_chunks_mut(cols).for_each(|row_slice| {
                fwht_inplace(row_slice);
                for x in row_slice.iter_mut() {
                    *x *= scale;
                }
            });
        }
    }

    let mut all_rp = Vec::<u32>::with_capacity(n_experts * (rows + 1));
    let mut all_ci = Vec::<u32>::new();
    let mut all_vals = Vec::<f32>::new();
    let mut expert_nnz = Vec::<u32>::with_capacity(n_experts);

    println!(
        "    {} experts, rows={}, cols={}, k={}",
        n_experts, rows, cols, k
    );

    let (all_u, all_v) = svd_batch_experts(&w_f32, rows, cols, k, n_experts, base_name, use_gpu)?;

    for e in 0..n_experts {
        let slice = &w_f32[e * rows * cols..(e + 1) * rows * cols];
        let u_sigma = &all_u[e * rows * k..(e + 1) * rows * k];
        let vt = &all_v[e * k * cols..(e + 1) * k * cols];
        let low_rank = matmul(u_sigma, vt, rows, k, cols);

        let mut residual: Vec<f32> = slice
            .iter()
            .zip(low_rank.iter())
            .map(|(w, l)| w - l)
            .collect();
        if let Some(mag) = residual_prune_threshold {
            for r in &mut residual {
                if r.abs() < mag {
                    *r = 0.0;
                }
            }
        }

        let mut row_ptr = vec![0u32; rows + 1];
        let mut col_idx = Vec::<u32>::new();
        let mut values = Vec::<f32>::new();
        for r in 0..rows {
            for c in 0..cols {
                let v = residual[r * cols + c];
                if v.abs() > 1e-9 {
                    col_idx.push(c as u32);
                    values.push(v);
                }
            }
            row_ptr[r + 1] = values.len() as u32;
        }
        let nnz = values.len();

        all_rp.extend_from_slice(&row_ptr);
        all_ci.extend_from_slice(&col_idx);
        all_vals.extend_from_slice(&values);
        expert_nnz.push(nnz as u32);
    }

    let total_nnz = all_ci.len();
    let avg_density = total_nnz as f64 / (rows * cols * n_experts).max(1) as f64;

    if sparse_threshold.map_or(false, |t| avg_density > t as f64) {
        if let Some(chi) = mpo_chi_max {
            let chi_usize = (chi as usize).min(rows.min(cols));
            println!(
                "    residual {:.1}% dense > threshold → MPO fallback (chi_max={})",
                avg_density * 100.0,
                chi_usize
            );

            // Site dims for 2-site MPO: [1, rows, chi, 1, chi, cols, 1, 1]
            let site_dims: Vec<u32> = vec![
                1,
                rows as u32,
                chi_usize as u32,
                1,
                chi_usize as u32,
                cols as u32,
                1,
                1,
            ];

            align_offset(writer, current_offset)?;
            let base_offset = *current_offset;

            // Write site_dims first (8 u32s)
            for &d in &site_dims {
                writer.write_all(&d.to_le_bytes())?;
            }

            // If chi_max <= k, truncate existing U/V. Otherwise recompute SVD.
            let (all_u_mpo, all_v_mpo) = if chi_usize <= k {
                let mut u_trunc = Vec::with_capacity(n_experts * rows * chi_usize);
                let mut v_trunc = Vec::with_capacity(n_experts * chi_usize * cols);
                for e in 0..n_experts {
                    let u_off = e * rows * k;
                    let v_off = e * k * cols;
                    u_trunc.extend_from_slice(&all_u[u_off..u_off + rows * chi_usize]);
                    v_trunc.extend_from_slice(&all_v[v_off..v_off + chi_usize * cols]);
                }
                (u_trunc, v_trunc)
            } else {
                svd_batch_experts(&w_f32, rows, cols, chi_usize, n_experts, base_name, use_gpu)?
            };

            // Write all expert site data: U_sigma followed by V^T for each expert
            for e in 0..n_experts {
                let u_offset = e * rows * chi_usize;
                let v_offset = e * chi_usize * cols;
                for &val in &all_u_mpo[u_offset..u_offset + rows * chi_usize] {
                    writer.write_all(&val.to_le_bytes())?;
                }
                for &val in &all_v_mpo[v_offset..v_offset + chi_usize * cols] {
                    writer.write_all(&val.to_le_bytes())?;
                }
            }

            let site_data_size = n_experts * (rows * chi_usize + chi_usize * cols);
            let payload_size = site_dims.len() * 4 + site_data_size * 4;
            *current_offset += payload_size as u64;

            entries.push(RfmTensorEntry {
                name: base_name.to_string(),
                dims: tensor.dims.to_vec(),
                wtype: RfmType::MoeExpertMpo {
                    n_experts: n_experts as u32,
                    n_sites: 2,
                    chi_max: chi_usize as u32,
                    rows: rows as u64,
                    cols: cols as u64,
                    value_type: 0,
                },
                offset: base_offset,
                size: payload_size as u64,
            });

            return Ok(true);
        }

        println!(
            "    residual {:.1}% dense > threshold → passthrough original",
            avg_density * 100.0
        );
        align_offset(writer, current_offset)?;
        let base_offset = *current_offset;
        let wtype = rfm_type_for_tensor(tensor, false, false);
        let payload_size = pack_tensor(tensor, writer, wtype.clone())?;
        *current_offset += payload_size;
        entries.push(RfmTensorEntry {
            name: base_name.to_string(),
            dims: tensor.dims.to_vec(),
            wtype,
            offset: base_offset,
            size: payload_size,
        });
        return Ok(false);
    }

    align_offset(writer, current_offset)?;
    let base_offset = *current_offset;

    for &x in &all_u {
        writer.write_all(&x.to_le_bytes())?;
    }
    for &x in &all_v {
        writer.write_all(&x.to_le_bytes())?;
    }
    for &x in &all_rp {
        writer.write_all(&x.to_le_bytes())?;
    }
    for &x in &all_ci {
        writer.write_all(&x.to_le_bytes())?;
    }
    for &x in &all_vals {
        writer.write_all(&x.to_le_bytes())?;
    }
    for &x in &expert_nnz {
        writer.write_all(&x.to_le_bytes())?;
    }

    let payload_size = (all_u.len() + all_v.len() + all_vals.len()) as u64 * 4
        + (all_rp.len() + all_ci.len() + expert_nnz.len()) as u64 * 4;
    *current_offset += payload_size;

    let wtype = if use_fwht {
        RfmType::MoeExpertSvdFwhtSparse {
            n_experts: n_experts as u32,
            k: k as u32,
            rows: rows as u64,
            cols: cols as u64,
            total_nnz: total_nnz as u64,
            index_bits: 32,
            value_type: 0,
        }
    } else {
        RfmType::MoeExpertSvdSparse {
            n_experts: n_experts as u32,
            k: k as u32,
            rows: rows as u64,
            cols: cols as u64,
            total_nnz: total_nnz as u64,
            index_bits: 32,
            value_type: 0,
        }
    };

    entries.push(RfmTensorEntry {
        name: base_name.to_string(),
        dims: tensor.dims.to_vec(),
        wtype,
        offset: base_offset,
        size: payload_size,
    });

    let avg_nnz = if n_experts > 0 {
        total_nnz as f64 / n_experts as f64
    } else {
        0.0
    };
    let sparsity = 1.0 - avg_nnz / (rows * cols).max(1) as f64;
    println!(
        "    avg nnz {:.0}/{} per expert ({:.1}% sparse), total_nnz={}",
        avg_nnz,
        rows * cols,
        sparsity * 100.0,
        total_nnz
    );

    Ok(true)
}

pub(super) fn convert_svd_quant_tensor(
    tensor: &TensorView,
    k_rank: u32,
    use_gpu: bool,
    base_name: &str,
    writer: &mut dyn Write,
    current_offset: &mut u64,
    entries: &mut Vec<RfmTensorEntry>,
    align_offset: &impl Fn(&mut dyn Write, &mut u64) -> Result<(), std::io::Error>,
) -> Result<(), Box<dyn std::error::Error>> {
    let in_dim = tensor.dims[0] as usize;
    let out_dim = tensor.dims[1] as usize;

    let w_f32 = match tensor.ggml_type {
        GgmlType::Q4_0 => dequantize_q4_0_to_f32(tensor.data, tensor.element_count()),
        GgmlType::Q4_K => {
            let mut out = vec![0.0f32; tensor.element_count()];
            rocmforge::cpu::quant::embed_q4_k(0, tensor.data, &mut out, tensor.element_count());
            out
        }
        GgmlType::Q6_K => dequantize_q6_k_to_f32(tensor.data, tensor.element_count()),
        GgmlType::Q8_0 => dequantize_q8_0_to_f32(tensor.data, tensor.element_count()),
        GgmlType::F16 => dequantize_f16_to_f32(tensor.data, tensor.element_count()),
        GgmlType::F32 => bytes_to_f32(tensor.data),
        other => {
            return Err(format!("Unsupported source type for SVD conversion: {:?}", other).into())
        }
    };

    println!("    Running SVD-Quant offline decomposition...");
    let min_mn = out_dim.min(in_dim);
    let k = (k_rank as usize).min(min_mn);
    let (u_k, vt_k) = svd_decompose(&w_f32, out_dim, in_dim, k, base_name, use_gpu)?;
    let low_rank_approx = matmul(&u_k, &vt_k, out_dim, k, in_dim);

    let mut residual = vec![0.0f32; out_dim * in_dim];
    for i in 0..out_dim * in_dim {
        residual[i] = w_f32[i] - low_rank_approx[i];
    }

    let q_residual = quantize_matrix_q4_0(&residual);
    let num_gguf_blocks = q_residual.len() / 18;
    let rfm_blocks = num_gguf_blocks / 8;

    let mut scales = Vec::with_capacity(rfm_blocks * 8 * 2);
    let zero_points = vec![0u8; rfm_blocks * 16];
    let mut nibbles = Vec::with_capacity(rfm_blocks * 128);

    for b in 0..rfm_blocks {
        let base_idx = b * 8;
        for i in 0..8 {
            let g_block = &q_residual[(base_idx + i) * 18..(base_idx + i + 1) * 18];
            scales.push(g_block[0]);
            scales.push(g_block[1]);
            nibbles.extend_from_slice(&g_block[2..18]);
        }
    }

    align_offset(writer, current_offset)?;
    let base_offset = *current_offset;
    writer.write_all(&scales)?;
    writer.write_all(&zero_points)?;
    writer.write_all(&nibbles)?;
    let base_size = (scales.len() + zero_points.len() + nibbles.len()) as u64;
    *current_offset += base_size;

    entries.push(RfmTensorEntry {
        name: base_name.to_string(),
        dims: tensor.dims.to_vec(),
        wtype: RfmType::Q4SvdQuant { k: k_rank },
        offset: base_offset,
        size: base_size,
    });

    align_offset(writer, current_offset)?;
    let u_offset = *current_offset;
    let mut u_bytes = Vec::with_capacity(u_k.len() * 4);
    for &x in &u_k {
        u_bytes.extend_from_slice(&x.to_le_bytes());
    }
    writer.write_all(&u_bytes)?;
    let u_size = u_bytes.len() as u64;
    *current_offset += u_size;

    entries.push(RfmTensorEntry {
        name: format!("{}.svd_u", base_name),
        dims: vec![k_rank as u64, out_dim as u64],
        wtype: RfmType::F32,
        offset: u_offset,
        size: u_size,
    });

    align_offset(writer, current_offset)?;
    let v_offset = *current_offset;
    let mut v_bytes = Vec::with_capacity(vt_k.len() * 4);
    for &x in &vt_k {
        v_bytes.extend_from_slice(&x.to_le_bytes());
    }
    writer.write_all(&v_bytes)?;
    let v_size = v_bytes.len() as u64;
    *current_offset += v_size;

    entries.push(RfmTensorEntry {
        name: format!("{}.svd_v", base_name),
        dims: vec![in_dim as u64, k_rank as u64],
        wtype: RfmType::F32,
        offset: v_offset,
        size: v_size,
    });

    Ok(())
}

/// Parse the layer index from a tensor name, supporting both `blk.N.` and `layers.N.` prefixes.
pub(super) fn parse_layer_idx(name: &str) -> Option<usize> {
    if let Some(idx) = name.find("blk.") {
        let rest = &name[idx + 4..];
        let end = rest.find('.').unwrap_or(rest.len());
        rest[..end].parse().ok()
    } else if let Some(idx) = name.find("layers.") {
        let rest = &name[idx + 7..];
        let end = rest.find('.').unwrap_or(rest.len());
        rest[..end].parse().ok()
    } else {
        None
    }
}

/// Returns true when the tensor should receive SVD correction during conversion.
///
/// Covers 3-D MoE expert tensors (`ffn_*_exps`) and 2-D weight matrices for
/// attention and FFN paths. Respects the `svd_attn_only` flag that restricts
/// SVD to attention projections only.
pub(super) fn should_svd_tensor(name: &str, tensor: &TensorView, svd_attn_only: bool) -> bool {
    // 3D MoE expert tensors: [cols, rows, n_experts]
    if tensor.dims.len() == 3 {
        if svd_attn_only {
            return false;
        }
        let n_experts = tensor.dims[2] as usize;
        let rows = tensor.dims[1] as usize;
        let cols = tensor.dims[0] as usize;
        if n_experts < 2 || rows < 64 || cols < 64 {
            return false;
        }
        return matches!(
            tensor.ggml_type,
            GgmlType::Q4_0
                | GgmlType::Q4_K
                | GgmlType::Q6_K
                | GgmlType::Q8_0
                | GgmlType::F16
                | GgmlType::F32
        ) && name.ends_with(".weight")
            && (name.contains("ffn_gate_exps")
                || name.contains("ffn_up_exps")
                || name.contains("ffn_down_exps"));
    }

    if tensor.dims.len() != 2 {
        return false;
    }

    // Skip tensors where either dimension is too small for meaningful SVD correction
    // (e.g. ffn_gate_inp_shexp.weight with dims=[2048,1]).
    if tensor.dims.iter().any(|&d| d < 64) {
        return false;
    }

    if svd_attn_only {
        return matches!(
            tensor.ggml_type,
            GgmlType::F32
                | GgmlType::F16
                | GgmlType::Q4_0
                | GgmlType::Q4_K
                | GgmlType::Q6_K
                | GgmlType::Q8_0
        ) && name.ends_with(".weight")
            && (name.contains("attn_q")
                || name.contains("attn_k")
                || name.contains("attn_v")
                || name.contains("attn_output")
                || name.contains("attn_gate"));
    }

    matches!(
        tensor.ggml_type,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::Q4_0
            | GgmlType::Q4_K
            | GgmlType::Q6_K
            | GgmlType::Q8_0
    ) && name.ends_with(".weight")
        && (name.contains("attn_q")
            || name.contains("attn_k")
            || name.contains("attn_v")
            || name.contains("attn_output")
            || name.contains("attn_gate")
            || name.contains("ssm_alpha")
            || name.contains("ssm_beta")
            || name.contains("ssm_out")
            || name.contains("ffn_gate")
            || name.contains("ffn_up")
            || name.contains("ffn_down"))
}

/// Align the file write position and tracked offset to a 256-byte boundary.
pub(super) fn align_to_256(writer: &mut dyn Write, offset: &mut u64) -> Result<(), std::io::Error> {
    let remainder = *offset % 256;
    if remainder > 0 {
        let padding = 256 - remainder;
        let pad_bytes = vec![0u8; padding as usize];
        writer.write_all(&pad_bytes)?;
        *offset += padding;
    }
    Ok(())
}

/// Convert all tensors from `gguf` and write their payloads into `out_file`.
///
/// Populates `entries` with the tensor table and advances `current_offset` for
/// each written payload. Honors all conversion options (SVD, sparse, MPO, etc.).
pub(super) fn convert_all_tensors(
    gguf: &GgufFile,
    options: &ConvertOptions,
    use_gpu: bool,
    out_file: &mut dyn Write,
    current_offset: &mut u64,
    entries: &mut Vec<RfmTensorEntry>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut tensor_names: Vec<String> = gguf.tensor_names().map(str::to_string).collect();
    tensor_names.sort();

    for tensor_name in tensor_names {
        if let Some(layer_idx) = parse_layer_idx(&tensor_name) {
            if let Some(ml) = options.max_layers {
                if layer_idx >= ml as usize {
                    continue;
                }
            }
        }

        let tensor = gguf
            .tensor(&tensor_name)?
            .ok_or_else(|| format!("tensor disappeared during conversion: {}", tensor_name))?;
        align_to_256(out_file, current_offset)?;

        if tensor.dims.len() == 3 && should_svd_tensor(&tensor_name, &tensor, options.svd_attn_only)
        {
            if let Some(k_val) = options.svd_k {
                let used_sparse = convert_moe_expert_svd_sparse(
                    &tensor,
                    k_val,
                    use_gpu,
                    options.sparse_threshold,
                    options.residual_prune_threshold,
                    options.use_fwht,
                    options.mpo_chi_max,
                    &tensor_name,
                    out_file,
                    current_offset,
                    entries,
                    &align_to_256,
                )?;
                if used_sparse {
                    println!(
                        "  MoE expert SVD+sparse (FWHT={}): {} ({} experts, k={})",
                        options.use_fwht, tensor_name, tensor.dims[2], k_val
                    );
                } else if let Some(chi_max) = options.mpo_chi_max {
                    println!(
                        "  MoE expert MPO fallback: {} ({} experts, chi_max={})",
                        tensor_name, tensor.dims[2], chi_max
                    );
                } else {
                    println!("  MoE passthrough: {} (residual too dense)", tensor_name);
                }
                continue;
            }
        }

        if let (Some(k_val), Some(threshold)) = (
            options
                .svd_k
                .filter(|_| should_svd_tensor(&tensor_name, &tensor, options.svd_attn_only)),
            options
                .sparse_threshold
                .filter(|_| should_compress_tensor(&tensor_name, &tensor)),
        ) {
            let used_sparse = convert_svd_sparse_tensor(
                &tensor,
                k_val,
                use_gpu,
                threshold,
                options.residual_prune_threshold,
                &tensor_name,
                out_file,
                current_offset,
                entries,
                &align_to_256,
            )?;
            if used_sparse {
                println!(
                    "  SVD+sparse residual: {} rank {} (sparse CSR residual)",
                    tensor_name, k_val
                );
            } else {
                println!(
                    "  SVD+sparse→dense fallback: {} rank {} (residual too dense, using Q4)",
                    tensor_name, k_val
                );
            }
        } else if let Some(k_val) = options
            .svd_k
            .filter(|_| should_svd_tensor(&tensor_name, &tensor, options.svd_attn_only))
        {
            convert_svd_quant_tensor(
                &tensor,
                k_val,
                use_gpu,
                &tensor_name,
                out_file,
                current_offset,
                entries,
                &align_to_256,
            )?;
            println!("  SVD: {} rank {}", tensor_name, k_val);
        } else if let Some(threshold) = options
            .sparse_threshold
            .filter(|_| should_compress_tensor(&tensor_name, &tensor))
        {
            let nnz_ratio = estimate_nnz_ratio(&tensor);
            if nnz_ratio < threshold {
                convert_sparse_csr_tensor(
                    &tensor,
                    &tensor_name,
                    out_file,
                    current_offset,
                    entries,
                    &align_to_256,
                )?;
                println!(
                    "  Converted to sparse CSR: {} (nnz ratio {:.2}%)",
                    tensor_name,
                    nnz_ratio * 100.0
                );
            } else {
                let wtype = rfm_type_for_tensor(&tensor, options.mq4, options.mq6);
                let payload_size = pack_tensor(&tensor, out_file, wtype)?;
                entries.push(RfmTensorEntry {
                    name: tensor_name.clone(),
                    dims: tensor.dims.to_vec(),
                    wtype,
                    offset: *current_offset,
                    size: payload_size,
                });
                *current_offset += payload_size;
                println!(
                    "  Packed tensor: {} with type {:?} (sparse skipped: nnz ratio {:.2}%)",
                    tensor_name,
                    wtype,
                    nnz_ratio * 100.0
                );
            }
        } else if let Some(chi_max) = options
            .mpo_chi_max
            .filter(|_| should_compress_tensor(&tensor_name, &tensor))
        {
            convert_mpo_tensor(
                &tensor,
                chi_max,
                use_gpu,
                &tensor_name,
                out_file,
                current_offset,
                entries,
                &align_to_256,
            )?;
            println!(
                "  Converted to MPO: {} with chi_max {}",
                tensor_name, chi_max
            );
        } else {
            let wtype = rfm_type_for_tensor(&tensor, options.mq4, options.mq6);
            let payload_size = pack_tensor(&tensor, out_file, wtype)?;
            entries.push(RfmTensorEntry {
                name: tensor_name.clone(),
                dims: tensor.dims.to_vec(),
                wtype,
                offset: *current_offset,
                size: payload_size,
            });
            *current_offset += payload_size;
            println!("  Packed tensor: {} with type {:?}", tensor_name, wtype);
        }
    }

    Ok(())
}
