use std::io::Write;
use rayon::prelude::*;
use rocmforge::loader::{GgmlType, TensorView, RfmTensorEntry, RfmType};
use super::super::quant::{
    bytes_to_f32, dequantize_f16_to_f32, dequantize_q4_0_to_f32, dequantize_q6_k_to_f32,
    dequantize_q8_0_to_f32, quantize_matrix_q4_0,
};
use super::super::math::{matmul, svd_batch_experts, svd_decompose, fwht_inplace};
use super::super::layout::{pack_tensor, rfm_type_for_tensor};

pub(crate) fn convert_sparse_csr_tensor(
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

pub(crate) fn convert_mpo_tensor(
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

pub(crate) fn convert_svd_sparse_tensor(
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

pub(crate) fn convert_moe_expert_mpo(
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

    let site_dims: Vec<u32> = vec![1, rows as u32, chi as u32, 1, chi as u32, cols as u32, 1, 1];

    align_offset(writer, current_offset)?;
    let base_offset = *current_offset;

    for &d in &site_dims {
        writer.write_all(&d.to_le_bytes())?;
    }

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

pub(crate) fn convert_moe_expert_svd_sparse(
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

            for &d in &site_dims {
                writer.write_all(&d.to_le_bytes())?;
            }

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

pub(crate) fn convert_svd_quant_tensor(
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
