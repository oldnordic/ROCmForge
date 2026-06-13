use std::io::Write;
use rocmforge::loader::{GgufFile, RfmTensorEntry};
use super::cli::ConvertOptions;
use super::layout::{pack_tensor, rfm_type_for_tensor};

#[path = "pipeline/utils.rs"]
mod utils;
#[path = "pipeline/conversion.rs"]
pub(crate) mod conversion;

pub(super) use conversion::{convert_mpo_tensor, convert_sparse_csr_tensor};
use utils::*;
use conversion::*;

/// Convert all tensors from `gguf` and write their payloads into `out_file`.
///
/// Populates `entries` with the tensor table and advances `current_offset` for
/// each written payload. Honors all conversion options (SVD, sparse, MPO, etc.).
pub(crate) fn convert_all_tensors(
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
