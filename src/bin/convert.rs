#![allow(warnings)]
//! ROCmForge Model (.rfm) Offline Converter Tool.
//!
//! Converts standard GGUF weights into highly aligned, fused-FFN
//! ROCmForge Model (.rfm) weights co-optimized for RDNA3 architecture.
//! Runs 100% on the CPU.

use std::env;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

use rayon::prelude::*;

use rocmforge::config::ModelConfig;
use rocmforge::loader::{GgmlType, GgufFile, TensorView};
use rocmforge::loader::{RfmMetadata, RfmTensorEntry, RfmType};

/// Magic bytes identifying the ROCmForge Model format.
pub const RFM_MAGIC: &[u8; 4] = b"RFM\0";
pub const RFM_VERSION: u32 = 1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let mut svd_k: Option<u32> = None;
    let mut sparse_threshold: Option<f32> = None;
    let mut residual_prune_threshold: Option<f32> = None;
    let mut mpo_chi_max: Option<u32> = None;
    let mut max_layers: Option<u32> = None;
    let mut use_fwht = false;
    let mut force_gpu = false;
    let mut force_cpu = false;
    let mut kv_lora_dim: Option<usize> = None;
    let mut kv_frame_codec = false;
    let mut adastate_anchors = false;
    let mut kv_quant_bits: Option<usize> = None;
    let mut qjl_scale: Option<f32> = None;
    let mut svd_attn_only = false;
    let mut input_path = String::new();
    let mut output_path = String::new();

    let mut idx = 1;
    while idx < args.len() {
        if args[idx] == "--svd-k" {
            if idx + 1 < args.len() {
                svd_k = Some(args[idx + 1].parse().expect("Invalid SVD rank k"));
                idx += 2;
            } else {
                eprintln!("Error: --svd-k requires a rank value");
                std::process::exit(1);
            }
        } else if args[idx] == "--sparse-threshold" {
            if idx + 1 < args.len() {
                sparse_threshold = Some(args[idx + 1].parse().expect("Invalid sparse threshold"));
                idx += 2;
            } else {
                eprintln!("Error: --sparse-threshold requires a value (e.g., 0.01)");
                std::process::exit(1);
            }
        } else if args[idx] == "--residual-prune-threshold" {
            if idx + 1 < args.len() {
                residual_prune_threshold = Some(
                    args[idx + 1]
                        .parse()
                        .expect("Invalid residual prune threshold"),
                );
                idx += 2;
            } else {
                eprintln!(
                    "Error: --residual-prune-threshold requires a magnitude value (e.g., 0.02)"
                );
                std::process::exit(1);
            }
        } else if args[idx] == "--mpo-chi-max" {
            if idx + 1 < args.len() {
                mpo_chi_max = Some(args[idx + 1].parse().expect("Invalid MPO chi max"));
                idx += 2;
            } else {
                eprintln!("Error: --mpo-chi-max requires a value");
                std::process::exit(1);
            }
        } else if args[idx] == "--max-layers" {
            if idx + 1 < args.len() {
                max_layers = Some(args[idx + 1].parse().expect("Invalid max layers"));
                idx += 2;
            } else {
                eprintln!("Error: --max-layers requires a value");
                std::process::exit(1);
            }
        } else if args[idx] == "--use-fwht" {
            use_fwht = true;
            idx += 1;
        } else if args[idx] == "--gpu" {
            force_gpu = true;
            idx += 1;
        } else if args[idx] == "--cpu" {
            force_cpu = true;
            idx += 1;
        } else if args[idx] == "--kv-lora-dim" {
            if idx + 1 < args.len() {
                let dim: usize = args[idx + 1].parse().expect("Invalid KV LoRA dim");
                let padded = dim.next_power_of_two();
                if padded != dim {
                    println!("💡 Model Converter: Padding --kv-lora-dim from {} to {} to satisfy Walsh-Hadamard power-of-two constraint.", dim, padded);
                }
                kv_lora_dim = Some(padded);
                idx += 2;
            } else {
                eprintln!("Error: --kv-lora-dim requires a value");
                std::process::exit(1);
            }
        } else if args[idx] == "--kv-quant-bits" {
            if idx + 1 < args.len() {
                kv_quant_bits = Some(args[idx + 1].parse().expect("Invalid KV quant bits"));
                idx += 2;
            } else {
                eprintln!("Error: --kv-quant-bits requires a value");
                std::process::exit(1);
            }
        } else if args[idx] == "--qjl-scale" {
            if idx + 1 < args.len() {
                qjl_scale = Some(args[idx + 1].parse().expect("Invalid QJL scale"));
                idx += 2;
            } else {
                eprintln!("Error: --qjl-scale requires a value");
                std::process::exit(1);
            }
        } else if args[idx] == "--kv-frame-codec" {
            kv_frame_codec = true;
            idx += 1;
        } else if args[idx] == "--adastate-anchors" {
            adastate_anchors = true;
            idx += 1;
        } else if args[idx] == "--svd-attn-only" {
            svd_attn_only = true;
            idx += 1;
        } else {
            if input_path.is_empty() {
                input_path = args[idx].clone();
            } else if output_path.is_empty() {
                output_path = args[idx].clone();
            }
            idx += 1;
        }
    }

    if input_path.is_empty() || output_path.is_empty() {
        eprintln!(concat!(
            "Usage: rocmforge-convert <input_gguf> <output_rfm>\n",
            "  [--svd-k <K>]                      SVD rank for low-rank correction\n",
            "  [--sparse-threshold <T>]            Combined with --svd-k: store sparse residual\n",
            "                                      when residual nnz ratio < T (0..1)\n",
            "  [--residual-prune-threshold <M>]    Combined with --svd-k+--sparse-threshold:\n",
            "                                      zero residual elements |r| < M before CSR\n",
            "  [--use-fwht]                        Apply Fast Walsh-Hadamard Transform before SVD\n",
            "  [--mpo-chi-max <C>]                 MPO bond dimension for FFN compression\n",
            "  [--max-layers <L>]                  Only convert first L layers (smoke testing)\n",
            "  [--gpu]                             Force GPU SVD (requires rocsolver & --features gpu)\n",
            "  [--cpu]                             Force CPU SVD (use power-iteration, slow)\n",
            "  [--kv-lora-dim <D>]                 Set latent KV cache compression dimension\n",
            "  [--kv-quant-bits <B>]               Set KV cache quantization bits (e.g. 3 for TurboQuant)\n",
            "  [--kv-frame-codec]                  Enable differential KV cache compression\n",
            "  [--svd-attn-only]                   Only apply SVD to attention projections (Q, K, V, O)\n",
            "  [--adastate-anchors]                Enable AdaState self-evolving dynamic anchors",
        ));
        std::process::exit(1);
    }

    if force_gpu && force_cpu {
        eprintln!("Error: Cannot specify both --gpu and --cpu");
        std::process::exit(1);
    }

    let use_gpu = if force_cpu {
        false
    } else if force_gpu {
        #[cfg(not(feature = "gpu"))]
        {
            eprintln!("Error: GPU support was not enabled at compile time. Please build with --features gpu.");
            std::process::exit(1);
        }
        #[cfg(feature = "gpu")]
        {
            true
        }
    } else {
        cfg!(feature = "gpu")
    };

    if use_gpu {
        println!("⚡ GPU acceleration enabled (rocSOLVER SVD)");
        #[cfg(feature = "gpu")]
        {
            println!("⚡ Initializing GPU device & checking VRAM safety...");
            let caps = match rocmforge::gpu::detect() {
                Some(c) => c,
                None => {
                    eprintln!("Error: GPU detection failed (no compatible GPU found). Refusing to proceed with GPU SVD.");
                    std::process::exit(1);
                }
            };
            match rocmforge::gpu::GpuDevice::init(caps.device_id) {
                Ok(_) => {
                    println!(
                        "⚡ GPU initialized successfully on device {} ({})",
                        caps.device_id, caps.device_name
                    );
                }
                Err(e) => {
                    eprintln!("Error: GPU initialization failed: {e}. Build with CPU support or fix ROCm environment.");
                    std::process::exit(1);
                }
            }
        }
    } else {
        println!("⚠️ Running SVD on CPU (GPU acceleration not enabled)");
    }

    println!("[1/4] Opening GGUF model: {}...", input_path);
    let gguf = GgufFile::open(&input_path)?;

    println!("[2/4] Parsing model configuration & tokenizer...");
    let config = ModelConfig::from_gguf(&gguf)?;

    let tok_data = gguf.tokenizer_data();
    let actual_num_layers = if let Some(ml) = max_layers {
        ml as usize
    } else {
        config.num_layers
    };

    let metadata = RfmMetadata {
        num_layers: actual_num_layers,
        hidden_size: config.hidden_size,
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        intermediate_size: config.intermediate_size,
        vocab_size: config.vocab_size,
        max_seq_len: config.max_seq_len,
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        rope_neox: config.rope_neox,
        use_attention_bias: config.use_attention_bias,
        architecture: config.architecture.clone(),

        tokens: tok_data.tokens.clone(),
        merges: tok_data.merges.clone(),
        bos_token_id: tok_data.bos_token_id,
        eos_token_id: tok_data.eos_token_id,
        unk_token_id: tok_data.unk_token_id,
        tokenizer_model: tok_data.model.clone(),
        tokenizer_pre: tok_data.pre.clone(),
        add_bos: tok_data.add_bos,
        add_eos: tok_data.add_eos,
        kv_lora_dim,
        kv_frame_codec_enabled: Some(kv_frame_codec),
        adastate_anchors_enabled: Some(adastate_anchors),
        kv_quant_bits,
        turboquant_centroids: kv_quant_bits
            .map(|_| vec![-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152]),
        qjl_scale: kv_quant_bits.map(|_| qjl_scale.unwrap_or(0.25f32)),
    };

    let metadata_bytes = serde_json::to_vec(&metadata)?;

    println!("[3/4] Preparing tensor layout mapping...");
    let mut entries = Vec::new();

    // Open target file
    let mut out_file = File::create(&output_path)?;

    // Write placeholder header (24 bytes):
    // Magic (4B) + Version (4B) + Metadata Size (8B) + Tensor Table Size (8B)
    out_file.write_all(RFM_MAGIC)?;
    out_file.write_all(&RFM_VERSION.to_le_bytes())?;
    out_file.write_all(&0u64.to_le_bytes())?; // placeholder metadata size
    out_file.write_all(&0u64.to_le_bytes())?; // placeholder tensor table size

    // Write metadata JSON
    out_file.write_all(&metadata_bytes)?;
    let metadata_size = metadata_bytes.len() as u64;

    // Write placeholder tensor table
    let table_pos = out_file.stream_position()?;
    let table_placeholder = vec![b' '; 4 * 1024 * 1024]; // Large enough for full-model tensor indexes.
    out_file.write_all(&table_placeholder)?;
    let tensor_table_allocated_size = table_placeholder.len() as u64;

    let mut current_offset = 0u64;

    println!("[4/4] Writing and converting weight payload...");

    // Helper function to align payload offsets to 256 bytes
    let align_offset = |file: &mut File, offset: &mut u64| -> Result<(), std::io::Error> {
        let remainder = *offset % 256;
        if remainder > 0 {
            let padding = 256 - remainder;
            let pad_bytes = vec![0u8; padding as usize];
            file.write_all(&pad_bytes)?;
            *offset += padding;
        }
        Ok(())
    };

    // Generic complete conversion path: preserve every GGUF tensor under its
    // original name. Architecture-specific runtime loaders can then decide
    // which tensors they understand without the converter silently dropping
    // fused QKV, SSM, MoE expert, or future tensors.
    let mut tensor_names: Vec<String> = gguf.tensor_names().map(str::to_string).collect();
    tensor_names.sort();

    for tensor_name in tensor_names {
        if let Some(layer_idx) = parse_layer_idx(&tensor_name) {
            if let Some(ml) = max_layers {
                if layer_idx >= ml as usize {
                    // Skip layers beyond max_layers
                    continue;
                }
            }
        }

        let tensor = gguf
            .tensor(&tensor_name)?
            .ok_or_else(|| format!("tensor disappeared during conversion: {}", tensor_name))?;
        align_offset(&mut out_file, &mut current_offset)?;

        // 3D MoE expert tensors: always use per-expert SVD+sparse when --svd-k is set.
        // Must be checked BEFORE the 2D combined path to avoid falling through.
        if tensor.dims.len() == 3 && should_svd_tensor(&tensor_name, &tensor, svd_attn_only) {
            if let Some(k_val) = svd_k {
                let used_sparse = convert_moe_expert_svd_sparse(
                    &tensor,
                    k_val,
                    use_gpu,
                    sparse_threshold,
                    residual_prune_threshold,
                    use_fwht,
                    &tensor_name,
                    &mut out_file,
                    &mut current_offset,
                    &mut entries,
                    &align_offset,
                )?;
                if used_sparse {
                    println!(
                        "  MoE expert SVD+sparse (FWHT={}): {} ({} experts, k={})",
                        use_fwht, tensor_name, tensor.dims[2], k_val
                    );
                } else {
                    println!("  MoE passthrough: {} (residual too dense)", tensor_name);
                }
                continue;
            }
        }

        // Combined SVD+sparse: when both --svd-k and --sparse-threshold are set
        // for a suitable tensor, decompose with SVD, check if the residual is
        // sparse, and store the residual as CSR.  Falls back to Q4 residual when
        // the residual is too dense.
        if let (Some(k_val), Some(threshold)) = (
            svd_k.filter(|_| should_svd_tensor(&tensor_name, &tensor, svd_attn_only)),
            sparse_threshold.filter(|_| should_compress_tensor(&tensor_name, &tensor)),
        ) {
            let used_sparse = convert_svd_sparse_tensor(
                &tensor,
                k_val,
                use_gpu,
                threshold,
                residual_prune_threshold,
                &tensor_name,
                &mut out_file,
                &mut current_offset,
                &mut entries,
                &align_offset,
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
        } else if let Some(k_val) =
            svd_k.filter(|_| should_svd_tensor(&tensor_name, &tensor, svd_attn_only))
        {
            convert_svd_quant_tensor(
                &tensor,
                k_val,
                use_gpu,
                &tensor_name,
                &mut out_file,
                &mut current_offset,
                &mut entries,
                &align_offset,
            )?;
            println!("  SVD: {} rank {}", tensor_name, k_val);
        } else if let Some(threshold) =
            sparse_threshold.filter(|_| should_compress_tensor(&tensor_name, &tensor))
        {
            let nnz_ratio = estimate_nnz_ratio(&tensor);
            if nnz_ratio < threshold {
                convert_sparse_csr_tensor(
                    &tensor,
                    &tensor_name,
                    &mut out_file,
                    &mut current_offset,
                    &mut entries,
                    &align_offset,
                )?;
                println!(
                    "  Converted to sparse CSR: {} (nnz ratio {:.2}%)",
                    tensor_name,
                    nnz_ratio * 100.0
                );
            } else {
                let wtype = rfm_type_for_tensor(&tensor);
                let payload_size = pack_tensor(&tensor, &mut out_file, wtype)?;
                entries.push(RfmTensorEntry {
                    name: tensor_name.clone(),
                    dims: tensor.dims.to_vec(),
                    wtype,
                    offset: current_offset,
                    size: payload_size,
                });
                current_offset += payload_size;
                println!(
                    "  Packed tensor: {} with type {:?} (sparse skipped: nnz ratio {:.2}%)",
                    tensor_name,
                    wtype,
                    nnz_ratio * 100.0
                );
            }
        } else if let Some(chi_max) =
            mpo_chi_max.filter(|_| should_compress_tensor(&tensor_name, &tensor))
        {
            convert_mpo_tensor(
                &tensor,
                chi_max,
                use_gpu,
                &tensor_name,
                &mut out_file,
                &mut current_offset,
                &mut entries,
                &align_offset,
            )?;
            println!(
                "  Converted to MPO: {} with chi_max {}",
                tensor_name, chi_max
            );
        } else {
            let wtype = rfm_type_for_tensor(&tensor);
            let payload_size = pack_tensor(&tensor, &mut out_file, wtype)?;
            entries.push(RfmTensorEntry {
                name: tensor_name.clone(),
                dims: tensor.dims.to_vec(),
                wtype,
                offset: current_offset,
                size: payload_size,
            });
            current_offset += payload_size;
            println!("  Packed tensor: {} with type {:?}", tensor_name, wtype);
        }
    }

    let table_bytes = serde_json::to_vec(&entries)?;
    if table_bytes.len() > tensor_table_allocated_size as usize {
        return Err(format!(
            "Tensor table JSON exceeds pre-allocated table space (actual: {} bytes, allocated: {} bytes)",
            table_bytes.len(),
            tensor_table_allocated_size
        )
        .into());
    }

    out_file.seek(SeekFrom::Start(table_pos))?;
    out_file.write_all(&table_bytes)?;

    let remainder = tensor_table_allocated_size as usize - table_bytes.len();
    if remainder > 0 {
        let padding = vec![b' '; remainder];
        out_file.write_all(&padding)?;
    }

    out_file.seek(SeekFrom::Start(8))?;
    out_file.write_all(&metadata_size.to_le_bytes())?;
    out_file.write_all(&tensor_table_allocated_size.to_le_bytes())?;

    println!(
        "\nConversion successful! Saved {} tensors to: {}",
        entries.len(),
        output_path
    );
    Ok(())
}

/// Rearranges and packs a standard GGUF tensor into .rfm layout.
fn pack_tensor(
    tensor: &TensorView,
    writer: &mut File,
    wtype: RfmType,
) -> Result<u64, Box<dyn std::error::Error>> {
    match wtype {
        RfmType::F32 => {
            writer.write_all(tensor.data)?;
            Ok(tensor.data.len() as u64)
        }
        RfmType::GgufPassthrough(_) => {
            writer.write_all(tensor.data)?;
            Ok(tensor.data.len() as u64)
        }
        RfmType::Q4Split => {
            if tensor.ggml_type != GgmlType::Q4_0 {
                return Err(format!(
                    "Unsupported GGUF quant type for split conversion: {:?}",
                    tensor.ggml_type
                )
                .into());
            }

            let num_gguf_blocks = tensor.data.len() / 18;
            let rfm_blocks = num_gguf_blocks / 8;
            if num_gguf_blocks % 8 != 0 {
                return Err(format!(
                    "Tensor {} blocks count is not divisible by 8: {}",
                    tensor.name, num_gguf_blocks
                )
                .into());
            }

            // Buffers for split arrays
            let mut scales = Vec::with_capacity(rfm_blocks * 8 * 2);
            let zero_points = vec![0u8; rfm_blocks * 16];
            let mut nibbles = Vec::with_capacity(rfm_blocks * 128);

            for b in 0..rfm_blocks {
                let base_idx = b * 8;
                for i in 0..8 {
                    let g_block = &tensor.data[(base_idx + i) * 18..(base_idx + i + 1) * 18];
                    scales.push(g_block[0]);
                    scales.push(g_block[1]);
                    nibbles.extend_from_slice(&g_block[2..18]);
                }
            }

            // Write split components sequentially
            writer.write_all(&scales)?;
            writer.write_all(&zero_points)?;
            writer.write_all(&nibbles)?;

            let total_size = (scales.len() + zero_points.len() + nibbles.len()) as u64;
            Ok(total_size)
        }
        _ => Err("Invalid tensor packing layout selected".into()),
    }
}

fn rfm_type_for_tensor(tensor: &TensorView) -> RfmType {
    match tensor.ggml_type {
        GgmlType::F32 => RfmType::F32,
        GgmlType::Q4_0 => RfmType::Q4Split,
        other => RfmType::GgufPassthrough(other as u32),
    }
}

fn parse_layer_idx(name: &str) -> Option<usize> {
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

fn should_svd_tensor(name: &str, tensor: &TensorView, svd_attn_only: bool) -> bool {
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
            GgmlType::Q4_0 | GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::F32
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
            GgmlType::F32 | GgmlType::Q4_0 | GgmlType::Q4_K | GgmlType::Q6_K
        ) && name.ends_with(".weight")
            && (name.contains("attn_q")
                || name.contains("attn_k")
                || name.contains("attn_v")
                || name.contains("attn_output")
                || name.contains("attn_gate"));
    }

    matches!(
        tensor.ggml_type,
        GgmlType::F32 | GgmlType::Q4_0 | GgmlType::Q4_K | GgmlType::Q6_K
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

fn should_compress_tensor(name: &str, tensor: &TensorView) -> bool {
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

/// Estimate the nnz ratio of a tensor by dequantizing a sample and counting nonzeros.
fn estimate_nnz_ratio(tensor: &TensorView) -> f32 {
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
        _ => return 1.0f32, // Unknown type, don't compress
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

/// Convert a dense tensor to sparse CSR format and write to RFM.
fn convert_sparse_csr_tensor(
    tensor: &TensorView,
    base_name: &str,
    writer: &mut File,
    current_offset: &mut u64,
    entries: &mut Vec<RfmTensorEntry>,
    align_offset: &impl Fn(&mut File, &mut u64) -> Result<(), std::io::Error>,
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

    // Build CSR from dense
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
    let index_bits: u8 = 32;
    let value_type: u32 = 0; // F32

    // Write payload: row_offsets (u32) + col_indices (u32) + values (f32)
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
            index_bits,
            value_type,
        },
        offset: payload_offset,
        size: payload_size as u64,
    });

    Ok(())
}

/// Convert a dense tensor to MPO (Matrix Product Operator) format using SVD-based tensor train.
fn convert_mpo_tensor(
    tensor: &TensorView,
    chi_max: u32,
    use_gpu: bool,
    base_name: &str,
    writer: &mut File,
    current_offset: &mut u64,
    entries: &mut Vec<RfmTensorEntry>,
    align_offset: &impl Fn(&mut File, &mut u64) -> Result<(), std::io::Error>,
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

    // Simple 2-site MPO decomposition: factor matrix into A1 * A2
    // A1: [rows, chi], A2: [chi, cols]
    let chi = (chi_max as usize).min(rows.min(cols));
    let (u_sigma, vt) = svd_decompose(&w_f32, rows, cols, chi, base_name, use_gpu)?;

    // site_dims layout: [chi_l, d_out, chi_r, 1] per site
    // Site 0: [1, rows, chi, 1]
    // Site 1: [chi, cols, 1, 1]
    let site_dims: Vec<u64> = vec![1, rows as u64, chi as u64, 1, chi as u64, cols as u64, 1, 1];

    // Flatten site data: u_sigma (rows * chi) followed by vt (chi * cols)
    let mut site_data = Vec::with_capacity(u_sigma.len() + vt.len());
    site_data.extend_from_slice(&u_sigma);
    site_data.extend_from_slice(&vt);

    // Write payload
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
            value_type: 0, // F32
        },
        offset: payload_offset,
        size: payload_size as u64,
    });

    Ok(())
}

/// Fuses two independent FFN Gate and Up Q4_0 tensors into a single interleaved layout.
fn pack_gate_up_fused(
    gate: &TensorView,
    up: &TensorView,
    writer: &mut File,
) -> Result<u64, Box<dyn std::error::Error>> {
    if gate.ggml_type != GgmlType::Q4_0 || up.ggml_type != GgmlType::Q4_0 {
        return Err("Only Q4_0 GGUF tensors can be fused into Gate-Up layout".into());
    }

    if gate.dims != up.dims {
        return Err("Gate and Up tensor dimensions must match exactly for fusion".into());
    }

    // Row-by-row layout mapping
    let intermediate_size = gate.dims[1] as usize;
    let hidden_size = gate.dims[0] as usize;

    let num_gguf_blocks_row = hidden_size / 32;
    let rfm_blocks_row = num_gguf_blocks_row / 8;

    if num_gguf_blocks_row % 8 != 0 {
        return Err(format!(
            "Hidden size {} is not a multiple of 256 elements",
            hidden_size
        )
        .into());
    }

    // Output arrays
    let mut scales = Vec::new();
    let mut zero_points = Vec::new();
    let mut nibbles = Vec::new();

    for r in 0..intermediate_size {
        let gate_row_offset = r * num_gguf_blocks_row * 18;
        let up_row_offset = r * num_gguf_blocks_row * 18;

        for b in 0..rfm_blocks_row {
            let base_gguf_blk = b * 8;

            // 1. Gather Gate Block (256 elements)
            let mut gate_scales = [0u8; 16];
            let mut gate_nibbles = [0u8; 128];
            for i in 0..8 {
                let blk_bytes = &gate.data[gate_row_offset + (base_gguf_blk + i) * 18
                    ..gate_row_offset + (base_gguf_blk + i + 1) * 18];
                gate_scales[i * 2] = blk_bytes[0];
                gate_scales[i * 2 + 1] = blk_bytes[1];
                gate_nibbles[i * 16..(i + 1) * 16].copy_from_slice(&blk_bytes[2..18]);
            }

            // 2. Gather Up Block (256 elements)
            let mut up_scales = [0u8; 16];
            let mut up_nibbles = [0u8; 128];
            for i in 0..8 {
                let blk_bytes = &up.data[up_row_offset + (base_gguf_blk + i) * 18
                    ..up_row_offset + (base_gguf_blk + i + 1) * 18];
                up_scales[i * 2] = blk_bytes[0];
                up_scales[i * 2 + 1] = blk_bytes[1];
                up_nibbles[i * 16..(i + 1) * 16].copy_from_slice(&blk_bytes[2..18]);
            }

            // Write interleaved segments to components:
            // Scales segment: Gate (16B) followed by Up (16B) = 32B total
            scales.extend_from_slice(&gate_scales);
            scales.extend_from_slice(&up_scales);

            // ZPs segment: Gate (16B of zeros) followed by Up (16B of zeros) = 32B total
            zero_points.extend_from_slice(&[0u8; 32]);

            // Nibbles segment: Gate (128B) followed by Up (128B) = 256B total
            nibbles.extend_from_slice(&gate_nibbles);
            nibbles.extend_from_slice(&up_nibbles);
        }
    }

    // Write complete payloads
    writer.write_all(&scales)?;
    writer.write_all(&zero_points)?;
    writer.write_all(&nibbles)?;

    let total_size = (scales.len() + zero_points.len() + nibbles.len()) as u64;
    Ok(total_size)
}

// ── SVD-Quant Low-Rank Outlier Decomposition ────────────────────────────────────────

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for p in 0..k {
            let aip = a[i * k + p];
            for j in 0..n {
                row[j] += aip * b[p * n + j];
            }
        }
    });
    c
}

fn normalize(v: &mut [f32]) -> f32 {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for x in v {
            *x *= inv;
        }
    }
    norm
}

fn orthogonalize(v: &mut [f32], basis: &[Vec<f32>]) {
    for b in basis {
        let dot = v.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        for (x, y) in v.iter_mut().zip(b) {
            *x -= dot * y;
        }
    }
}

fn matvec_w(a: &[f32], m: usize, n: usize, v: &[f32]) -> Vec<f32> {
    a.par_chunks(n)
        .take(m)
        .map(|row| row.iter().zip(v).map(|(x, y)| x * y).sum::<f32>())
        .collect()
}

fn matvec_wt(a: &[f32], m: usize, n: usize, u: &[f32]) -> Vec<f32> {
    (0..n)
        .into_par_iter()
        .map(|col| {
            let mut sum = 0.0f32;
            for row in 0..m {
                sum += a[row * n + col] * u[row];
            }
            sum
        })
        .collect()
}

fn deterministic_seed_vector(len: usize, component: usize) -> Vec<f32> {
    let mut state =
        0x9e37_79b9_7f4a_7c15u64 ^ ((component as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let unit = ((bits >> 40) as f32) / ((1u64 << 24) as f32);
        v.push(unit * 2.0 - 1.0);
    }
    normalize(&mut v);
    v
}

/// Compute top-k SVD for a single `[m, n]` row-major matrix.
/// Uses GPU (rocSOLVER) when the `gpu` feature is enabled and use_gpu is true;
/// falls back to CPU power iteration if GPU is unavailable or returns an error.
/// Returns `(u_scaled [m*k], vt [k*n])` — same layout as `top_k_svd_quant`.
fn svd_decompose(
    a: &[f32],
    m: usize,
    n: usize,
    k: usize,
    name: &str,
    use_gpu: bool,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    if use_gpu {
        #[cfg(feature = "gpu")]
        {
            match rocmforge::gpu::rocsolver::gpu_svd_single(a, m, n, k) {
                Ok(r) => return Ok(r),
                Err(e) => eprintln!("  GPU SVD failed for {name}: {e} — using CPU"),
            }
        }
    }
    let _ = name;
    Ok(top_k_svd_quant(a, m, n, k))
}

/// Compute top-k SVD for `n_experts` row-major `[rows, cols]` matrices packed contiguously.
/// Uses GPU batch SVD when the `gpu` feature is enabled and use_gpu is true;
/// falls back to per-expert CPU power iteration if GPU is unavailable or returns an error.
/// Returns `(all_u_scaled [n_experts*rows*k], all_vt [n_experts*k*cols])`.
fn svd_batch_experts(
    matrices: &[f32],
    rows: usize,
    cols: usize,
    k: usize,
    n_experts: usize,
    name: &str,
    use_gpu: bool,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    if use_gpu {
        #[cfg(feature = "gpu")]
        {
            match rocmforge::gpu::rocsolver::gpu_svd_batch(matrices, rows, cols, k, n_experts) {
                Ok(r) => return Ok(r),
                Err(e) => eprintln!("  GPU batch SVD failed for {name}: {e} — using CPU"),
            }
        }
    }
    let _ = name;
    let mut all_u = Vec::<f32>::with_capacity(n_experts * rows * k);
    let mut all_v = Vec::<f32>::with_capacity(n_experts * k * cols);
    for e in 0..n_experts {
        let slice = &matrices[e * rows * cols..(e + 1) * rows * cols];
        let (u, v) = top_k_svd_quant(slice, rows, cols, k);
        all_u.extend_from_slice(&u);
        all_v.extend_from_slice(&v);
    }
    Ok((all_u, all_v))
}

/// Deterministic top-k low-rank decomposition for SVD-Quant conversion.
///
/// The converter only stores rank-k correction matrices, so building a full
/// n-by-n Jacobi SVD is unnecessary for large LLM projections. This extracts
/// the leading singular directions with power iteration and explicit
/// orthogonalization, returning U with singular values already absorbed and
/// Vt in row-major [k, n] layout.
fn top_k_svd_quant(a: &[f32], m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let k = k.min(m.min(n));
    let iters = 8;
    let mut u_basis: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut v_basis: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut sigmas = Vec::with_capacity(k);

    for component in 0..k {
        let mut v = deterministic_seed_vector(n, component);
        orthogonalize(&mut v, &v_basis);
        if normalize(&mut v) <= 1e-12 {
            break;
        }

        let mut u = vec![0.0f32; m];
        for _ in 0..iters {
            u = matvec_w(a, m, n, &v);
            orthogonalize(&mut u, &u_basis);
            if normalize(&mut u) <= 1e-12 {
                break;
            }

            v = matvec_wt(a, m, n, &u);
            orthogonalize(&mut v, &v_basis);
            if normalize(&mut v) <= 1e-12 {
                break;
            }
        }

        u = matvec_w(a, m, n, &v);
        orthogonalize(&mut u, &u_basis);
        let sigma = normalize(&mut u);
        if sigma <= 1e-8 {
            break;
        }

        u_basis.push(u);
        v_basis.push(v);
        sigmas.push(sigma);
    }

    let actual_k = sigmas.len();
    let mut u_sigma = vec![0.0f32; m * k];
    let mut vt = vec![0.0f32; k * n];

    for col in 0..actual_k {
        for row in 0..m {
            u_sigma[row * k + col] = u_basis[col][row] * sigmas[col];
        }
        for j in 0..n {
            vt[col * n + j] = v_basis[col][j];
        }
    }

    (u_sigma, vt)
}

fn dequantize_q4_0_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    let mut out = vec![0.0f32; num_elements];
    for i in 0..num_blocks {
        let block_offset = i * 18;
        let scale = half::f16::from_bits(u16::from_le_bytes([
            data[block_offset],
            data[block_offset + 1],
        ]))
        .to_f32();
        for j in 0..32 {
            let byte_idx = j / 2;
            let nibble_idx = j % 2;
            let val_byte = data[block_offset + 2 + byte_idx];
            let val_nibble = if nibble_idx == 0 {
                val_byte & 0x0F
            } else {
                (val_byte >> 4) & 0x0F
            };
            let qval = (val_nibble as i8) - 8;
            out[i * 32 + j] = qval as f32 * scale;
        }
    }
    out
}

fn dequantize_q6_k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; num_elements];
    rocmforge::cpu::quant::embed_q6_k(0, data, &mut out, num_elements);
    out
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len() / 4];
    for i in 0..out.len() {
        out[i] = f32::from_le_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ]);
    }
    out
}

fn quantize_q4_0_block(block: &[f32]) -> [u8; 18] {
    let mut max_abs = 0.0f32;
    for &x in block {
        if x.abs() > max_abs {
            max_abs = x.abs();
        }
    }
    let scale = max_abs / 8.0;
    let scale_f16 = half::f16::from_f32(scale);
    let scale_f32 = scale_f16.to_f32();
    let inv_scale = if scale_f32 > 1e-10 {
        1.0 / scale_f32
    } else {
        0.0
    };

    let mut q = [0i8; 32];
    for j in 0..32 {
        let val = block[j] * inv_scale;
        q[j] = val.round().clamp(-8.0, 7.0) as i8;
    }

    let mut out = [0u8; 18];
    let scale_bytes = scale_f16.to_bits().to_le_bytes();
    out[0] = scale_bytes[0];
    out[1] = scale_bytes[1];

    for i in 0..16 {
        let low = (q[2 * i] + 8) as u8 & 0x0F;
        let high = (q[2 * i + 1] + 8) as u8 & 0x0F;
        out[2 + i] = low | (high << 4);
    }
    out
}

fn quantize_matrix_q4_0(data: &[f32]) -> Vec<u8> {
    let num_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(num_blocks * 18);
    for i in 0..num_blocks {
        let block = &data[i * 32..(i + 1) * 32];
        let q_block = quantize_q4_0_block(block);
        out.extend_from_slice(&q_block);
    }
    out
}

/// SVD + sparse-residual conversion.
///
/// 1. Dequantise tensor to F32.
/// 2. Compute top-k SVD → low-rank approximation U·Vᵀ.
/// 3. residual = W − U·Vᵀ.
/// 4. If `residual_prune_threshold` is set, zero residual elements with
///    |r| < threshold (magnitude pruning to introduce explicit sparsity).
/// 5. If residual nnz_ratio < `sparse_threshold` → store residual as sparse CSR
///    with type `SvdSparseCsr` and return `true`.
/// 6. Otherwise fall back to `convert_svd_quant_tensor` (Q4 residual) and
///    return `false`.
fn convert_svd_sparse_tensor(
    tensor: &TensorView,
    k_rank: u32,
    use_gpu: bool,
    sparse_threshold: f32,
    residual_prune_threshold: Option<f32>,
    base_name: &str,
    writer: &mut File,
    current_offset: &mut u64,
    entries: &mut Vec<RfmTensorEntry>,
    align_offset: &impl Fn(&mut File, &mut u64) -> Result<(), std::io::Error>,
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

    // residual = W - U·Vᵀ
    let mut residual: Vec<f32> = w_f32
        .iter()
        .zip(low_rank_approx.iter())
        .map(|(w, l)| w - l)
        .collect();

    // Optional magnitude pruning: zero out small residual elements to create
    // explicit sparsity that CSR can exploit.
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

    // Estimate sparsity of the residual (sample up to 4096 elements).
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
        // Residual too dense — fall back to Q4 quantised residual.
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

    // Build full CSR from residual.
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

    // Write sparse residual payload.
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
            value_type: 0, // F32
        },
        offset: base_offset,
        size: base_size,
    });

    // Write SVD U sub-tensor (F32 [out_dim, k]).
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

    // Write SVD V sub-tensor (F32 [k, in_dim]).
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

fn fwht_inplace(a: &mut [f32]) {
    let n = a.len();
    assert!(n.is_power_of_two(), "FWHT length must be a power of 2");
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in 0..h {
                let x = a[i + j];
                let y = a[i + j + h];
                a[i + j] = x + y;
                a[i + j + h] = x - y;
            }
        }
        h *= 2;
    }
}

/// Per-expert SVD + magnitude-pruned sparse CSR residual for 3D MoE expert tensors.
///
/// Tensor dims: `[cols, rows, n_experts]` (GGUF convention).
/// Expert `i` data in flat array: `w_f32[i * rows * cols .. (i+1) * rows * cols]`
/// as a `[rows, cols]` matrix in row-major.
///
/// Payload layout written matches `MoeExpertSvdSparse`:
/// U[n_experts*rows*k] | V[n_experts*k*cols] | row_ptr[n_experts*(rows+1)]
/// | col_idx[total_nnz] | values[total_nnz] | expert_nnz[n_experts]
fn convert_moe_expert_svd_sparse(
    tensor: &TensorView,
    k_rank: u32,
    use_gpu: bool,
    sparse_threshold: Option<f32>,
    residual_prune_threshold: Option<f32>,
    use_fwht: bool,
    base_name: &str,
    writer: &mut File,
    current_offset: &mut u64,
    entries: &mut Vec<RfmTensorEntry>,
    align_offset: &impl Fn(&mut File, &mut u64) -> Result<(), std::io::Error>,
) -> Result<bool, Box<dyn std::error::Error>> {
    assert_eq!(
        tensor.dims.len(),
        3,
        "convert_moe_expert_svd_sparse requires 3D tensor"
    );
    let cols = tensor.dims[0] as usize; // in_dim (fastest-varying in GGUF)
    let rows = tensor.dims[1] as usize; // out_dim
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
            for r in 0..rows {
                let row_slice = &mut expert_w[r * cols..(r + 1) * cols];
                fwht_inplace(row_slice);
                for x in row_slice.iter_mut() {
                    *x *= scale;
                }
            }
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

    // ── CPU residual, prune, and CSR build (fast per expert) ─────────────
    for e in 0..n_experts {
        let slice = &w_f32[e * rows * cols..(e + 1) * rows * cols];
        let u_sigma = &all_u[e * rows * k..(e + 1) * rows * k];
        let vt = &all_v[e * k * cols..(e + 1) * k * cols];

        // low_rank = U_k * Vt_k  [rows, cols]
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

    // CSR is only smaller than the original quantized tensor below ~7% density.
    // When the residual is denser than the threshold, fall back to the original
    // bytes verbatim — guaranteed no larger than the source.
    if sparse_threshold.map_or(false, |t| avg_density > t as f64) {
        println!(
            "    residual {:.1}% dense > threshold → passthrough original",
            avg_density * 100.0
        );
        align_offset(writer, current_offset)?;
        let base_offset = *current_offset;
        let wtype = rfm_type_for_tensor(tensor);
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
            value_type: 0, // F32
        }
    } else {
        RfmType::MoeExpertSvdSparse {
            n_experts: n_experts as u32,
            k: k as u32,
            rows: rows as u64,
            cols: cols as u64,
            total_nnz: total_nnz as u64,
            index_bits: 32,
            value_type: 0, // F32
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

fn convert_svd_quant_tensor(
    tensor: &TensorView,
    k_rank: u32,
    use_gpu: bool,
    base_name: &str,
    writer: &mut File,
    current_offset: &mut u64,
    entries: &mut Vec<RfmTensorEntry>,
    align_offset: &impl Fn(&mut File, &mut u64) -> Result<(), std::io::Error>,
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

    // Split the quantized residual into RFM Q4Split layout
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

    // 1. Write base quantized residual
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

    // 2. Write U sub-tensor (F32 row-major [out_dim, k])
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

    // 3. Write V sub-tensor (F32 row-major [k, in_dim])
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_and_split_roundtrip() {
        // A GGUF Q4_0 block has 2 bytes scale + 16 bytes nibbles.
        // Let's create mock Q4_0 data for 256 elements (8 blocks).
        let mut mock_data = Vec::new();
        for b in 0..8 {
            let scale_f16 = half::f16::from_f32(1.5 + b as f32);
            let scale_bits = scale_f16.to_bits().to_le_bytes();
            mock_data.extend_from_slice(&scale_bits);
            for i in 0..16 {
                mock_data.push(i as u8);
            }
        }

        // Run split packing logic
        let mut rfm_scales = Vec::new();
        let rfm_zps = vec![0u8; 16];
        let mut rfm_nibbles = Vec::new();

        for i in 0..8 {
            let g_block = &mock_data[i * 18..(i + 1) * 18];
            rfm_scales.push(g_block[0]);
            rfm_scales.push(g_block[1]);
            rfm_nibbles.extend_from_slice(&g_block[2..18]);
        }

        assert_eq!(rfm_scales.len(), 16);
        assert_eq!(rfm_zps.len(), 16);
        assert_eq!(rfm_nibbles.len(), 128);

        // Reconstruct the first block's scale
        let bits = u16::from_le_bytes([rfm_scales[0], rfm_scales[1]]);
        let scale = half::f16::from_bits(bits).to_f32();
        assert_eq!(scale, 1.5);
    }

    #[test]
    fn test_top_k_svd_quant_reconstructs_rank_one_matrix() {
        let m = 4;
        let n = 3;
        let left = [2.0f32, -1.0, 0.5, 3.0];
        let right = [1.5f32, -2.0, 0.25];
        let mut a = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                a[row * n + col] = left[row] * right[col];
            }
        }

        let (u_sigma, vt) = top_k_svd_quant(&a, m, n, 1);
        let reconstructed = matmul(&u_sigma, &vt, m, 1, n);
        let max_err = a
            .iter()
            .zip(reconstructed.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);

        assert!(max_err < 1e-4, "rank-one reconstruction error: {max_err}");
    }

    #[test]
    fn test_dequantize_q6_k_zero_block() {
        let data = vec![0u8; rocmforge::cpu::quant::Q6_K_BLOCK_BYTES];
        let out = dequantize_q6_k_to_f32(&data, rocmforge::cpu::quant::Q6_K_BLOCK_ELEMS);
        assert_eq!(out.len(), rocmforge::cpu::quant::Q6_K_BLOCK_ELEMS);
        assert!(out.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn test_convert_sparse_csr_tensor_basic() -> Result<(), Box<dyn std::error::Error>> {
        use std::fs;
        use std::io::Read;

        // Create a simple 4x4 dense matrix with some zeros
        let mut data = vec![0.0f32; 16];
        data[0] = 1.0;
        data[2] = 2.0;
        data[5] = 3.0;
        data[14] = 4.0;
        data[15] = 5.0;

        let dims: Vec<u64> = vec![4, 4];
        let tensor = TensorView {
            name: "test.weight",
            dims: &dims,
            ggml_type: GgmlType::F32,
            data: unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) },
        };

        let tmp_path = std::env::temp_dir().join("test_sparse_csr.rfm");
        let mut file = File::create(&tmp_path)?;
        let mut offset = 0u64;
        let mut entries = Vec::new();

        convert_sparse_csr_tensor(
            &tensor,
            "test.weight",
            &mut file,
            &mut offset,
            &mut entries,
            &|f, o| {
                let rem = *o % 256;
                if rem > 0 {
                    let pad = vec![0u8; (256 - rem) as usize];
                    f.write_all(&pad)?;
                    *o += 256 - rem;
                }
                Ok(())
            },
        )?;
        drop(file);

        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry.name, "test.weight");
        assert_eq!(entry.dims, vec![4, 4]);

        let RfmType::SparseCsr {
            rows,
            cols,
            nnz,
            index_bits,
            value_type,
        } = entry.wtype
        else {
            panic!("Expected SparseCsr type, got {:?}", entry.wtype);
        };
        assert_eq!(rows, 4);
        assert_eq!(cols, 4);
        assert_eq!(nnz, 5);
        assert_eq!(index_bits, 32);
        assert_eq!(value_type, 0);
        assert_eq!(entry.size, 60);

        // Verify payload bytes are written
        let mut file = File::open(&tmp_path)?;
        let mut payload = vec![0u8; entry.size as usize];
        file.seek(std::io::SeekFrom::Start(entry.offset))?;
        file.read_exact(&mut payload)?;

        // First 5 u32s are row_offsets: [0, 2, 3, 3, 5]
        let row_offsets: Vec<u32> = (0..5)
            .map(|i| {
                u32::from_le_bytes([
                    payload[i * 4],
                    payload[i * 4 + 1],
                    payload[i * 4 + 2],
                    payload[i * 4 + 3],
                ])
            })
            .collect();
        assert_eq!(row_offsets, vec![0, 2, 3, 3, 5]);

        fs::remove_file(&tmp_path)?;
        Ok(())
    }

    #[test]
    fn test_convert_mpo_tensor_basic() -> Result<(), Box<dyn std::error::Error>> {
        use std::fs;
        use std::io::Read;

        let left = [1.0f32, 2.0, 3.0, 4.0];
        let right = [0.5f32, 1.0, 1.5];
        let mut data = vec![0.0f32; 12];
        for row in 0..4 {
            for col in 0..3 {
                data[row * 3 + col] = left[row] * right[col];
            }
        }

        let dims: Vec<u64> = vec![4, 3];
        let tensor = TensorView {
            name: "test.weight",
            dims: &dims,
            ggml_type: GgmlType::F32,
            data: unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) },
        };

        let tmp_path = std::env::temp_dir().join("test_mpo.rfm");
        let mut file = File::create(&tmp_path)?;
        let mut offset = 0u64;
        let mut entries = Vec::new();

        convert_mpo_tensor(
            &tensor,
            2,
            false,
            "test.weight",
            &mut file,
            &mut offset,
            &mut entries,
            &|f, o| {
                let rem = *o % 256;
                if rem > 0 {
                    let pad = vec![0u8; (256 - rem) as usize];
                    f.write_all(&pad)?;
                    *o += 256 - rem;
                }
                Ok(())
            },
        )?;
        drop(file);

        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry.name, "test.weight");

        let RfmType::Mpo {
            n_sites,
            chi_max,
            value_type,
        } = entry.wtype
        else {
            panic!("Expected Mpo type, got {:?}", entry.wtype);
        };
        assert_eq!(n_sites, 2);
        assert_eq!(chi_max, 2);
        assert_eq!(value_type, 0);
        assert_eq!(entry.dims, vec![1, 4, 2, 1, 2, 3, 1, 1]);
        assert_eq!(entry.size, 56);

        // Verify payload bytes
        let mut file = File::open(&tmp_path)?;
        let mut payload = vec![0u8; entry.size as usize];
        file.seek(std::io::SeekFrom::Start(entry.offset))?;
        file.read_exact(&mut payload)?;
        assert_eq!(payload.len(), 56);

        fs::remove_file(&tmp_path)?;
        Ok(())
    }
}
