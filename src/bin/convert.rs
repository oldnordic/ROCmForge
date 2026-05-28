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
    let mut max_layers: Option<u32> = None;
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
        } else if args[idx] == "--max-layers" {
            if idx + 1 < args.len() {
                max_layers = Some(args[idx + 1].parse().expect("Invalid max layers"));
                idx += 2;
            } else {
                eprintln!("Error: --max-layers requires a value");
                std::process::exit(1);
            }
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
        eprintln!(
            "Usage: rocmforge-convert <input_gguf> <output_rfm> [--svd-k <K>] [--max-layers <L>]"
        );
        std::process::exit(1);
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

        if let Some(k_val) = svd_k.filter(|_| should_svd_tensor(&tensor_name, &tensor)) {
            convert_svd_quant_tensor(
                &tensor,
                k_val,
                &tensor_name,
                &mut out_file,
                &mut current_offset,
                &mut entries,
                &align_offset,
            )?;
            println!(
                "  Decomposed & packed tensor: {} with SVD rank {}",
                tensor_name, k_val
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

fn should_svd_tensor(name: &str, tensor: &TensorView) -> bool {
    if tensor.dims.len() != 2 {
        return false;
    }

    // Skip tensors where either dimension is too small for meaningful SVD correction
    // (e.g. ffn_gate_inp_shexp.weight with dims=[2048,1]).
    if tensor.dims.iter().any(|&d| d < 64) {
        return false;
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

fn convert_svd_quant_tensor(
    tensor: &TensorView,
    k_rank: u32,
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
    let (u_k, vt_k) = top_k_svd_quant(&w_f32, out_dim, in_dim, k);

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
}
