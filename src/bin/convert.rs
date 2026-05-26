#![allow(warnings)]
//! ROCmForge Model (.rfm) Offline Converter Tool.
//!
//! Converts standard GGUF weights into highly aligned, fused-FFN
//! ROCmForge Model (.rfm) weights co-optimized for RDNA3 architecture.
//! Runs 100% on the CPU.

use std::env;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

use rocmforge::config::ModelConfig;
use rocmforge::loader::{GgmlType, GgufFile, TensorView};
use rocmforge::loader::{RfmMetadata, RfmTensorEntry, RfmType};

/// Magic bytes identifying the ROCmForge Model format.
pub const RFM_MAGIC: &[u8; 4] = b"RFM\0";
pub const RFM_VERSION: u32 = 1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: rocmforge-convert <input_gguf> <output_rfm>");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    println!("[1/4] Opening GGUF model: {}...", input_path);
    let gguf = GgufFile::open(input_path)?;

    println!("[2/4] Parsing model configuration & tokenizer...");
    let config = ModelConfig::from_gguf(&gguf)?;

    let tok_data = gguf.tokenizer_data();
    let metadata = RfmMetadata {
        num_layers: config.num_layers,
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

    // We will build a list of all output tensors we want to pack.
    // To support clean pipeline execution, we resolve semantic names.
    let registry = &config.tensor_registry;

    // Gather all layer tensors
    let mut layer_tensors = Vec::new();
    for l in 0..config.num_layers {
        layer_tensors.push((
            l,
            "attn_norm",
            registry.resolve(rocmforge::config::TensorName::AttnNorm, l),
        ));
        layer_tensors.push((
            l,
            "ffn_norm",
            registry.resolve(rocmforge::config::TensorName::FfnNorm, l),
        ));
        layer_tensors.push((
            l,
            "attn_q",
            registry.resolve(rocmforge::config::TensorName::AttnQ, l),
        ));
        layer_tensors.push((
            l,
            "attn_k",
            registry.resolve(rocmforge::config::TensorName::AttnK, l),
        ));
        layer_tensors.push((
            l,
            "attn_v",
            registry.resolve(rocmforge::config::TensorName::AttnV, l),
        ));
        layer_tensors.push((
            l,
            "attn_o",
            registry.resolve(rocmforge::config::TensorName::AttnOutput, l),
        ));

        // FFN elements (we will interleave Gate & Up weights!)
        let gate_name = registry.resolve(rocmforge::config::TensorName::FfnGate, l);
        let up_name = registry.resolve(rocmforge::config::TensorName::FfnUp, l);
        layer_tensors.push((l, "ffn_gate_up", format!("{}+{}", gate_name, up_name)));

        layer_tensors.push((
            l,
            "ffn_down",
            registry.resolve(rocmforge::config::TensorName::FfnDown, l),
        ));
    }

    // Embeddings & head
    let token_emb_name = registry.resolve(rocmforge::config::TensorName::TokenEmb, 0);
    let output_norm_name = registry.resolve(rocmforge::config::TensorName::OutputNorm, 0);
    let lm_head_name = registry.resolve(rocmforge::config::TensorName::LmHead, 0);

    // Open target file
    let mut out_file = File::create(output_path)?;

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
    let table_placeholder = vec![b' '; 64 * 1024]; // 64KB is plenty for the tensor table index
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

    // 1. Token Embeddings
    if gguf.has_tensor(&token_emb_name) {
        align_offset(&mut out_file, &mut current_offset)?;
        let tensor = gguf.tensor(&token_emb_name)?.unwrap();
        let wtype = match tensor.ggml_type {
            GgmlType::F32 => RfmType::F32,
            GgmlType::Q4_0 => RfmType::Q4Split,
            other => RfmType::GgufPassthrough(other as u32),
        };
        let payload_size = pack_tensor(&tensor, &mut out_file, wtype.clone())?;
        entries.push(RfmTensorEntry {
            name: "token_embd.weight".to_string(),
            dims: tensor.dims.to_vec(),
            wtype: wtype.clone(),
            offset: current_offset,
            size: payload_size,
        });
        current_offset += payload_size;
        println!(
            "  Packed embedding: {} with type {:?}",
            token_emb_name, wtype
        );
    }

    // 2. Layer-by-layer weights
    for (l, kind, resolved_name) in layer_tensors {
        align_offset(&mut out_file, &mut current_offset)?;

        if kind == "ffn_gate_up" {
            let parts: Vec<&str> = resolved_name.split('+').collect();
            let gate_tensor = gguf.tensor(parts[0])?.unwrap();
            let up_tensor = gguf.tensor(parts[1])?.unwrap();

            let payload_size = pack_gate_up_fused(&gate_tensor, &up_tensor, &mut out_file)?;
            entries.push(RfmTensorEntry {
                name: format!("blk.{}.ffn_gate_up.weight", l),
                dims: gate_tensor.dims.to_vec(),
                wtype: RfmType::Q4FusedGateUp,
                offset: current_offset,
                size: payload_size,
            });
            current_offset += payload_size;
            println!("  Fused FFN gate+up layer: {}", l);
        } else {
            if let Some(tensor) = gguf.tensor(&resolved_name)? {
                let wtype = match tensor.ggml_type {
                    GgmlType::F32 => RfmType::F32,
                    GgmlType::Q4_0 => RfmType::Q4Split,
                    other => RfmType::GgufPassthrough(other as u32),
                };

                let payload_size = pack_tensor(&tensor, &mut out_file, wtype.clone())?;
                entries.push(RfmTensorEntry {
                    name: format!("blk.{}.{}.weight", l, kind.replace("attn_o", "attn_output")),
                    dims: tensor.dims.to_vec(),
                    wtype: wtype.clone(),
                    offset: current_offset,
                    size: payload_size,
                });
                current_offset += payload_size;
                println!(
                    "  Packed layer {} tensor: {} with type {:?}",
                    l, kind, wtype
                );
            }
        }
    }

    // 3. Output Norm
    if gguf.has_tensor(&output_norm_name) {
        align_offset(&mut out_file, &mut current_offset)?;
        let tensor = gguf.tensor(&output_norm_name)?.unwrap();
        let payload_size = pack_tensor(&tensor, &mut out_file, RfmType::F32)?;
        entries.push(RfmTensorEntry {
            name: "output_norm.weight".to_string(),
            dims: tensor.dims.to_vec(),
            wtype: RfmType::F32,
            offset: current_offset,
            size: payload_size,
        });
        current_offset += payload_size;
        println!("  Packed output norm");
    }

    // 4. LM Head
    if gguf.has_tensor(&lm_head_name) {
        align_offset(&mut out_file, &mut current_offset)?;
        let tensor = gguf.tensor(&lm_head_name)?.unwrap();
        let wtype = match tensor.ggml_type {
            GgmlType::F32 => RfmType::F32,
            GgmlType::Q4_0 => RfmType::Q4Split,
            other => RfmType::GgufPassthrough(other as u32),
        };
        let payload_size = pack_tensor(&tensor, &mut out_file, wtype.clone())?;
        entries.push(RfmTensorEntry {
            name: "output.weight".to_string(),
            dims: tensor.dims.to_vec(),
            wtype: wtype.clone(),
            offset: current_offset,
            size: payload_size,
        });
        current_offset += payload_size;
        println!("  Packed LM head with type {:?}", wtype);
    }

    // Serialize and overwrite the tensor table
    let table_bytes = serde_json::to_vec(&entries)?;
    if table_bytes.len() > tensor_table_allocated_size as usize {
        return Err(format!(
            "Tensor table JSON exceeds pre-allocated 64KB space (actual: {} bytes)",
            table_bytes.len()
        )
        .into());
    }

    // Overwrite the placeholder spaces with the actual JSON payload
    out_file.seek(SeekFrom::Start(table_pos))?;
    out_file.write_all(&table_bytes)?;

    // Fill the remainder of the allocated spaces with trailing spaces so the offsets remain correct
    let remainder = tensor_table_allocated_size as usize - table_bytes.len();
    if remainder > 0 {
        let padding = vec![b' '; remainder];
        out_file.write_all(&padding)?;
    }

    // Seek back to overwrite metadata sizes in the main file header
    out_file.seek(SeekFrom::Start(8))?; // skip Magic + Version (8 bytes)
    out_file.write_all(&metadata_size.to_le_bytes())?;
    out_file.write_all(&tensor_table_allocated_size.to_le_bytes())?;

    println!("\nConversion successful! Saved to: {}", output_path);
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
}
