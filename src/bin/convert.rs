#![allow(warnings)]
//! ROCmForge Model (.rfm) Offline Converter Tool.
//!
//! Converts standard GGUF weights into highly aligned, fused-FFN
//! ROCmForge Model (.rfm) weights co-optimized for RDNA3 architecture.
//! Runs 100% on the CPU.

#[path = "convert/cli.rs"]
mod cli;
#[path = "convert/layout.rs"]
mod layout;
#[path = "convert/math.rs"]
mod math;
#[path = "convert/pipeline.rs"]
mod pipeline;
#[path = "convert/quant.rs"]
mod quant;

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

use rocmforge::config::ModelConfig;
use rocmforge::loader::{GgmlType, GgufFile, TensorView};
use rocmforge::loader::{RfmMetadata, RfmTensorEntry, RfmType};

use self::cli::parse_args;
use self::pipeline::{convert_all_tensors, convert_mpo_tensor, convert_sparse_csr_tensor};

/// Magic bytes identifying the ROCmForge Model format.
pub const RFM_MAGIC: &[u8; 4] = b"RFM\0";
pub const RFM_VERSION: u32 = 1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_args(std::env::args());

    let use_gpu = if options.force_cpu {
        false
    } else if options.force_gpu {
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
            rocmforge::gpu::binary_vram_safety_preflight(caps.device_id);
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

    println!("[1/4] Opening GGUF model: {}...", options.input_path);
    let gguf = GgufFile::open(&options.input_path)?;

    println!("[2/4] Parsing model configuration & tokenizer...");
    let config = ModelConfig::from_gguf(&gguf)?;

    let tok_data = gguf.tokenizer_data();
    let actual_num_layers = if let Some(ml) = options.max_layers {
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
        kv_lora_dim: options.kv_lora_dim,
        kv_frame_codec_enabled: Some(options.kv_frame_codec),
        adastate_anchors_enabled: Some(options.adastate_anchors),
        kv_quant_bits: options.kv_quant_bits,
        turboquant_centroids: options
            .kv_quant_bits
            .map(|_| vec![-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152]),
        qjl_scale: options
            .kv_quant_bits
            .map(|_| options.qjl_scale.unwrap_or(0.25f32)),
    };

    let metadata_bytes = serde_json::to_vec(&metadata)?;

    println!("[3/4] Preparing tensor layout mapping...");
    let mut entries = Vec::new();

    // Open target file
    let mut out_file = File::create(&options.output_path)?;

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
    convert_all_tensors(
        &gguf,
        &options,
        use_gpu,
        &mut out_file,
        &mut current_offset,
        &mut entries,
    )?;

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
        options.output_path
    );
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
