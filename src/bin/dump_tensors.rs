use std::env;
use std::path::Path;

use rocmforge::config::ModelConfig;
use rocmforge::cpu::weights::CpuLayerWeights;
use rocmforge::loader::{GgufFile, RfmFile};

fn main() {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "/home/feanor/Projects/rocmforge/qwen3.6.gguf".to_string());
    let check_cpu_layers = args.any(|arg| arg == "--check-cpu-layers");

    match Path::new(&path).extension().and_then(|ext| ext.to_str()) {
        Some("rfm") => dump_rfm(&path, check_cpu_layers),
        _ => dump_gguf(&path),
    }
}

fn dump_gguf(path: &str) {
    println!("Opening GGUF: {}...", path);
    let gguf = GgufFile::open(path).expect("failed to open GGUF");
    println!("Metadata keys:");
    let mut keys: Vec<&String> = gguf.metadata.raw.keys().collect();
    keys.sort();
    for k in keys {
        let val = gguf.metadata.raw.get(k).map(|s| s.as_str()).unwrap_or("");
        println!("  {} = {}", k, val);
    }
    println!("\nFirst 100 tensor names:");
    let mut names: Vec<&str> = gguf.tensor_names().collect();
    names.sort();
    for (i, name) in names.iter().enumerate().take(100) {
        if let Ok(Some(t)) = gguf.tensor(name) {
            println!(
                "  [{}] {} | ggml_type={:?} | dims={:?}",
                i, t.name, t.ggml_type, t.dims
            );
        }
    }
}

fn dump_rfm(path: &str, check_cpu_layers: bool) {
    println!("Opening RFM: {}...", path);
    let rfm = RfmFile::open(path).expect("failed to open RFM");
    println!(
        "Metadata: arch={} layers={} hidden={} heads={} kv_heads={} head_dim={} intermediate={} vocab={}",
        rfm.metadata.architecture,
        rfm.metadata.num_layers,
        rfm.metadata.hidden_size,
        rfm.metadata.num_heads,
        rfm.metadata.num_kv_heads,
        rfm.metadata.head_dim,
        rfm.metadata.intermediate_size,
        rfm.metadata.vocab_size,
    );
    println!("Tensor count: {}", rfm.tensor_count());

    let mut names: Vec<&str> = rfm.tensor_names().collect();
    names.sort();
    for name in names {
        if let Ok(Some(t)) = rfm.tensor(name) {
            println!(
                "{} | {:?} | dims={:?} | bytes={}",
                t.name,
                t.wtype,
                t.dims,
                t.data.len()
            );
        }
    }

    if check_cpu_layers {
        let config = ModelConfig::from_rfm(&rfm.metadata).expect("failed to build RFM config");
        for layer in 0..config.num_layers {
            let weights =
                CpuLayerWeights::load_rfm(&rfm, layer, &config).expect("failed to load CPU layer");
            println!(
                "checked layer {}: fused_qkv={} q_norm={} ssm={}",
                layer,
                weights.attn_qkv.is_some(),
                weights.attn_q_norm.is_some(),
                weights.ssm.is_some()
            );
        }
    }
}
