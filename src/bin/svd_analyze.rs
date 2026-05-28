//! SVD-Quant Compression Analyzer for LLM models.
//!
//! Accepts GGUF or HuggingFace safetensors (FP16/BF16) as input.
//! Applies SVD outlier correction + Q4 quantization at various rank-k values,
//! reports per-layer compression ratios and reconstruction fidelity.
//!
//! Usage:
//!   cargo run --release --bin svd_analyze -- <model.gguf> [options]
//!   cargo run --release --bin svd_analyze -- <safetensors_dir/> [options]
//!
//! Options:
//!   --k 1,2,4,8,16    SVD ranks to sweep (default: 1,2,4,8,16)
//!   --max-layers N     Only analyze N layers (default: all)
//!   --error-target 0.05  Target error for optimal-k selection (default: 5%)

use rayon::prelude::*;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

// ── Safetensors loader (zero external deps) ──────────────────────────────

#[derive(Debug)]
struct SafeTensorInfo {
    dtype: String,
    shape: Vec<usize>,
    offsets: (usize, usize),
}

struct SafeTensorsFile {
    mmap: memmap2::Mmap,
    tensors: BTreeMap<String, SafeTensorInfo>,
    data_offset: usize,
}

impl SafeTensorsFile {
    fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let header_len = u64::from_le_bytes(mmap[..8].try_into()?) as usize;
        let header_json = &mmap[8..8 + header_len];
        let header: serde_json::Value = serde_json::from_slice(header_json)?;

        let mut tensors = BTreeMap::new();
        if let serde_json::Value::Object(map) = header {
            for (name, meta) in &map {
                if name == "__metadata__" {
                    continue;
                }
                if let serde_json::Value::Object(m) = meta {
                    let dtype = m["dtype"].as_str().unwrap_or("").to_string();
                    let shape = m["shape"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_u64().map(|x| x as usize))
                                .collect()
                        })
                        .unwrap_or_default();
                    let offsets = if let Some(arr) = m["data_offsets"].as_array() {
                        (
                            arr[0].as_u64().unwrap_or(0) as usize,
                            arr[1].as_u64().unwrap_or(0) as usize,
                        )
                    } else {
                        (0, 0)
                    };
                    tensors.insert(
                        name.clone(),
                        SafeTensorInfo {
                            dtype,
                            shape,
                            offsets,
                        },
                    );
                }
            }
        }

        Ok(Self {
            mmap,
            tensors,
            data_offset: 8 + header_len,
        })
    }

    fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    fn load_f32(&self, name: &str) -> Option<Vec<f32>> {
        let info = self.tensors.get(name)?;
        let start = self.data_offset + info.offsets.0;
        let end = self.data_offset + info.offsets.1;
        let data = &self.mmap[start..end];
        let n: usize = info.shape.iter().product();

        match info.dtype.as_str() {
            "F32" => {
                let mut out = vec![0.0f32; n];
                for i in 0..n {
                    out[i] = f32::from_le_bytes([
                        data[i * 4],
                        data[i * 4 + 1],
                        data[i * 4 + 2],
                        data[i * 4 + 3],
                    ]);
                }
                Some(out)
            }
            "BF16" => {
                let mut out = vec![0.0f32; n];
                for i in 0..n {
                    let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                    out[i] = bf16_to_f32(bits);
                }
                Some(out)
            }
            "F16" => {
                let mut out = vec![0.0f32; n];
                for i in 0..n {
                    let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                    out[i] = half::f16::from_bits(bits).to_f32();
                }
                Some(out)
            }
            _ => None,
        }
    }

    fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|i| i.shape.as_slice())
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    let f32_bits = (bits as u32) << 16;
    f32::from_bits(f32_bits)
}

struct ShardedSafeTensors {
    files: Vec<SafeTensorsFile>,
    weight_map: BTreeMap<String, usize>,
}

impl ShardedSafeTensors {
    fn open(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let index_path = dir.join("model.safetensors.index.json");
        let single_path = dir.join("model.safetensors");

        if index_path.exists() {
            let mut index_data = String::new();
            File::open(&index_path)?.read_to_string(&mut index_data)?;
            let index: serde_json::Value = serde_json::from_str(&index_data)?;

            let mut file_set: BTreeMap<String, usize> = BTreeMap::new();
            let mut weight_map = BTreeMap::new();

            if let serde_json::Value::Object(map) = index {
                if let Some(serde_json::Value::Object(wm_map)) = map.get("weight_map") {
                    for (tensor_name, file_val) in wm_map {
                        let fname = file_val.as_str().unwrap_or("");
                        let idx = if let Some(&i) = file_set.get(fname) {
                            i
                        } else {
                            let i = file_set.len();
                            file_set.insert(fname.to_string(), i);
                            i
                        };
                        weight_map.insert(tensor_name.clone(), idx);
                    }
                }
            }

            let mut files = Vec::new();
            let mut sorted_files: Vec<_> = file_set.into_iter().collect();
            sorted_files.sort_by_key(|(_, idx)| *idx);
            for (fname, _) in &sorted_files {
                files.push(SafeTensorsFile::open(&dir.join(fname))?);
            }

            Ok(Self { files, weight_map })
        } else if single_path.exists() {
            let f = SafeTensorsFile::open(&single_path)?;
            let weight_map = f.tensor_names().map(|n| (n.to_string(), 0)).collect();
            Ok(Self {
                files: vec![f],
                weight_map,
            })
        } else {
            Err("No safetensors found in directory".into())
        }
    }

    fn tensor_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.weight_map.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    fn load_f32(&self, name: &str) -> Option<Vec<f32>> {
        let &idx = self.weight_map.get(name)?;
        self.files.get(idx)?.load_f32(name)
    }

    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        let &idx = self.weight_map.get(name)?;
        self.files.get(idx)?.shape(name).map(|s| s.to_vec())
    }
}

// ── Unified input ────────────────────────────────────────────────────────

enum ModelInput {
    Gguf(Box<rocmforge::loader::GgufFile>),
    Safetensors(ShardedSafeTensors),
}

struct TensorMeta {
    name: String,
    rows: usize,
    cols: usize,
    orig_bytes: usize,
}

fn open_model(path: &str) -> Result<ModelInput, Box<dyn std::error::Error>> {
    let p = Path::new(path);
    if p.is_dir() {
        let st = ShardedSafeTensors::open(p)?;
        println!(
            "Opened safetensors directory: {} tensors",
            st.tensor_names().len()
        );
        Ok(ModelInput::Safetensors(st))
    } else {
        let gguf = rocmforge::loader::GgufFile::open(path)?;
        println!("Opened GGUF: {} tensors", gguf.tensor_count());
        Ok(ModelInput::Gguf(Box::new(gguf)))
    }
}

fn collect_tensors(model: &ModelInput, max_layers: Option<usize>) -> Vec<TensorMeta> {
    match model {
        ModelInput::Gguf(gguf) => collect_gguf_tensors(gguf, max_layers),
        ModelInput::Safetensors(st) => collect_safetensors_tensors(st, max_layers),
    }
}

fn collect_gguf_tensors(
    gguf: &rocmforge::loader::GgufFile,
    max_layers: Option<usize>,
) -> Vec<TensorMeta> {
    use rocmforge::loader::GgmlType;

    let mut names: Vec<String> = gguf.tensor_names().map(str::to_string).collect();
    names.sort();

    let layer_set = if let Some(max) = max_layers {
        let mut set = std::collections::BTreeSet::new();
        let mut count = 0usize;
        for name in &names {
            if !is_weight_tensor_gguf(name) {
                continue;
            }
            if let Some(idx) = name.find("blk.") {
                let rest = &name[idx + 5..];
                let end = rest.find('.').unwrap_or(rest.len());
                if set.insert(rest[..end].to_string()) {
                    count += 1;
                    if count > max {
                        break;
                    }
                }
            }
        }
        Some(set)
    } else {
        None
    };

    names
        .iter()
        .filter(|n| is_weight_tensor_gguf(n))
        .filter_map(|name| {
            let tv = gguf.tensor(name).ok()??;
            if tv.dims.len() != 2 {
                return None;
            }
            let in_dim = tv.dims[0] as usize;
            let out_dim = tv.dims[1] as usize;
            if in_dim < 4 || out_dim < 4 {
                return None;
            }

            if let Some(ref ls) = layer_set {
                if let Some(idx) = name.find("blk.") {
                    let rest = &name[idx + 5..];
                    let end = rest.find('.').unwrap_or(rest.len());
                    if !ls.contains(&rest[..end].to_string()) {
                        return None;
                    }
                }
            }

            let type_size = match tv.ggml_type {
                GgmlType::F32 => 4.0,
                GgmlType::F16 => 2.0,
                GgmlType::Q4_0 => 18.0 / 32.0,
                GgmlType::Q4_K => 144.0 / 256.0,
                GgmlType::Q6_K => 210.0 / 256.0,
                GgmlType::Q8_0 => 34.0 / 32.0,
                _ => 4.0,
            };

            Some(TensorMeta {
                name: name.clone(),
                rows: out_dim,
                cols: in_dim,
                orig_bytes: ((out_dim * in_dim) as f64 * type_size) as usize,
            })
        })
        .collect()
}

fn collect_safetensors_tensors(
    st: &ShardedSafeTensors,
    max_layers: Option<usize>,
) -> Vec<TensorMeta> {
    let names = st.tensor_names();

    let layer_set = if let Some(max) = max_layers {
        let mut set = std::collections::BTreeSet::new();
        let mut count = 0usize;
        for name in &names {
            if !is_weight_tensor_st(name) {
                continue;
            }
            if let Some(idx) = name.find("layers.") {
                let rest = &name[idx + 7..];
                let end = rest.find('.').unwrap_or(rest.len());
                if let Ok(layer_num) = rest[..end].parse::<usize>() {
                    if set.insert(layer_num) {
                        count += 1;
                        if count > max {
                            break;
                        }
                    }
                }
            }
        }
        Some(set)
    } else {
        None
    };

    names
        .iter()
        .filter(|n| is_weight_tensor_st(n))
        .filter_map(|name| {
            let shape = st.shape(name)?;
            if shape.len() != 2 {
                return None;
            }
            let rows = shape[0];
            let cols = shape[1];
            if rows < 4 || cols < 4 {
                return None;
            }

            if let Some(ref ls) = layer_set {
                if let Some(idx) = name.find("layers.") {
                    let rest = &name[idx + 7..];
                    let end = rest.find('.').unwrap_or(rest.len());
                    if let Ok(layer_num) = rest[..end].parse::<usize>() {
                        if !ls.contains(&layer_num) {
                            return None;
                        }
                    }
                }
            }

            Some(TensorMeta {
                name: name.to_string(),
                rows,
                cols,
                orig_bytes: ((rows * cols) as f64 * 2.0) as usize,
            })
        })
        .collect()
}

fn load_tensor_f32(model: &ModelInput, meta: &TensorMeta) -> Option<Vec<f32>> {
    match model {
        ModelInput::Gguf(gguf) => {
            let tv = gguf.tensor(&meta.name).ok()??;
            dequantize_gguf(&tv)
        }
        ModelInput::Safetensors(st) => st.load_f32(&meta.name),
    }
}

fn dequantize_gguf(tv: &rocmforge::loader::TensorView) -> Option<Vec<f32>> {
    use rocmforge::loader::GgmlType;
    let n = tv.element_count();
    match tv.ggml_type {
        GgmlType::F32 => {
            let out = (0..n)
                .map(|i| {
                    f32::from_le_bytes([
                        tv.data[i * 4],
                        tv.data[i * 4 + 1],
                        tv.data[i * 4 + 2],
                        tv.data[i * 4 + 3],
                    ])
                })
                .collect();
            Some(out)
        }
        GgmlType::Q4_0 => Some(deq_q4_0(tv.data, n)),
        GgmlType::Q8_0 => Some(deq_q8_0(tv.data, n)),
        GgmlType::Q4_K => {
            let mut out = vec![0.0f32; n];
            rocmforge::cpu::quant::embed_q4_k(0, tv.data, &mut out, n);
            Some(out)
        }
        GgmlType::Q6_K => {
            let mut out = vec![0.0f32; n];
            rocmforge::cpu::quant::embed_q6_k(0, tv.data, &mut out, n);
            Some(out)
        }
        _ => None,
    }
}

fn deq_q4_0(data: &[u8], n: usize) -> Vec<f32> {
    let nb = n / 32;
    let mut out = vec![0.0f32; n];
    for i in 0..nb {
        let off = i * 18;
        let s = half::f16::from_bits(u16::from_le_bytes([data[off], data[off + 1]])).to_f32();
        for j in 0..32 {
            let b = data[off + 2 + j / 2];
            let nib = if j % 2 == 0 {
                b & 0x0F
            } else {
                (b >> 4) & 0x0F
            };
            out[i * 32 + j] = ((nib as i8) - 8) as f32 * s;
        }
    }
    out
}

fn deq_q8_0(data: &[u8], n: usize) -> Vec<f32> {
    let nb = n / 32;
    let mut out = vec![0.0f32; n];
    for i in 0..nb {
        let off = i * 34;
        let s = half::f16::from_bits(u16::from_le_bytes([data[off], data[off + 1]])).to_f32();
        for j in 0..32 {
            out[i * 32 + j] = data[off + 2 + j] as i8 as f32 * s;
        }
    }
    out
}

fn is_weight_tensor_gguf(name: &str) -> bool {
    name.ends_with(".weight")
        && (name.contains("attn_q")
            || name.contains("attn_k")
            || name.contains("attn_v")
            || name.contains("attn_output")
            || name.contains("attn_qkv")
            || name.contains("ffn_gate")
            || name.contains("ffn_up")
            || name.contains("ffn_down"))
}

fn is_weight_tensor_st(name: &str) -> bool {
    name.ends_with(".weight")
        && (name.contains("q_proj")
            || name.contains("k_proj")
            || name.contains("v_proj")
            || name.contains("o_proj")
            || name.contains("qkv_proj")
            || name.contains("gate_proj")
            || name.contains("up_proj")
            || name.contains("down_proj"))
}

// ── SVD (power iteration from convert.rs) ────────────────────────────────

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
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    norm
}

fn orthogonalize(v: &mut [f32], basis: &[Vec<f32>]) {
    for b in basis {
        let dot: f32 = v.iter().zip(b).map(|(x, y)| x * y).sum();
        for (x, y) in v.iter_mut().zip(b) {
            *x -= dot * y;
        }
    }
}

fn matvec_w(a: &[f32], m: usize, n: usize, v: &[f32]) -> Vec<f32> {
    a.par_chunks(n)
        .take(m)
        .map(|row| row.iter().zip(v).map(|(x, y)| x * y).sum())
        .collect()
}

fn matvec_wt(a: &[f32], m: usize, n: usize, u: &[f32]) -> Vec<f32> {
    (0..n)
        .into_par_iter()
        .map(|col| (0..m).map(|row| a[row * n + col] * u[row]).sum())
        .collect()
}

fn seed_vector(len: usize, component: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ ((component as u64 + 1) * 0xbf58_476d_1ce4_e5b9);
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        v.push(((bits >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0);
    }
    normalize(&mut v);
    v
}

fn top_k_svd(a: &[f32], m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let k = k.min(m).min(n);
    let iters = 8;
    let mut u_basis: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut v_basis: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut sigmas = Vec::with_capacity(k);

    for component in 0..k {
        let mut v = seed_vector(n, component);
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

// ── CLI parsing ──────────────────────────────────────────────────────────

fn parse_k_args(args: &[String]) -> Vec<usize> {
    for i in 0..args.len() {
        if args[i] == "--k" && i + 1 < args.len() {
            return args[i + 1]
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
        }
    }
    vec![1, 2, 4, 8, 16]
}

fn parse_max_layers(args: &[String]) -> Option<usize> {
    for i in 0..args.len() {
        if args[i] == "--max-layers" && i + 1 < args.len() {
            return args[i + 1].trim().parse().ok();
        }
    }
    None
}

fn parse_error_target(args: &[String]) -> f32 {
    for i in 0..args.len() {
        if args[i] == "--error-target" && i + 1 < args.len() {
            return args[i + 1].trim().parse().unwrap_or(0.05);
        }
    }
    0.05
}

// ── Q4 quantization helpers ──────────────────────────────────────────────

fn q4_quant_block(block: &[f32]) -> (f32, Vec<i8>) {
    let mut max_abs = 0.0f32;
    for &x in block {
        if x.abs() > max_abs {
            max_abs = x.abs();
        }
    }
    let scale = max_abs / 8.0;
    let quant: Vec<i8> = block
        .iter()
        .map(|&x| {
            if scale > 1e-10 {
                (x / scale).round().clamp(-8.0, 7.0) as i8
            } else {
                0
            }
        })
        .collect();
    (scale, quant)
}

fn dequantize_q4_8bit(scale: f32, quant: &[i8]) -> Vec<f32> {
    quant.iter().map(|&q| q as f32 * scale).collect()
}

fn quantize_dequantize_matrix(w: &[f32], rows: usize, cols: usize, block_size: usize) -> Vec<f32> {
    let n = rows * cols;
    let mut out = vec![0.0f32; n];
    for start in (0..n).step_by(block_size) {
        let end = (start + block_size).min(n);
        let block = &w[start..end];
        let (scale, quant) = q4_quant_block(block);
        let recon = dequantize_q4_8bit(scale, &quant);
        out[start..end].copy_from_slice(&recon);
    }
    out
}

fn frob_rel_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    let diff_sq: f32 = original
        .iter()
        .zip(reconstructed)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let norm_sq: f32 = original.iter().map(|x| x * x).sum();
    if norm_sq < 1e-10 {
        0.0
    } else {
        (diff_sq / norm_sq).sqrt()
    }
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: svd_analyze <model.gguf | safetensors_dir/> [--k 1,2,4,8,16] [--max-layers N] [--error-target 0.05]");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let k_values = parse_k_args(&args);
    let max_layers = parse_max_layers(&args);
    let error_target = parse_error_target(&args);

    println!("=================================================================");
    println!("    SVD-QUANT COMPRESSION ANALYZER");
    println!("=================================================================");
    println!("Input: {}", model_path);
    println!("SVD ranks: {:?}", k_values);
    println!("Error target: {:.1}%", error_target * 100.0);
    println!();

    let model = open_model(model_path)?;
    let tensors = collect_tensors(&model, max_layers);
    println!("Analyzing {} weight tensors...\n", tensors.len());

    println!(
        "{:<55} {:>5} {:>5} {:>4} {:>8} {:>8} {:>8} {:>6}",
        "Tensor", "Rows", "Cols", "k", "NaiveQ4", "SVD+Q4", "Improv", "ms"
    );
    println!("{}", "-".repeat(104));

    let mut optimal_k: Vec<(String, usize, f32, f32, usize, usize)> = Vec::new();

    for meta in &tensors {
        let w_f32 = match load_tensor_f32(&model, meta) {
            Some(w) => w,
            None => continue,
        };

        let naive_q4 = quantize_dequantize_matrix(&w_f32, meta.rows, meta.cols, 32);
        let naive_err = frob_rel_error(&w_f32, &naive_q4);

        let mut best_k = 0usize;
        let mut best_err = 1.0f32;
        let mut best_naive = 0.0f32;
        let mut _best_ratio = 1.0f64;

        for &k in &k_values {
            let k = k.min(meta.rows).min(meta.cols);
            let t0 = Instant::now();

            let (u_sigma, vt) = top_k_svd(&w_f32, meta.rows, meta.cols, k);
            let low_rank = matmul(&u_sigma, &vt, meta.rows, k, meta.cols);

            let mut residual = vec![0.0f32; meta.rows * meta.cols];
            for i in 0..meta.rows * meta.cols {
                residual[i] = w_f32[i] - low_rank[i];
            }

            let q4_residual = quantize_dequantize_matrix(&residual, meta.rows, meta.cols, 32);

            let mut reconstructed = vec![0.0f32; meta.rows * meta.cols];
            for i in 0..meta.rows * meta.cols {
                reconstructed[i] = q4_residual[i] + low_rank[i];
            }

            let svd_q4_err = frob_rel_error(&w_f32, &reconstructed);
            let improvement = if naive_err > 1e-6 {
                naive_err / svd_q4_err
            } else {
                1.0
            };

            let svd_mb = (meta.rows * k + k * meta.cols) as f64 * 2.0 / 1e6;
            let q4_mb = (meta.rows * meta.cols) as f64 * 0.5 / 1e6;
            let total_mb = svd_mb + q4_mb;
            let orig_mb = meta.orig_bytes as f64 / 1e6;
            let ratio = total_mb / orig_mb;

            let elapsed = t0.elapsed().as_millis();

            println!(
                "{:<55} {:>5} {:>5} {:>4} {:>7.2}% {:>7.2}% {:>5.1}x {:>5}",
                meta.name,
                meta.rows,
                meta.cols,
                k,
                naive_err * 100.0,
                svd_q4_err * 100.0,
                improvement,
                elapsed
            );

            if svd_q4_err < error_target && (best_k == 0 || k < best_k) {
                best_k = k;
                best_err = svd_q4_err;
                best_naive = naive_err;
                _best_ratio = ratio;
            }
        }

        if best_k > 0 {
            optimal_k.push((
                meta.name.clone(),
                best_k,
                best_err,
                best_naive,
                meta.rows,
                meta.cols,
            ));
        }
    }

    println!("{}", "-".repeat(104));
    println!(
        "\n=== OPTIMAL k (SVD+Q4 error <{:.1}%) ===",
        error_target * 100.0
    );
    println!(
        "{:<55} {:>4} {:>8} {:>8} {:>8} {:>8}",
        "Tensor", "k", "NaiveQ4", "SVD+Q4", "Improv", "Ratio"
    );
    println!("{}", "-".repeat(96));

    for (name, k, err, naive, rows, cols) in &optimal_k {
        let improv = if *err > 1e-6 { *naive / *err } else { 1.0 };
        let n_elems = *rows * *cols;
        let orig_bytes = n_elems as f64 * 2.0; // BF16
        let q4_bytes = n_elems as f64 * 0.5;
        let svd_bytes = (*rows * *k + *k * *cols) as f64 * 2.0; // FP16 for U/V
        let ratio = (q4_bytes + svd_bytes) / orig_bytes;
        println!(
            "{:<55} {:>4} {:>7.2}% {:>7.2}% {:>6.1}x {:>7.1}%",
            name,
            k,
            naive * 100.0,
            err * 100.0,
            improv,
            ratio * 100.0
        );
    }

    Ok(())
}
