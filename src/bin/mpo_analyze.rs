//! MPO Weight Compression Analyzer for LLM models.
//!
//! Loads real GGUF model weights, applies Matrix Product Operator compression
//! at various bond dimensions, and reports per-layer compression ratios and
//! reconstruction fidelity. This provides concrete data on whether MPO
//! compression is viable for reducing VRAM/CPU memory in production inference.
//!
//! Run with:
//!   cargo run --bin mpo_analyze -- <model.gguf> [--chi 2,4,8] [--max-layers 4]

use rocmforge::loader::{GgmlType, GgufFile, TensorView};
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: mpo_analyze <model.gguf> [--chi 2,4,8] [--max-layers N]");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let chi_values = parse_chi_args(&args);
    let max_layers = parse_max_layers(&args);

    println!("=================================================================");
    println!("    MPO WEIGHT COMPRESSION ANALYZER FOR LLM MODELS");
    println!("=================================================================");
    println!("Model: {}", model_path);
    println!("Bond dimensions to test: {:?}", chi_values);
    println!();

    let gguf = GgufFile::open(model_path)?;
    println!(
        "Loaded GGUF: {} tensors, {:.1} MB file",
        gguf.tensor_count(),
        std::fs::metadata(model_path)?.len() as f64 / 1e6
    );

    let mut tensor_names: Vec<String> = gguf.tensor_names().map(str::to_string).collect();
    tensor_names.sort();

    let weight_tensors: Vec<_> = tensor_names
        .iter()
        .filter(|n| is_weight_tensor(n))
        .filter_map(|name| {
            let tv = gguf.tensor(name).ok()??;
            if tv.dims.len() != 2 {
                return None;
            }
            Some((name.clone(), tv))
        })
        .collect();

    let selected: Vec<_> = if let Some(max) = max_layers {
        let mut layer_set = std::collections::BTreeSet::new();
        let mut count = 0usize;
        for name in &tensor_names {
            if !is_weight_tensor(name) {
                continue;
            }
            if let Some(idx) = name.find(".blk.") {
                let rest = &name[idx + 5..];
                let end = rest.find('.').unwrap_or(rest.len());
                let layer_key = &name[idx + 5..idx + 5 + end];
                if layer_set.insert(layer_key.to_string()) {
                    count += 1;
                    if count > max {
                        break;
                    }
                }
            }
        }
        weight_tensors
            .into_iter()
            .filter(|(n, _)| {
                if let Some(idx) = n.find(".blk.") {
                    let rest = &n[idx + 5..];
                    let end = rest.find('.').unwrap_or(rest.len());
                    layer_set.contains(&n[idx + 5..idx + 5 + end])
                } else {
                    true
                }
            })
            .collect()
    } else {
        weight_tensors
    };

    println!("\nAnalyzing {} weight tensors...\n", selected.len());

    println!(
        "{:<50} {:>10} {:>10} {:>6} {:>8} {:>8} {:>10} {:>10}",
        "Tensor", "Rows", "Cols", "Chi", "Ratio", "Error%", "Params", "Time_ms"
    );
    println!("{}", "-".repeat(114));

    let mut total_original: f64 = 0.0;
    let mut total_compressed: Vec<f64> = vec![0.0; chi_values.len()];
    let mut total_error: Vec<f64> = vec![0.0; chi_values.len()];

    for (name, tv) in &selected {
        let rows = tv.dims[1].max(tv.dims[0]) as usize;
        let cols = tv.dims[0].min(tv.dims[1]) as usize;
        let n_elements = tv.element_count();

        if rows < 4 || cols < 4 {
            continue;
        }

        let w_f32 = match dequantize_tensor(tv) {
            Some(w) => w,
            None => {
                println!(
                    "{:<50} {:>10} {:>10} -- skipped (unsupported type {:?})",
                    name, rows, cols, tv.ggml_type
                );
                continue;
            }
        };

        total_original += n_elements as f64;

        for (ci, &chi) in chi_values.iter().enumerate() {
            let t0 = Instant::now();

            let factors_out = factor_dimension(rows);
            let factors_in = factor_dimension(cols);
            let n_sites = factors_out.len().max(factors_in.len());

            let padded_out: usize = {
                let mut f = factors_out.clone();
                while f.len() < n_sites {
                    f.insert(0, 1);
                }
                f.iter().product()
            };
            let padded_in: usize = {
                let mut f = factors_in.clone();
                while f.len() < n_sites {
                    f.insert(0, 1);
                }
                f.iter().product()
            };

            let padded_w = if padded_out == rows && padded_in == cols {
                w_f32.clone()
            } else {
                let mut padded = vec![0.0f32; padded_out * padded_in];
                for r in 0..rows {
                    let src = &w_f32[r * cols..(r + 1) * cols];
                    let dst = &mut padded[r * padded_in..(r + 1) * padded_in];
                    dst[..cols].copy_from_slice(src);
                }
                padded
            };

            let mpo = compress_mpo(&padded_w, padded_out, padded_in, chi);
            let elapsed = t0.elapsed().as_millis();

            let mpo_params: usize = mpo
                .sites
                .iter()
                .map(|s| s.chi_left * s.d_out * s.d_in * s.chi_right)
                .sum();
            let ratio = mpo_params as f64 / n_elements as f64;

            let error = compute_reconstruction_error(&mpo, &w_f32, rows, cols, cols);

            total_compressed[ci] += mpo_params as f64;
            total_error[ci] += error as f64 * n_elements as f64;

            println!(
                "{:<50} {:>10} {:>10} {:>6} {:>7.1}% {:>7.2}% {:>10} {:>10}",
                name,
                rows,
                cols,
                chi,
                ratio * 100.0,
                error * 100.0,
                mpo_params,
                elapsed
            );
        }
    }

    println!("{}", "-".repeat(114));
    println!("\n=== SUMMARY ===");
    println!("Original parameters: {:.0}", total_original);
    for (ci, &chi) in chi_values.iter().enumerate() {
        let ratio = total_compressed[ci] / total_original;
        println!(
            "chi={}: compressed params = {:.0} ({:.1}% of original), avg error = {:.2}%",
            chi,
            total_compressed[ci],
            ratio * 100.0,
            if total_original > 0.0 {
                total_error[ci] / total_original * 100.0
            } else {
                0.0
            }
        );
    }
    println!(
        "\nFor reference: Q4_0 uses 4.5 bits/weight = 56.25% of F32 size"
    );
    println!(
        "For reference: Q8_0 uses 8.5 bits/weight = 26.6% of F32 size"
    );

    Ok(())
}

fn parse_chi_args(args: &[String]) -> Vec<usize> {
    for i in 0..args.len() {
        if args[i] == "--chi" && i + 1 < args.len() {
            return args[i + 1]
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
        }
    }
    vec![2, 4, 8]
}

fn parse_max_layers(args: &[String]) -> Option<usize> {
    for i in 0..args.len() {
        if args[i] == "--max-layers" && i + 1 < args.len() {
            return args[i + 1].trim().parse().ok();
        }
    }
    None
}

fn is_weight_tensor(name: &str) -> bool {
    name.ends_with(".weight")
}

fn dequantize_tensor(tv: &TensorView) -> Option<Vec<f32>> {
    match tv.ggml_type {
        GgmlType::F32 => {
            let n = tv.element_count();
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                out[i] = f32::from_le_bytes([
                    tv.data[i * 4],
                    tv.data[i * 4 + 1],
                    tv.data[i * 4 + 2],
                    tv.data[i * 4 + 3],
                ]);
            }
            Some(out)
        }
        GgmlType::Q4_0 => Some(dequantize_q4_0_to_f32(tv.data, tv.element_count())),
        GgmlType::Q8_0 => Some(dequantize_q8_0_to_f32(tv.data, tv.element_count())),
        _ => None,
    }
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

fn dequantize_q8_0_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    let mut out = vec![0.0f32; num_elements];
    for i in 0..num_blocks {
        let block_offset = i * 34;
        let scale = half::f16::from_bits(u16::from_le_bytes([
            data[block_offset],
            data[block_offset + 1],
        ]))
        .to_f32();
        for j in 0..32 {
            let qval = data[block_offset + 2 + j] as i8;
            out[i * 32 + j] = qval as f32 * scale;
        }
    }
    out
}

fn factor_dimension(d: usize) -> Vec<usize> {
    if d < 2 {
        return vec![d];
    }
    let mut remaining = d;
    let mut factors = Vec::new();
    while remaining.is_multiple_of(4) {
        factors.push(4);
        remaining /= 4;
    }
    while remaining.is_multiple_of(2) {
        factors.push(2);
        remaining /= 2;
    }
    let mut odd = 3;
    while odd * odd <= remaining {
        while remaining.is_multiple_of(odd) {
            factors.push(odd);
            remaining /= odd;
        }
        odd += 2;
    }
    if remaining > 1 {
        factors.push(remaining);
    }
    factors
}

fn compress_mpo(weights: &[f32], n_out: usize, n_in: usize, chi_max: usize) -> MpoResult {
    let mut out_factors = factor_dimension(n_out);
    let mut in_factors = factor_dimension(n_in);
    let n_sites = out_factors.len().max(in_factors.len());
    while out_factors.len() < n_sites {
        out_factors.insert(0, 1);
    }
    while in_factors.len() < n_sites {
        in_factors.insert(0, 1);
    }

    let phys_dims: Vec<usize> = (0..n_sites)
        .map(|k| out_factors[k] * in_factors[k])
        .collect();

    let pair_strides: Vec<usize> = {
        let mut s = vec![0usize; n_sites];
        let mut stride = 1;
        for k in (0..n_sites).rev() {
            s[k] = stride;
            stride *= phys_dims[k];
        }
        s
    };

    let do_strides: Vec<usize> = {
        let mut s = vec![0usize; n_sites];
        let mut stride = 1;
        for k in (0..n_sites).rev() {
            s[k] = stride;
            stride *= out_factors[k];
        }
        s
    };

    let di_strides: Vec<usize> = {
        let mut s = vec![0usize; n_sites];
        let mut stride = 1;
        for k in (0..n_sites).rev() {
            s[k] = stride;
            stride *= in_factors[k];
        }
        s
    };

    let mut w_t = vec![0.0f32; n_out * n_in];
    for row in 0..n_out {
        for col in 0..n_in {
            let mut interleaved = 0usize;
            for k in 0..n_sites {
                let io_k = if do_strides[k] > 0 {
                    (row / do_strides[k]) % out_factors[k]
                } else {
                    0
                };
                let ii_k = if di_strides[k] > 0 {
                    (col / di_strides[k]) % in_factors[k]
                } else {
                    0
                };
                interleaved += (io_k * in_factors[k] + ii_k) * pair_strides[k];
            }
            w_t[interleaved] = weights[row * n_in + col];
        }
    }

    let mut current = w_t;
    let mut chi_left = 1usize;
    let mut sites = Vec::with_capacity(n_sites);

    for k in 0..n_sites {
        let phys_k = phys_dims[k];
        let d_out_k = out_factors[k];
        let d_in_k = in_factors[k];
        let total_right: usize = phys_dims[k..].iter().product();
        let n_svd = total_right / phys_k;
        let m_svd = chi_left * phys_k;

        let mut unfolded = vec![0.0f32; m_svd * n_svd];
        for il in 0..chi_left {
            for ip in 0..phys_k {
                let row = il * phys_k + ip;
                for col in 0..n_svd {
                    unfolded[row * n_svd + col] =
                        current[il * total_right + ip * n_svd + col];
                }
            }
        }

        let (u, sigma, vt, k_svd) = if m_svd >= n_svd {
            let (u, s, vt) = svd_thin(&unfolded, m_svd, n_svd);
            let k = s.len();
            (u, s, vt, k)
        } else {
            let ut = transpose(&unfolded, m_svd, n_svd);
            let (v, s, ut2) = svd_thin(&ut, n_svd, m_svd);
            let u2 = transpose(&ut2, m_svd, m_svd);
            let vt2 = transpose(&v, n_svd, m_svd);
            let k = s.len();
            (u2, s, vt2, k)
        };

        const SVD_EPS: f32 = 1e-9;
        let chi_new = sigma
            .iter()
            .filter(|&&s| s > SVD_EPS)
            .count()
            .min(chi_max)
            .max(1);
        let chi_right = if k == n_sites - 1 { 1 } else { chi_new };

        let mut data = vec![0.0f32; chi_left * d_out_k * d_in_k * chi_right];
        for il in 0..chi_left {
            for io in 0..d_out_k {
                for ii in 0..d_in_k {
                    let u_row = il * phys_k + io * d_in_k + ii;
                    for ir in 0..chi_right {
                        let u_val = if ir < k_svd { u[u_row * k_svd + ir] } else { 0.0 };
                        let s_val = if ir < k_svd { sigma[ir] } else { 0.0 };
                        data[il * d_out_k * d_in_k * chi_right
                            + io * d_in_k * chi_right
                            + ii * chi_right
                            + ir] = u_val * s_val;
                    }
                }
            }
        }
        sites.push(MpoSiteResult {
            chi_left,
            d_out: d_out_k,
            d_in: d_in_k,
            chi_right,
            data,
        });

        if k < n_sites - 1 {
            current = (0..chi_right * n_svd)
                .map(|idx| {
                    let ir = idx / n_svd;
                    let col = idx % n_svd;
                    if ir < k_svd {
                        vt[ir * n_svd + col]
                    } else {
                        0.0
                    }
                })
                .collect();
        }
        chi_left = chi_right;
    }

    MpoResult { sites }
}

fn compute_reconstruction_error(
    mpo: &MpoResult,
    original: &[f32],
    n_out: usize,
    n_in: usize,
    stride: usize,
) -> f32 {
    let n_samples = 8usize.min(n_in);
    let mut diff_sq = 0.0f32;
    let mut norm_sq = 0.0f32;

    let mut rng_state: u64 = 0x1234567890ABCDEF;
    for _ in 0..n_samples {
        let x: Vec<f32> = (0..n_in)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let bits = (rng_state >> 33) as i32;
                bits as f32 / (1i32 << 30) as f32
            })
            .collect();

        let y_mpo = mpo_apply(mpo, &x);
        let y_dense: Vec<f32> = (0..n_out)
            .map(|i| (0..n_in).map(|j| original[i * stride + j] * x[j]).sum())
            .collect();

        for i in 0..n_out {
            let d = y_mpo[i] - y_dense[i];
            diff_sq += d * d;
            norm_sq += y_dense[i] * y_dense[i];
        }
    }

    if norm_sq < 1e-10 {
        0.0
    } else {
        (diff_sq / norm_sq).sqrt()
    }
}

fn mpo_apply(mpo: &MpoResult, x: &[f32]) -> Vec<f32> {
    let n_sites = mpo.sites.len();
    if n_sites == 0 {
        return Vec::new();
    }

    let d_out_dims: Vec<usize> = mpo.sites.iter().map(|s| s.d_out).collect();
    let d_in_dims: Vec<usize> = mpo.sites.iter().map(|s| s.d_in).collect();
    let n_out: usize = d_out_dims.iter().product();
    let n_in: usize = d_in_dims.iter().product();

    let out_strides: Vec<usize> = {
        let mut s = vec![1usize; n_sites];
        for k in (0..n_sites - 1).rev() {
            s[k] = s[k + 1] * d_out_dims[k + 1];
        }
        s
    };
    let in_strides: Vec<usize> = {
        let mut s = vec![1usize; n_sites];
        for k in (0..n_sites - 1).rev() {
            s[k] = s[k + 1] * d_in_dims[k + 1];
        }
        s
    };

    let mut y = vec![0.0f32; n_out];

    for (out_idx, y_out) in y.iter_mut().enumerate() {
        let io: Vec<usize> = (0..n_sites)
            .map(|k| (out_idx / out_strides[k]) % d_out_dims[k])
            .collect();
        for (in_idx, &x_in) in x.iter().enumerate().take(n_in) {
            let ii: Vec<usize> = (0..n_sites)
                .map(|k| (in_idx / in_strides[k]) % d_in_dims[k])
                .collect();
            let mut bond = vec![1.0f32];
            for k in 0..n_sites {
                let site = &mpo.sites[k];
                let chi_r = site.chi_right;
                let mut next = vec![0.0f32; chi_r];
                for (il, &bond_val) in bond.iter().enumerate() {
                    for ir in 0..chi_r {
                        next[ir] += bond_val
                            * site.data
                                [il * site.d_out * site.d_in * chi_r
                                    + io[k] * site.d_in * chi_r
                                    + ii[k] * chi_r
                                    + ir];
                    }
                }
                bond = next;
            }
            *y_out += bond[0] * x_in;
        }
    }

    y
}

struct MpoSiteResult {
    chi_left: usize,
    d_out: usize,
    d_in: usize,
    chi_right: usize,
    data: Vec<f32>,
}

struct MpoResult {
    sites: Vec<MpoSiteResult>,
}

fn transpose(a: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; n * m];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

fn matmul_small(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
    c
}

fn svd_thin(a: &[f32], m: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    if m < n {
        let at = transpose(a, m, n);
        let (v, sigma, ut) = svd_thin_tall(&at, n, m);
        let u = transpose(&ut, m, m);
        let vt = transpose(&v, n, m);
        return (u, sigma, vt);
    }
    svd_thin_tall(a, m, n)
}

fn svd_thin_tall(a: &[f32], m: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(m >= n);

    let at = transpose(a, m, n);
    let mut c = matmul_small(&at, a, n, m, n);

    let mut v = vec![0.0f32; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _ in 0..100 {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let cpq = c[p * n + q];
                if cpq.abs() > 1e-15 {
                    converged = false;
                    let cpp = c[p * n + p];
                    let cqq = c[q * n + q];
                    let theta = (cqq - cpp) / (2.0 * cpq);
                    let t = 1.0 / (theta.abs() + (1.0 + theta * theta).sqrt());
                    let t = if theta < 0.0 { -t } else { t };
                    let cos = 1.0 / (1.0 + t * t).sqrt();
                    let sin = t * cos;
                    let tau = sin / (1.0 + cos);

                    c[p * n + p] = cpp - t * cpq;
                    c[q * n + q] = cqq + t * cpq;
                    c[p * n + q] = 0.0;
                    c[q * n + p] = 0.0;

                    for r in 0..n {
                        if r != p && r != q {
                            let crp = c[r * n + p];
                            let crq = c[r * n + q];
                            c[r * n + p] = crp - sin * (crq + tau * crp);
                            c[r * n + q] = crq + sin * (crp - tau * crq);
                            c[p * n + r] = c[r * n + p];
                            c[q * n + r] = c[r * n + q];
                        }
                    }

                    for i in 0..n {
                        let vip = v[i * n + p];
                        let viq = v[i * n + q];
                        v[i * n + p] = vip - sin * (viq + tau * vip);
                        v[i * n + q] = viq + sin * (vip - tau * viq);
                    }
                }
            }
        }
        if converged {
            break;
        }
    }

    let mut sigma: Vec<f32> = (0..n).map(|i| c[i * n + i].max(0.0).sqrt()).collect();

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| sigma[b].partial_cmp(&sigma[a]).unwrap_or(std::cmp::Ordering::Equal));
    sigma = order.iter().map(|&i| sigma[i]).collect();
    let mut v_sorted = vec![0.0f32; n * n];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..n {
            v_sorted[row * n + new_col] = v[row * n + old_col];
        }
    }

    let av = matmul_small(a, &v_sorted, m, n, n);
    let mut u = vec![0.0f32; m * n];
    for col in 0..n {
        let inv_s = if sigma[col] > 1e-10 {
            1.0 / sigma[col]
        } else {
            0.0
        };
        for row in 0..m {
            u[row * n + col] = av[row * n + col] * inv_s;
        }
    }

    let vt = transpose(&v_sorted, n, n);
    (u, sigma, vt)
}
