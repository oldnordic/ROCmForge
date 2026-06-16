use fastrand::Rng;
use rocmforge::cpu::kernels::gemm_q4k_q8::{gemm_q4_k_q8_k_dispatch_gemm, gemv_q4_k_q8_k_dispatch};
use rocmforge::cpu::kernels::gemm_q4k_q8_scalar::gemv_q4_k_q8_k;
use rocmforge::cpu::ops::gemm::gemm_q6_k_fallback;
use rocmforge::cpu::quant::{embed_q4_k, embed_q6_k};
use rocmforge::loader::{GgmlType, GgufFile};

const MODEL: &str = "/home/feanor/Projects/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf";

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs_err(a: &[f32], b: &[f32]) -> f32 {
    let sum: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f32
}

#[test]
fn real_q4_k_scalar_gemv_matches_dequant_reference() {
    let gguf = GgufFile::open(MODEL).expect("open model");
    let tv = gguf
        .tensor("blk.0.attn_q.weight")
        .expect("lookup")
        .expect("exists");
    assert_eq!(tv.ggml_type, GgmlType::Q4_K);
    let dims: Vec<usize> = tv.dims.iter().map(|&d| d as usize).collect();
    let out_dim = dims[0];
    let in_dim = dims[1];

    let mut rng = Rng::new();
    let x: Vec<f32> = (0..in_dim).map(|_| rng.f32() * 4.0 - 2.0).collect();

    let mut y_kernel = vec![0.0f32; out_dim];
    gemv_q4_k_q8_k(tv.data, &x, &mut y_kernel, out_dim, in_dim);

    let mut y_ref = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let mut deq = vec![0.0f32; in_dim];
        embed_q4_k(row, tv.data, &mut deq, in_dim);
        y_ref[row] = deq.iter().zip(&x).map(|(w, xi)| w * xi).sum::<f32>();
    }

    let max_err = max_abs_err(&y_kernel, &y_ref);
    let mean_err = mean_abs_err(&y_kernel, &y_ref);
    println!(
        "Q4_K scalar GEMV blk.0.attn_q.weight max_err={:.6} mean_err={:.6}",
        max_err, mean_err
    );
    assert!(
        max_err < 1.0,
        "Q4_K scalar GEMV max abs error too large: {}",
        max_err
    );
    assert!(
        mean_err < 0.05,
        "Q4_K scalar GEMV mean abs error too large: {}",
        mean_err
    );
}

#[test]
fn real_q4_k_avx2_gemv_matches_scalar() {
    let gguf = GgufFile::open(MODEL).expect("open model");
    let tv = gguf
        .tensor("blk.0.attn_q.weight")
        .expect("lookup")
        .expect("exists");
    assert_eq!(tv.ggml_type, GgmlType::Q4_K);
    let dims: Vec<usize> = tv.dims.iter().map(|&d| d as usize).collect();
    let out_dim = dims[0];
    let in_dim = dims[1];

    let mut rng = Rng::new();
    let x: Vec<f32> = (0..in_dim).map(|_| rng.f32() * 4.0 - 2.0).collect();

    let mut y_scalar = vec![0.0f32; out_dim];
    gemv_q4_k_q8_k(tv.data, &x, &mut y_scalar, out_dim, in_dim);

    let mut y_avx2 = vec![0.0f32; out_dim];
    gemv_q4_k_q8_k_dispatch(tv.data, &x, &mut y_avx2, out_dim, in_dim);

    let max_err = max_abs_err(&y_avx2, &y_scalar);
    let mean_err = mean_abs_err(&y_avx2, &y_scalar);
    println!(
        "Q4_K AVX2 vs scalar GEMV blk.0.attn_q.weight max_err={:.6} mean_err={:.6}",
        max_err, mean_err
    );
    assert!(
        max_err < 1.0,
        "Q4_K AVX2 GEMV max abs error too large: {}",
        max_err
    );
    assert!(
        mean_err < 0.05,
        "Q4_K AVX2 GEMV mean abs error too large: {}",
        mean_err
    );
}

#[test]
fn real_q4_k_avx2_gemm_matches_scalar() {
    let gguf = GgufFile::open(MODEL).expect("open model");
    let tv = gguf
        .tensor("blk.0.attn_q.weight")
        .expect("lookup")
        .expect("exists");
    assert_eq!(tv.ggml_type, GgmlType::Q4_K);
    let dims: Vec<usize> = tv.dims.iter().map(|&d| d as usize).collect();
    let out_dim = dims[0];
    let in_dim = dims[1];

    let mut rng = Rng::new();
    let m = 3;
    let x: Vec<f32> = (0..m * in_dim).map(|_| rng.f32() * 4.0 - 2.0).collect();

    let mut y_scalar = vec![0.0f32; m * out_dim];
    gemm_q4_k_q8_k_dispatch_gemm(tv.data, &x, &mut y_scalar, m, out_dim, in_dim);

    let mut y_avx2 = vec![0.0f32; m * out_dim];
    gemm_q4_k_q8_k_dispatch_gemm(tv.data, &x, &mut y_avx2, m, out_dim, in_dim);

    let max_err = max_abs_err(&y_avx2, &y_scalar);
    let mean_err = mean_abs_err(&y_avx2, &y_scalar);
    println!(
        "Q4_K AVX2 vs scalar GEMM blk.0.attn_q.weight max_err={:.6} mean_err={:.6}",
        max_err, mean_err
    );
    assert!(
        max_err < 1.0,
        "Q4_K AVX2 GEMM max abs error too large: {}",
        max_err
    );
    assert!(
        mean_err < 0.05,
        "Q4_K AVX2 GEMM mean abs error too large: {}",
        mean_err
    );
}

#[test]
fn real_q6_k_gemm_matches_dequant_reference() {
    let gguf = GgufFile::open(MODEL).expect("open model");
    let tv = gguf
        .tensor("blk.0.attn_v.weight")
        .expect("lookup")
        .expect("exists");
    assert_eq!(tv.ggml_type, GgmlType::Q6_K);
    let dims: Vec<usize> = tv.dims.iter().map(|&d| d as usize).collect();
    let out_dim = dims[0];
    let in_dim = dims[1];

    let mut rng = Rng::new();
    let x: Vec<f32> = (0..in_dim).map(|_| rng.f32() * 4.0 - 2.0).collect();

    let mut y_kernel = vec![0.0f32; out_dim];
    gemm_q6_k_fallback(tv.data, &x, &mut y_kernel, out_dim, in_dim);

    let mut y_ref = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let mut deq = vec![0.0f32; in_dim];
        embed_q6_k(row, tv.data, &mut deq, in_dim);
        y_ref[row] = deq.iter().zip(&x).map(|(w, xi)| w * xi).sum::<f32>();
    }

    let max_err = max_abs_err(&y_kernel, &y_ref);
    let mean_err = mean_abs_err(&y_kernel, &y_ref);
    println!(
        "Q6_K blk.0.attn_v.weight max_err={:.6} mean_err={:.6}",
        max_err, mean_err
    );
    assert!(
        max_err < 1.0,
        "Q6_K GEMM max abs error too large: {}",
        max_err
    );
    assert!(
        mean_err < 0.05,
        "Q6_K GEMM mean abs error too large: {}",
        mean_err
    );
}
