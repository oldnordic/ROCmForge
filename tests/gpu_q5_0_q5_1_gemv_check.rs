//! Temporary diagnostic: compare GPU GEMV vs CPU dequant reference for real Q5_0/Q5_1 weights.

#![cfg(feature = "gpu")]

use fastrand::Rng;
use rocmforge::cpu::ops::gemm::gemm_q8_0;
use rocmforge::cpu::quant::{embed_q5_0, embed_q5_1, embed_q8_0};
use rocmforge::gpu::kernels::{embed_q8_0_token, gemv_q8_0_f32};
use rocmforge::gpu::weights::GpuBuffer;
use rocmforge::gpu::{self, GpuDevice};
use rocmforge::loader::{GgmlType, GgufFile};

const Q4K_MODEL: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const Q5K_MODEL: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q5_k_m.gguf";

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

fn check_q5_0_gemv(model_path: &str, tensor_name: &str) {
    let gguf = GgufFile::open(model_path).expect("open model");
    let tv = gguf.tensor(tensor_name).expect("lookup").expect("exists");
    assert_eq!(tv.ggml_type, GgmlType::Q5_0);
    let dims: Vec<usize> = tv.dims.iter().map(|&d| d as usize).collect();
    let in_dim = dims[0];
    let out_dim = dims[1];

    let mut rng = Rng::new();
    let x: Vec<f32> = (0..in_dim).map(|_| rng.f32() * 4.0 - 2.0).collect();

    // CPU reference
    let mut y_ref = vec![0.0f32; out_dim];
    for (row, y_ref_row) in y_ref.iter_mut().enumerate().take(out_dim) {
        let mut deq = vec![0.0f32; in_dim];
        embed_q5_0(row, tv.data, &mut deq, in_dim);
        *y_ref_row = deq.iter().zip(&x).map(|(w, xi)| w * xi).sum::<f32>();
    }

    // GPU
    let caps = gpu::detect().expect("detect gpu");
    let _device = GpuDevice::init(caps.device_id).expect("init gpu");
    let mut w_gpu =
        GpuBuffer::alloc_for_device(tv.data.len(), caps.device_id).expect("alloc weights");
    w_gpu.copy_from_host(tv.data).expect("upload weights");
    let mut x_gpu = GpuBuffer::alloc_for_device(in_dim * 4, caps.device_id).expect("alloc input");
    x_gpu
        .copy_from_host(unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, in_dim * 4) })
        .expect("upload input");
    let y_gpu = GpuBuffer::alloc_for_device(out_dim * 4, caps.device_id).expect("alloc output");

    rocmforge::gpu::kernels::gemv_q5_0_f32(
        w_gpu.as_ptr(),
        x_gpu.as_ptr() as *const f32,
        y_gpu.as_ptr() as *mut f32,
        in_dim,
        out_dim,
    )
    .expect("gemv q5_0");

    let mut y_host = vec![0.0f32; out_dim];
    y_gpu
        .copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(y_host.as_mut_ptr() as *mut u8, out_dim * 4)
        })
        .expect("download output");

    let max_err = max_abs_err(&y_host, &y_ref);
    let mean_err = mean_abs_err(&y_host, &y_ref);
    println!(
        "Q5_0 {} out={} in={} max_err={:.6} mean_err={:.6}",
        tensor_name, out_dim, in_dim, max_err, mean_err
    );
    assert!(max_err < 1.0, "Q5_0 GEMV max error too large: {}", max_err);
}

fn check_q5_1_gemv(model_path: &str, tensor_name: &str) {
    let gguf = GgufFile::open(model_path).expect("open model");
    let tv = gguf.tensor(tensor_name).expect("lookup").expect("exists");
    assert_eq!(tv.ggml_type, GgmlType::Q5_1);
    let dims: Vec<usize> = tv.dims.iter().map(|&d| d as usize).collect();
    let in_dim = dims[0];
    let out_dim = dims[1];

    let mut rng = Rng::new();
    let x: Vec<f32> = (0..in_dim).map(|_| rng.f32() * 4.0 - 2.0).collect();

    let mut y_ref = vec![0.0f32; out_dim];
    for (row, y_ref_row) in y_ref.iter_mut().enumerate().take(out_dim) {
        let mut deq = vec![0.0f32; in_dim];
        embed_q5_1(row, tv.data, &mut deq, in_dim);
        *y_ref_row = deq.iter().zip(&x).map(|(w, xi)| w * xi).sum::<f32>();
    }

    let caps = gpu::detect().expect("detect gpu");
    let _device = GpuDevice::init(caps.device_id).expect("init gpu");
    let mut w_gpu =
        GpuBuffer::alloc_for_device(tv.data.len(), caps.device_id).expect("alloc weights");
    w_gpu.copy_from_host(tv.data).expect("upload weights");
    let mut x_gpu = GpuBuffer::alloc_for_device(in_dim * 4, caps.device_id).expect("alloc input");
    x_gpu
        .copy_from_host(unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, in_dim * 4) })
        .expect("upload input");
    let y_gpu = GpuBuffer::alloc_for_device(out_dim * 4, caps.device_id).expect("alloc output");

    rocmforge::gpu::kernels::gemv_q5_1_f32(
        w_gpu.as_ptr(),
        x_gpu.as_ptr() as *const f32,
        y_gpu.as_ptr() as *mut f32,
        in_dim,
        out_dim,
    )
    .expect("gemv q5_1");

    let mut y_host = vec![0.0f32; out_dim];
    y_gpu
        .copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(y_host.as_mut_ptr() as *mut u8, out_dim * 4)
        })
        .expect("download output");

    let max_err = max_abs_err(&y_host, &y_ref);
    let mean_err = mean_abs_err(&y_host, &y_ref);
    println!(
        "Q5_1 {} out={} in={} max_err={:.6} mean_err={:.6}",
        tensor_name, out_dim, in_dim, max_err, mean_err
    );
    assert!(max_err < 1.0, "Q5_1 GEMV max error too large: {}", max_err);
}

fn check_q4_k_gemv(model_path: &str, tensor_name: &str) {
    let gguf = GgufFile::open(model_path).expect("open model");
    let tv = gguf.tensor(tensor_name).expect("lookup").expect("exists");
    assert_eq!(tv.ggml_type, GgmlType::Q4_K);
    let dims: Vec<usize> = tv.dims.iter().map(|&d| d as usize).collect();
    // GGUF convention: dims[0]=in_dim, dims[1]=out_dim for standard 2D matrices.
    let in_dim = dims[0];
    let out_dim = dims[1];

    let mut rng = Rng::new();
    let x: Vec<f32> = (0..in_dim).map(|_| rng.f32() * 4.0 - 2.0).collect();

    let mut y_ref = vec![0.0f32; out_dim];
    rocmforge::cpu::kernels::gemm_q4k_q8_scalar::gemv_q4_k_q8_k(
        tv.data, &x, &mut y_ref, out_dim, in_dim,
    );

    let caps = gpu::detect().expect("detect gpu");
    let _device = GpuDevice::init(caps.device_id).expect("init gpu");
    let mut w_gpu =
        GpuBuffer::alloc_for_device(tv.data.len(), caps.device_id).expect("alloc weights");
    w_gpu.copy_from_host(tv.data).expect("upload weights");
    let mut x_gpu = GpuBuffer::alloc_for_device(in_dim * 4, caps.device_id).expect("alloc input");
    x_gpu
        .copy_from_host(unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, in_dim * 4) })
        .expect("upload input");
    let y_gpu = GpuBuffer::alloc_for_device(out_dim * 4, caps.device_id).expect("alloc output");

    rocmforge::gpu::kernels::gemv_q4_k_f32(
        w_gpu.as_ptr(),
        x_gpu.as_ptr() as *const f32,
        y_gpu.as_ptr() as *mut f32,
        in_dim,
        out_dim,
    )
    .expect("gemv q4_k");

    let mut y_host = vec![0.0f32; out_dim];
    y_gpu
        .copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(y_host.as_mut_ptr() as *mut u8, out_dim * 4)
        })
        .expect("download output");

    let max_err = max_abs_err(&y_host, &y_ref);
    let mean_err = mean_abs_err(&y_host, &y_ref);
    println!(
        "Q4_K {} out={} in={} max_err={:.6} mean_err={:.6}",
        tensor_name, out_dim, in_dim, max_err, mean_err
    );
    println!("  first 10 GPU: {:?}", &y_host[..10.min(y_host.len())]);
    println!("  first 10 CPU: {:?}", &y_ref[..10.min(y_ref.len())]);
    assert!(max_err < 1.0, "Q4_K GEMV max error too large: {}", max_err);
}

fn check_q4_k_gemm(model_path: &str, tensor_name: &str) {
    let gguf = GgufFile::open(model_path).expect("open model");
    let tv = gguf.tensor(tensor_name).expect("lookup").expect("exists");
    assert_eq!(tv.ggml_type, GgmlType::Q4_K);
    let dims: Vec<usize> = tv.dims.iter().map(|&d| d as usize).collect();
    let in_dim = dims[0];
    let out_dim = dims[1];

    let mut rng = Rng::new();
    let x: Vec<f32> = (0..in_dim).map(|_| rng.f32() * 4.0 - 2.0).collect();

    // CPU reference
    let mut y_ref = vec![0.0f32; out_dim];
    rocmforge::cpu::kernels::gemm_q4k_q8_scalar::gemv_q4_k_q8_k(
        tv.data, &x, &mut y_ref, out_dim, in_dim,
    );

    let caps = gpu::detect().expect("detect gpu");
    let _device = GpuDevice::init(caps.device_id).expect("init gpu");
    let mut w_gpu =
        GpuBuffer::alloc_for_device(tv.data.len(), caps.device_id).expect("alloc weights");
    w_gpu.copy_from_host(tv.data).expect("upload weights");
    let mut x_gpu = GpuBuffer::alloc_for_device(in_dim * 4, caps.device_id).expect("alloc input");
    x_gpu
        .copy_from_host(unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, in_dim * 4) })
        .expect("upload input");
    let y_gpu = GpuBuffer::alloc_for_device(out_dim * 4, caps.device_id).expect("alloc output");

    rocmforge::gpu::kernels::gemm_q4_k_f32(
        w_gpu.as_ptr(),
        x_gpu.as_ptr() as *const f32,
        y_gpu.as_ptr() as *mut f32,
        in_dim,
        out_dim,
        1,
    )
    .expect("gemm q4_k");

    let mut y_host = vec![0.0f32; out_dim];
    y_gpu
        .copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(y_host.as_mut_ptr() as *mut u8, out_dim * 4)
        })
        .expect("download output");

    let max_err = max_abs_err(&y_host, &y_ref);
    let mean_err = mean_abs_err(&y_host, &y_ref);
    println!(
        "Q4_K GEMM {} out={} in={} max_err={:.6} mean_err={:.6}",
        tensor_name, out_dim, in_dim, max_err, mean_err
    );
    println!("  first 10 GPU: {:?}", &y_host[..10.min(y_host.len())]);
    println!("  first 10 CPU: {:?}", &y_ref[..10.min(y_ref.len())]);
    assert!(max_err < 1.0, "Q4_K GEMM max error too large: {}", max_err);
}

#[test]
fn test_q4_k_gemv_vs_cpu() {
    check_q4_k_gemv(Q4K_MODEL, "blk.10.ffn_down.weight");
    check_q4_k_gemv(Q4K_MODEL, "blk.12.ffn_down.weight");
    check_q4_k_gemv(Q4K_MODEL, "blk.13.ffn_down.weight");
}

#[test]
fn test_q4_k_gemm_vs_cpu() {
    check_q4_k_gemm(Q4K_MODEL, "blk.10.ffn_down.weight");
    check_q4_k_gemm(Q4K_MODEL, "blk.12.ffn_down.weight");
    check_q4_k_gemm(Q4K_MODEL, "blk.13.ffn_down.weight");
}

#[test]
fn test_q5_0_gemv_vs_cpu() {
    check_q5_0_gemv(Q4K_MODEL, "blk.0.attn_q.weight");
    check_q5_0_gemv(Q4K_MODEL, "blk.0.attn_output.weight");
    check_q5_0_gemv(Q4K_MODEL, "blk.0.ffn_gate.weight");
    check_q5_0_gemv(Q4K_MODEL, "blk.0.attn_k.weight");
    check_q5_0_gemv(Q4K_MODEL, "blk.3.attn_v.weight");
}

#[test]
fn test_q5_1_gemv_vs_cpu() {
    check_q5_1_gemv(Q5K_MODEL, "blk.0.attn_q.weight");
    check_q5_1_gemv(Q5K_MODEL, "blk.0.attn_output.weight");
    check_q5_1_gemv(Q5K_MODEL, "blk.0.ffn_gate.weight");
}

fn check_q5_k_gemv(model_path: &str, tensor_name: &str) {
    let gguf = GgufFile::open(model_path).expect("open model");
    let tv = gguf.tensor(tensor_name).expect("lookup").expect("exists");
    assert_eq!(tv.ggml_type, GgmlType::Q5_K);
    let dims: Vec<usize> = tv.dims.iter().map(|&d| d as usize).collect();
    let in_dim = dims[0];
    let out_dim = dims[1];

    let mut rng = Rng::new();
    let x: Vec<f32> = (0..in_dim).map(|_| rng.f32() * 4.0 - 2.0).collect();

    // CPU reference
    let mut y_ref = vec![0.0f32; out_dim];
    for (row, y_ref_row) in y_ref.iter_mut().enumerate().take(out_dim) {
        let mut deq = vec![0.0f32; in_dim];
        rocmforge::cpu::quant::embed_q5_k(row, tv.data, &mut deq, in_dim);
        *y_ref_row = deq.iter().zip(&x).map(|(w, xi)| w * xi).sum::<f32>();
    }

    let caps = gpu::detect().expect("detect gpu");
    let _device = GpuDevice::init(caps.device_id).expect("init gpu");
    let mut w_gpu =
        GpuBuffer::alloc_for_device(tv.data.len(), caps.device_id).expect("alloc weights");
    w_gpu.copy_from_host(tv.data).expect("upload weights");
    let mut x_gpu = GpuBuffer::alloc_for_device(in_dim * 4, caps.device_id).expect("alloc input");
    x_gpu
        .copy_from_host(unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, in_dim * 4) })
        .expect("upload input");
    let y_gpu = GpuBuffer::alloc_for_device(out_dim * 4, caps.device_id).expect("alloc output");

    rocmforge::gpu::kernels::gemv_q5_k_f32(
        w_gpu.as_ptr(),
        x_gpu.as_ptr() as *const f32,
        y_gpu.as_ptr() as *mut f32,
        in_dim,
        out_dim,
    )
    .expect("gemv q5_k");

    let mut y_host = vec![0.0f32; out_dim];
    y_gpu
        .copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(y_host.as_mut_ptr() as *mut u8, out_dim * 4)
        })
        .expect("download output");

    let max_err = max_abs_err(&y_host, &y_ref);
    let mean_err = mean_abs_err(&y_host, &y_ref);
    println!(
        "Q5_K {} out={} in={} max_err={:.6} mean_err={:.6}",
        tensor_name, out_dim, in_dim, max_err, mean_err
    );
    assert!(max_err < 1.0, "Q5_K GEMV max error too large: {}", max_err);
}

#[test]
fn test_q5_k_gemv_vs_cpu() {
    check_q5_k_gemv(Q5K_MODEL, "blk.11.ffn_down.weight");
    check_q5_k_gemv(Q5K_MODEL, "blk.12.ffn_down.weight");
}

fn check_q6_k_gemv(model_path: &str, tensor_name: &str) {
    let gguf = GgufFile::open(model_path).expect("open model");
    let tv = gguf.tensor(tensor_name).expect("lookup").expect("exists");
    assert_eq!(tv.ggml_type, GgmlType::Q6_K);
    let dims: Vec<usize> = tv.dims.iter().map(|&d| d as usize).collect();
    let in_dim = dims[0];
    let out_dim = dims[1];

    let mut rng = Rng::new();
    let x: Vec<f32> = (0..in_dim).map(|_| rng.f32() * 4.0 - 2.0).collect();

    // CPU reference
    let mut y_ref = vec![0.0f32; out_dim];
    for (row, y_ref_row) in y_ref.iter_mut().enumerate().take(out_dim) {
        let mut deq = vec![0.0f32; in_dim];
        rocmforge::cpu::quant::embed_q6_k(row, tv.data, &mut deq, in_dim);
        *y_ref_row = deq.iter().zip(&x).map(|(w, xi)| w * xi).sum::<f32>();
    }

    let caps = gpu::detect().expect("detect gpu");
    let _device = GpuDevice::init(caps.device_id).expect("init gpu");
    let mut w_gpu =
        GpuBuffer::alloc_for_device(tv.data.len(), caps.device_id).expect("alloc weights");
    w_gpu.copy_from_host(tv.data).expect("upload weights");
    let mut x_gpu = GpuBuffer::alloc_for_device(in_dim * 4, caps.device_id).expect("alloc input");
    x_gpu
        .copy_from_host(unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, in_dim * 4) })
        .expect("upload input");
    let y_gpu = GpuBuffer::alloc_for_device(out_dim * 4, caps.device_id).expect("alloc output");

    rocmforge::gpu::kernels::gemv_q6_k_f32(
        w_gpu.as_ptr(),
        x_gpu.as_ptr() as *const f32,
        y_gpu.as_ptr() as *mut f32,
        in_dim,
        out_dim,
    )
    .expect("gemv q6_k");

    let mut y_host = vec![0.0f32; out_dim];
    y_gpu
        .copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(y_host.as_mut_ptr() as *mut u8, out_dim * 4)
        })
        .expect("download output");

    let max_err = max_abs_err(&y_host, &y_ref);
    let mean_err = mean_abs_err(&y_host, &y_ref);
    println!(
        "Q6_K {} out={} in={} max_err={:.6} mean_err={:.6}",
        tensor_name, out_dim, in_dim, max_err, mean_err
    );
    println!("  first 10 GPU: {:?}", &y_host[..10.min(y_host.len())]);
    println!("  first 10 CPU: {:?}", &y_ref[..10.min(y_ref.len())]);
    assert!(max_err < 1.0, "Q6_K GEMV max error too large: {}", max_err);
}

#[test]
fn test_q6_k_gemv_vs_cpu() {
    check_q6_k_gemv(Q5K_MODEL, "blk.0.ffn_down.weight");
    check_q6_k_gemv(Q5K_MODEL, "blk.1.ffn_down.weight");
    check_q6_k_gemv(Q4K_MODEL, "blk.0.ffn_down.weight");
}

fn check_q8_0_gemv(model_path: &str, tensor_name: &str) {
    let gguf = GgufFile::open(model_path).expect("open model");
    let tv = gguf.tensor(tensor_name).expect("lookup").expect("exists");
    assert_eq!(tv.ggml_type, GgmlType::Q8_0);
    let dims: Vec<usize> = tv.dims.iter().map(|&d| d as usize).collect();
    let in_dim = dims[0];
    let out_dim = dims[1];

    let mut rng = Rng::new();
    let x: Vec<f32> = (0..in_dim).map(|_| rng.f32() * 4.0 - 2.0).collect();

    // CPU reference
    let mut y_ref = vec![0.0f32; out_dim];
    gemm_q8_0(tv.data, &x, &mut y_ref, out_dim, in_dim);

    let caps = gpu::detect().expect("detect gpu");
    let _device = GpuDevice::init(caps.device_id).expect("init gpu");
    let mut w_gpu =
        GpuBuffer::alloc_for_device(tv.data.len(), caps.device_id).expect("alloc weights");
    w_gpu.copy_from_host(tv.data).expect("upload weights");
    let mut x_gpu = GpuBuffer::alloc_for_device(in_dim * 4, caps.device_id).expect("alloc input");
    x_gpu
        .copy_from_host(unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, in_dim * 4) })
        .expect("upload input");
    let y_gpu = GpuBuffer::alloc_for_device(out_dim * 4, caps.device_id).expect("alloc output");

    gemv_q8_0_f32(
        w_gpu.as_ptr(),
        x_gpu.as_ptr() as *const f32,
        y_gpu.as_ptr() as *mut f32,
        in_dim,
        out_dim,
    )
    .expect("gemv q8_0");

    let mut y_host = vec![0.0f32; out_dim];
    y_gpu
        .copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(y_host.as_mut_ptr() as *mut u8, out_dim * 4)
        })
        .expect("download output");

    let max_err = max_abs_err(&y_host, &y_ref);
    let mean_err = mean_abs_err(&y_host, &y_ref);
    println!(
        "Q8_0 {} out={} in={} max_err={:.6} mean_err={:.6}",
        tensor_name, out_dim, in_dim, max_err, mean_err
    );
    println!("  first 10 GPU: {:?}", &y_host[..10.min(y_host.len())]);
    println!("  first 10 CPU: {:?}", &y_ref[..10.min(y_host.len())]);
    assert!(max_err < 1.0, "Q8_0 GEMV max error too large: {}", max_err);
}

#[test]
fn test_q8_0_gemv_vs_cpu() {
    check_q8_0_gemv(Q4K_MODEL, "blk.0.attn_v.weight");
}

fn check_q8_0_embed(model_path: &str, token_ids: &[u32]) {
    let gguf = GgufFile::open(model_path).expect("open model");
    let tv = gguf
        .tensor("token_embd.weight")
        .expect("lookup")
        .expect("exists");
    assert_eq!(tv.ggml_type, GgmlType::Q8_0);
    let dims: Vec<usize> = tv.dims.iter().map(|&d| d as usize).collect();
    let hidden_size = dims[0];
    let vocab_size = dims[1];

    let caps = gpu::detect().expect("detect gpu");
    let _device = GpuDevice::init(caps.device_id).expect("init gpu");
    let mut emb_gpu =
        GpuBuffer::alloc_for_device(tv.data.len(), caps.device_id).expect("alloc embedding");
    emb_gpu.copy_from_host(tv.data).expect("upload embedding");

    for &token_id in token_ids {
        let mut out_ref = vec![0.0f32; hidden_size];
        embed_q8_0(token_id as usize, tv.data, &mut out_ref, hidden_size);

        let out_gpu =
            GpuBuffer::alloc_for_device(hidden_size * 4, caps.device_id).expect("alloc output");
        embed_q8_0_token(
            emb_gpu.as_ptr(),
            out_gpu.as_ptr() as *mut f32,
            hidden_size,
            vocab_size,
            token_id,
        )
        .expect("embed q8_0 token");

        let mut out_host = vec![0.0f32; hidden_size];
        out_gpu
            .copy_to_host(unsafe {
                std::slice::from_raw_parts_mut(out_host.as_mut_ptr() as *mut u8, hidden_size * 4)
            })
            .expect("download output");

        let max_err = max_abs_err(&out_host, &out_ref);
        let mean_err = mean_abs_err(&out_host, &out_ref);
        println!(
            "Q8_0 embed token_id={} hidden={} vocab={} max_err={:.6} mean_err={:.6}",
            token_id, hidden_size, vocab_size, max_err, mean_err
        );
        assert!(max_err < 1.0, "Q8_0 embed max error too large: {}", max_err);
    }
}

#[test]
fn test_q8_0_embed_vs_cpu() {
    check_q8_0_embed(Q4K_MODEL, &[0, 1, 100, 1000, 151935]);
}

fn check_q8_0_lm_head(model_path: &str) {
    let gguf = GgufFile::open(model_path).expect("open model");
    let tv = gguf
        .tensor("token_embd.weight")
        .expect("lookup")
        .expect("exists");
    assert_eq!(tv.ggml_type, GgmlType::Q8_0);
    let dims: Vec<usize> = tv.dims.iter().map(|&d| d as usize).collect();
    let hidden_size = dims[0];
    let vocab_size = dims[1];

    let mut rng = Rng::new();
    let x: Vec<f32> = (0..hidden_size).map(|_| rng.f32() * 4.0 - 2.0).collect();

    // CPU reference: y = W^T * x where W is [hidden_size, vocab_size]
    let mut y_ref = vec![0.0f32; vocab_size];
    gemm_q8_0(tv.data, &x, &mut y_ref, vocab_size, hidden_size);

    let caps = gpu::detect().expect("detect gpu");
    let _device = GpuDevice::init(caps.device_id).expect("init gpu");
    let mut w_gpu =
        GpuBuffer::alloc_for_device(tv.data.len(), caps.device_id).expect("alloc lm head weights");
    w_gpu.copy_from_host(tv.data).expect("upload weights");
    let mut x_gpu =
        GpuBuffer::alloc_for_device(hidden_size * 4, caps.device_id).expect("alloc input");
    x_gpu
        .copy_from_host(unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, hidden_size * 4)
        })
        .expect("upload input");
    let y_gpu = GpuBuffer::alloc_for_device(vocab_size * 4, caps.device_id).expect("alloc output");

    rocmforge::gpu::kernels::gemv_q8_0_f32_lm_head(
        w_gpu.as_ptr(),
        x_gpu.as_ptr() as *const f32,
        y_gpu.as_ptr() as *mut f32,
        hidden_size,
        vocab_size,
    )
    .expect("lm head q8_0");

    let mut y_host = vec![0.0f32; vocab_size];
    y_gpu
        .copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(y_host.as_mut_ptr() as *mut u8, vocab_size * 4)
        })
        .expect("download output");

    let max_err = max_abs_err(&y_host, &y_ref);
    let mean_err = mean_abs_err(&y_host, &y_ref);
    println!(
        "Q8_0 lm_head hidden={} vocab={} max_err={:.6} mean_err={:.6}",
        hidden_size, vocab_size, max_err, mean_err
    );
    println!(
        "  ref max={:.6} min={:.6}",
        y_ref.iter().copied().fold(f32::NAN, f32::max),
        y_ref.iter().copied().fold(f32::NAN, f32::min)
    );
    println!(
        "  gpu max={:.6} min={:.6}",
        y_host.iter().copied().fold(f32::NAN, f32::max),
        y_host.iter().copied().fold(f32::NAN, f32::min)
    );
    assert!(
        max_err < 1.0,
        "Q8_0 LM head max error too large: {}",
        max_err
    );
}

#[test]
fn test_q8_0_lm_head_vs_cpu() {
    check_q8_0_lm_head(Q4K_MODEL);
}
