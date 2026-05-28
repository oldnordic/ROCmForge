#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::gpu::kernels::dispatch_mpo_apply_f32;
use rocmforge::gpu::{GpuBuffer, GpuDevice};
use serial_test::serial;

fn skip_unless_experimental() {
    if !rocmforge::gpu::safety::run_experimental_gpu_tests_enabled() {
        eprintln!("Skipping — set ROCMFORGE_RUN_EXPERIMENTAL_GPU_TESTS=1 to enable experimental GPU tests");
        std::process::exit(0);
    }
}

fn upload_f32(data: &[f32]) -> rocmforge::gpu::GpuResult<GpuBuffer> {
    let mut buf = GpuBuffer::alloc(std::mem::size_of_val(data))?;
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    buf.copy_from_host(bytes)?;
    Ok(buf)
}

fn upload_u32(data: &[u32]) -> rocmforge::gpu::GpuResult<GpuBuffer> {
    let mut buf = GpuBuffer::alloc(std::mem::size_of_val(data))?;
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    buf.copy_from_host(bytes)?;
    Ok(buf)
}

fn download_f32(buf: &GpuBuffer, len: usize) -> rocmforge::gpu::GpuResult<Vec<f32>> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    buf.copy_to_host(&mut bytes)?;
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() })
}

fn init_device() -> GpuDevice {
    GpuDevice::init(0).expect("GPU should initialize")
}

fn expect<T>(r: rocmforge::gpu::GpuResult<T>) -> T {
    r.expect("GPU operation should succeed")
}

/// Build a simple 2-site MPO: A = A1 * A2 where A1 is [d1, chi] and A2 is [chi, d2]
/// The flattened input is x[d2], output y[d1] = A1 * A2 * x
fn mpo_2site_reference(
    a1: &[f32], // [d1, chi]
    a2: &[f32], // [chi, d2]
    d1: usize,
    chi: usize,
    d2: usize,
    x: &[f32],
) -> Vec<f32> {
    // temp = A2 * x  [chi]
    let mut temp = vec![0.0f32; chi];
    for i in 0..chi {
        let mut sum = 0.0f32;
        for j in 0..d2 {
            sum += a2[i * d2 + j] * x[j];
        }
        temp[i] = sum;
    }
    // y = A1 * temp  [d1]
    let mut y = vec![0.0f32; d1];
    for i in 0..d1 {
        let mut sum = 0.0f32;
        for j in 0..chi {
            sum += a1[i * chi + j] * temp[j];
        }
        y[i] = sum;
    }
    y
}

#[test]
#[serial]
fn test_mpo_apply_2site_basic() {
    skip_unless_experimental();
    let device = init_device();
    let d1 = 8usize;
    let chi = 4usize;
    let d2 = 6usize;

    // Random-ish site tensors
    let mut a1 = vec![0.0f32; d1 * chi];
    for (i, v) in a1.iter_mut().enumerate() {
        *v = (i as f32) * 0.01 - 0.15;
    }
    let mut a2 = vec![0.0f32; chi * d2];
    for (i, v) in a2.iter_mut().enumerate() {
        *v = (i as f32) * 0.02 + 0.05;
    }
    let mut x = vec![0.0f32; d2];
    for (i, v) in x.iter_mut().enumerate() {
        *v = (i as f32) * 0.03 - 0.1;
    }

    let ref_y = mpo_2site_reference(&a1, &a2, d1, chi, d2, &x);

    // GPU: site_dims = [1, d1, chi, 1, chi, d2, 1, 1] for n_sites=2
    let site_dims: Vec<u32> = vec![1, d1 as u32, chi as u32, 1, chi as u32, d2 as u32, 1, 1];
    let site_data: Vec<f32> = a1.iter().chain(a2.iter()).copied().collect();

    let gpu_sites = expect(upload_f32(&site_data));
    let gpu_dims = expect(upload_u32(&site_dims));
    let gpu_x = expect(upload_f32(&x));
    let gpu_y = expect(upload_f32(&vec![0.0f32; d1]));

    expect(dispatch_mpo_apply_f32(
        gpu_sites.as_ptr() as *const f32,
        gpu_dims.as_ptr() as *const u32,
        2,  // n_sites
        d1, // out_dim
        d2, // in_dim
        gpu_x.as_ptr() as *const f32,
        gpu_y.as_ptr() as *mut f32,
        device.stream(),
    ));

    expect(device.synchronize());

    let out_y = expect(download_f32(&gpu_y, d1));

    for i in 0..d1 {
        assert!(
            (out_y[i] - ref_y[i]).abs() < 1e-4,
            "MPO apply mismatch at {}: actual={}, ref={}",
            i,
            out_y[i],
            ref_y[i]
        );
    }
}

#[test]
#[serial]
fn test_mpo_apply_2site_identity_like() {
    skip_unless_experimental();
    let device = init_device();
    let d1 = 4usize;
    let chi = 2usize;
    let d2 = 4usize;

    // A1 = [[1,0],[0,1],[1,0],[0,1]] (d1=4, chi=2)
    // A2 = [[1,0,0,0],[0,1,0,0]] (chi=2, d2=4)
    // A = A1*A2 = identity on first 2 dims, zeros elsewhere
    let a1 = vec![
        1.0f32, 0.0f32, 0.0f32, 1.0f32, 1.0f32, 0.0f32, 0.0f32, 1.0f32,
    ];
    let a2 = vec![
        1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32,
    ];
    let x = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];

    let ref_y = mpo_2site_reference(&a1, &a2, d1, chi, d2, &x);

    let site_data: Vec<f32> = a1.iter().chain(a2.iter()).copied().collect();
    let site_dims: Vec<u32> = vec![1, d1 as u32, chi as u32, 1, chi as u32, d2 as u32, 1, 1];

    let gpu_sites = expect(upload_f32(&site_data));
    let gpu_dims = expect(upload_u32(&site_dims));
    let gpu_x = expect(upload_f32(&x));
    let gpu_y = expect(upload_f32(&vec![0.0f32; d1]));

    expect(dispatch_mpo_apply_f32(
        gpu_sites.as_ptr() as *const f32,
        gpu_dims.as_ptr() as *const u32,
        2,
        d1,
        d2,
        gpu_x.as_ptr() as *const f32,
        gpu_y.as_ptr() as *mut f32,
        device.stream(),
    ));

    expect(device.synchronize());

    let out_y = expect(download_f32(&gpu_y, d1));

    for i in 0..d1 {
        assert!(
            (out_y[i] - ref_y[i]).abs() < 1e-4,
            "MPO identity-like mismatch at {}: actual={}, ref={}",
            i,
            out_y[i],
            ref_y[i]
        );
    }
}

#[test]
#[serial]
fn test_mpo_apply_3site_basic() {
    skip_unless_experimental();
    let device = init_device();
    let d1 = 4usize;
    let d2 = 3usize;
    let d3 = 5usize;
    let chi1 = 2usize;
    let chi2 = 3usize;

    // 3-site MPO: A1[d1, chi1], A2[chi1, d2, chi2], A3[chi2, d3]
    // Flattened: A1 is [d1, chi1], A2 is [chi1, d2, chi2], A3 is [chi2, d3]
    // Actually per MPO convention: site i has shape [chi_left, d_i, chi_right]
    // So A1: [1, d1, chi1], A2: [chi1, d2, chi2], A3: [chi2, d3, 1]
    let a1: Vec<f32> = (0..(d1 * chi1)).map(|i| (i as f32) * 0.01 - 0.1).collect();
    let a2: Vec<f32> = (0..(chi1 * d2 * chi2))
        .map(|i| (i as f32) * 0.02 + 0.05)
        .collect();
    let a3: Vec<f32> = (0..(chi2 * d3)).map(|i| (i as f32) * 0.03 - 0.05).collect();

    let mut x = vec![0.0f32; d3];
    for (i, v) in x.iter_mut().enumerate() {
        *v = (i as f32) * 0.04 + 0.1;
    }

    // CPU reference: contract A3*x -> temp[chi2], then A2*temp -> temp2[chi1*d2], then A1*temp2 -> y[d1]
    // Actually the full contraction is:
    //   y[i1] = sum_{i2,i3,j1,j2} A1[0,i1,j1] * A2[j1,i2,j2] * A3[j2,i3,0] * x[i3]
    // Step 1: t2[j2] = sum_{i3} A3[j2,i3,0] * x[i3]
    let mut t2 = vec![0.0f32; chi2];
    for j2 in 0..chi2 {
        let mut sum = 0.0f32;
        for i3 in 0..d3 {
            sum += a3[j2 * d3 + i3] * x[i3];
        }
        t2[j2] = sum;
    }
    // Step 2: t1[j1] = sum_{i2,j2} A2[j1,i2,j2] * t2[j2]
    let mut t1 = vec![0.0f32; chi1];
    for j1 in 0..chi1 {
        let mut sum = 0.0f32;
        for i2 in 0..d2 {
            for j2 in 0..chi2 {
                sum += a2[j1 * d2 * chi2 + i2 * chi2 + j2] * t2[j2];
            }
        }
        t1[j1] = sum;
    }
    // Step 3: y[i1] = sum_{j1} A1[0,i1,j1] * t1[j1]
    let mut ref_y = vec![0.0f32; d1];
    for i1 in 0..d1 {
        let mut sum = 0.0f32;
        for j1 in 0..chi1 {
            sum += a1[i1 * chi1 + j1] * t1[j1];
        }
        ref_y[i1] = sum;
    }

    let site_data: Vec<f32> = a1
        .iter()
        .chain(a2.iter())
        .chain(a3.iter())
        .copied()
        .collect();
    let site_dims: Vec<u32> = vec![
        1,
        d1 as u32,
        chi1 as u32,
        1,
        chi1 as u32,
        d2 as u32,
        chi2 as u32,
        1,
        chi2 as u32,
        d3 as u32,
        1,
        1,
    ];

    let gpu_sites = expect(upload_f32(&site_data));
    let gpu_dims = expect(upload_u32(&site_dims));
    let gpu_x = expect(upload_f32(&x));
    let gpu_y = expect(upload_f32(&vec![0.0f32; d1]));

    expect(dispatch_mpo_apply_f32(
        gpu_sites.as_ptr() as *const f32,
        gpu_dims.as_ptr() as *const u32,
        3,
        d1,
        d3,
        gpu_x.as_ptr() as *const f32,
        gpu_y.as_ptr() as *mut f32,
        device.stream(),
    ));

    expect(device.synchronize());

    let out_y = expect(download_f32(&gpu_y, d1));

    for i in 0..d1 {
        assert!(
            (out_y[i] - ref_y[i]).abs() < 1e-3,
            "MPO 3-site mismatch at {}: actual={}, ref={}",
            i,
            out_y[i],
            ref_y[i]
        );
    }
}
