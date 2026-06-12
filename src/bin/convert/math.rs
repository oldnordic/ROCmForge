use rayon::prelude::*;

pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

pub fn svd_decompose(
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

pub fn svd_batch_experts(
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
    let results: Vec<_> = (0..n_experts)
        .into_par_iter()
        .map(|e| {
            let slice = &matrices[e * rows * cols..(e + 1) * rows * cols];
            top_k_svd_quant(slice, rows, cols, k)
        })
        .collect();

    let mut all_u = Vec::<f32>::with_capacity(n_experts * rows * k);
    let mut all_v = Vec::<f32>::with_capacity(n_experts * k * cols);
    for (u, v) in results {
        all_u.extend_from_slice(&u);
        all_v.extend_from_slice(&v);
    }
    Ok((all_u, all_v))
}

pub fn top_k_svd_quant(a: &[f32], m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
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

pub fn fwht_inplace(a: &mut [f32]) {
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

#[cfg(test)]
mod tests {
    use super::{matmul, top_k_svd_quant};

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
}
