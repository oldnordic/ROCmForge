#[cfg(feature = "gpu")]
#[macro_use]
mod common;

#[cfg(feature = "gpu")]
mod turboquant_tests {
    use rocmforge::config::{ModelConfig, TensorNameRegistry, TensorNamingScheme};
    use rocmforge::gpu::{GpuBuffer, GpuDevice, GpuForwardScratch, GpuKvCache};
    use serial_test::serial;

    fn make_turboquant_test_config() -> ModelConfig {
        ModelConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 128,
            max_seq_len: 256,
            hidden_size: 128,
            num_heads: 1,
            intermediate_size: 128,
            vocab_size: 1000,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_neox: false,
            use_attention_bias: false,
            attention_layout: rocmforge::config::AttentionLayout::SplitQkv,
        ffn_layout: rocmforge::config::FfnLayout::SwiGLU,
            architecture: "test".to_string(),
            tensor_registry: TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf),
            shortconv_l_cache: None,
            num_dense_layers: None,
            num_experts_per_tok: None,
            use_expert_bias: false,
            expert_weights_scale: 1.0,
            rope_freq: (0..64)
                .map(|i| 1.0 / 10000.0f32.powf((2 * i) as f32 / 128.0f32))
                .collect(),
            kv_lora_dim: Some(128),
            kv_frame_codec_enabled: None,
            adastate_anchors_enabled: None,
            kv_quant_bits: Some(3),
            turboquant_centroids: Some(vec![-3.0, -1.8, -1.0, -0.4, 0.4, 1.0, 1.8, 3.0]),
            qjl_scale: Some(0.25),
        }
    }

    fn host_fwht(data: &mut [f32]) {
        let n = data.len();
        let mut len = 1;
        while len < n {
            let chunk = len * 2;
            for i in (0..n).step_by(chunk) {
                for j in 0..len {
                    let u = data[i + j];
                    let v = data[i + j + len];
                    data[i + j] = u + v;
                    data[i + j + len] = u - v;
                }
            }
            len <<= 1;
        }
    }

    fn get_host_random_projection_sign(i: i32, j: i32) -> f32 {
        let mut seed = (i as u32)
            .wrapping_mul(1664525)
            .wrapping_add(j as u32)
            .wrapping_add(1013904223);
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        if (seed & 1) != 0 {
            1.0f32
        } else {
            -1.0f32
        }
    }

    fn host_turboquant_simulate(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        centroids: &[f32],
        _qjl_scale: f32,
    ) -> (f32, Vec<f32>) {
        let d = q.len();

        // 1. Scale input vectors before FWHT dynamically to match GPU kernel sequence
        let mut rot_q = q.to_vec();
        let mut rot_k = k.to_vec();
        let mut rot_v = v.to_vec();

        let mut sum_sq_k = 0.0f32;
        let mut sum_sq_v = 0.0f32;
        for i in 0..d {
            sum_sq_k += rot_k[i] * rot_k[i];
            sum_sq_v += rot_v[i] * rot_v[i];
        }
        let rms_k = if sum_sq_k > 0.0f32 {
            (sum_sq_k / (d as f32)).sqrt()
        } else {
            1.0f32
        };
        let rms_v = if sum_sq_v > 0.0f32 {
            (sum_sq_v / (d as f32)).sqrt()
        } else {
            1.0f32
        };

        println!("HOST: rms_k = {}, rms_v = {}", rms_k, rms_v);

        let scale_k = rms_k * (d as f32).sqrt();
        let scale_v = rms_v * (d as f32).sqrt();

        for i in 0..d {
            rot_k[i] /= scale_k;
            rot_v[i] /= scale_v;
        }

        host_fwht(&mut rot_q);
        host_fwht(&mut rot_k);
        host_fwht(&mut rot_v);

        print!("HOST rot_v: ");
        for &val in rot_v.iter().take(16) {
            print!("{} ", val);
        }
        println!();

        // 2. 3-bit Quantize rot_k
        let mut idx_k = vec![0; d];
        let mut deq_k = vec![0.0f32; d];
        for i in 0..d {
            let val = rot_k[i];
            let mut best_c = 0;
            let mut min_dist = (val - centroids[0]).abs();
            for (c, &centroid) in centroids.iter().enumerate().take(8).skip(1) {
                let dist = (val - centroid).abs();
                if dist < min_dist {
                    min_dist = dist;
                    best_c = c;
                }
            }
            idx_k[i] = best_c;
            deq_k[i] = centroids[best_c];
        }

        // 3. QJL residuals of K
        let mut sign_k = vec![0.0f32; d];
        for i in 0..d {
            let residual = rot_k[i] - deq_k[i];
            let mut sum_k = 0.0f32;
            for col in 0..d {
                let r_sign = get_host_random_projection_sign(i as i32, col as i32);
                sum_k += r_sign * residual;
            }
            sign_k[i] = if sum_k >= 0.0f32 { 1.0 } else { -1.0 };
        }

        // 4. QJL sign of Q
        let mut sign_q = vec![0.0f32; d];
        for (i, val) in sign_q.iter_mut().enumerate() {
            let mut sum_q = 0.0f32;
            for (col, &q_val) in rot_q.iter().enumerate() {
                let r_sign = get_host_random_projection_sign(i as i32, col as i32);
                sum_q += r_sign * q_val;
            }
            *val = if sum_q >= 0.0f32 { 1.0 } else { -1.0 };
        }

        // 5. Score computation
        let score_base = rot_q
            .iter()
            .zip(deq_k.iter())
            .map(|(&qi, &ki)| qi * ki)
            .sum::<f32>()
            / (d as f32).sqrt();
        let score_base = score_base * rms_k;

        let corr_sum = sign_q
            .iter()
            .zip(sign_k.iter())
            .map(|(&sq, &sk)| sq * sk)
            .sum::<f32>();
        let correction = (rms_k / (d as f32).sqrt()) * corr_sum;

        let score = score_base + correction;

        // 6. Dequantize V
        let mut idx_v = vec![0; d];
        let mut deq_v = vec![0.0f32; d];
        for i in 0..d {
            let val = rot_v[i];
            let mut best_c = 0;
            let mut min_dist = (val - centroids[0]).abs();
            for (c, &centroid) in centroids.iter().enumerate().take(8).skip(1) {
                let dist = (val - centroid).abs();
                if i == 3 {
                    println!("  c={}: centroid={}, dist={}", c, centroid, dist);
                }
                if dist < min_dist {
                    min_dist = dist;
                    best_c = c;
                }
            }
            if i == 3 {
                println!("  => best_c={}", best_c);
            }
            idx_v[i] = best_c;
            deq_v[i] = centroids[best_c] * (rms_v / (d as f32).sqrt());
        }
        print!("HOST idx_v: ");
        for &val in idx_v.iter().take(16) {
            print!("{} ", val);
        }
        println!();
        println!(
            "HOST BEFORE FWHT: deq_v[0] = {}, idx_v[0] = {}",
            deq_v[0], idx_v[0]
        );
        let mut restored_v = deq_v.clone();
        host_fwht(&mut restored_v);
        println!("HOST AFTER FWHT: restored_v[0] = {}", restored_v[0]);

        (score, restored_v)
    }

    #[test]
    #[serial]
    fn test_turboquant_kv_write_and_decode_parity() {
        require_gpu!();
        require_vram!(4);

        let config = make_turboquant_test_config();
        let device = GpuDevice::init(0).expect("Initialize device");
        let mut cache = GpuKvCache::new(&config, 256).expect("KV cache allocation");

        // Generate random vector data
        let mut test_k = vec![0.0f32; 128];
        let mut test_v = vec![0.0f32; 128];
        let mut test_q = vec![0.0f32; 128];

        for i in 0..128 {
            test_k[i] = ((i as f32 * 12.345f32).sin() * 1.5f32) - 0.1f32;
            test_v[i] = ((i as f32 * 8.567f32).sin() * 0.8f32) + 0.2f32;
            test_q[i] = ((i as f32 * 5.789f32).sin() * 1.2f32) - 0.3f32;
        }

        // Host reference simulation
        let centroids = config
            .turboquant_centroids
            .as_ref()
            .expect("invariant: turboquant_centroids set after TurboQuant conversion");
        let qjl_scale = config
            .qjl_scale
            .expect("invariant: qjl_scale set after TurboQuant conversion");
        let (_expected_score, expected_v) =
            host_turboquant_simulate(&test_q, &test_k, &test_v, centroids, qjl_scale);

        // Upload Query Q to scratch
        let mut scratch = GpuForwardScratch::new(&config).expect("Alloc scratch");
        let q_bytes = unsafe {
            std::slice::from_raw_parts(
                test_q.as_ptr() as *const u8,
                128 * std::mem::size_of::<f32>(),
            )
        };
        scratch.q.copy_from_host(q_bytes).expect("Query upload");

        // Copy Key and Value vectors to GPU inputs
        let mut k_gpu = GpuBuffer::alloc(128 * std::mem::size_of::<f32>()).expect("Alloc K");
        let mut v_gpu = GpuBuffer::alloc(128 * std::mem::size_of::<f32>()).expect("Alloc V");
        let k_bytes = unsafe {
            std::slice::from_raw_parts(
                test_k.as_ptr() as *const u8,
                128 * std::mem::size_of::<f32>(),
            )
        };
        let v_bytes = unsafe {
            std::slice::from_raw_parts(
                test_v.as_ptr() as *const u8,
                128 * std::mem::size_of::<f32>(),
            )
        };
        k_gpu.copy_from_host(k_bytes).expect("Upload K");
        v_gpu.copy_from_host(v_bytes).expect("Upload V");

        // Write Key/Value vectors into cache (TurboQuant 3-bit compression)
        cache
            .write(
                0,
                0,
                k_gpu.as_ptr() as *const f32,
                v_gpu.as_ptr() as *const f32,
            )
            .expect("KV cache compressed write");

        // Execute attention decode (Dequantization + FWHT dot product + QJL residual correction)
        let scale = 1.0f32 / (128.0f32).sqrt();
        rocmforge::gpu::flash_attn_decode_turboquant(
            scratch.attn_out.as_ptr() as *mut f32,
            scratch.q.as_ptr() as *const f32,
            cache.k_ptr(0).expect("K ptr") as *const u8,
            cache.v_ptr(0).expect("V ptr") as *const u8,
            1,   // seq_len
            1,   // num_heads
            1,   // num_kv_heads
            128, // head_dim
            scale,
            128, // kv_lora_dim
            3,   // bits
            8,   // num_centroids
            cache.centroids_ptr().expect("centroids ptr"),
            cache.qjl_scale,
            std::ptr::null(), // w_up_k
            std::ptr::null(), // w_up_v
            device.stream(),
        )
        .expect("Attention decode");

        // Download attention output
        let mut downloaded_out = vec![0.0f32; 128];
        let out_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                downloaded_out.as_mut_ptr() as *mut u8,
                128 * std::mem::size_of::<f32>(),
            )
        };
        scratch
            .attn_out
            .copy_to_host(out_bytes)
            .expect("Download output");

        // Softmax of a single position is 1.0, so downloaded_out should exactly match dequantized/restored_v!
        // We assert L-infinity error is <= 10^-5
        for idx in 0..128 {
            let actual = downloaded_out[idx];
            let expected = expected_v[idx];
            let err = (actual - expected).abs();
            assert!(
                err <= 1e-5,
                "Numerical mismatch at index {}: actual={:.6}, expected={:.6} (err={:.6e})",
                idx,
                actual,
                expected,
                err
            );
        }
    }
}
