//! CPU vs GPU attention numerical accuracy tests

#[cfg(feature = "rocm")]
use proptest::prelude::*;
#[cfg(feature = "rocm")]
use proptest::test_runner::TestRunner;
#[cfg(feature = "rocm")]
use serial_test::serial;

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_cpu_gpu_attention_close_enough() {
    let mut runner = TestRunner::default();

    runner
        .run(
            &generate_attention_test_case(),
            |(batch_size, seq_len, dim)| {
                let q = Tensor::random_seeded(batch_size * seq_len * dim, 42);
                let k = Tensor::random_seeded(batch_size * seq_len * dim, 42);
                let v = Tensor::random_seeded(batch_size * seq_len * dim, 42);

                // Run CPU attention
                let cpu_attention = Attention::with_backend(dim, AttentionBackend::Cpu);
                let cpu_output = cpu_attention
                    .forward(&q.data, &k.data, &v.data, None, None)
                    .unwrap();

                // Run GPU attention
                let gpu_attention = Attention::with_backend(dim, AttentionBackend::Gpu);
                let gpu_output = gpu_attention
                    .forward(&q.data, &k.data, &v.data, None, None)
                    .unwrap();

                // Compare outputs
                let max_diff = max_abs_diff(&cpu_output, &gpu_output);

                prop_assert!(
                    max_diff <= 1e-4,
                    "CPU vs GPU max difference {} exceeds tolerance 1e-4",
                    max_diff
                );

                Ok(())
            },
        )
        .unwrap();
}

#[cfg(feature = "rocm")]
fn generate_attention_test_case() -> impl Strategy<Value = (usize, usize, usize)> {
    (1usize..=4usize).prop_flat_map(|batch_size| {
        (64usize..=257usize).prop_flat_map(move |seq_len| {
            (32usize..=129usize).prop_map(move |dim| (batch_size, seq_len, dim))
        })
    })
}
