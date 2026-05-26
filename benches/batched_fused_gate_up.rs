#![allow(warnings)]
//! Isolated benchmark for batched_fused_gate_up_q4_0_f32_prefill_kernel.
//!
//! Measures kernel performance without full model loading noise.
//! Uses synthetic Q4_0 data at realistic Qwen2.5-0.5B dimensions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rocmforge::gpu::kernels::batched_fused_gate_up_q4_0_f32;
use rocmforge::gpu::quant::{Q4_0_BLOCK_SIZE, QK4_0};
use rocmforge::gpu::{self, GpuDevice};

// Realistic dimensions from Qwen2.5-0.5B-Instruct
const HIDDEN_SIZE: usize = 896;
const FF_SIZE: usize = 4864; // Intermediate FFN dimension

struct GateUpBenchContext {
    device: GpuDevice,
    gate_weights_gpu: gpu::GpuBuffer,
    up_weights_gpu: gpu::GpuBuffer,
    input_gpu: gpu::GpuBuffer,
    output_gpu: gpu::GpuBuffer,
}

impl GateUpBenchContext {
    fn new() -> Result<Self, String> {
        if !gpu::run_gpu_benches_enabled() {
            return Err(format!(
                "set {}=1 to run GPU benchmarks",
                gpu::RUN_GPU_BENCHES_ENV
            ));
        }

        let caps = gpu::detect().ok_or_else(|| "GPU not detected".to_string())?;
        let device =
            GpuDevice::init(caps.device_id).map_err(|err| format!("GPU init failed: {}", err))?;

        // Calculate Q4_0 weight matrix sizes
        // Weights layout: [ff_size][n_rows / QK4_0][Q4_0_BLOCK_SIZE]
        let n_blocks_per_col = (HIDDEN_SIZE + QK4_0 - 1) / QK4_0;
        let gate_weights_bytes = FF_SIZE * n_blocks_per_col * Q4_0_BLOCK_SIZE;
        let up_weights_bytes = FF_SIZE * n_blocks_per_col * Q4_0_BLOCK_SIZE;

        // Allocate GPU buffers
        let mut gate_weights_gpu = gpu::GpuBuffer::alloc(gate_weights_bytes)
            .map_err(|err| format!("gate weights alloc failed: {}", err))?;

        let mut up_weights_gpu = gpu::GpuBuffer::alloc(up_weights_bytes)
            .map_err(|err| format!("up weights alloc failed: {}", err))?;

        // Generate synthetic Q4_0 weights on CPU
        let gate_weights_cpu: Vec<u8> = vec![0; gate_weights_bytes];
        let up_weights_cpu: Vec<u8> = vec![0; up_weights_bytes];

        // Copy weights to GPU
        gate_weights_gpu
            .copy_from_host(&gate_weights_cpu)
            .map_err(|err| format!("gate weights H2D failed: {}", err))?;

        up_weights_gpu
            .copy_from_host(&up_weights_cpu)
            .map_err(|err| format!("up weights H2D failed: {}", err))?;

        Ok(Self {
            device,
            gate_weights_gpu,
            up_weights_gpu,
            input_gpu: gpu::GpuBuffer::empty(),
            output_gpu: gpu::GpuBuffer::empty(),
        })
    }

    fn alloc_io(&mut self, seq_len: usize) -> Result<(), String> {
        // Input: [seq_len][hidden_size]
        let input_bytes = seq_len * HIDDEN_SIZE * std::mem::size_of::<f32>();
        let mut input_gpu = gpu::GpuBuffer::alloc(input_bytes)
            .map_err(|err| format!("input alloc failed: {}", err))?;

        // Output: [seq_len][ff_size]
        let output_bytes = seq_len * FF_SIZE * std::mem::size_of::<f32>();
        let output_gpu = gpu::GpuBuffer::alloc(output_bytes)
            .map_err(|err| format!("output alloc failed: {}", err))?;

        // Initialize input with synthetic data (all 0.0 to stay in safe range)
        let input_cpu: Vec<u8> = vec![0; input_bytes];

        input_gpu
            .copy_from_host(&input_cpu)
            .map_err(|err| format!("input H2D failed: {}", err))?;

        self.input_gpu = input_gpu;
        self.output_gpu = output_gpu;

        Ok(())
    }

    fn run_kernel(&self, seq_len: usize) -> Result<(), String> {
        let stream = self.device.stream();

        unsafe {
            batched_fused_gate_up_q4_0_f32(
                self.gate_weights_gpu.as_ptr(),
                self.up_weights_gpu.as_ptr(),
                self.input_gpu.as_ptr() as *const f32,
                self.output_gpu.as_ptr() as *mut f32,
                HIDDEN_SIZE,
                FF_SIZE,
                seq_len,
                stream,
            )
        }
        .map_err(|err| format!("kernel launch failed: {}", err))?;

        self.device
            .synchronize()
            .map_err(|err| format!("stream sync failed: {}", err))?;

        Ok(())
    }
}

fn bench_batched_fused_gate_up(c: &mut Criterion) {
    let mut ctx = match GateUpBenchContext::new() {
        Ok(ctx) => ctx,
        Err(err) => {
            println!("Skipping benchmark: {}", err);
            return;
        }
    };

    let mut group = c.benchmark_group("batched_fused_gate_up_q4_0");
    group.throughput(Throughput::Elements(1));

    for &seq_len in &[1, 8, 16, 32, 64] {
        if let Err(err) = ctx.alloc_io(seq_len) {
            println!("Skipping seq_len {}: {}", seq_len, err);
            continue;
        }

        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &seq_len,
            |b, &seq_len| {
                b.iter(|| {
                    let result = ctx.run_kernel(black_box(seq_len));
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_batched_fused_gate_up);
criterion_main!(benches);
