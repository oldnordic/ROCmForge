#![cfg(feature = "gpu")]
#![allow(warnings)]

//! Correctness test for FFN gate_up operation using synthetic data.

mod common;

use common::helpers::*;
use rocmforge::config::{AttentionLayout, FfnLayout, ModelConfig, TensorNameRegistry, TensorNamingScheme, TensorRole};
use rocmforge::cpu::ops::dispatch_gemv as cpu_dispatch_gemv;
use rocmforge::gpu::{GpuBuffer, GpuDevice, GpuForwardScratch, WeightMeta};
use rocmforge::loader::GgmlType;
use serial_test::serial;

#[test]
#[serial]
fn test_gate_up_operation_correctness_synthetic() {
    require_gpu!();
    let device = GpuDevice::init(0).expect("GPU init failed");
    let dev_id = device.device_id();
    let config = mock_model_config();
    let h = config.hidden_size;
    let ff = config.intermediate_size;

    eprintln!("=== Gate_Up Operation Correctness (Synthetic) ===");

    // 1. Setup synthetic weights (Q4_0)
    let gate_data_f32 = (0..ff*h).map(|i| (i as f32).sin() * 0.1).collect::<Vec<f32>>();
    let up_data_f32 = (0..ff*h).map(|i| (i as f32).cos() * 0.1).collect::<Vec<f32>>();
    
    let gate_q4 = quantize_q4_0(&gate_data_f32);
    let up_q4 = quantize_q4_0(&up_data_f32);
    
    let gate_gpu = upload_raw(dev_id, &gate_q4);
    let up_gpu = upload_raw(dev_id, &up_q4);
    
    let meta = mock_gpu_meta(ff, h, GgmlType::Q4_0, TensorRole::Generic);
    let cpu_meta = mock_cpu_meta(ff, h, GgmlType::Q4_0, TensorRole::Generic);

    // 2. Setup input
    let input_f32 = (0..h).map(|i| (i as f32) * 0.01).collect::<Vec<f32>>();
    let input_gpu = upload_f32(dev_id, &input_f32);

    // 3. Run CPU reference
    let mut cpu_gate = vec![0.0f32; ff];
    let mut cpu_up = vec![0.0f32; ff];
    
    cpu_dispatch_gemv(&gate_q4, &cpu_meta, &input_f32, &mut cpu_gate, ff, h, None).expect("cpu gate gemv");
    cpu_dispatch_gemv(&up_q4, &cpu_meta, &input_f32, &mut cpu_up, ff, h, None).expect("cpu up gemv");
    
    let mut cpu_swiglu = vec![0.0f32; ff];
    for i in 0..ff {
        let silu = cpu_gate[i] / (1.0 + (-cpu_gate[i]).exp());
        cpu_swiglu[i] = silu * cpu_up[i];
    }

    // 4. Run GPU fused gate_up
    let mut gpu_gate = GpuBuffer::alloc_for_device(ff * 4, dev_id).expect("alloc gpu gate");
    let mut gpu_swiglu = GpuBuffer::alloc_for_device(ff * 4, dev_id).expect("alloc gpu swiglu");

    rocmforge::gpu::ops::gpu_dispatch_fused_gate_up_on_stream(
        &device,
        &gate_gpu,
        &meta,
        &up_gpu,
        &meta,
        None, // interleaved
        None, // interleaved_tile4
        input_gpu.as_ptr() as *const f32,
        gpu_gate.as_ptr() as *mut f32,
        gpu_swiglu.as_ptr() as *mut f32,
        ff,
        h,
        device.stream(),
    ).expect("gpu dispatch gate_up");
    device.synchronize().expect("sync");

    // 5. Compare
    let gpu_swiglu_res = download_f32(&gpu_swiglu, ff);
    
    let err = max_abs_error(&cpu_swiglu, &gpu_swiglu_res);
    eprintln!("Max abs error: {:.6}", err);
    assert!(err < 1e-3, "FFN fused gate_up parity failed: err={}", err);
}

fn upload_raw(device_id: i32, bytes: &[u8]) -> GpuBuffer {
    let mut buf = GpuBuffer::alloc_for_device(bytes.len(), device_id).expect("upload_raw alloc");
    buf.copy_from_host(bytes).expect("upload_raw copy");
    buf
}
