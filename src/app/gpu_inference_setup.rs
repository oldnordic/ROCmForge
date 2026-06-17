#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(feature = "gpu")]
use rocmforge::config::ModelConfig;
#[cfg(feature = "gpu")]
use rocmforge::cpu::{cache::CpuForwardScratch, weights::CpuModelWeights};
#[cfg(feature = "gpu")]
use rocmforge::gpu;
#[cfg(feature = "gpu")]
use rocmforge::gpu::weights::GpuModelWeights;
#[cfg(feature = "gpu")]
use rocmforge::loader::ModelFile;

#[cfg(feature = "gpu")]
use super::cli::Args;

#[cfg(feature = "gpu")]
pub(crate) struct GpuInferenceSetupState {
    pub cpu_weights: CpuModelWeights,
    pub gpu_weights: GpuModelWeights,
    pub kv: gpu::GpuKvCache,
    pub gpu_scratch: gpu::GpuForwardScratch,
    pub host_scratch: CpuForwardScratch,
    pub use_greedy: bool,
    pub use_gpu_greedy_fastpath: bool,
}

#[cfg(feature = "gpu")]
fn prepare_expert_scratch(
    gpu_scratch: &mut gpu::GpuForwardScratch,
    gpu_weights: &GpuModelWeights,
) -> Result<(), Box<dyn std::error::Error>> {
    'expert_scratch: for layer in &gpu_weights.layers {
        let all_compressed = [
            layer.ffn_gate_compressed.as_ref(),
            layer.ffn_up_compressed.as_ref(),
            layer.ffn_down_compressed.as_ref(),
        ];
        if all_compressed.iter().all(|x| x.is_some()) {
            let k = layer
                .ffn_gate_compressed
                .as_ref()
                .map(|c| c.k)
                .unwrap_or(32);
            let max_rows = all_compressed
                .iter()
                .filter_map(|x| *x)
                .map(|c| c.rows)
                .max()
                .unwrap_or(1);
            let max_cols = all_compressed
                .iter()
                .filter_map(|x| *x)
                .map(|c| c.cols)
                .max()
                .unwrap_or(1);
            let max_nnz = all_compressed
                .iter()
                .filter_map(|x| *x)
                .map(|c| c.max_nnz())
                .max()
                .unwrap_or(1);
            gpu_scratch
                .init_expert_scratch(k as u32, max_rows, max_cols, max_nnz)
                .map_err(|e| format!("expert scratch init: {}", e))?;
            eprintln!(
                "  Expert scratch: k={}, max_rows={}, max_cols={}, max_nnz={}",
                k, max_rows, max_cols, max_nnz
            );
            break 'expert_scratch;
        }

        let all_mpo = [
            layer.ffn_gate_mpo_experts.as_ref(),
            layer.ffn_up_mpo_experts.as_ref(),
            layer.ffn_down_mpo_experts.as_ref(),
        ];
        if all_mpo.iter().all(|x| x.is_some()) {
            let k = layer
                .ffn_gate_mpo_experts
                .as_ref()
                .map(|c| c.chi_max)
                .unwrap_or(32);
            let max_rows = all_mpo
                .iter()
                .filter_map(|x| *x)
                .map(|c| c.rows)
                .max()
                .unwrap_or(1);
            let max_cols = all_mpo
                .iter()
                .filter_map(|x| *x)
                .map(|c| c.cols)
                .max()
                .unwrap_or(1);
            gpu_scratch
                .init_expert_scratch(k as u32, max_rows, max_cols, 0)
                .map_err(|e| format!("expert scratch init: {}", e))?;
            eprintln!(
                "  Expert scratch (MPO): k={}, max_rows={}, max_cols={}",
                k, max_rows, max_cols
            );
            break 'expert_scratch;
        }
    }
    Ok(())
}

#[cfg(feature = "gpu")]
pub(crate) fn prepare_gpu_inference_state(
    file: &ModelFile,
    config: &ModelConfig,
    args: &Args,
    device_id: i32,
    max_seq: usize,
) -> Result<GpuInferenceSetupState, Box<dyn std::error::Error>> {
    eprint!("Loading CPU weights... ");
    let t_cpu_load = Instant::now();
    let cpu_weights = file
        .load_cpu_weights(config)
        .map_err(|e| format!("cpu weight load: {}", e))?;
    eprintln!("done in {:.1}s", t_cpu_load.elapsed().as_secs_f64());

    eprint!("Loading GPU weights... ");
    let t_gpu_load = Instant::now();
    let gpu_weights = file
        .load_gpu_weights(config, device_id)
        .map_err(|e| format!("gpu weight load: {}", e))?;
    eprintln!("done in {:.1}s", t_gpu_load.elapsed().as_secs_f64());

    let kv = gpu::GpuKvCache::new(config, max_seq).map_err(|e| format!("gpu kv: {}", e))?;
    let mut gpu_scratch =
        gpu::GpuForwardScratch::new(config).map_err(|e| format!("gpu scratch: {}", e))?;
    prepare_expert_scratch(&mut gpu_scratch, &gpu_weights)?;

    let host_scratch = CpuForwardScratch::new(config);
    let use_greedy = args.top_p >= 1.0;
    let use_gpu_greedy_fastpath = use_greedy && !args.debug;

    Ok(GpuInferenceSetupState {
        cpu_weights,
        gpu_weights,
        kv,
        gpu_scratch,
        host_scratch,
        use_greedy,
        use_gpu_greedy_fastpath,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn greedy_fastpath_requires_greedy_sampling_and_non_debug() {
        let use_greedy = true;
        let debug = false;
        let use_gpu_greedy_fastpath = use_greedy && !debug;
        assert!(use_gpu_greedy_fastpath);

        let debug = true;
        let use_gpu_greedy_fastpath = use_greedy && !debug;
        assert!(!use_gpu_greedy_fastpath);
    }
}
