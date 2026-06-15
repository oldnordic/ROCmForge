use super::cli::Args;
use super::cpu_decode::run_cpu_decode_loop;
use super::cpu_prefill::run_cpu_prefill;
use super::cpu_runtime::prepare_cpu_runtime;
use super::cpu_setup::prepare_cpu_inference_state;

#[cfg(feature = "cpu-graph")]
use super::cpu_decode::run_cpu_decode_loop_with_ctx;

#[cfg(feature = "cpu-graph")]
use rocmforge::cpu::graph::{CaptureContext, GraphMap, ScoreMetric};

#[cfg(feature = "cpu-graph")]
fn parse_score_metric(name: &str) -> ScoreMetric {
    match name.to_lowercase().as_str() {
        "cosine" | "cosine-similarity" => ScoreMetric::CosineSimilarity,
        "l2" | "l2-similarity" => ScoreMetric::L2Similarity,
        "mean" | "mean-activation" => ScoreMetric::MeanActivation,
        "cross-entropy" => ScoreMetric::CrossEntropy,
        "entropy" | "neg-entropy" => ScoreMetric::NegEntropy,
        _ => ScoreMetric::NegEntropy,
    }
}

pub(crate) fn run_cpu_inference(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = prepare_cpu_runtime(args)?;
    let caps = runtime.caps;
    let run = prepare_cpu_inference_state(args, &caps)?;
    let config = run.config;
    let tok = run.tok;
    let weights = run.weights;
    let batch_config = run.batch_config;
    let prompt_tokens = run.prompt_tokens;
    let mut kv = run.kv;
    let mut scratch = run.scratch;
    let use_greedy = run.use_greedy;

    let n_prompt = prompt_tokens.len();

    #[cfg(feature = "cpu-graph")]
    if let Some(load_dir) = &args.load_graph_map_dir {
        let map = GraphMap::load(std::path::Path::new(load_dir))?;
        eprintln!("Loaded GraphMap from {}", load_dir);
        eprintln!("  branches: {}", map.branch_scores().len());
        eprintln!("  annotations: {}", map.branch_annotations.len());
    }

    run_cpu_prefill(
        args,
        &config,
        &tok,
        &weights,
        &batch_config,
        &prompt_tokens,
        &mut kv,
        &mut scratch,
    )?;

    #[cfg(feature = "cpu-graph")]
    {
        if let Some(save_dir) = &args.graph_map_dir {
            let score_metric = parse_score_metric(&args.graph_score_metric);
            let mut ctx = CaptureContext::new(0, 0);
            run_cpu_decode_loop_with_ctx(
                args,
                &config,
                &tok,
                &weights,
                &mut kv,
                &mut scratch,
                use_greedy,
                n_prompt,
                &mut ctx,
                score_metric,
            )?;
            let map = GraphMap::from_context(&ctx);
            map.save(std::path::Path::new(save_dir))?;
            eprintln!("Saved GraphMap to {}", save_dir);
            eprintln!("  branches: {}", map.branch_scores().len());
        } else {
            run_cpu_decode_loop(
                args,
                &config,
                &tok,
                &weights,
                &mut kv,
                &mut scratch,
                use_greedy,
                n_prompt,
            )?;
        }
    }

    #[cfg(not(feature = "cpu-graph"))]
    {
        if args.graph_map_dir.is_some() || args.load_graph_map_dir.is_some() {
            return Err(
                "--graph-map-dir and --load-graph-map-dir require the cpu-graph feature".into(),
            );
        }
        run_cpu_decode_loop(
            args,
            &config,
            &tok,
            &weights,
            &mut kv,
            &mut scratch,
            use_greedy,
            n_prompt,
        )?;
    }

    Ok(())
}
