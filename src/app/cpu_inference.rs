use super::cli::Args;
use super::cpu_decode::run_cpu_decode_loop;
use super::cpu_prefill::run_cpu_prefill;
use super::cpu_runtime::prepare_cpu_runtime;
use super::cpu_setup::prepare_cpu_inference_state;

#[cfg(feature = "cpu-graph")]
use super::cpu_decode::{run_cpu_decode_beam_loop_with_ctx, run_cpu_decode_loop_with_ctx};

#[cfg(feature = "cpu-graph")]
use rocmforge::cpu::graph::{BranchValueHead, CaptureContext, GraphMap, ScoreMetric};

pub(crate) fn run_cpu_inference(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cpu-graph")]
    if let Some(trace_dir) = &args.train_value_head_from_traces {
        let save_path = args
            .save_value_head
            .as_deref()
            .ok_or("--save-value-head is required when training a value head")?;
        let file = rocmforge::loader::ModelFile::open(&args.model)?;
        let config = file.config()?;
        let tokenizer = file.tokenizer();
        let weights = file
            .load_cpu_weights(&config)
            .map_err(|e| format!("weight load: {}", e))?;
        rocmforge::cpu::graph::train_value_head_from_trace_dir(
            std::path::Path::new(trace_dir),
            &weights,
            &config,
            &tokenizer,
            100,
            0.01,
            std::path::Path::new(save_path),
        )?;
        return Ok(());
    }

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
    let value_head = if let Some(path) = &args.value_head_path {
        Some(BranchValueHead::load(std::path::Path::new(path))?)
    } else {
        None
    };

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
        if value_head.is_some() && args.rerank_beam_width > 1 {
            let mut ctx = CaptureContext::new(0, 0);
            run_cpu_decode_beam_loop_with_ctx(
                args,
                &config,
                &tok,
                &weights,
                &kv,
                &scratch,
                n_prompt,
                &mut ctx,
                value_head
                    .as_ref()
                    .expect("beam width > 1 requires value head"),
                args.rerank_top_k,
                args.rerank_scale,
                args.rerank_beam_depth,
                args.rerank_beam_width,
            )?;
            if let Some(save_dir) = &args.graph_map_dir {
                let map = GraphMap::from_context(&ctx);
                map.save(std::path::Path::new(save_dir))?;
                eprintln!("Saved GraphMap to {}", save_dir);
                eprintln!("  branches: {}", map.branch_scores().len());
            }
        } else if let Some(save_dir) = &args.graph_map_dir {
            let score_metric = ScoreMetric::from_name(&args.graph_score_metric);
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
                value_head.as_ref(),
                args.rerank_top_k,
                args.rerank_scale,
                args.rerank_beam_depth,
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
        if args.graph_map_dir.is_some()
            || args.load_graph_map_dir.is_some()
            || args.value_head_path.is_some()
            || args.train_value_head_from_traces.is_some()
        {
            return Err(
                "--graph-map-dir, --load-graph-map-dir, --value-head-path, and \
                 --train-value-head-from-traces require the cpu-graph feature"
                    .into(),
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
