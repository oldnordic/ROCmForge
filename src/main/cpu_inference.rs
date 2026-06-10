use super::cli::Args;
use super::cpu_decode::run_cpu_decode_loop;
use super::cpu_prefill::run_cpu_prefill;
use super::cpu_runtime::prepare_cpu_runtime;
use super::cpu_setup::prepare_cpu_run;

pub(crate) fn run_cpu_inference(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = prepare_cpu_runtime(args)?;
    let caps = runtime.caps;
    let run = prepare_cpu_run(args, &caps)?;
    let config = run.config;
    let tok = run.tok;
    let weights = run.weights;
    let batch_config = run.batch_config;
    let prompt_tokens = run.prompt_tokens;
    let mut kv = run.kv;
    let mut scratch = run.scratch;
    let use_greedy = run.use_greedy;

    let n_prompt = prompt_tokens.len();
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

    Ok(())
}
