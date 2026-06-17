pub(crate) struct Args {
    pub model: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub no_template: bool,
    pub list_tensors: bool,
    pub debug: bool,
    pub gpu: bool,
    pub prefill_only_validate: bool,
    pub draft_model: Option<String>,
    pub speculative_tokens: usize,
    pub kv_dump: Option<String>,
    pub server: bool,
    pub port: u16,
    pub threads: Option<usize>,
    pub ctx_size: Option<usize>,
    pub graph_map_dir: Option<String>,
    pub load_graph_map_dir: Option<String>,
    pub graph_score_metric: String,
    pub value_head_path: Option<String>,
    pub rerank_top_k: usize,
    pub rerank_scale: f32,
    pub rerank_beam_depth: usize,
    pub rerank_beam_width: usize,
    pub rerank_beam_length_penalty: f32,
    pub train_value_head_from_traces: Option<String>,
    pub save_value_head: Option<String>,
    pub forward_graph_trace: Option<String>,
    pub expected_attention: Option<String>,
}

fn usage() -> ! {
    eprintln!("rocmforge - AMD-first LLM inference engine");
    eprintln!();
    eprintln!("Usage: rocmforge --model <path> --prompt <text> [OPTIONS]");
    eprintln!("       rocmforge --model <path> --server [--port N]");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --model <path>         Path to GGUF model file");
    eprintln!();
    eprintln!("Generation mode:");
    eprintln!("  --prompt <text>        Input prompt");
    eprintln!("  --max-tokens N         Maximum tokens to generate [default: 256]");
    eprintln!("  --temperature F        Sampling temperature [default: 1.0]");
    eprintln!("  --top-p F              Nucleus sampling threshold [default: 0.9]");
    eprintln!("  --no-template          Disable chat template");
    eprintln!("  --list-tensors         List tensors in model file and exit");
    eprintln!("  --debug                Show debug info (top logits, etc.)");
    eprintln!("  --gpu                  Use GPU backend (requires ROCm/HIP)");
    eprintln!("  --threads N, -t N      Number of CPU threads/cores to use [default: auto-detect]");
    eprintln!(
        "  --ctx-size N, -c N     Override maximum context window size [default: model default]"
    );
    eprintln!(
        "  --graph-map-dir <path> Save session GraphMap to directory (requires cpu-graph feature; GPU capture records token-level trace)"
    );
    eprintln!("  --load-graph-map-dir <path> Load a previous session GraphMap from directory");
    eprintln!(
        "  --graph-score-metric <name> Score metric for captured branches [default: neg-entropy]"
    );
    eprintln!(
        "  --value-head-path <path>  Load a trained BranchValueHead for token-level reranking"
    );
    eprintln!(
        "  --rerank-top-k <N>        Number of top candidates to score with the value head [default: 3]"
    );
    eprintln!(
        "  --rerank-scale <F>        Scale factor applied to value-head scores before biasing logits [default: 1.0]"
    );
    eprintln!(
        "  --rerank-beam-depth <D>   Number of tokens to lookahead when scoring each candidate [default: 1]"
    );
    eprintln!(
        "  --rerank-beam-width <B>   Number of hypotheses to keep alive across steps [default: 1]"
    );
    eprintln!(
        "  --rerank-beam-length-penalty <F>  Length normalization exponent for beam pruning [default: 1.0]"
    );
    eprintln!(
        "  --train-value-head-from-traces <dir> Train a BranchValueHead from persisted GraphMap traces"
    );
    eprintln!("  --save-value-head <path>  Destination file for --train-value-head-from-traces");
    eprintln!("  --prefill-only-validate Run prefill only, exit with validation status");
    eprintln!("  --kv-dump <path>       Dump post-prefill KV cache to binary file");
    eprintln!(
        "  --draft-model <path>   Path to draft GGUF/RFM model file for speculative decoding"
    );
    eprintln!("  --speculative-tokens N Number of draft tokens to speculate per step [default: 4]");
    eprintln!(
        "  --forward-graph-trace <path>  Write a JSONL forward-graph trace for visualization"
    );
    eprintln!(
        "  --expected-attention <json>   Optional meta mapping query pos -> expected key positions"
    );
    eprintln!();
    eprintln!("Server mode:");
    eprintln!("  --server               Start OpenAI-compatible HTTP API server");
    eprintln!("  --port N               Port to bind [default: 8080]");
    eprintln!();
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  rocmforge --model qwen2.5-7b.gguf --prompt \"Hello, world!\"");
    std::process::exit(1);
}

pub(crate) fn parse_args() -> Args {
    parse_args_from(std::env::args().skip(1))
}

fn parse_args_from(args: impl IntoIterator<Item = String>) -> Args {
    let mut args = args.into_iter();
    let mut model = None;
    let mut prompt = None;
    let mut max_tokens = 256usize;
    let mut temperature = 1.0f32;
    let mut top_p = 0.9f32;
    let mut no_template = false;
    let mut list_tensors = false;
    let mut debug = false;
    let mut gpu = false;
    let mut server = false;
    let mut port = 8080u16;
    let mut prefill_only_validate = false;
    let mut draft_model = None;
    let mut speculative_tokens = 4usize;
    let mut kv_dump: Option<String> = None;
    let mut threads = None;
    let mut ctx_size = None;
    let mut graph_map_dir: Option<String> = None;
    let mut load_graph_map_dir: Option<String> = None;
    let mut graph_score_metric = "neg-entropy".to_string();
    let mut value_head_path: Option<String> = None;
    let mut rerank_top_k = 3usize;
    let mut rerank_scale = 1.0f32;
    let mut rerank_beam_depth = 1usize;
    let mut rerank_beam_width = 1usize;
    let mut rerank_beam_length_penalty = 1.0f32;
    let mut train_value_head_from_traces: Option<String> = None;
    let mut save_value_head: Option<String> = None;
    let mut forward_graph_trace: Option<String> = None;
    let mut expected_attention: Option<String> = None;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "-m" | "--model" => model = Some(args.next().unwrap_or_else(|| usage())),
            "-p" | "--prompt" => prompt = Some(args.next().unwrap_or_else(|| usage())),
            "--max-tokens" => {
                max_tokens = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--temp" | "--temperature" => {
                temperature = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--top-p" => {
                top_p = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--no-template" => no_template = true,
            "--list-tensors" => list_tensors = true,
            "--debug" => debug = true,
            "--gpu" => gpu = true,
            "--server" => server = true,
            "--port" => {
                port = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "-t" | "--threads" => {
                threads = Some(
                    args.next()
                        .unwrap_or_else(|| usage())
                        .parse()
                        .unwrap_or_else(|_| usage()),
                )
            }
            "-c" | "--ctx-size" => {
                ctx_size = Some(
                    args.next()
                        .unwrap_or_else(|| usage())
                        .parse()
                        .unwrap_or_else(|_| usage()),
                )
            }
            "--prefill-only-validate" => prefill_only_validate = true,
            "--graph-map-dir" => graph_map_dir = Some(args.next().unwrap_or_else(|| usage())),
            "--load-graph-map-dir" => {
                load_graph_map_dir = Some(args.next().unwrap_or_else(|| usage()))
            }
            "--graph-score-metric" => graph_score_metric = args.next().unwrap_or_else(|| usage()),
            "--value-head-path" => value_head_path = Some(args.next().unwrap_or_else(|| usage())),
            "--rerank-top-k" => {
                rerank_top_k = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage());
                if rerank_top_k == 0 {
                    eprintln!("--rerank-top-k must be > 0");
                    usage();
                }
            }
            "--rerank-scale" => {
                rerank_scale = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage());
            }
            "--rerank-beam-depth" => {
                rerank_beam_depth = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage());
                if rerank_beam_depth == 0 {
                    rerank_beam_depth = 1;
                }
            }
            "--rerank-beam-width" => {
                rerank_beam_width = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage());
                if rerank_beam_width == 0 {
                    rerank_beam_width = 1;
                }
            }
            "--rerank-beam-length-penalty" => {
                rerank_beam_length_penalty = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage());
            }
            "--train-value-head-from-traces" => {
                train_value_head_from_traces = Some(args.next().unwrap_or_else(|| usage()))
            }
            "--save-value-head" => save_value_head = Some(args.next().unwrap_or_else(|| usage())),
            "--kv-dump" => kv_dump = Some(args.next().unwrap_or_else(|| usage())),
            "--draft-model" => draft_model = Some(args.next().unwrap_or_else(|| usage())),
            "--speculative-tokens" => {
                speculative_tokens = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--forward-graph-trace" => {
                forward_graph_trace = Some(args.next().unwrap_or_else(|| usage()))
            }
            "--expected-attention" => {
                expected_attention = Some(args.next().unwrap_or_else(|| usage()))
            }
            "-h" | "--help" => usage(),
            other => {
                eprintln!("Unknown flag: {}", other);
                usage();
            }
        }
    }

    if rerank_beam_width > 1 && value_head_path.is_none() {
        eprintln!("--rerank-beam-width > 1 requires --value-head-path");
        usage();
    }

    let result = Args {
        model: model.unwrap_or_else(|| usage()),
        prompt: prompt.unwrap_or_default(),
        max_tokens,
        temperature,
        top_p,
        no_template,
        list_tensors,
        debug,
        gpu,
        prefill_only_validate,
        draft_model,
        speculative_tokens,
        kv_dump,
        server,
        port,
        threads,
        ctx_size,
        graph_map_dir,
        load_graph_map_dir,
        graph_score_metric,
        value_head_path,
        rerank_top_k,
        rerank_scale,
        rerank_beam_depth,
        rerank_beam_width,
        rerank_beam_length_penalty,
        train_value_head_from_traces,
        save_value_head,
        forward_graph_trace,
        expected_attention,
    };

    // Touch fields used only in gpu/server code paths to keep clippy happy
    // in default (cpu-only) builds. The values are already parsed and stored.
    #[cfg(not(any(feature = "gpu", feature = "server")))]
    {
        let _ = result.prefill_only_validate;
        let _ = result.draft_model;
        let _ = result.speculative_tokens;
        let _ = result.kv_dump;
    }
    #[cfg(not(feature = "cpu-graph"))]
    {
        let _ = result.graph_map_dir;
        let _ = result.load_graph_map_dir;
        let _ = result.graph_score_metric;
        let _ = result.value_head_path;
        let _ = result.rerank_top_k;
        let _ = result.rerank_scale;
        let _ = result.rerank_beam_depth;
        let _ = result.rerank_beam_width;
        let _ = result.rerank_beam_length_penalty;
        let _ = result.train_value_head_from_traces;
        let _ = result.save_value_head;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::parse_args_from;

    #[test]
    fn parse_args_reads_generation_flags() {
        let args = parse_args_from([
            "--model".to_string(),
            "model.gguf".to_string(),
            "--prompt".to_string(),
            "hi".to_string(),
            "--max-tokens".to_string(),
            "32".to_string(),
            "--temperature".to_string(),
            "0.7".to_string(),
            "--top-p".to_string(),
            "0.8".to_string(),
            "--threads".to_string(),
            "12".to_string(),
        ]);

        assert_eq!(args.model, "model.gguf");
        assert_eq!(args.prompt, "hi");
        assert_eq!(args.max_tokens, 32);
        assert_eq!(args.temperature, 0.7);
        assert_eq!(args.top_p, 0.8);
        assert_eq!(args.threads, Some(12));
    }

    #[test]
    fn parse_args_reads_server_and_optional_flags() {
        let args = parse_args_from([
            "--model".to_string(),
            "model.rfm".to_string(),
            "--server".to_string(),
            "--port".to_string(),
            "9090".to_string(),
            "--gpu".to_string(),
            "--ctx-size".to_string(),
            "4096".to_string(),
            "--draft-model".to_string(),
            "draft.gguf".to_string(),
            "--speculative-tokens".to_string(),
            "6".to_string(),
        ]);

        assert_eq!(args.model, "model.rfm");
        assert!(args.server);
        assert!(args.gpu);
        assert_eq!(args.port, 9090);
        assert_eq!(args.ctx_size, Some(4096));
        assert_eq!(args.draft_model.as_deref(), Some("draft.gguf"));
        assert_eq!(args.speculative_tokens, 6);
    }
}
