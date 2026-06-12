pub struct ConvertOptions {
    pub svd_k: Option<u32>,
    pub sparse_threshold: Option<f32>,
    pub residual_prune_threshold: Option<f32>,
    pub mpo_chi_max: Option<u32>,
    pub max_layers: Option<u32>,
    pub use_fwht: bool,
    pub mq4: bool,
    pub mq6: bool,
    pub force_gpu: bool,
    pub force_cpu: bool,
    pub kv_lora_dim: Option<usize>,
    pub kv_frame_codec: bool,
    pub adastate_anchors: bool,
    pub kv_quant_bits: Option<usize>,
    pub qjl_scale: Option<f32>,
    pub svd_attn_only: bool,
    pub input_path: String,
    pub output_path: String,
}

const USAGE: &str = concat!(
    "Usage: rocmforge-convert <input_gguf> <output_rfm>\n",
    "  [--svd-k <K>]                      SVD rank for low-rank correction\n",
    "  [--sparse-threshold <T>]            Combined with --svd-k: store sparse residual\n",
    "                                      when residual nnz ratio < T (0..1)\n",
    "  [--residual-prune-threshold <M>]    Combined with --svd-k+--sparse-threshold:\n",
    "                                      zero residual elements |r| < M before CSR\n",
    "  [--use-fwht]                        Apply Fast Walsh-Hadamard Transform before SVD\n",
    "  [--mpo-chi-max <C>]                 MPO bond dimension for FFN compression\n",
    "  [--max-layers <L>]                  Only convert first L layers (smoke testing)\n",
    "  [--gpu]                             Force GPU SVD (requires rocsolver & --features gpu)\n",
    "  [--cpu]                             Force CPU SVD (use power-iteration, slow)\n",
    "  [--kv-lora-dim <D>]                 Set latent KV cache compression dimension\n",
    "  [--kv-quant-bits <B>]               Set KV cache quantization bits (e.g. 3 for TurboQuant)\n",
    "  [--kv-frame-codec]                  Enable differential KV cache compression\n",
    "  [--svd-attn-only]                   Only apply SVD to attention projections (Q, K, V, O)\n",
    "  [--adastate-anchors]                Enable AdaState self-evolving dynamic anchors",
);

pub fn parse_args(args: impl IntoIterator<Item = String>) -> ConvertOptions {
    let args: Vec<String> = args.into_iter().collect();
    let mut options = ConvertOptions {
        svd_k: None,
        sparse_threshold: None,
        residual_prune_threshold: None,
        mpo_chi_max: None,
        max_layers: None,
        use_fwht: false,
        mq4: false,
        mq6: false,
        force_gpu: false,
        force_cpu: false,
        kv_lora_dim: None,
        kv_frame_codec: false,
        adastate_anchors: false,
        kv_quant_bits: None,
        qjl_scale: None,
        svd_attn_only: false,
        input_path: String::new(),
        output_path: String::new(),
    };

    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "--svd-k" => {
                options.svd_k = Some(parse_value(&args, idx, "--svd-k", "Invalid SVD rank k"));
                idx += 2;
            }
            "--sparse-threshold" => {
                options.sparse_threshold = Some(parse_value(
                    &args,
                    idx,
                    "--sparse-threshold",
                    "Invalid sparse threshold",
                ));
                idx += 2;
            }
            "--residual-prune-threshold" => {
                options.residual_prune_threshold = Some(parse_value(
                    &args,
                    idx,
                    "--residual-prune-threshold",
                    "Invalid residual prune threshold",
                ));
                idx += 2;
            }
            "--mpo-chi-max" => {
                options.mpo_chi_max = Some(parse_value(
                    &args,
                    idx,
                    "--mpo-chi-max",
                    "Invalid MPO chi max",
                ));
                idx += 2;
            }
            "--max-layers" => {
                options.max_layers = Some(parse_value(
                    &args,
                    idx,
                    "--max-layers",
                    "Invalid max layers",
                ));
                idx += 2;
            }
            "--use-fwht" => {
                options.use_fwht = true;
                idx += 1;
            }
            "--mq4" => {
                options.mq4 = true;
                idx += 1;
            }
            "--mq6" => {
                options.mq6 = true;
                idx += 1;
            }
            "--gpu" => {
                options.force_gpu = true;
                idx += 1;
            }
            "--cpu" => {
                options.force_cpu = true;
                idx += 1;
            }
            "--kv-lora-dim" => {
                let dim: usize = parse_value(&args, idx, "--kv-lora-dim", "Invalid KV LoRA dim");
                let padded = dim.next_power_of_two();
                if padded != dim {
                    println!(
                        "💡 Model Converter: Padding --kv-lora-dim from {} to {} to satisfy Walsh-Hadamard power-of-two constraint.",
                        dim, padded
                    );
                }
                options.kv_lora_dim = Some(padded);
                idx += 2;
            }
            "--kv-quant-bits" => {
                let bits: usize =
                    parse_value(&args, idx, "--kv-quant-bits", "Invalid KV quant bits");
                if bits < 1 || bits > 4 {
                    eprintln!(
                        "Error: --kv-quant-bits must be 1, 2, 3, or 4 (got {})",
                        bits
                    );
                    std::process::exit(1);
                }
                options.kv_quant_bits = Some(bits);
                idx += 2;
            }
            "--qjl-scale" => {
                options.qjl_scale =
                    Some(parse_value(&args, idx, "--qjl-scale", "Invalid QJL scale"));
                idx += 2;
            }
            "--kv-frame-codec" => {
                options.kv_frame_codec = true;
                idx += 1;
            }
            "--adastate-anchors" => {
                options.adastate_anchors = true;
                idx += 1;
            }
            "--svd-attn-only" => {
                options.svd_attn_only = true;
                idx += 1;
            }
            _ => {
                if options.input_path.is_empty() {
                    options.input_path = args[idx].clone();
                } else if options.output_path.is_empty() {
                    options.output_path = args[idx].clone();
                }
                idx += 1;
            }
        }
    }

    if options.input_path.is_empty() || options.output_path.is_empty() {
        eprintln!("{USAGE}");
        std::process::exit(1);
    }

    if options.force_gpu && options.force_cpu {
        eprintln!("Error: Cannot specify both --gpu and --cpu");
        std::process::exit(1);
    }

    options
}

fn parse_value<T>(args: &[String], idx: usize, flag: &str, invalid_message: &str) -> T
where
    T: std::str::FromStr,
{
    if idx + 1 >= args.len() {
        eprintln!("Error: {flag} requires a value");
        std::process::exit(1);
    }
    args[idx + 1].parse().unwrap_or_else(|_| {
        eprintln!("Error: {invalid_message}");
        std::process::exit(1);
    })
}

#[cfg(test)]
mod tests {
    use super::parse_args;

    #[test]
    fn parse_args_pads_kv_lora_dim_to_power_of_two() {
        let options = parse_args([
            "rocmforge-convert".to_string(),
            "--kv-lora-dim".to_string(),
            "96".to_string(),
            "in.gguf".to_string(),
            "out.rfm".to_string(),
        ]);
        assert_eq!(options.kv_lora_dim, Some(128));
        assert_eq!(options.input_path, "in.gguf");
        assert_eq!(options.output_path, "out.rfm");
    }

    #[test]
    fn parse_args_reads_basic_flags_and_paths() {
        let options = parse_args([
            "rocmforge-convert".to_string(),
            "--svd-k".to_string(),
            "16".to_string(),
            "--mq4".to_string(),
            "--cpu".to_string(),
            "model.gguf".to_string(),
            "model.rfm".to_string(),
        ]);
        assert_eq!(options.svd_k, Some(16));
        assert!(options.mq4);
        assert!(options.force_cpu);
        assert_eq!(options.input_path, "model.gguf");
        assert_eq!(options.output_path, "model.rfm");
    }
}
