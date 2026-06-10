use rocmforge::tokenizer::BpeTokenizer;

use super::debug::print_top_k_tokens;

#[derive(Debug, Clone, PartialEq)]
struct SliceStats {
    mean: f32,
    std: f32,
    min: f32,
    max: f32,
}

fn slice_stats(values: &[f32]) -> SliceStats {
    let mean: f32 = values.iter().copied().sum::<f32>() / values.len() as f32;
    let std: f32 =
        ((values.iter().map(|x| x * x).sum::<f32>() / values.len() as f32) - mean * mean).sqrt();
    let min: f32 = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max: f32 = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    SliceStats {
        mean,
        std,
        min,
        max,
    }
}

pub(crate) fn print_cpu_hardware_summary(
    physical_cores: usize,
    logical_cpus: usize,
    simd_description: &str,
    l3_cache_mb: Option<f64>,
    total_memory_gb: f64,
    kernel_description: &str,
) {
    eprintln!("done");
    eprintln!("  Physical cores: {}", physical_cores);
    eprintln!("  Logical CPUs: {}", logical_cpus);
    eprintln!("  SIMD features: {}", simd_description);
    if let Some(l3) = l3_cache_mb {
        eprintln!("  L3 cache: {:.1} MB", l3);
    } else {
        eprintln!("  L3 cache: undetectable (using fallback)");
    }
    eprintln!("  Total memory: {:.1} GB", total_memory_gb);
    eprintln!("  Kernel preference: {}", kernel_description);
}

pub(crate) fn print_batch_config(max_tokens_per_batch: usize, num_cores: usize) {
    eprintln!(
        "Batch config: max {} tokens/batch, use {} cores",
        max_tokens_per_batch, num_cores
    );
}

pub(crate) fn print_prompt_summary(template_name: &str, prompt_tokens_len: usize) {
    eprintln!("Chat template: {}", template_name);
    eprintln!("Prompt: {} tokens", prompt_tokens_len);
}

pub(crate) fn print_prefill_debug(first_tok: u32, hidden: &[f32]) {
    let stats = slice_stats(hidden);
    eprintln!(
        "[Prefill] first token {} embedding: mean={:.4} std={:.4}",
        first_tok, stats.mean, stats.std
    );
    eprintln!("  hidden[0..5]: {:?}", &hidden[0..5]);
}

pub(crate) fn print_prefill_stats(prefill_ms: f64, n_prompt: usize) {
    eprintln!(
        "Prefill: {:.1}ms ({:.1} tok/s)",
        prefill_ms,
        n_prompt as f64 / prefill_ms * 1000.0
    );
}

pub(crate) fn print_decode_token_debug(next_token: u32, text: &str) {
    eprintln!("[Generated] token_id={} text={:?}", next_token, text);
}

pub(crate) fn print_hidden_stats(n_generated: usize, next_token: u32, hidden: &[f32]) {
    let stats = slice_stats(hidden);
    eprintln!(
        "\n[Token {} embed] id={} mean={:.4} std={:.4} range=[{:.4}, {:.4}]",
        n_generated, next_token, stats.mean, stats.std, stats.min, stats.max
    );
    eprintln!("  hidden[0..5]: {:?}", &hidden[0..5]);
}

pub(crate) fn print_logits_stats(n_generated: usize, logits: &[f32]) {
    let stats = slice_stats(logits);
    eprintln!(
        "[Token {} logits] mean={:.4} std={:.4} range=[{:.4}, {:.4}]",
        n_generated, stats.mean, stats.std, stats.min, stats.max
    );
}

pub(crate) fn print_top_logits_debug(
    n_generated: usize,
    logits: &[f32],
    tok: &BpeTokenizer,
    k: usize,
) {
    eprintln!("\n[Token {} logits]", n_generated);
    print_top_k_tokens(logits, tok, k);
}

pub(crate) fn print_generation_stats(n_generated: usize, gen_ms: f64) {
    eprintln!(
        "\n{} tokens in {:.1}ms = {:.1} tok/s",
        n_generated,
        gen_ms,
        n_generated as f64 / gen_ms * 1000.0
    );
}

pub(crate) fn print_eos_stats() {
    eprintln!("\n[EOS on first token]");
}

#[cfg(test)]
mod tests {
    use super::slice_stats;

    #[test]
    fn slice_stats_reports_expected_summary() {
        let stats = slice_stats(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(stats.mean, 2.5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 4.0);
        assert!((stats.std - 1.118034).abs() < 1e-5);
    }
}
