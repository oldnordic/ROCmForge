//! RoPE positional embeddings.

// ── RoPE positional embeddings ───────────────────────────────────────────────────

/// Apply RoPE (Rotary Position Embeddings) to a vector.
///
/// Input shape: [num_heads, head_dim]
/// Position: current token position in sequence
/// freq: precomputed frequencies, length = head_dim/2,
///       where freq[i] = 1/theta^(2i/head_dim)
///
/// # Modes
/// - `neox = true`: GPT-NeoX style (pairs within same half) - Qwen2, GPT-NeoX
/// - `neox = false`: Classic RoPE (consecutive pairs) - LLaMA, Mistral
pub fn rope(x: &mut [f32], num_heads: usize, head_dim: usize, pos: usize, freq: &[f32], neox: bool) {
    let total_len = num_heads * head_dim;
    debug_assert_eq!(x.len(), total_len, "rope input dimension mismatch");
    debug_assert_eq!(freq.len(), head_dim / 2, "rope freq length mismatch");

    for h in 0..num_heads {
        let base = h * head_dim;
        let half = head_dim / 2;

        for i in 0..half {
            let angle = pos as f32 * freq[i];
            let (sin_a, cos_a) = angle.sin_cos();

            if neox {
                // NeoX: pairs are (i, i+half)
                let x0 = x[base + i];
                let x1 = x[base + i + half];
                x[base + i] = x0 * cos_a - x1 * sin_a;
                x[base + i + half] = x0 * sin_a + x1 * cos_a;
            } else {
                // Classic: pairs are (2i, 2i+1)
                let x0 = x[base + 2 * i];
                let x1 = x[base + 2 * i + 1];
                x[base + 2 * i] = x0 * cos_a - x1 * sin_a;
                x[base + 2 * i + 1] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

/// Batched RoPE: apply to each row with position start_pos + s
pub fn rope_batch(
    x: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    start_pos: usize,
    freq: &[f32],
    neox: bool,
) {
    let row_len = num_heads * head_dim;
    let seq_len = x.len() / row_len;

    for s in 0..seq_len {
        let row = &mut x[s * row_len..(s + 1) * row_len];
        rope(row, num_heads, head_dim, start_pos + s, freq, neox);
    }
}
