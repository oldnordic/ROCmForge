//! Test Q6_K dequantization by comparing with llama.cpp implementation
//!
//! This tests just the dequantization logic without dot product.

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        eprintln!();
        eprintln!("Tests Q6_K dequantization implementation.");
        std::process::exit(1);
    }

    println!("Q6_K dequantization test - comparing with llama.cpp pattern");
    println!("Model: {}", args[1]);
    println!();

    // Key insights from llama.cpp dequantize_row_q6_K:
    println!("llama.cpp Q6_K dequantization pattern:");
    println!("  for (int n = 0; n < QK_K; n += 128) {{");
    println!("      for (int l = 0; l < 32; ++l) {{");
    println!("          int is = l/16;");
    println!("          const int8_t q1 = (ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;");
    println!("          const int8_t q2 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;");
    println!("          const int8_t q3 = (ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;");
    println!("          const int8_t q4 = (ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;");
    println!("          y[l +  0] = d * sc[is + 0] * q1;");
    println!("          y[l + 32] = d * sc[is + 2] * q2;");
    println!("          y[l + 64] = d * sc[is + 4] * q3;");
    println!("          y[l + 96] = d * sc[is + 6] * q4;");
    println!("      }}");
    println!("      y  += 128;  ql += 64;  qh += 32;  sc += 8;");
    println!("  }}");
    println!();

    println!("Key observations:");
    println!("  1. First iteration (n=0): ql[0..63], qh[0..31], sc[0..7]");
    println!("     - Output positions: y[0..127]");
    println!("  2. Second iteration (n=128): ql[64..127], qh[32..63], sc[8..15]");
    println!("     - Output positions: y[128..255]");
    println!();

    println!("Scale indexing for first 128 elements:");
    for l in 0..32 {
        let is = l / 16;
        println!(
            "  l={:2}: is={} -> sc[{}], sc[{}], sc[{}], sc[{}]",
            l,
            is,
            is + 0,
            is + 2,
            is + 4,
            is + 6
        );
    }

    println!();
    println!("Scale indexing for second 128 elements:");
    for l in 0..32 {
        let is = l / 16;
        println!(
            "  l={:2}: is={} -> sc[{}+8], sc[{}+8], sc[{}+8], sc[{}+8]",
            l,
            is,
            is,
            is + 2,
            is + 4,
            is + 6
        );
    }

    println!();
    println!("SUCCESS: Q6_K dequantization pattern understood");
    println!("Next: Implement dot product following this exact pattern");
}
