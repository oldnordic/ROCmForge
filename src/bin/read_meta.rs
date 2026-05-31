use rocmforge::loader::RfmFile;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: read_meta <path_to_rfm>");
        return;
    }
    let file = RfmFile::open(&args[1]).expect("Failed to open RFM file");
    println!("Metadata:");
    println!("  Architecture: {}", file.metadata.architecture);
    println!("  num_layers: {}", file.metadata.num_layers);
    println!("  kv_lora_dim: {:?}", file.metadata.kv_lora_dim);
    println!(
        "  kv_frame_codec_enabled: {:?}",
        file.metadata.kv_frame_codec_enabled
    );
    println!(
        "  adastate_anchors_enabled: {:?}",
        file.metadata.adastate_anchors_enabled
    );

    // Count SVD tensors
    let mut svd_count = 0;
    for name in file.tensor_names() {
        if name.contains("svd") {
            svd_count += 1;
        }
    }
    println!("  SVD tensors: {}", svd_count);
}
