// Test dimension validation for edge cases

use rocmforge::cpu::ops::{dispatch_gemv, dispatch_gemm};
use rocmforge::cpu::weights::WeightMeta;
use rocmforge::loader::GgmlType;

#[test]
fn test_validate_block_size() {
    use rocmforge::cpu::quant::{validate_block_size, Q4_BLOCK_ELEMS};

    // Valid dimensions
    assert!(validate_block_size(32, Q4_BLOCK_ELEMS, "test").is_ok());
    assert!(validate_block_size(64, Q4_BLOCK_ELEMS, "test").is_ok());
    assert!(validate_block_size(256, Q4_BLOCK_ELEMS, "test").is_ok());

    // Invalid dimensions
    assert!(validate_block_size(33, Q4_BLOCK_ELEMS, "test").is_err());
    assert!(validate_block_size(100, Q4_BLOCK_ELEMS, "test").is_err());
}

#[test]
fn test_block_remainder() {
    use rocmforge::cpu::quant::{block_remainder, Q4_BLOCK_ELEMS};

    // Exact multiple
    let (blocks, remainder) = block_remainder(64, Q4_BLOCK_ELEMS);
    assert_eq!(blocks, 2);
    assert_eq!(remainder, 0);

    // Non-multiple
    let (blocks, remainder) = block_remainder(100, Q4_BLOCK_ELEMS);
    assert_eq!(blocks, 3);  // 96 elements
    assert_eq!(remainder, 4); // 4 extra elements
}

#[test]
fn test_padded_dim() {
    use rocmforge::cpu::quant::{padded_dim, Q4_BLOCK_ELEMS};

    assert_eq!(padded_dim(32, Q4_BLOCK_ELEMS), 32);
    assert_eq!(padded_dim(33, Q4_BLOCK_ELEMS), 64);
    assert_eq!(padded_dim(65, Q4_BLOCK_ELEMS), 96);
}

#[test]
fn test_dispatch_gemv_rejects_misaligned_dimensions() {
    // Q4_0 requires dimensions to be multiples of 32
    let w = vec![0u8; 100 * 18]; // out_dim=100, but we'll use smaller
    let x: Vec<f32> = vec![0.0; 33];  // 33 is NOT a multiple of 32
    let mut y = vec![0.0f32; 100];

    let meta = WeightMeta {
        wtype: GgmlType::Q4_0,
        dims: vec![33, 100],
        needs_transpose: false,
    };

    let result = dispatch_gemv(&w, &meta, &x, &mut y, 100, 33);
    assert!(result.is_err());
    if let Err(e) = result {
        let error_msg = format!("{:?}", e);
        assert!(error_msg.contains("not a multiple") || error_msg.contains("block size"));
    }
}

#[test]
fn test_dispatch_gemm_rejects_misaligned_dimensions() {
    // Q6_K requires dimensions to be multiples of 256
    let w = vec![0u8; 100 * 210];
    let x: Vec<f32> = vec![0.0; 300];  // 300 is NOT a multiple of 256
    let mut y = vec![0.0f32; 100 * 300];

    let meta = WeightMeta {
        wtype: GgmlType::Q6_K,
        dims: vec![300, 100],
        needs_transpose: false,
    };

    let result = dispatch_gemm(&w, &meta, &x, &mut y, 100, 300);
    assert!(result.is_err());
    if let Err(e) = result {
        let error_msg = format!("{:?}", e);
        assert!(error_msg.contains("not a multiple") || error_msg.contains("block size"));
    }
}

#[test]
fn test_dispatch_gemv_accepts_aligned_dimensions() {
    // Q4_0 with aligned dimensions should work
    let w = vec![0u8; 100 * 32]; // out_dim=100, in_dim=32
    let x: Vec<f32> = vec![0.0; 32];   // 32 IS a multiple of 32
    let mut y = vec![0.0f32; 100];

    let meta = WeightMeta {
        wtype: GgmlType::Q4_0,
        dims: vec![32, 100],
        needs_transpose: false,
    };

    // Should not panic
    let result = dispatch_gemv(&w, &meta, &x, &mut y, 100, 32);
    assert!(result.is_ok());
}
