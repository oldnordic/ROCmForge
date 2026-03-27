//! Q4_K quantization type definitions

/// Number of elements per Q4_K block (from llama.cpp)
pub const QK_K: usize = 256;

/// Scales array size (from llama.cpp)
pub const K_SCALE_SIZE: usize = 12;

/// Total bytes per Q4_K block
pub const Q4_K_BLOCK_SIZE: usize = 128 + 12 + 4; // qs + scales + d/dmin

/// Rust-owned Q4_K block
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Q4KBlock {
    pub d: half::f16,              // delta/scale (2 bytes)
    pub dmin: half::f16,           // minimum scale (2 bytes)
    pub scales: [u8; 12],    // quantized scales (12 bytes)
    pub qs: [u8; 128],       // quants, 4-bit values (128 bytes)
}

impl Default for Q4KBlock {
    fn default() -> Self {
        Self {
            d: half::f16::from_f32(1.0),
            dmin: half::f16::from_f32(0.0),
            scales: [0; 12],
            qs: [0; 128],
        }
    }
}
