//! Q4_K quantization type definitions

/// Number of elements per Q4_K block (from llama.cpp)
pub const QK_K: usize = 256;

/// Scales array size (from llama.cpp)
pub const K_SCALE_SIZE: usize = 12;

/// Total bytes per Q4_K block
pub const Q4_K_BLOCK_SIZE: usize = 128 + 12 + 4; // qs + scales + d/dmin

// Q8_0 constants (from llama.cpp)
/// Number of elements per Q8_0 block
pub const QK8_0: usize = 32;

/// Total bytes per Q8_0 block
pub const Q8_0_BLOCK_SIZE: usize = 34; // 2 (scale) + 32 (data)

/// Maximum quantized value for Q8_0
pub const Q8_0_MAX: f32 = 127.0;

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

/// Total bytes per Q5_K block (176 bytes, not 196 - llama.cpp static_assert confirms)
pub const Q5_K_BLOCK_SIZE: usize = 2 + 2 + 12 + 32 + 128; // d + dmin + scales + qh + qs

/// Rust-owned Q5_K block
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Q5KBlock {
    pub d: half::f16,              // delta/scale (2 bytes)
    pub dmin: half::f16,           // minimum scale (2 bytes)
    pub scales: [u8; 12],          // quantized scales (12 bytes)
    pub qh: [u8; 32],              // quants, high bit (32 bytes)
    pub qs: [u8; 128],             // quants, low 4 bits (128 bytes)
}

impl Default for Q5KBlock {
    fn default() -> Self {
        Self {
            d: half::f16::from_f32(1.0),
            dmin: half::f16::from_f32(0.0),
            scales: [0; 12],
            qh: [0; 32],
            qs: [0; 128],
        }
    }
}

/// Rust-owned Q8_0 block
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Q8_0Block {
    pub d: half::f16,        // scale (2 bytes)
    pub qs: [i8; 32],       // quantized values (32 bytes)
}

impl Default for Q8_0Block {
    fn default() -> Self {
        Self {
            d: half::f16::from_f32(1.0),
            qs: [0; 32],
        }
    }
}
