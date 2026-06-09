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
    pub d: half::f16,     // delta/scale (2 bytes)
    pub dmin: half::f16,  // minimum scale (2 bytes)
    pub scales: [u8; 12], // quantized scales (12 bytes)
    pub qs: [u8; 128],    // quants, 4-bit values (128 bytes)
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
    pub d: half::f16,     // delta/scale (2 bytes)
    pub dmin: half::f16,  // minimum scale (2 bytes)
    pub scales: [u8; 12], // quantized scales (12 bytes)
    pub qh: [u8; 32],     // quants, high bit (32 bytes)
    pub qs: [u8; 128],    // quants, low 4 bits (128 bytes)
}

/// Q6_K: 256 elements per block, 210 bytes (128 ql + 64 qh + 16 scales + 2 d)
///
/// Q6_K block format (from llama.cpp ggml-common.h):
/// - ql[128]: low 4-bit quantized weights (2 elements per byte)
/// - qh[64]: high 2-bit quantized weights (4 elements per byte)
/// - scales[16]: signed 8-bit scales (int8_t)
/// - d: f16 super-block scale (2 bytes, AT THE END!)
///
/// Each weight is 6 bits: low 4 bits from ql, high 2 bits from qh
/// Total: 210 bytes for 256 values (~6.56 bits per weight)
pub const Q6_K_BLOCK_SIZE: usize = 128 + 64 + 16 + 2; // ql + qh + scales + d

/// Rust-owned Q6_K block matching C layout in HIP kernels
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Q6KBlock {
    pub ql: [u8; 128],    // quants, low 4 bits
    pub qh: [u8; 64],     // quants, high 2 bits
    pub scales: [i8; 16], // signed scales (int8_t)
    pub d: half::f16,     // super-block scale
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
    pub d: half::f16, // scale (2 bytes)
    pub qs: [i8; 32], // quantized values (32 bytes)
}

impl Default for Q8_0Block {
    fn default() -> Self {
        Self {
            d: half::f16::from_f32(1.0),
            qs: [0; 32],
        }
    }
}

// Q4_0 constants (from llama.cpp)
/// Number of elements per Q4_0 block
pub const QK4_0: usize = 32;

/// Total bytes per Q4_0 block
pub const Q4_0_BLOCK_SIZE: usize = 18; // 2 (scale) + 16 (data)

/// Rust-owned Q4_0 block
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Q4_0Block {
    pub d: half::f16, // scale (2 bytes)
    pub qs: [u8; 16], // 4-bit values packed (16 bytes)
}

impl Default for Q4_0Block {
    fn default() -> Self {
        Self {
            d: half::f16::from_f32(1.0),
            qs: [0; 16],
        }
    }
}

// Q4_1 constants (from llama.cpp)
/// Number of elements per Q4_1 block
pub const QK4_1: usize = 32;

/// Total bytes per Q4_1 block
pub const Q4_1_BLOCK_SIZE: usize = 20; // 2 (scale) + 2 (min) + 16 (data)

// Q5_0 constants (from llama.cpp)
/// Number of elements per Q5_0 block
pub const QK5_0: usize = 32;

/// Total bytes per Q5_0 block
pub const Q5_0_BLOCK_SIZE: usize = 22; // 2 (scale) + 4 (qh) + 16 (qs)

/// Rust-owned Q5_0 block
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Q5_0Block {
    pub d: half::f16, // scale (2 bytes)
    pub qh: [u8; 4],  // high bits (4 bytes)
    pub qs: [u8; 16], // low 4 bits packed (16 bytes)
}

impl Default for Q5_0Block {
    fn default() -> Self {
        Self {
            d: half::f16::from_f32(1.0),
            qh: [0; 4],
            qs: [0; 16],
        }
    }
}

// Q5_1 constants (from llama.cpp)
/// Number of elements per Q5_1 block
pub const QK5_1: usize = 32;

/// Total bytes per Q5_1 block
pub const Q5_1_BLOCK_SIZE: usize = 24; // 2 (scale) + 2 (min) + 4 (qh) + 16 (qs)

/// Rust-owned Q5_1 block
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Q5_1Block {
    pub d: half::f16, // scale (2 bytes)
    pub m: half::f16, // min offset (2 bytes)
    pub qh: [u8; 4],  // high bits (4 bytes)
    pub qs: [u8; 16], // low 4 bits packed (16 bytes)
}

impl Default for Q5_1Block {
    fn default() -> Self {
        Self {
            d: half::f16::from_f32(1.0),
            m: half::f16::from_f32(0.0),
            qh: [0; 4],
            qs: [0; 16],
        }
    }
}

/// Rust-owned Q4_1 block (llama.cpp format: d + m + qs)
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Q4_1Block {
    pub d: half::f16, // scale (2 bytes)
    pub m: half::f16, // min offset (2 bytes)
    pub qs: [u8; 16], // 4-bit values packed (16 bytes)
}

impl Default for Q4_1Block {
    fn default() -> Self {
        Self {
            d: half::f16::from_f32(1.0),
            m: half::f16::from_f32(0.0),
            qs: [0; 16],
        }
    }
}
