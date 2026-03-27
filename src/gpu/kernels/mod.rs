//! GPU kernel wrappers organized by functionality.
//!
//! Safety-first design:
//! - All kernels validate bounds before launching
//! - All hipError_t return values checked
//! - Never panic, always return GpuError

pub mod norm;
pub mod rope;
pub mod elementwise;
pub mod attention;

pub use norm::{rms_norm, rms_norm_batched};
pub use rope::{rope, rope_batched};
pub use elementwise::{add, mul, scale, gelu, silu, add_batched, mul_batched, zero_fill};
pub use attention::{kv_write, kv_write_batched, flash_attn_decode, flash_attn_prefill};
