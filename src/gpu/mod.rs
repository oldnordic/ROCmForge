//! AMD GPU inference backend with HIP.
//!
//! Safety-first design:
//! - All HIP API calls wrapped with error handling
//! - Never panic, always return GpuError
//! - CPU fallback when GPU unavailable
//!
//! ## Usage
//!
//! ```no_run
//! use rocmforge::gpu::{detect, GpuDevice};
//!
//! // Detect GPU
//! let caps = match detect() {
//!     Some(gpu) => {
//!         println!("Found: {} ({} GB VRAM)", gpu.device_name, gpu.total_vram_gb());
//!         gpu
//!     }
//!     None => {
//!         println!("No GPU, using CPU");
//!         return;
//!     }
//! };
//!
//! // Check if model fits
//! if !caps.can_fit_model(model_size) {
//!     println!("Model too large for GPU, using CPU");
//!     return;
//! }
//!
//! // Initialize device
//! let device = GpuDevice::init(caps.device_id)?;
//! // ... run inference ...
//! # Ok::<(), rocmforge::gpu::GpuError>(())
//! ```

mod error;
mod ffi;
mod detect;
mod device;
mod weights;
mod kernels;
mod cache;
mod dynamic_loader;

pub use error::{GpuError, GpuResult};
pub use detect::GpuCapabilities;
pub use device::GpuDevice;
pub use weights::{WeightMeta, GpuBuffer, GpuLayerWeights, GpuModelWeights};
pub use kernels::{kv_write, kv_write_batched, rms_norm, rms_norm_batched, rope, rope_batched, add, mul, scale, gelu, silu, add_batched, mul_batched, flash_attn_decode, flash_attn_prefill};
pub use cache::{GpuKvCache, GpuForwardScratch};
pub use dynamic_loader::{DynamicLibrary, library_info, LibraryInfo};

/// Detect AMD GPU capabilities (safe wrapper).
///
/// Returns None if HIP unavailable or no GPU found.
/// Never panics.
///
/// # Example
///
/// ```no_run
/// use rocmforge::gpu::detect;
///
/// match detect() {
///     Some(gpu) => println!("Found: {}", gpu.device_name),
///     None => println!("No GPU detected"),
/// }
/// ```
pub fn detect() -> Option<GpuCapabilities> {
    GpuCapabilities::detect()
}
