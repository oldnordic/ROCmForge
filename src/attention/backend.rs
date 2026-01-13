//! Backend abstraction for attention computation

/// Backend types for attention computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionBackend {
    /// CPU backend using standard Rust computation
    #[default]
    Cpu,
    /// GPU backend using ROCm/HIP acceleration
    #[cfg(feature = "rocm")]
    Gpu,
}
