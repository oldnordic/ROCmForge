//! GPU Kernel Tuning Configuration
//!
//! This module provides configurable tuning parameters for HIP kernels
//! optimized for different AMD GPU architectures (RDNA2, RDNA3, CDNA).
//!
//! # Architecture Detection
//!
//! Tuning parameters are selected based on the detected GPU architecture:
//! - **RDNA3** (gfx1100+): Wave32, 256 threads/block, optimized LDS usage
//! - **RDNA2** (gfx1030+): Wave32, 256 threads/block
//! - **CDNA2** (gfx90a): Wave64, 256 or 512 threads/block
//!
//! # Tuning Parameters
//!
//! ```rust,ignore
//! use rocmforge::ggml::hip_backend::tuning::{GpuArchitecture, KernelTuning};
//!
//! let arch = GpuArchitecture::from_gfx_ip("gfx1100");
//! let config = arch.get_tuning();
//! assert_eq!(config.block_size, 256);
//! assert_eq!(config.wave_size, 32);  // RDNA3 wave32
//! ```

use std::env;

/// GPU architecture identifiers for AMD GPUs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuArchitecture {
    /// RDNA3 architecture (gfx1100, gfx1101, gfx1102)
    /// - Radeon RX 7000 series
    /// - Wave32 execution
    /// - 256 KB LDS per CU
    Rdna3,
    /// RDNA2 architecture (gfx1030, gfx1031, gfx1032, gfx1034, gfx1035)
    /// - Radeon RX 6000 series
    /// - Wave32 execution
    /// - 128 KB LDS per CU
    Rdna2,
    /// CDNA2 architecture (gfx90a)
    /// - Instinct MI200 series
    /// - Wave64 execution
    /// - 128 KB LDS per CU
    Cdna2,
    /// CDNA3 architecture (gfx940, gfx941, gfx942)
    /// - Instinct MI300 series
    /// - Wave64 execution
    /// - Large LDS
    Cdna3,
    /// Unknown/fallback architecture
    Unknown,
}

impl GpuArchitecture {
    /// Parse architecture from HIP device name or GFX IP
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use rocmforge::ggml::hip_backend::tuning::GpuArchitecture;
    ///
    /// assert_eq!(GpuArchitecture::from_gfx_ip("gfx1100"), GpuArchitecture::Rdna3);
    /// assert_eq!(GpuArchitecture::from_gfx_ip("gfx1030"), GpuArchitecture::Rdna2);
    /// assert_eq!(GpuArchitecture::from_gfx_ip("gfx90a"), GpuArchitecture::Cdna2);
    /// ```
    pub fn from_gfx_ip(gfx_ip: &str) -> Self {
        match gfx_ip {
            // RDNA3 (gfx1100 series)
            ip if ip.starts_with("gfx110") => GpuArchitecture::Rdna3,
            // RDNA2 (gfx1030 series)
            ip if ip.starts_with("gfx103") => GpuArchitecture::Rdna2,
            // CDNA3 (gfx940 series)
            ip if ip.starts_with("gfx94") => GpuArchitecture::Cdna3,
            // CDNA2 (gfx90a)
            "gfx90a" => GpuArchitecture::Cdna2,
            // Other CDNA (gfx900 series)
            ip if ip.starts_with("gfx90") => GpuArchitecture::Cdna2,
            _ => GpuArchitecture::Unknown,
        }
    }

    /// Get the wavefront size (threads per wave) for this architecture
    ///
    /// - RDNA2/RDNA3: Wave32 (32 threads)
    /// - CDNA2/CDNA3: Wave64 (64 threads)
    pub fn wave_size(&self) -> u32 {
        match self {
            GpuArchitecture::Rdna2 | GpuArchitecture::Rdna3 => 32,
            GpuArchitecture::Cdna2 | GpuArchitecture::Cdna3 => 64,
            GpuArchitecture::Unknown => 32, // Default to wave32
        }
    }

    /// Get the recommended block size for matrix multiplication kernels
    ///
    /// Larger blocks can improve occupancy but use more registers/LDS
    pub fn recommended_matmul_block_size(&self) -> u32 {
        match self {
            GpuArchitecture::Rdna3 => 256,
            GpuArchitecture::Rdna2 => 256,
            GpuArchitecture::Cdna2 => 512,
            GpuArchitecture::Cdna3 => 512,
            GpuArchitecture::Unknown => 256,
        }
    }

    /// Get the LDS size per compute unit in bytes
    ///
    /// Useful for determining how much shared memory to use
    pub fn lds_size_per_cu(&self) -> u32 {
        match self {
            GpuArchitecture::Rdna3 => 256 * 1024, // 256 KB
            GpuArchitecture::Rdna2 => 128 * 1024, // 128 KB
            GpuArchitecture::Cdna2 => 128 * 1024, // 128 KB
            GpuArchitecture::Cdna3 => 256 * 1024, // 256 KB (estimated)
            GpuArchitecture::Unknown => 128 * 1024,
        }
    }

    /// Get the maximum number of waves per compute unit
    ///
    /// This affects occupancy calculations
    pub fn max_waves_per_cu(&self) -> u32 {
        match self {
            GpuArchitecture::Rdna3 => 16,  // Up to 16 wave32 waves
            GpuArchitecture::Rdna2 => 20,  // Up to 20 wave32 waves
            GpuArchitecture::Cdna2 => 8,   // Up to 8 wave64 waves
            GpuArchitecture::Cdna3 => 12,  // Up to 12 wave64 waves (estimated)
            GpuArchitecture::Unknown => 16,
        }
    }
}

/// Kernel tuning configuration
///
/// Contains all configurable parameters for HIP kernel optimization.
/// These parameters can be set via environment variables or auto-detected.
#[derive(Debug, Clone)]
pub struct KernelTuning {
    /// Number of threads per block (block dimension)
    pub block_size: u32,
    /// Wavefront size (32 for RDNA, 64 for CDNA)
    pub warp_size: u32,
    /// Use Local Data Share (LDS/Shared Memory) for optimization
    pub use_lds: bool,
    /// LDS size per block in bytes (for shared memory allocation)
    pub lds_size_per_block: u32,
    /// Tile size for matrix tiling (K dimension)
    pub tile_size_k: u32,
    /// Tile size for matrix tiling (N dimension)
    pub tile_size_n: u32,
    /// Number of accumulators per thread (reduces memory traffic)
    pub accumulators_per_thread: u32,
}

impl Default for KernelTuning {
    fn default() -> Self {
        // Default to RDNA3 tuning (most common consumer GPUs)
        KernelTuning {
            block_size: 256,
            warp_size: 32,
            use_lds: true,
            lds_size_per_block: 32 * 1024,  // 32 KB per block
            tile_size_k: 32,
            tile_size_n: 32,
            accumulators_per_thread: 4,
        }
    }
}

impl KernelTuning {
    /// Create tuning configuration for a specific architecture
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use rocmforge::ggml::hip_backend::tuning::KernelTuning;
    ///
    /// let config = KernelTuning::for_architecture("gfx1100");
    /// ```
    pub fn for_architecture(gfx_ip: &str) -> Self {
        let arch = GpuArchitecture::from_gfx_ip(gfx_ip);
        Self::for_gpu_arch(arch)
    }

    /// Create tuning configuration for a GPU architecture
    pub fn for_gpu_arch(arch: GpuArchitecture) -> Self {
        match arch {
            GpuArchitecture::Rdna3 => KernelTuning {
                block_size: 256,
                warp_size: 32,
                use_lds: true,
                lds_size_per_block: 32 * 1024,  // 32 KB
                tile_size_k: 32,
                tile_size_n: 32,
                accumulators_per_thread: 4,
            },
            GpuArchitecture::Rdna2 => KernelTuning {
                block_size: 256,
                warp_size: 32,
                use_lds: true,
                lds_size_per_block: 16 * 1024,  // 16 KB (smaller LDS on RDNA2)
                tile_size_k: 32,
                tile_size_n: 32,
                accumulators_per_thread: 4,
            },
            GpuArchitecture::Cdna2 => KernelTuning {
                block_size: 512,
                warp_size: 64,
                use_lds: true,
                lds_size_per_block: 64 * 1024,  // 64 KB (CDNA has more LDS)
                tile_size_k: 64,
                tile_size_n: 64,
                accumulators_per_thread: 8,
            },
            GpuArchitecture::Cdna3 => KernelTuning {
                block_size: 512,
                warp_size: 64,
                use_lds: true,
                lds_size_per_block: 64 * 1024,
                tile_size_k: 64,
                tile_size_n: 64,
                accumulators_per_thread: 8,
            },
            GpuArchitecture::Unknown => KernelTuning::default(),
        }
    }

    /// Create tuning configuration from environment variables
    ///
    /// Environment variables:
    /// - `ROCFORGE_BLOCK_SIZE`: Threads per block
    /// - `ROCFORGE_WARP_SIZE`: Wavefront size (32 or 64)
    /// - `ROCFORGE_USE_LDS`: Enable LDS (0 or 1)
    /// - `ROCFORGE_LDS_SIZE`: LDS per block in bytes
    /// - `ROCFORGE_TILE_K`: K tile size
    /// - `ROCFORGE_TILE_N`: N tile size
    pub fn from_env() -> Self {
        let mut config = KernelTuning::default();

        if let Ok(block_size) = env::var("ROCFORGE_BLOCK_SIZE") {
            if let Ok(size) = block_size.parse::<u32>() {
                config.block_size = size;
            }
        }

        if let Ok(warp_size) = env::var("ROCFORGE_WARP_SIZE") {
            if let Ok(size) = warp_size.parse::<u32>() {
                config.warp_size = size;
            }
        }

        if let Ok(use_lds) = env::var("ROCFORGE_USE_LDS") {
            config.use_lds = use_lds != "0" && use_lds.to_lowercase() != "false";
        }

        if let Ok(lds_size) = env::var("ROCFORGE_LDS_SIZE") {
            if let Ok(size) = lds_size.parse::<u32>() {
                config.lds_size_per_block = size;
            }
        }

        if let Ok(tile_k) = env::var("ROCFORGE_TILE_K") {
            if let Ok(size) = tile_k.parse::<u32>() {
                config.tile_size_k = size;
            }
        }

        if let Ok(tile_n) = env::var("ROCFORGE_TILE_N") {
            if let Ok(size) = tile_n.parse::<u32>() {
                config.tile_size_n = size;
            }
        }

        config
    }

    /// Create tuning configuration with custom overrides
    pub fn with_override(&self, f: impl FnOnce(&mut KernelTuning)) -> Self {
        let mut config = self.clone();
        f(&mut config);
        config
    }

    /// Calculate number of waves per block
    pub fn waves_per_block(&self) -> u32 {
        self.block_size / self.warp_size
    }

    /// Validate the tuning configuration
    ///
    /// Returns an error if the configuration is invalid
    pub fn validate(&self) -> Result<(), String> {
        if self.block_size == 0 {
            return Err("block_size cannot be zero".to_string());
        }

        if self.warp_size != 32 && self.warp_size != 64 {
            return Err(format!("warp_size must be 32 or 64, got {}", self.warp_size));
        }

        if self.block_size % self.warp_size != 0 {
            return Err(format!(
                "block_size ({}) must be a multiple of warp_size ({})",
                self.block_size, self.warp_size
            ));
        }

        if self.tile_size_k == 0 || self.tile_size_n == 0 {
            return Err("tile sizes cannot be zero".to_string());
        }

        if self.accumulators_per_thread == 0 {
            return Err("accumulators_per_thread cannot be zero".to_string());
        }

        Ok(())
    }

    /// Generate kernel launch parameters as C preprocessor defines
    ///
    /// This is useful for passing tuning parameters to HIP kernels
    pub fn kernel_defines(&self) -> Vec<String> {
        vec![
            format!("-DBLOCK_SIZE={}", self.block_size),
            format!("-DWARP_SIZE={}", self.warp_size),
            format!("-DTILE_SIZE_K={}", self.tile_size_k),
            format!("-DTILE_SIZE_N={}", self.tile_size_n),
            format!("-DACC_PER_THREAD={}", self.accumulators_per_thread),
            if self.use_lds {
                "-DUSE_LDS=1".to_string()
            } else {
                "-DUSE_LDS=0".to_string()
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_detection() {
        assert_eq!(GpuArchitecture::from_gfx_ip("gfx1100"), GpuArchitecture::Rdna3);
        assert_eq!(GpuArchitecture::from_gfx_ip("gfx1101"), GpuArchitecture::Rdna3);
        assert_eq!(GpuArchitecture::from_gfx_ip("gfx1102"), GpuArchitecture::Rdna3);
        assert_eq!(GpuArchitecture::from_gfx_ip("gfx1030"), GpuArchitecture::Rdna2);
        assert_eq!(GpuArchitecture::from_gfx_ip("gfx1035"), GpuArchitecture::Rdna2);
        assert_eq!(GpuArchitecture::from_gfx_ip("gfx90a"), GpuArchitecture::Cdna2);
        assert_eq!(GpuArchitecture::from_gfx_ip("gfx940"), GpuArchitecture::Cdna3);
        assert_eq!(GpuArchitecture::from_gfx_ip("unknown"), GpuArchitecture::Unknown);
    }

    #[test]
    fn test_wave_size() {
        assert_eq!(GpuArchitecture::Rdna3.wave_size(), 32);
        assert_eq!(GpuArchitecture::Rdna2.wave_size(), 32);
        assert_eq!(GpuArchitecture::Cdna2.wave_size(), 64);
        assert_eq!(GpuArchitecture::Cdna3.wave_size(), 64);
    }

    #[test]
    fn test_tuning_default() {
        let config = KernelTuning::default();
        assert_eq!(config.block_size, 256);
        assert_eq!(config.warp_size, 32);
        assert!(config.use_lds);
        assert_eq!(config.tile_size_k, 32);
        assert_eq!(config.tile_size_n, 32);
    }

    #[test]
    fn test_tuning_for_architecture() {
        let rdna3 = KernelTuning::for_architecture("gfx1100");
        assert_eq!(rdna3.block_size, 256);
        assert_eq!(rdna3.warp_size, 32);

        let cdna2 = KernelTuning::for_architecture("gfx90a");
        assert_eq!(cdna2.block_size, 512);
        assert_eq!(cdna2.warp_size, 64);
    }

    #[test]
    fn test_waves_per_block() {
        let config = KernelTuning::default();
        assert_eq!(config.waves_per_block(), 8); // 256 / 32 = 8

        let cdna_config = KernelTuning::for_architecture("gfx90a");
        assert_eq!(cdna_config.waves_per_block(), 8); // 512 / 64 = 8
    }

    #[test]
    fn test_validation() {
        let mut config = KernelTuning::default();
        assert!(config.validate().is_ok());

        config.warp_size = 48; // Invalid
        assert!(config.validate().is_err());

        config.warp_size = 32;
        config.block_size = 100; // Not divisible by warp_size
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_with_override() {
        let config = KernelTuning::default()
            .with_override(|c| c.block_size = 512);

        assert_eq!(config.block_size, 512);
        assert_eq!(config.warp_size, 32); // Unchanged
    }

    #[test]
    fn test_kernel_defines() {
        let config = KernelTuning::default();
        let defines = config.kernel_defines();

        assert!(defines.iter().any(|d| d.contains("BLOCK_SIZE=256")));
        assert!(defines.iter().any(|d| d.contains("WARP_SIZE=32")));
        assert!(defines.iter().any(|d| d.contains("USE_LDS=1")));
    }
}
