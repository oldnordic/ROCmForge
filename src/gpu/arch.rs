//! AMD GPU architecture identification for ROCm-Forge.
//!
//! Provides architecture-specific parameters for quantization kernel optimization.
//! Each architecture has different capabilities (warp size, shared memory, etc.).

/// AMD GPU architecture identifiers for ROCm-Forge
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuArchitecture {
    /// RDNA3 (RX 7000 series, W7900, 7900XTX) - gfx1100
    Gfx1100,
    /// RDNA2 (RX 6000 series) - gfx1030
    Gfx1030,
    /// CDNA2 (MI210) - gfx90a
    Gfx90a,
    /// CDNA1 (MI100) - gfx908
    Gfx908,
    /// Vega (Vega 64, 56, 20, MI25) - gfx900
    Gfx900,
    /// Unknown or unsupported architecture
    Unknown(u32),
}

impl GpuArchitecture {
    /// Maximum threads per block for this architecture
    pub fn max_threads_per_block(&self) -> u32 {
        match self {
            Self::Gfx1100 | Self::Gfx1030 => 1024,
            Self::Gfx90a | Self::Gfx908 => 1024,
            Self::Gfx900 => 1024,
            Self::Unknown(_) => 256, // Conservative default
        }
    }

    /// Warp size (wavefront size) for this architecture
    pub fn warp_size(&self) -> u32 {
        match self {
            Self::Gfx1100 | Self::Gfx1030 => 32, // RDNA
            Self::Gfx90a | Self::Gfx908 | Self::Gfx900 => 64, // CDNA/Vega
            Self::Unknown(_) => 32,
        }
    }

    /// Shared memory per block (bytes)
    pub fn shared_mem_per_block(&self) -> usize {
        match self {
            Self::Gfx1100 | Self::Gfx1030 => 64 * 1024,
            Self::Gfx90a | Self::Gfx908 => 64 * 1024,
            Self::Gfx900 => 64 * 1024,
            Self::Unknown(_) => 32 * 1024,
        }
    }

    /// Parse from device name string (e.g., "gfx1100")
    pub fn from_name(name: &str) -> Option<Self> {
        // Handle format: "gfx1100" or "gfx1100_architecture" or similar
        let name_lower = name.to_lowercase();
        let gfx_name = name_lower
            .split('_')
            .next()?
            .trim_start_matches("gfx");

        let arch_id = u32::from_str_radix(gfx_name, 16).ok()?;
        Some(match arch_id {
            0x1100 => Self::Gfx1100,
            0x1030 => Self::Gfx1030,
            0x90a => Self::Gfx90a,
            0x908 => Self::Gfx908,
            0x900 => Self::Gfx900,
            id => Self::Unknown(id),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gfx1100() {
        let arch = GpuArchitecture::from_name("gfx1100").unwrap();
        assert_eq!(arch, GpuArchitecture::Gfx1100);
        assert_eq!(arch.warp_size(), 32);
    }

    #[test]
    fn test_parse_gfx1030() {
        let arch = GpuArchitecture::from_name("gfx1030").unwrap();
        assert_eq!(arch, GpuArchitecture::Gfx1030);
    }

    #[test]
    fn test_parse_gfx90a() {
        let arch = GpuArchitecture::from_name("gfx90a").unwrap();
        assert_eq!(arch, GpuArchitecture::Gfx90a);
        assert_eq!(arch.warp_size(), 64); // CDNA has 64-wide wavefronts
    }

    #[test]
    fn test_unknown_architecture() {
        let arch = GpuArchitecture::from_name("gfx9999").unwrap();
        assert!(matches!(arch, GpuArchitecture::Unknown(0x9999)));
        assert_eq!(arch.warp_size(), 32); // Conservative default
    }

    #[test]
    fn test_invalid_name_returns_none() {
        assert!(GpuArchitecture::from_name("invalid").is_none());
        assert!(GpuArchitecture::from_name("cuda").is_none());
    }

    #[test]
    fn test_max_threads_per_block() {
        assert_eq!(GpuArchitecture::Gfx1100.max_threads_per_block(), 1024);
        assert_eq!(GpuArchitecture::Gfx90a.max_threads_per_block(), 1024);
        assert_eq!(GpuArchitecture::Unknown(999).max_threads_per_block(), 256);
    }

    #[test]
    fn test_shared_mem_per_block() {
        assert_eq!(GpuArchitecture::Gfx1100.shared_mem_per_block(), 64 * 1024);
        assert_eq!(GpuArchitecture::Unknown(999).shared_mem_per_block(), 32 * 1024);
    }
}
