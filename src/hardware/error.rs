//! Hardware detection errors.

/// Errors that can occur during hardware capability detection.
#[derive(Debug)]
pub enum HardwareError {
    /// Hardware detection failed with a descriptive message.
    DetectionFailed(String),
    /// The current platform is not supported for hardware detection.
    UnsupportedPlatform(String),
}

impl std::fmt::Display for HardwareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HardwareError::DetectionFailed(msg) => write!(f, "hardware detection failed: {}", msg),
            HardwareError::UnsupportedPlatform(msg) => {
                write!(f, "unsupported platform: {}", msg)
            }
        }
    }
}

impl std::error::Error for HardwareError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_detection_failed() {
        let e = HardwareError::DetectionFailed("no CPU found".to_string());
        assert_eq!(e.to_string(), "hardware detection failed: no CPU found");
    }

    #[test]
    fn display_unsupported_platform() {
        let e = HardwareError::UnsupportedPlatform(" wasm32".to_string());
        assert_eq!(e.to_string(), "unsupported platform:  wasm32");
    }
}
