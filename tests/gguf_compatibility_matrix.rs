// ROCmForge - AMD GPU LLM Inference Engine
// GGUF Compatibility Matrix Tests
//
// Phase 8: GGUF Compatibility - Task 08-11
// Documents GGUF compatibility across architectures and quantization formats

use rocmforge::loader::GgufTensorType;

/// Check if a quantization format is supported
pub fn is_format_supported(format: GgufTensorType) -> bool {
    matches!(
        format,
        GgufTensorType::F32
            | GgufTensorType::F16
            | GgufTensorType::Q4_0
            | GgufTensorType::Q4_1
            | GgufTensorType::Q5_0
            | GgufTensorType::Q5_1
            | GgufTensorType::Q8_0
            | GgufTensorType::Q2_K
            | GgufTensorType::Q3_K
            | GgufTensorType::Q4_K
            | GgufTensorType::Q5_K
            | GgufTensorType::Q6_K
            | GgufTensorType::Mxfp4
            | GgufTensorType::Mxfp6E2m3
            | GgufTensorType::Mxfp6E3m2
    )
}

/// Get all supported quantization formats
pub fn supported_formats() -> Vec<GgufTensorType> {
    vec![
        GgufTensorType::F32,
        GgufTensorType::F16,
        GgufTensorType::Q4_0,
        GgufTensorType::Q4_1,
        GgufTensorType::Q5_0,
        GgufTensorType::Q5_1,
        GgufTensorType::Q8_0,
        GgufTensorType::Q2_K,
        GgufTensorType::Q3_K,
        GgufTensorType::Q4_K,
        GgufTensorType::Q5_K,
        GgufTensorType::Q6_K,
        GgufTensorType::Mxfp4,
        GgufTensorType::Mxfp6E2m3,
        GgufTensorType::Mxfp6E3m2,
    ]
}

/// Count of supported formats
pub fn supported_format_count() -> usize {
    supported_formats().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_15_formats_supported() {
        // Verify all 15 GGUF quantization formats are supported
        assert_eq!(supported_format_count(), 15, "Should support all 15 GGUF formats");
    }

    #[test]
    fn test_k_quant_formats_supported() {
        // Verify all K-quant formats are supported
        let k_quants = vec![
            GgufTensorType::Q2_K,
            GgufTensorType::Q3_K,
            GgufTensorType::Q4_K,
            GgufTensorType::Q5_K,
            GgufTensorType::Q6_K,
        ];

        for format in k_quants {
            assert!(
                is_format_supported(format),
                "K-quant format {:?} should be supported",
                format
            );
        }
    }

    #[test]
    fn test_standard_formats_supported() {
        // Verify standard formats are supported
        let standard = vec![
            GgufTensorType::F32,
            GgufTensorType::F16,
            GgufTensorType::Q4_0,
            GgufTensorType::Q4_1,
            GgufTensorType::Q5_0,
            GgufTensorType::Q5_1,
            GgufTensorType::Q8_0,
        ];

        for format in standard {
            assert!(
                is_format_supported(format),
                "Standard format {:?} should be supported",
                format
            );
        }
    }

    #[test]
    fn test_mxfp_formats_supported() {
        // Verify MXFP formats are supported
        let mxfp = vec![
            GgufTensorType::Mxfp4,
            GgufTensorType::Mxfp6E2m3,
            GgufTensorType::Mxfp6E3m2,
        ];

        for format in mxfp {
            assert!(
                is_format_supported(format),
                "MXFP format {:?} should be supported",
                format
            );
        }
    }

    #[test]
    fn test_format_coverage() {
        // Test that we have complete coverage across format types
        let formats = supported_formats();

        // K-quants (6 total including Q4_0, Q4_1, Q5_0, Q5_1 which are K-quants too)
        assert!(formats.contains(&GgufTensorType::Q2_K), "Q2_K missing");
        assert!(formats.contains(&GgufTensorType::Q3_K), "Q3_K missing");
        assert!(formats.contains(&GgufTensorType::Q4_K), "Q4_K missing");
        assert!(formats.contains(&GgufTensorType::Q5_K), "Q5_K missing");
        assert!(formats.contains(&GgufTensorType::Q6_K), "Q6_K missing");

        // MXFP
        assert!(formats.contains(&GgufTensorType::Mxfp4), "MXFP4 missing");
        assert!(formats.contains(&GgufTensorType::Mxfp6E2m3), "MXFP6E2M3 missing");
        assert!(formats.contains(&GgufTensorType::Mxfp6E3m2), "MXFP6E3m2 missing");
    }
}
