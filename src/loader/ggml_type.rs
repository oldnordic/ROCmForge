use crate::loader::LoadError;

/// GGUF tensor data types — matches ggml_type from ggml.h
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
}

impl GgmlType {
    /// Convert a GGML type code to a GgmlType enum variant.
    ///
    /// Returns `LoadError::UnknownTensorType` for invalid codes.
    pub fn from_u32(v: u32) -> Result<Self, LoadError> {
        match v {
            0 => Ok(GgmlType::F32),
            1 => Ok(GgmlType::F16),
            2 => Ok(GgmlType::Q4_0),
            3 => Ok(GgmlType::Q4_1),
            6 => Ok(GgmlType::Q5_0),
            7 => Ok(GgmlType::Q5_1),
            8 => Ok(GgmlType::Q8_0),
            10 => Ok(GgmlType::Q2_K),
            11 => Ok(GgmlType::Q3_K),
            12 => Ok(GgmlType::Q4_K),
            13 => Ok(GgmlType::Q5_K),
            14 => Ok(GgmlType::Q6_K),
            _ => Err(LoadError::UnknownTensorType(v)),
        }
    }

    /// Bytes required to store `n` elements of this type.
    ///
    /// Block sizes verified against llama.cpp ggml.h:
    ///   Q4_0: 32 elems / 18 bytes   Q4_1: 32 / 20
    ///   Q5_0: 32 / 22               Q5_1: 32 / 24
    ///   Q8_0: 32 / 34
    ///   Q2_K: 256 / 256             Q3_K: 256 / 256
    ///   Q4_K: 256 / 144             Q5_K: 256 / 176
    ///   Q6_K: 256 / 210
    pub fn bytes_for_elements(&self, n: usize) -> usize {
        match self {
            GgmlType::F32 => n * 4,
            GgmlType::F16 => n * 2,
            GgmlType::Q4_0 => n.div_ceil(32) * 18,
            GgmlType::Q4_1 => n.div_ceil(32) * 20,
            GgmlType::Q5_0 => n.div_ceil(32) * 22,
            GgmlType::Q5_1 => n.div_ceil(32) * 24,
            GgmlType::Q8_0 => n.div_ceil(32) * 34,
            GgmlType::Q2_K => n.div_ceil(256) * 256,
            GgmlType::Q3_K => n.div_ceil(256) * 256,
            GgmlType::Q4_K => n.div_ceil(256) * 144,
            GgmlType::Q5_K => n.div_ceil(256) * 176,
            GgmlType::Q6_K => n.div_ceil(256) * 210,
        }
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::Q4_0 => "Q4_0",
            GgmlType::Q4_1 => "Q4_1",
            GgmlType::Q5_0 => "Q5_0",
            GgmlType::Q5_1 => "Q5_1",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q2_K => "Q2_K",
            GgmlType::Q3_K => "Q3_K",
            GgmlType::Q4_K => "Q4_K",
            GgmlType::Q5_K => "Q5_K",
            GgmlType::Q6_K => "Q6_K",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_all_types() {
        let cases: &[(u32, GgmlType)] = &[
            (0, GgmlType::F32),
            (1, GgmlType::F16),
            (2, GgmlType::Q4_0),
            (3, GgmlType::Q4_1),
            (6, GgmlType::Q5_0),
            (7, GgmlType::Q5_1),
            (8, GgmlType::Q8_0),
            (10, GgmlType::Q2_K),
            (11, GgmlType::Q3_K),
            (12, GgmlType::Q4_K),
            (13, GgmlType::Q5_K),
            (14, GgmlType::Q6_K),
        ];
        for &(v, expected) in cases {
            assert_eq!(GgmlType::from_u32(v).unwrap(), expected);
        }
    }

    #[test]
    fn unknown_type_is_error() {
        assert!(GgmlType::from_u32(99).is_err());
    }

    #[test]
    fn bytes_f32() {
        assert_eq!(GgmlType::F32.bytes_for_elements(100), 400);
    }

    #[test]
    fn bytes_q4_0_exact_block() {
        // 32 elements = 1 block = 18 bytes
        assert_eq!(GgmlType::Q4_0.bytes_for_elements(32), 18);
        // 33 elements = 2 blocks = 36 bytes
        assert_eq!(GgmlType::Q4_0.bytes_for_elements(33), 36);
    }

    #[test]
    fn bytes_q4_k() {
        // 256 elements = 1 superblock = 144 bytes
        assert_eq!(GgmlType::Q4_K.bytes_for_elements(256), 144);
    }
}
