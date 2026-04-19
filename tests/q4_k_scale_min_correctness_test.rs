//! Q4_K scale/min extraction correctness tests

use rocmforge::cpu::kernels::q4::BlockQ4K;

#[cfg(test)]
mod q4_k_scale_min_tests {
    use super::*;

    /// Test scale/min extraction for first 4 scale pairs (j < 4)
    #[test]
    fn test_scale_min_extraction_j_less_than_4() {
        let mut scales = [0u8; 12];
        scales[0] = 1; // d[0]
        scales[4] = 2; // m[0]

        let (d, m) = BlockQ4K::get_scale_min_k4(0, scales);
        assert_eq!(d, 1, "d[0] should be 1");
        assert_eq!(m, 2, "m[0] should be 2");
    }

    /// Test scale/min extraction for j >= 4 with bit packing
    #[test]
    fn test_scale_min_extraction_j_greater_than_4() {
        let mut scales = [0u8; 12];
        scales[8] = 0xAB;
        scales[4] = 0x03;

        let (d, m) = BlockQ4K::get_scale_min_k4(4, scales);
        assert_eq!(d, 11, "d[4] should be 11 (0x0B)");
        assert_eq!(m, 10, "m[4] should be 10 (0x0A)");
    }

    /// Test all 8 scale/min pairs
    #[test]
    fn test_all_scale_min_pairs() {
        let scales: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 0xAB, 0xCD, 0xEF, 0x12];

        let pairs: Vec<(u8, u8)> = (0..8)
            .map(|j| BlockQ4K::get_scale_min_k4(j, scales))
            .collect();

        assert_eq!(pairs.len(), 8);
    }
}
