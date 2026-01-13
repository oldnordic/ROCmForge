//! Shape helpers for ggml IR.

pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}
