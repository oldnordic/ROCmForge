//! Validation utilities for ggml graphs.

use crate::ggml::{GgmlResult, Graph};

pub fn validate_graph(_graph: &Graph) -> GgmlResult<()> {
    Ok(())
}
