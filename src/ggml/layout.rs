//! Tensor layout definitions.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
    ColMajor,
    Strided,
}
