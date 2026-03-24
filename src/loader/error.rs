use std::fmt;

/// Errors that can occur during GGUF file loading.
#[derive(Debug)]
pub enum LoadError {
    /// I/O error while reading the file.
    Io(std::io::Error),
    /// GGUF magic bytes do not match "GGUF".
    InvalidMagic([u8; 4]),
    /// GGUF version is outside the supported range (2-3).
    UnsupportedVersion(u32),
    /// Unknown GGML tensor type code.
    UnknownTensorType(u32),
    /// Tensor data extends past the end of the file.
    OutOfBounds {
        offset: u64,
        size: usize,
        file_size: usize,
    },
    /// Invalid UTF-8 in a string field.
    InvalidUtf8(Vec<u8>),
    /// String exceeds the sanity limit.
    StringTooLong(usize),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "I/O error: {}", e),
            LoadError::InvalidMagic(got) => write!(f, "invalid GGUF magic: {:?}", got),
            LoadError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version: {}", v),
            LoadError::UnknownTensorType(t) => write!(f, "unknown tensor type: {}", t),
            LoadError::OutOfBounds {
                offset,
                size,
                file_size,
            } => {
                write!(
                    f,
                    "slice out of bounds: offset={} size={} file_size={}",
                    offset, size, file_size
                )
            }
            LoadError::InvalidUtf8(_) => write!(f, "invalid UTF-8 in string field"),
            LoadError::StringTooLong(n) => write!(f, "string too long: {} bytes", n),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let LoadError::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}
