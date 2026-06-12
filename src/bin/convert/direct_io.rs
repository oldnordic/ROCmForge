//! Direct I/O (O_DIRECT) writer for RFM conversion output.
//!
//! Bypasses the Linux page cache to avoid double-caching multi-GB model files.
//! All writes use 512-byte aligned buffers — required by O_DIRECT.
//!
//! On non-Linux platforms this degrades to a normal `File` passthrough.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

/// Buffered writer backed by a Linux O_DIRECT file descriptor.
///
/// Accumulates writes in a 512-byte aligned buffer and flushes full 512-byte
/// chunks.  The caller must call [`Self::flush`] before drop to ensure any
/// trailing partial chunk is padded and written.  After closing, the file
/// must be reopened with normal I/O to perform seek-and-patch operations
/// (O_DIRECT does not support unaligned writes).
pub(super) struct DirectIoWriter {
    file: File,
    #[cfg(target_os = "linux")]
    buf: rocmforge::aligned::AlignedVec<u8>,
    #[cfg(target_os = "linux")]
    buf_filled: usize,
}

impl DirectIoWriter {
    const BUF_CAP: usize = 64 * 1024; // 64 KB
    const ALIGN: usize = 512;

    /// Create a new file with O_DIRECT on Linux, or a normal file elsewhere.
    pub(super) fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::OpenOptionsExt;
            const O_DIRECT: i32 = 0x4000;

            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .custom_flags(O_DIRECT)
                .open(path)?;

            let buf =
                rocmforge::aligned::AlignedVec::new_zeroed(Self::BUF_CAP, Self::ALIGN);

            Ok(Self {
                file,
                buf,
                buf_filled: 0,
            })
        }

        #[cfg(not(target_os = "linux"))]
        {
            let file = File::create(path)?;
            Ok(Self { file })
        }
    }

    #[cfg(target_os = "linux")]
    fn flush_aligned(&mut self) -> io::Result<()> {
        let aligned = (self.buf_filled / Self::ALIGN) * Self::ALIGN;
        if aligned == 0 {
            return Ok(());
        }
        self.file.write_all(&self.buf[..aligned])?;
        let remainder = self.buf_filled - aligned;
        if remainder > 0 {
            let src = self.buf[aligned..self.buf_filled].to_vec();
            self.buf[..remainder].copy_from_slice(&src);
        }
        self.buf_filled = remainder;
        Ok(())
    }
}

impl Write for DirectIoWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        #[cfg(target_os = "linux")]
        {
            let mut remaining = data;
            while !remaining.is_empty() {
                let space = self.buf.len() - self.buf_filled;
                if space == 0 {
                    self.flush_aligned()?;
                    continue;
                }
                let n = remaining.len().min(space);
                self.buf[self.buf_filled..self.buf_filled + n]
                    .copy_from_slice(&remaining[..n]);
                self.buf_filled += n;
                remaining = &remaining[n..];
            }
            Ok(data.len())
        }

        #[cfg(not(target_os = "linux"))]
        {
            self.file.write(data)
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        #[cfg(target_os = "linux")]
        {
            self.flush_aligned()?;
            if self.buf_filled > 0 {
                let padded =
                    ((self.buf_filled + Self::ALIGN - 1) / Self::ALIGN) * Self::ALIGN;
                for i in self.buf_filled..padded {
                    self.buf[i] = 0;
                }
                self.file.write_all(&self.buf[..padded])?;
                self.buf_filled = 0;
            }
        }
        self.file.flush()
    }
}

impl Drop for DirectIoWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
