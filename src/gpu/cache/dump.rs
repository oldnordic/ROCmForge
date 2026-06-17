use super::GpuKvCache;

/// Magic bytes that identify a KV cache dump file.
pub const KV_DUMP_MAGIC: &[u8; 8] = b"KVCACHE1";

/// In-memory representation of a KV cache dump loaded from disk.
///
/// Layout:
/// - `k[layer]`: flat `Vec<f32>` of shape `[num_tokens × kv_size]`
///   where `kv_size = num_kv_heads × head_dim`.
/// - `v[layer]`: same shape.
#[derive(Debug)]
pub struct KvDump {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_tokens: usize,
    /// Key vectors per layer. `k[l][t * kv_size .. (t+1) * kv_size]` = token t.
    pub k: Vec<Vec<f32>>,
    /// Value vectors per layer. Same layout as `k`.
    pub v: Vec<Vec<f32>>,
}

impl KvDump {
    /// Load a KV cache dump written by [`GpuKvCache::dump_to_file`].
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        use std::io::Read;
        let mut data = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut data)?;

        // Header: 8 magic + 4×4 fields + 8 padding = 32 bytes
        if data.len() < 32 {
            return Err("KvDump: file too short to contain header".into());
        }
        if &data[..8] != KV_DUMP_MAGIC {
            return Err(format!(
                "KvDump: bad magic {:?}, expected {:?}",
                &data[..8],
                KV_DUMP_MAGIC
            )
            .into());
        }
        let num_layers = u32::from_le_bytes(data[8..12].try_into()?) as usize;
        let num_kv_heads = u32::from_le_bytes(data[12..16].try_into()?) as usize;
        let head_dim = u32::from_le_bytes(data[16..20].try_into()?) as usize;
        let num_tokens = u32::from_le_bytes(data[20..24].try_into()?) as usize;
        // bytes 24..32 are padding

        let kv_size = num_kv_heads * head_dim;
        let floats_per_layer = num_tokens * kv_size;
        let bytes_per_layer = floats_per_layer * 4;
        let expected_len = 32 + 2 * num_layers * bytes_per_layer;

        if data.len() < expected_len {
            return Err(format!(
                "KvDump: truncated — expected {} bytes, got {}",
                expected_len,
                data.len()
            )
            .into());
        }

        let mut k = Vec::with_capacity(num_layers);
        let mut v = Vec::with_capacity(num_layers);
        let mut offset = 32usize;

        for _ in 0..num_layers {
            let k_floats: Vec<f32> = data[offset..offset + bytes_per_layer]
                .chunks_exact(4)
                .map(|b| {
                    f32::from_le_bytes(
                        b.try_into()
                            .expect("invariant: chunks_exact(4) produces 4-byte slices"),
                    )
                })
                .collect();
            offset += bytes_per_layer;

            let v_floats: Vec<f32> = data[offset..offset + bytes_per_layer]
                .chunks_exact(4)
                .map(|b| {
                    f32::from_le_bytes(
                        b.try_into()
                            .expect("invariant: chunks_exact(4) produces 4-byte slices"),
                    )
                })
                .collect();
            offset += bytes_per_layer;

            k.push(k_floats);
            v.push(v_floats);
        }

        Ok(KvDump {
            num_layers,
            num_kv_heads,
            head_dim,
            num_tokens,
            k,
            v,
        })
    }

    /// Return the key vector for a specific layer and token position.
    ///
    /// Returns a slice of length `num_kv_heads * head_dim`.
    pub fn key(&self, layer: usize, token: usize) -> &[f32] {
        let kv_size = self.num_kv_heads * self.head_dim;
        &self.k[layer][token * kv_size..(token + 1) * kv_size]
    }

    /// Return the value vector for a specific layer and token position.
    pub fn val(&self, layer: usize, token: usize) -> &[f32] {
        let kv_size = self.num_kv_heads * self.head_dim;
        &self.v[layer][token * kv_size..(token + 1) * kv_size]
    }
}

impl GpuKvCache {
    /// Dump the first `num_tokens` positions of every layer's KV cache to a
    /// binary file for off-GPU analysis.
    ///
    /// The file format is:
    /// ```text
    /// [u8; 8]  magic = "KVCACHE1"
    /// u32      num_layers
    /// u32      num_kv_heads
    /// u32      head_dim
    /// u32      num_tokens
    /// [u8; 8]  padding
    /// -- for each layer l in 0..num_layers:
    ///    [f32; num_tokens * num_kv_heads * head_dim]  K[l]
    ///    [f32; num_tokens * num_kv_heads * head_dim]  V[l]
    /// ```
    ///
    /// This is a research / analysis tool; it synchronises the GPU stream and
    /// copies VRAM → host, so it is not suitable for hot inference paths.
    pub fn dump_to_file(
        &self,
        path: &std::path::Path,
        num_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        if num_tokens == 0 || num_tokens > self.max_seq_len {
            return Err(format!(
                "dump_to_file: num_tokens {} out of range [1, {}]",
                num_tokens, self.max_seq_len
            )
            .into());
        }

        let kv_size = num_kv_heads * head_dim;
        let floats_per_layer = num_tokens * kv_size;

        let mut file = std::fs::File::create(path)?;

        // Header
        file.write_all(KV_DUMP_MAGIC)?;
        file.write_all(&(self.num_layers as u32).to_le_bytes())?;
        file.write_all(&(num_kv_heads as u32).to_le_bytes())?;
        file.write_all(&(head_dim as u32).to_le_bytes())?;
        file.write_all(&(num_tokens as u32).to_le_bytes())?;
        file.write_all(&[0u8; 8])?; // padding

        // Body: K then V for each layer
        for layer in 0..self.num_layers {
            // copy_to_host_vec reads the whole layer buffer (max_seq_len * kv_size)
            let full_k = self.k[layer].copy_to_host_vec()?;
            let full_v = self.v[layer].copy_to_host_vec()?;

            // Write only the populated prefix
            let k_bytes: Vec<u8> = full_k[..floats_per_layer]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            file.write_all(&k_bytes)?;

            let v_bytes: Vec<u8> = full_v[..floats_per_layer]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            file.write_all(&v_bytes)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{KvDump, KV_DUMP_MAGIC};

    /// A KvDump written and re-parsed must round-trip all header fields.
    #[test]
    fn kv_dump_header_round_trips() -> Result<(), Box<dyn std::error::Error>> {
        use std::io::{Read, Write};
        use tempfile::NamedTempFile;

        let mut f = NamedTempFile::new()?;

        let num_layers: u32 = 4;
        let num_kv_heads: u32 = 8;
        let head_dim: u32 = 128;
        let num_tokens: u32 = 16;

        // Write header
        f.write_all(KV_DUMP_MAGIC)?;
        f.write_all(&num_layers.to_le_bytes())?;
        f.write_all(&num_kv_heads.to_le_bytes())?;
        f.write_all(&head_dim.to_le_bytes())?;
        f.write_all(&num_tokens.to_le_bytes())?;
        f.write_all(&[0u8; 8])?; // padding

        // Write zero-filled payload bytes
        let floats_per_layer = num_tokens as usize * num_kv_heads as usize * head_dim as usize;
        let zeros = vec![0u8; floats_per_layer * 4 * 2 * num_layers as usize];
        f.write_all(&zeros)?;
        f.flush()?;

        // Read back and check header
        let path = f.path().to_owned();
        let mut buf = Vec::new();
        std::fs::File::open(&path)?.read_to_end(&mut buf)?;

        assert_eq!(&buf[..8], KV_DUMP_MAGIC, "magic mismatch");
        assert_eq!(u32::from_le_bytes(buf[8..12].try_into()?), num_layers);
        assert_eq!(u32::from_le_bytes(buf[12..16].try_into()?), num_kv_heads);
        assert_eq!(u32::from_le_bytes(buf[16..20].try_into()?), head_dim);
        assert_eq!(u32::from_le_bytes(buf[20..24].try_into()?), num_tokens);

        // Total size: 32 header + 2 * num_layers * floats_per_layer * 4
        let expected_len = 32 + 2 * num_layers as usize * floats_per_layer * 4;
        assert_eq!(buf.len(), expected_len, "file size mismatch");
        Ok(())
    }

    /// A KvDump with wrong magic returns an error.
    #[test]
    fn kv_dump_parse_rejects_bad_magic() -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut f = NamedTempFile::new()?;
        f.write_all(b"BADMAGIC")?;
        f.write_all(&[0u8; 100])?;
        f.flush()?;

        let result = KvDump::load(f.path());
        assert!(result.is_err(), "should fail on bad magic");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("magic"), "error should mention magic: {msg}");
        Ok(())
    }

    /// A KvDump with truncated data returns an error.
    #[test]
    fn kv_dump_parse_rejects_truncated_file() -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut f = NamedTempFile::new()?;
        f.write_all(KV_DUMP_MAGIC)?;
        // Write a header claiming 4 layers / 8 heads / 128 dim / 16 tokens
        // but no body data
        f.write_all(&4u32.to_le_bytes())?;
        f.write_all(&8u32.to_le_bytes())?;
        f.write_all(&128u32.to_le_bytes())?;
        f.write_all(&16u32.to_le_bytes())?;
        f.write_all(&[0u8; 8])?;
        f.flush()?;

        let result = KvDump::load(f.path());
        assert!(result.is_err(), "should fail on truncated body");
        Ok(())
    }
}
