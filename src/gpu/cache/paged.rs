use super::{GpuBuffer, GpuKvCache, GpuResult};
use std::sync::Arc;

#[derive(Clone, Copy)]
struct BlockOverlap {
    logical_block_start_token: usize,
    overlap_start: usize,
    overlap_tokens: usize,
}

fn overlap_for_block(
    start_pos: usize,
    seq_len: usize,
    logical_block: usize,
    block_size: usize,
) -> Option<BlockOverlap> {
    let block_start = logical_block * block_size;
    let block_end = (logical_block + 1) * block_size;
    let overlap_start = std::cmp::max(start_pos, block_start);
    let overlap_end = std::cmp::min(start_pos + seq_len, block_end);
    let overlap_tokens = overlap_end.saturating_sub(overlap_start);

    if overlap_tokens == 0 {
        None
    } else {
        Some(BlockOverlap {
            logical_block_start_token: block_start,
            overlap_start,
            overlap_tokens,
        })
    }
}

impl GpuKvCache {
    fn ensure_paged_block(
        &mut self,
        layer: usize,
        physical_block: usize,
        block_bytes: usize,
    ) -> GpuResult<()> {
        if physical_block >= self.paged_k[layer].len() {
            self.paged_k[layer].resize_with(physical_block + 1, || None);
            self.paged_v[layer].resize_with(physical_block + 1, || None);
        }

        if self.paged_k[layer][physical_block].is_none() {
            let k_buf = GpuBuffer::alloc(block_bytes)?;
            self.paged_k[layer][physical_block] = Some(Arc::new(k_buf));
        }
        if self.paged_v[layer][physical_block].is_none() {
            let v_buf = GpuBuffer::alloc(block_bytes)?;
            self.paged_v[layer][physical_block] = Some(Arc::new(v_buf));
        }

        Ok(())
    }

    /// Sync a range of tokens from the contiguous working view to the paged cache.
    pub fn scatter_to_paged(
        &mut self,
        layer: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> GpuResult<()> {
        let block_size = self.block_size_tokens;
        let pos_bytes = self.pos_bytes;
        let block_bytes = block_size * pos_bytes;

        if seq_len == 0 {
            return Ok(());
        }
        let start_block = start_pos / block_size;
        let end_block = (start_pos + seq_len - 1) / block_size;

        for logical_block in start_block..=end_block {
            while logical_block >= self.block_table.block_ids.len() {
                let physical_block = self.block_allocator.allocate();
                self.block_table.block_ids.push(physical_block);
            }
            let physical_block = self.block_table.block_ids[logical_block];
            self.ensure_paged_block(layer, physical_block, block_bytes)?;

            if let Some(overlap) = overlap_for_block(start_pos, seq_len, logical_block, block_size)
            {
                let k_block = self.paged_k[layer][physical_block]
                    .as_ref()
                    .expect("invariant: paged K block allocated above");
                let v_block = self.paged_v[layer][physical_block]
                    .as_ref()
                    .expect("invariant: paged V block allocated above");

                let contig_offset = overlap.overlap_start * pos_bytes;
                let block_offset =
                    (overlap.overlap_start - overlap.logical_block_start_token) * pos_bytes;
                let copy_size = overlap.overlap_tokens * pos_bytes;

                unsafe {
                    let contig_k_ptr = (self.k[layer].as_ptr() as *const u8).add(contig_offset);
                    let block_k_ptr = (k_block.as_ptr() as *mut u8).add(block_offset);
                    super::super::ffi::hip_memcpy_d2d(block_k_ptr, contig_k_ptr, copy_size)?;

                    let contig_v_ptr = (self.v[layer].as_ptr() as *const u8).add(contig_offset);
                    let block_v_ptr = (v_block.as_ptr() as *mut u8).add(block_offset);
                    super::super::ffi::hip_memcpy_d2d(block_v_ptr, contig_v_ptr, copy_size)?;
                }
            }
        }
        Ok(())
    }

    /// Scatter on an explicit HIP stream without synchronizing the host.
    ///
    /// The caller must ensure that all writes to the contiguous K/V buffers have
    /// been submitted to `stream` before this call, and must synchronize `stream`
    /// before reading the paged cache from another stream.
    pub fn scatter_to_paged_on_stream(
        &mut self,
        layer: usize,
        start_pos: usize,
        seq_len: usize,
        stream: crate::gpu::ffi::hipStream_t,
    ) -> GpuResult<()> {
        let block_size = self.block_size_tokens;
        let pos_bytes = self.pos_bytes;
        let block_bytes = block_size * pos_bytes;

        if seq_len == 0 {
            return Ok(());
        }
        let start_block = start_pos / block_size;
        let end_block = (start_pos + seq_len - 1) / block_size;

        for logical_block in start_block..=end_block {
            while logical_block >= self.block_table.block_ids.len() {
                let physical_block = self.block_allocator.allocate();
                self.block_table.block_ids.push(physical_block);
            }
            let physical_block = self.block_table.block_ids[logical_block];
            self.ensure_paged_block(layer, physical_block, block_bytes)?;

            if let Some(overlap) = overlap_for_block(start_pos, seq_len, logical_block, block_size)
            {
                let k_block = self.paged_k[layer][physical_block]
                    .as_ref()
                    .expect("invariant: paged K block allocated above");
                let v_block = self.paged_v[layer][physical_block]
                    .as_ref()
                    .expect("invariant: paged V block allocated above");

                let contig_offset = overlap.overlap_start * pos_bytes;
                let block_offset =
                    (overlap.overlap_start - overlap.logical_block_start_token) * pos_bytes;
                let copy_size = overlap.overlap_tokens * pos_bytes;

                unsafe {
                    let contig_k_ptr = (self.k[layer].as_ptr() as *const u8).add(contig_offset);
                    let block_k_ptr = (k_block.as_ptr() as *mut u8).add(block_offset);
                    super::super::ffi::hip_memcpy_d2d_async(
                        block_k_ptr,
                        contig_k_ptr,
                        copy_size,
                        stream,
                    )?;

                    let contig_v_ptr = (self.v[layer].as_ptr() as *const u8).add(contig_offset);
                    let block_v_ptr = (v_block.as_ptr() as *mut u8).add(block_offset);
                    super::super::ffi::hip_memcpy_d2d_async(
                        block_v_ptr,
                        contig_v_ptr,
                        copy_size,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Sync all blocks from the paged cache back to the contiguous working view for a layer.
    pub fn gather_to_contiguous(&self, layer: usize) -> GpuResult<()> {
        let block_size = self.block_size_tokens;
        let pos_bytes = self.pos_bytes;

        for (logical_block, &physical_block) in self.block_table.block_ids.iter().enumerate() {
            if physical_block >= self.paged_k[layer].len() {
                continue;
            }

            let contig_offset = logical_block * block_size * pos_bytes;
            let copy_size = block_size * pos_bytes;

            if let Some(ref k_block) = self.paged_k[layer][physical_block] {
                unsafe {
                    let contig_k_ptr = (self.k[layer].as_ptr() as *mut u8).add(contig_offset);
                    super::super::ffi::hip_memcpy_d2d(contig_k_ptr, k_block.as_ptr(), copy_size)?;
                }
            }
            if let Some(ref v_block) = self.paged_v[layer][physical_block] {
                unsafe {
                    let contig_v_ptr = (self.v[layer].as_ptr() as *mut u8).add(contig_offset);
                    super::super::ffi::hip_memcpy_d2d(contig_v_ptr, v_block.as_ptr(), copy_size)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::overlap_for_block;

    #[test]
    fn overlap_for_block_skips_non_overlapping_block() {
        assert!(overlap_for_block(32, 4, 1, 16).is_none());
    }

    #[test]
    fn overlap_for_block_handles_partial_overlap() {
        let overlap = overlap_for_block(10, 12, 0, 16).expect("expected overlap");
        assert_eq!(overlap.logical_block_start_token, 0);
        assert_eq!(overlap.overlap_start, 10);
        assert_eq!(overlap.overlap_tokens, 6);
    }

    #[test]
    fn overlap_for_block_handles_middle_block_full_coverage() {
        let overlap = overlap_for_block(10, 40, 1, 16).expect("expected overlap");
        assert_eq!(overlap.logical_block_start_token, 16);
        assert_eq!(overlap.overlap_start, 16);
        assert_eq!(overlap.overlap_tokens, 16);
    }
}
