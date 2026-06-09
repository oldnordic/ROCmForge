//! Block allocator and logical block table for paged KV cache storage.

#[derive(Clone, Debug)]
pub struct BlockTable {
    pub block_ids: Vec<usize>,
}

#[derive(Debug)]
pub struct BlockAllocator {
    pub block_size_tokens: usize,
    pub free_list: Vec<usize>,
    pub refcounts: Vec<usize>,
    pub total_blocks: usize,
}

impl BlockAllocator {
    pub fn new(block_size_tokens: usize) -> Self {
        Self {
            block_size_tokens,
            free_list: Vec::new(),
            refcounts: Vec::new(),
            total_blocks: 0,
        }
    }

    pub fn allocate(&mut self) -> usize {
        if let Some(id) = self.free_list.pop() {
            self.refcounts[id] = 1;
            id
        } else {
            let id = self.total_blocks;
            self.total_blocks += 1;
            self.refcounts.push(1);
            id
        }
    }

    pub fn release(&mut self, id: usize) -> bool {
        if id < self.refcounts.len() && self.refcounts[id] > 0 {
            self.refcounts[id] -= 1;
            if self.refcounts[id] == 0 {
                self.free_list.push(id);
                return true;
            }
        }
        false
    }

    pub fn retain(&mut self, id: usize) {
        if id < self.refcounts.len() {
            self.refcounts[id] += 1;
        }
    }
}
