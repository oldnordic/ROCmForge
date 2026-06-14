//! GeoGraph-backed CPU execution engine.

use crate::cpu::weights::WeightMeta;
use crate::cpu::CpuError;
use crate::loader::GgmlType;

#[cfg(feature = "cpu-graph")]
pub use geographdb_core::algorithms::four_d::{GraphNode4D, TemporalEdge, TemporalWindow};

use std::collections::HashMap;

/// Stable handle to a contiguous f32 tensor inside a `CpuGraphArena`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct F32Handle {
    pub offset: usize,
    pub len: usize,
}

/// Stable handle to a contiguous u8 tensor inside a `CpuGraphArena`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct U8Handle {
    pub offset: usize,
    pub len: usize,
}

/// Stable storage arena for all tensor data referenced by a captured CPU graph.
///
/// Instead of recording raw pointer addresses that can become invalid between
/// capture and replay, every captured op stores handles (offsets) into this
/// arena.  The arena owns the bytes; handles remain valid as long as the arena
/// itself is alive.
pub struct CpuGraphArena {
    f32_data: Vec<f32>,
    u8_data: Vec<u8>,
    /// Maps the original caller pointer to the current f32 handle for that slice.
    /// Used by `read_back` to copy replay results back to caller-owned buffers.
    f32_bindings: HashMap<usize, F32Handle>,
    /// Same, for u8 slices.
    u8_bindings: HashMap<usize, U8Handle>,
}

impl CpuGraphArena {
    pub fn new() -> Self {
        Self {
            f32_data: Vec::new(),
            u8_data: Vec::new(),
            f32_bindings: HashMap::new(),
            u8_bindings: HashMap::new(),
        }
    }

    /// Allocate a new, zero-initialized f32 slot.
    pub fn alloc_f32(&mut self, len: usize) -> F32Handle {
        let offset = self.f32_data.len();
        self.f32_data.resize(offset + len, 0.0f32);
        F32Handle { offset, len }
    }

    /// Copy an f32 slice into the arena and return its handle.
    pub fn copy_f32(&mut self, src: &[f32]) -> F32Handle {
        let handle = self.alloc_f32(src.len());
        self.f32_data[handle.offset..handle.offset + handle.len].copy_from_slice(src);
        handle
    }

    /// Bind a caller-owned f32 slice to an arena slot.  The slot is initialized
    /// from the current slice contents and registered for `read_back`.
    pub fn bind_f32(&mut self, ptr: usize, src: &[f32]) -> F32Handle {
        let handle = self.copy_f32(src);
        self.f32_bindings.insert(ptr, handle);
        handle
    }

    /// Rebind an already-allocated f32 handle to a caller-owned slice for read-back.
    pub fn rebind_f32(&mut self, ptr: usize, handle: F32Handle) {
        self.f32_bindings.insert(ptr, handle);
    }

    pub fn f32(&self, handle: F32Handle) -> &[f32] {
        &self.f32_data[handle.offset..handle.offset + handle.len]
    }

    pub fn f32_mut(&mut self, handle: F32Handle) -> &mut [f32] {
        &mut self.f32_data[handle.offset..handle.offset + handle.len]
    }

    /// Allocate a new, zero-initialized u8 slot.
    pub fn alloc_u8(&mut self, len: usize) -> U8Handle {
        let offset = self.u8_data.len();
        self.u8_data.resize(offset + len, 0u8);
        U8Handle { offset, len }
    }

    /// Copy a u8 slice into the arena and return its handle.
    pub fn copy_u8(&mut self, src: &[u8]) -> U8Handle {
        let handle = self.alloc_u8(src.len());
        self.u8_data[handle.offset..handle.offset + handle.len].copy_from_slice(src);
        handle
    }

    /// Bind a caller-owned u8 slice to an arena slot.
    pub fn bind_u8(&mut self, ptr: usize, src: &[u8]) -> U8Handle {
        let handle = self.copy_u8(src);
        self.u8_bindings.insert(ptr, handle);
        handle
    }

    pub fn u8(&self, handle: U8Handle) -> &[u8] {
        &self.u8_data[handle.offset..handle.offset + handle.len]
    }

    pub fn u8_mut(&mut self, handle: U8Handle) -> &mut [u8] {
        &mut self.u8_data[handle.offset..handle.offset + handle.len]
    }

    /// Copy all current bindings back to the original caller slices.
    ///
    /// # Safety
    /// The original slices must still be alive and have the same length as when
    /// they were bound.  After this call the caller-owned buffers reflect the
    /// final replay state.
    pub unsafe fn read_back(&self) {
        for (&ptr, &handle) in &self.f32_bindings {
            let src = self.f32(handle);
            let dst = std::slice::from_raw_parts_mut(ptr as *mut f32, handle.len);
            dst.copy_from_slice(src);
        }
        for (&ptr, &handle) in &self.u8_bindings {
            let src = self.u8(handle);
            let dst = std::slice::from_raw_parts_mut(ptr as *mut u8, handle.len);
            dst.copy_from_slice(src);
        }
    }
}

impl Default for CpuGraphArena {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU operator node types.
///
/// Each tensor is addressed by a stable handle into a `CpuGraphArena` rather
/// than a raw pointer, so the graph remains valid after the original buffers
/// are moved or dropped.
#[derive(Debug, Clone)]
pub enum CpuOpNode {
    RmsNorm {
        hidden: F32Handle,
        weight: F32Handle,
        out: F32Handle,
        n: usize,
        eps: f32,
    },
    Gemv {
        weight: U8Handle,
        weight_bytes: usize,
        input: F32Handle,
        out: F32Handle,
        scratch: Option<U8Handle>,
        m: usize,
        n: usize,
        wtype: GgmlType,
        needs_transpose: bool,
    },
    RoPE {
        x: F32Handle,
        sin: F32Handle,
        cos: F32Handle,
        out: F32Handle,
        n_heads: usize,
        head_dim: usize,
        neox: bool,
    },
    Attention {
        q: F32Handle,
        k: F32Handle,
        v: F32Handle,
        out: F32Handle,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    },
    SiLU {
        gate: F32Handle,
        up: F32Handle,
        out: F32Handle,
        h: usize,
    },
    ResidualAdd {
        a: F32Handle,
        b: F32Handle,
        out: F32Handle,
        h: usize,
    },
}

#[cfg(feature = "cpu-graph")]
pub struct CpuGraph {
    /// Temporal graph nodes.  Exposed so tests and tooling can inspect timestamps.
    pub nodes: Vec<GraphNode4D>,
    ops: Vec<CpuOpNode>,
}

/// Abstract context for executing CPU operations.
/// Allows same forward code to be used for direct execution or graph capture.
#[cfg(test)]
pub mod tests;

pub trait CpuExecutionContext {
    #[allow(clippy::too_many_arguments)]
    fn execute_gemv(
        &mut self,
        w: &[u8],
        meta: &WeightMeta,
        x: &[f32],
        y: &mut [f32],
        out_dim: usize,
        in_dim: usize,
        q8_scratch: Option<&mut [u8]>,
    ) -> Result<(), CpuError>;

    fn execute_rms_norm(&mut self, x: &[f32], w: &[f32], out: &mut [f32], eps: f32);

    fn execute_rope(
        &mut self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        sin: &[f32],
        cos: &[f32],
        neox: bool,
    );

    #[allow(clippy::too_many_arguments)]
    fn execute_attention(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    );

    fn execute_silu(&mut self, gate: &[f32], up: &mut [f32]);

    fn execute_residual_add(&mut self, a: &mut [f32], b: &[f32]);
}

/// Direct execution context that runs kernels immediately.
pub struct DirectContext;

impl CpuExecutionContext for DirectContext {
    #[allow(clippy::too_many_arguments)]
    fn execute_gemv(
        &mut self,
        w: &[u8],
        meta: &WeightMeta,
        x: &[f32],
        y: &mut [f32],
        out_dim: usize,
        in_dim: usize,
        q8_scratch: Option<&mut [u8]>,
    ) -> Result<(), CpuError> {
        crate::cpu::ops::dispatch_gemv(w, meta, x, y, out_dim, in_dim, q8_scratch)
    }

    fn execute_rms_norm(&mut self, x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
        crate::cpu::ops::rms_norm(x, w, out, eps);
    }

    fn execute_rope(
        &mut self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        sin: &[f32],
        cos: &[f32],
        neox: bool,
    ) {
        crate::cpu::ops::rope(x, n_heads, head_dim, sin, cos, neox);
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_attention(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        _max_seq_len: usize,
    ) {
        crate::cpu::ops::flash_attn_decode(
            q,
            k,
            v,
            out,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        );
    }

    fn execute_silu(&mut self, gate: &[f32], up: &mut [f32]) {
        crate::cpu::ops::silu_fuse(gate, up);
    }

    fn execute_residual_add(&mut self, a: &mut [f32], b: &[f32]) {
        crate::cpu::ops::residual_add(a, b);
    }
}

#[cfg(feature = "cpu-graph")]
pub struct CaptureContext {
    pub graph: CpuGraph,
    pub arena: CpuGraphArena,
    pub layer: usize,
    pub step: usize,
    pub timestamp: u64,
    /// History of caller-pointer -> output-handle bindings, annotated with the
    /// timestamp at which they became valid.  Used by `rebind_after_regress` to
    /// restore the arena bindings that correspond to a rolled-back temporal
    /// state.
    pub output_log: Vec<(u64, usize, F32Handle)>,
}

#[cfg(feature = "cpu-graph")]
impl CaptureContext {
    /// Convenience constructor.
    pub fn new(layer: usize, timestamp: u64) -> Self {
        Self {
            graph: CpuGraph::new(),
            arena: CpuGraphArena::new(),
            layer,
            step: 0,
            timestamp,
            output_log: Vec::new(),
        }
    }

    /// Copy all replay results back to the caller-owned slices that were bound
    /// during capture.
    ///
    /// # Safety
    /// The caller must ensure all bound slices are still alive with their
    /// original lengths.
    pub unsafe fn read_back(&self) {
        self.arena.read_back();
    }

    /// Restore the arena's caller-pointer bindings to the state that existed
    /// just after `timestamp`.  This must be called after `graph.regress` so
    /// that `read_back` reflects the rolled-back computation rather than the
    /// most-recently-captured branch.
    pub fn rebind_after_regress(&mut self, timestamp: u64) {
        let mut surviving: std::collections::HashMap<usize, F32Handle> =
            std::collections::HashMap::new();
        for (ts, ptr, handle) in &self.output_log {
            if *ts <= timestamp {
                surviving.insert(*ptr, *handle);
            }
        }
        for (ptr, handle) in surviving {
            self.arena.rebind_f32(ptr, handle);
        }
    }
}

#[cfg(feature = "cpu-graph")]
impl CpuExecutionContext for CaptureContext {
    #[allow(clippy::too_many_arguments)]
    fn execute_gemv(
        &mut self,
        w: &[u8],
        meta: &WeightMeta,
        x: &[f32],
        y: &mut [f32],
        out_dim: usize,
        in_dim: usize,
        mut q8_scratch: Option<&mut [u8]>,
    ) -> Result<(), CpuError> {
        let weight = self.arena.copy_u8(w);
        let input = self.arena.copy_f32(x);
        let out = self.arena.alloc_f32(y.len());
        self.arena.rebind_f32(y.as_ptr() as usize, out);
        self.output_log
            .push((self.timestamp, y.as_ptr() as usize, out));
        let scratch = q8_scratch.as_ref().map(|s| self.arena.copy_u8(s));

        let op = CpuOpNode::Gemv {
            weight,
            weight_bytes: w.len(),
            input,
            out,
            scratch,
            m: out_dim,
            n: in_dim,
            wtype: meta.wtype,
            needs_transpose: meta.needs_transpose,
        };
        self.graph
            .add_node(op, self.layer, self.step, self.timestamp);
        self.step += 1;
        let scratch_borrow = q8_scratch.as_deref_mut();
        crate::cpu::ops::dispatch_gemv(w, meta, x, y, out_dim, in_dim, scratch_borrow)?;
        self.arena.f32_mut(out).copy_from_slice(y);
        if let (Some(h), Some(s)) = (scratch, q8_scratch.as_mut()) {
            self.arena.u8_mut(h).copy_from_slice(s);
        }
        Ok(())
    }

    fn execute_rms_norm(&mut self, x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
        let hidden = self.arena.copy_f32(x);
        let weight = self.arena.copy_f32(w);
        let h_out = self.arena.alloc_f32(out.len());
        self.arena.rebind_f32(out.as_ptr() as usize, h_out);
        self.output_log
            .push((self.timestamp, out.as_ptr() as usize, h_out));

        let op = CpuOpNode::RmsNorm {
            hidden,
            weight,
            out: h_out,
            n: x.len(),
            eps,
        };
        self.graph
            .add_node(op, self.layer, self.step, self.timestamp);
        self.step += 1;
        crate::cpu::ops::rms_norm(x, w, out, eps);
        self.arena.f32_mut(h_out).copy_from_slice(out);
    }

    fn execute_rope(
        &mut self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        sin: &[f32],
        cos: &[f32],
        neox: bool,
    ) {
        let x_in = self.arena.copy_f32(x);
        let sin_h = self.arena.copy_f32(sin);
        let cos_h = self.arena.copy_f32(cos);
        let h_out = self.arena.alloc_f32(x.len());
        self.arena.rebind_f32(x.as_ptr() as usize, h_out);
        self.output_log
            .push((self.timestamp, x.as_ptr() as usize, h_out));

        let op = CpuOpNode::RoPE {
            x: x_in,
            sin: sin_h,
            cos: cos_h,
            out: h_out,
            n_heads,
            head_dim,
            neox,
        };
        self.graph
            .add_node(op, self.layer, self.step, self.timestamp);
        self.step += 1;
        crate::cpu::ops::rope(x, n_heads, head_dim, sin, cos, neox);
        self.arena.f32_mut(h_out).copy_from_slice(x);
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_attention(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) {
        let q_h = self.arena.copy_f32(q);
        let k_h = self.arena.copy_f32(k);
        let v_h = self.arena.copy_f32(v);
        let h_out = self.arena.alloc_f32(out.len());
        self.arena.rebind_f32(out.as_ptr() as usize, h_out);
        self.output_log
            .push((self.timestamp, out.as_ptr() as usize, h_out));

        let op = CpuOpNode::Attention {
            q: q_h,
            k: k_h,
            v: v_h,
            out: h_out,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
        };
        self.graph
            .add_node(op, self.layer, self.step, self.timestamp);
        self.step += 1;
        crate::cpu::ops::flash_attn_decode(
            q,
            k,
            v,
            out,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        );
        self.arena.f32_mut(h_out).copy_from_slice(out);
    }

    fn execute_silu(&mut self, gate: &[f32], up: &mut [f32]) {
        let gate_h = self.arena.copy_f32(gate);
        let up_in = self.arena.copy_f32(up);
        let h_out = self.arena.alloc_f32(up.len());
        self.arena.rebind_f32(up.as_ptr() as usize, h_out);
        self.output_log
            .push((self.timestamp, up.as_ptr() as usize, h_out));

        let op = CpuOpNode::SiLU {
            gate: gate_h,
            up: up_in,
            out: h_out,
            h: gate.len(),
        };
        self.graph
            .add_node(op, self.layer, self.step, self.timestamp);
        self.step += 1;
        crate::cpu::ops::silu_fuse(gate, up);
        self.arena.f32_mut(h_out).copy_from_slice(up);
    }

    fn execute_residual_add(&mut self, a: &mut [f32], b: &[f32]) {
        let a_in = self.arena.copy_f32(a);
        let b_h = self.arena.copy_f32(b);
        let h_out = self.arena.alloc_f32(a.len());
        self.arena.rebind_f32(a.as_ptr() as usize, h_out);
        self.output_log
            .push((self.timestamp, a.as_ptr() as usize, h_out));

        let op = CpuOpNode::ResidualAdd {
            a: a_in,
            b: b_h,
            out: h_out,
            h: a.len(),
        };
        self.graph
            .add_node(op, self.layer, self.step, self.timestamp);
        self.step += 1;
        crate::cpu::ops::residual_add(a, b);
        self.arena.f32_mut(h_out).copy_from_slice(a);
    }
}

#[cfg(feature = "cpu-graph")]
impl CpuGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            ops: Vec::new(),
        }
    }

    pub fn add_node(&mut self, op: CpuOpNode, layer: usize, step: usize, timestamp: u64) {
        let id = self.ops.len() as u64;
        self.ops.push(op);

        let node = GraphNode4D {
            id,
            x: layer as f32,
            y: step as f32,
            z: 0.0,
            begin_ts: timestamp,
            end_ts: 0,
            properties: std::collections::BTreeMap::new(),
            successors: Vec::new(),
        };
        self.nodes.push(node);
    }

    /// Execute nodes valid within the given temporal window.
    pub fn execute_window(
        &self,
        arena: &mut CpuGraphArena,
        window: TemporalWindow,
    ) -> Result<(), CpuError> {
        let mut active_nodes: Vec<&GraphNode4D> = self
            .nodes
            .iter()
            .filter(|n| window.overlaps(n.begin_ts, n.end_ts))
            .collect();

        // Sort by time first, then spatial coordinates (Layer -> Step)
        active_nodes.sort_by(|a, b| {
            a.begin_ts
                .cmp(&b.begin_ts)
                .then_with(|| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
        });

        for node in active_nodes {
            let op = &self.ops[node.id as usize];
            self.execute_op(op, arena)?;
        }
        Ok(())
    }

    fn execute_op(&self, op: &CpuOpNode, arena: &mut CpuGraphArena) -> Result<(), CpuError> {
        // Use raw base pointers so we can form slices for multiple disjoint
        // handles without fighting the borrow checker.  All handles are
        // non-overlapping offsets allocated by the arena, so this is sound.
        let f32_base = arena.f32_data.as_mut_ptr();
        let u8_base = arena.u8_data.as_mut_ptr();

        let f32_slice =
            |h: F32Handle| unsafe { std::slice::from_raw_parts(f32_base.add(h.offset), h.len) };
        let f32_slice_mut =
            |h: F32Handle| unsafe { std::slice::from_raw_parts_mut(f32_base.add(h.offset), h.len) };
        let u8_slice =
            |h: U8Handle| unsafe { std::slice::from_raw_parts(u8_base.add(h.offset), h.len) };
        let u8_slice_mut =
            |h: U8Handle| unsafe { std::slice::from_raw_parts_mut(u8_base.add(h.offset), h.len) };

        match op {
            CpuOpNode::RmsNorm {
                hidden,
                weight,
                out,
                eps,
                n: _,
            } => {
                let hidden = f32_slice(*hidden);
                let weight = f32_slice(*weight);
                let out = f32_slice_mut(*out);
                crate::cpu::ops::rms_norm(hidden, weight, out, *eps);
            }
            CpuOpNode::Gemv {
                weight,
                input,
                out,
                scratch,
                m,
                n,
                wtype,
                needs_transpose,
                weight_bytes: _,
            } => {
                let w = u8_slice(*weight);
                let x = f32_slice(*input);
                let y = f32_slice_mut(*out);
                let mut q8_scratch = scratch.map(|h| u8_slice_mut(h));
                let meta = WeightMeta {
                    wtype: *wtype,
                    dims: vec![*m as u64, *n as u64],
                    needs_transpose: *needs_transpose,
                    role: crate::config::TensorRole::Generic,
                    svd_k: None,
                };
                crate::cpu::ops::dispatch_gemv(w, &meta, x, y, *m, *n, q8_scratch.as_deref_mut())?;
            }
            CpuOpNode::RoPE {
                x,
                sin,
                cos,
                out,
                n_heads,
                head_dim,
                neox,
            } => {
                let x_out = f32_slice_mut(*out);
                // RoPE is an in-place rotation; copy the source values into the
                // output slot first so the kernel can read and write the same buffer.
                // If x and out share a handle this is a no-op.
                if *x != *out {
                    x_out.copy_from_slice(f32_slice(*x));
                }
                let sin = f32_slice(*sin);
                let cos = f32_slice(*cos);
                crate::cpu::ops::rope(x_out, *n_heads, *head_dim, sin, cos, *neox);
            }
            CpuOpNode::Attention {
                q,
                k,
                v,
                out,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                max_seq_len: _,
            } => {
                let q = f32_slice(*q);
                let k = f32_slice(*k);
                let v = f32_slice(*v);
                let out = f32_slice_mut(*out);
                crate::cpu::ops::flash_attn_decode(
                    q,
                    k,
                    v,
                    out,
                    *seq_len,
                    *num_heads,
                    *num_kv_heads,
                    *head_dim,
                );
            }
            CpuOpNode::SiLU {
                gate,
                up,
                out,
                h: _,
            } => {
                let gate = f32_slice(*gate);
                let up_in = f32_slice(*up);
                let out_handle = *out;
                let out = f32_slice_mut(out_handle);
                if *up != out_handle {
                    out.copy_from_slice(up_in);
                }
                crate::cpu::ops::silu_fuse(gate, out);
            }
            CpuOpNode::ResidualAdd { a, b, out, h: _ } => {
                let a_in = f32_slice(*a);
                let out_handle = *out;
                let out = f32_slice_mut(out_handle);
                if *a != out_handle {
                    out.copy_from_slice(a_in);
                }
                let b = f32_slice(*b);
                crate::cpu::ops::residual_add(out, b);
            }
        }
        Ok(())
    }

    /// Invalidate all nodes captured after the given timestamp.
    pub fn regress(&mut self, timestamp: u64) {
        for node in self.nodes.iter_mut() {
            if node.begin_ts > timestamp {
                // Fully invalidate nodes that started after the regression point
                node.end_ts = node.begin_ts;
            } else if node.end_ts == 0 || node.end_ts > timestamp {
                // Cap nodes that were active across the regression point
                node.end_ts = timestamp;
            }
        }
    }
}
