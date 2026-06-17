//! Structured forward-pass graph tracer for CPU inference.
//!
//! Emits a JSONL file with node and edge records that
//! `geographdb-core/examples/plot_forward_graph.py` can render.

use serde::Serialize;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Result, Write};
use std::path::Path;

/// Components that can appear as nodes in the forward graph.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceComponent {
    InputEmbedding,
    Query,
    Key,
    Value,
    AttentionOutput,
    MlpHidden,
    Logits,
    Confidence,
}

/// Edge kinds connecting forward-graph nodes.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceEdgeKind {
    Attention,
}

#[derive(Clone, Debug)]
struct TraceNode {
    component: TraceComponent,
    layer: usize,
    position: Option<usize>,
    token_id: u32,
    values: Vec<f32>,
}

#[derive(Clone, Debug)]
struct TraceEdge {
    kind: TraceEdgeKind,
    src_component: TraceComponent,
    src_layer: usize,
    src_position: Option<usize>,
    src_token_id: u32,
    dst_component: TraceComponent,
    dst_layer: usize,
    dst_position: Option<usize>,
    dst_token_id: u32,
    weight: f32,
    head_id: Option<usize>,
}

/// Collects a forward-pass graph and writes it as JSONL.
#[derive(Debug)]
pub struct ForwardGraphRecorder {
    token_ids: Vec<u32>,
    nodes: Vec<TraceNode>,
    edges: Vec<TraceEdge>,
    emitted_kv: HashSet<(usize, usize)>,
    max_context: usize,
    max_value_len: usize,
    attention_threshold: f32,
    meta_expected_attention: Option<serde_json::Value>,
    last_confidence: Option<f32>,
}

impl ForwardGraphRecorder {
    /// Create a recorder seeded with the prompt token IDs.
    pub fn new(prompt_tokens: &[u32]) -> Self {
        Self {
            token_ids: prompt_tokens.to_vec(),
            nodes: Vec::new(),
            edges: Vec::new(),
            emitted_kv: HashSet::new(),
            max_context: 512,
            max_value_len: 128,
            attention_threshold: 0.01,
            meta_expected_attention: None,
            last_confidence: None,
        }
    }

    /// Store the expected-attention metadata that the plotter will use to
    /// highlight positions.
    pub fn set_expected_attention(&mut self, value: serde_json::Value) {
        self.meta_expected_attention = Some(value);
    }

    /// Number of tokens currently in the traced context.
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    /// Append the next generated token to the traced context.
    pub fn push_token(&mut self, token_id: u32) {
        self.token_ids.push(token_id);
    }

    fn token_id_at(&self, pos: usize) -> u32 {
        self.token_ids.get(pos).copied().unwrap_or(0)
    }

    /// Record an arbitrary node. `values` is cloned because callers reuse
    /// scratch buffers.
    pub fn record_node(
        &mut self,
        component: TraceComponent,
        layer: usize,
        position: Option<usize>,
        values: &[f32],
    ) {
        let token_id = position.map(|p| self.token_id_at(p)).unwrap_or(0);
        let values: Vec<f32> = values.iter().take(self.max_value_len).copied().collect();
        self.nodes.push(TraceNode {
            component,
            layer,
            position,
            token_id,
            values,
        });
    }

    /// Ensure key/value nodes exist for the given layer and position.
    /// Does nothing if they were already emitted.
    pub fn ensure_kv_nodes(
        &mut self,
        layer: usize,
        pos: usize,
        k_values: &[f32],
        v_values: &[f32],
    ) {
        if self.emitted_kv.insert((layer, pos)) {
            self.record_node(TraceComponent::Key, layer, Some(pos), k_values);
            self.record_node(TraceComponent::Value, layer, Some(pos), v_values);
        }
    }

    /// Record a single attention edge from the query position to a key/value
    /// position. The edge is annotated with the head id when per-head tracing is
    /// active.
    pub fn record_attention_edge(
        &mut self,
        head_id: Option<usize>,
        layer: usize,
        src_pos: usize,
        dst_pos: usize,
        weight: f32,
    ) {
        let src_token_id = self.token_id_at(src_pos);
        let dst_token_id = self.token_id_at(dst_pos);
        self.edges.push(TraceEdge {
            kind: TraceEdgeKind::Attention,
            src_component: TraceComponent::Query,
            src_layer: layer,
            src_position: Some(src_pos),
            src_token_id,
            dst_component: TraceComponent::Value,
            dst_layer: layer,
            dst_position: Some(dst_pos),
            dst_token_id,
            weight,
            head_id,
        });
    }

    /// Return the current context length clamped to the configured maximum.
    pub fn effective_seq_len(&self, requested: usize) -> usize {
        requested.min(self.max_context)
    }

    /// Threshold below which attention edges are dropped.
    pub fn attention_threshold(&self) -> f32 {
        self.attention_threshold
    }

    /// Record a confidence readout derived from `logits` for `token_id`.
    pub fn record_confidence(&mut self, position: usize, token_id: u32, logits: &[f32]) {
        if logits.is_empty() {
            return;
        }
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut exp_values = vec![0.0f32; logits.len()];
        for (i, &logit) in logits.iter().enumerate() {
            let e = (logit - max_logit).exp();
            exp_values[i] = e;
            sum += e;
        }
        let prob = if sum > 0.0 && (token_id as usize) < exp_values.len() {
            exp_values[token_id as usize] / sum
        } else {
            0.0
        };
        self.last_confidence = Some(prob);
        self.nodes.push(TraceNode {
            component: TraceComponent::Confidence,
            layer: 0,
            position: Some(position),
            token_id,
            values: vec![prob],
        });
    }

    /// Write the collected trace to a JSONL file.
    pub fn write_jsonl(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        #[derive(Serialize)]
        #[serde(tag = "record")]
        enum JsonRecord<'a> {
            #[serde(rename = "meta")]
            Meta {
                #[serde(skip_serializing_if = "Option::is_none")]
                expected_attention: Option<&'a serde_json::Value>,
                token_ids: &'a [u32],
                #[serde(skip_serializing_if = "Option::is_none")]
                predicted_token: Option<u32>,
                #[serde(skip_serializing_if = "Option::is_none")]
                confidence: Option<f32>,
            },
            #[serde(rename = "node")]
            Node {
                component: TraceComponent,
                layer: usize,
                position: Option<usize>,
                token_id: u32,
                values: &'a [f32],
            },
            #[serde(rename = "edge")]
            Edge {
                kind: TraceEdgeKind,
                src_component: TraceComponent,
                src_layer: usize,
                src_position: Option<usize>,
                src_token_id: u32,
                dst_component: TraceComponent,
                dst_layer: usize,
                dst_position: Option<usize>,
                dst_token_id: u32,
                weight: f32,
                #[serde(skip_serializing_if = "Option::is_none")]
                head_id: Option<usize>,
            },
        }

        let predicted_token = self
            .nodes
            .iter()
            .rfind(|n| matches!(n.component, TraceComponent::Confidence))
            .map(|n| n.token_id);

        serde_json::to_writer(
            &mut writer,
            &JsonRecord::Meta {
                expected_attention: self.meta_expected_attention.as_ref(),
                token_ids: &self.token_ids,
                predicted_token,
                confidence: self.last_confidence,
            },
        )?;
        writeln!(writer)?;

        for node in &self.nodes {
            serde_json::to_writer(
                &mut writer,
                &JsonRecord::Node {
                    component: node.component,
                    layer: node.layer,
                    position: node.position,
                    token_id: node.token_id,
                    values: &node.values,
                },
            )?;
            writeln!(writer)?;
        }

        for edge in &self.edges {
            serde_json::to_writer(
                &mut writer,
                &JsonRecord::Edge {
                    kind: edge.kind,
                    src_component: edge.src_component,
                    src_layer: edge.src_layer,
                    src_position: edge.src_position,
                    src_token_id: edge.src_token_id,
                    dst_component: edge.dst_component,
                    dst_layer: edge.dst_layer,
                    dst_position: edge.dst_position,
                    dst_token_id: edge.dst_token_id,
                    weight: edge.weight,
                    head_id: edge.head_id,
                },
            )?;
            writeln!(writer)?;
        }

        writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_records_nodes_and_edges() {
        let mut r = ForwardGraphRecorder::new(&[1, 2, 3]);
        r.record_node(TraceComponent::InputEmbedding, 0, Some(0), &[0.1, -0.2]);
        r.record_attention_edge(Some(0), 0, 2, 0, 0.75);
        r.write_jsonl(std::env::temp_dir().join("rocmforge_trace_test.jsonl"))
            .expect("write trace jsonl");
    }
}
