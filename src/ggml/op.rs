//! Supported ggml operations.

#[derive(Debug, Clone)]
pub enum Op {
    GetRows,
    MatMul,
    Add,
    Mask,
    Scale { factor: f32 },
    LayerNorm { eps: f32 },
    RmsNorm { eps: f32 },
    Rope,
    Softmax,
    Attention,
    SwiGlu,
    MlpSwiglu,
    SplitQkv,
    Reshape,
    View,
    Copy,
}
