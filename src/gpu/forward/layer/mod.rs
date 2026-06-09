mod decode;
mod moe;
mod ssm;

// Private re-imports make cross-sibling functions available to all layer/* children.
use moe::gpu_dispatch_moe_ffn_on_stream;
use ssm::gpu_layer_forward_ssm_on_stream;

pub use decode::gpu_layer_forward_hybrid;
pub(in crate::gpu::forward) use decode::{
    gpu_attention_decode, gpu_attention_decode_from_state, gpu_layer_forward_from_state_on_stream,
};
