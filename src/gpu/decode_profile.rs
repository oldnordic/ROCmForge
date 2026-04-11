//! Decode-stage profiling helpers for the GPU forward path.
//!
//! Keeping the profiling state machine out of `forward.rs` reduces coupling
//! between instrumentation and the actual decode logic.

use super::device::GpuDevice;
use super::error::GpuResult;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub(crate) const PROFILE_DECODE_STAGES_ENV: &str = "ROCMFORGE_PROFILE_DECODE_STAGES";
const ENV_UNKNOWN: u8 = 0;
const ENV_DISABLED: u8 = 1;
const ENV_ENABLED: u8 = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuDecodeStageProfileSnapshot {
    pub layer_invocations: u64,
    pub tail_invocations: u64,
    pub attn_norm_ns: u128,
    pub qkv_ns: u128,
    pub q_rope_ns: u128,
    pub k_rope_ns: u128,
    pub kv_write_ns: u128,
    pub attention_ns: u128,
    pub attn_proj_ns: u128,
    pub attn_residual_ns: u128,
    pub ffn_norm_ns: u128,
    pub gate_up_ns: u128,
    pub ffn_down_ns: u128,
    pub ffn_residual_ns: u128,
    pub logits_norm_ns: u128,
    pub logits_proj_ns: u128,
    pub argmax_ns: u128,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DecodeStage {
    AttnNorm,
    Qkv,
    QRope,
    KRope,
    KvWrite,
    Attention,
    AttnProj,
    AttnResidual,
    FfnNorm,
    GateUp,
    FfnDown,
    FfnResidual,
    LogitsNorm,
    LogitsProj,
    Argmax,
}

fn decode_stage_profile_store() -> &'static Mutex<GpuDecodeStageProfileSnapshot> {
    static STORE: OnceLock<Mutex<GpuDecodeStageProfileSnapshot>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(GpuDecodeStageProfileSnapshot::default()))
}

static PROFILE_DECODE_STAGES_FLAG: AtomicU8 = AtomicU8::new(ENV_UNKNOWN);

pub(crate) fn decode_stage_profiling_enabled() -> bool {
    match PROFILE_DECODE_STAGES_FLAG.load(Ordering::Relaxed) {
        ENV_DISABLED => false,
        ENV_ENABLED => true,
        _ => {
            let enabled = std::env::var_os(PROFILE_DECODE_STAGES_ENV).is_some();
            PROFILE_DECODE_STAGES_FLAG.store(
                if enabled { ENV_ENABLED } else { ENV_DISABLED },
                Ordering::Relaxed,
            );
            enabled
        }
    }
}

pub(crate) fn refresh_decode_profile_env_flag() {
    PROFILE_DECODE_STAGES_FLAG.store(ENV_UNKNOWN, Ordering::Relaxed);
}

fn record_decode_stage(stage: DecodeStage, elapsed_ns: u128) {
    let mut guard = decode_stage_profile_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    match stage {
        DecodeStage::AttnNorm => guard.attn_norm_ns += elapsed_ns,
        DecodeStage::Qkv => guard.qkv_ns += elapsed_ns,
        DecodeStage::QRope => guard.q_rope_ns += elapsed_ns,
        DecodeStage::KRope => guard.k_rope_ns += elapsed_ns,
        DecodeStage::KvWrite => guard.kv_write_ns += elapsed_ns,
        DecodeStage::Attention => guard.attention_ns += elapsed_ns,
        DecodeStage::AttnProj => guard.attn_proj_ns += elapsed_ns,
        DecodeStage::AttnResidual => guard.attn_residual_ns += elapsed_ns,
        DecodeStage::FfnNorm => guard.ffn_norm_ns += elapsed_ns,
        DecodeStage::GateUp => guard.gate_up_ns += elapsed_ns,
        DecodeStage::FfnDown => guard.ffn_down_ns += elapsed_ns,
        DecodeStage::FfnResidual => guard.ffn_residual_ns += elapsed_ns,
        DecodeStage::LogitsNorm => guard.logits_norm_ns += elapsed_ns,
        DecodeStage::LogitsProj => guard.logits_proj_ns += elapsed_ns,
        DecodeStage::Argmax => guard.argmax_ns += elapsed_ns,
    }
}

pub(crate) fn record_tail_invocation() {
    let mut guard = decode_stage_profile_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.tail_invocations += 1;
}

pub(crate) fn record_layer_invocation() {
    let mut guard = decode_stage_profile_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.layer_invocations += 1;
}

pub(crate) fn profile_decode_stage<T>(
    device: &GpuDevice,
    stage: DecodeStage,
    op: impl FnOnce() -> GpuResult<T>,
) -> GpuResult<T> {
    if !decode_stage_profiling_enabled() {
        return op();
    }

    let start = Instant::now();
    let result = op()?;
    device.synchronize()?;
    record_decode_stage(stage, start.elapsed().as_nanos());
    Ok(result)
}

pub fn reset_decode_stage_profile() {
    let mut guard = decode_stage_profile_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    *guard = GpuDecodeStageProfileSnapshot::default();
}

pub fn decode_stage_profile_snapshot() -> GpuDecodeStageProfileSnapshot {
    *decode_stage_profile_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}
