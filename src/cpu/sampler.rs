//! CPU sampling functions.
//!
//! Greedy and top-p sampling.

pub fn cpu_sample_greedy(_logits: &[f32]) -> u32 {
    todo!("implement cpu_sample_greedy")
}

pub fn cpu_sample_top_p(_logits: &[f32], _temperature: f32, _top_p: f32, _seed: u64) -> u32{
    todo!("implement cpu_sample_top_p")
}
