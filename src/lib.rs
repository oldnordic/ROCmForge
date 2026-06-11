#![allow(
    dead_code,
    unused_variables,
    unused_imports,
    unused_mut,
    unused_unsafe,
    unused_must_use,
    clippy::all
)]

pub mod bench;
pub mod config;
pub mod cpu;
pub mod error;
pub mod hardware;
pub mod loader;
pub mod tokenizer;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "server")]
pub mod api;
