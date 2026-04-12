//! Runnable binaries that exercise fuel's Phase 6a lazy graph layer.
//!
//! This crate exists as a sibling of `fuel-examples` specifically to
//! avoid dragging in the eager-path model code in `fuel-transformers`
//! and `fuel-nn`. Every binary in `src/bin/` depends only on
//! `fuel-core` (aliased as `fuel`), so builds here are isolated from
//! any work-in-progress in the broader workspace.
//!
//! # Binaries
//!
//! - `llama-lazy` — end-to-end LLaMA-family inference through the
//!   lazy graph + fast CPU executor. Defaults to TinyLlama-1.1B so
//!   no HuggingFace authentication is required. Pass
//!   `meta-llama/Meta-Llama-3-8B` as the first arg to run Llama 3
//!   (gated; requires `HF_TOKEN`).
