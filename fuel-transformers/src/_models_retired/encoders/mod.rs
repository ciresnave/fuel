//! Encoder-only model implementations.
//!
//! Bidirectional transformer encoders for text embeddings, retrieval,
//! classification, and other representation-learning tasks.

pub use super::common::with_tracing;

pub mod bert;
pub mod debertav2;
pub mod distilbert;
pub mod jina_bert;
pub mod modernbert;
pub mod nomic_bert;
pub mod nvembed_v2;
pub mod stella_en_v5;
pub mod xlm_roberta;
