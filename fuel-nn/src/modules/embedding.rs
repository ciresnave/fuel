// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lazy `Embedding` layer — row lookup against a `[vocab_size, hidden]`
//! weight table.
//!
//! The weight table is held as an `Arc<[f32]>` and emitted as a Const
//! node anchored on the token-id graph at forward time. The forward
//! accepts a U32 index tensor of any rank, flattens it, runs an
//! `index_select` along dim 0, and reshapes the result to append a
//! trailing `hidden` dim — matching the eager `fuel-nn::Embedding`
//! semantics.

use crate::modules::Module;
use fuel_core::Result;
use fuel_core::lazy::Tensor;
use fuel_ir::Shape;
use std::sync::Arc;

/// Lookup-table embedding over `Tensor`.
#[derive(Debug, Clone)]
pub struct Embedding {
    table: Arc<[f32]>,
    vocab_size: usize,
    hidden: usize,
}

impl Embedding {
    /// Build an embedding from a `[vocab_size, hidden]` weight buffer.
    pub fn new(table: Arc<[f32]>, vocab_size: usize, hidden: usize) -> Result<Self> {
        if table.len() != vocab_size * hidden {
            return Err(fuel_core::Error::Msg(format!(
                "Embedding::new: table has {} elements but \
                 vocab_size * hidden = {} * {} = {}",
                table.len(),
                vocab_size,
                hidden,
                vocab_size * hidden,
            ))
            .bt());
        }
        Ok(Self {
            table,
            vocab_size,
            hidden,
        })
    }

    /// Vocabulary size (first axis of the embedding table).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Hidden size (second axis of the embedding table).
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Reference to the underlying embedding weight buffer.
    pub fn table(&self) -> &Arc<[f32]> {
        &self.table
    }

    /// Look up the rows of the embedding table corresponding to
    /// `token_ids`. The input must be a U32 tensor of any rank; the
    /// output has the input's shape with a trailing `hidden` dim
    /// appended.
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        if token_ids.dtype() != fuel_core::DType::U32 {
            return Err(fuel_core::Error::Msg(format!(
                "Embedding::forward: token_ids must be U32, got {:?}",
                token_ids.dtype(),
            ))
            .bt());
        }
        let input_shape = token_ids.shape();
        let input_dims = input_shape.dims().to_vec();
        let flat_len: usize = input_dims.iter().product();
        let table_t = token_ids.const_f32_like(
            Arc::clone(&self.table),
            Shape::from_dims(&[self.vocab_size, self.hidden]),
        );
        let flat_ids = if input_dims.len() == 1 {
            token_ids.clone()
        } else {
            token_ids.reshape(Shape::from_dims(&[flat_len]))?
        };
        let rows = table_t.index_select(0_usize, &flat_ids)?;
        let mut out_dims = input_dims;
        out_dims.push(self.hidden);
        rows.reshape(Shape::from_dims(&out_dims))
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Embedding::forward(self, xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core::Device;

    fn make_table(vocab: usize, hidden: usize) -> Vec<f32> {
        (0..(vocab * hidden))
            .map(|i| (i as f32) * 0.01 - 0.5)
            .collect()
    }

    #[test]
    fn embedding_forward_shape_for_token_seq() {
        let vocab = 7;
        let hidden = 4;
        let seq = 5;
        let table = make_table(vocab, hidden);
        let emb = Embedding::new(Arc::from(table), vocab, hidden).unwrap();
        let tokens: Vec<u32> = vec![0, 3, 1, 6, 2];
        let token_ids = Tensor::from_u32(tokens, Shape::from_dims(&[seq]), &Device::cpu());
        let out = emb.forward(&token_ids).unwrap();
        assert_eq!(out.shape().dims(), &[seq, hidden]);
        let got = out.realize_f32();
        assert_eq!(got.len(), seq * hidden);
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "embedding out[{i}] = {v} not finite");
        }
    }

    #[test]
    fn embedding_lookup_matches_index_select_baseline() {
        let vocab = 6;
        let hidden = 3;
        let tokens: Vec<u32> = vec![2, 0, 5, 1];
        let table = make_table(vocab, hidden);

        // Reference: literal row gather.
        let mut expected = Vec::with_capacity(tokens.len() * hidden);
        for &tid in &tokens {
            let base = (tid as usize) * hidden;
            expected.extend_from_slice(&table[base..base + hidden]);
        }

        let emb = Embedding::new(Arc::from(table), vocab, hidden).unwrap();
        let token_ids = Tensor::from_u32(
            tokens.clone(),
            Shape::from_dims(&[tokens.len()]),
            &Device::cpu(),
        );
        let out = emb.forward(&token_ids).unwrap();
        assert_eq!(out.shape().dims(), &[tokens.len(), hidden]);
        let got = out.realize_f32();
        assert_eq!(got.len(), expected.len());
        for (i, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-6, "embedding[{i}] expected {e}, got {a}",);
        }
    }
}
