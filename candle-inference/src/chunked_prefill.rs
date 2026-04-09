//! Chunked prefill for bounded time-to-first-token.
//!
//! When a prompt is very long (thousands of tokens), processing it in a single forward
//! pass creates a latency spike — both for the prefill request itself and for any
//! concurrent decode requests waiting for the GPU. **Chunked prefill** breaks the
//! prompt into smaller chunks and processes them sequentially, allowing decode steps
//! to interleave between chunks.
//!
//! # Benefits
//!
//! - **Bounded TTFT**: The maximum latency per step is proportional to `chunk_size`,
//!   not total prompt length.
//! - **Decode interleaving**: Between chunks, the scheduler can run decode steps for
//!   other in-flight requests, improving overall throughput.
//! - **Memory smoothing**: Peak activation memory is proportional to `chunk_size`
//!   rather than the full prompt.
//!
//! # Architecture
//!
//! [`ChunkedPrefill`] splits a token sequence into chunks and tracks the prefill
//! progress. It does not own the model — the caller drives the forward passes.
//!
//! ```text
//! Prompt: [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9]  (10 tokens)
//! Chunk size: 4
//!
//! Step 1: forward([t0, t1, t2, t3], index_pos=0)  → KV cache updated
//! Step 2: forward([t4, t5, t6, t7], index_pos=4)  → KV cache updated
//! Step 3: forward([t8, t9],         index_pos=8)  → KV cache updated, logits returned
//! ```
//!
//! # Example
//!
//! ```rust
//! use candle_inference::chunked_prefill::ChunkedPrefill;
//!
//! let prompt_tokens: Vec<u32> = (0..1000).collect();
//! let mut prefill = ChunkedPrefill::new(&prompt_tokens, 256);
//!
//! assert!(!prefill.is_done());
//! assert_eq!(prefill.num_chunks(), 4); // ceil(1000/256) = 4
//!
//! while let Some(chunk) = prefill.next_chunk() {
//!     let tokens = chunk.tokens();
//!     let index_pos = chunk.index_pos();
//!     let is_last = chunk.is_last();
//!
//!     // model.forward(&Tensor::new(tokens, &device)?, index_pos, &mut cache)?;
//!     // if is_last { /* extract logits */ }
//! }
//!
//! assert!(prefill.is_done());
//! ```

/// A single chunk of tokens to be processed in one forward pass.
#[derive(Debug, Clone)]
pub struct PrefillChunk<'a> {
    /// The token IDs in this chunk.
    tokens: &'a [u32],
    /// The absolute position offset for this chunk (for KV cache / RoPE).
    index_pos: usize,
    /// Whether this is the last chunk (logits should be extracted).
    is_last: bool,
    /// Chunk index (0-based).
    chunk_idx: usize,
    /// Total number of chunks.
    total_chunks: usize,
}

impl<'a> PrefillChunk<'a> {
    /// Returns the token IDs in this chunk.
    pub fn tokens(&self) -> &[u32] {
        self.tokens
    }

    /// Returns the absolute position offset for this chunk.
    ///
    /// Pass this as `index_pos` to the model's forward method so that
    /// positional embeddings (RoPE, ALiBi, etc.) and the KV cache are
    /// updated at the correct positions.
    pub fn index_pos(&self) -> usize {
        self.index_pos
    }

    /// Returns `true` if this is the final chunk.
    ///
    /// The caller should extract logits only from the last chunk's output
    /// (typically the logits for the last token position).
    pub fn is_last(&self) -> bool {
        self.is_last
    }

    /// Returns the 0-based index of this chunk.
    pub fn chunk_idx(&self) -> usize {
        self.chunk_idx
    }

    /// Returns the total number of chunks.
    pub fn total_chunks(&self) -> usize {
        self.total_chunks
    }

    /// Returns the number of tokens in this chunk.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns `true` if this chunk is empty (should not normally happen).
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Splits a prompt into chunks for progressive prefill.
///
/// Use [`next_chunk`](ChunkedPrefill::next_chunk) in a loop to get each chunk.
/// The final chunk's output contains the logits for sampling the first generated token.
#[derive(Debug, Clone)]
pub struct ChunkedPrefill<'a> {
    tokens: &'a [u32],
    chunk_size: usize,
    current_pos: usize,
    total_chunks: usize,
    current_chunk: usize,
}

impl<'a> ChunkedPrefill<'a> {
    /// Creates a new chunked prefill iterator.
    ///
    /// * `tokens` — The full prompt token sequence.
    /// * `chunk_size` — Maximum tokens per forward pass. Must be ≥ 1.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    pub fn new(tokens: &'a [u32], chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        let total_chunks = if tokens.is_empty() {
            0
        } else {
            (tokens.len() + chunk_size - 1) / chunk_size
        };
        Self {
            tokens,
            chunk_size,
            current_pos: 0,
            total_chunks,
            current_chunk: 0,
        }
    }

    /// Returns the next chunk, or `None` if all chunks have been yielded.
    pub fn next_chunk(&mut self) -> Option<PrefillChunk<'a>> {
        if self.current_pos >= self.tokens.len() {
            return None;
        }

        let start = self.current_pos;
        let end = (start + self.chunk_size).min(self.tokens.len());
        let chunk_tokens = &self.tokens[start..end];
        let chunk_idx = self.current_chunk;

        self.current_pos = end;
        self.current_chunk += 1;

        Some(PrefillChunk {
            tokens: chunk_tokens,
            index_pos: start,
            is_last: end >= self.tokens.len(),
            chunk_idx,
            total_chunks: self.total_chunks,
        })
    }

    /// Returns `true` if all chunks have been yielded.
    pub fn is_done(&self) -> bool {
        self.current_pos >= self.tokens.len()
    }

    /// Returns the total number of chunks.
    pub fn num_chunks(&self) -> usize {
        self.total_chunks
    }

    /// Returns the number of chunks remaining (not yet yielded).
    pub fn remaining_chunks(&self) -> usize {
        self.total_chunks - self.current_chunk
    }

    /// Returns the total number of tokens in the prompt.
    pub fn total_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Returns the number of tokens already yielded.
    pub fn tokens_processed(&self) -> usize {
        self.current_pos
    }

    /// Returns the configured chunk size.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Resets the iterator to the beginning.
    pub fn reset(&mut self) {
        self.current_pos = 0;
        self.current_chunk = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_chunking() {
        let tokens: Vec<u32> = (0..10).collect();
        let mut prefill = ChunkedPrefill::new(&tokens, 4);

        assert_eq!(prefill.num_chunks(), 3); // ceil(10/4) = 3
        assert!(!prefill.is_done());

        // Chunk 0: [0,1,2,3]
        let c = prefill.next_chunk().unwrap();
        assert_eq!(c.tokens(), &[0, 1, 2, 3]);
        assert_eq!(c.index_pos(), 0);
        assert!(!c.is_last());
        assert_eq!(c.chunk_idx(), 0);
        assert_eq!(c.len(), 4);

        // Chunk 1: [4,5,6,7]
        let c = prefill.next_chunk().unwrap();
        assert_eq!(c.tokens(), &[4, 5, 6, 7]);
        assert_eq!(c.index_pos(), 4);
        assert!(!c.is_last());

        // Chunk 2: [8,9] (partial)
        let c = prefill.next_chunk().unwrap();
        assert_eq!(c.tokens(), &[8, 9]);
        assert_eq!(c.index_pos(), 8);
        assert!(c.is_last());
        assert_eq!(c.chunk_idx(), 2);

        assert!(prefill.is_done());
        assert!(prefill.next_chunk().is_none());
    }

    #[test]
    fn exact_division() {
        let tokens: Vec<u32> = (0..8).collect();
        let mut prefill = ChunkedPrefill::new(&tokens, 4);

        assert_eq!(prefill.num_chunks(), 2);

        let c0 = prefill.next_chunk().unwrap();
        assert_eq!(c0.tokens(), &[0, 1, 2, 3]);
        assert!(!c0.is_last());

        let c1 = prefill.next_chunk().unwrap();
        assert_eq!(c1.tokens(), &[4, 5, 6, 7]);
        assert!(c1.is_last());

        assert!(prefill.is_done());
    }

    #[test]
    fn single_chunk() {
        let tokens: Vec<u32> = (0..3).collect();
        let mut prefill = ChunkedPrefill::new(&tokens, 10);

        assert_eq!(prefill.num_chunks(), 1);

        let c = prefill.next_chunk().unwrap();
        assert_eq!(c.tokens(), &[0, 1, 2]);
        assert_eq!(c.index_pos(), 0);
        assert!(c.is_last());
        assert_eq!(c.chunk_idx(), 0);

        assert!(prefill.is_done());
    }

    #[test]
    fn chunk_size_equals_length() {
        let tokens: Vec<u32> = (0..5).collect();
        let mut prefill = ChunkedPrefill::new(&tokens, 5);

        assert_eq!(prefill.num_chunks(), 1);

        let c = prefill.next_chunk().unwrap();
        assert_eq!(c.tokens().len(), 5);
        assert!(c.is_last());
    }

    #[test]
    fn chunk_size_one() {
        let tokens: Vec<u32> = (0..3).collect();
        let mut prefill = ChunkedPrefill::new(&tokens, 1);

        assert_eq!(prefill.num_chunks(), 3);

        for i in 0..3 {
            let c = prefill.next_chunk().unwrap();
            assert_eq!(c.tokens(), &[i]);
            assert_eq!(c.index_pos(), i as usize);
            assert_eq!(c.is_last(), i == 2);
        }
    }

    #[test]
    fn empty_tokens() {
        let tokens: Vec<u32> = vec![];
        let mut prefill = ChunkedPrefill::new(&tokens, 4);

        assert_eq!(prefill.num_chunks(), 0);
        assert!(prefill.is_done());
        assert!(prefill.next_chunk().is_none());
    }

    #[test]
    fn reset_works() {
        let tokens: Vec<u32> = (0..6).collect();
        let mut prefill = ChunkedPrefill::new(&tokens, 4);

        // Consume all chunks
        while prefill.next_chunk().is_some() {}
        assert!(prefill.is_done());

        // Reset
        prefill.reset();
        assert!(!prefill.is_done());
        assert_eq!(prefill.remaining_chunks(), 2);

        let c = prefill.next_chunk().unwrap();
        assert_eq!(c.tokens(), &[0, 1, 2, 3]);
    }

    #[test]
    fn remaining_chunks_decrements() {
        let tokens: Vec<u32> = (0..10).collect();
        let mut prefill = ChunkedPrefill::new(&tokens, 3);

        assert_eq!(prefill.num_chunks(), 4); // ceil(10/3)
        assert_eq!(prefill.remaining_chunks(), 4);

        prefill.next_chunk();
        assert_eq!(prefill.remaining_chunks(), 3);
        assert_eq!(prefill.tokens_processed(), 3);

        prefill.next_chunk();
        assert_eq!(prefill.remaining_chunks(), 2);
        assert_eq!(prefill.tokens_processed(), 6);
    }

    #[test]
    fn total_chunks_info() {
        let tokens: Vec<u32> = (0..10).collect();
        let mut prefill = ChunkedPrefill::new(&tokens, 4);

        let c = prefill.next_chunk().unwrap();
        assert_eq!(c.total_chunks(), 3);
    }

    #[test]
    #[should_panic(expected = "chunk_size must be > 0")]
    fn zero_chunk_size_panics() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        ChunkedPrefill::new(&tokens, 0);
    }

    #[test]
    fn large_prompt() {
        let tokens: Vec<u32> = (0..10_000).collect();
        let mut prefill = ChunkedPrefill::new(&tokens, 512);

        assert_eq!(prefill.num_chunks(), 20); // ceil(10000/512) = 20

        let mut chunk_count = 0;
        let mut last_was_last = false;
        while let Some(c) = prefill.next_chunk() {
            assert!(c.len() <= 512);
            assert!(c.len() > 0);
            assert_eq!(c.index_pos(), chunk_count * 512);
            last_was_last = c.is_last();
            chunk_count += 1;
        }

        assert_eq!(chunk_count, 20);
        assert!(last_was_last);
    }
}
