//! Hash-based prefix caching for KV state reuse.
//!
//! When multiple inference requests share a common prefix (e.g., a system prompt,
//! few-shot examples, or conversation history), the KV states for that prefix can be
//! computed once and reused across requests. This can reduce time-to-first-token by
//! 15–50% for repeated prefixes.
//!
//! # Architecture
//!
//! [`PrefixCache`] stores per-layer KV tensor pairs keyed by a hash of the token
//! sequence. When a new request arrives:
//!
//! 1. Compute the hash of the prompt token prefix.
//! 2. Look up the hash in the cache.
//! 3. **Hit**: Clone the cached KV tensors and skip the prefill for the matching prefix.
//! 4. **Miss**: Run prefill normally, then store the result for future reuse.
//!
//! The cache uses LRU eviction with a configurable maximum entry count to bound memory.
//!
//! # Example
//!
//! ```rust
//! use candle_inference::prefix_cache::PrefixCache;
//! use candle::{DType, Device, Tensor};
//!
//! # fn main() -> candle::Result<()> {
//! let mut cache = PrefixCache::new(16); // up to 16 cached prefixes
//!
//! // Simulate token IDs for a system prompt
//! let system_tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
//!
//! // First request: cache miss
//! assert!(cache.lookup(&system_tokens).is_none());
//!
//! // After prefill, store the KV states (one pair per layer)
//! let k = Tensor::zeros((1, 4, 5, 64), DType::F32, &Device::Cpu)?;
//! let v = Tensor::zeros((1, 4, 5, 64), DType::F32, &Device::Cpu)?;
//! let kv_states = vec![(k, v)]; // 1-layer example
//! cache.insert(&system_tokens, kv_states);
//!
//! // Second request with same prefix: cache hit!
//! let cached = cache.lookup(&system_tokens);
//! assert!(cached.is_some());
//! assert_eq!(cached.unwrap().len(), 1); // 1 layer
//! # Ok(())
//! # }
//! ```

use candle::Tensor;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// A cached KV state for one transformer layer.
pub type LayerKvState = (Tensor, Tensor);

/// Hash key for a token sequence prefix.
type PrefixHash = u64;

/// An entry in the prefix cache, storing per-layer KV tensors and LRU metadata.
struct CacheEntry {
    /// Per-layer (key, value) tensor pairs. Index = layer index.
    kv_states: Vec<LayerKvState>,
    /// The token sequence length this entry covers.
    seq_len: usize,
    /// Monotonically increasing access counter for LRU eviction.
    last_access: u64,
}

/// Hash-based prefix cache for KV state reuse across inference requests.
///
/// Entries are keyed by a hash of the token sequence. When the cache exceeds
/// `max_entries`, the least-recently-used entry is evicted.
pub struct PrefixCache {
    entries: HashMap<PrefixHash, CacheEntry>,
    max_entries: usize,
    access_counter: u64,
}

impl PrefixCache {
    /// Creates a new prefix cache with the given maximum number of entries.
    ///
    /// Each entry stores the full KV state for one token prefix across all layers.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries: max_entries.max(1),
            access_counter: 0,
        }
    }

    /// Looks up KV states for the given token prefix.
    ///
    /// Returns cloned KV tensor pairs (one per layer) if a cache entry exists,
    /// or `None` on a cache miss. The returned tensors can be used to initialize
    /// the KV cache for a new request, skipping the prefill phase for the
    /// matching prefix.
    ///
    /// On a hit, the entry's LRU timestamp is updated.
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<Vec<LayerKvState>> {
        let hash = hash_tokens(tokens);
        if let Some(entry) = self.entries.get_mut(&hash) {
            self.access_counter += 1;
            entry.last_access = self.access_counter;
            Some(
                entry
                    .kv_states
                    .iter()
                    .map(|(k, v): &LayerKvState| (k.clone(), v.clone()))
                    .collect(),
            )
        } else {
            None
        }
    }

    /// Stores KV states for the given token prefix.
    ///
    /// If an entry with the same prefix hash already exists, it is replaced.
    /// If the cache is full, the least-recently-used entry is evicted first.
    pub fn insert(&mut self, tokens: &[u32], kv_states: Vec<LayerKvState>) {
        let hash = hash_tokens(tokens);
        let seq_len = tokens.len();

        // Evict if at capacity and this is a new entry
        if !self.entries.contains_key(&hash) && self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        self.access_counter += 1;
        self.entries.insert(
            hash,
            CacheEntry {
                kv_states,
                seq_len,
                last_access: self.access_counter,
            },
        );
    }

    /// Returns the number of prefix entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the maximum number of entries the cache can hold.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Clears all cached entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_counter = 0;
    }

    /// Checks if a token prefix has a cached entry without updating LRU metadata.
    pub fn contains(&self, tokens: &[u32]) -> bool {
        let hash = hash_tokens(tokens);
        self.entries.contains_key(&hash)
    }

    /// Returns the sequence length of the cached entry for the given prefix,
    /// or `None` if not cached.
    pub fn cached_seq_len(&self, tokens: &[u32]) -> Option<usize> {
        let hash = hash_tokens(tokens);
        self.entries.get(&hash).map(|e| e.seq_len)
    }

    /// Finds the longest cached prefix that matches the beginning of `tokens`.
    ///
    /// Tries progressively shorter prefixes until a cache hit is found. Returns
    /// `None` if no prefix has been cached. The returned tuple contains the
    /// matched prefix length and the cloned KV states.
    ///
    /// This is more expensive than [`lookup`](PrefixCache::lookup) because it probes
    /// multiple hash keys. Use it when you don't know the exact prefix boundary.
    pub fn longest_prefix_match(
        &mut self,
        tokens: &[u32],
    ) -> Option<(usize, Vec<LayerKvState>)> {
        // Try from longest to shortest
        for len in (1..=tokens.len()).rev() {
            let prefix = &tokens[..len];
            if let Some(kv) = self.lookup(prefix) {
                return Some((len, kv));
            }
        }
        None
    }

    /// Evicts the least-recently-used entry.
    fn evict_lru(&mut self) {
        if let Some((&lru_key, _)) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
        {
            self.entries.remove(&lru_key);
        }
    }
}

/// Computes a hash of a token sequence using the standard library's `DefaultHasher`.
///
/// This uses SipHash-2-4 internally, which provides good collision resistance
/// for non-adversarial inputs. For prefix caching, we need deterministic hashing
/// (same tokens → same hash) but do not need cryptographic properties.
fn hash_tokens(tokens: &[u32]) -> PrefixHash {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn make_kv(layers: usize, seq_len: usize) -> Vec<LayerKvState> {
        (0..layers)
            .map(|_| {
                let k =
                    Tensor::zeros((1, 4, seq_len, 64), DType::F32, &Device::Cpu).unwrap();
                let v =
                    Tensor::zeros((1, 4, seq_len, 64), DType::F32, &Device::Cpu).unwrap();
                (k, v)
            })
            .collect()
    }

    #[test]
    fn basic_insert_and_lookup() {
        let mut cache = PrefixCache::new(4);
        let tokens = vec![1u32, 2, 3, 4, 5];
        assert!(cache.lookup(&tokens).is_none());

        let kv = make_kv(2, 5);
        cache.insert(&tokens, kv);

        let cached = cache.lookup(&tokens);
        assert!(cached.is_some());
        let cached = cached.unwrap();
        assert_eq!(cached.len(), 2);
        assert_eq!(cached[0].0.dims(), &[1, 4, 5, 64]);
    }

    #[test]
    fn different_prefixes_are_separate() {
        let mut cache = PrefixCache::new(4);
        let tokens_a = vec![1u32, 2, 3];
        let tokens_b = vec![4u32, 5, 6];

        cache.insert(&tokens_a, make_kv(1, 3));
        cache.insert(&tokens_b, make_kv(1, 3));

        assert!(cache.lookup(&tokens_a).is_some());
        assert!(cache.lookup(&tokens_b).is_some());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn lru_eviction() {
        let mut cache = PrefixCache::new(2);
        let a = vec![1u32];
        let b = vec![2u32];
        let c = vec![3u32];

        cache.insert(&a, make_kv(1, 1));
        cache.insert(&b, make_kv(1, 1));
        assert_eq!(cache.len(), 2);

        // Access 'a' to make it more recent than 'b'
        cache.lookup(&a);

        // Insert 'c' — should evict 'b' (least recently used)
        cache.insert(&c, make_kv(1, 1));
        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&a));
        assert!(!cache.contains(&b));
        assert!(cache.contains(&c));
    }

    #[test]
    fn replace_existing_entry() {
        let mut cache = PrefixCache::new(4);
        let tokens = vec![1u32, 2, 3];

        cache.insert(&tokens, make_kv(1, 3));
        assert_eq!(cache.len(), 1);

        // Replace with different data
        cache.insert(&tokens, make_kv(2, 3));
        assert_eq!(cache.len(), 1);

        let cached = cache.lookup(&tokens).unwrap();
        assert_eq!(cached.len(), 2); // should be the updated version
    }

    #[test]
    fn clear_empties_cache() {
        let mut cache = PrefixCache::new(4);
        cache.insert(&[1u32], make_kv(1, 1));
        cache.insert(&[2u32], make_kv(1, 1));
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn cached_seq_len() {
        let mut cache = PrefixCache::new(4);
        let tokens = vec![1u32, 2, 3, 4, 5];
        cache.insert(&tokens, make_kv(1, 5));

        assert_eq!(cache.cached_seq_len(&tokens), Some(5));
        assert_eq!(cache.cached_seq_len(&[99u32]), None);
    }

    #[test]
    fn longest_prefix_match_finds_longest() {
        let mut cache = PrefixCache::new(4);
        // Cache a short prefix and a long prefix
        cache.insert(&[1u32, 2], make_kv(1, 2));
        cache.insert(&[1u32, 2, 3, 4], make_kv(1, 4));

        // Query with tokens [1, 2, 3, 4, 5] — should match [1, 2, 3, 4]
        let result = cache.longest_prefix_match(&[1, 2, 3, 4, 5]);
        assert!(result.is_some());
        let (matched_len, kv) = result.unwrap();
        assert_eq!(matched_len, 4);
        assert_eq!(kv[0].0.dims(), &[1, 4, 4, 64]);
    }

    #[test]
    fn longest_prefix_match_no_match() {
        let mut cache = PrefixCache::new(4);
        cache.insert(&[1u32, 2, 3], make_kv(1, 3));

        let result = cache.longest_prefix_match(&[4, 5, 6]);
        assert!(result.is_none());
    }

    #[test]
    fn hash_determinism() {
        let tokens = vec![42u32, 100, 200, 300];
        let h1 = hash_tokens(&tokens);
        let h2 = hash_tokens(&tokens);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_different_sequences() {
        let h1 = hash_tokens(&[1, 2, 3]);
        let h2 = hash_tokens(&[3, 2, 1]);
        assert_ne!(h1, h2);
    }
}
