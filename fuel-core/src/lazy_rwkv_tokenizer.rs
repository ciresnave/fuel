//! Byte-pair tokenizer for RWKV models.
//!
//! Self-contained helper (no tensor math) ported from the retired eager
//! [`fuel_transformers::_models_retired::llm::rwkv_v5::Tokenizer`] so the
//! lazy `rwkv` example binary can be revived without pulling the retired
//! eager `Tensor` substrate in.
//!
//! Loads a JSON vocabulary (`token → id`) and runs a greedy longest-match
//! byte-pair encode/decode. The original Python reference lives at:
//! <https://github.com/BlinkDL/ChatRWKV/blob/095e812aef15a1f74107f6c39d13578a2412dc46/RWKV_v5_demo.py#L14>

use crate::Result;
use std::collections::{HashMap, HashSet};

type Bytes = Vec<u8>;

/// Byte-pair tokenizer for RWKV models loaded from a JSON vocabulary file.
pub struct Tokenizer {
    table: Vec<Vec<Vec<Bytes>>>,
    good: Vec<HashSet<u8>>,
    idx2token: HashMap<u32, Vec<u8>>,
    token2idx: HashMap<Vec<u8>, u32>,
}

impl Tokenizer {
    /// Load a tokenizer from a JSON vocabulary file at path `p`.
    pub fn new<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let file = std::fs::File::open(p)?;
        let token2idx: HashMap<String, u32> =
            serde_json::from_reader(file).map_err(crate::Error::wrap)?;
        Self::from_token2idx(token2idx)
    }

    /// Build a tokenizer directly from an already-parsed `token → id` map.
    ///
    /// Primarily for tests / in-memory vocabularies; the JSON loader simply
    /// defers to this after deserialization.
    pub fn from_token2idx(token2idx: HashMap<String, u32>) -> Result<Self> {
        let token2idx = token2idx
            .into_iter()
            .map(|(key, value)| (key.into_bytes(), value))
            .collect::<HashMap<_, _>>();
        let idx2token = token2idx
            .iter()
            .map(|(key, value)| (*value, key.to_vec()))
            .collect::<HashMap<_, _>>();

        let max_idx = token2idx.values().copied().max().unwrap_or(0);

        let mut table = vec![vec![vec![]; 256]; 256];
        let mut good = vec![HashSet::new(); 256];
        for idx in (0..(1 + max_idx)).rev() {
            let s = match idx2token.get(&idx) {
                None => continue,
                Some(s) => s,
            };
            if s.len() >= 2 {
                let (s0, s1) = (s[0], s[1]);
                table[s0 as usize][s1 as usize].push(s.to_vec());
                good[s0 as usize].insert(s1);
            }
        }
        Ok(Self {
            table,
            good,
            idx2token,
            token2idx,
        })
    }

    /// Decode a sequence of token IDs to raw bytes.
    pub fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        let mut v = Vec::new();
        for token_id in tokens.iter() {
            if let Some(token) = self.idx2token.get(token_id) {
                v.extend_from_slice(token.as_slice())
            }
        }
        v
    }

    /// Decode a sequence of token IDs to a UTF-8 string.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let bytes = self.decode_bytes(tokens);
        String::from_utf8(bytes).map_err(crate::Error::wrap)
    }

    /// Encode raw bytes to a sequence of token IDs.
    pub fn encode_bytes(&self, bytes: &[u8]) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let mut s = vec![bytes[i]];
            if i + 1 < bytes.len() && self.good[bytes[i] as usize].contains(&bytes[i + 1]) {
                let table = &self.table[bytes[i] as usize][bytes[i + 1] as usize];
                for table_elem in table.iter() {
                    if bytes[i..].starts_with(table_elem) {
                        s = table_elem.to_vec();
                        break;
                    }
                }
            }
            i += s.len();
            let token = match self.token2idx.get(&s) {
                None => crate::bail!("unexpected token '{}' {s:?}", String::from_utf8_lossy(&s)),
                Some(token) => *token,
            };
            tokens.push(token)
        }
        Ok(tokens)
    }

    /// Encode a UTF-8 string to a sequence of token IDs.
    pub fn encode(&self, str: &str) -> Result<Vec<u32>> {
        self.encode_bytes(str.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny fixture: covers every ASCII letter the roundtrip test needs, plus
    /// a couple of multi-byte merges so the byte-pair table is non-empty.
    fn tiny_vocab() -> HashMap<String, u32> {
        let mut v = HashMap::new();
        // Single-byte tokens for each character we encode below.
        // Ids are arbitrary but unique.
        let mut next_id: u32 = 1;
        for ch in b"abcdefghijklmnopqrstuvwxyz ,!" {
            v.insert((*ch as char).to_string(), next_id);
            next_id += 1;
        }
        // Multi-byte merges: prefer longest greedy match.
        v.insert("ab".to_string(), 100);
        v.insert("abc".to_string(), 101);
        v.insert("hello".to_string(), 200);
        v.insert("world".to_string(), 201);
        v
    }

    #[test]
    fn encode_decode_roundtrip_ascii() {
        let tok = Tokenizer::from_token2idx(tiny_vocab()).unwrap();

        // "hello world" should pick up the "hello" and "world" merges.
        let ids = tok.encode("hello world").unwrap();
        assert_eq!(ids.len(), 3, "expected hello + space + world, got {ids:?}");
        assert_eq!(ids[0], 200, "first token should be 'hello' merge");
        assert_eq!(ids[2], 201, "third token should be 'world' merge");

        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(decoded, "hello world");

        // Plain "abc" should resolve to the longest "abc" merge, not "ab" + "c".
        let ids = tok.encode("abc").unwrap();
        assert_eq!(ids, vec![101]);
        assert_eq!(tok.decode(&ids).unwrap(), "abc");

        // A char that requires only single-byte tokens still roundtrips.
        let ids = tok.encode("cab").unwrap();
        assert_eq!(tok.decode(&ids).unwrap(), "cab");
    }

    #[test]
    fn decode_bytes_known_ids() {
        let tok = Tokenizer::from_token2idx(tiny_vocab()).unwrap();

        // 200 = "hello", 201 = "world"; we look up their byte lookup directly
        // so this test exercises decode_bytes independently of encode.
        let bytes = tok.decode_bytes(&[200, 201]);
        assert_eq!(bytes, b"helloworld".to_vec());

        // Unknown ids are silently skipped, matching the eager behavior.
        let bytes = tok.decode_bytes(&[200, 99_999, 201]);
        assert_eq!(bytes, b"helloworld".to_vec());

        // An empty token list returns an empty Vec.
        assert!(tok.decode_bytes(&[]).is_empty());
    }
}
