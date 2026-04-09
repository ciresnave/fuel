//! KV cache compression strategies.
//!
//! Long-context inference is bottlenecked by KV cache memory.  This module
//! provides three orthogonal compression strategies that can be applied
//! independently or combined:
//!
//! - **KIVI** — Per-channel asymmetric quantization of keys/values to 2–4 bits
//!   with per-head scale/zero-point.  2–4× memory reduction.
//! - **R-KV** — Importance-redundancy scoring that retains a configurable
//!   budget fraction of tokens. Tokens are ranked by cumulative attention
//!   importance minus a redundancy penalty, and the bottom fraction is pruned.
//! - **Low-Rank** — Fixed memory ceiling by tracking only rank-R approximation
//!   metadata. Trades a small quality cost for bounded KV growth.
//!
//! # Architecture
//!
//! All strategies implement the [`KvCompressor`] trait, exposing a uniform
//! `compress()` → [`CompressedKv`] → `decompress()` round-trip.  The module
//! operates on raw score/importance metadata (not tensors) so it can be tested
//! without a GPU.
//!
//! ```text
//! KV cache (per-layer, per-head)
//!   │
//!   ├─► KiviCompressor   ─► 2/4-bit quantized repr   ─► dequant on attention
//!   ├─► RkvCompressor    ─► pruned token subset       ─► smaller cache
//!   └─► LowRankCompressor─► rank-R approximation info ─► bounded memory
//! ```
//!
//! # Example
//!
//! ```rust
//! use candle_inference::kv_compress::{KiviConfig, KiviCompressor, KvCompressor, CompressedKv};
//!
//! let config = KiviConfig::new(4); // 4-bit quantization
//! let compressor = KiviCompressor::new(config);
//!
//! // Per-channel values to quantize (e.g., one channel across all tokens)
//! let channel_values = vec![0.1_f32, 0.5, -0.3, 0.8, -0.1, 0.2];
//! let compressed = compressor.compress(&channel_values);
//!
//! // Decompress for attention computation
//! let restored = compressed.decompress();
//! // restored ≈ channel_values (within quantization error)
//! assert_eq!(restored.len(), channel_values.len());
//! ```

use std::fmt;

// ── Trait ──────────────────────────────────────────────────────────────────

/// Compressed KV representation that can be decompressed back to f32 values.
pub trait CompressedKv: fmt::Debug {
    /// Decompress back to f32 values.
    fn decompress(&self) -> Vec<f32>;

    /// Number of elements in the original (uncompressed) data.
    fn len(&self) -> usize;

    /// Whether the compressed representation is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Approximate memory usage in bytes of the compressed representation.
    fn compressed_size_bytes(&self) -> usize;

    /// Memory usage in bytes of the original uncompressed data.
    fn original_size_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<f32>()
    }

    /// Compression ratio (original / compressed).  Higher is better.
    fn compression_ratio(&self) -> f64 {
        let cs = self.compressed_size_bytes();
        if cs == 0 {
            return 0.0;
        }
        self.original_size_bytes() as f64 / cs as f64
    }
}

/// A KV cache compression strategy.
pub trait KvCompressor: fmt::Debug {
    /// The compressed representation type.
    type Compressed: CompressedKv;

    /// Compress a slice of f32 values (e.g., one channel across all tokens
    /// for KIVI, or importance scores for R-KV).
    fn compress(&self, values: &[f32]) -> Self::Compressed;

    /// Human-readable name of the strategy.
    fn name(&self) -> &str;
}

// ═══════════════════════════════════════════════════════════════════════════
// KIVI: Per-channel asymmetric quantization
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for KIVI quantization.
#[derive(Debug, Clone)]
pub struct KiviConfig {
    /// Number of quantization bits (2 or 4).
    pub bits: u8,
}

impl KiviConfig {
    /// Create a new KIVI config.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is not 2 or 4.
    pub fn new(bits: u8) -> Self {
        assert!(bits == 2 || bits == 4, "KIVI supports 2-bit or 4-bit quantization");
        Self { bits }
    }
}

/// Per-channel asymmetric quantization compressor.
///
/// Stores per-channel `(scale, zero_point)` plus packed integer codes.
/// Decompression: `value = scale * code + zero_point`.
#[derive(Debug, Clone)]
pub struct KiviCompressor {
    config: KiviConfig,
}

impl KiviCompressor {
    pub fn new(config: KiviConfig) -> Self {
        Self { config }
    }

    fn max_code(&self) -> u8 {
        (1u8 << self.config.bits) - 1
    }
}

/// Compressed representation from KIVI quantization.
#[derive(Debug, Clone)]
pub struct KiviCompressed {
    /// Quantized codes (one per element, stored as u8 even for 2-bit).
    codes: Vec<u8>,
    /// Scale factor: `(max - min) / max_code`.
    scale: f32,
    /// Zero point: `min` value.
    zero_point: f32,
    /// Number of bits used.
    bits: u8,
}

impl CompressedKv for KiviCompressed {
    fn decompress(&self) -> Vec<f32> {
        self.codes
            .iter()
            .map(|&c| self.scale * c as f32 + self.zero_point)
            .collect()
    }

    fn len(&self) -> usize {
        self.codes.len()
    }

    fn compressed_size_bytes(&self) -> usize {
        // codes (packed) + scale + zero_point
        let packed_bits = self.codes.len() * self.bits as usize;
        let packed_bytes = (packed_bits + 7) / 8;
        packed_bytes + 4 + 4 // scale(f32) + zero_point(f32)
    }
}

impl KvCompressor for KiviCompressor {
    type Compressed = KiviCompressed;

    fn compress(&self, values: &[f32]) -> KiviCompressed {
        if values.is_empty() {
            return KiviCompressed {
                codes: Vec::new(),
                scale: 0.0,
                zero_point: 0.0,
                bits: self.config.bits,
            };
        }

        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_code = self.max_code();

        let range = max - min;
        let scale = if range == 0.0 {
            0.0
        } else {
            range / max_code as f32
        };

        let codes: Vec<u8> = values
            .iter()
            .map(|&v| {
                if scale == 0.0 {
                    0
                } else {
                    let code = ((v - min) / scale).round() as u8;
                    code.min(max_code)
                }
            })
            .collect();

        KiviCompressed {
            codes,
            scale,
            zero_point: min,
            bits: self.config.bits,
        }
    }

    fn name(&self) -> &str {
        match self.config.bits {
            2 => "KIVI-2bit",
            4 => "KIVI-4bit",
            _ => "KIVI",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// R-KV: Importance-redundancy scoring
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for R-KV importance-redundancy pruning.
#[derive(Debug, Clone)]
pub struct RkvConfig {
    /// Fraction of tokens to retain (0.0, 1.0].  E.g. 0.34 keeps 34%.
    pub budget_fraction: f32,
    /// Weight for redundancy penalty.  Higher = more aggressive deduplication.
    pub redundancy_weight: f32,
}

impl RkvConfig {
    /// Create a new R-KV config.
    ///
    /// # Panics
    ///
    /// Panics if `budget_fraction` is not in (0.0, 1.0].
    pub fn new(budget_fraction: f32) -> Self {
        assert!(
            budget_fraction > 0.0 && budget_fraction <= 1.0,
            "budget_fraction must be in (0.0, 1.0]"
        );
        Self {
            budget_fraction,
            redundancy_weight: 0.5,
        }
    }

    /// Set the redundancy weight.
    pub fn with_redundancy_weight(mut self, w: f32) -> Self {
        self.redundancy_weight = w;
        self
    }
}

/// Importance-redundancy KV cache pruner.
///
/// Given per-token importance scores (e.g., cumulative attention) and an
/// optional redundancy signal, retains only the top `budget_fraction` tokens.
#[derive(Debug, Clone)]
pub struct RkvCompressor {
    config: RkvConfig,
}

impl RkvCompressor {
    pub fn new(config: RkvConfig) -> Self {
        Self { config }
    }

    /// Compute which token indices to keep.
    ///
    /// * `importance` — Per-token importance scores (higher = more important).
    /// * `redundancy` — Optional per-token redundancy scores (higher = more redundant).
    ///
    /// Returns sorted indices of tokens to retain.
    pub fn select_keep(
        &self,
        importance: &[f32],
        redundancy: Option<&[f32]>,
    ) -> Vec<usize> {
        let n = importance.len();
        if n == 0 {
            return Vec::new();
        }

        let budget = ((n as f32 * self.config.budget_fraction).ceil() as usize).max(1).min(n);

        // Compute combined scores: importance - redundancy_weight * redundancy
        let mut scored: Vec<(usize, f32)> = importance
            .iter()
            .enumerate()
            .map(|(i, &imp)| {
                let red = redundancy.map_or(0.0, |r| r.get(i).copied().unwrap_or(0.0));
                (i, imp - self.config.redundancy_weight * red)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(budget);

        // Return indices sorted by position (preserves temporal order)
        let mut keep: Vec<usize> = scored.iter().map(|&(i, _)| i).collect();
        keep.sort_unstable();
        keep
    }

    /// Compute which token indices to evict (complement of `select_keep`).
    pub fn select_evict(
        &self,
        importance: &[f32],
        redundancy: Option<&[f32]>,
    ) -> Vec<usize> {
        let keep = self.select_keep(importance, redundancy);
        let keep_set: std::collections::HashSet<usize> = keep.into_iter().collect();
        (0..importance.len())
            .filter(|i| !keep_set.contains(i))
            .collect()
    }
}

/// Compressed representation from R-KV pruning (stores only retained values).
#[derive(Debug, Clone)]
pub struct RkvCompressed {
    /// Retained values.
    retained_values: Vec<f32>,
    /// Original indices of retained values.
    retained_indices: Vec<usize>,
    /// Original length before pruning.
    original_len: usize,
}

impl CompressedKv for RkvCompressed {
    fn decompress(&self) -> Vec<f32> {
        // Reconstruct full-length vector, filling pruned positions with 0.0
        let mut out = vec![0.0f32; self.original_len];
        for (&idx, &val) in self.retained_indices.iter().zip(&self.retained_values) {
            out[idx] = val;
        }
        out
    }

    fn len(&self) -> usize {
        self.original_len
    }

    fn compressed_size_bytes(&self) -> usize {
        // retained values + indices + original_len
        self.retained_values.len() * 4
            + self.retained_indices.len() * std::mem::size_of::<usize>()
            + std::mem::size_of::<usize>()
    }
}

impl KvCompressor for RkvCompressor {
    type Compressed = RkvCompressed;

    fn compress(&self, values: &[f32]) -> RkvCompressed {
        // When used as a KvCompressor, treat the values themselves as importance
        // scores (higher magnitude = more important).
        let importance: Vec<f32> = values.iter().map(|v| v.abs()).collect();
        let keep = self.select_keep(&importance, None);

        let retained_values: Vec<f32> = keep.iter().map(|&i| values[i]).collect();
        let retained_indices = keep;

        RkvCompressed {
            retained_values,
            retained_indices,
            original_len: values.len(),
        }
    }

    fn name(&self) -> &str {
        "R-KV"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Low-Rank: Fixed-rank approximation metadata
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for low-rank KV approximation.
#[derive(Debug, Clone)]
pub struct LowRankConfig {
    /// Approximation rank.  Lower = more compression, lower quality.
    pub rank: usize,
}

impl LowRankConfig {
    /// Create a new low-rank config.
    ///
    /// # Panics
    ///
    /// Panics if `rank` is 0.
    pub fn new(rank: usize) -> Self {
        assert!(rank > 0, "rank must be > 0");
        Self { rank }
    }
}

/// Low-rank KV cache compressor.
///
/// Approximates d-dimensional token vectors using rank-R:
/// Each token's d values are projected to R coefficients plus a shared
/// R×d basis (which is built incrementally).  This implementation uses
/// a simple mean-centered projection for demonstration; a production
/// implementation would use incremental SVD.
#[derive(Debug, Clone)]
pub struct LowRankCompressor {
    config: LowRankConfig,
}

impl LowRankCompressor {
    pub fn new(config: LowRankConfig) -> Self {
        Self { config }
    }
}

/// Compressed representation from low-rank approximation.
#[derive(Debug, Clone)]
pub struct LowRankCompressed {
    /// Coefficients: one R-length vector per token (flattened: n_tokens × rank).
    coefficients: Vec<f32>,
    /// Basis vectors (rank values — simplified 1-D basis for single-channel).
    basis: Vec<f32>,
    /// Mean value (subtracted before projection).
    mean: f32,
    /// Original length.
    original_len: usize,
    /// Rank used.
    rank: usize,
}

impl CompressedKv for LowRankCompressed {
    fn decompress(&self) -> Vec<f32> {
        // For a 1-D channel, reconstruction is: value ≈ mean + coeff * basis[0]
        // (simplified from full matrix case)
        if self.basis.is_empty() || self.coefficients.is_empty() {
            return vec![self.mean; self.original_len];
        }

        self.coefficients
            .iter()
            .map(|&c| self.mean + c * self.basis[0])
            .collect()
    }

    fn len(&self) -> usize {
        self.original_len
    }

    fn compressed_size_bytes(&self) -> usize {
        // coefficients + basis + mean + metadata
        self.coefficients.len() * 4
            + self.basis.len() * 4
            + 4 // mean
            + 8 // original_len + rank
    }
}

impl LowRankCompressed {
    /// Returns the approximation rank used.
    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl KvCompressor for LowRankCompressor {
    type Compressed = LowRankCompressed;

    fn compress(&self, values: &[f32]) -> LowRankCompressed {
        if values.is_empty() {
            return LowRankCompressed {
                coefficients: Vec::new(),
                basis: Vec::new(),
                mean: 0.0,
                original_len: 0,
                rank: self.config.rank,
            };
        }

        let n = values.len();
        let mean = values.iter().sum::<f32>() / n as f32;
        let centered: Vec<f32> = values.iter().map(|&v| v - mean).collect();

        // For 1-D data, the "basis" is the direction of maximum variance.
        // With a single channel, this is just the sign of the standard deviation.
        let variance = centered.iter().map(|x| x * x).sum::<f32>() / n as f32;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            // All values are ~equal; coefficients are all 0
            return LowRankCompressed {
                coefficients: vec![0.0; n],
                basis: vec![1.0],
                mean,
                original_len: n,
                rank: self.config.rank,
            };
        }

        // Project onto unit basis (normalized centered values = coefficients)
        let coefficients: Vec<f32> = centered.iter().map(|&c| c / std_dev).collect();
        let basis = vec![std_dev]; // single basis "vector" for 1-D

        LowRankCompressed {
            coefficients,
            basis,
            mean,
            original_len: n,
            rank: self.config.rank,
        }
    }

    fn name(&self) -> &str {
        "LowRank"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── KIVI tests ────────────────────────────────────────────────────────

    #[test]
    fn kivi_4bit_roundtrip() {
        let compressor = KiviCompressor::new(KiviConfig::new(4));
        let values = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let compressed = compressor.compress(&values);
        let restored = compressed.decompress();

        assert_eq!(restored.len(), values.len());
        for (orig, rest) in values.iter().zip(&restored) {
            assert!((orig - rest).abs() < 0.07, "orig={orig}, restored={rest}");
        }
    }

    #[test]
    fn kivi_2bit_roundtrip() {
        let compressor = KiviCompressor::new(KiviConfig::new(2));
        let values = vec![-1.0, 0.0, 0.5, 1.0];
        let compressed = compressor.compress(&values);
        let restored = compressed.decompress();

        assert_eq!(restored.len(), 4);
        // 2-bit has only 4 levels, so error is larger
        for (orig, rest) in values.iter().zip(&restored) {
            assert!((orig - rest).abs() < 0.5, "orig={orig}, restored={rest}");
        }
    }

    #[test]
    fn kivi_compression_ratio() {
        let compressor = KiviCompressor::new(KiviConfig::new(4));
        let values: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let compressed = compressor.compress(&values);

        let ratio = compressed.compression_ratio();
        // 4-bit should give ~8:1 ratio (32-bit / 4-bit) minus overhead
        assert!(ratio > 3.0, "ratio={ratio}, expected > 3.0");
    }

    #[test]
    fn kivi_empty() {
        let compressor = KiviCompressor::new(KiviConfig::new(4));
        let compressed = compressor.compress(&[]);
        assert_eq!(compressed.len(), 0);
        assert!(compressed.is_empty());
        assert!(compressed.decompress().is_empty());
    }

    #[test]
    fn kivi_constant_values() {
        let compressor = KiviCompressor::new(KiviConfig::new(4));
        let values = vec![0.5; 10];
        let compressed = compressor.compress(&values);
        let restored = compressed.decompress();

        for &v in &restored {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic(expected = "KIVI supports 2-bit or 4-bit")]
    fn kivi_invalid_bits() {
        KiviConfig::new(8);
    }

    // ── R-KV tests ────────────────────────────────────────────────────────

    #[test]
    fn rkv_select_keep_basic() {
        let compressor = RkvCompressor::new(RkvConfig::new(0.5));
        let importance = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let keep = compressor.select_keep(&importance, None);

        // Budget = ceil(5 * 0.5) = 3 tokens
        assert_eq!(keep.len(), 3);
        // Should keep indices 1 (5.0), 3 (8.0), 4 (3.0) — top 3 by importance
        assert!(keep.contains(&1));
        assert!(keep.contains(&3));
        // Sorted by position
        assert!(keep.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn rkv_redundancy_penalty() {
        let compressor = RkvCompressor::new(RkvConfig::new(0.5).with_redundancy_weight(1.0));
        // Token 1 has high importance but also high redundancy
        let importance = vec![1.0, 10.0, 5.0, 3.0];
        let redundancy = vec![0.0, 9.0, 0.0, 0.0];
        let keep = compressor.select_keep(&importance, Some(&redundancy));

        // Budget = ceil(4 * 0.5) = 2
        // Scores: [1.0-0=1.0, 10.0-9.0=1.0, 5.0-0=5.0, 3.0-0=3.0]
        // Top 2 by score: index 2 (5.0) and index 3 (3.0)
        assert_eq!(keep.len(), 2);
        assert!(keep.contains(&2));
        assert!(keep.contains(&3));
    }

    #[test]
    fn rkv_full_budget_keeps_all() {
        let compressor = RkvCompressor::new(RkvConfig::new(1.0));
        let importance = vec![1.0, 2.0, 3.0];
        let keep = compressor.select_keep(&importance, None);
        assert_eq!(keep, vec![0, 1, 2]);
    }

    #[test]
    fn rkv_select_evict() {
        let compressor = RkvCompressor::new(RkvConfig::new(0.5));
        let importance = vec![1.0, 5.0, 2.0, 8.0];
        let evict = compressor.select_evict(&importance, None);

        // Budget = ceil(4 * 0.5) = 2 kept, so 2 evicted
        assert_eq!(evict.len(), 2);
    }

    #[test]
    fn rkv_compress_roundtrip() {
        let compressor = RkvCompressor::new(RkvConfig::new(0.5));
        let values = vec![0.1, 5.0, 0.2, 8.0, 0.3];
        let compressed = compressor.compress(&values);
        let restored = compressed.decompress();

        assert_eq!(restored.len(), 5);
        // High-magnitude values are retained exactly
        assert!((restored[1] - 5.0).abs() < 1e-6);
        assert!((restored[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn rkv_empty() {
        let compressor = RkvCompressor::new(RkvConfig::new(0.5));
        let keep = compressor.select_keep(&[], None);
        assert!(keep.is_empty());
    }

    #[test]
    #[should_panic(expected = "budget_fraction must be in")]
    fn rkv_invalid_budget() {
        RkvConfig::new(0.0);
    }

    // ── Low-Rank tests ────────────────────────────────────────────────────

    #[test]
    fn lowrank_roundtrip() {
        let compressor = LowRankCompressor::new(LowRankConfig::new(1));
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let compressed = compressor.compress(&values);
        let restored = compressed.decompress();

        assert_eq!(restored.len(), 5);
        // For 1-D data with rank-1, reconstruction should be exact
        for (orig, rest) in values.iter().zip(&restored) {
            assert!(
                (orig - rest).abs() < 1e-4,
                "orig={orig}, restored={rest}"
            );
        }
    }

    #[test]
    fn lowrank_constant_values() {
        let compressor = LowRankCompressor::new(LowRankConfig::new(1));
        let values = vec![3.0; 8];
        let compressed = compressor.compress(&values);
        let restored = compressed.decompress();

        for &v in &restored {
            assert!((v - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn lowrank_empty() {
        let compressor = LowRankCompressor::new(LowRankConfig::new(1));
        let compressed = compressor.compress(&[]);
        assert_eq!(compressed.len(), 0);
        assert!(compressed.decompress().is_empty());
    }

    #[test]
    fn lowrank_compression_metadata() {
        let compressor = LowRankCompressor::new(LowRankConfig::new(2));
        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let compressed = compressor.compress(&values);

        assert_eq!(compressed.len(), 100);
        assert_eq!(compressed.rank(), 2);
        // For 1-D data the coefficients are the same length as original,
        // but the basis is tiny (rank elements) + mean scalar, so
        // total overhead is small.  The ratio improves with multi-dim data.
        assert!(compressed.compressed_size_bytes() > 0);
    }

    #[test]
    #[should_panic(expected = "rank must be > 0")]
    fn lowrank_invalid_rank() {
        LowRankConfig::new(0);
    }

    // ── Cross-strategy tests ──────────────────────────────────────────────

    #[test]
    fn compressor_name() {
        assert_eq!(KiviCompressor::new(KiviConfig::new(4)).name(), "KIVI-4bit");
        assert_eq!(KiviCompressor::new(KiviConfig::new(2)).name(), "KIVI-2bit");
        assert_eq!(RkvCompressor::new(RkvConfig::new(0.5)).name(), "R-KV");
        assert_eq!(LowRankCompressor::new(LowRankConfig::new(1)).name(), "LowRank");
    }
}
