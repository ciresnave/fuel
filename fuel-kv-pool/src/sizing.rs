// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sizing a pool from a device budget (ROADMAP item 17, row H19).
//!
//! Before a pool exists the consumer has to pick `num_blocks`. Guessing it is
//! how a pool either wastes the card or takes the memory the weights and the
//! decode graph's own activations need and OOMs on the first step. This module
//! is the arithmetic that replaces the guess: given what the device can give
//! this process and what everything *else* will occupy, how many whole blocks
//! fit?
//!
//! **Mechanism, not policy** ([15-consumer-contract](../../docs/architecture/15-consumer-contract.md)):
//! the device capacity, the utilization fraction (C-5's "quantity" axis) and
//! the non-KV residency all come from the caller; nothing here decides how
//! much of the card a consumer *may* use. It is a pure function, so it is
//! testable without a device.
//!
//! The non-KV residency is the part that is easy to get wrong. A decode graph
//! analysed by `fuel_dispatch::residency::ResidencyPlanner` reports a peak that
//! already includes the pool buffers *if the pool is a graph leaf*, and the pool
//! size is exactly what we are solving for. [`non_kv_resident_bytes`] removes a
//! probe pool's bytes from a measured peak so the caller can analyse the graph
//! once with a small probe pool and size the real one from that.

use crate::KvGeometry;

/// What the device can give the pool, and what else will be resident.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvBudget {
    /// Device memory this process can draw on, in bytes (total, or currently
    /// free if other programs share the card: the caller's backend query).
    pub device_bytes: u64,
    /// Fraction of `device_bytes` this process may use, in `(0, 1]`. The
    /// consumer's entitlement; the remainder is headroom for the driver and
    /// other programs.
    pub utilization: f64,
    /// Bytes resident on the device besides the KV pool: weights, activations
    /// and workspace at their peak (see [`non_kv_resident_bytes`]).
    pub non_kv_resident_bytes: u64,
}

/// Why a pool could not be sized.
#[derive(Debug, Clone, PartialEq)]
pub enum KvSizingError {
    /// `utilization` was not a finite number in `(0, 1]`.
    InvalidUtilization(f64),
    /// The geometry has a zero-byte block (a zero layer count, block size,
    /// head count, head dim or element size).
    ZeroSizedBlock,
    /// Not even one block fits: the usable bytes, after the non-KV residency,
    /// are smaller than a block.
    NoRoomForABlock {
        /// `device_bytes * utilization`.
        usable_bytes: u64,
        /// The non-KV residency that was subtracted.
        non_kv_resident_bytes: u64,
        /// Bytes one block needs.
        block_bytes: u64,
    },
}

impl std::fmt::Display for KvSizingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidUtilization(u) => {
                write!(f, "KV pool sizing: utilization {u} is not in (0, 1]")
            }
            Self::ZeroSizedBlock => write!(f, "KV pool sizing: the geometry has a zero-byte block"),
            Self::NoRoomForABlock {
                usable_bytes,
                non_kv_resident_bytes,
                block_bytes,
            } => write!(
                f,
                "KV pool sizing: {usable_bytes} usable bytes minus {non_kv_resident_bytes} \
                 resident leaves no room for one {block_bytes}-byte block"
            ),
        }
    }
}

impl std::error::Error for KvSizingError {}

impl KvBudget {
    /// Bytes left for the KV pool: `floor(device_bytes * utilization)` minus the
    /// non-KV residency. Errors on an invalid utilization; saturates to zero
    /// (rather than wrapping) when the residency exceeds the usable bytes.
    pub fn kv_bytes(&self) -> Result<u64, KvSizingError> {
        Ok(self
            .usable_bytes()?
            .saturating_sub(self.non_kv_resident_bytes))
    }

    fn usable_bytes(&self) -> Result<u64, KvSizingError> {
        let u = self.utilization;
        if !u.is_finite() || u <= 0.0 || u > 1.0 {
            return Err(KvSizingError::InvalidUtilization(u));
        }
        // f64 has 53 mantissa bits; device sizes are far below 2^53 bytes.
        // The cast saturates, and u <= 1 keeps the product <= device_bytes.
        Ok(((self.device_bytes as f64) * u).floor() as u64)
    }
}

impl KvGeometry {
    /// Bytes one physical block occupies across all layers and both K and V.
    pub fn block_bytes(&self) -> u64 {
        self.bytes_per_block()
    }

    /// This geometry with `num_blocks` set to the most whole blocks that fit in
    /// `budget`. Never returns a geometry that overcommits: the result's pool
    /// bytes plus `budget.non_kv_resident_bytes` are at most the usable bytes.
    pub fn fit_to_budget(self, budget: &KvBudget) -> Result<KvGeometry, KvSizingError> {
        let block_bytes = self.block_bytes();
        if block_bytes == 0 {
            return Err(KvSizingError::ZeroSizedBlock);
        }
        let usable_bytes = budget.usable_bytes()?;
        let kv_bytes = usable_bytes.saturating_sub(budget.non_kv_resident_bytes);
        let blocks = kv_bytes / block_bytes;
        if blocks == 0 {
            return Err(KvSizingError::NoRoomForABlock {
                usable_bytes,
                non_kv_resident_bytes: budget.non_kv_resident_bytes,
                block_bytes,
            });
        }
        Ok(KvGeometry {
            num_blocks: usize::try_from(blocks).unwrap_or(usize::MAX),
            ..self
        })
    }
}

/// The non-KV part of a measured peak: `graph_peak_bytes` (a residency analysis
/// of the decode graph built with `probe`'s pool as graph leaves) minus the
/// probe pool's own bytes. Saturates at zero.
pub fn non_kv_resident_bytes(graph_peak_bytes: u64, probe: &KvGeometry) -> u64 {
    graph_peak_bytes.saturating_sub(probe.block_bytes().saturating_mul(probe.num_blocks as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom(num_blocks: usize) -> KvGeometry {
        // 2 layers * 16 tokens * 2 kv heads * 8 head_dim * 2 (K,V) * 4 bytes = 4096 B/block
        KvGeometry {
            n_layers: 2,
            num_blocks,
            block_size: 16,
            n_kv_heads: 2,
            head_dim: 8,
            elem_size: 4,
        }
    }

    #[test]
    fn block_bytes_is_the_documented_product() {
        assert_eq!(geom(1).block_bytes(), 4096);
    }

    #[test]
    fn fits_the_most_whole_blocks_and_never_overcommits() {
        // usable = 1_000_000 * 0.5 = 500_000; minus 100_000 resident = 400_000;
        // 400_000 / 4096 = 97 whole blocks (397_312 B).
        let b = KvBudget {
            device_bytes: 1_000_000,
            utilization: 0.5,
            non_kv_resident_bytes: 100_000,
        };
        let g = geom(1).fit_to_budget(&b).unwrap();
        assert_eq!(g.num_blocks, 97);
        let used = g.block_bytes() * g.num_blocks as u64 + b.non_kv_resident_bytes;
        assert!(used <= 500_000, "overcommitted: {used}");
        // One more block would have overcommitted: the result is maximal.
        assert!(used + g.block_bytes() > 500_000);
    }

    #[test]
    fn an_exact_fit_takes_every_block() {
        let b = KvBudget {
            device_bytes: 10 * 4096,
            utilization: 1.0,
            non_kv_resident_bytes: 0,
        };
        assert_eq!(geom(1).fit_to_budget(&b).unwrap().num_blocks, 10);
    }

    #[test]
    fn residency_beyond_the_usable_bytes_is_an_error_not_a_wrap() {
        let b = KvBudget {
            device_bytes: 1_000,
            utilization: 1.0,
            non_kv_resident_bytes: u64::MAX,
        };
        assert!(matches!(
            geom(1).fit_to_budget(&b),
            Err(KvSizingError::NoRoomForABlock { .. })
        ));
        assert_eq!(b.kv_bytes().unwrap(), 0);
    }

    #[test]
    fn less_than_one_block_of_room_is_an_error() {
        let b = KvBudget {
            device_bytes: 4095,
            utilization: 1.0,
            non_kv_resident_bytes: 0,
        };
        assert!(matches!(
            geom(1).fit_to_budget(&b),
            Err(KvSizingError::NoRoomForABlock { .. })
        ));
    }

    #[test]
    fn bad_utilization_and_zero_blocks_are_typed_errors() {
        for u in [0.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
            let b = KvBudget {
                device_bytes: 1 << 20,
                utilization: u,
                non_kv_resident_bytes: 0,
            };
            assert!(
                matches!(
                    geom(1).fit_to_budget(&b),
                    Err(KvSizingError::InvalidUtilization(_))
                ),
                "utilization {u} must be rejected"
            );
        }
        let mut zero = geom(1);
        zero.head_dim = 0;
        let b = KvBudget {
            device_bytes: 1 << 20,
            utilization: 1.0,
            non_kv_resident_bytes: 0,
        };
        assert_eq!(zero.fit_to_budget(&b), Err(KvSizingError::ZeroSizedBlock));
    }

    #[test]
    fn utilization_scales_the_pool_monotonically() {
        let blocks = |u: f64| {
            geom(1)
                .fit_to_budget(&KvBudget {
                    device_bytes: 1 << 24,
                    utilization: u,
                    non_kv_resident_bytes: 1 << 20,
                })
                .unwrap()
                .num_blocks
        };
        assert!(blocks(0.5) < blocks(0.75) && blocks(0.75) < blocks(0.9));
    }

    #[test]
    fn the_probe_pool_is_removed_from_a_measured_peak() {
        // Peak 1_000_000 with a 3-block (12_288 B) probe pool as graph leaves.
        assert_eq!(
            non_kv_resident_bytes(1_000_000, &geom(3)),
            1_000_000 - 12_288
        );
        // A peak smaller than the probe pool saturates instead of wrapping.
        assert_eq!(non_kv_resident_bytes(100, &geom(3)), 0);
    }
}
