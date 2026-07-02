//! `CostVector` — the per-path cost VECTOR the rankers produce, and
//! the Pareto-dominance relation over it.
//!
//! Phase B PR-B1 of the "plan IS the graph" rebuild. This replaces the
//! scalar `composite_ns` *sort key* with a structured vector, per
//! `docs/architecture/04-optimization.md` §"Rankers and the cost
//! model: a per-path cost vector, ranked" and decisions-log #7/#10.
//!
//! # The axes (kept low-dimensional and discrete on purpose)
//!
//! Keeping the ranker vector low-dimensional — ONE central time
//! metric, memory as discrete tiers, discrete precision/accuracy — is
//! what keeps the per-device Pareto frontier naturally small (~10²
//! paths across a deep model) and lossless. See §"Bounding the
//! frontier: Pareto per device + crowding cap".
//!
//! - **`time`** — ONE central wall-clock nanosecond metric. For B1
//!   this is exactly the existing scalar:
//!   `composite_ns(&static_cost).saturating_add(inbound_transfer_ns)`.
//!   `t_min` ("fastest best case") is explicitly **NOT** an axis
//!   (decisions-log #7/#10). The throughput-median-vs-latency-p99
//!   *mode selection* the constitution describes is a later
//!   refinement; for B1 there is one `time` number and the Judge's
//!   distribution doesn't reach this struct yet.
//! - **memory, per tier** — host-RAM and device-VRAM footprints
//!   tracked separately (decisions-log #10: "memory as discrete
//!   tiers"), attributed to the candidate's placement tier. Which
//!   tier *binds* depends on the target machine, so they cannot be
//!   collapsed into one number. Disk is reserved (the Layer-1
//!   [`CostEstimate`] exposes no disk-footprint axis yet).
//! - **`precision`** (digits) — derived from the candidate's
//!   [`PrecisionGuarantee`]. Higher is better.
//! - **`accuracy`** (ULP / rounding / monotonicity descriptor) —
//!   derived from the candidate's [`PrecisionGuarantee`]. Higher is
//!   better.
//!
//! # Orientation (documented once, used everywhere)
//!
//! Every axis is oriented so that **lower `time`, lower memory,
//! higher `precision`, higher `accuracy` is better.** [`dominates`]
//! and the tie-break order both read this orientation.
//!
//! [`dominates`]: CostVector::dominates
//! [`CostEstimate`]: crate::fused::CostEstimate
//! [`PrecisionGuarantee`]: crate::fused::PrecisionGuarantee

use fuel_ir::DeviceLocation;

use crate::fused::{CostEstimate, PrecisionGuarantee};

use super::candidate::Candidate;
use super::cost::{composite_ns, default_backend_rates};

/// Discrete precision level in **decimal digits** — higher is better.
///
/// Derived from a [`PrecisionGuarantee`]'s relative-error bound when
/// one is stated (`digits ≈ floor(-log10(max_relative))`), otherwise
/// inferred from the strongest qualitative claim the guarantee makes.
/// Kept as a small integer (not a continuous float) so the Pareto
/// frontier stays discrete and small.
pub type PrecisionDigits = u16;

/// Discrete accuracy level — higher is better. Summarizes the
/// guarantee's ULP / rounding / monotonicity character into one
/// ordered descriptor so it can sit on the cost vector as a discrete
/// axis. Higher means tighter-rounded / more deterministic.
///
/// The ladder is intentionally coarse (a handful of rungs), matching
/// the constitution's "discrete accuracy levels."
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccuracyClass {
    /// No static bound and not bit-stable — the weakest claim
    /// (e.g. scheduler-dependent subgroup reductions).
    Unbounded,
    /// Bit-stable on same hardware but no static ULP bound stated.
    Stable,
    /// A loose ULP bound is claimed (> 2 ULP) — vendor transcendental
    /// territory.
    BoundedLoose,
    /// A tight ULP bound is claimed (1–2 ULP) — IEEE-754 elementwise.
    BoundedTight,
    /// Correctly-rounded: ULP = 0 (the reference grade).
    CorrectlyRounded,
}

impl AccuracyClass {
    /// Map to a small ordered integer (higher = better) for the
    /// tie-break / dominance arithmetic.
    pub fn rank(self) -> u8 {
        match self {
            AccuracyClass::Unbounded => 0,
            AccuracyClass::Stable => 1,
            AccuracyClass::BoundedLoose => 2,
            AccuracyClass::BoundedTight => 3,
            AccuracyClass::CorrectlyRounded => 4,
        }
    }
}

/// Per-tier memory footprint of a candidate's placement, in bytes.
///
/// Tracked per tier (not collapsed) because which tier binds depends
/// on the target machine — a fast-but-VRAM-heavy path and a slow-but-
/// host-light path are both legitimately Pareto-optimal. Disk is
/// reserved for when the Layer-1 cost model grows a disk-footprint
/// axis; today's [`CostEstimate`] has none, so it stays zero.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryTiers {
    /// Host-RAM footprint (bytes) attributed to a CPU-tier placement.
    pub host_ram_bytes: u64,
    /// Device-VRAM footprint (bytes) attributed to a GPU-tier
    /// placement (CUDA / Vulkan / Metal).
    pub device_vram_bytes: u64,
}

impl MemoryTiers {
    /// Attribute `bytes` to the tier implied by `device`: GPU devices
    /// land on VRAM, the CPU on host RAM. The Layer-1 cost model has
    /// one footprint number (`bytes_moved`); placement decides which
    /// tier it loads.
    pub fn for_placement(device: DeviceLocation, bytes: u64) -> Self {
        match device {
            DeviceLocation::Cpu => MemoryTiers {
                host_ram_bytes: bytes,
                device_vram_bytes: 0,
            },
            DeviceLocation::Cuda { .. }
            | DeviceLocation::Vulkan { .. }
            | DeviceLocation::Metal { .. } => MemoryTiers {
                host_ram_bytes: 0,
                device_vram_bytes: bytes,
            },
        }
    }

    /// `true` iff `self` is no worse than `other` on every tier
    /// (lower-or-equal bytes). Helper for [`CostVector::dominates`].
    fn no_worse_than(&self, other: &MemoryTiers) -> bool {
        self.host_ram_bytes <= other.host_ram_bytes
            && self.device_vram_bytes <= other.device_vram_bytes
    }

    /// `true` iff `self` is strictly better than `other` on at least
    /// one tier (strictly fewer bytes).
    fn strictly_better_on_some(&self, other: &MemoryTiers) -> bool {
        self.host_ram_bytes < other.host_ram_bytes
            || self.device_vram_bytes < other.device_vram_bytes
    }
}

/// The per-path cost vector the rankers produce. Pareto dominance
/// (the per-device frontier in PR-B2) is over this whole vector;
/// ties break **precision → accuracy → memory** (decisions-log #7).
///
/// Orientation: **lower `time`, lower `memory`, higher `precision`,
/// higher `accuracy` is better.**
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CostVector {
    /// ONE central wall-clock metric (ns). For B1: the old scalar
    /// `composite_ns(static_cost) + inbound_transfer_ns`. Lower is
    /// better. `t_min` is deliberately not an axis.
    pub time: u64,
    /// Per-tier memory footprint. Lower is better, per tier.
    pub memory: MemoryTiers,
    /// Numerical precision in decimal digits. Higher is better.
    pub precision: PrecisionDigits,
    /// Accuracy descriptor (ULP / rounding / monotonicity). Higher is
    /// better.
    pub accuracy: AccuracyClass,
}

impl CostVector {
    /// Build a cost vector from a candidate.
    ///
    /// - `time` = [`composite_ns`]`(&static_cost) + inbound_transfer_ns`
    ///   (saturating) — the exact B1 scalar.
    /// - `memory` = the candidate's `static_cost.bytes_moved`
    ///   attributed to the placement tier of `candidate.device`.
    /// - `precision` / `accuracy` are derived from
    ///   `candidate.precision`.
    pub fn from_candidate(candidate: &Candidate) -> Self {
        // The rank sees only `Candidate::backend` (no registered caps in
        // hand), so it uses the per-backend throughput prior — kept
        // consistent with the placement DP's authoritative caps figure
        // via `default_backend_rates`.
        let (compute_rate, mem_bandwidth) = default_backend_rates(candidate.backend);
        let time = composite_ns(&candidate.static_cost, compute_rate, mem_bandwidth)
            .saturating_add(candidate.inbound_transfer_ns);
        let memory =
            MemoryTiers::for_placement(candidate.device, candidate.static_cost.bytes_moved);
        CostVector {
            time,
            memory,
            precision: precision_digits(&candidate.precision),
            accuracy: accuracy_class(&candidate.precision),
        }
    }

    /// Pareto dominance over the whole vector.
    ///
    /// Returns `true` iff `self` is **no worse than `other` on every
    /// axis** AND **strictly better on at least one** — the standard
    /// Pareto relation. Per the documented orientation: lower `time`,
    /// lower memory (per tier), higher `precision`, higher `accuracy`
    /// is better.
    ///
    /// This is the relation PR-B2's per-device Pareto frontier will
    /// filter on. It is intentionally *not* what selects the winner —
    /// the winner stays time-first (see
    /// [`super::alternative_set::AlternativeSet::rank_by_cost`]) so
    /// realize behavior is preserved.
    pub fn dominates(&self, other: &CostVector) -> bool {
        let no_worse = self.time <= other.time
            && self.memory.no_worse_than(&other.memory)
            && self.precision >= other.precision
            && self.accuracy >= other.accuracy;
        if !no_worse {
            return false;
        }
        let strictly_better = self.time < other.time
            || self.memory.strictly_better_on_some(&other.memory)
            || self.precision > other.precision
            || self.accuracy > other.accuracy;
        strictly_better
    }

    /// Total-order key that keeps the **winner time-first**: the
    /// lowest-`time` candidate wins (preserving the old `composite_ns`
    /// winner), with ties broken **precision → accuracy → memory**
    /// (the constitution's order). Memory ties break on total bytes
    /// (host + VRAM, saturating), lower first.
    ///
    /// Precision and accuracy are oriented "higher is better," so the
    /// key negates them (via `Reverse`-style inversion) to sort
    /// descending while `time` and memory sort ascending.
    ///
    /// Winner vs frontier: the winner (index 0 / arm-0) uses THIS
    /// time-first order for realize-behavior preservation; the
    /// *frontier retention* (PR-B2) will use [`Self::dominates`].
    pub fn total_order_key(&self) -> (u64, u16, u8, u64) {
        let total_memory = self
            .memory
            .host_ram_bytes
            .saturating_add(self.memory.device_vram_bytes);
        (
            self.time,
            // Higher precision should sort earlier → invert.
            PrecisionDigits::MAX - self.precision,
            // Higher accuracy should sort earlier → invert.
            u8::MAX - self.accuracy.rank(),
            total_memory,
        )
    }
}

/// Derive a discrete decimal-digit precision from a guarantee.
///
/// Priority:
/// 1. A stated relative-error bound → `floor(-log10(max_relative))`
///    (clamped to a sane ceiling); `max_relative == 0` means
///    correctly-rounded → the ceiling.
/// 2. ULP = 0 → correctly-rounded → the ceiling.
/// 3. Bit-stable with no bound → a mid rung (deterministic but
///    unquantified).
/// 4. Otherwise (UNAUDITED / scheduler-dependent) → 0.
///
/// The ceiling (`MAX_DIGITS`) keeps the axis discrete and bounded;
/// f64 carries ~15-16 significant digits, so 18 is a comfortable cap
/// that never under-reports a correctly-rounded kernel.
fn precision_digits(p: &PrecisionGuarantee) -> PrecisionDigits {
    const MAX_DIGITS: PrecisionDigits = 18;
    if let Some(rel) = p.max_relative {
        if rel <= 0.0 {
            return MAX_DIGITS;
        }
        let digits = (-rel.log10()).floor();
        if digits.is_finite() && digits > 0.0 {
            return (digits as u32).min(MAX_DIGITS as u32) as PrecisionDigits;
        }
        return 0;
    }
    if p.max_ulp == Some(0) {
        return MAX_DIGITS;
    }
    if p.bit_stable_on_same_hardware {
        // Deterministic but no quantified bound — a mid rung so a
        // bit-stable kernel outranks an unaudited one without
        // claiming reference-grade digits it never promised.
        return MAX_DIGITS / 2;
    }
    0
}

/// Derive the discrete [`AccuracyClass`] from a guarantee's ULP /
/// rounding / stability character.
fn accuracy_class(p: &PrecisionGuarantee) -> AccuracyClass {
    match p.max_ulp {
        Some(0) => AccuracyClass::CorrectlyRounded,
        Some(1..=2) => AccuracyClass::BoundedTight,
        Some(_) => AccuracyClass::BoundedLoose,
        None => {
            if p.bit_stable_on_same_hardware {
                AccuracyClass::Stable
            } else {
                AccuracyClass::Unbounded
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::{CostEstimate, PrecisionGuarantee};
    use crate::kernel::{KernelCaps, OpParams};
    use fuel_ir::probe::BackendId;
    use fuel_ir::{DeviceLocation, Layout, Result};
    use fuel_memory::Storage;
    use std::sync::{Arc, RwLock};

    fn noop(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn candidate(
        device: DeviceLocation,
        cost: CostEstimate,
        inbound: u64,
        precision: PrecisionGuarantee,
    ) -> Candidate {
        Candidate {
            kernel: noop,
            caps: KernelCaps::empty(),
            backend: BackendId::Cpu,
            device,
            precision,
            static_cost: cost,
            inbound_transfer_ns: inbound,
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "",
        }
    }

    // ===== (a) CostVector is built correctly from a CostEstimate +
    //            PrecisionGuarantee, axis by axis. =====

    #[test]
    fn cost_vector_time_axis_matches_old_scalar() {
        // time == composite_ns(static_cost) + inbound_transfer_ns.
        let cost = CostEstimate {
            flops: 1000,
            bytes_moved: 4000,
            kernel_overhead_ns: 50,
        };
        let c = candidate(
            DeviceLocation::Cpu,
            cost,
            777,
            PrecisionGuarantee::REFERENCE,
        );
        let v = CostVector::from_candidate(&c);
        // composite_ns: max(1000, 4000/4=1000) + 50 = 1050; + 777.
        assert_eq!(v.time, 1050 + 777);
    }

    #[test]
    fn cost_vector_memory_attributed_to_cpu_host_tier() {
        let cost = CostEstimate {
            flops: 0,
            bytes_moved: 8192,
            kernel_overhead_ns: 0,
        };
        let c = candidate(
            DeviceLocation::Cpu,
            cost,
            0,
            PrecisionGuarantee::REFERENCE,
        );
        let v = CostVector::from_candidate(&c);
        assert_eq!(v.memory.host_ram_bytes, 8192, "CPU loads host RAM");
        assert_eq!(v.memory.device_vram_bytes, 0, "no VRAM on CPU");
    }

    #[test]
    fn cost_vector_memory_attributed_to_gpu_vram_tier() {
        let cost = CostEstimate {
            flops: 0,
            bytes_moved: 8192,
            kernel_overhead_ns: 0,
        };
        let c = candidate(
            DeviceLocation::Cuda { gpu_id: 0 },
            cost,
            0,
            PrecisionGuarantee::REFERENCE,
        );
        let v = CostVector::from_candidate(&c);
        assert_eq!(v.memory.device_vram_bytes, 8192, "GPU loads VRAM");
        assert_eq!(v.memory.host_ram_bytes, 0, "no host RAM on GPU");
    }

    #[test]
    fn cost_vector_precision_and_accuracy_from_guarantee() {
        // REFERENCE: max_ulp Some(0), max_relative Some(0.0) →
        // correctly-rounded + max digits.
        let reference = candidate(
            DeviceLocation::Cpu,
            CostEstimate::default(),
            0,
            PrecisionGuarantee::REFERENCE,
        );
        let v = CostVector::from_candidate(&reference);
        assert_eq!(v.accuracy, AccuracyClass::CorrectlyRounded);
        assert_eq!(v.precision, 18, "ULP=0 / rel=0 → max digits");

        // PRIMITIVE_DETERMINISTIC_CPU: bit-stable, no ULP bound →
        // Stable accuracy, mid precision digits.
        let prim = candidate(
            DeviceLocation::Cpu,
            CostEstimate::default(),
            0,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        let v = CostVector::from_candidate(&prim);
        assert_eq!(v.accuracy, AccuracyClass::Stable);
        assert_eq!(v.precision, 9, "bit-stable-no-bound → mid rung");

        // UNAUDITED: not bit-stable, no bound → weakest.
        let unaudited = candidate(
            DeviceLocation::Cpu,
            CostEstimate::default(),
            0,
            PrecisionGuarantee::UNAUDITED,
        );
        let v = CostVector::from_candidate(&unaudited);
        assert_eq!(v.accuracy, AccuracyClass::Unbounded);
        assert_eq!(v.precision, 0);

        // A loose ULP bound (4) → BoundedLoose.
        let loose = PrecisionGuarantee {
            bit_stable_on_same_hardware: true,
            max_ulp: Some(4),
            max_relative: None,
            max_absolute: None,
            notes: "loose",
        };
        let v = CostVector::from_candidate(&candidate(
            DeviceLocation::Cpu,
            CostEstimate::default(),
            0,
            loose,
        ));
        assert_eq!(v.accuracy, AccuracyClass::BoundedLoose);

        // A tight ULP bound (1) → BoundedTight.
        let tight = PrecisionGuarantee {
            bit_stable_on_same_hardware: true,
            max_ulp: Some(1),
            max_relative: None,
            max_absolute: None,
            notes: "tight",
        };
        let v = CostVector::from_candidate(&candidate(
            DeviceLocation::Cpu,
            CostEstimate::default(),
            0,
            tight,
        ));
        assert_eq!(v.accuracy, AccuracyClass::BoundedTight);
    }

    // ===== (b) dominates: clearly-dominated pair → true; genuinely
    //            non-dominated pair → false both ways. =====

    #[test]
    fn dominates_clearly_dominated_pair() {
        // a: faster, less memory, higher precision/accuracy than b →
        // a dominates b, and b does NOT dominate a.
        let a = CostVector {
            time: 100,
            memory: MemoryTiers {
                host_ram_bytes: 10,
                device_vram_bytes: 0,
            },
            precision: 10,
            accuracy: AccuracyClass::CorrectlyRounded,
        };
        let b = CostVector {
            time: 200,
            memory: MemoryTiers {
                host_ram_bytes: 20,
                device_vram_bytes: 0,
            },
            precision: 5,
            accuracy: AccuracyClass::Stable,
        };
        assert!(a.dominates(&b), "a strictly better on every axis");
        assert!(!b.dominates(&a), "dominated cannot dominate back");
    }

    #[test]
    fn dominates_non_dominated_pair_is_false_both_ways() {
        // Faster-but-lower-precision vs slower-but-higher-precision —
        // a genuine tradeoff; neither dominates.
        let fast_low_prec = CostVector {
            time: 100,
            memory: MemoryTiers::default(),
            precision: 4,
            accuracy: AccuracyClass::BoundedLoose,
        };
        let slow_high_prec = CostVector {
            time: 500,
            memory: MemoryTiers::default(),
            precision: 12,
            accuracy: AccuracyClass::CorrectlyRounded,
        };
        assert!(!fast_low_prec.dominates(&slow_high_prec));
        assert!(!slow_high_prec.dominates(&fast_low_prec));
    }

    #[test]
    fn dominates_equal_vectors_is_false() {
        // Equal on all axes → no strict improvement → no dominance.
        let a = CostVector {
            time: 100,
            memory: MemoryTiers {
                host_ram_bytes: 5,
                device_vram_bytes: 5,
            },
            precision: 7,
            accuracy: AccuracyClass::Stable,
        };
        assert!(!a.dominates(&a));
    }

    #[test]
    fn dominates_memory_tradeoff_across_tiers_is_non_dominated() {
        // Same time/precision/accuracy, but one is lighter on host
        // and heavier on VRAM and vice-versa → genuine cross-tier
        // tradeoff, neither dominates.
        let host_light = CostVector {
            time: 100,
            memory: MemoryTiers {
                host_ram_bytes: 1,
                device_vram_bytes: 100,
            },
            precision: 7,
            accuracy: AccuracyClass::Stable,
        };
        let vram_light = CostVector {
            time: 100,
            memory: MemoryTiers {
                host_ram_bytes: 100,
                device_vram_bytes: 1,
            },
            precision: 7,
            accuracy: AccuracyClass::Stable,
        };
        assert!(!host_light.dominates(&vram_light));
        assert!(!vram_light.dominates(&host_light));
    }

    // ===== (c) tie-break precision → accuracy → memory among
    //            equal-time candidates (via total_order_key). =====

    #[test]
    fn tie_break_precision_first_among_equal_time() {
        let lower_prec = CostVector {
            time: 100,
            memory: MemoryTiers::default(),
            precision: 5,
            accuracy: AccuracyClass::CorrectlyRounded,
        };
        let higher_prec = CostVector {
            time: 100,
            memory: MemoryTiers::default(),
            precision: 9,
            accuracy: AccuracyClass::Unbounded,
        };
        // Higher precision sorts FIRST even though its accuracy is
        // worse — precision dominates the tie-break.
        assert!(
            higher_prec.total_order_key() < lower_prec.total_order_key(),
            "precision is the first tie-breaker",
        );
    }

    #[test]
    fn tie_break_accuracy_then_memory() {
        // Equal time + precision → accuracy decides.
        let lo_acc = CostVector {
            time: 100,
            memory: MemoryTiers::default(),
            precision: 7,
            accuracy: AccuracyClass::Stable,
        };
        let hi_acc = CostVector {
            time: 100,
            memory: MemoryTiers::default(),
            precision: 7,
            accuracy: AccuracyClass::CorrectlyRounded,
        };
        assert!(hi_acc.total_order_key() < lo_acc.total_order_key());

        // Equal time + precision + accuracy → memory decides
        // (lower total bytes first).
        let heavy = CostVector {
            time: 100,
            memory: MemoryTiers {
                host_ram_bytes: 1000,
                device_vram_bytes: 0,
            },
            precision: 7,
            accuracy: AccuracyClass::Stable,
        };
        let light = CostVector {
            time: 100,
            memory: MemoryTiers {
                host_ram_bytes: 10,
                device_vram_bytes: 0,
            },
            precision: 7,
            accuracy: AccuracyClass::Stable,
        };
        assert!(light.total_order_key() < heavy.total_order_key());
    }

    #[test]
    fn total_order_key_is_time_first() {
        // Lower time wins regardless of every other axis — the
        // realize-behavior-preserving contract.
        let fast_but_worse = CostVector {
            time: 100,
            memory: MemoryTiers {
                host_ram_bytes: u64::MAX,
                device_vram_bytes: 0,
            },
            precision: 0,
            accuracy: AccuracyClass::Unbounded,
        };
        let slow_but_better = CostVector {
            time: 101,
            memory: MemoryTiers::default(),
            precision: 18,
            accuracy: AccuracyClass::CorrectlyRounded,
        };
        assert!(
            fast_but_worse.total_order_key() < slow_but_better.total_order_key(),
            "time is the primary key",
        );
    }
}
