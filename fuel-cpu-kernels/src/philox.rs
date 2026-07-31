//! Philox-4x32-10 counter-based RNG, and Fuel's `RandomBits` counter derivation.
//!
//! Design: `docs/superpowers/specs/2026-07-31-rng-generator-seam-design.md`.
//!
//! This is **increment 1** of the RNG seam: a plain function plus the §8 counter
//! mapping, gated on the published Random123 vectors. There is no `Op::RandomBits`
//! here — the basis addition is gated on a KISS RFC (spec §11).
//!
//! Philox is *counter-based*: the output is a pure function of `(counter, key)`
//! with no internal state. That is what makes the eventual graph op CSE-sound,
//! replay byte-exact, and forward/backward mask-consistent for free (spec §1, §3).

/// Philox-4x32 round constants (Random123).
const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
/// Weyl sequence key bumps: golden ratio and sqrt(3)-1 in 32-bit fixed point.
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

/// 32x32 -> 64 multiply, split into (hi, lo).
#[inline(always)]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = (a as u64) * (b as u64);
    ((p >> 32) as u32, p as u32)
}

/// One Philox-4x32 round.
#[inline(always)]
fn philox4x32_round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(PHILOX_M0, ctr[0]);
    let (hi1, lo1) = mulhilo(PHILOX_M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// Bump the key by the Weyl constants (applied between rounds).
#[inline(always)]
fn philox4x32_bumpkey(key: [u32; 2]) -> [u32; 2] {
    [key[0].wrapping_add(PHILOX_W0), key[1].wrapping_add(PHILOX_W1)]
}

/// Philox-4x32-10: ten rounds over a 4x32 counter with a 2x32 key.
///
/// Pure: the same `(ctr, key)` always yields the same four words, on every
/// backend. Conformance is gated on the upstream Random123 vectors
/// ([`crate::philox_kat::PHILOX4X32_10_KAT`]) — see spec §10.
#[inline]
pub fn philox4x32_10(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    philox4x32_r::<10>(ctr, key)
}

/// Philox-4x32 with `R` rounds.
///
/// Round 1 uses the key as given; rounds `2..=R` bump the key first. So `R`
/// rounds perform `R - 1` bumps — the schedule Random123 specifies, and the
/// detail the published vectors exist to pin.
#[inline]
pub fn philox4x32_r<const R: usize>(mut ctr: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    if R == 0 {
        return ctr;
    }
    ctr = philox4x32_round(ctr, key);
    for _ in 1..R {
        key = philox4x32_bumpkey(key);
        ctr = philox4x32_round(ctr, key);
    }
    ctr
}

// ---------------------------------------------------------------------------
// Fuel's `RandomBits` counter derivation — spec §8 (normative).
// ---------------------------------------------------------------------------

/// Split a `u64` seed into Philox's 2x32 key, little-endian words.
///
/// The same little-endian rule as [`derive_counter`]'s block split, deliberately:
/// one convention to remember rather than two.
#[inline]
pub fn key_from_seed(seed: u64) -> [u32; 2] {
    [seed as u32, (seed >> 32) as u32]
}

/// Build the Philox counter for a logical element, per spec §8.
///
/// ```text
/// counter[0] = block_lo    block = linear_index / 4, little-endian split
/// counter[1] = block_hi
/// counter[2] = base        runtime-bound scalar (a per-(seed, stream) STEP index)
/// counter[3] = stream      build-time stream id
/// ```
///
/// `linear_index` is the element's **logical** row-major (C-order, last axis
/// fastest) position in the op's declared output shape — never a physical
/// iteration order, and never a partition- or rank-local index (spec §9).
///
/// `block_lo` occupies `counter[0]` so that advancing `block_index` by one is
/// byte-identical to a single Random123 `incr()`: the carry runs
/// `counter[0] -> counter[1]` and structurally cannot reach `base` or `stream`,
/// so distinct `(base, stream)` own provably disjoint counter spaces.
#[inline]
pub fn derive_counter(linear_index: u64, base: u32, stream: u32) -> [u32; 4] {
    let block = linear_index / 4;
    [block as u32, (block >> 32) as u32, base, stream]
}

/// The `U32` output word for one logical element — the whole §8 mapping.
///
/// Evaluates one Philox block and selects lane `linear_index % 4`, in
/// Random123's published output order. Defined element-wise; an implementation
/// should walk 4-aligned logical blocks (one Philox eval per four outputs)
/// rather than calling this per element, which would run the 10-round core four
/// times and discard three lanes.
#[inline]
pub fn random_bits_word(linear_index: u64, seed: u64, base: u32, stream: u32) -> u32 {
    let out = philox4x32_10(derive_counter(linear_index, base, stream), key_from_seed(seed));
    out[(linear_index % 4) as usize]
}

/// Fill `dst` with `RandomBits` output for logical indices
/// `start_index .. start_index + dst.len()`.
///
/// `start_index` is the **global** logical index of `dst[0]` — for a sharded
/// tensor that is `shard_start + local_offset`, never `0` (spec §9).
pub fn random_bits_fill(dst: &mut [u32], start_index: u64, seed: u64, base: u32, stream: u32) {
    let key = key_from_seed(seed);
    let mut i = 0usize;
    while i < dst.len() {
        let gi = start_index + i as u64;
        let lane = (gi % 4) as usize;
        let block = philox4x32_10(derive_counter(gi, base, stream), key);
        // Emit the rest of this 4-lane block, or the rest of dst.
        let take = core::cmp::min(4 - lane, dst.len() - i);
        dst[i..i + take].copy_from_slice(&block[lane..lane + take]);
        i += take;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::philox_kat::PHILOX4X32_10_KAT;

    /// Stock Random123-style 128-bit counter increment, `counter[0]` fastest.
    /// The reference the §8 layout was chosen to agree with.
    fn incr(mut c: [u32; 4]) -> [u32; 4] {
        for w in c.iter_mut() {
            *w = w.wrapping_add(1);
            if *w != 0 {
                break;
            }
        }
        c
    }

    /// **Structural** assertion, not a value one: advancing `block_index` by one
    /// must equal exactly one stock `incr()` of the counter. This is the
    /// invariant the `ctr_lo -> counter[0]` little-endian layout was chosen for,
    /// so a wrong endianness or slot order fails *here* — naming the invariant —
    /// rather than as a mystery byte mismatch at some element.
    ///
    /// Includes the `block_lo` overflow boundary, which is precisely where a
    /// reversed split would diverge while looking fine everywhere else.
    #[test]
    fn block_advance_equals_one_counter_incr() {
        for &(base, stream) in &[(0u32, 0u32), (7, 3), (0xDEAD_BEEF, 0x0BAD_F00D)] {
            for &block in &[0u64, 1, 2, 0xFFFF_FFFE, 0xFFFF_FFFF, 0x1_0000_0000, 0x1_0000_0001] {
                let c_n = derive_counter(block * 4, base, stream);
                let c_n1 = derive_counter((block + 1) * 4, base, stream);
                assert_eq!(
                    c_n1,
                    incr(c_n),
                    "block {block} -> {} must be one incr(); base={base:#x} stream={stream:#x}",
                    block + 1
                );
            }
        }
    }

    /// The carry from `block` must never reach `base` or `stream` — that is what
    /// makes distinct `(base, stream)` provably disjoint rather than
    /// disjoint-by-inspection.
    #[test]
    fn block_carry_cannot_reach_base_or_stream() {
        // Largest representable logical index => largest block => both block
        // words saturated. base/stream must still be untouched.
        let c = derive_counter(u64::MAX, 0xAAAA_AAAA, 0x5555_5555);
        assert_eq!(c[0], 0xFFFF_FFFF, "block_lo should be saturated here");
        assert_eq!(c[1], 0x3FFF_FFFF, "block_hi = (u64::MAX / 4) >> 32");
        assert_eq!(c[2], 0xAAAA_AAAA, "base perturbed by block");
        assert_eq!(c[3], 0x5555_5555, "stream perturbed by block");
    }

    /// **Non-circular join** between the mapping layer and the algorithm anchor.
    ///
    /// With `seed = 0, base = 0, stream = 0`, elements 0..=3 sit in block 0, so
    /// the derived counter is `[0,0,0,0]` and the key is `[0,0]` — which is
    /// literally the first upstream KAT vector. So the mapping's first four
    /// outputs must BE that vector's four words, in order. This ties §8 to the
    /// published anchor without re-deriving Philox.
    #[test]
    fn element_0_3_are_the_all_zero_kat_words() {
        let (ctr, key, expected) = PHILOX4X32_10_KAT[0];
        assert_eq!(ctr, [0, 0, 0, 0], "KAT[0] is expected to be the all-zero vector");
        assert_eq!(key, [0, 0]);
        for i in 0..4u64 {
            assert_eq!(
                random_bits_word(i, 0, 0, 0),
                expected[i as usize],
                "element {i} must be lane {i} of the all-zero KAT vector",
            );
        }
    }

    /// `random_bits_fill` must agree with per-element evaluation for every
    /// start alignment — the blocked walk is an optimization, never a semantic.
    #[test]
    fn fill_matches_elementwise_at_every_alignment() {
        for start in 0..9u64 {
            for len in 0..13usize {
                let mut got = vec![0u32; len];
                random_bits_fill(&mut got, start, 0x0123_4567_89AB_CDEF, 42, 7);
                let want: Vec<u32> = (0..len as u64)
                    .map(|k| random_bits_word(start + k, 0x0123_4567_89AB_CDEF, 42, 7))
                    .collect();
                assert_eq!(got, want, "start={start} len={len}");
            }
        }
    }

    /// The anchor. 55 vectors: 3 canonical (all-zero, all-one, digits-of-pi)
    /// from upstream `tests/kat_vectors`, plus 52 systematic single-bit and
    /// structured inputs from `tests/old_kat_vectors`.
    ///
    /// A wrong round schedule cannot match even the three canonical vectors by
    /// accident — 384 bits of 10-round avalanche. The 52 exist for the
    /// cross-implementation diff, where a *shared* misreading of the spec is the
    /// failure class a narrow corpus cannot catch.
    #[test]
    fn philox4x32_10_matches_upstream_kat() {
        assert_eq!(PHILOX4X32_10_KAT.len(), 55, "corpus size changed unexpectedly");
        let mut failures = Vec::new();
        for (i, (ctr, key, expected)) in PHILOX4X32_10_KAT.iter().enumerate() {
            let got = philox4x32_10(*ctr, *key);
            if got != *expected {
                failures.push(format!(
                    "  [{i}] ctr={ctr:08x?} key={key:08x?}\n       want {expected:08x?}\n       got  {got:08x?}"
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} upstream vectors mismatched:\n{}",
            failures.len(),
            PHILOX4X32_10_KAT.len(),
            failures.join("\n"),
        );
    }
}
