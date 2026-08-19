// SPDX-License-Identifier: MIT OR Apache-2.0
//! Born-red for the `vec_dot_f32` target-dependent-semantics defect.
//!
//! The SIMD arm OVERWRITES `*c` (`vec_reduce` stores, then the leftover loop
//! accumulates), so it always yields `*c = dot(a, b)` regardless of `*c`'s
//! incoming value. The scalar fallback ACCUMULATES into `*c` and never
//! initialises it, so it yields `*c = incoming_c + dot(a, b)`. A caller that is
//! correct on avx2/neon therefore produces silent garbage on a target without
//! them.
//!
//! Contract (7 of the 8 arms in this crate already agree — the scalar
//! `vec_dot_f32` is the lone deviation, so this is a defect in one arm, not a
//! design choice between two): OVERWRITE — `vec_dot_f32` must set
//! `*c = dot(a, b)` regardless of `*c`'s incoming value. Corroborated by ggml's
//! `ggml_vec_dot_f32` (`*s = sumf`).
//!
//! Run the SIMD arm (default on an avx2 box):
//!     cargo test -p fuel-cpu-kernels --test vec_dot_f32_overwrite
//! Run the scalar arm:
//!     RUSTFLAGS="-C target-feature=-avx2,-avx" \
//!         cargo test -p fuel-cpu-kernels --test vec_dot_f32_overwrite
//! Positive control that the scalar arm actually compiled: under those flags,
//! `rustc --print cfg` must NOT list `target_feature="avx2"`. Without that
//! control a "born-red" run that silently used the SIMD arm would false-green.

use fuel_cpu_kernels::vec_dot_f32;

/// Assert overwrite semantics for a given length `k`.
///
/// Inputs are small integer-valued floats, so every product and every partial
/// sum is exact in f32 and SIMD lane-reassociation cannot change the result.
/// Consequently the ONLY thing that can move `c` away from the reference dot is
/// the accumulate-vs-overwrite semantics — which is exactly what is under test.
fn check_overwrite(k: usize) {
    let a: Vec<f32> = (0..k).map(|i| (i % 4 + 1) as f32).collect();
    let b: Vec<f32> = (0..k).map(|i| (i % 3 + 1) as f32).collect();
    let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

    // Non-zero: an overwrite discards it, an accumulate adds the dot on top.
    const SENTINEL: f32 = 123.0;
    let mut c = SENTINEL;
    // SAFETY: `a` and `b` each hold exactly `k` contiguous, initialised f32s;
    // `c` is a valid, initialised, exclusively-borrowed f32.
    unsafe { vec_dot_f32(a.as_ptr(), b.as_ptr(), &mut c, k) };

    assert_eq!(
        c,
        expected,
        "k={k}: vec_dot_f32 must OVERWRITE *c with the dot product; got {c}, \
         expected {expected} (an accumulate-into-uninitialised arm would give {})",
        SENTINEL + expected
    );
}

#[test]
fn vec_dot_f32_overwrites_c_across_step_boundary() {
    // Record which arm compiled, so the run log is self-describing.
    let arm = if cfg!(any(target_feature = "avx2", target_feature = "neon")) {
        "SIMD"
    } else {
        "scalar"
    };
    eprintln!("vec_dot_f32 arm under test: {arm}");

    // STEP = 32. Exercise all three regions of the SIMD arm and their scalar
    // equivalents:
    //   k = 32  exact multiple of STEP  -> leftover loop empty (store only)
    //   k = 40  multiple + tail         -> store, then 8-element leftover
    //   k = 5   k < STEP                -> np = 0, store of 0, all-leftover
    check_overwrite(32);
    check_overwrite(40);
    check_overwrite(5);
}
