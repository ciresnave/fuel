// SPDX-License-Identifier: MIT OR Apache-2.0
//! Correctness and GPU time of the M = 1 f32 matmul kernels at the Qwen3-0.6B
//! decode shapes: the original per-column-workgroup `matvec` against the
//! column-parallel `matvec_cols` (ROADMAP item 22 follow-up: the M = 1 matvec was
//! 95% of decode GPU time at about 11% of nominal memory bandwidth).
//!
//! For each `(K, N)`: both kernels are checked against an f64 CPU reference, then
//! timed with the Vulkan timestamp queries over a batch of dispatches that cycle
//! through distinct right-hand buffers (so the weights do not sit in L2 between
//! dispatches, as in a decode step). Prints GPU microseconds per dispatch and the
//! effective bandwidth over the right operand.
//!
//! ```sh
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release \
//!     -p fuel-vulkan-backend --example matvec_probe
//! ```
//!
//! An example rather than an `#[ignore]` test (the ignore total sits at its
//! ceiling). Exits non-zero if either kernel is off its reference.

use fuel_vulkan_backend::{BATCH_TOTAL, DeviceSelection, VulkanBackend, VulkanStorageBytes};
use std::sync::Arc;

fn bytes_of(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn floats_of(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 16_777_216.0) - 0.5
        })
        .collect()
}

fn reference(a: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0f64; n];
    for kk in 0..k {
        let ak = a[kk] as f64;
        let row = &b[kk * n..(kk + 1) * n];
        for (o, &w) in out.iter_mut().zip(row) {
            *o += ak * w as f64;
        }
    }
    out
}

fn run(
    be: &VulkanBackend,
    a: &VulkanStorageBytes,
    b: &VulkanStorageBytes,
    out: &mut VulkanStorageBytes,
    k: usize,
    n: usize,
) {
    be.matmul_f32_bytes(a, b, out, &[], &[], 1, n, k).unwrap();
}

/// GPU microseconds per dispatch over `reps` passes through all of `bs`.
fn gpu_us(
    be: &VulkanBackend,
    a: &VulkanStorageBytes,
    bs: &[VulkanStorageBytes],
    out: &mut VulkanStorageBytes,
    k: usize,
    n: usize,
    reps: usize,
) -> f64 {
    let mut samples = Vec::new();
    for round in 0..5 {
        be.reset_gpu_time();
        for _ in 0..reps {
            for b in bs {
                run(be, a, b, out, k, n);
            }
        }
        be.download_bytes(out).unwrap();
        let ns: u64 = be
            .gpu_time_snapshot()
            .iter()
            .filter(|r| r.0 == BATCH_TOTAL)
            .map(|r| r.2)
            .sum();
        if round >= 1 {
            samples.push(ns as f64 / 1e3 / (reps * bs.len()) as f64);
        }
    }
    samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    let be = Arc::new(
        VulkanBackend::with_selection(DeviceSelection::PreferDiscrete)
            .expect("a Vulkan device (PreferDiscrete)"),
    );
    assert!(be.set_gpu_timing(true), "timestamp queries unsupported");
    // (K, N, what it is in Qwen3-0.6B)
    let shapes = [
        (1024usize, 1024usize, "k_proj / v_proj"),
        (1024, 2048, "q_proj"),
        (2048, 1024, "o_proj"),
        (1024, 3072, "gate / up"),
        (3072, 1024, "down"),
        (1024, 151_936, "lm head"),
    ];
    println!(
        "{:>6} {:>8} {:<16} {:>10} {:>10} {:>9} {:>10} {:>10} {:>8}",
        "K",
        "N",
        "role",
        "matvec us",
        "cols us",
        "speedup",
        "matvec GB/s",
        "cols GB/s",
        "max rel err"
    );
    // Warm-up: bring the GPU out of its idle clocks before the first timed row
    // (otherwise the first shape reads slow for both kernels).
    {
        let a = be.upload_bytes_handle(&bytes_of(&fill(1024, 1))).unwrap();
        let b = be
            .upload_bytes_handle(&bytes_of(&fill(1024 * 3072, 2)))
            .unwrap();
        let mut out = be.alloc_bytes_handle(3072 * 4).unwrap();
        for _ in 0..2000 {
            run(&be, &a, &b, &mut out, 1024, 3072);
        }
        be.download_bytes(&out).unwrap();
    }
    let mut bad = 0;
    for (k, n, role) in shapes {
        let bytes_b = k * n * 4;
        let copies = (400usize << 20).div_ceil(bytes_b).clamp(2, 24);
        let a_host = fill(k, 1);
        let b_host = fill(k * n, 2);
        let a = be.upload_bytes_handle(&bytes_of(&a_host)).unwrap();
        let mut bs = Vec::new();
        for _ in 0..copies {
            bs.push(be.upload_bytes_handle(&bytes_of(&b_host)).unwrap());
        }
        let mut out = be.alloc_bytes_handle(n * 4).unwrap();
        let want = reference(&a_host, &b_host, k, n);
        let scale = want.iter().fold(0f64, |m, x| m.max(x.abs())).max(1e-30);

        let mut errs = [0f64; 2];
        for (i, cols) in [false, true].into_iter().enumerate() {
            be.set_matvec_cols(cols);
            run(&be, &a, &bs[0], &mut out, k, n);
            let got = floats_of(&be.download_bytes(&out).unwrap());
            errs[i] = got
                .iter()
                .zip(&want)
                .fold(0f64, |m, (g, w)| m.max((*g as f64 - w).abs()))
                / scale;
        }
        be.set_matvec_cols(false);
        let old = gpu_us(&be, &a, &bs, &mut out, k, n, 3);
        be.set_matvec_cols(true);
        let new = gpu_us(&be, &a, &bs, &mut out, k, n, 3);
        let gbs = |us: f64| bytes_b as f64 / (us * 1e-6) / 1e9;
        println!(
            "{k:>6} {n:>8} {role:<16} {old:>10.1} {new:>10.1} {:>8.2}x {:>10.1} {:>10.1} {:>8.1e}/{:.1e}",
            old / new,
            gbs(old),
            gbs(new),
            errs[0],
            errs[1],
        );
        if errs[0] > 1e-4 || errs[1] > 1e-4 {
            bad += 1;
        }
    }
    if bad != 0 {
        eprintln!("{bad} shape(s) off the f64 reference by more than 1e-4 (relative to max |C|)");
        std::process::exit(1);
    }
}
