// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live-device probe for split-KV decode attention (ROADMAP item 17, row H12).
//!
//! The single-pass `flash_attn_f32` kernel launches one workgroup per
//! (b, h_q, q_i), so a decode step (sq = 1) with few heads leaves most of the
//! GPU idle and walks the whole key range serially. `flash_attn_f32_split_bytes`
//! cuts the key range into chunks that run as separate workgroups and merges
//! them in fixed order.
//!
//! * `check` compares both kernels against an f64 CPU reference (max abs
//!   error), including causal and ALiBi cases, and exits non-zero on a miss.
//! * `latency` prints per-call wall time (record + submit + wait + readback)
//!   for the unsplit kernel and several split counts. Variants are interleaved
//!   round-robin so a slow stretch of the (bimodal) GPU hits all of them alike.
//!   It asserts nothing about speed.
//!
//! An example rather than an `#[ignore]` test: it needs a GPU, so it runs by
//! hand under the machine-wide mutex, and the ignore ceiling stays where it is.
//!
//! ```sh
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release \
//!     -p fuel-vulkan-backend --example flash_attn_split_probe -- check
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release \
//!     -p fuel-vulkan-backend --example flash_attn_split_probe -- latency
//! ```

use fuel_vulkan_backend::{DeviceSelection, VulkanBackend, VulkanStorageBytes};
use std::time::Instant;

fn backend() -> VulkanBackend {
    VulkanBackend::with_selection(DeviceSelection::PreferDiscrete)
        .expect("a Vulkan device (PreferDiscrete)")
}

/// The only argument is a fixed subcommand name matched against two literals;
/// it is never used as a path, command or query.
fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("check") => check(),
        Some("latency") => latency(),
        _ => {
            eprintln!("usage: flash_attn_split_probe <check|latency>");
            std::process::exit(2);
        }
    }
}

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

/// One attention problem. `splits` applies to the split kernel only.
#[derive(Clone, Copy)]
struct Case {
    hq: usize,
    hkv: usize,
    sq: usize,
    sk: usize,
    d: usize,
    causal: bool,
    alibi: bool,
    splits: usize,
}

impl Case {
    fn scale(&self) -> f32 {
        1.0 / (self.d as f32).sqrt()
    }
}

/// Host-side inputs for a [`Case`].
struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    alibi: Vec<f32>,
}

impl Inputs {
    fn new(c: &Case) -> Self {
        Self {
            q: fill(c.hq * c.sq * c.d, 1),
            k: fill(c.hkv * c.sk * c.d, 2),
            v: fill(c.hkv * c.sk * c.d, 3),
            alibi: fill(c.hq, 4).iter().map(|x| x * 0.02).collect(),
        }
    }
}

/// Scaled, masked scores of one query row against every key (f64).
fn score_row(inp: &Inputs, c: &Case, h: usize, qi: usize) -> Vec<f64> {
    let kvh = h / (c.hq / c.hkv);
    let qrow = &inp.q[(h * c.sq + qi) * c.d..][..c.d];
    (0..c.sk)
        .map(|kj| {
            if c.causal && kj > qi {
                return f64::NEG_INFINITY;
            }
            let krow = &inp.k[(kvh * c.sk + kj) * c.d..][..c.d];
            let dot: f64 = (0..c.d).map(|x| qrow[x] as f64 * krow[x] as f64).sum();
            let bias = if c.alibi {
                inp.alibi[h] as f64 * (kj as f64 - qi as f64)
            } else {
                0.0
            };
            dot * c.scale() as f64 + bias
        })
        .collect()
}

/// f64 reference: softmax(q k^T * scale [+ alibi * (kj - qi)] [causal]) v.
fn reference(inp: &Inputs, c: &Case) -> Vec<f32> {
    let mut out = vec![0f32; c.hq * c.sq * c.d];
    for h in 0..c.hq {
        let kvh = h / (c.hq / c.hkv);
        for qi in 0..c.sq {
            let sc = score_row(inp, c, h, qi);
            let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if m == f64::NEG_INFINITY {
                continue;
            }
            let w: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
            let l: f64 = w.iter().sum();
            for x in 0..c.d {
                let acc: f64 = (0..c.sk)
                    .map(|kj| w[kj] * inp.v[(kvh * c.sk + kj) * c.d + x] as f64)
                    .sum();
                out[(h * c.sq + qi) * c.d + x] = (acc / l) as f32;
            }
        }
    }
    out
}

/// Device buffers for a [`Case`].
struct Gpu {
    q: VulkanStorageBytes,
    k: VulkanStorageBytes,
    v: VulkanStorageBytes,
    alibi: VulkanStorageBytes,
    out: VulkanStorageBytes,
}

impl Gpu {
    fn upload(be: &VulkanBackend, c: &Case, inp: &Inputs) -> Self {
        let up = |x: &[f32]| be.upload_bytes(&bytes_of(x)).unwrap();
        Self {
            q: up(&inp.q),
            k: up(&inp.k),
            v: up(&inp.v),
            alibi: up(&inp.alibi),
            out: be.alloc_bytes(c.hq * c.sq * c.d * 4).unwrap(),
        }
    }

    /// Run the unsplit kernel (`splits == 0`) or the split kernel.
    fn run(&mut self, be: &VulkanBackend, c: &Case, splits: usize) {
        let alibi = c.alibi.then_some(&self.alibi);
        let (b, scale) = (1, c.scale());
        if splits == 0 {
            be.flash_attn_f32_bytes(
                &self.q,
                &self.k,
                &self.v,
                alibi,
                &mut self.out,
                b,
                c.hq,
                c.hkv,
                c.sq,
                c.sk,
                c.d,
                scale,
                c.causal,
            )
        } else {
            be.flash_attn_f32_split_bytes(
                &self.q,
                &self.k,
                &self.v,
                alibi,
                &mut self.out,
                b,
                c.hq,
                c.hkv,
                c.sq,
                c.sk,
                c.d,
                scale,
                c.causal,
                splits,
            )
        }
        .unwrap();
    }

    fn read(&self, be: &VulkanBackend) -> Vec<f32> {
        floats_of(&be.download_bytes(&self.out).unwrap())
    }
}

fn max_abs_err(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

const CASE_DEFAULT: Case = Case {
    hq: 8,
    hkv: 8,
    sq: 1,
    sk: 4096,
    d: 128,
    causal: false,
    alibi: false,
    splits: 4,
};

/// Decode and prefill shapes: GQA 1:1, 4:1 and 2:1, causal, ALiBi, splits 1-16.
fn check_cases() -> Vec<Case> {
    let prefill = Case {
        hq: 4,
        hkv: 2,
        sq: 16,
        sk: 2048,
        d: 64,
        causal: true,
        ..CASE_DEFAULT
    };
    vec![
        CASE_DEFAULT,
        Case {
            hq: 32,
            hkv: 8,
            sk: 3000,
            splits: 8,
            ..CASE_DEFAULT
        },
        Case {
            hkv: 4,
            sk: 1000,
            d: 64,
            alibi: true,
            splits: 3,
            ..CASE_DEFAULT
        },
        Case {
            splits: 2,
            ..prefill
        },
        Case {
            alibi: true,
            splits: 16,
            ..prefill
        },
        Case {
            sk: 700,
            splits: 1,
            ..CASE_DEFAULT
        },
    ]
}

/// Run both kernels on `c` and assert each is within 1e-4 of the f64 reference.
fn check_case(be: &VulkanBackend, c: &Case) {
    let inp = Inputs::new(c);
    let want = reference(&inp, c);
    let mut gpu = Gpu::upload(be, c, &inp);
    gpu.run(be, c, 0);
    let e_unsplit = max_abs_err(&gpu.read(be), &want);
    gpu.run(be, c, c.splits);
    let e_split = max_abs_err(&gpu.read(be), &want);
    println!(
        "hq={} hkv={} sq={} sk={} d={} causal={} alibi={} splits={}: max|err| unsplit={e_unsplit:.2e} split={e_split:.2e}",
        c.hq, c.hkv, c.sq, c.sk, c.d, c.causal, c.alibi, c.splits
    );
    assert!(
        e_unsplit < 1e-4,
        "unsplit kernel off its reference: {e_unsplit}"
    );
    assert!(e_split < 1e-4, "split kernel off the reference: {e_split}");
}

fn check() {
    let be = backend();
    for c in check_cases() {
        check_case(&be, &c);
    }
}

/// 0 = the unsplit kernel.
const VARIANTS: [usize; 6] = [0, 2, 4, 8, 16, 32];

fn variant_name(s: usize) -> String {
    if s == 0 {
        "unsplit".to_string()
    } else {
        format!("split{s}")
    }
}

/// Median wall time in us of `run(variant)` per usable variant, interleaved.
fn median_times(be: &VulkanBackend, c: &Case, gpu: &mut Gpu, usable: &[usize]) -> Vec<f64> {
    let mut one = |s: usize| {
        let t0 = Instant::now();
        gpu.run(be, c, s);
        gpu.read(be);
        t0.elapsed().as_secs_f64() * 1e6
    };
    for &s in usable {
        for _ in 0..5 {
            one(s);
        }
    }
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); usable.len()];
    for _ in 0..60 {
        for (i, &s) in usable.iter().enumerate() {
            times[i].push(one(s));
        }
    }
    times
        .into_iter()
        .map(|mut t| {
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t[t.len() / 2]
        })
        .collect()
}

fn latency() {
    let be = backend();
    print!("{:>4} {:>4} {:>6}", "hq", "hkv", "sk");
    for s in VARIANTS {
        print!(" {:>9}", variant_name(s));
    }
    println!("   (median us per call incl. readback)");
    for &(hq, hkv) in &[(8usize, 8usize), (32, 8), (64, 8)] {
        for &sk in &[1024usize, 2048, 4096] {
            let c = Case {
                hq,
                hkv,
                sk,
                ..CASE_DEFAULT
            };
            let mut gpu = Gpu::upload(&be, &c, &Inputs::new(&c));
            let usable: Vec<usize> = VARIANTS
                .into_iter()
                .filter(|&s| s == 0 || sk.div_ceil(s) <= 1024)
                .collect();
            let medians = median_times(&be, &c, &mut gpu, &usable);
            print!("{hq:>4} {hkv:>4} {sk:>6}");
            for s in VARIANTS {
                match usable.iter().position(|&u| u == s) {
                    Some(i) => print!(" {:>9.1}", medians[i]),
                    None => print!(" {:>9}", "-"),
                }
            }
            println!();
        }
    }
}
