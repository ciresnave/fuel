// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live-device probe for split-KV decode attention (ROADMAP item 17, row H12).
//!
//! The single-pass `flash_attn_f32` kernel launches one workgroup per
//! (b, h_q, q_i), so a decode step (sq = 1) with few heads leaves most of the
//! GPU idle and walks the whole key range serially. `flash_attn_f32_split_bytes`
//! cuts the key range into chunks that run as separate workgroups and merges
//! them in fixed order.
//!
//! * `check [f32|bf16|f16]` compares both kernels against an f64 CPU reference
//!   (f32: max abs error; bf16/f16: error relative to the output's own half
//!   precision, with the reference computed from the dtype-rounded inputs),
//!   including causal and ALiBi cases, and exits non-zero on a miss. No dtype
//!   argument checks all three.
//! * `latency [f32|bf16|f16]` (default f32) prints per-call wall time (record + submit + wait + readback)
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
//!     -p fuel-vulkan-backend --example flash_attn_split_probe -- check bf16
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release \
//!     -p fuel-vulkan-backend --example flash_attn_split_probe -- latency
//! ```

use fuel_vulkan_backend::{DeviceSelection, VulkanBackend, VulkanStorageBytes};
use half::{bf16, f16};
use std::time::Instant;

/// Element type of Q/K/V/alibi/out.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Dtype {
    F32,
    Bf16,
    F16,
}

impl Dtype {
    fn parse(a: Option<&str>) -> Option<Self> {
        match a {
            Some("f32") => Some(Self::F32),
            Some("bf16") => Some(Self::Bf16),
            Some("f16") => Some(Self::F16),
            _ => None,
        }
    }

    fn elem(self) -> usize {
        if self == Self::F32 { 4 } else { 2 }
    }

    /// Round-trip through the dtype, so a reference sees what the GPU sees.
    fn round(self, x: f32) -> f32 {
        match self {
            Self::F32 => x,
            Self::Bf16 => bf16::from_f32(x).to_f32(),
            Self::F16 => f16::from_f32(x).to_f32(),
        }
    }

    /// Half-ulp relative bound of a correctly rounded output (with 1% slack).
    fn rel_tol(self) -> f32 {
        match self {
            Self::F32 => 0.0,
            Self::Bf16 => 1.01 * 2f32.powi(-8),
            Self::F16 => 1.01 * 2f32.powi(-11),
        }
    }

    fn bytes_of(self, v: &[f32]) -> Vec<u8> {
        match self {
            Self::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::Bf16 => v
                .iter()
                .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
                .collect(),
            Self::F16 => v
                .iter()
                .flat_map(|x| f16::from_f32(*x).to_le_bytes())
                .collect(),
        }
    }

    fn floats_of(self, b: &[u8]) -> Vec<f32> {
        match self {
            Self::F32 => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            Self::Bf16 => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::from_le_bytes(*c).to_f32())
                .collect(),
            Self::F16 => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_le_bytes(*c).to_f32())
                .collect(),
        }
    }
}

fn backend() -> VulkanBackend {
    VulkanBackend::with_selection(DeviceSelection::PreferDiscrete)
        .expect("a Vulkan device (PreferDiscrete)")
}

/// The arguments are a fixed subcommand name and a dtype name, each matched
/// against literals; neither is used as a path, command or query.
fn main() {
    let arg2 = std::env::args().nth(2);
    match std::env::args().nth(1).as_deref() {
        Some("check") => match arg2.as_deref() {
            None => [Dtype::F32, Dtype::Bf16, Dtype::F16]
                .into_iter()
                .for_each(check),
            a => check(Dtype::parse(a).unwrap_or_else(usage)),
        },
        Some("latency") => {
            latency(Dtype::parse(arg2.as_deref().or(Some("f32"))).unwrap_or_else(usage))
        }
        _ => usage(),
    }
}

fn usage<T>() -> T {
    eprintln!("usage: flash_attn_split_probe <check|latency> [f32|bf16|f16]");
    std::process::exit(2);
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
    /// Values are rounded to `dt` up front, so the f64 reference computes from
    /// exactly the numbers the GPU reads.
    fn new(c: &Case, dt: Dtype) -> Self {
        let r = |v: Vec<f32>| v.into_iter().map(|x| dt.round(x)).collect::<Vec<f32>>();
        Self {
            q: r(fill(c.hq * c.sq * c.d, 1)),
            k: r(fill(c.hkv * c.sk * c.d, 2)),
            v: r(fill(c.hkv * c.sk * c.d, 3)),
            alibi: r(fill(c.hq, 4).iter().map(|x| x * 0.02).collect()),
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
    dt: Dtype,
    q: VulkanStorageBytes,
    k: VulkanStorageBytes,
    v: VulkanStorageBytes,
    alibi: VulkanStorageBytes,
    out: VulkanStorageBytes,
}

impl Gpu {
    fn upload(be: &VulkanBackend, c: &Case, inp: &Inputs, dt: Dtype) -> Self {
        let up = |x: &[f32]| be.upload_bytes(&dt.bytes_of(x)).unwrap();
        Self {
            dt,
            q: up(&inp.q),
            k: up(&inp.k),
            v: up(&inp.v),
            alibi: up(&inp.alibi),
            out: be.alloc_bytes(c.hq * c.sq * c.d * dt.elem()).unwrap(),
        }
    }

    /// Run the unsplit kernel (`splits == 0`) or the split kernel.
    fn run(&mut self, be: &VulkanBackend, c: &Case, splits: usize) {
        let alibi = c.alibi.then_some(&self.alibi);
        let (b, scale) = (1, c.scale());
        let (q, k, v, out) = (&self.q, &self.k, &self.v, &mut self.out);
        let (hq, hkv, sq, sk, d, cz) = (c.hq, c.hkv, c.sq, c.sk, c.d, c.causal);
        match (self.dt, splits) {
            (Dtype::F32, 0) => {
                be.flash_attn_f32_bytes(q, k, v, alibi, out, b, hq, hkv, sq, sk, d, scale, cz)
            }
            (Dtype::Bf16, 0) => {
                be.flash_attn_bf16_bytes(q, k, v, alibi, out, b, hq, hkv, sq, sk, d, scale, cz)
            }
            (Dtype::F16, 0) => {
                be.flash_attn_f16_bytes(q, k, v, alibi, out, b, hq, hkv, sq, sk, d, scale, cz)
            }
            (Dtype::F32, n) => be.flash_attn_f32_split_bytes(
                q, k, v, alibi, out, b, hq, hkv, sq, sk, d, scale, cz, n,
            ),
            (Dtype::Bf16, n) => be.flash_attn_bf16_split_bytes(
                q, k, v, alibi, out, b, hq, hkv, sq, sk, d, scale, cz, n,
            ),
            (Dtype::F16, n) => be.flash_attn_f16_split_bytes(
                q, k, v, alibi, out, b, hq, hkv, sq, sk, d, scale, cz, n,
            ),
        }
        .unwrap();
    }

    fn read(&self, be: &VulkanBackend) -> Vec<f32> {
        self.dt.floats_of(&be.download_bytes(&self.out).unwrap())
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

/// Worst `|got - want| - rel * |want|` over the output (<= the absolute slack
/// means every element is within the relative bound).
fn excess(got: &[f32], want: &[f32], rel: f32) -> f32 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs() - rel * b.abs())
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Run both kernels on `c` and assert each is within the dtype's bound of the
/// f64 reference: 1e-4 absolute for f32; for bf16/f16 a half-ulp relative
/// bound (+1% slack) plus 2e-6 absolute for the f32 accumulation noise.
fn check_case(be: &VulkanBackend, c: &Case, dt: Dtype) {
    let inp = Inputs::new(c, dt);
    let want = reference(&inp, c);
    let mut gpu = Gpu::upload(be, c, &inp, dt);
    gpu.run(be, c, 0);
    let unsplit = gpu.read(be);
    gpu.run(be, c, c.splits);
    let split = gpu.read(be);
    let (slack, rel) = if dt == Dtype::F32 {
        (1e-4, 0.0)
    } else {
        (2e-6, dt.rel_tol())
    };
    let (x_un, x_sp) = (excess(&unsplit, &want, rel), excess(&split, &want, rel));
    println!(
        "{dt:?} hq={} hkv={} sq={} sk={} d={} causal={} alibi={} splits={}: max|err| unsplit={:.2e} split={:.2e}; split-vs-unsplit max|diff|={:.2e}; excess over bound unsplit={x_un:.2e} split={x_sp:.2e}",
        c.hq,
        c.hkv,
        c.sq,
        c.sk,
        c.d,
        c.causal,
        c.alibi,
        c.splits,
        max_abs_err(&unsplit, &want),
        max_abs_err(&split, &want),
        max_abs_err(&split, &unsplit),
    );
    assert!(
        x_un <= slack,
        "unsplit {dt:?} off its reference by {x_un:e} past the bound"
    );
    assert!(
        x_sp <= slack,
        "split {dt:?} off the reference by {x_sp:e} past the bound"
    );
}

fn check(dt: Dtype) {
    let be = backend();
    for c in check_cases() {
        check_case(&be, &c, dt);
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

fn latency(dt: Dtype) {
    let be = backend();
    println!("dtype {dt:?}");
    print!("{:>4} {:>4} {:>6}", "hq", "hkv", "sk");
    for s in VARIANTS {
        print!(" {:>9}", variant_name(s));
    }
    println!("   (median us per call incl. readback)");
    for &(hq, hkv) in &[(8usize, 8usize), (16, 8), (32, 8), (64, 8)] {
        for &sk in &[1024usize, 2048, 4096] {
            let c = Case {
                hq,
                hkv,
                sk,
                ..CASE_DEFAULT
            };
            let mut gpu = Gpu::upload(&be, &c, &Inputs::new(&c, dt), dt);
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
