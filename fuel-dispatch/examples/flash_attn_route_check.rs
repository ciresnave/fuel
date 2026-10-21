// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live-GPU check that the Vulkan flash-attention dispatch takes the split-KV
//! route for a decode shape, for f32, bf16 and f16 (ROADMAP 17, H12 wiring), as
//! an example rather than an `#[ignore]` test (the ignore total sits at its
//! ceiling).
//!
//! Run: `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release
//! -p fuel-dispatch --features vulkan --example flash_attn_route_check`
//!
//! Per dtype, on `b=1 hq=8 hkv=2 sq=1 sk=3000 d=64`, inputs rounded to the
//! dtype first so the reference sees what the GPU reads:
//! 1. dispatch vs an f64 CPU reference (f32: 1e-4 absolute; half: a half-ulp
//!    relative bound plus 2e-6);
//! 2. dispatch vs the split wrapper called directly with 8 splits: must be
//!    bit-identical (the split path is deterministic), which proves the route;
//! 3. dispatch vs the unsplit wrapper: printed only.
use std::sync::{Arc, RwLock};

use fuel_dispatch::{
    kernel::{KernelBindingTable, OpParams},
    vulkan_dispatch::register_vulkan_kernels,
};
use fuel_ir::{DType, Layout, Shape, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};
use fuel_vulkan_backend::{VulkanBackend, VulkanStorageBytes};
use half::{bf16, f16};

#[derive(Clone, Copy)]
struct Dims {
    b: usize,
    hq: usize,
    hkv: usize,
    sq: usize,
    sk: usize,
    d: usize,
    scale: f32,
}

struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Element type under test.
#[derive(Clone, Copy, Debug)]
enum Elem {
    F32,
    Bf16,
    F16,
}

impl Elem {
    fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Bf16 => DType::BF16,
            Self::F16 => DType::F16,
        }
    }

    fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            _ => 2,
        }
    }

    fn round(self, x: f32) -> f32 {
        match self {
            Self::F32 => x,
            Self::Bf16 => bf16::from_f32(x).to_f32(),
            Self::F16 => f16::from_f32(x).to_f32(),
        }
    }

    /// (relative, absolute) bound against the f64 reference.
    fn tol(self) -> (f32, f32) {
        match self {
            Self::F32 => (0.0, 1e-4),
            Self::Bf16 => (1.01 * 2f32.powi(-8), 2e-6),
            Self::F16 => (1.01 * 2f32.powi(-11), 2e-6),
        }
    }

    fn encode(self, v: &[f32]) -> Vec<u8> {
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

    fn decode(self, b: &[u8]) -> Vec<f32> {
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

fn synth(n: usize, a: f32, c: f32, e: Elem) -> Vec<f32> {
    (0..n)
        .map(|i| e.round(((i as f32 + a) * c).sin() * 0.5))
        .collect()
}

fn cpu_ref(x: &Inputs, s: Dims) -> Vec<f32> {
    let mut out = vec![0f32; s.hq * s.d];
    for h in 0..s.hq {
        let kv = h / (s.hq / s.hkv);
        let dot = |j: usize| -> f64 {
            let qk: f64 = (0..s.d)
                .map(|t| x.q[h * s.d + t] as f64 * x.k[(kv * s.sk + j) * s.d + t] as f64)
                .sum();
            qk * s.scale as f64
        };
        let scores: Vec<f64> = (0..s.sk).map(dot).collect();
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = scores.iter().map(|v| (v - m).exp()).collect();
        let l: f64 = e.iter().sum();
        for t in 0..s.d {
            let acc: f64 = (0..s.sk)
                .map(|j| e[j] * x.v[(kv * s.sk + j) * s.d + t] as f64)
                .sum();
            out[h * s.d + t] = (acc / l) as f32;
        }
    }
    out
}

/// Worst `|got - want| - rel * |want|`.
fn excess(got: &[f32], want: &[f32], rel: f32) -> f32 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs() - rel * b.abs())
        .fold(f32::NEG_INFINITY, f32::max)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn download(backend: &VulkanBackend, e: Elem, s: &VulkanStorageBytes) -> Vec<f32> {
    e.decode(&backend.download_bytes(s).expect("d2h"))
}

/// The registered dispatch kernel, the production route.
fn via_dispatch(backend: &Arc<VulkanBackend>, e: Elem, x: &Inputs, s: Dims) -> Vec<f32> {
    let up = |h: &[f32]| backend.upload_bytes_handle(&e.encode(h)).expect("upload");
    let st = |bytes| {
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Vulkan(bytes),
            e.dtype(),
        )))
    };
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let kernel =
        table.lookup_alternatives(OpKind::FlashAttn, &[e.dtype(); 4], BackendId::Vulkan)[0].kernel;
    let out = st(backend
        .alloc_bytes_handle(s.hq * s.d * e.size())
        .expect("alloc"));
    let (q_dims, kv_dims) = ([s.b, s.hq, s.sq, s.d], [s.b, s.hkv, s.sk, s.d]);
    let layouts = [
        Layout::contiguous(Shape::from_dims(&q_dims)),
        Layout::contiguous(Shape::from_dims(&kv_dims)),
        Layout::contiguous(Shape::from_dims(&kv_dims)),
        Layout::contiguous(Shape::from_dims(&q_dims)),
    ];
    let params = OpParams::FlashAttn {
        b: s.b,
        hq: s.hq,
        hkv: s.hkv,
        sq: s.sq,
        sk: s.sk,
        d: s.d,
        k_len: s.sk,
        softmax_scale: s.scale,
        causal: false,
        window_size_left: None,
        window_size_right: None,
        softcap: None,
    };
    kernel(
        &[st(up(&x.q)), st(up(&x.k)), st(up(&x.v))],
        &mut [Arc::clone(&out)],
        &layouts,
        &params,
    )
    .expect("dispatch");
    match &out.read().unwrap().inner {
        BackendStorage::Vulkan(o) => download(backend, e, o),
        _ => unreachable!("output was allocated on Vulkan"),
    }
}

/// A backend wrapper called directly: `Some(n)` = split-KV with `n` splits,
/// `None` = the unsplit kernel.
fn direct(
    backend: &Arc<VulkanBackend>,
    e: Elem,
    x: &Inputs,
    s: Dims,
    splits: Option<usize>,
) -> Vec<f32> {
    let up = |h: &[f32]| backend.upload_bytes_handle(&e.encode(h)).expect("upload");
    let (q, k, v) = (up(&x.q), up(&x.k), up(&x.v));
    let mut out = backend
        .alloc_bytes_handle(s.hq * s.d * e.size())
        .expect("alloc");
    let (b, hq, hkv, sq, sk, d, sc) = (s.b, s.hq, s.hkv, s.sq, s.sk, s.d, s.scale);
    let r = match (e, splits) {
        (Elem::F32, Some(n)) => backend.flash_attn_f32_split_bytes(
            &q, &k, &v, None, &mut out, b, hq, hkv, sq, sk, d, sc, false, n,
        ),
        (Elem::Bf16, Some(n)) => backend.flash_attn_bf16_split_bytes(
            &q, &k, &v, None, &mut out, b, hq, hkv, sq, sk, d, sc, false, n,
        ),
        (Elem::F16, Some(n)) => backend.flash_attn_f16_split_bytes(
            &q, &k, &v, None, &mut out, b, hq, hkv, sq, sk, d, sc, false, n,
        ),
        (Elem::F32, None) => backend
            .flash_attn_f32_bytes(&q, &k, &v, None, &mut out, b, hq, hkv, sq, sk, d, sc, false),
        (Elem::Bf16, None) => backend
            .flash_attn_bf16_bytes(&q, &k, &v, None, &mut out, b, hq, hkv, sq, sk, d, sc, false),
        (Elem::F16, None) => backend
            .flash_attn_f16_bytes(&q, &k, &v, None, &mut out, b, hq, hkv, sq, sk, d, sc, false),
    };
    r.expect("direct wrapper");
    download(backend, e, &out)
}

fn check(backend: &Arc<VulkanBackend>, e: Elem) {
    let (hq, hkv, sk, d) = (8usize, 2usize, 3000usize, 64usize);
    let s = Dims {
        b: 1,
        hq,
        hkv,
        sq: 1,
        sk,
        d,
        scale: 1.0 / (d as f32).sqrt(),
    };
    let x = Inputs {
        q: synth(hq * d, 0.0, 0.013, e),
        k: synth(hkv * sk * d, 1.0, 0.0071, e),
        v: synth(hkv * sk * d, 2.0, 0.0053, e),
    };
    let got = via_dispatch(backend, e, &x, s);
    let want = cpu_ref(&x, s);
    let split = direct(backend, e, &x, s, Some(8));
    let unsplit = direct(backend, e, &x, s, None);

    let (rel, slack) = e.tol();
    let past = excess(&got, &want, rel);
    let same_split = got
        .iter()
        .zip(&split)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let same_unsplit = got
        .iter()
        .zip(&unsplit)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    println!(
        "{e:?}: dispatch vs cpu f64 max abs {:e}, excess over bound {past:e}",
        max_abs(&got, &want)
    );
    println!("{e:?}: dispatch bit-identical to direct split(8): {same_split}");
    println!(
        "{e:?}: dispatch bit-identical to unsplit: {same_unsplit}; max abs vs unsplit {:e}",
        max_abs(&got, &unsplit)
    );
    assert!(past <= slack, "{e:?}: numeric bound");
    assert!(same_split, "{e:?}: dispatch did not take the split route");
}

fn main() {
    let backend = Arc::new(VulkanBackend::new().expect("vulkan backend"));
    for e in [Elem::F32, Elem::Bf16, Elem::F16] {
        check(&backend, e);
    }
}
