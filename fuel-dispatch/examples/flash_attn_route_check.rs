// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live-GPU check that the f32 Vulkan flash-attention dispatch takes the
//! split-KV route for a decode shape (ROADMAP 17, H12 wiring), as an example
//! rather than an `#[ignore]` test (the ignore total sits at its ceiling).
//!
//! Run: `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release
//! -p fuel-dispatch --features vulkan --example flash_attn_route_check`
//!
//! Three comparisons on `b=1 hq=8 hkv=2 sq=1 sk=3000 d=64`:
//! 1. dispatch vs an f64 CPU reference (tolerance 1e-4);
//! 2. dispatch vs `flash_attn_f32_split_bytes(.., 8)` called directly: must be
//!    bit-identical (the split path is deterministic), which proves the route;
//! 3. dispatch vs `flash_attn_f32_bytes`: printed, expected to differ in the
//!    last bits or not at all (reordered adds), never by more than 1e-4.
use std::sync::{Arc, RwLock};

use fuel_dispatch::{
    kernel::{KernelBindingTable, OpParams},
    vulkan_dispatch::register_vulkan_kernels,
};
use fuel_ir::{DType, Layout, Shape, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};
use fuel_vulkan_backend::{VulkanBackend, VulkanStorageBytes};

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

fn synth(n: usize, a: f32, c: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + a) * c).sin() * 0.5).collect()
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

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn bit_equal(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn download(backend: &Arc<VulkanBackend>, s: &VulkanStorageBytes) -> Vec<f32> {
    bytemuck::cast_slice::<u8, f32>(&backend.download_bytes(s).expect("d2h")).to_vec()
}

/// The registered dispatch kernel, the production route.
fn via_dispatch(backend: &Arc<VulkanBackend>, x: &Inputs, s: Dims) -> Vec<f32> {
    let up = |h: &[f32]| {
        backend
            .upload_bytes_handle(bytemuck::cast_slice(h))
            .expect("upload")
    };
    let st = |bytes| {
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Vulkan(bytes),
            DType::F32,
        )))
    };
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let kernel =
        table.lookup_alternatives(OpKind::FlashAttn, &[DType::F32; 4], BackendId::Vulkan)[0].kernel;
    let out = st(backend.alloc_bytes_handle(s.hq * s.d * 4).expect("alloc"));
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
        BackendStorage::Vulkan(o) => download(backend, o),
        _ => unreachable!("output was allocated on Vulkan"),
    }
}

/// A backend wrapper called directly: `Some(n)` = split-KV with `n` splits,
/// `None` = the unsplit kernel.
fn direct(backend: &Arc<VulkanBackend>, x: &Inputs, s: Dims, splits: Option<usize>) -> Vec<f32> {
    let up = |h: &[f32]| {
        backend
            .upload_bytes_handle(bytemuck::cast_slice(h))
            .expect("upload")
    };
    let (q, k, v) = (up(&x.q), up(&x.k), up(&x.v));
    let mut out = backend.alloc_bytes_handle(s.hq * s.d * 4).expect("alloc");
    let r = match splits {
        Some(n) => backend.flash_attn_f32_split_bytes(
            &q, &k, &v, None, &mut out, s.b, s.hq, s.hkv, s.sq, s.sk, s.d, s.scale, false, n,
        ),
        None => backend.flash_attn_f32_bytes(
            &q, &k, &v, None, &mut out, s.b, s.hq, s.hkv, s.sq, s.sk, s.d, s.scale, false,
        ),
    };
    r.expect("direct wrapper");
    download(backend, &out)
}

fn main() {
    let backend = Arc::new(VulkanBackend::new().expect("vulkan backend"));
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
        q: synth(hq * d, 0.0, 0.013),
        k: synth(hkv * sk * d, 1.0, 0.0071),
        v: synth(hkv * sk * d, 2.0, 0.0053),
    };
    let got = via_dispatch(&backend, &x, s);
    let want = cpu_ref(&x, s);
    let split = direct(&backend, &x, s, Some(8));
    let unsplit = direct(&backend, &x, s, None);

    let (e_cpu, e_un) = (max_abs(&got, &want), max_abs(&got, &unsplit));
    let same_split = bit_equal(&got, &split);
    println!("dispatch vs cpu f64: max abs {e_cpu:e}");
    println!("dispatch bit-identical to direct split(8): {same_split}");
    println!(
        "dispatch bit-identical to unsplit: {}; max abs vs unsplit {e_un:e}",
        bit_equal(&got, &unsplit)
    );
    assert!(e_cpu < 1e-4 && e_un < 1e-4, "numeric bound");
    assert!(same_split, "dispatch did not take the split route");
}
