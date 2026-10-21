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
use fuel_vulkan_backend::VulkanBackend;

fn cpu_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    hq: usize,
    hkv: usize,
    sk: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; hq * d];
    for h in 0..hq {
        let kv = h / (hq / hkv);
        let s: Vec<f64> = (0..sk)
            .map(|j| {
                (0..d)
                    .map(|x| q[h * d + x] as f64 * k[(kv * sk + j) * d + x] as f64)
                    .sum::<f64>()
                    * scale as f64
            })
            .collect();
        let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
        let l: f64 = e.iter().sum();
        for x in 0..d {
            let acc: f64 = (0..sk)
                .map(|j| e[j] * v[(kv * sk + j) * d + x] as f64)
                .sum();
            out[h * d + x] = (acc / l) as f32;
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

fn main() {
    let backend = Arc::new(VulkanBackend::new().expect("vulkan backend"));
    let (b, hq, hkv, sq, sk, d) = (1usize, 8usize, 2usize, 1usize, 3000usize, 64usize);
    let scale = 1.0 / (d as f32).sqrt();
    let synth = |n: usize, a: f32, c: f32| -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + a) * c).sin() * 0.5).collect()
    };
    let (q, k, v) = (
        synth(hq * d, 0.0, 0.013),
        synth(hkv * sk * d, 1.0, 0.0071),
        synth(hkv * sk * d, 2.0, 0.0053),
    );
    let want = cpu_ref(&q, &k, &v, hq, hkv, sk, d, scale);

    let up = |h: &[f32]| {
        backend
            .upload_bytes_handle(bytemuck::cast_slice(h))
            .expect("upload")
    };
    let (qb, kb, vb) = (up(&q), up(&k), up(&v));

    // 1. Through the registered dispatch kernel.
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let kernel =
        table.lookup_alternatives(OpKind::FlashAttn, &[DType::F32; 4], BackendId::Vulkan)[0].kernel;
    let st = |bytes| {
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Vulkan(bytes),
            DType::F32,
        )))
    };
    let out_arc = st(backend.alloc_bytes_handle(hq * d * 4).expect("alloc"));
    let layouts = [
        Layout::contiguous(Shape::from_dims(&[b, hq, sq, d])),
        Layout::contiguous(Shape::from_dims(&[b, hkv, sk, d])),
        Layout::contiguous(Shape::from_dims(&[b, hkv, sk, d])),
        Layout::contiguous(Shape::from_dims(&[b, hq, sq, d])),
    ];
    kernel(
        &[st(up(&q)), st(up(&k)), st(up(&v))],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::FlashAttn {
            b,
            hq,
            hkv,
            sq,
            sk,
            d,
            k_len: sk,
            softmax_scale: scale,
            causal: false,
            window_size_left: None,
            window_size_right: None,
            softcap: None,
        },
    )
    .expect("dispatch");
    let got: Vec<f32> = match &out_arc.read().unwrap().inner {
        BackendStorage::Vulkan(o) => {
            bytemuck::cast_slice::<u8, f32>(&backend.download_bytes(o).unwrap()).to_vec()
        }
        _ => unreachable!(),
    };

    // 2. Direct split and 3. direct unsplit.
    let mut o_split = backend.alloc_bytes_handle(hq * d * 4).unwrap();
    backend
        .flash_attn_f32_split_bytes(
            &qb,
            &kb,
            &vb,
            None,
            &mut o_split,
            b,
            hq,
            hkv,
            sq,
            sk,
            d,
            scale,
            false,
            8,
        )
        .unwrap();
    let split: Vec<f32> =
        bytemuck::cast_slice::<u8, f32>(&backend.download_bytes(&o_split).unwrap()).to_vec();
    let mut o_un = backend.alloc_bytes_handle(hq * d * 4).unwrap();
    backend
        .flash_attn_f32_bytes(
            &qb, &kb, &vb, None, &mut o_un, b, hq, hkv, sq, sk, d, scale, false,
        )
        .unwrap();
    let unsplit: Vec<f32> =
        bytemuck::cast_slice::<u8, f32>(&backend.download_bytes(&o_un).unwrap()).to_vec();

    let (e_cpu, e_un) = (max_abs(&got, &want), max_abs(&got, &unsplit));
    let identical_to_split = got
        .iter()
        .zip(&split)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let identical_to_unsplit = got
        .iter()
        .zip(&unsplit)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    println!("dispatch vs cpu f64: max abs {e_cpu:e}");
    println!("dispatch bit-identical to direct split(8): {identical_to_split}");
    println!(
        "dispatch bit-identical to unsplit: {identical_to_unsplit}; max abs vs unsplit {e_un:e}"
    );
    assert!(e_cpu < 1e-4 && e_un < 1e-4, "numeric bound");
    assert!(identical_to_split, "dispatch did not take the split route");
}
