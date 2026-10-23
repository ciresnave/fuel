// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live-CUDA test for the `OpKind::AwqMatmul` binding (ROADMAP row 31a): the
//! baracuda W4A16 AWQ kernel reached through the contract-registered wrapper,
//! against an f64 dequantise-then-matmul reference.
//!
//! The operands are built from LOGICAL codes with the kernel layout written out
//! longhand below (OC-major; nibble `k % 8` of word `k / 8`; zeros nibble
//! `g % 8` of word `g / 8`; scales entry `g`), independently of
//! `fuel_quantized::awq_repack`, so a layout error in either would show here.
//!
//! Not `#[ignore]`d: the file is `cfg(feature = "cuda")`, so no CI leg builds it,
//! and `require_gpu_run_lock` plus `CudaDevice::new` fail loudly without the
//! wrapper or a device.
//!
//! Run: `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-dispatch
//! --features cuda --test baracuda_awq_live` (sm_80+ required).

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_dispatch::{
    baracuda_dispatch::register_baracuda_cuda_kernels,
    dispatch::register_cuda_kernels,
    kernel::{KernelBindingTable, OpParams},
};
use fuel_ir::{DType, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};
use half::f16;

const KEY: [DType; 5] = [DType::F16, DType::U32, DType::U32, DType::F16, DType::F16];

fn upload(dev: &CudaDevice, bytes: &[u8], dt: DType) -> Arc<RwLock<Storage>> {
    let cuda = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(cuda), dt)))
}

struct Problem {
    ic: usize,
    oc: usize,
    group: usize,
    q: Vec<u8>,  // [oc, ic] logical codes
    z: Vec<u8>,  // [oc, groups]
    s: Vec<f16>, // [oc, groups]
}

fn problem(ic: usize, oc: usize, group: usize) -> Problem {
    let groups = ic / group;
    Problem {
        ic,
        oc,
        group,
        q: (0..oc * ic).map(|i| ((i * 7 + i / 3) % 16) as u8).collect(),
        z: (0..oc * groups).map(|i| ((i * 5 + 3) % 16) as u8).collect(),
        s: (0..oc * groups)
            .map(|i| f16::from_f32(0.01 + 0.002 * (i % 9) as f32))
            .collect(),
    }
}

/// Kernel-layout buffers, written longhand from the layout table.
fn kernel_buffers(p: &Problem) -> (Vec<u32>, Vec<u32>, Vec<f16>, usize) {
    let groups = p.ic / p.group;
    let mult = 128 / p.group;
    let zeros_w = groups.div_ceil(8).div_ceil(mult) * mult;
    let mut w = vec![0u32; p.oc * p.ic / 8];
    let mut z = vec![0u32; p.oc * zeros_w];
    let mut s = vec![f16::ZERO; p.oc * zeros_w * 8];
    for o in 0..p.oc {
        for k in 0..p.ic {
            w[o * (p.ic / 8) + k / 8] |= u32::from(p.q[o * p.ic + k]) << (4 * (k % 8));
        }
        for g in 0..groups {
            z[o * zeros_w + g / 8] |= u32::from(p.z[o * groups + g]) << (4 * (g % 8));
            s[o * zeros_w * 8 + g] = p.s[o * groups + g];
        }
    }
    (w, z, s, zeros_w)
}

fn reference(p: &Problem, x: &[f16], rows: usize, zero_bias: f64) -> Vec<f64> {
    let groups = p.ic / p.group;
    let mut y = vec![0.0f64; rows * p.oc];
    for r in 0..rows {
        for o in 0..p.oc {
            let mut acc = 0.0f64;
            for k in 0..p.ic {
                let g = k / p.group;
                let w = (f64::from(p.q[o * p.ic + k]) - f64::from(p.z[o * groups + g]) + zero_bias)
                    * f64::from(p.s[o * groups + g].to_f32());
                acc += f64::from(x[r * p.ic + k].to_f32()) * w;
            }
            y[r * p.oc + o] = acc;
        }
    }
    y
}

fn run(rows: usize, ic: usize, oc: usize, group: usize, split_k: i32) {
    fuel_test_support::require_gpu_run_lock();
    let dev = CudaDevice::new(0).expect("cuda device");
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);
    let kernel = table
        .lookup(OpKind::AwqMatmul, &KEY, BackendId::Cuda)
        .expect("AwqMatmul F16 binding must be registered on CUDA");

    let p = problem(ic, oc, group);
    let (w, z, s, _zeros_w) = kernel_buffers(&p);
    let x: Vec<f16> = (0..rows * ic)
        .map(|i| f16::from_f32(((i * 5) % 17) as f32 * 0.1 - 0.8))
        .collect();
    let launch = || -> Vec<u8> {
        let out = {
            let buf = CudaStorageBytes::alloc(&dev, rows * oc * 2).expect("alloc");
            Arc::new(RwLock::new(Storage::new(
                BackendStorage::Cuda(buf),
                DType::F16,
            )))
        };
        kernel(
            &[
                upload(&dev, bytemuck::cast_slice(&x), DType::F16),
                upload(&dev, bytemuck::cast_slice(&w), DType::U32),
                upload(&dev, bytemuck::cast_slice(&z), DType::U32),
                upload(&dev, bytemuck::cast_slice(&s), DType::F16),
            ],
            &mut [out.clone()],
            &[],
            &OpParams::AwqMatmul {
                batch: 1,
                m: rows,
                n: oc,
                k: ic,
                group_size: group,
                split_k_iters: split_k,
            },
        )
        .expect("awq kernel call");
        match &out.read().unwrap().inner {
            BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("d2h"),
            _ => panic!("output not on CUDA"),
        }
    };
    let got_bytes = launch();
    // The FKC contract declares `same_hardware_bitwise` (author-declared, not
    // audited). This is the run-to-run half of that claim on this device: a
    // second launch on identical inputs must reproduce every output byte. It
    // does not cover other hardware or other drivers.
    assert_eq!(
        launch(),
        got_bytes,
        "rows={rows} ic={ic} oc={oc} g={group}: two launches on identical inputs differ"
    );
    let got: &[f16] = bytemuck::cast_slice(&got_bytes);
    assert_eq!(got.len(), rows * oc);
    let want = reference(&p, &x, rows, 0.0);
    let scale = want.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-3);
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (f64::from(g.to_f32()) - w).abs())
        .fold(0.0, f64::max);
    // F16 staging of split-K partials and an F16 output: a few F16 ulps of the
    // largest magnitude, not a bit-match.
    assert!(
        max_err <= 2e-2 * scale,
        "rows={rows} ic={ic} oc={oc} g={group}: max_err={max_err} scale={scale}"
    );
    // Sabotage sibling: a reference with every zero-point off by one (the
    // classic AWQ bug) must be FAR from the kernel's output, so the tolerance
    // above cannot be satisfied by a wrong layout.
    let wrong = reference(&p, &x, rows, 1.0);
    let wrong_err = got
        .iter()
        .zip(&wrong)
        .map(|(g, w)| (f64::from(g.to_f32()) - w).abs())
        .fold(0.0, f64::max);
    assert!(
        wrong_err > 10.0 * max_err.max(1e-6),
        "tolerance cannot distinguish a wrong zero-point: wrong_err={wrong_err} max_err={max_err}"
    );
}

#[test]
fn awq_matmul_f16_matches_the_dequantise_then_matmul_reference() {
    // group 128 and 64; M below, at and above the kernel's 128-row tile; a
    // ragged M that is not a multiple of anything.
    run(1, 256, 64, 128, 8);
    run(17, 512, 128, 64, 8);
    run(130, 256, 64, 128, 8);
}

/// Shapes the kernel cannot run are a typed `Err` from the wrapper, before any
/// launch: `n % 64`, `split_k_iters > 0`, `k % (32 * split_k_iters)`,
/// `group_size` in {64, 128}, `k % group_size`. The operands are those of a
/// VALID 256x64 problem, so a wrapper that skipped the check and launched would
/// read buffers that do not match the claimed shape; the control row proves the
/// same buffers run when the parameters are honest.
#[test]
fn awq_matmul_f16_refuses_unrunnable_shapes_before_launch() {
    fuel_test_support::require_gpu_run_lock();
    let dev = CudaDevice::new(0).expect("cuda device");
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);
    let kernel = table
        .lookup(OpKind::AwqMatmul, &KEY, BackendId::Cuda)
        .expect("AwqMatmul F16 binding must be registered on CUDA");
    let p = problem(256, 64, 128);
    let (w, z, s, _) = kernel_buffers(&p);
    let x: Vec<f16> = vec![f16::from_f32(0.5); 4 * 256];
    let try_call = |n: usize, k: usize, group: usize, split_k: i32| {
        let out = CudaStorageBytes::alloc(&dev, 4 * 128 * 2).expect("alloc");
        let out = Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(out),
            DType::F16,
        )));
        kernel(
            &[
                upload(&dev, bytemuck::cast_slice(&x), DType::F16),
                upload(&dev, bytemuck::cast_slice(&w), DType::U32),
                upload(&dev, bytemuck::cast_slice(&z), DType::U32),
                upload(&dev, bytemuck::cast_slice(&s), DType::F16),
            ],
            &mut [out],
            &[],
            &OpParams::AwqMatmul {
                batch: 1,
                m: 4,
                n,
                k,
                group_size: group,
                split_k_iters: split_k,
            },
        )
    };
    try_call(64, 256, 128, 8).expect("control: the honest parameters run");
    for (what, n, k, group, split_k) in [
        ("n=96 (not a multiple of 64)", 96, 256, 128, 8),
        ("split_k_iters=0", 64, 256, 128, 0),
        ("split_k_iters=-1", 64, 256, 128, -1),
        (
            "k=256 with split_k_iters=7 (256 % 224 != 0)",
            64,
            256,
            128,
            7,
        ),
        ("group_size=96", 64, 256, 96, 8),
        ("k=200 (not a multiple of group_size)", 64, 200, 128, 1),
    ] {
        assert!(
            try_call(n, k, group, split_k).is_err(),
            "{what}: expected Err, the wrapper accepted it"
        );
    }
}
