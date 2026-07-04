//! Live-CUDA tests for baracuda-kernels-sys-backed attention
//! primitives.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_ir::{dispatch::OpKind, probe::BackendId, DType};
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_dispatch::{baracuda_dispatch::register_baracuda_cuda_kernels, dispatch::register_cuda_kernels, kernel::{KernelBindingTable, OpParams}};
use fuel_memory::{BackendStorage, Storage};

fn dev_or_skip() -> Option<CudaDevice> {
    CudaDevice::new(0).ok()
}

fn upload_f32(dev: &CudaDevice, host: &[f32]) -> Storage {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    let cuda_bytes = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda_bytes), DType::F32)
}

fn download_f32(s: &Storage) -> Vec<f32> {
    let bytes = match &s.inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("d2h"),
        _ => panic!("not on CUDA"),
    };
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

fn pick_alt(
    table: &KernelBindingTable,
    op: OpKind,
    expected: fuel_dispatch::KernelRef,
) -> fuel_dispatch::KernelRef {
    let alternatives =
        table.lookup_alternatives(op, &[DType::F32, DType::F32], BackendId::Cuda);
    let expected_ptr = expected as usize;
    for alt in alternatives {
        if (alt.kernel as usize) == expected_ptr {
            return alt.kernel;
        }
    }
    panic!("expected baracuda KernelRef not found")
}

/// RoPE applied at sequence position 0 with default base 10000 is
/// the identity transform (angle θ = 0 → cos=1, sin=0, output = x).
/// Use this for a no-arithmetic-error smoke test on the live GPU.
#[test]
#[ignore]
fn baracuda_rope_f32_at_seq_position_zero_is_identity() {
    let Some(_dev) = dev_or_skip() else { return };
    let dev = CudaDevice::new(0).expect("cuda");
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // [outer_count=1, seq=1, head_dim=4] — RoPE rotates pairs
    // (x_0, x_1) and (x_2, x_3). At seq position 0, angle is 0 →
    // identity.
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let src = upload_f32(&dev, &input);
    let out_bytes = CudaStorageBytes::alloc(&dev, input.len() * 4).expect("alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));
    let kernel = pick_alt(
        &table,
        OpKind::Rope,
        fuel_dispatch::baracuda_dispatch::attention::rope_f32,
    );
    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::Rope {
            outer_count: 1,
            seq: 1,
            head_dim: 4,
        },
    )
    .expect("kernel call");

    let got = download_f32(&out_arc.read().unwrap());
    // Identity at pos 0: output == input within fp32 tolerance.
    for (g, e) in got.iter().zip(input.iter()) {
        assert!(
            (g - e).abs() < 1e-5,
            "got {got:?} expected {input:?}",
        );
    }
}

/// Two sequence positions, head_dim=4. Verify RoPE produces a
/// non-trivial rotation at pos 1 — angle θ_0 = 1 (i=0 →
/// base^0 = 1). The pair (x_0, x_1) at pos 1 rotates by 1 rad.
///
/// Reference: `(cos(1)·x_0 - sin(1)·x_1, sin(1)·x_0 + cos(1)·x_1)`.
#[test]
#[ignore]
fn baracuda_rope_f32_at_seq_position_one_rotates_pair() {
    let Some(_dev) = dev_or_skip() else { return };
    let dev = CudaDevice::new(0).expect("cuda");
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // [outer_count=1, seq=2, head_dim=4] flat = 8 elements:
    // [pos0_x0, pos0_x1, pos0_x2, pos0_x3, pos1_x0, pos1_x1, pos1_x2, pos1_x3]
    let input: [f32; 8] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let src = upload_f32(&dev, &input);
    let out_bytes = CudaStorageBytes::alloc(&dev, input.len() * 4).expect("alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));
    let kernel = pick_alt(
        &table,
        OpKind::Rope,
        fuel_dispatch::baracuda_dispatch::attention::rope_f32,
    );
    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::Rope {
            outer_count: 1,
            seq: 2,
            head_dim: 4,
        },
    )
    .expect("kernel call");

    let got = download_f32(&out_arc.read().unwrap());

    // Pos 0 (angle = 0): identity.
    for i in 0..4 {
        assert!(
            (got[i] - input[i]).abs() < 1e-5,
            "pos0 idx {i}: got {} expected {}",
            got[i],
            input[i],
        );
    }
    // Pos 1, pair (x_0=1, x_1=0): θ_0 = 1 · base^0 = 1.
    //   out_x0 = cos(1) · 1 - sin(1) · 0 = cos(1) ≈ 0.5403
    //   out_x1 = sin(1) · 1 + cos(1) · 0 = sin(1) ≈ 0.8415
    let cos1 = 1.0_f32.cos();
    let sin1 = 1.0_f32.sin();
    assert!(
        (got[4] - cos1).abs() < 1e-4,
        "pos1 x0: got {} expected cos(1) = {cos1}",
        got[4],
    );
    assert!(
        (got[5] - sin1).abs() < 1e-4,
        "pos1 x1: got {} expected sin(1) = {sin1}",
        got[5],
    );
}

// ===========================================================================
// FlashDecoding (decode-flash, seq_q==1, capacity-K) — Phase D step 2
// ===========================================================================

use fuel_ir::{Layout, Shape};
use half::{bf16, f16};

/// Upload raw `T` host data as a CUDA `Storage` of dtype `dt`.
fn upload<T: bytemuck::Pod>(dev: &CudaDevice, dt: DType, host: &[T]) -> Storage {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    let cuda = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda), dt)
}

fn download<T: bytemuck::Pod>(s: &Storage) -> Vec<T> {
    let bytes = match &s.inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("d2h"),
        _ => panic!("not on CUDA"),
    };
    bytemuck::cast_slice::<u8, T>(&bytes).to_vec()
}

/// Find the flash_decoding KernelRef at the FlashAttn no-alibi [T;4] key.
fn pick_flash(
    table: &KernelBindingTable,
    dt: DType,
    expected: fuel_dispatch::KernelRef,
) -> fuel_dispatch::KernelRef {
    let alternatives =
        table.lookup_alternatives(OpKind::FlashAttn, &[dt, dt, dt, dt], BackendId::Cuda);
    let expected_ptr = expected as usize;
    for alt in alternatives {
        if (alt.kernel as usize) == expected_ptr {
            return alt.kernel;
        }
    }
    panic!("expected flash_decoding KernelRef not found at ({dt:?};4)")
}

/// Host f32 oracle for decode SDPA (the decomposed base map): for each
/// (b, hq) attend the LOGICAL prefix `0..k_len` of the capacity K/V with
/// GQA head mapping. Inputs are the f16/bf16-rounded values (as f32) so the
/// reference matches the kernel's f32-accumulation-of-half-inputs numerics.
#[allow(clippy::too_many_arguments)]
fn decode_reference(
    q: &[f32], k: &[f32], v: &[f32],
    b: usize, hq: usize, hkv: usize, d: usize, sk: usize, k_len: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; b * hq * d];
    if k_len == 0 {
        return out; // no KV → zeros
    }
    let groups = hq / hkv;
    for bi in 0..b {
        for hi in 0..hq {
            let kv_h = hi / groups;
            let q_off = (bi * hq + hi) * d;
            let mut scores = vec![0.0f32; k_len];
            let mut m = f32::NEG_INFINITY;
            for (kj, sc) in scores.iter_mut().enumerate() {
                let k_off = ((bi * hkv + kv_h) * sk + kj) * d;
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q[q_off + dd] * k[k_off + dd];
                }
                *sc = dot * scale;
                if *sc > m {
                    m = *sc;
                }
            }
            let mut sum = 0.0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - m).exp();
                sum += *sc;
            }
            let inv = 1.0 / sum;
            let o_off = (bi * hq + hi) * d;
            for (kj, sc) in scores.iter().enumerate() {
                let p = *sc * inv;
                let v_off = ((bi * hkv + kv_h) * sk + kj) * d;
                for dd in 0..d {
                    out[o_off + dd] += p * v[v_off + dd];
                }
            }
        }
    }
    out
}

/// Deterministic pseudo-random f32 in [-0.5, 0.5).
fn prng(seed: &mut u64) -> f32 {
    // xorshift64*
    let mut x = *seed;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *seed = x;
    let u = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32;
    u - 0.5
}

/// Registration gate (NO GPU needed): the two FlashDecoding bindings must
/// register at the FlashAttn no-alibi [f16;4] / [bf16;4] CUDA keys. This is
/// the born-red gate — red before `register_baracuda_cuda_kernels` grows the
/// two `register_full(FlashAttn, …, flash_decoding::…)` lines, green after.
#[test]
fn register_baracuda_binds_flash_decoding_f16_bf16() {
    let mut table = KernelBindingTable::new();
    register_baracuda_cuda_kernels(&mut table);

    let f16_alts = table.lookup_alternatives(
        OpKind::FlashAttn, &[DType::F16, DType::F16, DType::F16, DType::F16], BackendId::Cuda);
    let want_f16 =
        fuel_dispatch::baracuda_dispatch::flash_decoding::flash_decoding_f16 as *const () as usize;
    assert!(
        f16_alts.iter().any(|a| (a.kernel as usize) == want_f16),
        "flash_decoding_f16 must register at (FlashAttn, [F16;4], Cuda)",
    );

    let bf16_alts = table.lookup_alternatives(
        OpKind::FlashAttn, &[DType::BF16, DType::BF16, DType::BF16, DType::BF16], BackendId::Cuda);
    let want_bf16 =
        fuel_dispatch::baracuda_dispatch::flash_decoding::flash_decoding_bf16 as *const () as usize;
    assert!(
        bf16_alts.iter().any(|a| (a.kernel as usize) == want_bf16),
        "flash_decoding_bf16 must register at (FlashAttn, [BF16;4], Cuda)",
    );
}

/// Shared live driver: GQA decode over a capacity-K buffer, compared to the
/// decomposed base-map reference within `tol`. Returns the max abs diff.
#[allow(clippy::too_many_arguments)]
fn run_flash_decode_case(
    dev: &CudaDevice,
    dt: DType,
    kernel: fuel_dispatch::KernelRef,
    to_half: impl Fn(f32) -> f32, // round-trip through the storage dtype
    upload_half: impl Fn(&CudaDevice, &[f32]) -> Storage,
    download_half: impl Fn(&Storage) -> Vec<f32>,
    b: usize, hq: usize, hkv: usize, d: usize, sk: usize, k_len: usize,
    tol: f32,
    seed0: u64,
) -> f32 {
    let scale = 1.0f32 / (d as f32).sqrt();
    // Generate f32, then round to the storage dtype so the reference sees
    // the exact values the kernel reads.
    let mut seed = seed0;
    let q_f32: Vec<f32> = (0..b * hq * d).map(|_| to_half(prng(&mut seed))).collect();
    let k_f32: Vec<f32> = (0..b * hkv * sk * d).map(|_| to_half(prng(&mut seed))).collect();
    let v_f32: Vec<f32> = (0..b * hkv * sk * d).map(|_| to_half(prng(&mut seed))).collect();

    let q = Arc::new(RwLock::new(upload_half(dev, &q_f32)));
    let k = Arc::new(RwLock::new(upload_half(dev, &k_f32)));
    let v = Arc::new(RwLock::new(upload_half(dev, &v_f32)));
    let out_bytes = b * hq * d * dt.size_in_bytes();
    let out = Arc::new(RwLock::new(Storage::new(
        BackendStorage::Cuda(CudaStorageBytes::alloc(dev, out_bytes).expect("alloc")),
        dt,
    )));

    let q_layout = Layout::contiguous(Shape::from_dims(&[b, hq, 1, d]));
    let k_layout = Layout::contiguous(Shape::from_dims(&[b, hkv, sk, d]));
    let v_layout = k_layout.clone();

    kernel(
        &[q.clone(), k.clone(), v.clone()],
        &mut [out.clone()],
        &[q_layout, k_layout, v_layout],
        &OpParams::FlashAttn {
            b, hq, hkv, sq: 1, sk, d, k_len,
            softmax_scale: scale,
            causal: true,
            window_size_left: None,
            window_size_right: None,
            softcap: None,
        },
    )
    .expect("flash_decoding launch");

    let got = download_half(&out.read().unwrap());
    let reference = decode_reference(&q_f32, &k_f32, &v_f32, b, hq, hkv, d, sk, k_len, scale);
    let mut max_diff = 0.0f32;
    for (g, r) in got.iter().zip(reference.iter()) {
        max_diff = max_diff.max((g - r).abs());
    }
    assert!(
        max_diff <= tol,
        "{dt:?} decode k_len={k_len}: max abs diff {max_diff} exceeds tol {tol}",
    );
    max_diff
}

/// f16 GQA decode (B=2, Hq=8, Hkv=2, D=64, cap=128) at k_len ∈ {1,37,128}
/// vs the decomposed base map. Tolerance is the f16 half-storage epsilon.
#[test]
#[ignore]
fn flash_decoding_f16_gqa_matches_base_map() {
    let Some(dev) = dev_or_skip() else { return };
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);
    let kernel = pick_flash(
        &table, DType::F16,
        fuel_dispatch::baracuda_dispatch::flash_decoding::flash_decoding_f16,
    );
    // (cap, k_len): single-split cases {1,37,128} + a multi-split case
    // (cap=384, k_len=300 ⇒ 2 splits) that exercises the combine kernel.
    for &(cap, k_len) in &[(128usize, 1usize), (128, 37), (128, 128), (384, 300)] {
        let diff = run_flash_decode_case(
            &dev, DType::F16, kernel,
            |x| f16::from_f32(x).to_f32(),
            |d, h| {
                let hv: Vec<f16> = h.iter().map(|&x| f16::from_f32(x)).collect();
                upload(d, DType::F16, &hv)
            },
            |s| download::<f16>(s).iter().map(|x| x.to_f32()).collect(),
            2, 8, 2, 64, cap, k_len,
            3.0e-2, 0x1234_5678 + (cap as u64) * 1000 + k_len as u64,
        );
        eprintln!("flash_decoding_f16 cap={cap} k_len={k_len}: max abs diff = {diff}");
    }
}

/// bf16 GQA decode — same shapes, coarser bf16 tolerance.
#[test]
#[ignore]
fn flash_decoding_bf16_gqa_matches_base_map() {
    let Some(dev) = dev_or_skip() else { return };
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);
    let kernel = pick_flash(
        &table, DType::BF16,
        fuel_dispatch::baracuda_dispatch::flash_decoding::flash_decoding_bf16,
    );
    for &k_len in &[1usize, 37, 128] {
        let diff = run_flash_decode_case(
            &dev, DType::BF16, kernel,
            |x| bf16::from_f32(x).to_f32(),
            |d, h| {
                let hv: Vec<bf16> = h.iter().map(|&x| bf16::from_f32(x)).collect();
                upload(d, DType::BF16, &hv)
            },
            |s| download::<bf16>(s).iter().map(|x| x.to_f32()).collect(),
            2, 8, 2, 64, 128, k_len,
            8.0e-2, 0xABCD_0001 + k_len as u64,
        );
        eprintln!("flash_decoding_bf16 k_len={k_len}: max abs diff = {diff}");
    }
}

/// k_len == 0 edge: the kernel writes nothing; the output must be the
/// zero-initialized buffer (all zeros).
#[test]
#[ignore]
fn flash_decoding_f16_klen_zero_is_zeros() {
    let Some(dev) = dev_or_skip() else { return };
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);
    let kernel = pick_flash(
        &table, DType::F16,
        fuel_dispatch::baracuda_dispatch::flash_decoding::flash_decoding_f16,
    );
    let (b, hq, hkv, d, sk) = (2usize, 4, 2, 32, 64);
    let q_f32: Vec<f16> = (0..b * hq * d).map(|i| f16::from_f32((i % 7) as f32)).collect();
    let kv: Vec<f16> = (0..b * hkv * sk * d).map(|i| f16::from_f32((i % 5) as f32)).collect();
    let q = Arc::new(RwLock::new(upload(&dev, DType::F16, &q_f32)));
    let k = Arc::new(RwLock::new(upload(&dev, DType::F16, &kv)));
    let v = Arc::new(RwLock::new(upload(&dev, DType::F16, &kv)));
    let out = Arc::new(RwLock::new(Storage::new(
        BackendStorage::Cuda(CudaStorageBytes::alloc(&dev, b * hq * d * 2).expect("alloc")),
        DType::F16,
    )));
    kernel(
        &[q, k, v],
        &mut [out.clone()],
        &[
            Layout::contiguous(Shape::from_dims(&[b, hq, 1, d])),
            Layout::contiguous(Shape::from_dims(&[b, hkv, sk, d])),
            Layout::contiguous(Shape::from_dims(&[b, hkv, sk, d])),
        ],
        &OpParams::FlashAttn {
            b, hq, hkv, sq: 1, sk, d, k_len: 0,
            softmax_scale: 0.125, causal: true,
            window_size_left: None, window_size_right: None, softcap: None,
        },
    )
    .expect("flash_decoding k_len=0");
    let got: Vec<f16> = download::<f16>(&out.read().unwrap());
    assert!(
        got.iter().all(|x| x.to_f32() == 0.0),
        "k_len==0 must leave the zero-initialized output untouched",
    );
}

/// Static-gate rejections: the wrapper hard-errors (fail-fast backstop) on
/// out-of-contract shapes the ranker is expected to have excluded —
/// seq_q!=1, GQA non-divisibility, and head_dim>128 (via `_can_implement`).
#[test]
#[ignore]
fn flash_decoding_f16_rejects_unsupported_shapes() {
    let Some(dev) = dev_or_skip() else { return };
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);
    let kernel = pick_flash(
        &table, DType::F16,
        fuel_dispatch::baracuda_dispatch::flash_decoding::flash_decoding_f16,
    );
    let mk = |b: usize, hq: usize, hkv: usize, sq: usize, sk: usize, d: usize| {
        let q = Arc::new(RwLock::new(upload(
            &dev, DType::F16,
            &vec![f16::from_f32(0.1); b * hq * sq * d],
        )));
        let kv = Arc::new(RwLock::new(upload(
            &dev, DType::F16,
            &vec![f16::from_f32(0.1); b * hkv * sk * d],
        )));
        let v = Arc::new(RwLock::new(upload(
            &dev, DType::F16,
            &vec![f16::from_f32(0.1); b * hkv * sk * d],
        )));
        let out = Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(
                CudaStorageBytes::alloc(&dev, b * hq * sq * d * 2).expect("alloc"),
            ),
            DType::F16,
        )));
        (q, kv, v, out)
    };

    // seq_q != 1 → decode kernel declines.
    {
        let (q, k, v, out) = mk(1, 2, 2, 3, 8, 32);
        let r = kernel(
            &[q, k, v], &mut [out],
            &[
                Layout::contiguous(Shape::from_dims(&[1, 2, 3, 32])),
                Layout::contiguous(Shape::from_dims(&[1, 2, 8, 32])),
                Layout::contiguous(Shape::from_dims(&[1, 2, 8, 32])),
            ],
            &OpParams::FlashAttn {
                b: 1, hq: 2, hkv: 2, sq: 3, sk: 8, d: 32, k_len: 8,
                softmax_scale: 0.1, causal: true,
                window_size_left: None, window_size_right: None, softcap: None,
            },
        );
        assert!(r.is_err(), "seq_q != 1 must be rejected");
    }
    // GQA non-divisible (Hq=8, Hkv=3).
    {
        let (q, k, v, out) = mk(1, 8, 3, 1, 8, 32);
        let r = kernel(
            &[q, k, v], &mut [out],
            &[
                Layout::contiguous(Shape::from_dims(&[1, 8, 1, 32])),
                Layout::contiguous(Shape::from_dims(&[1, 3, 8, 32])),
                Layout::contiguous(Shape::from_dims(&[1, 3, 8, 32])),
            ],
            &OpParams::FlashAttn {
                b: 1, hq: 8, hkv: 3, sq: 1, sk: 8, d: 32, k_len: 8,
                softmax_scale: 0.1, causal: true,
                window_size_left: None, window_size_right: None, softcap: None,
            },
        );
        assert!(r.is_err(), "Hq % Hkv != 0 must be rejected");
    }
    // head_dim > 128 → _can_implement returns 3.
    {
        let (q, k, v, out) = mk(1, 2, 2, 1, 4, 160);
        let r = kernel(
            &[q, k, v], &mut [out],
            &[
                Layout::contiguous(Shape::from_dims(&[1, 2, 1, 160])),
                Layout::contiguous(Shape::from_dims(&[1, 2, 4, 160])),
                Layout::contiguous(Shape::from_dims(&[1, 2, 4, 160])),
            ],
            &OpParams::FlashAttn {
                b: 1, hq: 2, hkv: 2, sq: 1, sk: 4, d: 160, k_len: 4,
                softmax_scale: 0.1, causal: true,
                window_size_left: None, window_size_right: None, softcap: None,
            },
        );
        assert!(r.is_err(), "head_dim > 128 must be rejected by _can_implement");
    }
}
