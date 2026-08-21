// SPDX-License-Identifier: MIT OR Apache-2.0
//! On-device integration test for the live CUDA JIT `load_kernel`
//! (`fuel_dispatch::jit_cuda_load::load_synth_kernel`) — the device-specific
//! step `jit_adopt::adopt_from_response` needs, per kernel-seam-interop §5.2.
//!
//! `#[ignore]`'d: needs a real CUDA device + the NVRTC runtime. Run manually
//! with `cargo test -p fuel-dispatch --features cuda,jit -- --ignored`.
//!
//! ## Why this drives a mock — and why that reason is now STALE (2026-08-06)
//!
//! **CORRECTION. This block previously claimed `baracuda-kernelgen` is
//! `publish = false` in its own `Cargo.toml` — never shipped to crates.io — and
//! concluded that wiring the real `BaracudaSynthesizer` would need a path dep
//! into the reference-only `../baracuda` checkout, i.e. an override of CLAUDE.md's
//! build discipline requiring explicit approval. THAT IS FALSE, and it was false
//! when written or shortly after.**
//!
//! Verified: `baracuda-kernelgen` **0.0.1-alpha.76 and alpha.77 are present in
//! the local crates.io registry cache** (`~/.cargo/registry/{src,cache}/
//! index.crates.io-*/baracuda-kernelgen-0.0.1-alpha.7{6,7}[.crate]`) — a `.crate`
//! tarball under `index.crates.io` can only have come from the registry. The
//! upstream manifest carries no `publish` key (so it defaults to publishable) and
//! comments that it was published **specifically so Fuel could construct
//! `BaracudaSynthesizer` from crates.io**, precisely because our build discipline
//! forbids path deps.
//!
//! So there is **no rule to override**: depending on it is an ordinary registry
//! dependency, pinned like every other baracuda crate. The blocker this comment
//! described does not exist.
//!
//! Cost of the stale claim, recorded because it is the reusable part: it was
//! read as authoritative and propagated — Fuel told a sibling project its JIT
//! seam was "blocked on nothing but the publish boundary" and asked to be pinged
//! when the crate published, which it already had. **A doc comment asserting an
//! external fact has a shelf life, and this one outlived its truth without any
//! signal.** External facts (is X published? what version?) belong in a check,
//! not a comment.
//!
//! What remains true: this test drives the seam (`fuel_kernel_seam::Synthesizer`)
//! with a small mock whose "compiled artifact" is real PTX — compiled at test
//! time by `baracuda-nvrtc` from hand-written CUDA-C matching the exact scalar
//! ABI `load_synth_kernel` expects. That exercises everything novel *here*
//! (module load, symbol resolve, slot claim, launch marshaling, real device
//! execution + result verification), and remains a useful isolation of the
//! loader from the generator.
//!
//! **Open follow-up, no longer gated on approval:** drive the seam end-to-end
//! against the real `BaracudaSynthesizer` (`baracuda-cuda-emit`, exact-pinned,
//! `--features seam,nvrtc`) — the first time Fuel's JIT path would meet a real
//! generator. The marquee region is `PagedAttn`'s dense recipe, which no GPU
//! backend implements as a fused op.

#![cfg(all(feature = "cuda", feature = "jit"))]

use std::sync::{Arc, RwLock};

use baracuda_kernels_types::{ArchSku, ElementKind, OperandDesc};
use baracuda_nvrtc::{CompileOptions, Program};
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_dispatch::jit_adopt::adopt_from_response;
use fuel_dispatch::jit_cuda_load::load_synth_kernel;
use fuel_dispatch::kernel::OpParams;
use fuel_dispatch::runtime_fused_kernels::{fused_kernel_available, lookup_runtime_kernel};
use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, Layout, Shape};
use fuel_kernel_seam::{
    ArtifactKind, JitBudget, JitRequest, JitResponse, LinkEntry, SynthArtifact, Synthesizer,
};
use fuel_memory::{BackendStorage, Storage};

fn dev_or_skip() -> Option<CudaDevice> {
    CudaDevice::new(0).ok()
}

/// The scalar-ABI source `load_synth_kernel` expects: `(const float* in0,
/// const float* in1, float* out, long long n)`, one grid-stride thread per
/// output element — byte-for-byte the shape `baracuda-cuda-emit`'s
/// `emit_scalar` builds for `relu(add(a, b))` at F32 (see
/// `jit_cuda_load.rs`'s module docs).
const ENTRY: &str = "fuel_test_jit_relu_add_f32_scalar";

fn relu_add_cuda_source() -> String {
    // Whitespace is cosmetic to the C compiler — no line-continuation
    // escaping subtleties needed, just a plain `.join("\n")`.
    [
        format!("extern \"C\" __global__ void {ENTRY}("),
        "    const float* __restrict__ in0,".to_string(),
        "    const float* __restrict__ in1,".to_string(),
        "    float* __restrict__ out,".to_string(),
        "    long long n) {".to_string(),
        "    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;".to_string(),
        "    long long step = (long long)gridDim.x * blockDim.x;".to_string(),
        "    for (; i < n; i += step) {".to_string(),
        "        float v = in0[i] + in1[i];".to_string(),
        "        out[i] = v > 0.0f ? v : 0.0f;".to_string(),
        "    }".to_string(),
        "}".to_string(),
    ]
    .join("\n")
}

fn compile_relu_add_ptx() -> Vec<u8> {
    let source = relu_add_cuda_source();
    let opts = CompileOptions::default();
    let ptx = Program::compile_with(&source, ENTRY, &opts)
        .unwrap_or_else(|e| panic!("nvrtc compile of the test relu(add) kernel failed: {e}"));
    ptx.into_bytes()
}

fn relu_add_region() -> PatternNode {
    PatternNode::Op {
        op: OpTag::Relu,
        attrs: OpAttrs::default(),
        operands: vec![PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![
                PatternNode::Bind { index: 0 },
                PatternNode::Bind { index: 1 },
            ],
        }],
    }
}

/// A mock mirroring Baracuda's real two-step handover shape (see this file's
/// module docs for why it's a mock, not the real `BaracudaSynthesizer`):
/// `synthesize` always accepts and retains one artifact; `take_kernel` hands
/// it over once, per `Synthesizer`'s single-adopt contract.
struct MockSynth {
    art: std::sync::Mutex<Option<SynthArtifact>>,
}

impl Synthesizer for MockSynth {
    fn synthesize(&self, _req: &JitRequest) -> JitResponse {
        JitResponse::Synthesized {
            entry_point: ENTRY.into(),
        }
    }
    fn take_kernel(&self, entry_point: &str) -> Option<SynthArtifact> {
        if entry_point != ENTRY {
            return None;
        }
        self.art.lock().unwrap().take()
    }
}

fn upload_f32(dev: &CudaDevice, host: &[f32]) -> Storage {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    let cuda_bytes = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda_bytes), DType::F32)
}

fn download_f32(s: &Storage) -> Vec<f32> {
    match &s.inner {
        BackendStorage::Cuda(c) => {
            bytemuck::cast_slice::<u8, f32>(&c.to_cpu_bytes().expect("d2h")).to_vec()
        }
        _ => panic!("not on CUDA"),
    }
}

/// Output buffer pre-filled with `NaN` — **NOT** `CudaStorageBytes::alloc`,
/// and the difference is the whole point.
///
/// `CudaStorageBytes::alloc` is zero-initialized (`byte_storage.rs:122`,
/// "`byte_count` zero-initialized bytes ... via `device.alloc_zeros::<u8>`").
/// With a zero-filled destination, **"the kernel wrote zeros" and "the kernel
/// never wrote at all" are byte-identical observations** — the instrument
/// cannot separate a body defect from a launch defect, and GAP-001's entire
/// recorded symptom is *"all-zero output, no error"*.
///
/// It is worse than the generic case here, because `relu(a + b)` on these
/// fixtures **legitimately produces mostly zeros**: 4 of 7 expected outputs in
/// [`live_baracuda_synthesizer_full_loop_scalar`] and 3 of 4 in
/// [`jit_adopt_loads_and_launches_a_synthesized_cuda_kernel`] are exactly
/// `0.0`. A zero-filled buffer agrees with a never-launched kernel on the
/// majority of the oracle.
///
/// `NaN` is the right sentinel rather than some improbable magic number: no
/// arithmetic in `relu(a + b)` or `x * p0` can *produce* `NaN` from finite
/// inputs, so a surviving `NaN` means "nothing was written to this element",
/// unambiguously. (`NaN != NaN` also makes it invisible to `assert_eq!`, which
/// is exactly why the survivor check below is a SEPARATE assertion — see
/// `assert_fully_written`.)
fn alloc_out_nan_filled(dev: &CudaDevice, n: usize) -> Arc<RwLock<Storage>> {
    Arc::new(RwLock::new(upload_f32(dev, &vec![f32::NAN; n])))
}

/// Separates *"never wrote"* from *"wrote the wrong values"*, and must be
/// called BEFORE the value comparison.
///
/// Order matters: `assert_eq!(got, want)` on a NaN-survivor reports a value
/// mismatch, which reads as a numerical/body defect and points the reader at
/// the generator. The survivor check names the real condition — the element
/// was never written — which is a launch/marshaling question and lands on a
/// different owner.
///
/// The *pattern* of survivors carries more than their count, so this reports it:
///
/// - **none** — every element was written; any failure after this point is a
///   VALUE defect, and the emitted body is a fair place to look.
/// - **all** — this buffer never received a write.
/// - **a suffix** — the kernel ran with too small an element count: a
///   count-unit/marshaling defect, not a body defect.
/// - **scattered** — a partial write, pointing at indexing/stride handling
///   inside the body.
///
/// The suffix case is the one a bare *"is anything still NaN?"* check gets
/// wrong, and it is worth spelling out because it is the obvious way to misuse
/// this sentinel: a kernel launched with the wrong `n` faithfully writes the
/// range it was told to, so the untouched tail reads as *never wrote*.
/// Separating the two needs the PATTERN, not the boolean. A zero-initialized
/// buffer cannot show either one.
///
/// **Bound on the "all" verdict — stated because it is easy to over-read.**
/// All-survivors means *this buffer was never written*, which is not the same
/// claim as *the kernel never launched*. A kernel that launched and wrote to a
/// different allocation — a freed or rebound output binding — produces the
/// byte-identical observation. Both candidate mechanisms sit on Fuel's side of
/// the seam, so the ownership verdict survives the ambiguity; the mechanism
/// does not follow from this assertion alone and must not be reported as if it
/// did.
fn assert_fully_written(got: &[f32], label: &str) {
    let unwritten: Vec<usize> = got
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_nan())
        .map(|(i, _)| i)
        .collect();
    if unwritten.is_empty() {
        return;
    }
    let n = got.len();
    // Indices arrive ascending, so "starts at n - len" is sufficient for a suffix.
    let is_suffix = unwritten.first() == Some(&(n - unwritten.len()));
    let verdict = if unwritten.len() == n {
        "NOTHING was written to this buffer. The kernel either never launched, or          launched and wrote somewhere else (a freed/rebound output binding). Both          mechanisms are on the launch/marshaling side rather than in the emitted          body — but this assertion does not distinguish them, so do not report          that it does."
    } else if is_suffix {
        "the written region is a PREFIX — the kernel ran with too small an element          count. That is a count-unit/marshaling defect (elements vs bytes, or a grid          sized off the wrong extent), NOT a defect in the emitted body."
    } else {
        "the survivors are SCATTERED — some elements were written and others          skipped, which points at indexing/stride handling inside the body."
    };
    panic!(
        "{label}: {} of {n} output elements still hold the pre-fill NaN —          {verdict}
With the zero-initialized alloc this test used before, every one          of these would have read as 0.0 and been indistinguishable from a correct          relu output.
Unwritten indices: {unwritten:?}
got: {got:?}",
        unwritten.len(),
    );
}

#[test]
#[ignore]
fn jit_adopt_loads_and_launches_a_synthesized_cuda_kernel() {
    let Some(device) = dev_or_skip() else {
        eprintln!(
            "skipping jit_adopt_loads_and_launches_a_synthesized_cuda_kernel: no CUDA device"
        );
        return;
    };

    let artifact = SynthArtifact {
        artifact: compile_relu_add_ptx(),
        kind: ArtifactKind::Ptx,
        link: LinkEntry {
            entry_point: ENTRY.into(),
            symbol: ENTRY.into(),
            structure_key: "elementwise:f32".into(),
            revision_hash: 1,
        },
        contract: "## fused_op: fuel_test_jit_relu_add\ncost: n\n".into(),
    };
    let synth = MockSynth {
        art: std::sync::Mutex::new(Some(artifact)),
    };

    let req = JitRequest {
        region: relu_add_region(),
        operands: vec![
            OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256),
            OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256),
        ],
        arch: ArchSku::Sm89,
        budget: JitBudget {
            max_compile_ms: 5_000,
        },
    };

    let adopted = adopt_from_response(&synth, &req, BackendId::Cuda, |art| {
        load_synth_kernel(art, &device)
    })
    .expect("adopt_from_response should not error")
    .expect("the mock synthesizer always synthesizes");
    let id = adopted.id;

    assert!(id.is_runtime(), "adopted a runtime FusedOpId");
    assert!(
        fused_kernel_available(id, BackendId::Cuda),
        "the adopted op's kernel is visible to the capability gate on Cuda",
    );

    // Exercise the loaded kernel for real: relu(a + b) on the device.
    let kernel = lookup_runtime_kernel(id, BackendId::Cuda)
        .expect("kernel bound on Cuda")
        .kernel;
    let a = [1.0_f32, -5.0, 2.0, -0.5];
    let b = [2.0_f32, 3.0, -10.0, 0.5];
    let lhs = Arc::new(RwLock::new(upload_f32(&device, &a)));
    let rhs = Arc::new(RwLock::new(upload_f32(&device, &b)));
    // NaN-prefilled, not zero-alloc'd: 3 of these 4 expected outputs are 0.0.
    let out = alloc_out_nan_filled(&device, a.len());

    let layout = Layout::contiguous(Shape::from_dims(&[a.len()]));
    kernel(
        &[lhs, rhs],
        &mut [out.clone()],
        &[layout.clone(), layout.clone(), layout],
        &OpParams::None,
    )
    .expect("launch");

    let got = download_f32(&out.read().unwrap());
    let want: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x + y).max(0.0))
        .collect();
    // CONTROL for the NaN prefill: this kernel is a hand-written PTX mock that
    // is known to work on the current pin. If the prefill itself were broken
    // (e.g. the upload didn't land, or `download_f32` re-read a stale buffer),
    // THIS test would report unwritten elements — so a clean run here is what
    // licenses reading a NaN survivor in the real-synthesizer test as a fact
    // about that kernel rather than about the instrument.
    assert_fully_written(&got, "mock relu(add) scalar kernel");
    assert_eq!(got, want, "relu(a + b) via the JIT-loaded CUDA kernel");
}

/// **GAP-214 born-red.** A JIT-loaded `CudaFunc` is valid only inside the CUDA
/// context that loaded it. `dispatch_slot` records the loading device's
/// [`DeviceId`] and refuses to launch against operands on ANY other device — a
/// typed error, never a launch. We assert the REFUSAL, not the bad launch: the
/// failure mode is UB, which does not reliably fail, so observing it is not a
/// test.
///
/// Two `CudaDevice::new(0)` on the SAME ordinal yield DISTINCT `DeviceId`s (the
/// id is minted from a monotonic counter), so this exercises
/// same-ordinal-different-context — GAP-001's actual symptom — not the easy
/// cross-ordinal case a much weaker check would also pass.
///
/// Three arms + the standing sabotage discipline:
/// 1. NEGATIVE CONTROL — all operands on the loading device → normal launch.
///    Without it the test passes for a dispatcher that refuses everything.
/// 2. all operands on a foreign context → typed refusal.
/// 3. PER-OPERAND DISCRIMINATOR — inputs on dev1, OUTPUT on dev2. A guard that
///    checks only one operand launches this (cross-context UB on the output);
///    the per-operand guard refuses. The arm a reviewer cannot see missing.
#[test]
#[ignore]
fn jit_kernel_refuses_operands_on_a_foreign_cuda_context() {
    let (Some(dev1), Some(dev2)) = (dev_or_skip(), dev_or_skip()) else {
        eprintln!(
            "skipping jit_kernel_refuses_operands_on_a_foreign_cuda_context: no CUDA device"
        );
        return;
    };
    assert_ne!(
        dev1.id(),
        dev2.id(),
        "two CudaDevice::new(0) on one ordinal must have distinct DeviceIds — the premise",
    );

    let artifact = SynthArtifact {
        artifact: compile_relu_add_ptx(),
        kind: ArtifactKind::Ptx,
        link: LinkEntry {
            entry_point: ENTRY.into(),
            symbol: ENTRY.into(),
            structure_key: "elementwise:f32".into(),
            revision_hash: 1,
        },
        contract: "## fused_op: fuel_test_jit_relu_add_gap214\ncost: n\n".into(),
    };
    // The slot records dev1's DeviceId at claim time.
    let kernel = load_synth_kernel(&artifact, &dev1).expect("load on dev1");

    let a = [1.0_f32, -5.0, 2.0, -0.5];
    let b = [2.0_f32, 3.0, -10.0, 0.5];
    let layout = Layout::contiguous(Shape::from_dims(&[a.len()]));
    let layouts = [layout.clone(), layout.clone(), layout];

    // Arm 1 — NEGATIVE CONTROL: matching device launches and computes relu(a+b).
    {
        let lhs = Arc::new(RwLock::new(upload_f32(&dev1, &a)));
        let rhs = Arc::new(RwLock::new(upload_f32(&dev1, &b)));
        let out = alloc_out_nan_filled(&dev1, a.len());
        kernel(&[lhs, rhs], &mut [out.clone()], &layouts, &OpParams::None)
            .expect("arm1: matching-device dispatch must LAUNCH, not refuse");
        let got = download_f32(&out.read().unwrap());
        assert_fully_written(&got, "gap214 arm1 (matching device)");
        let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| (x + y).max(0.0)).collect();
        assert_eq!(got, want, "arm1: relu(a + b) on the loading device");
    }

    // Arm 2 — all operands on dev2 (a foreign context) → typed refusal, no launch.
    {
        let lhs = Arc::new(RwLock::new(upload_f32(&dev2, &a)));
        let rhs = Arc::new(RwLock::new(upload_f32(&dev2, &b)));
        let out = alloc_out_nan_filled(&dev2, a.len());
        let err = kernel(&[lhs, rhs], &mut [out], &layouts, &OpParams::None)
            .expect_err("arm2: operands on a foreign context must be REFUSED, not launched");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("different") && msg.contains("context"),
            "arm2: the refusal must name the device-context mismatch, got: {msg}",
        );
    }

    // Arm 3 — PER-OPERAND DISCRIMINATOR: inputs on dev1, OUTPUT on dev2. A
    // check-one-operand impl (first input, on dev1) would launch and corrupt the
    // dev2 output buffer from dev1's context; the per-operand guard refuses.
    {
        let lhs = Arc::new(RwLock::new(upload_f32(&dev1, &a)));
        let rhs = Arc::new(RwLock::new(upload_f32(&dev1, &b)));
        let out = alloc_out_nan_filled(&dev2, a.len()); // OUTPUT on the wrong device
        kernel(&[lhs, rhs], &mut [out], &layouts, &OpParams::None)
            .expect_err("arm3: a mixed-device operand set (output on dev2) must be REFUSED");
    }
}

// ---- scalar-Param kernel (the trailing `float p{i}` ABI) -------------------

/// `mul_scalar` with ONE runtime param — the emitter's param'd scalar ABI:
/// `(const float* in0, float* out, long long n, float p0)` (the `p{i}` suffix
/// rides AFTER `long long n`, always `float`).
const PARAM_ENTRY: &str = "fuel_test_jit_mul_param_f32_scalar";

fn mul_param_cuda_source() -> String {
    [
        format!("extern \"C\" __global__ void {PARAM_ENTRY}("),
        "    const float* __restrict__ in0,".to_string(),
        "    float* __restrict__ out,".to_string(),
        "    long long n,".to_string(),
        "    float p0) {".to_string(),
        "    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;".to_string(),
        "    long long step = (long long)gridDim.x * blockDim.x;".to_string(),
        "    for (; i < n; i += step) {".to_string(),
        "        out[i] = in0[i] * p0;".to_string(),
        "    }".to_string(),
        "}".to_string(),
    ]
    .join("\n")
}

/// `mul_scalar(x)` with the value left OPEN (a slot template — the `extract:`
/// value rides the fused node's `Runtime { scalars }`, not the region).
fn mul_scalar_slot_region() -> PatternNode {
    PatternNode::Op {
        op: OpTag::MulScalar,
        attrs: OpAttrs::default(),
        operands: vec![PatternNode::Bind { index: 0 }],
    }
}

/// End-to-end scalar-Param launch: adopt a slot-template region whose kernel
/// takes a trailing `float p0`, then launch with `OpParams::JitScalars` and
/// verify the device computed `x * p0` with the LIVE value — proving the
/// `extract:` → `JitScalars` → trailing-`p{i}` marshaling on real hardware.
#[test]
#[ignore]
fn jit_scalar_param_kernel_launches_with_live_value() {
    let Some(device) = dev_or_skip() else {
        eprintln!("skipping jit_scalar_param_kernel_launches_with_live_value: no CUDA device");
        return;
    };

    let source = mul_param_cuda_source();
    let opts = CompileOptions::default();
    let ptx = Program::compile_with(&source, PARAM_ENTRY, &opts)
        .unwrap_or_else(|e| panic!("nvrtc compile of the mul_param kernel failed: {e}"));
    let artifact = SynthArtifact {
        artifact: ptx.into_bytes(),
        kind: ArtifactKind::Ptx,
        link: LinkEntry {
            entry_point: PARAM_ENTRY.into(),
            symbol: PARAM_ENTRY.into(),
            structure_key: "elementwise:f32:p1".into(),
            revision_hash: 2,
        },
        contract: "## fused_op: fuel_test_jit_mul_param\ncost: n\n".into(),
    };
    struct ParamSynth {
        art: std::sync::Mutex<Option<SynthArtifact>>,
    }
    impl Synthesizer for ParamSynth {
        fn synthesize(&self, _req: &JitRequest) -> JitResponse {
            JitResponse::Synthesized {
                entry_point: PARAM_ENTRY.into(),
            }
        }
        fn take_kernel(&self, entry_point: &str) -> Option<SynthArtifact> {
            if entry_point != PARAM_ENTRY {
                return None;
            }
            self.art.lock().unwrap().take()
        }
    }
    let synth = ParamSynth {
        art: std::sync::Mutex::new(Some(artifact)),
    };

    let req = JitRequest {
        region: mul_scalar_slot_region(),
        operands: vec![OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256)],
        arch: ArchSku::Sm89,
        budget: JitBudget {
            max_compile_ms: 5_000,
        },
    };
    let adopted = adopt_from_response(&synth, &req, BackendId::Cuda, |art| {
        load_synth_kernel(art, &device)
    })
    .expect("adopt_from_response should not error")
    .expect("the mock synthesizer always synthesizes");
    let id = adopted.id;
    assert!(id.is_runtime());

    let kernel = lookup_runtime_kernel(id, BackendId::Cuda)
        .expect("kernel bound on Cuda")
        .kernel;
    let x = [1.0_f32, -5.0, 2.0, -0.5];
    let inp = Arc::new(RwLock::new(upload_f32(&device, &x)));
    // NaN-prefilled. This test's oracle happens to contain no zeros, so it is
    // the ONE case where the old zero-alloc was not ambiguous — kept uniform
    // anyway, because "which fixtures happen to avoid 0.0" is not a property
    // anyone should have to re-derive when editing the fixture.
    let out = alloc_out_nan_filled(&device, x.len());

    let layout = Layout::contiguous(Shape::from_dims(&[x.len()]));
    kernel(
        &[inp],
        &mut [out.clone()],
        &[layout.clone(), layout],
        // The live `extract:` value — exactly what compile_one's is_runtime arm
        // produces from the fused node's Runtime { scalars }.
        &OpParams::JitScalars { scalars: vec![2.5] },
    )
    .expect("launch with a trailing float p0");

    let got = download_f32(&out.read().unwrap());
    let want: Vec<f32> = x.iter().map(|v| v * 2.5).collect();
    assert_fully_written(&got, "mock mul_param scalar-Param kernel");
    assert_eq!(
        got, want,
        "x * p0 via the JIT-loaded scalar-Param CUDA kernel"
    );
}

// ---- the REAL BaracudaSynthesizer (alpha.76, published) — the milestone -----

/// **THE MILESTONE**: the full JIT-on-request loop end-to-end on PUBLISHED
/// crates, driving Baracuda's OWN synthesizer (`baracuda-kernelgen` alpha.76's
/// `seam::BaracudaSynthesizer`) rather than a mock — `synthesize -> take_kernel
/// -> load_synth_kernel -> launch -> verify`, for a scalar-schedule relu(add).
///
/// The operands are declared **element-aligned (4 B)**, so Baracuda's emitter
/// keys `vec_width = 1` and picks the **Scalar** schedule → a `baracuda_gen_
/// ..._scalar` kernel that `load_synth_kernel` handles today (the vectorized /
/// strided ABIs are the documented loader follow-up). Gated behind `jit-synth`
/// (= jit + cuda + baracuda-cuda-emit{seam,nvrtc}); run with
/// `cargo test -p fuel-dispatch --features jit-synth -- --ignored`.
#[test]
#[ignore]
#[cfg(feature = "jit-synth")]
fn live_baracuda_synthesizer_full_loop_scalar() {
    use baracuda_cuda_emit::seam::BaracudaSynthesizer;
    use fuel_kernel_seam::{JitBudget, JitRequest, JitResponse, Synthesizer};

    let Some(device) = dev_or_skip() else {
        eprintln!("skipping live_baracuda_synthesizer_full_loop_scalar: no CUDA device");
        return;
    };

    let synth = BaracudaSynthesizer::new(5_000);
    // Baracuda's OpDef contract: `operands` holds exactly n_inputs + 1 entries —
    // the region's inputs (in bind order) THEN the output. relu(add(a,b)) has 2
    // inputs, so 3 operands ([a, b, out]); this also IS the binding-key dtype
    // tuple `build_lookup_dtypes` produces (inputs then output). Element-aligned
    // (4 B) + a non-vector-multiple count → the Scalar schedule our loader handles.
    let operand = || OperandDesc::new(1, &[7], &[1], ElementKind::F32, 4);
    let req = JitRequest {
        region: relu_add_region(),
        operands: vec![operand(), operand(), operand()],
        arch: ArchSku::Sm89,
        budget: JitBudget {
            max_compile_ms: 5_000,
        },
    };

    // (1) The synthesizer accepts + builds the region (independent of our loader).
    match synth.synthesize(&req) {
        JitResponse::Synthesized { entry_point } => {
            eprintln!("BaracudaSynthesizer synthesized: {entry_point}");
        }
        JitResponse::Declined { reason } => {
            panic!("BaracudaSynthesizer declined relu(add): {reason}");
        }
    }

    // (2) The full adopt path: (re-)synthesize -> take_kernel -> load_synth_kernel.
    let adopted = adopt_from_response(&synth, &req, BackendId::Cuda, |art| {
        eprintln!(
            "synth emitted symbol: {}  (kind {:?})",
            art.link.symbol, art.kind
        );
        // GAP-001's "First diagnostic" (docs/gaps.md), made runnable: dump what the
        // synthesizer DECLARES, so it can be compared against what Fuel COMPUTES.
        // The row records that diagnostic as "cuda-build-blocked"; it is closer to
        // NOT RUNNABLE AS WRITTEN, because the seam carries no launch-geometry
        // field at all. `SynthArtifact` is {artifact, kind, link{entry_point,
        // symbol, structure_key, revision_hash}, contract} — so if a count/launch
        // declaration exists anywhere it can only be inside `contract`, and no
        // checked-in `.fkc.md` in this tree declares a `count_unit:`. Whether
        // Baracuda EMITS one at runtime is the open half, and printing settles it.
        //
        // Observation only, deliberately: no assertion, and it does not touch the
        // launch path, so the experiment stays single-variable.
        //
        // MEASURED 2026-08-19, kernelgen =0.0.1-alpha.78, and it REFUTES the
        // paragraph above (kept, because the wrong prediction is the point):
        // the synthesizer DOES declare the field, at runtime, inside `contract`:
        //
        //     count_unit: elements
        //     class: elementwise
        //
        // Fuel computes `n` in ELEMENTS (`layouts[n_inputs].shape().elem_count()`
        // = 7). The declaration says ELEMENTS. **They agree**, so by the GAP-001
        // row's own rule — agree => body, disagree => contract — this is NOT a
        // contract mismatch, and the row's "cuda-build-blocked" status was
        // mislabelling a diagnostic that runs fine.
        //
        // Also emitted: `structure_key: sk3|bin|f32|cuda:sm89|ix32|warp|r1|...`
        // — an sk3 key from a retired crate line. Flagged, not interpreted.
        eprintln!("  structure_key: {}", art.link.structure_key);
        eprintln!("  revision_hash: {:#x}", art.link.revision_hash);
        let mut declared = 0usize;
        for line in art.contract.lines() {
            let l = line.to_ascii_lowercase();
            if l.contains("count")
                || l.contains("launch")
                || l.contains("grid")
                || l.contains("block")
                || l.contains("schedule")
                || l.contains("elem")
            {
                declared += 1;
                eprintln!("  contract| {}", line.trim_end());
            }
        }
        eprintln!("  contract lines declaring count/launch geometry: {declared}");
        load_synth_kernel(art, &device)
    })
    .expect("adopt_from_response: the full loop reached adopt")
    .expect("the real synthesizer produced an adoptable kernel");
    let id = adopted.id;

    assert!(id.is_runtime(), "adopted a runtime FusedOpId");
    assert!(
        fused_kernel_available(id, BackendId::Cuda),
        "the adopted kernel is visible to the capability gate on Cuda",
    );

    // (3) Launch Baracuda's generated kernel and verify relu(a + b) on-device.
    // Dispatch through the kernel THIS adoption loaded, never through a lookup
    // by id. Was `lookup_runtime_kernel(id, ..).kernel`, and that is the whole
    // GAP-001 defect (see the assertion below).
    let kernel = adopted.kernel;
    // *** GAP-001 (b): the kernel we LAUNCH must be the kernel we ADOPTED. ***
    // Born red on purpose. A `FusedOpId` names a RECIPE, not an artifact, so
    // resolving one back to "the" kernel returns an ALTERNATIVE — today the
    // first-registered (GAP-213). An earlier test in this binary synthesizes
    // the same `relu_add_region()` at f32, so it registers first and this
    // lookup hands back ITS kernel, bound to a `CudaDevice` already dropped.
    // Both kernels compute relu(add) on f32, so whenever that stale launch
    // happens to work the values are CORRECT and this test passes while
    // exercising the mock instead of the live synthesizer.
    // Trivially true as written — and that is the point: it fails the moment
    // anyone reintroduces a lookup-by-id here. Observed RED before the line
    // above changed (~2 of 5 full-suite runs); never RED alone, because alone
    // there is no second alternative.
    //
    // MEASURED mechanism, and it is worse than "first-registered wins"
    // (GAP-213). `bindings` is keyed on `(BindingKey, KernelDTypes, BackendId)`
    // — but `first_runtime_fused`'s predicate matches on `fid` and `backend`
    // ONLY and ignores dtypes. An earlier test in this binary adopts the same
    // `relu_add_region()` declaring 2 operands (`[F32, F32]`); this one
    // declares 3 (`[F32, F32, F32]`). Same recipe id, same backend, DIFFERENT
    // dtype keys ⇒ two distinct HashMap entries, BOTH matching the predicate,
    // and `HashMap::iter()` order decides which one answers. Order is seeded
    // per process, so the selection — and therefore whether this test launched
    // its own kernel or a kernel bound to an already-dropped `CudaDevice` — was
    // a coin flip on every run. That is the ~40%.
    assert!(
        std::ptr::fn_addr_eq(kernel, adopted.kernel),
        "launching a DIFFERENT kernel than was adopted — an id resolved to another          alternative. Dispatch through `adopted.kernel`; an id names a RECIPE, and          a recipe legitimately has many kernels (GAP-213).",
    );
    let a = [1.0_f32, -5.0, 2.0, -0.5, 3.0, -7.0, 0.0];
    let b = [2.0_f32, 3.0, -10.0, 0.5, -1.0, 7.0, 4.0];
    let lhs = Arc::new(RwLock::new(upload_f32(&device, &a)));
    let rhs = Arc::new(RwLock::new(upload_f32(&device, &b)));
    // ⚠️ THE GAP-001 DISCRIMINATOR. Expected output here is
    // [3, 0, 0, 0, 2, 0, 4] — FOUR of seven elements are legitimately 0.0.
    // Against the zero-filled alloc this line used to use, a kernel that never
    // executed produced [0, 0, 0, 0, 0, 0, 0], agreeing with the correct answer
    // on the majority of the oracle and differing only where relu happened to
    // be positive. That is the instrument that recorded GAP-001's symptom as
    // "all-zero output, no error" — a description that fits BOTH a .78 body
    // defect (theirs) and a launch/marshaling failure (ours), which is why the
    // 77→78 bisect could never settle the root cause.
    let out = alloc_out_nan_filled(&device, a.len());
    let layout = Layout::contiguous(Shape::from_dims(&[a.len()]));
    kernel(
        &[lhs, rhs],
        &mut [out.clone()],
        &[layout.clone(), layout.clone(), layout],
        &OpParams::None,
    )
    .expect("launch Baracuda's synthesized relu(add)");

    let got = download_f32(&out.read().unwrap());
    let want: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x + y).max(0.0))
        .collect();
    // ⚠️ THE DISCRIMINATOR, and it must be read BEFORE the value comparison.
    //
    //   surviving NaN  ⇒ the kernel NEVER WROTE these elements. A launch /
    //                    marshaling failure — OURS. The 77→78 bisect would then
    //                    be pinned to the wrong side entirely, and root
    //                    `Cargo.toml`'s exact `baracuda-kernelgen = "=0.0.1-
    //                    alpha.77"` would be holding back a generator that was
    //                    never at fault.
    //   genuine zeros  ⇒ the kernel RAN and produced zeros. A body/contract
    //                    defect on the generator side — THEIRS, and the bisect
    //                    stands as recorded.
    //
    // Reporting either answer closes the "root-cause sub-form UNRESOLVED" note
    // on the GAP-001 row; the point of this line is that they are now
    // DISTINGUISHABLE, which under the previous zero-filled alloc they were not.
    assert_fully_written(&got, "REAL BaracudaSynthesizer scalar kernel (GAP-001)");
    assert_eq!(
        got, want,
        "relu(a + b) via Baracuda's OWN alpha.76-synthesized CUDA kernel — the LIVE LOOP",
    );
}

/// **Does a real synthesizer accept `PagedAttn`'s dense region?** — the probe
/// KISS is gating its region-synthesis RFC on.
///
/// This is the marquee case for JIT-from-a-discovered-region: `Op::PagedAttn`
/// has **no GPU implementation anywhere**, so under CUDA the fused node is
/// host-placed and the KV caches cross the bus every token. Lowering to
/// primitives needs every primitive bound; handing the whole region to a
/// synthesizer needs none of them. It is also far larger than the
/// `relu(add(a,b))` cell the rest of this file exercises — two `IndexSelect`
/// gathers, two `MatMul`s, a nested `Fused(SOFTMAX_LAST_DIM)`, and a
/// variable-length mask chain.
///
/// **The question is willingness, not correctness.** `JitResponse::Declined` is
/// a first-class answer and is reported as a RESULT, not a failure — a decline
/// tells KISS that JIT-from-a-region is out of scope for this generator, which
/// is as useful for the RFC as acceptance.
///
/// The region comes from Fuel's own `recipe_for` — the same `PatternNode`
/// `decompose` lowers — so this asks about the region Fuel actually runs, not a
/// hand-written approximation.
///
/// **Operands are typed honestly.** `block_table` and `context_lens` are `U32`
/// (`ElementKind::U32`, the seam's designated gather/scatter index ctype). An
/// earlier attempt at this probe was abandoned rather than substitute `I32`:
/// an answer obtained by misrepresenting the operands is an answer about a
/// different kernel.
#[test]
#[ignore]
#[cfg(feature = "jit-synth")]
fn live_baracuda_synthesizer_paged_attn_dense_region() {
    use baracuda_cuda_emit::seam::BaracudaSynthesizer;
    use fuel_graph::registry::{FusedOpParams, FusedOps};
    use fuel_graph::{Graph, Node, NodeId, Op};

    const B: usize = 1;
    const HQ: usize = 4;
    const HKV: usize = 4;
    const SQ: usize = 1;
    const D: usize = 4;
    const NUM_BLOCKS: usize = 32;
    const BLOCK_SIZE: usize = 4;
    const MAX_BLK: usize = 8;

    let mut g = Graph::new();
    let leaf = |g: &mut Graph, dims: &[usize], dtype: DType| {
        g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(dims),
            dtype,
        })
    };
    let q = leaf(&mut g, &[B, HQ, SQ, D], DType::F32);
    let kc = leaf(&mut g, &[NUM_BLOCKS, BLOCK_SIZE, HKV, D], DType::F32);
    let vc = leaf(&mut g, &[NUM_BLOCKS, BLOCK_SIZE, HKV, D], DType::F32);
    let bt = leaf(&mut g, &[B, MAX_BLK], DType::U32);
    let cl = leaf(&mut g, &[B], DType::U32);

    let params = FusedOpParams::PagedAttn {
        softmax_scale: 1.0 / (D as f32).sqrt(),
        block_size: BLOCK_SIZE,
        softcap: None,
    };
    let fused = g.push(Node {
        op: Op::Fused(FusedOps::PAGED_ATTN, params.clone()),
        inputs: vec![q, kc, vc, bt, cl],
        shape: Shape::from_dims(&[B, HQ, SQ, D]),
        dtype: DType::F32,
    });

    // Fuel's OWN region — the same PatternNode `decompose` lowers.
    let region = fuel_graph::registry::paged_attn::recipe_for(&g, fused, &params)
        .expect("well-formed PagedAttn node yields a recipe");
    // CONTROL: a real Op tree, not a degenerate stub. Without this a
    // `recipe_for` returning a bare Bind would make any answer meaningless.
    assert!(
        matches!(region, PatternNode::Op { .. }),
        "the region must be an Op tree",
    );

    // Operands in bind order, then the output (the seam's OpDef convention).
    let od = |dims: &[usize], k: ElementKind, elem: u32| {
        let mut strides = vec![1usize; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        let d: Vec<i64> = dims.iter().map(|&x| x as i64).collect();
        let s: Vec<i64> = strides.iter().map(|&x| x as i64).collect();
        OperandDesc::new(dims.len(), &d, &s, k, elem)
    };
    let operands = vec![
        od(&[B, HQ, SQ, D], ElementKind::F32, 4), // 0 q
        od(&[NUM_BLOCKS, BLOCK_SIZE, HKV, D], ElementKind::F32, 4), // 1 k_cache
        od(&[NUM_BLOCKS, BLOCK_SIZE, HKV, D], ElementKind::F32, 4), // 2 v_cache
        od(&[B, MAX_BLK], ElementKind::U32, 4),   // 3 block_table
        od(&[B], ElementKind::U32, 4),            // 4 context_lens
        od(&[B, HQ, SQ, D], ElementKind::F32, 4), // out
    ];

    let synth = BaracudaSynthesizer::new(10_000);
    let req = JitRequest {
        region,
        operands,
        arch: ArchSku::Sm89,
        budget: JitBudget {
            max_compile_ms: 10_000,
        },
    };

    println!("\n=== PagedAttn dense region -> real BaracudaSynthesizer ===");
    match synth.synthesize(&req) {
        JitResponse::Synthesized { entry_point } => {
            println!("RESULT: ACCEPTED — entry_point = {entry_point}");
            println!(
                "The generator will synthesize a fused kernel for a discovered region \
                 of this size. JIT-from-a-region is in scope for it."
            );
        }
        JitResponse::Declined { reason } => {
            println!("RESULT: DECLINED — reason: {reason}");
            println!(
                "A decline is a first-class answer, not a failure: it says \
                 JIT-from-a-discovered-region of this shape is out of scope for this \
                 generator, which is what the RFC needs to know."
            );
        }
    }
    println!("=== END RESULT ===\n");
}
