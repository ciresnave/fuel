//! On-device integration test for the live CUDA JIT `load_kernel`
//! (`fuel_dispatch::jit_cuda_load::load_synth_kernel`) — the device-specific
//! step `jit_adopt::adopt_from_response` needs, per kernel-seam-interop §5.2.
//!
//! `#[ignore]`'d: needs a real CUDA device + the NVRTC runtime. Run manually
//! with `cargo test -p fuel-dispatch --features cuda,jit -- --ignored`.
//!
//! ## Why this doesn't drive `baracuda_kernelgen::jit::seam::BaracudaSynthesizer`
//!
//! The task that produced this test asked for the real `BaracudaSynthesizer`
//! (`baracuda-kernelgen`, `--features seam,nvrtc`). That crate is
//! `publish = false` in its own `Cargo.toml` — never shipped to crates.io —
//! and CLAUDE.md's build-discipline section is explicit that "baracuda...
//! comes from crates.io pinned... a local `../baracuda` checkout is
//! **reference-only**". Depending on it here would mean a path dependency
//! into that checkout, which is exactly what that rule rules out. So this
//! test instead drives the same seam (`fuel_kernel_seam::Synthesizer`) with a
//! small mock synthesizer (the same shape `jit_adopt.rs`'s own unit tests
//! use), whose "compiled artifact" is real PTX — compiled at test time by
//! `baracuda-nvrtc` (a properly crates.io-pinned baracuda crate, not the
//! reference-only checkout) from a hand-written CUDA-C source matching the
//! exact scalar ABI `load_synth_kernel` expects. This exercises every part of
//! `load_synth_kernel` that is actually novel here (module load, symbol
//! resolve, slot claim, launch marshaling, real device execution + result
//! verification) without the disputed dependency. If the real
//! `BaracudaSynthesizer` wiring is wanted anyway, that's a follow-up someone
//! should explicitly approve (it means overriding the reference-only rule).

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
/// output element — byte-for-byte the shape `baracuda-kernelgen`'s
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
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
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
        JitResponse::Synthesized { entry_point: ENTRY.into() }
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

#[test]
#[ignore]
fn jit_adopt_loads_and_launches_a_synthesized_cuda_kernel() {
    let Some(device) = dev_or_skip() else {
        eprintln!("skipping jit_adopt_loads_and_launches_a_synthesized_cuda_kernel: no CUDA device");
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
    let synth = MockSynth { art: std::sync::Mutex::new(Some(artifact)) };

    let req = JitRequest {
        region: relu_add_region(),
        operands: vec![
            OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256),
            OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256),
        ],
        arch: ArchSku::Sm89,
        budget: JitBudget { max_compile_ms: 5_000 },
    };

    let id = adopt_from_response(&synth, &req, BackendId::Cuda, |art| {
        load_synth_kernel(art, &device)
    })
    .expect("adopt_from_response should not error")
    .expect("the mock synthesizer always synthesizes");

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
    let out_bytes = CudaStorageBytes::alloc(&device, a.len() * 4).expect("alloc out");
    let out = Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(out_bytes), DType::F32)));

    let layout = Layout::contiguous(Shape::from_dims(&[a.len()]));
    kernel(
        &[lhs, rhs],
        &mut [out.clone()],
        &[layout.clone(), layout.clone(), layout],
        &OpParams::None,
    )
    .expect("launch");

    let got = download_f32(&out.read().unwrap());
    let want: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| (x + y).max(0.0)).collect();
    assert_eq!(got, want, "relu(a + b) via the JIT-loaded CUDA kernel");
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
            JitResponse::Synthesized { entry_point: PARAM_ENTRY.into() }
        }
        fn take_kernel(&self, entry_point: &str) -> Option<SynthArtifact> {
            if entry_point != PARAM_ENTRY {
                return None;
            }
            self.art.lock().unwrap().take()
        }
    }
    let synth = ParamSynth { art: std::sync::Mutex::new(Some(artifact)) };

    let req = JitRequest {
        region: mul_scalar_slot_region(),
        operands: vec![OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256)],
        arch: ArchSku::Sm89,
        budget: JitBudget { max_compile_ms: 5_000 },
    };
    let id = adopt_from_response(&synth, &req, BackendId::Cuda, |art| {
        load_synth_kernel(art, &device)
    })
    .expect("adopt_from_response should not error")
    .expect("the mock synthesizer always synthesizes");
    assert!(id.is_runtime());

    let kernel = lookup_runtime_kernel(id, BackendId::Cuda)
        .expect("kernel bound on Cuda")
        .kernel;
    let x = [1.0_f32, -5.0, 2.0, -0.5];
    let inp = Arc::new(RwLock::new(upload_f32(&device, &x)));
    let out_bytes = CudaStorageBytes::alloc(&device, x.len() * 4).expect("alloc out");
    let out = Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(out_bytes), DType::F32)));

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
    assert_eq!(got, want, "x * p0 via the JIT-loaded scalar-Param CUDA kernel");
}
