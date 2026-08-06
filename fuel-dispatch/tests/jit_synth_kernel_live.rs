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
//! against the real `BaracudaSynthesizer` (`baracuda-kernelgen`, exact-pinned,
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
/// (= jit + cuda + baracuda-kernelgen{seam,nvrtc}); run with
/// `cargo test -p fuel-dispatch --features jit-synth -- --ignored`.
#[test]
#[ignore]
#[cfg(feature = "jit-synth")]
fn live_baracuda_synthesizer_full_loop_scalar() {
    use baracuda_kernelgen::jit::seam::BaracudaSynthesizer;
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
        budget: JitBudget { max_compile_ms: 5_000 },
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
    let id = adopt_from_response(&synth, &req, BackendId::Cuda, |art| {
        eprintln!("synth emitted symbol: {}  (kind {:?})", art.link.symbol, art.kind);
        load_synth_kernel(art, &device)
    })
    .expect("adopt_from_response: the full loop reached adopt")
    .expect("the real synthesizer produced an adoptable kernel");

    assert!(id.is_runtime(), "adopted a runtime FusedOpId");
    assert!(
        fused_kernel_available(id, BackendId::Cuda),
        "the adopted kernel is visible to the capability gate on Cuda",
    );

    // (3) Launch Baracuda's generated kernel and verify relu(a + b) on-device.
    let kernel = lookup_runtime_kernel(id, BackendId::Cuda)
        .expect("kernel bound on Cuda")
        .kernel;
    let a = [1.0_f32, -5.0, 2.0, -0.5, 3.0, -7.0, 0.0];
    let b = [2.0_f32, 3.0, -10.0, 0.5, -1.0, 7.0, 4.0];
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
    .expect("launch Baracuda's synthesized relu(add)");

    let got = download_f32(&out.read().unwrap());
    let want: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| (x + y).max(0.0)).collect();
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
    use baracuda_kernelgen::jit::seam::BaracudaSynthesizer;
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
        g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(dims), dtype })
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
        od(&[B, HQ, SQ, D], ElementKind::F32, 4),                 // 0 q
        od(&[NUM_BLOCKS, BLOCK_SIZE, HKV, D], ElementKind::F32, 4), // 1 k_cache
        od(&[NUM_BLOCKS, BLOCK_SIZE, HKV, D], ElementKind::F32, 4), // 2 v_cache
        od(&[B, MAX_BLK], ElementKind::U32, 4),                   // 3 block_table
        od(&[B], ElementKind::U32, 4),                            // 4 context_lens
        od(&[B, HQ, SQ, D], ElementKind::F32, 4),                 // out
    ];

    let synth = BaracudaSynthesizer::new(10_000);
    let req = JitRequest {
        region,
        operands,
        arch: ArchSku::Sm89,
        budget: JitBudget { max_compile_ms: 10_000 },
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
