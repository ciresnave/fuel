//! Live CUDA `load_kernel` for a JIT-synthesized kernel (kernel-seam-interop
//! §5.2's device-specific step): given a [`SynthArtifact`] (PTX bytes + a
//! symbol name), load it as a CUDA module, resolve the function, and wrap it
//! as a Fuel [`KernelRef`] — the closure
//! [`crate::jit_adopt::adopt_from_response`] takes as `load_kernel`.
//!
//! ## Why this lives in fuel-dispatch, not fuel-cuda-backend
//!
//! The task shape suggests a `fuel-cuda-backend` function, but `KernelRef`
//! is defined in [`crate::kernel`] over `fuel_memory::Storage` — and
//! `fuel-memory` already depends on `fuel-cuda-backend` (its `cuda` feature
//! re-exports `CudaStorageBytes`), so the reverse dependency
//! (`fuel-cuda-backend` → `fuel-memory`/`fuel-kernel-seam`) would be a cycle.
//! `fuel-dispatch` already depends on both (optionally, behind `cuda` and
//! `jit`), so the loader lives here, gated behind both features — the actual
//! module-load + symbol-resolve step reuses
//! [`fuel_cuda_backend::CudaDevice::get_or_load_custom_func`] verbatim
//! (already exactly "load PTX as a module, resolve a symbol, cache by name,
//! return a launchable `CudaFunc`" — no new fuel-cuda-backend code needed).
//!
//! ## The closure problem: why a slot table
//!
//! [`KernelRef`] is a bare `fn` pointer with no environment — a *capturing*
//! closure over the just-loaded `CudaFunc` cannot coerce to it. So a
//! dynamically loaded PTX kernel's identity (which `CudaFunc` to launch) has
//! to live in a process-global slot, and the `KernelRef` this module hands
//! back is one of a small, fixed bank of monomorphized `dispatch_slot::<N>`
//! functions — each a genuinely distinct compiled function (its own address),
//! so `N` is baked into the *code* at compile time, not captured at runtime.
//! [`load_synth_kernel`] claims the next free slot and returns its dispatcher.
//! Slots are never freed (a `FusedOpId`'s kernel lives for the process, per
//! [`crate::runtime_fused_kernels::adopt_runtime_fused`]'s docs), so the bank
//! is sized generously; bump [`MAX_JIT_SLOTS`] + the `dispatch_table!` call if
//! a workload ever adopts more distinct JIT kernels than that in one process.
//!
//! ## Launch-marshaling scope (read before trusting this on a new region shape)
//!
//! `baracuda-kernelgen`'s emitter chooses one of several kernel ABIs per
//! `Schedule` (`Scalar` / `Vectorized{width}` / `Strided`) — see its
//! `cuda.rs::Cuda::lower`. Only the **scalar** ABI is implemented here:
//! `(const T* in0, .., const T* inK, T* out, long long n)`, one thread per
//! output element, `n` = the output's element count. This is what
//! `emit_scalar` builds for a small, fully-contiguous, uniform-dtype,
//! parameter-free elementwise region — the common case a Fuel-chosen fusion
//! region is expected to hit. It is **not** what `Vectorized` (pointee
//! reinterpreted as `float4`/`float2`/a packed f16-pair struct, `n` counted in
//! *vector* units per `effective_count_width`) or `Strided` (extra by-value
//! shape/stride args, needed for broadcast/gather/scatter/non-contiguous
//! regions) build. Rather than launch the wrong ABI silently — the exact
//! failure mode this loader must never produce — [`load_synth_kernel`]
//! declines (a typed `Err`) any artifact whose `link.symbol` doesn't end in
//! `_scalar`. A robust version would read the FKC contract's schedule/
//! `count_unit:` metadata instead of sniffing the symbol suffix; sniffing is
//! today's honest shortcut (documented, not hidden).
//!
//! Runtime scalar `Param`s (from `AddScalar`/`MulScalar` regions) ARE
//! supported: `compile_one` maps the fused node's `Runtime { scalars }` to
//! [`OpParams::JitScalars`], and [`launch_scalar`] appends them as the
//! kernel's trailing `, float p0, float p1, …` args (the emitter's
//! `param_args` suffix — always `float`, after `long long n`, ascending),
//! narrowing each `f64` slot value to `f32` per that ABI. A param-COUNT
//! mismatch between the node and the kernel signature remains undetectable
//! from `SynthArtifact` alone (the contract's `op_params` metadata is the
//! eventual cross-check); the slot machinery keeps them aligned by
//! construction (one canonical pattern pre-order on both sides).

use std::sync::{Arc, OnceLock, RwLock};

use fuel_cuda_backend::{CudaDevice, CudaFunc, LaunchConfig, WrapErr};
use fuel_ir::{Error, Layout, Result};
use fuel_kernel_seam::{ArtifactKind, SynthArtifact};
use fuel_memory::{BackendStorage, Storage};

use crate::kernel::{KernelRef, OpParams};

/// Size of the process-global JIT-kernel slot bank. See the module docs'
/// "why a slot table" section for why this exists at all.
const MAX_JIT_SLOTS: usize = 64;

/// One claimed slot: the loaded+resolved `CudaFunc` plus its entry point
/// (carried only for diagnostics in launch-time error messages).
struct Slot {
    func: CudaFunc,
    entry_point: String,
}

fn slots() -> &'static RwLock<Vec<Option<Slot>>> {
    static SLOTS: OnceLock<RwLock<Vec<Option<Slot>>>> = OnceLock::new();
    SLOTS.get_or_init(|| RwLock::new((0..MAX_JIT_SLOTS).map(|_| None).collect()))
}

/// Load `art` as a CUDA module on `device`, resolve `art.link.symbol`, and
/// wrap it as a [`KernelRef`] — the `load_kernel` closure
/// [`crate::jit_adopt::adopt_from_response`] needs. See the module docs for
/// the scalar-ABI-only scope this implements.
///
/// # Errors
/// - `art.kind` isn't [`ArtifactKind::Ptx`] (Cubin loading would need
///   `Module::load_raw`, not wired here).
/// - `art.artifact` isn't valid UTF-8 PTX text.
/// - `art.link.symbol` doesn't look like a scalar-schedule kernel (see the
///   module docs' launch-marshaling scope) — an honest decline rather than a
///   silent wrong-ABI launch.
/// - the driver fails to load the module / resolve the symbol
///   ([`CudaDevice::get_or_load_custom_func`]'s errors).
/// - the slot bank ([`MAX_JIT_SLOTS`]) is full.
pub fn load_synth_kernel(art: &SynthArtifact, device: &CudaDevice) -> Result<KernelRef> {
    if !matches!(art.kind, ArtifactKind::Ptx) {
        return Err(Error::Msg(format!(
            "load_synth_kernel({}): unsupported artifact kind {:?} (only Ptx is loadable here; \
             Cubin loading would need Module::load_raw, not yet wired)",
            art.link.entry_point, art.kind,
        ))
        .bt());
    }
    // See the module docs' "launch-marshaling scope": only the scalar ABI's
    // launch shape is implemented, so anything else is an honest decline
    // rather than a silently wrong launch.
    if !art.link.symbol.ends_with("_scalar") {
        return Err(Error::Msg(format!(
            "load_synth_kernel({}): symbol '{}' isn't a recognized scalar-ABI kernel — \
             vectorized/strided synth kernels aren't supported by this loader yet",
            art.link.entry_point, art.link.symbol,
        ))
        .bt());
    }
    let ptx_src = std::str::from_utf8(&art.artifact).map_err(|e| {
        Error::Msg(format!(
            "load_synth_kernel({}): PTX artifact isn't valid UTF-8: {e}",
            art.link.entry_point,
        ))
        .bt()
    })?;
    let func = device.get_or_load_custom_func(&art.link.symbol, &art.link.entry_point, ptx_src)?;
    claim_slot(func, art.link.entry_point.clone())
}

/// Claim the next free slot in the bank and return its dispatcher.
fn claim_slot(func: CudaFunc, entry_point: String) -> Result<KernelRef> {
    let mut guard = slots().write().unwrap();
    let idx = guard.iter().position(Option::is_none).ok_or_else(|| {
        Error::Msg(format!(
            "load_synth_kernel({entry_point}): the {MAX_JIT_SLOTS}-slot JIT-kernel bank is full \
             — bump MAX_JIT_SLOTS + the dispatch_table! call in jit_cuda_load.rs",
        ))
        .bt()
    })?;
    guard[idx] = Some(Slot { func, entry_point });
    drop(guard);
    Ok(DISPATCH_TABLE[idx])
}

/// The output element count for the scalar-ABI launch: prefer the output
/// layout Fuel passed (the [`KernelRef`] contract's `layouts[inputs.len()]`);
/// fall back to the output storage's raw byte length / dtype size when no
/// layout was supplied (mirrors `fuel_cuda_backend::baracuda::elementwise`'s
/// same fallback for binding-table callers that haven't migrated to
/// layout-passing).
fn output_numel(out: &Storage, layouts: &[Layout], n_inputs: usize) -> usize {
    match layouts.get(n_inputs) {
        Some(l) => l.shape().elem_count(),
        None => {
            let elem = out.dtype.size_in_bytes().max(1);
            out.inner.len_bytes() / elem
        }
    }
}

fn cuda_storage<'a>(s: &'a Storage, entry_point: &str) -> Result<&'a fuel_cuda_backend::CudaStorageBytes> {
    match &s.inner {
        BackendStorage::Cuda(c) => Ok(c),
        #[allow(unreachable_patterns)]
        _ => Err(Error::Msg(format!(
            "load_synth_kernel({entry_point}): called with a non-CUDA storage",
        ))
        .bt()),
    }
}

fn cuda_storage_mut<'a>(
    s: &'a mut Storage,
    entry_point: &str,
) -> Result<&'a mut fuel_cuda_backend::CudaStorageBytes> {
    match &mut s.inner {
        BackendStorage::Cuda(c) => Ok(c),
        #[allow(unreachable_patterns)]
        _ => Err(Error::Msg(format!(
            "load_synth_kernel({entry_point}): called with a non-CUDA storage",
        ))
        .bt()),
    }
}

/// The scalar-ABI launch shared by every slot dispatcher: a pointer arg for
/// each input (in order), the output pointer, then the output element count
/// as `long long` — exactly the parameter list `baracuda-kernelgen`'s
/// `emit_scalar` builds. Modeled on `CudaUgIOp1::fwd`
/// (`fuel-cuda-backend/src/ug.rs`) — the crate's other "launch a `CudaFunc`
/// against Fuel storage" site — adapted from one typed `CudaSlice<T>` arg to N
/// byte-erased `CudaStorageBytes` buffers: a synth kernel is dtype-generic at
/// this Rust call boundary (the PTX signature carries the concrete C type;
/// CUDA kernel params are addresses regardless), so the launch only ever
/// pushes the raw device pointer via `&DeviceBuffer<u8>`'s `KernelArg` impl.
fn launch_scalar(
    func: &CudaFunc,
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
    params: &OpParams,
    entry_point: &str,
) -> Result<()> {
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "load_synth_kernel({entry_point}): expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guards = inputs
        .iter()
        .map(|arc| {
            arc.read().map_err(|_| {
                Error::Msg(format!(
                    "load_synth_kernel({entry_point}): input storage RwLock poisoned",
                ))
                .bt()
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut out_guard = outputs[0].write().map_err(|_| {
        Error::Msg(format!(
            "load_synth_kernel({entry_point}): output storage RwLock poisoned",
        ))
        .bt()
    })?;

    let numel = output_numel(&out_guard, layouts, inputs.len());

    let mut builder = func.builder();
    for g in &in_guards {
        let cuda = cuda_storage(g, entry_point)?;
        builder.arg(cuda.buffer());
    }
    let out_cuda = cuda_storage_mut(&mut out_guard, entry_point)?;
    builder.arg(out_cuda.buffer());
    let n: i64 = numel as i64;
    builder.arg(&n);
    // The kernel's runtime scalar params: the emitter's signature suffix is
    // `, float p0, float p1, …` — ALWAYS `float`, regardless of the kernel's
    // element dtype, appended after `long long n`, indices ascending (the
    // synthesizer's `param_args`). `JitScalars` carries the live values in
    // that same canonical (pattern pre-order) order; narrow each to f32 here.
    let p32: Vec<f32> = match params {
        OpParams::JitScalars { scalars } => scalars.iter().map(|&v| v as f32).collect(),
        _ => Vec::new(),
    };
    for p in &p32 {
        builder.arg(p);
    }

    let cfg = LaunchConfig::for_num_elems(numel as u32);
    // SAFETY: pointer args are live device buffers held alive by `in_guards` /
    // `out_guard` for the duration of this call; `p32` outlives the launch.
    // `n` matches the emitted kernel's `long long n` parameter (the scalar
    // ABI's element count, pinned by the `_scalar`-suffix gate in
    // `load_synth_kernel`). Argument count/order matches `emit_scalar`'s
    // signature exactly: one pointer per input in order, the output pointer,
    // `n`, then the `float p{i}` params in ascending order (empty for a
    // param-free kernel).
    unsafe { builder.launch(cfg) }.w()?;
    Ok(())
}

/// One dispatcher per slot — a genuinely distinct `fn` item per `N` (via
/// monomorphization), so a bare-`fn`-pointer `KernelRef` can carry a
/// runtime-loaded kernel's identity without a captured environment. See the
/// module docs' "why a slot table" section.
fn dispatch_slot<const N: usize>(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    let guard = slots().read().unwrap();
    let slot = guard[N].as_ref().ok_or_else(|| {
        Error::Msg(format!(
            "load_synth_kernel: slot {N} dispatched with no kernel claimed \
             (a KernelRef outliving its slot bank reset?)",
        ))
        .bt()
    })?;
    launch_scalar(&slot.func, inputs, outputs, layouts, params, &slot.entry_point)
}

macro_rules! dispatch_table {
    ($($n:literal),* $(,)?) => {
        [$(dispatch_slot::<$n> as KernelRef),*]
    };
}

static DISPATCH_TABLE: [KernelRef; MAX_JIT_SLOTS] = dispatch_table!(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
);

// No unit tests in this module: every code path that runs before a live
// device call (the `ArtifactKind`/`_scalar`-symbol declines) still requires a
// `&CudaDevice` argument, and this crate's live-CUDA convention (see
// `tests/baracuda_*_live.rs`) is that anything constructing a `CudaDevice`
// lives in an `#[ignore]`'d integration test, never a always-run unit test —
// see `tests/jit_synth_kernel_live.rs` for the on-device coverage.
