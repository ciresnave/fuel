//! # fuel-mkl-cpu-backend
//!
//! Intel oneMKL-backed CPU matmul/conv2d kernels for the fuel lazy-graph
//! layer.
//!
//! Mirrors `fuel-aocl-cpu-backend`'s shape but routes the matmul fast
//! path through `onemkl::blas::level3::gemm` instead of AOCL-BLAS or
//! the cross-vendor `gemm` crate. The kernels register as sibling
//! alternatives on the unified binding table at `BackendId::Cpu` with
//! the `"mkl"` source tag (see [`register_mkl_cpu_kernels`]); every
//! other op is served by fuel-cpu-backend's portable kernels.
//!
//! On Intel CPUs (and many AMD CPUs too — MKL detects vendor at runtime
//! and dispatches accordingly), this should be the matmul winner. The
//! Judge profiles MKL alongside any other registered CPU kernel source
//! and the dispatch table picks per `(op, dtype, size_class)`.
//!
//! The legacy `GraphBackend for MklBackend` executor adapter was retired
//! in executor-unification Session 7; the binding-table registration is
//! now the sole production surface.
//!
//! # Availability gate
//!
//! [`probe_mkl_loadable`] runs a 2×2 sgemm; if `mkl_rt` doesn't load on
//! this machine it returns `Err` and the caller skips registering the
//! MKL kernels. Backends in Fuel's design own their own availability
//! check — there is no HardwareQuery layer gating them.

pub mod binding_table;
mod dll_path;
// `probe` module retired 2026-06-08: MKL is no longer a separate
// backend (no `BackendId::Mkl`); its kernels register as siblings
// of fuel-cpu-backend at `BackendId::Cpu` with `kernel_source:
// "mkl"`. The runtime `probe_mkl_loadable()` check below stays
// — it's the gate that determines whether MKL kernels register
// into the binding table at startup.

pub use binding_table::register_mkl_cpu_kernels;

use fuel_core_types::Result;

// onemkl v0.2 service-module surface re-exported so callers can
// reach for these without taking a direct `onemkl` dependency.
//
// * `IsaLevel`        — pin MKL's vector-ISA dispatch tier (see
//                       [`pin_isa`]). Useful for benchmarking and for
//                       working around a misbehaving fallback path.
// * `ThreadCountGuard`— RAII scope guard that overrides MKL's local
//                       thread count and restores it on drop. Lets a
//                       caller say "this matmul should use 1 thread"
//                       without disturbing the rest of the process.
pub use onemkl::service::{IsaLevel, ThreadCountGuard};

/// Pin oneMKL's vector-ISA dispatch tier to `level`.
///
/// Must be called **before** any MKL routine (including
/// [`probe_mkl_loadable`] and `register_mkl_cpu_kernels`) —
/// MKL caches its dispatched code path on first use. Returns `Err`
/// if the CPU does not support the requested level.
///
/// On Windows this best-effort extends `PATH` so `mkl_rt` can load
/// even when `setvars.bat` hasn't been run.
pub fn pin_isa(level: IsaLevel) -> Result<()> {
    dll_path::ensure_loadable();
    onemkl::service::enable_instructions(level)
        .map_err(|e| fuel_core_types::Error::Msg(format!("MKL_Enable_Instructions: {e}")))
}

/// Probe `mkl_rt` (or whichever oneMKL runtime resolves) with a 2×2
/// sgemm. Returns `Ok` on a successful call producing the right
/// answer, `Err` if the library can't be loaded or the probe gemm
/// returns wrong values. Public so callers that just want a
/// "is oneMKL available?" signal can use it without constructing the
/// backend.
///
/// On Windows, this best-effort extends `PATH` with the standard
/// oneMKL bin directory if it isn't already there — see [`dll_path`]
/// for the discovery order. Without this, a default-launched
/// `cargo run --features onemkl` would crash with
/// `STATUS_DLL_NOT_FOUND` unless the user pre-runs `setvars.bat`.
pub fn probe_mkl_loadable() -> Result<()> {
    dll_path::ensure_loadable();
    use onemkl::enums::{Layout as MklLayout, Transpose};
    use onemkl::matrix::{MatrixMut, MatrixRef};

    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [1.0_f32, 0.0, 0.0, 1.0];
    let mut c = [0.0_f32; 4];

    let a_ref = MatrixRef::new(&a, 2, 2, MklLayout::RowMajor)
        .map_err(|e| fuel_core_types::Error::Msg(format!("MKL probe MatrixRef::new(a) failed: {e}")))?;
    let b_ref = MatrixRef::new(&b, 2, 2, MklLayout::RowMajor)
        .map_err(|e| fuel_core_types::Error::Msg(format!("MKL probe MatrixRef::new(b) failed: {e}")))?;
    let mut c_mut = MatrixMut::new(&mut c, 2, 2, MklLayout::RowMajor)
        .map_err(|e| fuel_core_types::Error::Msg(format!("MKL probe MatrixMut::new(c) failed: {e}")))?;

    onemkl::blas::level3::gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        1.0_f32,
        &a_ref,
        &b_ref,
        0.0_f32,
        &mut c_mut,
    ).map_err(|e| fuel_core_types::Error::Msg(format!("MKL probe gemm failed: {e}")))?;

    if c != [1.0, 2.0, 3.0, 4.0] {
        return Err(fuel_core_types::Error::Msg(format!(
            "MKL probe gemm produced wrong result: {c:?} != [1, 2, 3, 4]"
        )));
    }
    Ok(())
}
