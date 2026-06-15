//! # fuel-aocl-cpu-backend
//!
//! AMD AOCL-BLAS-backed CPU matmul/conv2d kernels for the fuel
//! lazy-graph layer.
//!
//! This is the first per-vendor CPU backend (Phase 7b spike). It
//! routes the matmul fast path through `aocl_blas::gemm` instead of the
//! cross-vendor `gemm` crate. The kernels register as sibling
//! alternatives on the unified binding table at `BackendId::Cpu` with
//! the `"aocl"` source tag (see [`register_aocl_cpu_kernels`]); every
//! other op is served by fuel-cpu-backend's portable kernels.
//!
//! On Zen-class AMD CPUs `aocl_blas::gemm` calls into AOCL-BLAS (BLIS),
//! which exploits per-microarch tuning that the portable `gemm` crate
//! can't match. The Judge profiles both at startup and the dispatch
//! table picks per `(op, dtype, size_class)`.
//!
//! The legacy `GraphBackend for AoclBackend` executor adapter was
//! retired in executor-unification Session 7; the binding-table
//! registration is now the sole production surface.
//!
//! # Availability gate
//!
//! [`probe_aocl_loadable`] runs a 2×2 sgemm; if `libaocl_blas` doesn't
//! load on this machine it returns `Err` and the caller skips
//! registering the AOCL kernels. Backends in Fuel's design own their
//! own availability check — there is no HardwareQuery layer gating them.

pub mod binding_table;
mod dll_path;
// `probe` module retired 2026-06-08: AOCL is no longer a separate
// backend (no `BackendId::Aocl`); its kernels register as siblings
// of fuel-cpu-backend at `BackendId::Cpu` with `kernel_source:
// "aocl"`. The runtime `probe_aocl_loadable()` check below stays
// — it's the gate that determines whether AOCL kernels register
// into the binding table at startup.

pub use binding_table::register_aocl_cpu_kernels;

use fuel_core_types::Result;

/// Probe `libaocl_blas` with a 2×2 sgemm. Returns `Ok` on a successful
/// call, `Err` if the library can't be loaded (or any deeper failure
/// surfaces). Public so callers that just want a "is AOCL available?"
/// signal can use it without constructing the backend.
///
/// On Windows, this best-effort extends `PATH` with the standard
/// AOCL BLIS install directory if it isn't already there — see
/// [`dll_path`] for the discovery order. The AMD installer doesn't
/// add the BLIS dir to system PATH, so without this, every Windows
/// run would need a manual `set PATH=...` before invocation.
pub fn probe_aocl_loadable() -> Result<()> {
    dll_path::ensure_loadable();
    use aocl_types::Trans;
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [1.0_f32, 0.0, 0.0, 1.0];
    let mut c = [0.0_f32; 4];
    aocl_blas::gemm(
        Trans::No, Trans::No,
        2, 2, 2,
        1.0_f32,
        &a, &b,
        0.0_f32,
        &mut c,
    ).map_err(|e| fuel_core_types::Error::Msg(
        format!("AOCL probe gemm failed: {e}")
    ))?;
    if c != [1.0, 2.0, 3.0, 4.0] {
        return Err(fuel_core_types::Error::Msg(format!(
            "AOCL probe gemm produced wrong result: {c:?} != [1, 2, 3, 4]"
        )));
    }
    Ok(())
}
