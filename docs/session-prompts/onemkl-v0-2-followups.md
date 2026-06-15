# onemkl v0.2 followups — parked until a concrete consumer lands

**Status:** v0.2.0 landed 2026-05-15. Service-module integration
(`ThreadCountGuard`, `IsaLevel`, `version_string`, `cpu_frequency_*`,
`AlignedBuffer`) shipped into `fuel-mkl-cpu-backend` in the same window.
The three v0.2 categories below are deliberately deferred — each needs
new Fuel-side seams that don't exist yet.

> Reconciled 2026-06-15 against the 2026-06-14 redirection + current git:
> the CUDA Tier-2 / baracuda-pin gates are now largely MET, so the real
> remaining gate is "a concrete consumer lands," not the CUDA work being
> in flight.

Pick up when a concrete consumer lands that needs one of these. The
deferral was originally gated behind in-flight CUDA work; that gate is
now largely met (CUDA Tier 2 — Rope / RmsNorm / LayerNorm / FlashAttn —
has landed against the current baracuda pin, alpha.67), so the live
trigger is now a real consumer rather than the CUDA work itself:

- the build-time RNG seam (see §1), or
- a sparse op family (see §3), or
- a linear-solve need (see §2).

## 1. RNG distributions

v0.2 added Gumbel / Laplace / Rayleigh / ChiSquare / GaussianMV /
TruncatedNormal / Dirichlet / Multinomial / Categorical / Geometric /
Hypergeometric / NegBinomial / quasi-random.

Why parked: Fuel has no backend-level RNG seam. Sampling and weight
init currently live in `fuel-transformers` and `fuel-nn` against the
`rand` crate. Wiring MKL's RNG only matters once we design:

- Where does the `Generator` live? Per-backend? Per-device? Per-graph?
- How does it interact with autograd (no gradient through sampling but
  the sample value participates in the tape).
- How do CUDA (cuRAND), Vulkan (no native RNG), AOCL participate?

This is a Fuel-graph design problem, not a backend-update problem.
The most natural first consumer is the Gumbel-max-trick at the end of
LLM generation — once that lands as a graph op, plumb MKL's
`Gumbel` distribution into the CPU dispatch arm.

## 2. Mixed-precision iterative-refinement LAPACK

v0.2 added `iter_refine_{gesv,posv}_{f64,c64}` — solve `Ax=b` in f32
with f64 residual refinement.

Why parked: no consumer in Fuel's current op surface. We do **no**
linear solves anywhere in the matmul / conv / reduce / softmax /
norm / attention path. Useful if Fuel ever grows into scientific
compute; not interesting for LLM inference. Revisit only on a
concrete ask.

## 3. Sparse SYRK / SYPR + summary-stats preprocessing

v0.2 added sparse symmetric rank-k update / packed inverse, plus
streaming quantiles, Tukey robust covariance, BACON outliers, MI
imputation.

Why parked: Fuel has no sparse tensor path at all. Adding one is
larger than this update — sparse format design (CSR? COO? blocked
CSR for attention?), how it interacts with the rest of the op
surface, storage variants, layout, etc. Summary stats sit even
further afield: they're preprocessing / observability, not graph
kernels.

Revisit if/when a sparse op family (sparse attention, MoE routing
table) becomes a Fuel priority. The summary-stats helpers might
also surface in the empirical-judge profiler later — quantiles on
latency distributions instead of single percentile points.

## Done in this update

- Workspace `onemkl` pin bumped 0.1 → 0.2.
- `fuel-mkl-cpu-backend::probe::enumerate_devices` now reports
  `version_string()` + current/max CPU GHz + MKL's `max_threads()`
  instead of `std::env::consts::ARCH`.
- `MklBackend::try_new_with_threads(n)` added.
- `IsaLevel` + `ThreadCountGuard` re-exported from
  `fuel_mkl_cpu_backend`.
- `pin_isa(level)` free function added (wraps `enable_instructions`).
- `conv2d` im2col scratch now allocated via `AlignedBuffer<f32>` at
  64-byte alignment instead of `vec![0.0_f32; …]`.
