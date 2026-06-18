# Kernel inventory — `fuel-mkl-cpu-backend` + `fuel-aocl-cpu-backend`

> Scope: kernels Fuel **itself** provides in these two BLAS-backed CPU
> backend crates. Both crates are thin sibling wrappers: they register
> a small set of CPU-side kernel wrappers as *sibling alternatives* on
> the unified `KernelBindingTable` at `BackendId::Cpu`, tagged with the
> `kernel_source` `"mkl"` / `"aocl"` respectively. The actual GEMM math
> is delegated to `onemkl::blas::level3::gemm` (MKL) and
> `aocl_blas::gemm` (AOCL); conv2d is delegated to the shared
> `fuel_conv::conv2d_via_gemm` im2col+gemm driver. Everything not listed
> here is served by `fuel-cpu-backend`'s portable kernels.
>
> Source crates:
> - `fuel-mkl-cpu-backend/src/binding_table.rs`
> - `fuel-mkl-cpu-backend/src/lib.rs`
> - `fuel-aocl-cpu-backend/src/binding_table.rs`
> - `fuel-aocl-cpu-backend/src/lib.rs`
>
> Shared helper (output/layout semantics referenced below):
> `fuel-conv/src/lib.rs` (`ConvShape`, `conv2d_via_gemm`).

## Summary

Each crate provides **2 distinct kernels** (4 binding-table
registrations because Conv2D registers under both a 3-operand and a
4-operand dtype key but is served by one wrapper). The two crates are
near-identical structurally; MKL and AOCL are listed as separate kernel
entries because they are genuinely different implementations (different
BLAS libraries, different probe/availability gates, different
precision/determinism characteristics, separate registration fns), not
dtype-monomorphized variants of one kernel.

| Crate | Kernel | Op | DTypes | Binding keys |
|---|---|---|---|---|
| mkl | `matmul_f32_mkl_cpu_wrapper` | MatMul | F32 | `(MatMul,[F32,F32,F32],Cpu)` |
| mkl | `conv2d_f32_mkl_cpu_wrapper` | Conv2D | F32 | `(Conv2D,[F32×3],Cpu)` + `(Conv2D,[F32×4],Cpu)` |
| aocl | `matmul_f32_aocl_cpu_wrapper` | MatMul | F32 | `(MatMul,[F32,F32,F32],Cpu)` |
| aocl | `conv2d_f32_aocl_cpu_wrapper` | Conv2D | F32 | `(Conv2D,[F32×3],Cpu)` + `(Conv2D,[F32×4],Cpu)` |

Not kernels (registration / availability plumbing, listed for
completeness, excluded from the per-kernel contract set):
`register_mkl_cpu_kernels` / `register_aocl_cpu_kernels` (binding-table
wiring); `probe_mkl_loadable` / `probe_aocl_loadable` (2×2 sgemm
availability gate); `pin_isa` (MKL-only ISA pin); `dll_path::ensure_loadable`
(Windows PATH discovery); re-exports `IsaLevel`, `ThreadCountGuard` (MKL).

---

## 1. MatMul (MKL) — `matmul_f32_mkl_cpu_wrapper`

- **Source:** `fuel-mkl-cpu-backend/src/binding_table.rs:109` (wrapper),
  inner `matmul_f32_mkl_bytes` at `:168`. Registered at `:66`.
- **Op kind:** `OpKind::MatMul`. Batched row-major `[m,k] @ [k,n] -> [m,n]`.
- **DTypes:** F32 only. Binding key `[F32, F32, F32]` (lhs, rhs, out).
- **Backend / source tag:** `BackendId::Cpu`, `kernel_source = "mkl"`.
- **Input layout handling:** **Contiguous-only, zero-offset.** `_layouts`
  is *ignored* (param prefixed `_`, never read). Inputs are assumed
  contiguous (executor auto-Contiguize pass guarantees this). The wrapper
  validates exact packed byte counts:
  `need_lhs = lhs_batch_count * m*k * 4`, `need_rhs = rhs_batch_count *
  k*n * 4`, `need_out = lhs_batch_count * m*n * 4` (`:212-238`); any
  stride/offset/broadcast layout would mismatch these and error. No
  `StridedIndex`, no `is_contiguous()` branch — pure flat-slice math via
  `as_slice()`/`as_slice_mut()` (`:239-241`).
- **Op params (`OpParams::Matmul`):** `lhs_batch_dims: Vec<usize>`,
  `rhs_batch_dims: Vec<usize>`, `m`, `n`, `k` (`:129-143`).
- **Batch semantics:** Batch ranks must match (`:181`). Per-axis the dims
  must be equal **or** GQA-divisible (`lhs>rhs && lhs%rhs==0`, n_rep =
  lhs/rhs); anything else errors (`:190-205`). Iterates `lhs_batch_count`
  slots; maps each to an rhs slot via per-axis `lhs_idx/n_rep` row-major
  unravel/ravel (`:243-265`). One `onemkl::blas::level3::gemm` call per
  batch slot, `NoTrans/NoTrans`, alpha=1, beta=0 (`:276-285`).
- **Output behavior:** dtype F32 (same as inputs). Shape =
  `lhs_batch_dims + [m, n]` (output batch follows lhs, the GQA "larger"
  side). Output written contiguous row-major; **caller pre-allocates**
  the output storage (no alloc in kernel). beta=0 ⇒ full overwrite, no
  read of prior output, **no accumulation, no aliasing/in-place**
  (separate out storage).
- **Precision:** `MKL_PRECISION` (`:49-58`): `bit_stable_on_same_hardware
  = true` (run-to-run deterministic on fixed CPU + thread count; set
  `MKL_CBWR` for cross-machine repro). `max_ulp/relative/absolute = None`
  (uncalibrated — placeholder pending step-8 framework). NOT bit-equal to
  the scalar reference (MKL's blocked accumulation order differs). Parity
  test asserts only relative error `< 1e-4`.
- **Cost model:** `unknown_cost` (no static cost; Judge profiles it).
- **Caps:** `KernelCaps::empty()`.

## 2. Conv2D (MKL) — `conv2d_f32_mkl_cpu_wrapper`

- **Source:** `fuel-mkl-cpu-backend/src/binding_table.rs:299` (wrapper),
  inner `run_mkl_conv2d_via_gemm` at `:411`. Registered at `:79`
  (3-operand) and `:89` (4-operand, with bias).
- **Op kind:** `OpKind::Conv2D`. NCHW im2col + per-(batch,group) sgemm.
- **DTypes:** F32 only. Two binding keys: `[F32,F32,F32]` (x, w, out — no
  bias) and `[F32,F32,F32,F32]` (x, w, bias, out). One wrapper serves
  both; distinguishes by `inputs.len()` (2 vs 3) at `:305`, `:339`.
- **Backend / source tag:** `BackendId::Cpu`, `kernel_source = "mkl"`.
- **Input layout handling:** **Contiguous-only, zero-offset, NCHW.**
  `_layouts` ignored. Operands consumed as flat slices via `as_slice()`
  (`:380-386`); im2col + gemm assume tightly-packed NCHW x / OIHW weight /
  per-channel bias. No stride/offset/broadcast support.
- **Op params (`OpParams::Conv2D`):** `x_shape: [N,Cin,H,W]`,
  `w_shape: [Cout,Cin,kH,kW]`, `out_shape: [N,Cout,Hout,Wout]`,
  `stride:(usize,usize)`, `padding:(usize,usize)`,
  `dilation:(usize,usize)`, `groups:usize` (`:319-335`). Asymmetric
  stride/padding supported; groups (incl. depthwise) supported.
- **Fallback paths (important):** This wrapper does **not** handle all
  conv shapes itself. It **delegates to the scalar
  `fuel_cpu_backend::byte_kernels::conv2d_f32`** when (a) `dilation !=
  (1,1)` (`:355-360`) — `ConvShape` carries no dilation field; or (b)
  `ConvShape::validate()` fails (`:373-378`). Only the (1,1)-dilation,
  valid-shape case runs through MKL's im2col+gemm.
- **MKL fast path:** builds `fuel_conv::ConvShape` (`:361-372`), allocates
  an im2col scratch buffer of `s.im2col_len()` f32 via
  `onemkl::service::AlignedBuffer` (64-byte aligned, AVX-512 cache line),
  **falling back to a plain `vec![0.0; ...]` if MKL_malloc fails — never
  panics** (`:392-404`). Then `run_mkl_conv2d_via_gemm` drives
  `fuel_conv::conv2d_via_gemm`: one `onemkl gemm` per (batch,group) with
  `m=cout_per_g, n=Hout*Wout, k=cin_per_g*kH*kW`, `NoTrans/NoTrans`,
  alpha=1, beta=0 (`:421-436`).
- **Output behavior:** dtype F32. Shape `[N, Cout, Hout, Wout]`
  (`Hout/Wout` from `ConvShape::h_out()/w_out()`). Output written
  contiguous row-major NCHW; **caller pre-allocates.** Per-(batch,group)
  GEMM does `c = a@b` (beta=0, no accumulate). **Bias add (if present)
  is done after gemm by `conv2d_via_gemm` itself**, per-output-channel
  broadcast over the spatial plane (`fuel-conv/src/lib.rs:343-355`), not
  by the BLAS call. No in-place/aliasing on inputs.
- **Precision notes:** `MKL_PRECISION` (same struct as MatMul):
  deterministic on fixed hardware/threads; not bit-equal to scalar
  conv (blocked accumulation + im2col reordering). Parity test asserts
  rel err `< 1e-4`. The fallback (scalar) path inherits scalar precision.
  Inner gemm closure uses `.expect()` on `MatrixRef/MatrixMut::new` and
  the gemm call (`:425-434`) — a **panic on production path** if those
  fail (CLAUDE.md flags `.expect()` on production paths as a violation;
  note for the contract).
- **Cost model:** `unknown_cost`. **Caps:** `KernelCaps::empty()`.

## 3. MatMul (AOCL) — `matmul_f32_aocl_cpu_wrapper`

- **Source:** `fuel-aocl-cpu-backend/src/binding_table.rs:109` (wrapper),
  inner `matmul_f32_aocl_bytes` at `:168`. Registered at `:67`.
- **Op kind / DTypes / backend tag:** identical to the MKL MatMul kernel
  but `kernel_source = "aocl"`, OpKind `MatMul`, F32, `BackendId::Cpu`.
- **Input layout handling:** **Contiguous-only, zero-offset.** Same as
  MKL: `_layouts` ignored; exact packed-byte-count validation
  (`:211-237`); flat `as_slice()` access (`:238-240`). No strided/offset/
  broadcast support.
- **Op params:** `OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m,
  n, k }` (`:129-143`).
- **Batch semantics:** identical to MKL — ranks must match, per-axis
  equal-or-GQA-divisible, row-major unravel/ravel batch mapping
  (`:180-265`). One `aocl_blas::gemm(Trans::No, Trans::No, m, n, k,
  1.0, a, b, 0.0, c)` per batch slot (`:266-278`). Note AOCL's gemm takes
  `m,n,k` directly (no MatrixRef wrapper, unlike MKL).
- **Output behavior:** dtype F32; shape `lhs_batch_dims + [m,n]`;
  contiguous row-major; caller-allocated; beta=0 full overwrite, no
  accumulate/aliasing.
- **Precision:** `AOCL_PRECISION` (`:48-56`): `bit_stable_on_same_hardware
  = true` (BLIS deterministic on fixed CPU + thread count). `max_*` =
  None (uncalibrated). Not bit-equal to scalar (BLIS blocked
  accumulation). Parity test: rel err `< 1e-4`.
- **Cost model:** `unknown_cost`. **Caps:** `KernelCaps::empty()`.

## 4. Conv2D (AOCL) — `conv2d_f32_aocl_cpu_wrapper`

- **Source:** `fuel-aocl-cpu-backend/src/binding_table.rs:292`.
  Registered at `:80` (3-operand) and `:90` (4-operand, with bias).
- **Op kind / DTypes / backend tag / keys:** identical to MKL Conv2D but
  `kernel_source = "aocl"`. Two keys (`[F32×3]` no-bias, `[F32×4]`
  with-bias); one wrapper, dispatched by `inputs.len()` (`:298`, `:332`).
- **Input layout handling:** **Contiguous-only, zero-offset, NCHW.**
  `_layouts` ignored; flat `as_slice()` access (`:373-379`). No
  strided/offset/broadcast.
- **Op params:** `OpParams::Conv2D { x_shape, w_shape, out_shape, stride,
  padding, dilation, groups }` (`:312-328`). Asymmetric stride/pad +
  groups/depthwise supported on the fast path.
- **Fallback paths:** delegates to scalar
  `fuel_cpu_backend::byte_kernels::conv2d_f32` when `dilation != (1,1)`
  (`:348-353`) or `ConvShape::validate()` fails (`:366-371`).
- **AOCL fast path:** builds `ConvShape` (`:354-365`); allocates im2col
  scratch as a plain `vec![0.0; s.im2col_len()]` (`:380` — **no aligned
  buffer**, unlike MKL); drives `fuel_conv::conv2d_via_gemm` with an
  `aocl_blas::gemm(Trans::No, Trans::No, m, n, k, 1.0, a, b, 0.0, c)`
  closure per (batch,group) (`:382-394`).
- **Output behavior:** dtype F32; shape `[N,Cout,Hout,Wout]`; contiguous
  row-major NCHW; caller-allocated; beta=0 no-accumulate. Bias add (if
  present) done post-gemm by `conv2d_via_gemm` per-channel (see
  `fuel-conv/src/lib.rs:343-355`). No in-place/aliasing.
- **Precision notes:** `AOCL_PRECISION`: deterministic on fixed
  hardware/threads; not bit-equal to scalar. Parity test rel err
  `< 1e-4`. The gemm closure uses `.expect("aocl_blas::gemm in
  conv2d_via_gemm")` (`:392`) — **panic on production path** if the BLAS
  call errors (CLAUDE.md never-panic violation; note for the contract).
- **Cost model:** `unknown_cost`. **Caps:** `KernelCaps::empty()`.

---

## Cross-cutting contract notes

- **No strided/broadcast/offset support anywhere.** All four kernels rely
  on the executor's auto-Contiguize pass and ignore the `layouts`
  argument entirely. The only "layout" check is an exact contiguous
  byte-count validation in matmul; conv relies on shape params + flat
  slices. A per-kernel contract should record these as **contiguous-only,
  zero-offset** with no fast-path for already-strided input.
- **F32-only.** No f16/bf16/f64/int variants exist in either crate (the
  MKL header comment notes "Int GEMM follows in its own commit" — not yet
  present). These are not dtype-monomorphized families; they are single
  F32 kernels.
- **Conv2D is partial.** Both conv wrappers silently fall back to the
  scalar CPU conv for dilation≠(1,1) and for any shape that fails
  `ConvShape::validate`. The BLAS path only owns the (1,1)-dilation
  valid-shape case. A contract must capture this fallback boundary.
- **Output is always caller-allocated, fully overwritten (beta=0), never
  aliased with inputs, never accumulated.** Bias for conv is applied by
  the shared `conv2d_via_gemm` driver, not the BLAS gemm.
- **Determinism vs bit-equality.** Both backends claim
  `bit_stable_on_same_hardware` (run-to-run determinism on fixed CPU +
  thread count) but explicitly are **not** bit-equal to the scalar
  reference. ULP/relative/absolute bounds are uncalibrated (`None`).
- **Panic risk on production paths:** the conv gemm closures
  (`run_mkl_conv2d_via_gemm` `.expect(...)` and the AOCL inline
  `.expect(...)`) can panic if the BLAS call or matrix-view construction
  fails. Flagged against the CLAUDE.md never-panic rule.
- **Availability is self-gated** by `probe_*_loadable` (a 2×2 sgemm) plus
  Windows DLL-path discovery in `dll_path::ensure_loadable`; kernels are
  only registered after a successful probe. These are not kernels.
