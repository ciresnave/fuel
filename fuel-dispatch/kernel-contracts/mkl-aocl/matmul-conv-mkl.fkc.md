---
fkc_version: 1
provider:
  name: fuel-mkl-cpu-backend
  backend: Cpu                                  # maps to BackendId::Cpu
  kernel_source: "mkl"                          # the BindingEntry.kernel_source tag
  link_registry: fuel_mkl_cpu_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map; every entry_point below roots in THIS crate
  revision_base: "git:f41137b4"                # provider build id, folded into kernel_revision_hash
---

# fuel-mkl-cpu-backend — kernel contracts (matmul family)

This crate registers a small set of Intel-MKL-backed kernel wrappers on the unified
`KernelBindingTable` at `BackendId::Cpu`, tagged `kernel_source = "mkl"`. They are *sibling
alternatives* to the AOCL/BLIS wrappers (`matmul-conv-aocl.fkc.md`) and to `fuel-cpu-backend`'s
portable kernels at the same `Cpu` key. The GEMM math is delegated to
`onemkl::blas::level3::gemm`; Conv2D is delegated to the shared `fuel_conv::conv2d_via_gemm`
im2col+gemm driver. Everything not contracted here is served by `fuel-cpu-backend`'s portable
kernels.

> **Split rationale (link-registry resolution).** MKL and AOCL ship as two **separate provider
> crates** with two **separate link registries** — `fuel_mkl_cpu_backend::fkc::ENTRY_POINTS` and
> `fuel_aocl_cpu_backend::fkc::ENTRY_POINTS`. An `entry_point` is resolved against **its own
> file's** `link_registry` (§12.6), so an MKL `entry_point` rooted at `fuel_mkl_cpu_backend::*`
> can only resolve against the MKL registry. The two providers therefore live in two bundle
> files (this one + `matmul-conv-aocl.fkc.md`), each with its own front-matter (provider, backend,
> kernel_source, link_registry). They still register as sibling alternatives at the same `Cpu`
> key — the binding table is shared, the *contract files* are per-provider.

Both kernels are **F32-only**, **contiguous-only / zero-offset** (they ignore the `layouts`
argument entirely and rely on the executor's auto-Contiguize pass), and produce a
**caller-pre-allocated, fully-overwritten (beta=0), non-aliased** output. Cost is empirically
profiled by the Judge (`unknown_cost` as built); every cost block here is marked
`provenance: judge_measured` and carries only an honest derivable FLOPs/bandwidth formula *hint*
for the Judge to refine — no fabricated coefficients, and `overhead_ns: ~` (the absolute launch
constant is the Judge's to measure, never authored).

---

## matmul_f32_mkl_cpu_wrapper  (batched row-major F32 GEMM via Intel MKL)

One-line: Batched row-major `[m,k] @ [k,n] -> [m,n]` F32 GEMM delegated to `onemkl` Level-3 sgemm; contiguous-only.

Batched row-major single-precision matrix multiply. For each batch slot the wrapper issues one
`onemkl::blas::level3::gemm` call with `NoTrans/NoTrans`, `alpha = 1.0`, `beta = 0.0`
(`binding_table.rs:276-285`). Inputs are consumed as flat, tightly-packed slices via
`as_slice()`/`as_slice_mut()` (`:239-241`); the `_layouts` argument is **ignored** (prefixed `_`,
never read) — the kernel assumes contiguous, zero-offset operands, which the executor's
auto-Contiguize pass guarantees. The wrapper validates exact packed byte counts up front
(`need_lhs = lhs_batch_count * m*k * 4`, `need_rhs = rhs_batch_count * k*n * 4`,
`need_out = lhs_batch_count * m*n * 4`, `:212-238`); any stride/offset/broadcast layout mismatches
these and errors, so there is no `StridedIndex` path. Batch ranks must match (`:181`); per-axis the
dims must be equal **or** GQA-divisible (`lhs > rhs && lhs % rhs == 0`, `n_rep = lhs/rhs`),
anything else errors (`:190-205`). The output batch follows the lhs ("larger" GQA side); each lhs
slot maps to an rhs slot via per-axis `lhs_idx / n_rep` row-major unravel/ravel (`:243-265`).

**Numerics/perf.** Math is MKL's blocked-accumulation sgemm: deterministic on fixed CPU + thread
count, but **not bit-equal** to the scalar reference (different accumulation order); the parity test
asserts only relative error `< 1e-4`. ULP/relative/absolute bounds are uncalibrated as built.
`beta = 0` means a full overwrite with no read of prior output — no accumulation, no in-place,
separate output storage. Cross-machine reproducibility requires `MKL_CBWR`. Availability is
self-gated by a 2×2 sgemm probe (`probe_mkl_loadable`) plus Windows DLL-path discovery; the kernel
is only registered after a successful probe.

**Limitations.** F32 only (no f16/bf16/f64/int — "Int GEMM follows in its own commit", not yet
present). Contiguous-only, zero-offset; no strided/broadcast/offset fast path. Caps are
`KernelCaps::empty()` as built.

```fkc
kernel: matmul_f32_mkl_cpu_wrapper
op_kind: MatMul
blurb: "Batched row-major [m,k] @ [k,n] -> [m,n] F32 GEMM delegated to onemkl Level-3 sgemm; contiguous-only."
backend: Cpu
kernel_source: "mkl"
entry_point: "fuel_mkl_cpu_backend::binding_table::matmul_f32_mkl_cpu_wrapper"   # §12.6 — resolves against fuel_mkl_cpu_backend::fkc::ENTRY_POINTS
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any                              # [..lhs_batch_dims.., m, k]
      shape_constraint: last_dim_eq=rhs      # k = lhs.dim[-1] == rhs.dim[-2]
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any                              # [..rhs_batch_dims.., k, n]
      shape_constraint: same_rank=lhs        # batch ranks must match (:181); per-axis equal-or-GQA-divisible
  op_params:
    variant: Matmul                          # OpParams::Matmul
    fields:
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis equal OR GQA-divisible (lhs%rhs==0) vs lhs_batch_dims" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)           # F32, same as inputs
      shape_rule: matmul(lhs, rhs)           # lhs_batch_dims + [m, n] (batch follows lhs / GQA-larger side)
      layout_guarantee: contiguous           # contiguous row-major; caller pre-allocates (preallocated ABI)
      aliasing: none                          # beta=0 full overwrite; no accumulate, no in-place

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths: []                                  # no declared already-strided fast path; contiguous-only
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured                 # unknown_cost as built; Judge bootstraps. FLOPs/bandwidth below are derivable hints only.
  class: gemm_like
  flops: "2 * lhs_batch_count * m * n * k"   # 2*M*N*K per batch slot, summed over lhs_batch_count
  bytes_moved: "lhs_batch_count * (m*k + k*n + m*n) * 4"   # F32 = 4 bytes; rhs reused under GQA but bounded here
  overhead_ns: ~                             # judge_measured — absolute launch constant is the Judge's to measure
  memory: { device_bytes: 0, host_bytes: "lhs_batch_count * m * n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true          # MKL_PRECISION: run-to-run deterministic on fixed CPU + thread count
  max_ulp: ~                                 # uncalibrated (None as built)
  max_relative: ~
  max_absolute: ~
  audited: true                              # bounds null + reason => PrecisionGuarantee::none(notes)
  notes: "MKL blocked-accumulation sgemm; deterministic on fixed hardware/threads (set MKL_CBWR for cross-machine repro). NOT bit-equal to scalar reference; parity test asserts rel err < 1e-4. ULP/rel/abs uncalibrated."

determinism: same_hardware_bitwise
```

---

## conv2d_f32_mkl_cpu_wrapper  (NCHW F32 conv2d via MKL im2col + per-(batch,group) sgemm)

One-line: NCHW F32 Conv2D via im2col + per-(batch,group) `onemkl` sgemm; (1,1)-dilation fast path, scalar fallback otherwise; optional bias.

Single-precision 2D convolution. The (1,1)-dilation, valid-shape case runs through MKL's
im2col + GEMM: the wrapper builds a `fuel_conv::ConvShape` (`:361-372`), allocates an im2col
scratch buffer of `s.im2col_len()` f32 via `onemkl::service::AlignedBuffer` (64-byte aligned,
AVX-512 cache line) **falling back to a plain `vec![0.0; ...]` if `MKL_malloc` fails — never
panics** (`:392-404`), then drives `fuel_conv::conv2d_via_gemm` with one `onemkl gemm` per
(batch, group): `m = cout_per_g`, `n = Hout*Wout`, `k = cin_per_g*kH*kW`, `NoTrans/NoTrans`,
`alpha = 1`, `beta = 0` (`:421-436`). Operands are consumed as flat slices via `as_slice()`
(`:380-386`); `_layouts` is ignored. The kernel assumes tightly-packed NCHW `x`, OIHW weight `w`,
and per-channel bias. Asymmetric stride/padding and groups (including depthwise) are supported on
the fast path. Two binding keys serve this one wrapper, distinguished by `inputs.len()` at `:305` /
`:339`: `[F32,F32,F32]` (x, w, out — no bias) and `[F32,F32,F32,F32]` (x, w, bias, out).

**Fallback boundary (load-bearing).** This wrapper does **not** own all conv shapes. It **delegates
to the scalar `fuel_cpu_backend::byte_kernels::conv2d_f32`** when (a) `dilation != (1,1)`
(`:355-360`) — `ConvShape` carries no dilation field — or (b) `ConvShape::validate()` fails
(`:373-378`). Only the (1,1)-dilation valid-shape case runs the MKL im2col+gemm path; the fallback
path inherits scalar-conv precision and cost. The bias add (when present) is done **after** gemm by
`conv2d_via_gemm` itself, per-output-channel broadcast over the spatial plane
(`fuel-conv/src/lib.rs:343-355`), not by the BLAS call.

**Numerics/perf.** Same `MKL_PRECISION` struct as the matmul kernel: deterministic on fixed
hardware/thread count, **not bit-equal** to scalar conv (blocked accumulation + im2col reordering);
parity test asserts rel err `< 1e-4`; bounds uncalibrated. `beta = 0` ⇒ full overwrite, no
accumulate, no in-place/aliasing.

**Known defect (recorded, not endorsed).** The inner gemm closure in `run_mkl_conv2d_via_gemm`
calls `.expect()` on `MatrixRef`/`MatrixMut::new` and on the gemm call (`:425-434`) — a **panic on
a production path** if those fail, which violates the CLAUDE.md never-panic rule. The contract
records this as a precision/determinism note; it is a bug to fix in the kernel, not a property to
advertise.

**Limitations.** F32 only. Contiguous-only, zero-offset, NCHW. Partial coverage (BLAS path owns
only (1,1)-dilation valid shapes; everything else falls back to scalar). Caps `KernelCaps::empty()`.

```fkc
kernel: conv2d_f32_mkl_cpu_wrapper
op_kind: Conv2D
blurb: "NCHW F32 Conv2D via im2col + per-(batch,group) onemkl sgemm; (1,1)-dilation fast path, scalar fallback otherwise; optional bias."
backend: Cpu
kernel_source: "mkl"
entry_point: "fuel_mkl_cpu_backend::binding_table::conv2d_f32_mkl_cpu_wrapper"   # §12.6 — resolves against fuel_mkl_cpu_backend::fkc::ENTRY_POINTS; one wrapper serves both binding keys
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # NCHW [N, Cin, H, W]
    - name: w
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # OIHW [Cout, Cin, kH, kW]
    - name: bias                             # OPTIONAL: present only on the [F32x4] binding key (inputs.len()==3)
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                                # [Cout], per-output-channel; added post-gemm by conv2d_via_gemm
  op_params:
    variant: Conv2D                          # OpParams::Conv2D
    fields:
      x_shape: { kind: "[usize; 4]", constraint: "[N, Cin, H, W]" }
      w_shape: { kind: "[usize; 4]", constraint: "[Cout, Cin, kH, kW]" }
      out_shape: { kind: "[usize; 4]", constraint: "[N, Cout, Hout, Wout]" }
      stride: { kind: "(usize, usize)", constraint: "asymmetric supported" }
      padding: { kind: "(usize, usize)", constraint: "asymmetric supported" }
      dilation: { kind: "(usize, usize)", constraint: "(1,1) => MKL fast path; otherwise scalar fallback" }
      groups: { kind: usize, constraint: "groups incl. depthwise supported on fast path" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)             # F32
      shape_rule: conv2d(params)             # [N, Cout, Hout, Wout] from ConvShape::h_out()/w_out()
      layout_guarantee: contiguous           # contiguous row-major NCHW; caller pre-allocates
      aliasing: none                          # beta=0 full overwrite; bias added post-gemm; no in-place

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize + sums its cost
  fast_paths:
    - { when: "dilation == (1,1) && ConvShape::validate() ok", note: "MKL im2col+gemm path; otherwise delegates to scalar fuel_cpu_backend::byte_kernels::conv2d_f32" }
  in_place: false
  alignment_bytes: 64                        # MKL AlignedBuffer (AVX-512 cache line); falls back to vec! if MKL_malloc fails
  access_granularity_bits: 32

cost:
  provenance: judge_measured                 # unknown_cost as built; Judge bootstraps. Formula below is a derivable hint only.
  class: conv
  flops: "2 * N * groups * cout_per_g * (Hout * Wout) * (cin_per_g * kH * kW)"   # 2*M*N*K per (batch,group) gemm, summed
  bytes_moved: ~                             # judge_measured (im2col scratch + gemm traffic; not cleanly derivable)
  overhead_ns: ~                             # judge_measured — absolute launch constant is the Judge's to measure
  memory: { device_bytes: 0, host_bytes: "N * Cout * Hout * Wout * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true          # MKL_PRECISION: deterministic on fixed hardware/threads
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                              # bounds null + reason => none(notes)
  notes: "MKL im2col+blocked-gemm; deterministic on fixed hardware/threads; NOT bit-equal to scalar conv (blocked accumulation + im2col reordering); parity rel err < 1e-4; bounds uncalibrated. Scalar-fallback path (dilation!=(1,1) or invalid ConvShape) inherits scalar-conv precision. NOTE: inner gemm closure uses .expect() (:425-434) -> panic-on-production-path defect (CLAUDE.md never-panic), bug to fix."

determinism: same_hardware_bitwise
```
