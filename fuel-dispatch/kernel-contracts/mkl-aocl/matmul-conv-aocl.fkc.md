---
fkc_version: 1
provider:
  name: fuel-aocl-cpu-backend
  backend: Cpu                                  # maps to BackendId::Cpu
  kernel_source: "aocl"                         # the BindingEntry.kernel_source tag
  link_registry: fuel_aocl_cpu_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map; every entry_point below roots in THIS crate
  revision_base: "git:f41137b4"                # provider build id, folded into kernel_revision_hash
---

# fuel-aocl-cpu-backend — kernel contracts (matmul family)

This crate registers a small set of AMD-AOCL/BLIS-backed kernel wrappers on the unified
`KernelBindingTable` at `BackendId::Cpu`, tagged `kernel_source = "aocl"`. They are *sibling
alternatives* to the Intel-MKL wrappers (`matmul-conv-mkl.fkc.md`) and to `fuel-cpu-backend`'s
portable kernels at the same `Cpu` key. The GEMM math is delegated to `aocl_blas::gemm` (AMD
BLIS); Conv2D is delegated to the shared `fuel_conv::conv2d_via_gemm` im2col+gemm driver.
Everything not contracted here is served by `fuel-cpu-backend`'s portable kernels.

> **Split rationale (link-registry resolution).** MKL and AOCL ship as two **separate provider
> crates** with two **separate link registries** — `fuel_mkl_cpu_backend::fkc::ENTRY_POINTS` and
> `fuel_aocl_cpu_backend::fkc::ENTRY_POINTS`. An `entry_point` is resolved against **its own
> file's** `link_registry` (§12.6), so an AOCL `entry_point` rooted at `fuel_aocl_cpu_backend::*`
> can only resolve against the AOCL registry. The two providers therefore live in two bundle
> files (this one + `matmul-conv-mkl.fkc.md`), each with its own front-matter (provider, backend,
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

## matmul_f32_aocl_cpu_wrapper  (batched row-major F32 GEMM via AMD AOCL/BLIS)

One-line: Batched row-major `[m,k] @ [k,n] -> [m,n]` F32 GEMM delegated to `aocl_blas::gemm` (BLIS); contiguous-only.

Structurally identical to the MKL matmul kernel but delegated to `aocl_blas::gemm` (AMD BLIS).
For each batch slot the wrapper issues one
`aocl_blas::gemm(Trans::No, Trans::No, m, n, k, 1.0, a, b, 0.0, c)` (`:266-278`) — note AOCL's gemm
takes `m, n, k` directly with no `MatrixRef` wrapper, unlike MKL. Inputs are consumed as flat,
tightly-packed slices via `as_slice()` (`:238-240`); `_layouts` is ignored. The same exact
packed-byte-count validation (`:211-237`) rejects any stride/offset/broadcast layout. Batch ranks
must match; per-axis dims must be equal **or** GQA-divisible (`lhs % rhs == 0`, `n_rep = lhs/rhs`),
with the same row-major unravel/ravel batch mapping (`:180-265`). Output batch follows lhs.

**Numerics/perf.** `AOCL_PRECISION` (`:48-56`): BLIS blocked accumulation, deterministic on fixed
CPU + thread count, **not bit-equal** to scalar (different accumulation order); parity test asserts
rel err `< 1e-4`; ULP/relative/absolute uncalibrated. `beta = 0` ⇒ full overwrite, no accumulation,
no in-place. Availability self-gated by a 2×2 sgemm probe (`probe_aocl_loadable`) plus Windows
DLL-path discovery; registered only after a successful probe.

**Limitations.** F32 only. Contiguous-only, zero-offset; no strided/broadcast/offset fast path.
Caps `KernelCaps::empty()`.

```fkc
kernel: matmul_f32_aocl_cpu_wrapper
op_kind: MatMul
blurb: "Batched row-major [m,k] @ [k,n] -> [m,n] F32 GEMM delegated to aocl_blas::gemm (BLIS); contiguous-only."
backend: Cpu
kernel_source: "aocl"
entry_point: "fuel_aocl_cpu_backend::binding_table::matmul_f32_aocl_cpu_wrapper"   # §12.6 — resolves against fuel_aocl_cpu_backend::fkc::ENTRY_POINTS
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
      shape_constraint: same_rank=lhs        # batch ranks must match; per-axis equal-or-GQA-divisible
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
      dtype_rule: passthrough(lhs)           # F32
      shape_rule: matmul(lhs, rhs)           # lhs_batch_dims + [m, n]
      layout_guarantee: contiguous           # contiguous row-major; caller pre-allocates
      aliasing: none                          # beta=0 full overwrite; no accumulate, no in-place

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize + sums its cost
  fast_paths: []                                  # no already-strided fast path; contiguous-only
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured                 # unknown_cost as built; Judge bootstraps. FLOPs/bandwidth below are derivable hints only.
  class: gemm_like
  flops: "2 * lhs_batch_count * m * n * k"   # 2*M*N*K per batch slot, summed
  bytes_moved: "lhs_batch_count * (m*k + k*n + m*n) * 4"
  overhead_ns: ~                             # judge_measured — absolute launch constant is the Judge's to measure
  memory: { device_bytes: 0, host_bytes: "lhs_batch_count * m * n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true          # AOCL_PRECISION: BLIS deterministic on fixed CPU + thread count
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                              # bounds null + reason => none(notes)
  notes: "AOCL/BLIS blocked-accumulation sgemm; deterministic on fixed hardware/threads; NOT bit-equal to scalar reference; parity test asserts rel err < 1e-4; ULP/rel/abs uncalibrated."

determinism: same_hardware_bitwise
```

---

## conv2d_f32_aocl_cpu_wrapper  (NCHW F32 conv2d via AOCL im2col + per-(batch,group) BLIS sgemm)

One-line: NCHW F32 Conv2D via im2col + per-(batch,group) `aocl_blas` sgemm; (1,1)-dilation fast path, scalar fallback otherwise; optional bias.

Structurally identical to the MKL conv kernel but delegated to AOCL/BLIS. The (1,1)-dilation,
valid-shape case builds a `fuel_conv::ConvShape` (`:354-365`), allocates the im2col scratch as a
plain `vec![0.0; s.im2col_len()]` (`:380` — **no aligned buffer**, unlike MKL), and drives
`fuel_conv::conv2d_via_gemm` with an
`aocl_blas::gemm(Trans::No, Trans::No, m, n, k, 1.0, a, b, 0.0, c)` closure per (batch, group)
(`:382-394`). Operands are flat `as_slice()` reads (`:373-379`); `_layouts` ignored; NCHW `x`,
OIHW `w`, per-channel bias assumed. Asymmetric stride/padding and groups/depthwise supported on the
fast path. Two binding keys (one wrapper), dispatched by `inputs.len()` at `:298` / `:332`:
`[F32,F32,F32]` (no bias) and `[F32,F32,F32,F32]` (with bias).

**Fallback boundary (load-bearing).** Delegates to the scalar
`fuel_cpu_backend::byte_kernels::conv2d_f32` when `dilation != (1,1)` (`:348-353`) or
`ConvShape::validate()` fails (`:366-371`); only the (1,1)-dilation valid case runs AOCL
im2col+gemm. The bias add (when present) is done **after** gemm by `conv2d_via_gemm`,
per-output-channel over the spatial plane (`fuel-conv/src/lib.rs:343-355`), not by the BLAS call.

**Numerics/perf.** `AOCL_PRECISION`: deterministic on fixed hardware/threads, **not bit-equal** to
scalar conv (blocked accumulation + im2col reordering); parity rel err `< 1e-4`; bounds
uncalibrated. `beta = 0` ⇒ full overwrite, no accumulate, no in-place.

**Known defect (recorded, not endorsed).** The AOCL gemm closure calls
`.expect("aocl_blas::gemm in conv2d_via_gemm")` (`:392`) — a **panic on a production path** if the
BLAS call errors, violating the CLAUDE.md never-panic rule. Recorded as a note; a bug to fix.

**Limitations.** F32 only. Contiguous-only, zero-offset, NCHW. Partial coverage (BLAS path owns
only (1,1)-dilation valid shapes; scalar fallback otherwise). Caps `KernelCaps::empty()`.

```fkc
kernel: conv2d_f32_aocl_cpu_wrapper
op_kind: Conv2D
blurb: "NCHW F32 Conv2D via im2col + per-(batch,group) aocl_blas sgemm; (1,1)-dilation fast path, scalar fallback otherwise; optional bias."
backend: Cpu
kernel_source: "aocl"
entry_point: "fuel_aocl_cpu_backend::binding_table::conv2d_f32_aocl_cpu_wrapper"   # §12.6 — resolves against fuel_aocl_cpu_backend::fkc::ENTRY_POINTS; one wrapper serves both binding keys
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
      dilation: { kind: "(usize, usize)", constraint: "(1,1) => AOCL fast path; otherwise scalar fallback" }
      groups: { kind: usize, constraint: "groups incl. depthwise supported on fast path" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)             # F32
      shape_rule: conv2d(params)             # [N, Cout, Hout, Wout]
      layout_guarantee: contiguous           # contiguous row-major NCHW; caller pre-allocates
      aliasing: none                          # beta=0 full overwrite; bias post-gemm; no in-place

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize + sums its cost
  fast_paths:
    - { when: "dilation == (1,1) && ConvShape::validate() ok", note: "AOCL im2col+gemm path; otherwise delegates to scalar fuel_cpu_backend::byte_kernels::conv2d_f32" }
  in_place: false
  alignment_bytes: 16                        # plain vec! im2col scratch (no aligned buffer, unlike MKL); default CPU alignment
  access_granularity_bits: 32

cost:
  provenance: judge_measured                 # unknown_cost as built; Judge bootstraps. Formula below is a derivable hint only.
  class: conv
  flops: "2 * N * groups * cout_per_g * (Hout * Wout) * (cin_per_g * kH * kW)"   # 2*M*N*K per (batch,group) gemm, summed
  bytes_moved: ~                             # judge_measured
  overhead_ns: ~                             # judge_measured — absolute launch constant is the Judge's to measure
  memory: { device_bytes: 0, host_bytes: "N * Cout * Hout * Wout * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true          # AOCL_PRECISION: deterministic on fixed hardware/threads
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                              # bounds null + reason => none(notes)
  notes: "AOCL/BLIS im2col+blocked-gemm; deterministic on fixed hardware/threads; NOT bit-equal to scalar conv (blocked accumulation + im2col reordering); parity rel err < 1e-4; bounds uncalibrated. Scalar-fallback path (dilation!=(1,1) or invalid ConvShape) inherits scalar-conv precision. NOTE: gemm closure uses .expect(\"aocl_blas::gemm in conv2d_via_gemm\") (:392) -> panic-on-production-path defect (CLAUDE.md never-panic), bug to fix."

determinism: same_hardware_bitwise
```
