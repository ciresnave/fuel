---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: "fuel_dispatch::fkc::verify::harness::RopeApplyLinkRegistry (Task 4.6 verification-harness-local resolver; NOT part of the shared production fuel_dispatch::fkc::CudaLinkRegistry / cuda_link.rs table — see notes below)"
  revision_base: "git:4d852819"
---

# baracuda `rope_apply` — Task 4.6 FKC gap-closure acceptance kernel

This contract documents baracuda's standalone `rope_apply_<dt>_run` FFI family
(`baracuda-kernels-sys` 0.0.1-alpha.77, `kernels/include/baracuda_attention.cuh:1703`
the `BARACUDA_KERNELS_ROPE_APPLY_INSTANTIATE` macro; FFI decl `src/lib.rs:49243`) — a kernel
baracuda ships for Fuel that had **exactly one prior caller in the whole repo**:
`fuel_cuda_backend::storage::CudaStorage::rope` (`fuel-cuda-backend/src/storage.rs:3883-3960`,
labeled "Phase 6c.4 migration"), a method on the **legacy** dtype-tagged `CudaStorage` /
`CudaStorageSlice` storage representation that predates the current byte-oriented
`CudaStorageBytes` (the type every live dispatch wrapper — `cuda_input`/`cuda_output` in
`fuel-dispatch/src/dispatch.rs` — actually reads/writes) and has **no call sites anywhere**
(confirmed by repo-wide grep). "Shipped, never wired in" — the designated acceptance kernel for
the FKC verification harness per
`docs/session-prompts/capturedrun-4b-paused-pending-fkc-verification.md`.

**This is a DIFFERENT kernel from the already-wired `docs/kernel-contracts/cuda/rope.fkc.md`**
(`OpKind::Rope` / `entry_point: fuel_cuda_backend::fkc::rope`, resolved to
`crate::baracuda_dispatch::attention::rope_f32` and siblings, which call baracuda's
`rope_<dt>_run` / `rope_<dt>_strided_run` — a device-computed-trig, `base`+`positions` ABI, no
caller-supplied `cos`/`sin` at all). `rope_apply_<dt>_run` is a **third** RoPE FFI family in
baracuda's surface (the others being `rope_<dt>_run` above and `rope_apply_interleaved_<dt>_run`,
out of scope here) that takes CALLER-SUPPLIED `cos`/`sin` trig tables. Both sections legitimately
share `op_kind: Rope` and coexist as sibling `BindingEntry` alternatives at the same
`(Rope, dtypes, Cuda)` decision point (Phase 7.6 step 9a) — they are not duplicates.

**Real, non-obvious ABI gap this contract surfaces**: baracuda's `cos`/`sin` tables here are
**ALWAYS F32** regardless of the operand dtype, and are **HALF-WIDTH** — `[seq, head_dim/2]` (one
trig value per rotation *pair*) — per `storage.rs:3879`'s own doc comment and the FFI's
`stride_b == 0 ⇒ [td/2]`-shared-table semantics. This is narrower than the Fuel-wide
`OpParams::Rope` convention the CPU `rope_<dt>` family and the already-wired CUDA `rope_<dt>`
family both use, where `cos`/`sin` are **FULL-WIDTH** `[seq, head_dim]` (values duplicated across
both rotation halves; see `docs/kernel-contracts/cpu/rope.fkc.md`). A future production wiring of
this kernel must adapt (slice/precompute a half-width table) rather than pass a Fuel-convention
`cos`/`sin` buffer straight through — flagged here, not silently assumed away.

**Verification-harness-local link registry (not yet production).** This contract's `entry_point`
resolves through a `LinkRegistry` implementation that lives in the FKC verification harness
(`fuel-dispatch/src/fkc/verify/harness.rs`, Task 4.6), NOT the shared production
`fuel_dispatch::fkc::CudaLinkRegistry` (`cuda_link.rs`) every other CUDA contract in this
directory resolves through. This is deliberate: Task 4.6 is scaffolding for the empirical
verification harness (the acceptance test for the paused CapturedRun executor build-out), not a
production dispatch-wiring PR. Because this contract can only be *imported* (parsed **and**
lowered against a real link registry) under `--features cuda`, and no default-feature test globs
`docs/kernel-contracts/**/*.fkc.md` today (checked: no corpus-lint / `import_glob` test exists over
this directory as of this commit), this file cannot be swept into any default-feature test by
accident. The default-feature `fuel-dispatch/src/fkc/mod.rs` test suite DOES `parse_file` +
`validate_file` + a stub-linked `lower_file` over this contract (structural verification, no real
CUDA symbols needed) — see `parses_and_lowers_real_rope_apply_contract`.

---

## rope_apply  (Rope — {F32, F16, BF16, F64} varying; cos/sin fixed F32; contiguous only)

Apply RoPE rotation to `x [outer_count, seq, head_dim]` using caller-supplied, HALF-WIDTH
`cos`/`sin` tables of shape `[seq, head_dim/2]` (always F32; see the ABI-gap note above). Backs
`OpKind::Rope` as a sibling alternative to the already-wired `rope_<dt>` family. `stride_b` is
always `0` in the wiring below (Fuel's cos/sin tables are model-wide, never per-batch — the
per-row-table `stride_b == td/2` case in baracuda's ABI is out of scope for this contract).
Contiguous input only (no strided path) — mirrors the CPU `rope_<dt>` family's posture, not the
already-wired CUDA `rope_<dt>` family's stride support. Output: fresh, contiguous, no aliasing
(baracuda's own doc comment: aliasing `y` with `x` or with `cos`/`sin` is UNSAFE — two threads per
rotated pair both read both pair elements).

```fkc
kernel: rope_apply
op_kind: Rope
blurb: "apply RoPE rotation via baracuda's caller-supplied-cos/sin rope_apply_<dt>_run (CUDA/baracuda) {F32, F16, BF16, F64}; contiguous only; cos/sin always F32 half-width [seq, head_dim/2]."
backend: Cuda
kernel_source: "baracuda"
entry_point: "baracuda_kernels_rope_apply"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [outer_count, seq, head_dim]
      shape_constraint: "divisible(x.dim[2], 2)"   # head_dim even (baracuda ABI requirement)
    - name: cos
      dtypes: [F32]                        # ALWAYS F32 regardless of x's dtype (baracuda ABI)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [seq, head_dim/2] — HALF-WIDTH, NOT [seq, head_dim]
      shape_constraint: "cos.dim[0] == x.dim[1] (seq); cos.dim[1] == x.dim[2]/2 (half head_dim — baracuda ABI, narrower than the Fuel-wide full-width cos/sin convention)"
    - name: sin
      dtypes: [F32]                        # ALWAYS F32 regardless of x's dtype (baracuda ABI)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [seq, head_dim/2]
      shape_constraint: "same_as=cos"
  op_params:
    variant: Rope                          # OpParams::Rope (primitive namespace; §3.7)
    fields:
      outer_count: { kind: usize, note: "batch×heads flattened; baracuda's bh; stride_b is always 0 (one shared cos/sin table across every row)" }
      seq:         { kind: usize, constraint: "== x.dim[1] == cos.dim[0]" }
      head_dim:    { kind: usize, constraint: "== x.dim[2]; head_dim % 2 == 0; cos.dim[1] == head_dim / 2 (baracuda half-width ABI)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)               # [outer_count, seq, head_dim]
      layout_guarantee: contiguous
      aliasing: none                       # baracuda ABI: aliasing y with x (or cos/sin) is UNSAFE

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: normalization
  flops: "4 * outer_count * seq * head_dim"
  bytes_moved: "(2 * outer_count * seq * head_dim) * dtype_bytes + (2 * seq * head_dim / 2) * 4"
  memory: { device_bytes: "outer_count * seq * head_dim * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "deterministic rope apply; caller-supplied cos/sin. Author-declared `audited: true`, but the import-time V-FKC-9 gate (`fkc::verify::gate_precision`) downgrades this to UNAUDITED for any (backend, dtypes, kernel_revision_hash) tuple lacking a passing `.fkc-verified-ledger.json` entry — earning the pass is exactly what the Task 4.6 harness / `fkc_verify_rope_apply_writes_a_pass_ledger_entry` acceptance test does. Not yet cross-checked against a CPU reference (`verify_precision_bound`): that helper's current single-shared-`BindingEntry`/single-shared-`ProbeInputs` signature cannot express a CUDA-candidate-vs-CPU-reference comparison for this op, because baracuda's half-width cos/sin ABI needs DIFFERENT probe bytes than the CPU family's full-width convention for the same logical rotation — a real follow-on gap, not silently assumed sound."

determinism: same_hardware_bitwise
```
