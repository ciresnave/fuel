---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                       # maps to BackendId::Cpu
  kernel_source: "portable-cpu"      # the BindingEntry.kernel_source tag
  link_registry: fuel_cpu_backend::byte_kernels::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — SSM / Mamba kernel contracts

The state-space-model (Mamba-1 / Mamba-2) and sequence-loss kernels for the portable
`CpuStorageBytes` surface, plus the fused softmax cross-entropy loss they are commonly trained
against. Four logical ops, each monomorphized over the four float dtypes `{F32, F64, BF16, F16}`,
so this bundle registers 16 primitive `op_kind` contracts:

- **`fused_softmax_cross_entropy_{f32,f64,bf16,f16}`** — `OpKind::FusedSoftmaxCrossEntropy`,
  `OpParams::FusedSoftmaxCrossEntropy` (`fuel-dispatch/src/kernel.rs:604`; `OpKind` at
  `fuel-core-types/src/dispatch.rs:463`). Source: the `fused_softmax_cross_entropy_kernel!` macro,
  `fuel-cpu-backend/src/byte_kernels.rs:5155`, instantiated `:5251-5262`.
- **`causal_conv1d_{f32,f64,bf16,f16}`** — `OpKind::CausalConv1d`, `OpParams::CausalConv1d`
  (`kernel.rs:621`; `OpKind` `dispatch.rs:472`). Source: `causal_conv1d_native_kernel!` (`:5348`,
  f32/f64 at `:5404-5405`) + `causal_conv1d_half_kernel!` (`:5411`, bf16/f16 at `:5466-5467`).
- **`selective_scan_{f32,f64,bf16,f16}`** — `OpKind::SelectiveScan`, `OpParams::SelectiveScan`
  (`kernel.rs:638`; `OpKind` `dispatch.rs:479`). Source: `selective_scan_kernel!` (`:5534`,
  instantiated `:5627-5634`). **Multi-output bundle** (`[y ; last_state]`, Option C).
- **`ssd_chunk_scan_{f32,f64,bf16,f16}`** — `OpKind::SsdChunkScan`, `OpParams::SsdChunkScan`
  (`kernel.rs:657`; `OpKind` `dispatch.rs:486`). Source: `ssd_chunk_scan_kernel!` (`:5724`,
  instantiated `:5822-5829`). **Multi-output bundle** (`[y ; last_state]`, Option C).

All four ops are **primitive `op_kind` contracts** (none is a `fused_op` — "fused" in
`FusedSoftmaxCrossEntropy` names the *softmax+NLL fusion inside one op*, not a graph `FusedOpId`;
its param carrier is `OpParams`, the primitive namespace, §3.7). They are the production
`CpuStorageBytes` path that the dispatch wrapper (`fuel_dispatch::dispatch::cpu_wrappers`) extracts
and calls; each consumes flat contiguous zero-offset row-major slices plus an explicit `usize`
geometry, never a `Layout`/strides/offset (the cross-cutting CPU byte-kernel rule — the pipelined
executor's auto-Contiguize pass realizes any strided/broadcast/offset input *before* these kernels
run). Hence every operand below is `contiguous: required`, every output is `contiguous`, and the
`awkward_layout_strategy` is `requires_contiguous` throughout.

**Family precision invariant.** The two scans (`selective_scan`, `ssd_chunk_scan`) accumulate the
recurrent hidden state in an **f64** accumulator *regardless of element dtype T*, narrowing to T on
the `y` and `last_state` stores (`:5581-5621`, `:5785-5816`). `fused_softmax_cross_entropy` does all
log-sum-exp / NLL math in **f64** (f64 lossless, the other three promote) and always outputs F32
(`:5215-5241`). `causal_conv1d` follows the ordinary CPU half-float rule: f32/f64 native, bf16/f16
widen to **f32**, narrow on store (`:5404-5405` native vs `:5466-5467` half).

---

## fused_softmax_cross_entropy_f32  (fused softmax + NLL cross-entropy loss, f32 logits, f64 math)

Single-pass fused softmax cross-entropy: combines a numerically-stable log-softmax (row-max
subtract → log-sum-exp), the negative-log-likelihood pick at the target class, `ignore_index`
masking, and the requested reduction in one walk over `logits [n_rows, vocab]`
(`byte_kernels.rs:5155`). `logits` arrive in T = f32; `targets` are **I64** class indices `[n_rows]`;
the output is **always F32** (`fixed(F32)`). All loss math runs in **f64**: the row max is taken in
T (order-preserving), then promoted to f64 before the shift so even half inputs get f64 dynamic
range under `exp()`; the per-row `nll = -(logit[target] - row_max - ln Σ exp(logit - row_max))` is
accumulated in f64 (`:5215-5221`). Reduction tags (ABI `u8`, mapping `Mean=0, Sum=1, None=2`,
`:5084-5086`, matching `Reduction::as_tag()`): **Mean** and **Sum** produce a single F32 scalar
(output shape `[]`, 1 element), **None** produces F32 `[n_rows]` (`:5125-5126`). `ignore_index` rows
are skipped (no contribution to the sum / per-row output stays 0.0); **Mean over zero valid rows is
defined as 0.0** (not NaN, `:5234`); `vocab == 0` is defined as loss 0.0 (`:5174-5180`); a non-ignored
target outside `[0, vocab)` is a build/run **Error** (`:5195-5202`), never a panic. Known
limitations: contiguous zero-offset only (the `[..., V]` logits are flattened to `[n_rows, vocab]`
upstream and must be contiguous); targets must be I64; no in-place.

```fkc
kernel: fused_softmax_cross_entropy_f32
op_kind: FusedSoftmaxCrossEntropy
blurb: "Fused softmax cross-entropy loss, f32 logits + I64 targets; f64 log-sum-exp; output always F32 (scalar for Mean/Sum, [n_rows] for None)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::fused_softmax_cross_entropy_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: logits
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [n_rows, vocab] (flattened from [..., V] upstream)
    - name: targets
      dtypes: [I64]                        # class indices; fixed I64 regardless of logits dtype
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [n_rows]
      shape_constraint: "dim[0]=logits"    # n_rows matches logits.dim[0]
  op_params:
    variant: FusedSoftmaxCrossEntropy      # OpParams::FusedSoftmaxCrossEntropy (primitive namespace; §3.7)
    fields:
      n_rows:       { kind: usize, constraint: "== logits.dim[0] == targets.dim[0]" }
      vocab:        { kind: usize, constraint: "== logits.dim[1]" }
      reduction:    { kind: "Reduction", note: "Mean|Sum|None; ABI tag Mean=0,Sum=1,None=2" }
      ignore_index: { kind: i64, note: "target rows equal to this are skipped; conventional sentinel -100" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)               # always F32 regardless of logits dtype
      shape_rule: from_params(reduction)   # Mean/Sum -> [] (scalar); None -> [n_rows]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "vocab == 0", note: "pathological no-classes early return: loss 0.0" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: reduction
  # Two passes over the [n_rows, vocab] logit grid (row-max then sum-exp), plus a target pick:
  # ~3 ops/element over n_rows*vocab => derivable FLOPs. Bandwidth = read logits + targets + write out.
  flops: "3 * n_rows * vocab"
  bytes_moved: "(n_rows * vocab * dtype_bytes) + (n_rows * 8) + out_bytes"
  memory: { device_bytes: 0, host_bytes: "n_rows * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic two-pass loop; f64 log-sum-exp accumulator
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                         # CPU primitive: family default (PRIMITIVE_DETERMINISTIC_CPU) applies (§4.8/§12.4)
  notes: "all loss math in f64 (f32 logits promoted); stable row-max log-sum-exp; output F32. Deterministic; not bit-stable cross-hardware (FMA/exp contraction may differ)."

determinism: same_hardware_bitwise
```

## fused_softmax_cross_entropy_f64  (fused softmax + NLL cross-entropy loss, f64 logits, f64 math)

Identical algorithm to `fused_softmax_cross_entropy_f32` with f64 logits (`byte_kernels.rs:5254`).
The f64 path is the lossless case of the family invariant: the loss math is f64, the f64 logits feed
it directly (no promotion). Same I64 targets, same **F32** output (`fixed(F32)`, scalar for
Mean/Sum, `[n_rows]` for None), same `ignore_index` / Mean-over-zero=0.0 / `vocab==0`=0.0 /
out-of-range-target-is-Error semantics, same contiguous zero-offset row-major byte-length
validation (now against an 8-byte logit element). Limitations match the family: contiguous
zero-offset only, I64 targets, no in-place.

```fkc
kernel: fused_softmax_cross_entropy_f64
op_kind: FusedSoftmaxCrossEntropy
blurb: "Fused softmax cross-entropy loss, f64 logits + I64 targets; f64 log-sum-exp; output always F32 (scalar for Mean/Sum, [n_rows] for None)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::fused_softmax_cross_entropy_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: logits
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
    - name: targets
      dtypes: [I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "dim[0]=logits"
  op_params:
    variant: FusedSoftmaxCrossEntropy
    fields:
      n_rows:       { kind: usize, constraint: "== logits.dim[0] == targets.dim[0]" }
      vocab:        { kind: usize, constraint: "== logits.dim[1]" }
      reduction:    { kind: "Reduction", note: "Mean|Sum|None; ABI tag Mean=0,Sum=1,None=2" }
      ignore_index: { kind: i64, note: "target rows equal to this are skipped; conventional sentinel -100" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(reduction)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "vocab == 0", note: "pathological no-classes early return: loss 0.0" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: reduction
  flops: "3 * n_rows * vocab"
  bytes_moved: "(n_rows * vocab * dtype_bytes) + (n_rows * 8) + out_bytes"
  memory: { device_bytes: 0, host_bytes: "n_rows * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "loss math in f64 (f64 logits fed directly, lossless); stable row-max log-sum-exp; output F32. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## fused_softmax_cross_entropy_bf16  (fused softmax + NLL cross-entropy loss, bf16 logits, f64 math)

The bf16-logit instantiation (`byte_kernels.rs:5257`). Same algorithm and the same **f64** loss
math as the rest of the family: the row max is taken in bf16 (comparison is order-preserving), then
each logit is widened via `.to_f64()` before the shift/exp so the bf16 inputs get the full f64
dynamic range; the per-row NLL accumulates in f64. Output is **F32** (`fixed(F32)`), scalar for
Mean/Sum, `[n_rows]` for None. Same I64 targets, `ignore_index` masking, Mean-over-zero=0.0,
`vocab==0`=0.0, out-of-range-target-is-Error. bf16 logit element is 2 bytes. Limitations match the
family: contiguous zero-offset only, I64 targets, no in-place.

```fkc
kernel: fused_softmax_cross_entropy_bf16
op_kind: FusedSoftmaxCrossEntropy
blurb: "Fused softmax cross-entropy loss, bf16 logits + I64 targets; f64 log-sum-exp; output always F32 (scalar for Mean/Sum, [n_rows] for None)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::fused_softmax_cross_entropy_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: logits
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
    - name: targets
      dtypes: [I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "dim[0]=logits"
  op_params:
    variant: FusedSoftmaxCrossEntropy
    fields:
      n_rows:       { kind: usize, constraint: "== logits.dim[0] == targets.dim[0]" }
      vocab:        { kind: usize, constraint: "== logits.dim[1]" }
      reduction:    { kind: "Reduction", note: "Mean|Sum|None; ABI tag Mean=0,Sum=1,None=2" }
      ignore_index: { kind: i64, note: "target rows equal to this are skipped; conventional sentinel -100" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(reduction)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "vocab == 0", note: "pathological no-classes early return: loss 0.0" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: reduction
  flops: "3 * n_rows * vocab"
  bytes_moved: "(n_rows * vocab * dtype_bytes) + (n_rows * 8) + out_bytes"
  memory: { device_bytes: 0, host_bytes: "n_rows * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic loop; bf16 widened to f64 for all loss math
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "loss math in f64 (bf16 logits widened on load); stable row-max log-sum-exp; output F32. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## fused_softmax_cross_entropy_f16  (fused softmax + NLL cross-entropy loss, f16 logits, f64 math)

Byte-for-byte the same path as `fused_softmax_cross_entropy_bf16` with `half::f16` substituted for
`half::bf16` (`byte_kernels.rs:5260`): row max in f16, `.to_f64()` widen before shift/exp, f64 NLL
accumulator, **F32** output (`fixed(F32)`, scalar for Mean/Sum, `[n_rows]` for None). Differs from
bf16 only in the IEEE half storage format (10-bit mantissa, narrower exponent range). Same I64
targets, `ignore_index` / Mean-over-zero=0.0 / `vocab==0`=0.0 / out-of-range-target-is-Error
semantics, 2-byte logit element. Limitations match the family: contiguous zero-offset only, I64
targets, no in-place.

```fkc
kernel: fused_softmax_cross_entropy_f16
op_kind: FusedSoftmaxCrossEntropy
blurb: "Fused softmax cross-entropy loss, f16 logits + I64 targets; f64 log-sum-exp; output always F32 (scalar for Mean/Sum, [n_rows] for None)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::fused_softmax_cross_entropy_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: logits
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
    - name: targets
      dtypes: [I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "dim[0]=logits"
  op_params:
    variant: FusedSoftmaxCrossEntropy
    fields:
      n_rows:       { kind: usize, constraint: "== logits.dim[0] == targets.dim[0]" }
      vocab:        { kind: usize, constraint: "== logits.dim[1]" }
      reduction:    { kind: "Reduction", note: "Mean|Sum|None; ABI tag Mean=0,Sum=1,None=2" }
      ignore_index: { kind: i64, note: "target rows equal to this are skipped; conventional sentinel -100" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(reduction)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "vocab == 0", note: "pathological no-classes early return: loss 0.0" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: reduction
  flops: "3 * n_rows * vocab"
  bytes_moved: "(n_rows * vocab * dtype_bytes) + (n_rows * 8) + out_bytes"
  memory: { device_bytes: 0, host_bytes: "n_rows * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic loop; f16 widened to f64 for all loss math
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "loss math in f64 (f16 logits widened on load); stable row-max log-sum-exp; output F32. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## causal_conv1d_f32  (depthwise causal 1-D conv + bias + optional fused SiLU, f32 native)

The Mamba prefill convolution: a **depthwise** 1-D causal convolution with a per-channel bias and an
optional fused SiLU activation on the output store (`byte_kernels.rs:5348`). Inputs are
`x [batch, channels, seq_in]` (the **caller** left-pads with `kernel - 1` zeros so that
`seq_in == seq_out + kernel - 1`, validated `:5297-5303`), `weight [channels, 1, kernel]` (depthwise
— one length-`kernel` filter per channel), and `bias [channels]`. For each `(b, c, t)` it computes
`y = bias[c] + Σ_{k} weight[c,0,k] · x[b, c, t+k]` over the `kernel` consecutive input positions, and
if `use_silu` applies `SiLU(y) = y / (1 + exp(-y))` on store (`:5385-5396`). The output is
`[batch, channels, seq_out]` in T. f32 arithmetic and accumulator throughout (no widen). Pure
nested positional walk (`batch × channels × seq_out × kernel`) over contiguous zero-offset row-major
buffers; fully overwrites a caller-preallocated `out`. Empty work (`seq_out==0 || channels==0 ||
batch==0`) returns `Ok(())` after validation (`:5372-5374`). Validation is byte-length checks
returning `Result`, never a panic (`:5314-5341`). Known limitations: contiguous zero-offset only
(caller is responsible for the left-pad and for contiguizing strided inputs); no in-place; the
per-output cost is `O(kernel)` FMAs — bandwidth-bound for the small Mamba kernel size (4).

```fkc
kernel: causal_conv1d_f32
op_kind: CausalConv1d
blurb: "Depthwise causal 1-D conv (caller left-pads) + per-channel bias + optional fused SiLU, f32 native; x[batch,channels,seq_in] -> out[batch,channels,seq_out]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::causal_conv1d_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [batch, channels, seq_in] (caller left-padded: seq_in = seq_out + kernel - 1)
    - name: weight
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [channels, 1, kernel] depthwise
      shape_constraint: "dim[0]=x"         # channels matches x.dim[1]
    - name: bias
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [channels]
      shape_constraint: "dim[0]=x"         # channels matches x.dim[1]
  op_params:
    variant: CausalConv1d                  # OpParams::CausalConv1d (primitive namespace; §3.7)
    fields:
      batch:    { kind: usize, constraint: "== x.dim[0]" }
      channels: { kind: usize, constraint: "== x.dim[1] == weight.dim[0] == bias.dim[0]" }
      seq_in:   { kind: usize, constraint: "== x.dim[2] == seq_out + kernel - 1" }
      seq_out:  { kind: usize }
      kernel:   { kind: usize, constraint: "== weight.dim[2]" }
      use_silu: { kind: bool, note: "fuse SiLU(y) = y / (1 + exp(-y)) on the output store" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(out)         # [batch, channels, seq_out]; symbolic seq_out preserved
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: conv }
    - { when: "use_silu == false", note: "no SiLU branch on store" }
    - { when: "seq_out == 0", note: "empty-work early return after validation" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: conv
  # Depthwise: kernel FMAs per output element over batch*channels*seq_out outputs => derivable FLOPs.
  # Bandwidth: read x + weight + bias, write out.
  flops: "2 * batch * channels * seq_out * kernel"
  bytes_moved: "(batch * channels * (seq_in + seq_out) + channels * kernel + channels) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic nested loop; native f32 accumulate
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                         # CPU primitive: family default (PRIMITIVE_DETERMINISTIC_CPU) applies (§4.8/§12.4)
  notes: "native f32 depthwise conv + bias; optional fused SiLU. Deterministic; not bit-stable cross-hardware (FMA contraction may differ)."

determinism: same_hardware_bitwise
```

## causal_conv1d_f64  (depthwise causal 1-D conv + bias + optional fused SiLU, f64 native)

Identical algorithm to `causal_conv1d_f32` evaluated in native f64 throughout
(`byte_kernels.rs:5405`, same `causal_conv1d_native_kernel!` template): same depthwise
`y = bias[c] + Σ_k weight[c,0,k]·x[b,c,t+k]`, same optional fused SiLU on store, same caller-left-pad
contract (`seq_in == seq_out + kernel - 1`), same `[batch, channels, seq_out]` overwrite, same
byte-length validation (now against an 8-byte element). f64 gives the widest precision of the family
(no widen/narrow round-trip). Limitations match the family: contiguous zero-offset only, no in-place.

```fkc
kernel: causal_conv1d_f64
op_kind: CausalConv1d
blurb: "Depthwise causal 1-D conv (caller left-pads) + per-channel bias + optional fused SiLU, f64 native; x[batch,channels,seq_in] -> out[batch,channels,seq_out]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::causal_conv1d_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
    - name: weight
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[0]=x"
    - name: bias
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "dim[0]=x"
  op_params:
    variant: CausalConv1d
    fields:
      batch:    { kind: usize, constraint: "== x.dim[0]" }
      channels: { kind: usize, constraint: "== x.dim[1] == weight.dim[0] == bias.dim[0]" }
      seq_in:   { kind: usize, constraint: "== x.dim[2] == seq_out + kernel - 1" }
      seq_out:  { kind: usize }
      kernel:   { kind: usize, constraint: "== weight.dim[2]" }
      use_silu: { kind: bool, note: "fuse SiLU(y) = y / (1 + exp(-y)) on the output store" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(out)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: conv }
    - { when: "use_silu == false", note: "no SiLU branch on store" }
    - { when: "seq_out == 0", note: "empty-work early return after validation" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: conv
  flops: "2 * batch * channels * seq_out * kernel"
  bytes_moved: "(batch * channels * (seq_in + seq_out) + channels * kernel + channels) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64 depthwise conv + bias; optional fused SiLU; widest precision of the family. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## causal_conv1d_bf16  (depthwise causal 1-D conv + bias + optional fused SiLU, bf16 I/O with f32 compute)

The `causal_conv1d_half_kernel!`-instantiated bf16 kernel (`byte_kernels.rs:5466`, macro `:5411`).
Same depthwise conv + bias + optional fused SiLU as `causal_conv1d_f32`, but **bf16 in/out with an
f32 accumulator**: bias, weights, and x are widened via `.to_f32()`, the FMA accumulation and the
optional `SiLU = acc / (1 + exp(-acc))` run in f32, then `<bf16>::from_f32(...)` narrows on store
(`:5445-5457`). This is the family's half-precision invariant (compute f32, I/O bf16). Same
caller-left-pad contract, same `[batch, channels, seq_out]` overwrite, 2-byte element width.
Limitations match the family: contiguous zero-offset only, no in-place.

```fkc
kernel: causal_conv1d_bf16
op_kind: CausalConv1d
blurb: "Depthwise causal 1-D conv (caller left-pads) + per-channel bias + optional fused SiLU, bf16 I/O with f32 compute; x[batch,channels,seq_in] -> out[batch,channels,seq_out]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::causal_conv1d_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
    - name: weight
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[0]=x"
    - name: bias
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "dim[0]=x"
  op_params:
    variant: CausalConv1d
    fields:
      batch:    { kind: usize, constraint: "== x.dim[0]" }
      channels: { kind: usize, constraint: "== x.dim[1] == weight.dim[0] == bias.dim[0]" }
      seq_in:   { kind: usize, constraint: "== x.dim[2] == seq_out + kernel - 1" }
      seq_out:  { kind: usize }
      kernel:   { kind: usize, constraint: "== weight.dim[2]" }
      use_silu: { kind: bool, note: "fuse SiLU(y) = y / (1 + exp(-y)) on the output store (in f32)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(out)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: conv }
    - { when: "use_silu == false", note: "no SiLU branch on store" }
    - { when: "seq_out == 0", note: "empty-work early return after validation" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: conv
  flops: "2 * batch * channels * seq_out * kernel"
  bytes_moved: "(batch * channels * (seq_in + seq_out) + channels * kernel + channels) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic loop; f32 accumulate, bf16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "compute in f32 (widen on load, narrow on store); bf16 I/O; optional fused SiLU in f32. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## causal_conv1d_f16  (depthwise causal 1-D conv + bias + optional fused SiLU, f16 I/O with f32 compute)

Byte-for-byte the same code path as `causal_conv1d_bf16` with `half::f16` substituted for
`half::bf16` (`byte_kernels.rs:5467`, same `causal_conv1d_half_kernel!` template): depthwise conv +
bias, f32-compute round-trip (`.to_f32()` widen, f32 FMA + optional SiLU, `<f16>::from_f32(...)`
narrow on store), same caller-left-pad contract, same `[batch, channels, seq_out]` overwrite, 2-byte
element width. Differs from bf16 only in the IEEE half storage format (10-bit mantissa, narrower
exponent range). Limitations match the family: contiguous zero-offset only, no in-place.

```fkc
kernel: causal_conv1d_f16
op_kind: CausalConv1d
blurb: "Depthwise causal 1-D conv (caller left-pads) + per-channel bias + optional fused SiLU, f16 I/O with f32 compute; x[batch,channels,seq_in] -> out[batch,channels,seq_out]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::causal_conv1d_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
    - name: weight
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[0]=x"
    - name: bias
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "dim[0]=x"
  op_params:
    variant: CausalConv1d
    fields:
      batch:    { kind: usize, constraint: "== x.dim[0]" }
      channels: { kind: usize, constraint: "== x.dim[1] == weight.dim[0] == bias.dim[0]" }
      seq_in:   { kind: usize, constraint: "== x.dim[2] == seq_out + kernel - 1" }
      seq_out:  { kind: usize }
      kernel:   { kind: usize, constraint: "== weight.dim[2]" }
      use_silu: { kind: bool, note: "fuse SiLU(y) = y / (1 + exp(-y)) on the output store (in f32)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(out)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: conv }
    - { when: "use_silu == false", note: "no SiLU branch on store" }
    - { when: "seq_out == 0", note: "empty-work early return after validation" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: conv
  flops: "2 * batch * channels * seq_out * kernel"
  bytes_moved: "(batch * channels * (seq_in + seq_out) + channels * kernel + channels) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic loop; f32 accumulate, f16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "compute in f32 (widen on load, narrow on store); f16 I/O (IEEE half); optional fused SiLU in f32. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## selective_scan_f32  (Mamba-1 selective state-space scan, f32 I/O with f64 state accumulator)

Mamba-1's selective state-space scan forward pass (`byte_kernels.rs:5534`). Five inputs:
`u [batch, seqlen, dim]`, `delta [batch, seqlen, dim]`, `a [dim, dstate]` (the log-A state matrix),
`b [batch, seqlen, dstate]`, `c [batch, seqlen, dstate]`. Per timestep, per `(batch, dim)`:
`d = softplus(delta)` if `delta_softplus` else `delta` (stable softplus
`max(x,0) + ln(1 + exp(-|x|))`, `:5594-5601`); then the per-state recurrence
`h[b,i,j] = exp(d·a[i,j])·h[b,i,j] + d·b[b,t,j]·u[b,t,i]` and output
`y[b,t,i] = Σ_j h[b,i,j]·c[b,t,j]` (`:5606-5613`). The recurrent hidden state `h [batch, dim,
dstate]` is allocated internally, **zero-initialized**, and threaded across every timestep
(`:5583`). The hidden-state accumulator is **f64 regardless of T** (the load-bearing precision
invariant), with the final `h` narrowed to T on store.

**Multi-output bundle (Option C, `:5486-5487`, `:5574-5579`).** This op emits **one** output buffer
that is the concatenation `[y_bytes ; last_state_bytes]`: slot 0 = `y [batch, seqlen, dim]`, slot 1 =
`last_state [batch, dim, dstate]`, both in T. The executor splits the preallocated buffer at the
`y` element count (`:5579`). Empty work (`batch|seqlen|dim|dstate == 0`) zero-fills the bundle and
returns `Ok(())` (`:5562-5566`). Byte-length validation returns `Result`, never a panic
(`:5509-5524`). Known limitations: contiguous zero-offset only; no in-place; complexity is
`O(batch·seqlen·dim·dstate)` FMAs (sequential recurrence over the time axis).

```fkc
kernel: selective_scan_f32
op_kind: SelectiveScan
blurb: "Mamba-1 selective state-space scan, f32 I/O with f64 state accumulator; inputs u,delta,a,b,c; bundled output [y; last_state]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::selective_scan_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: u
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [batch, seqlen, dim]
    - name: delta
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [batch, seqlen, dim]
      shape_constraint: "same_as=u"
    - name: a
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [dim, dstate] (log-A state matrix)
    - name: b
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [batch, seqlen, dstate]
    - name: c
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [batch, seqlen, dstate]
      shape_constraint: "same_as=b"
  op_params:
    variant: SelectiveScan                 # OpParams::SelectiveScan (primitive namespace; §3.7)
    fields:
      batch:          { kind: usize, constraint: "== u.dim[0]" }
      seqlen:         { kind: usize, constraint: "== u.dim[1]" }
      dim:            { kind: usize, constraint: "== u.dim[2] == a.dim[0]" }
      dstate:         { kind: usize, constraint: "== a.dim[1] == b.dim[2]" }
      delta_softplus: { kind: bool, note: "apply softplus(delta) before use" }

return:
  bundle:                                  # multi-output Option C: one buffer [y ; last_state]
    - { name: y,          dtype_rule: passthrough(u), shape_rule: from_params(y),          layout_guarantee: contiguous }   # [batch, seqlen, dim]
    - { name: last_state, dtype_rule: passthrough(u), shape_rule: from_params(last_state), layout_guarantee: contiguous }   # [batch, dim, dstate]

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: attention }
    - { when: "delta_softplus == false", note: "no softplus branch per element" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: attention
  # Sequential recurrence: per (batch, seqlen, dim, dstate) the inner loop does ~1 exp + a few FMAs.
  # FLOPs scale with batch*seqlen*dim*dstate; the transcendental exp dominates, so this is a coarse hint.
  flops: "8 * batch * seqlen * dim * dstate"
  bytes_moved: "(2 * batch * seqlen * dim + dim * dstate + 2 * batch * seqlen * dstate + batch * seqlen * dim + batch * dim * dstate) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "batch * dim * dstate * 8", disk_bytes: 0 }   # internal f64 hidden state h

precision:
  bit_stable_on_same_hardware: true      # deterministic sequential scan; f64 hidden-state accumulator
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                         # CPU primitive: family default (PRIMITIVE_DETERMINISTIC_CPU) applies (§4.8/§12.4)
  notes: "f64 hidden-state accumulator regardless of T (f32 narrowed on store); stable softplus. Deterministic; not bit-stable cross-hardware (exp/FMA contraction may differ)."

determinism: same_hardware_bitwise
```

## selective_scan_f64  (Mamba-1 selective state-space scan, f64 native)

Identical algorithm to `selective_scan_f32` with f64 I/O (`byte_kernels.rs:5628`). The hidden-state
accumulator is f64 (as for every dtype in this family), and with f64 I/O it is the lossless case (no
widen/narrow on the `y` / `last_state` stores). Same five inputs `u,delta,a,b,c`, same stable
softplus, same per-state recurrence, same internally-allocated zero-initialized `h`, same
**multi-output bundle** `[y ; last_state]` (slot 0 `y [batch, seqlen, dim]`, slot 1
`last_state [batch, dim, dstate]`). Byte-length validation against an 8-byte element. Limitations
match the family: contiguous zero-offset only, no in-place.

```fkc
kernel: selective_scan_f64
op_kind: SelectiveScan
blurb: "Mamba-1 selective state-space scan, f64 native (f64 state accumulator); inputs u,delta,a,b,c; bundled output [y; last_state]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::selective_scan_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: u
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
    - name: delta
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "same_as=u"
    - name: a
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
    - name: b
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
    - name: c
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "same_as=b"
  op_params:
    variant: SelectiveScan
    fields:
      batch:          { kind: usize, constraint: "== u.dim[0]" }
      seqlen:         { kind: usize, constraint: "== u.dim[1]" }
      dim:            { kind: usize, constraint: "== u.dim[2] == a.dim[0]" }
      dstate:         { kind: usize, constraint: "== a.dim[1] == b.dim[2]" }
      delta_softplus: { kind: bool, note: "apply softplus(delta) before use" }

return:
  bundle:
    - { name: y,          dtype_rule: passthrough(u), shape_rule: from_params(y),          layout_guarantee: contiguous }
    - { name: last_state, dtype_rule: passthrough(u), shape_rule: from_params(last_state), layout_guarantee: contiguous }

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: attention }
    - { when: "delta_softplus == false", note: "no softplus branch per element" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: attention
  flops: "8 * batch * seqlen * dim * dstate"
  bytes_moved: "(2 * batch * seqlen * dim + dim * dstate + 2 * batch * seqlen * dstate + batch * seqlen * dim + batch * dim * dstate) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "batch * dim * dstate * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f64 hidden-state accumulator with f64 I/O (lossless, no narrow); stable softplus. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## selective_scan_bf16  (Mamba-1 selective state-space scan, bf16 I/O with f64 state accumulator)

The bf16 instantiation (`byte_kernels.rs:5629`, same `selective_scan_kernel!` template). Same
algorithm, five inputs, stable softplus, recurrence, internally-allocated zero-initialized `h`, and
**multi-output bundle** `[y ; last_state]` as `selective_scan_f32`. The precision conversion is the
family invariant taken to its widest: every input element is widened to **f64** via `.to_f64()`, the
entire recurrence accumulates in f64, and the `y` / `last_state` outputs are narrowed bf16 via
`half::bf16::from_f32(v as f32)` (`:5629-5631`) — so the bf16 path keeps the full f64 accumulator
precision through to the narrow on store, 2-byte element width. Limitations match the family:
contiguous zero-offset only, no in-place.

```fkc
kernel: selective_scan_bf16
op_kind: SelectiveScan
blurb: "Mamba-1 selective state-space scan, bf16 I/O with f64 state accumulator; inputs u,delta,a,b,c; bundled output [y; last_state]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::selective_scan_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: u
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
    - name: delta
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "same_as=u"
    - name: a
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
    - name: b
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
    - name: c
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "same_as=b"
  op_params:
    variant: SelectiveScan
    fields:
      batch:          { kind: usize, constraint: "== u.dim[0]" }
      seqlen:         { kind: usize, constraint: "== u.dim[1]" }
      dim:            { kind: usize, constraint: "== u.dim[2] == a.dim[0]" }
      dstate:         { kind: usize, constraint: "== a.dim[1] == b.dim[2]" }
      delta_softplus: { kind: bool, note: "apply softplus(delta) before use" }

return:
  bundle:
    - { name: y,          dtype_rule: passthrough(u), shape_rule: from_params(y),          layout_guarantee: contiguous }
    - { name: last_state, dtype_rule: passthrough(u), shape_rule: from_params(last_state), layout_guarantee: contiguous }

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: attention }
    - { when: "delta_softplus == false", note: "no softplus branch per element" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: attention
  flops: "8 * batch * seqlen * dim * dstate"
  bytes_moved: "(2 * batch * seqlen * dim + dim * dstate + 2 * batch * seqlen * dstate + batch * seqlen * dim + batch * dim * dstate) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "batch * dim * dstate * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic scan; f64 accumulator, bf16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f64 hidden-state accumulator regardless of T (bf16 widened on load, narrowed on store). Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## selective_scan_f16  (Mamba-1 selective state-space scan, f16 I/O with f64 state accumulator)

Byte-for-byte the same code path as `selective_scan_bf16` with `half::f16` substituted for
`half::bf16` (`byte_kernels.rs:5632`, same `selective_scan_kernel!` template): every input widened
to f64 via `.to_f64()`, the full recurrence in f64, `y` / `last_state` narrowed via
`half::f16::from_f32(v as f32)`. Same five inputs, stable softplus, internally-allocated
zero-initialized `h`, and **multi-output bundle** `[y ; last_state]`. Differs from bf16 only in the
IEEE half storage format (10-bit mantissa, narrower exponent range), 2-byte element width.
Limitations match the family: contiguous zero-offset only, no in-place.

```fkc
kernel: selective_scan_f16
op_kind: SelectiveScan
blurb: "Mamba-1 selective state-space scan, f16 I/O with f64 state accumulator; inputs u,delta,a,b,c; bundled output [y; last_state]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::selective_scan_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: u
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
    - name: delta
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "same_as=u"
    - name: a
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
    - name: b
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
    - name: c
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "same_as=b"
  op_params:
    variant: SelectiveScan
    fields:
      batch:          { kind: usize, constraint: "== u.dim[0]" }
      seqlen:         { kind: usize, constraint: "== u.dim[1]" }
      dim:            { kind: usize, constraint: "== u.dim[2] == a.dim[0]" }
      dstate:         { kind: usize, constraint: "== a.dim[1] == b.dim[2]" }
      delta_softplus: { kind: bool, note: "apply softplus(delta) before use" }

return:
  bundle:
    - { name: y,          dtype_rule: passthrough(u), shape_rule: from_params(y),          layout_guarantee: contiguous }
    - { name: last_state, dtype_rule: passthrough(u), shape_rule: from_params(last_state), layout_guarantee: contiguous }

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: attention }
    - { when: "delta_softplus == false", note: "no softplus branch per element" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: attention
  flops: "8 * batch * seqlen * dim * dstate"
  bytes_moved: "(2 * batch * seqlen * dim + dim * dstate + 2 * batch * seqlen * dstate + batch * seqlen * dim + batch * dim * dstate) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "batch * dim * dstate * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic scan; f64 accumulator, f16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f64 hidden-state accumulator regardless of T (f16 widened on load, narrowed on store; IEEE half). Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## ssd_chunk_scan_f32  (Mamba-2 SSD chunked scan, f32 I/O with f64 state accumulator)

Mamba-2's State-Space Duality chunked scan forward pass (`byte_kernels.rs:5724`). Five inputs:
`x [batch, seqlen, heads, head_dim]`, `dt [batch, seqlen, heads]`, `a [heads]` (scalar log-A per
head), `b [batch, seqlen, heads, state_dim]`, `c [batch, seqlen, heads, state_dim]`. Per timestep,
per `(batch, head, head_dim)`: `exp_d_a = exp(dt·a[head])` and the per-state recurrence
`h[b,h,i,j] = exp_d_a·h[b,h,i,j] + dt·b[b,t,h,j]·x[b,t,h,i]` with output
`y[b,t,h,i] = Σ_j h[b,h,i,j]·c[b,t,h,j]` (`:5790-5807`). The per-head, per-head-dim recurrent state
`h_state [batch, heads, head_dim, state_dim]` is allocated internally, **zero-initialized**, and
threaded across the full sequence; the accumulator is **f64 regardless of T**, narrowed on store.

**`chunk_size` is an ABI-compat / GPU-parallelism knob only — the CPU result is identical for any
valid value.** v1 runs a single sequential scan over the full sequence regardless of `chunk_size`;
the param is validated `chunk_size > 0` and `seqlen % chunk_size == 0` (`:5746-5759`) but the inner
loop does not depend on it (`:5707-5723`). **Multi-output bundle (Option C, `:5779-5783`):** one
buffer `[y_bytes ; last_state_bytes]`, slot 0 = `y [batch, seqlen, heads, head_dim]` (matches `x`),
slot 1 = `last_state [batch, heads, head_dim, state_dim]`, both T. Empty work zero-fills the bundle
(`:5767-5771`). Validation returns `Result`, never a panic. Known limitations: contiguous zero-offset
only; no in-place; **v1 is mathematically the single-chunk case** (the chunked algorithm degenerates
to a multi-head selective scan when `chunk_size == seqlen`; multi-chunk inter-chunk decay propagation
is a documented follow-up, `:5640-5643`).

```fkc
kernel: ssd_chunk_scan_f32
op_kind: SsdChunkScan
blurb: "Mamba-2 SSD chunked scan, f32 I/O with f64 state accumulator; inputs x,dt,a,b,c; chunk_size ABI-compat knob; bundled output [y; last_state]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::ssd_chunk_scan_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [batch, seqlen, heads, head_dim]
    - name: dt
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [batch, seqlen, heads]
    - name: a
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [heads] (scalar log-A per head)
    - name: b
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [batch, seqlen, heads, state_dim]
    - name: c
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [batch, seqlen, heads, state_dim]
      shape_constraint: "same_as=b"
  op_params:
    variant: SsdChunkScan                  # OpParams::SsdChunkScan (primitive namespace; §3.7)
    fields:
      batch:      { kind: usize, constraint: "== x.dim[0]" }
      seqlen:     { kind: usize, constraint: "== x.dim[1]" }
      heads:      { kind: usize, constraint: "== x.dim[2] == dt.dim[2] == a.dim[0] == b.dim[2]" }
      head_dim:   { kind: usize, constraint: "== x.dim[3]" }
      state_dim:  { kind: usize, constraint: "== b.dim[3]" }
      chunk_size: { kind: usize, constraint: "chunk_size > 0; seqlen % chunk_size == 0", note: "GPU-parallelism knob; CPU result identical for any valid value" }

return:
  bundle:                                  # multi-output Option C: one buffer [y ; last_state]
    - { name: y,          dtype_rule: passthrough(x), shape_rule: same_as(x),               layout_guarantee: contiguous }   # [batch, seqlen, heads, head_dim]
    - { name: last_state, dtype_rule: passthrough(x), shape_rule: from_params(last_state),  layout_guarantee: contiguous }   # [batch, heads, head_dim, state_dim]

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: attention }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: attention
  # Sequential recurrence over (batch, seqlen, heads, head_dim, state_dim); inner loop is FMA-heavy
  # with one exp per (batch, seqlen, head). FLOPs scale with batch*seqlen*heads*head_dim*state_dim.
  flops: "6 * batch * seqlen * heads * head_dim * state_dim"
  bytes_moved: "(2 * batch * seqlen * heads * head_dim + batch * seqlen * heads + heads + 2 * batch * seqlen * heads * state_dim + batch * heads * head_dim * state_dim) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "batch * heads * head_dim * state_dim * 8", disk_bytes: 0 }   # internal f64 hidden state

precision:
  bit_stable_on_same_hardware: true      # deterministic sequential scan; f64 hidden-state accumulator
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                         # CPU primitive: family default (PRIMITIVE_DETERMINISTIC_CPU) applies (§4.8/§12.4)
  notes: "f64 hidden-state accumulator regardless of T (f32 narrowed on store); chunk_size does not affect the result. Deterministic; not bit-stable cross-hardware (exp/FMA contraction may differ)."

determinism: same_hardware_bitwise
```

## ssd_chunk_scan_f64  (Mamba-2 SSD chunked scan, f64 native)

Identical algorithm to `ssd_chunk_scan_f32` with f64 I/O (`byte_kernels.rs:5823`). The hidden-state
accumulator is f64 (as for every dtype in this family); with f64 I/O it is lossless (no widen/narrow
on store). Same five inputs `x,dt,a,b,c`, same `chunk_size` ABI-compat semantics (validated, result
unaffected), same internally-allocated zero-initialized `h_state`, same **multi-output bundle**
`[y ; last_state]` (slot 0 `y` matches `x [batch, seqlen, heads, head_dim]`, slot 1
`last_state [batch, heads, head_dim, state_dim]`). Byte-length validation against an 8-byte element.
Limitations match the family: contiguous zero-offset only, no in-place, v1 single-chunk math.

```fkc
kernel: ssd_chunk_scan_f64
op_kind: SsdChunkScan
blurb: "Mamba-2 SSD chunked scan, f64 native (f64 state accumulator); inputs x,dt,a,b,c; chunk_size ABI-compat knob; bundled output [y; last_state]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::ssd_chunk_scan_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: dt
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
    - name: a
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
    - name: b
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: c
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=b"
  op_params:
    variant: SsdChunkScan
    fields:
      batch:      { kind: usize, constraint: "== x.dim[0]" }
      seqlen:     { kind: usize, constraint: "== x.dim[1]" }
      heads:      { kind: usize, constraint: "== x.dim[2] == dt.dim[2] == a.dim[0] == b.dim[2]" }
      head_dim:   { kind: usize, constraint: "== x.dim[3]" }
      state_dim:  { kind: usize, constraint: "== b.dim[3]" }
      chunk_size: { kind: usize, constraint: "chunk_size > 0; seqlen % chunk_size == 0", note: "GPU-parallelism knob; CPU result identical for any valid value" }

return:
  bundle:
    - { name: y,          dtype_rule: passthrough(x), shape_rule: same_as(x),               layout_guarantee: contiguous }
    - { name: last_state, dtype_rule: passthrough(x), shape_rule: from_params(last_state),  layout_guarantee: contiguous }

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: attention }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: attention
  flops: "6 * batch * seqlen * heads * head_dim * state_dim"
  bytes_moved: "(2 * batch * seqlen * heads * head_dim + batch * seqlen * heads + heads + 2 * batch * seqlen * heads * state_dim + batch * heads * head_dim * state_dim) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "batch * heads * head_dim * state_dim * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f64 hidden-state accumulator with f64 I/O (lossless, no narrow); chunk_size does not affect the result. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## ssd_chunk_scan_bf16  (Mamba-2 SSD chunked scan, bf16 I/O with f64 state accumulator)

The bf16 instantiation (`byte_kernels.rs:5824`, same `ssd_chunk_scan_kernel!` template). Same
algorithm, five inputs, recurrence, `chunk_size` ABI-compat semantics, internally-allocated
zero-initialized `h_state`, and **multi-output bundle** `[y ; last_state]` as `ssd_chunk_scan_f32`.
Every input element is widened to **f64** via `.to_f64()`, the entire recurrence accumulates in f64,
and `y` / `last_state` are narrowed via `half::bf16::from_f32(v as f32)` (`:5824-5826`) — the bf16
path keeps the full f64 accumulator precision through to store, 2-byte element width. Limitations
match the family: contiguous zero-offset only, no in-place, v1 single-chunk math.

```fkc
kernel: ssd_chunk_scan_bf16
op_kind: SsdChunkScan
blurb: "Mamba-2 SSD chunked scan, bf16 I/O with f64 state accumulator; inputs x,dt,a,b,c; chunk_size ABI-compat knob; bundled output [y; last_state]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::ssd_chunk_scan_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: dt
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
    - name: a
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
    - name: b
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: c
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=b"
  op_params:
    variant: SsdChunkScan
    fields:
      batch:      { kind: usize, constraint: "== x.dim[0]" }
      seqlen:     { kind: usize, constraint: "== x.dim[1]" }
      heads:      { kind: usize, constraint: "== x.dim[2] == dt.dim[2] == a.dim[0] == b.dim[2]" }
      head_dim:   { kind: usize, constraint: "== x.dim[3]" }
      state_dim:  { kind: usize, constraint: "== b.dim[3]" }
      chunk_size: { kind: usize, constraint: "chunk_size > 0; seqlen % chunk_size == 0", note: "GPU-parallelism knob; CPU result identical for any valid value" }

return:
  bundle:
    - { name: y,          dtype_rule: passthrough(x), shape_rule: same_as(x),               layout_guarantee: contiguous }
    - { name: last_state, dtype_rule: passthrough(x), shape_rule: from_params(last_state),  layout_guarantee: contiguous }

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: attention }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: attention
  flops: "6 * batch * seqlen * heads * head_dim * state_dim"
  bytes_moved: "(2 * batch * seqlen * heads * head_dim + batch * seqlen * heads + heads + 2 * batch * seqlen * heads * state_dim + batch * heads * head_dim * state_dim) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "batch * heads * head_dim * state_dim * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic scan; f64 accumulator, bf16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f64 hidden-state accumulator regardless of T (bf16 widened on load, narrowed on store); chunk_size does not affect the result. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## ssd_chunk_scan_f16  (Mamba-2 SSD chunked scan, f16 I/O with f64 state accumulator)

Byte-for-byte the same code path as `ssd_chunk_scan_bf16` with `half::f16` substituted for
`half::bf16` (`byte_kernels.rs:5827`, same `ssd_chunk_scan_kernel!` template): every input widened
to f64 via `.to_f64()`, the full recurrence in f64, `y` / `last_state` narrowed via
`half::f16::from_f32(v as f32)`. Same five inputs, recurrence, `chunk_size` ABI-compat semantics,
internally-allocated zero-initialized `h_state`, and **multi-output bundle** `[y ; last_state]`.
Differs from bf16 only in the IEEE half storage format (10-bit mantissa, narrower exponent range),
2-byte element width. Limitations match the family: contiguous zero-offset only, no in-place, v1
single-chunk math.

```fkc
kernel: ssd_chunk_scan_f16
op_kind: SsdChunkScan
blurb: "Mamba-2 SSD chunked scan, f16 I/O with f64 state accumulator; inputs x,dt,a,b,c; chunk_size ABI-compat knob; bundled output [y; last_state]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::ssd_chunk_scan_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: dt
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
    - name: a
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
    - name: b
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: c
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=b"
  op_params:
    variant: SsdChunkScan
    fields:
      batch:      { kind: usize, constraint: "== x.dim[0]" }
      seqlen:     { kind: usize, constraint: "== x.dim[1]" }
      heads:      { kind: usize, constraint: "== x.dim[2] == dt.dim[2] == a.dim[0] == b.dim[2]" }
      head_dim:   { kind: usize, constraint: "== x.dim[3]" }
      state_dim:  { kind: usize, constraint: "== b.dim[3]" }
      chunk_size: { kind: usize, constraint: "chunk_size > 0; seqlen % chunk_size == 0", note: "GPU-parallelism knob; CPU result identical for any valid value" }

return:
  bundle:
    - { name: y,          dtype_rule: passthrough(x), shape_rule: same_as(x),               layout_guarantee: contiguous }
    - { name: last_state, dtype_rule: passthrough(x), shape_rule: from_params(last_state),  layout_guarantee: contiguous }

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: attention }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: attention
  flops: "6 * batch * seqlen * heads * head_dim * state_dim"
  bytes_moved: "(2 * batch * seqlen * heads * head_dim + batch * seqlen * heads + heads + 2 * batch * seqlen * heads * state_dim + batch * heads * head_dim * state_dim) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "batch * heads * head_dim * state_dim * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic scan; f64 accumulator, f16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f64 hidden-state accumulator regardless of T (f16 widened on load, narrowed on store; IEEE half); chunk_size does not affect the result. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```
