---
fkc_version: 1
provider:
  name: fuel-graph-fused
  backend: Cpu                       # maps to BackendId::Cpu (the always-built cpu_fallback path)
  kernel_source: "portable-cpu"      # the BackendImpl.kernel_source tag (part of the dispatch key); the CPU fused kernel registered through register_default_fused_kernels
  link_registry: fuel_dispatch::dispatch::FUSED_ENTRY_POINTS   # §12.6 symbol→KernelRef map (fused-side)
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-graph fused registry — conv / RoPE / SSM kernel contracts (conv-attn family)

Six **`fused_op`** kernel-level ops from the `fuel-graph` `FusedOpRegistry`, joined at runtime to
their CPU `BackendImpl` payload by `fuel_dispatch::dispatch::register_default_fused_kernels`. Each
op is a graph-side `FusedOpEntry` (`shape_rule` / `dtype_rule` / `decompose` / `backward` /
`pattern` / `output_views`, `fuel-graph/src/registry.rs`) bound to a kernel-side `BackendImpl`
(`cost` / `precision` / `caps` / `revision`, `fuel-dispatch/src/fused.rs:63`). The six covered here
(family `conv-attn`):

- **`rope`** — `FusedOps::ROPE` (`FusedOpId(5)`; `registry.rs:854`). Param carrier
  `FusedOpParams::Rope` (no fields; `registry.rs:178`). Graph entry
  `fuel-graph/src/registry/rope.rs:30`. CPU kernel `rope_f32_cpu_wrapper`
  (`fuel-dispatch/src/dispatch.rs:1296`).
- **`conv2d`** — `FusedOps::CONV2D` (`FusedOpId(6)`; `registry.rs:858`). Param carrier
  `FusedOpParams::Conv2D { stride, padding, groups }` (`registry.rs:184`). Graph entry
  `fuel-graph/src/registry/conv2d.rs:54`.
- **`conv_transpose2d`** — `FusedOps::CONV_TRANSPOSE2D` (`FusedOpId(11)`; `registry.rs:882`). Param
  carrier `FusedOpParams::ConvTranspose2D { stride, padding, output_padding, dilation, groups }`
  (`registry.rs:210`). Graph entry `fuel-graph/src/registry/conv_transpose_2d.rs:32`.
- **`causal_conv1d`** — `FusedOps::CAUSAL_CONV1D` (`FusedOpId(18)`; `registry.rs:932`). Param
  carrier `FusedOpParams::CausalConv1d { use_silu }` (`registry.rs:339`). Graph entry
  `fuel-graph/src/registry/causal_conv1d.rs:66`.
- **`selective_scan`** — `FusedOps::SELECTIVE_SCAN` (`FusedOpId(19)`; `registry.rs:941`). Param
  carrier `FusedOpParams::SelectiveScan { delta_softplus }` (`registry.rs:323`). Graph entry
  `fuel-graph/src/registry/selective_scan.rs:86`. **Multi-output bundle** (`[y ; last_state]`,
  Option C, via `output_views`).
- **`ssd_chunk_scan`** — `FusedOps::SSD_CHUNK_SCAN` (`FusedOpId(20)`; `registry.rs:951`). Param
  carrier `FusedOpParams::SsdChunkScan { chunk_size }` (`registry.rs:274`). Graph entry
  `fuel-graph/src/registry/ssd_chunk_scan.rs:75`. **Multi-output bundle** (`[y ; last_state]`,
  Option C, via `output_views`).

All six are **`fused_op` contracts** (their param carrier is `FusedOpParams`, the fused namespace,
§3.7; not `OpParams`). Per §4.4 their cost compiles to the **fused** cost-fn shape
`fn(&[Shape], &FusedOpParams, &BackendCapabilities) -> CostEstimate` (`fused.rs:63`) — **no
`&[DType]` argument**; their `shape_rule` / `dtype_rule` compile to the graph-side
`fn(&[Shape], &FusedOpParams) -> Shape` / `fn(&[DType], &FusedOpParams) -> DType`
(`registry.rs:104,108`), and the two bundled ops' slot specs compile to
`output_views: fn(&[Shape], &[DType], &FusedOpParams) -> Vec<OutputViewSpec>` (`registry.rs:133`,
§5.5 / §12.7).

**Cross-cutting layout fact (LOAD-BEARING).** The graph-side registry does not encode layout, and
the kernel-side CPU wrappers in `fuel-dispatch/src/dispatch.rs` take a `_layouts: &[Layout]`
argument and **ignore it** (underscore-prefixed; verified `rope_f32_cpu_wrapper` at
`dispatch.rs:1296-1332`); they call `cpu_input()` (`dispatch.rs:224-233`) which returns the raw
byte buffer with **no stride application**. No `register_fused!` call in
`register_default_fused_kernels` (`dispatch.rs:5238+`) passes `caps = …`, so caps default to
`KernelCaps::empty()` (macro `fused.rs:368-400`). Consequence: **every fused kernel here is
contiguous-only, offset-0, row-major**; a strided/broadcast/offset input is realized to contiguous
by the executor's auto-Contiguize step (`StridedInputPreferenceFilter`, `strided_input_pref.rs`)
before the kernel runs. Hence every operand below is `contiguous: required` (all other layout flags
`rejected`, including `reverse_strides`), every output `layout_guarantee: contiguous`, and
`awkward_layout_strategy: requires_contiguous` throughout — the planner inserts `Op::Contiguize`
(itself an FKC kernel) and sums its cost (§4.3 / §4.4).

**dtype monomorphization.** CPU coverage is registered per-dtype over the four float dtypes
`{F32, F64, BF16, F16}`. One contract section per op lists that dtype set on its operands (the
registry's `dtype_rule` is `passthrough`); the per-dtype kernel instances differ only in element
width and the accumulate/narrow rule, never in the contract shape.

**Precision family invariant (CPU).** Every CPU fused kernel here claims
`bit_stable_on_same_hardware: true` with no static ULP/relative/absolute bound, accumulating in a
wider type than its element dtype: `rope` / `conv2d` / `conv_transpose2d` / `causal_conv1d` widen
BF16/F16 to **f32** (F64 native for F64 input), and the two scans accumulate the recurrent hidden
state in **f64** regardless of element dtype, narrowing to T on store. Per the **2026-07-03
maintainer decision (CireSnave)**, every section here declares `audited: true`: the FKC import is now
the production registration path (`register_cpu_conv_rope_fused_from_contract`), so each kernel's
bit-stable claim **relocates** from its `*_CPU_PRECISION` const onto the contract — same author, same
guarantee (bit-stable, no static bound; §4.8 / §12.4), so the flip moves the evidentiary bar, it does
not lower it. Without the flip the import would lower to `UNAUDITED` and DOWNGRADE production
metadata. The Judge still audits/refines these bit-stable seeds (§4.8); cost is `judge_measured` and
the Judge bootstraps it (§4.4).

**Backward / decompose (informational; not advertised in these forward contracts).** All six are
`NotDifferentiable` in the registry (real grads, where present, are emitted by `Tensor::backward`),
and all six **panic in `decompose`** (no primitive subgraph form) — so a backend without a native
kernel relies on the executor's `cpu_fallback` to the always-built CPU kernel contracted here. None
has a live pattern matcher (`pattern` is stub `None`).

**`param(N)` index tables (C-4 param threading, 2026-07-23).** The shape-oracle return cross-check
(`fuel-dispatch/src/fkc/return_check.rs`) evaluates a declared `shape_rule`'s `param(N)` atoms
against synthesized per-variant, per-combo values (`synth_probe_param_points`);
`param(N)` indexes the `FusedOpParams::key().ints` flattening (`FusedOpParamsKey.ints`,
`fuel-graph/src/registry.rs`) — the same encoding CSE keys on, so the declared-rule evaluator and
the real registry fn see identical values by construction. Float fields ride `key().bits`, **not**
`ints`, so they have NO `param(N)` slot; `Rope` carries no fields at all (no row). Index order per
variant (pinned by `corpus_prose_pins_param_index_tables_matching_key_ints`):

| variant | `key().ints` → `param(N)` |
|---|---|
| `Conv2D` | `param(0)=stride.0 (sh)` · `param(1)=stride.1 (sw)` · `param(2)=padding.0 (ph)` · `param(3)=padding.1 (pw)` · `param(4)=groups` |
| `ConvTranspose2D` | `param(0)=stride.0 (sh)` · `param(1)=stride.1 (sw)` · `param(2)=padding.0 (ph)` · `param(3)=padding.1 (pw)` · `param(4)=output_padding.0 (oph)` · `param(5)=output_padding.1 (opw)` · `param(6)=dilation.0 (dh)` · `param(7)=dilation.1 (dw)` · `param(8)=groups` |
| `CausalConv1d` | `param(0)=use_silu (0/1)` |
| `SelectiveScan` | `param(0)=delta_softplus (0/1)` |
| `SsdChunkScan` | `param(0)=chunk_size` |

Threading alone does NOT activate this family's whole-shape rules: `conv2d(params)` /
`conv_transpose2d(params)` / `from_params(seq_out)` and both scan slot-1 `from_params(last_state)`
bundle rules each also need a whole-shape constructor (`Dims`, KISS §6.20-reserved `0x0B`) — they
stay warned skips, **KISS-gated** on the filed §6.4 extension-registry entry
(`docs/outreach/kiss-dims-withdim-extension-registry-filed.md`). What IS live now at the
synthesized param points: the params-dependent variants' `passthrough` dtype rules (previously
dead — synth returned `None` for them).

---

## rope  (rotary position embedding with caller cos/sin tables, fused; rotate_half)

Fused rotary position embedding. Three inputs: `x [outer_count, seq, head_dim]` (the batch×heads
axis flattened into `outer_count`), and precomputed `cos` / `sin` tables of shape
`[seq, head_dim]` that broadcast over the `outer_count` axis (the broadcast is realized by the
kernel **re-indexing** the `[seq, head_dim]` tables per outer, NOT by a stride-0 view — so the
operand is an ordinary contiguous tensor, not a broadcast). Uses the **rotate_half** convention:
the head dimension splits into two halves of size `h = head_dim/2`, and for each pair `(lo, hi)`
the kernel computes `out[lo] = x[lo]·cos[lo] − x[hi]·sin[lo]` and
`out[hi] = x[hi]·cos[hi] + x[lo]·sin[hi]`. Requires rank ≥ 2 with an **even** `head_dim` (odd is a
build/run Error, never a panic). Shape and dtype are pass-through (= `x`); `seq` / `head_dim` are
recovered from the input shapes at execution time (`FusedOpParams::Rope` carries no fields). f32/f64
native; bf16/f16 widen to f32 and narrow on store. Backward is another `rope` with negated sin
(emitted by `Tensor::backward`, not a registry rule). Known limitations: contiguous zero-offset
only; no in-place; head_dim must be even; cos/sin are re-indexed, not a broadcast view.

```fkc
kernel: rope
fused_op: ROPE
blurb: "Fused rotary position embedding (rotate_half); x[outer,seq,head_dim] with cos/sin[seq,head_dim] re-indexed (broadcast) over outer; pass-through shape/dtype."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::rope_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [outer_count, seq, head_dim]
      shape_constraint: "divisible(x.dim[2], 2)"   # head_dim even (h = head_dim/2)
    - name: cos
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [seq, head_dim]; re-indexed per outer (NOT a stride-0 view)
      shape_constraint: "last_dim_eq=x"    # head_dim matches x; seq matches x.dim[1]
    - name: sin
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [seq, head_dim]
      shape_constraint: "same_as=cos"
  op_params:
    variant: Rope                          # FusedOpParams::Rope (fused namespace; no fields; §3.7)
    fields: {}                             # seq / head_dim recovered from shapes at exec time

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)               # [outer_count, seq, head_dim]; symbolic seq preserved (§5.2)
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "seq == 0", note: "empty-work early return after validation" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); the formula hints below are the derivable prior
  class: normalization
  # n = outer_count * seq * head_dim (output element count). Two rotation planes -> two FMA pairs
  # per element => derivable FLOPs; bandwidth = read x + write out + the [seq,head_dim] table reads.
  flops: "4 * outer_count * seq * head_dim"
  bytes_moved: "(2 * outer_count * seq * head_dim + 2 * seq * head_dim) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "outer_count * seq * head_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic positional nested loop; f32 compute for half, f64 for F64
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                          # 2026-07-03 maintainer flip (CireSnave): relocates the ROPE_CPU_PRECISION bit-stable claim onto the contract (same author, same guarantee); FKC import is now production — false would DOWNGRADE to UNAUDITED (§4.8/§12.4)
  notes: "rotate_half; f32/f64 native, bf16/f16 widen to f32 then narrow on store. Deterministic; not bit-stable cross-hardware (FMA contraction may differ)."

determinism: same_hardware_bitwise
```

---

## conv2d  (2-D cross-correlation with stride / padding / groups, fused)

Fused 2-D convolution (cross-correlation convention). Two or three inputs: `x [N, Cin, H, W]`,
`weight [Cout, Cin/groups, Kh, Kw]`, and an optional `bias [Cout]` (the with-bias and without-bias
forms are distinct registered dtype tuples). Computes the standard grouped cross-correlation with
the given `stride` / `padding` / `groups`; **spatial dilation is always 1** (no dilation field on
`FusedOpParams::Conv2D` — the param payload omits it until backward's dilation-as-stride trick
forces it, per the `registry.rs:180-183` comment). Output is rank-4
`[N, Cout, (H + 2·ph − Kh)/sh + 1, (W + 2·pw − Kw)/sw + 1]` (the `conv2d(params)` geometry, §5.2),
dtype = `x`. Compute widens BF16/F16 to f32 (F64 native), narrowing on store. The forward op is
`NotDifferentiable` in the registry; the real gradient is emitted by `Tensor::backward`
(dX = ConvTranspose2D, dW = a transposed conv, dB = reduce_sum_to). `decompose` panics (no
`Op::Im2Col` primitive) — backends without a native kernel use `cpu_fallback`. Known limitations:
contiguous zero-offset (NCHW packed) only; dilation fixed at 1; no in-place.

```fkc
kernel: conv2d
fused_op: CONV2D
blurb: "Fused 2-D cross-correlation with stride/padding/groups (dilation fixed at 1); x[N,Cin,H,W], weight[Cout,Cin/g,Kh,Kw], optional bias[Cout]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::conv2d_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [N, Cin, H, W]
    - name: weight
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [Cout, Cin/groups, Kh, Kw]
      shape_constraint: "divisible(x.dim[1], groups)"   # Cin divisible by groups; weight.dim[1] = Cin/groups
    - name: bias
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Cout]
      shape_constraint: "dim[0]=weight"    # Cout matches weight.dim[0]
      optional: true                       # 2-input (no-bias) vs 3-input (with-bias) registered tuples
  op_params:
    variant: Conv2D                        # FusedOpParams::Conv2D (fused namespace; §3.7)
    fields:
      stride:  { kind: "(usize, usize)", note: "(sh, sw)" }
      padding: { kind: "(usize, usize)", note: "(ph, pw)" }
      groups:  { kind: usize, constraint: "x.dim[1] % groups == 0; weight.dim[0] % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: conv2d(params)           # [N, Cout, (H+2ph-Kh)/sh+1, (W+2pw-Kw)/sw+1] (§5.2)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "groups == 1", note: "dense (non-grouped) convolution" }
    - { when: "depthwise", note: "groups == Cin == Cout depthwise path" }
    - { when: "all_inputs_contiguous", class: conv }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); the formula hint below is the derivable prior
  class: conv
  # MACs = N * Cout * Hout * Wout * (Cin/groups) * Kh * Kw; 2 FLOPs per MAC => derivable FLOPs.
  # Output spatial extent (Hout/Wout) is a geometry function of the params, so the FLOPs hint is
  # expressed over the output element count out_elems and the per-output reduction width.
  flops: "2 * out_elems * (x.dim[1] / groups) * weight.dim[2] * weight.dim[3]"
  bytes_moved: "(x_elems + weight_elems + out_elems) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic nested loop; f32 accumulate for half, f64 for F64
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                          # 2026-07-03 maintainer flip (CireSnave): relocates the CONV2D_CPU_PRECISION bit-stable claim onto the contract (same author, same guarantee); FKC import is now production — false would DOWNGRADE to UNAUDITED (§4.8/§12.4)
  notes: "cross-correlation; f32/f64 native, bf16/f16 widen to f32 then narrow on store. Deterministic; not bit-stable cross-hardware (FMA/reduction order may differ)."

determinism: same_hardware_bitwise
```

---

## conv_transpose2d  (2-D transposed / fractionally-strided convolution, optional bias, fused)

Fused 2-D transposed (fractionally-strided) convolution. Two or three inputs: `x [N, Cin, H, W]`,
`weight [Cin, Cout/groups, Kh, Kw]` (note the transposed weight layout — `Cin` leads, unlike
`conv2d`), and an optional `bias [Cout]` (the with-bias and without-bias forms are distinct
registered dtype tuples, mirroring `conv2d`; the CPU scatter kernel **seeds the output with
`bias[co]`** — or `0` when the bias is absent — then scatter-accumulates, `byte_kernels.rs`, the
same kernel the primitive `cpu/conv.fkc.md` `conv_transpose2d_f32` section documents). Carries the
full `stride` / `padding` / `output_padding` / `dilation` / `groups` bundle
(`FusedOpParams::ConvTranspose2D`, `registry.rs:210`). Output is rank-4 with
`Hout = (H − 1)·sh − 2·ph + dh·(Kh − 1) + out_pad_h + 1` (saturating; `Wout` analogous) and
`Cout = (Cout/groups)·groups`, dtype = `x`. Compute widens BF16/F16 to f32 (F64 native), narrowing
on store. Forward is `NotDifferentiable` in the registry (the `Tensor::backward` forward arm panics
— no autograd through transposed conv); `decompose` panics, so backends without a native kernel use
`cpu_fallback`. Known limitations: contiguous zero-offset only; no in-place.

```fkc
kernel: conv_transpose2d
fused_op: CONV_TRANSPOSE2D
blurb: "Fused 2-D transposed (fractionally-strided) convolution, optional bias; x[N,Cin,H,W], weight[Cin,Cout/g,Kh,Kw], optional bias[Cout]; stride/padding/output_padding/dilation/groups."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::conv_transpose2d_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [N, Cin, H, W]
    - name: weight
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [Cin, Cout/groups, Kh, Kw] (transposed layout: Cin leads)
      shape_constraint: "dim[0]=x"         # weight.dim[0] (Cin) matches x.dim[1]
    - name: bias
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Cout]; kernel seeds the output with bias[co] (or 0 when absent), NOT a stride-0 view
      shape_constraint: "out.dim[1] == bias.dim[0]"   # Cout = (Cout/groups)*groups
      optional: true                       # 2-input (no-bias) vs 3-input (with-bias) registered tuples — same CPU wrapper; the optional operand rides op-params, not a distinct entry_point
  op_params:
    variant: ConvTranspose2D               # FusedOpParams::ConvTranspose2D (fused namespace; §3.7)
    fields:
      stride:         { kind: "(usize, usize)", note: "(sh, sw)" }
      padding:        { kind: "(usize, usize)", note: "(ph, pw)" }
      output_padding: { kind: "(usize, usize)", note: "(out_pad_h, out_pad_w); resolves the stride>1 output ambiguity" }
      dilation:       { kind: "(usize, usize)", note: "(dh, dw)" }
      groups:         { kind: usize, constraint: "x.dim[1] % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: conv_transpose2d(params) # Hout=(H-1)*sh - 2*ph + dh*(Kh-1) + out_pad_h + 1 (saturating); Cout=(Cout/g)*groups (§5.2)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "groups == 1", note: "dense (non-grouped) transposed convolution" }
    - { when: "all_inputs_contiguous", class: conv }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); the formula hint below is the derivable prior
  class: conv
  # Transposed conv scatters each input element across the kernel window: MACs scale with the INPUT
  # element count times the per-position reduction width (Cout/groups)*Kh*Kw; 2 FLOPs per MAC.
  flops: "2 * x_elems * (weight.dim[1]) * weight.dim[2] * weight.dim[3]"
  bytes_moved: "(x_elems + weight_elems + out_elems) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic nested loop; f32 accumulate for half, f64 for F64
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                          # 2026-07-03 maintainer flip (CireSnave): relocates the CONV_TRANSPOSE2D_CPU_PRECISION bit-stable claim onto the contract (same author, same guarantee); FKC import is now production — false would DOWNGRADE to UNAUDITED (§4.8/§12.4)
  notes: "fractionally-strided conv (saturating output geometry); f32/f64 native, bf16/f16 widen to f32 then narrow. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## causal_conv1d  (depthwise 1-D causal conv + per-channel bias + optional fused SiLU, fused)

The Mamba prefill convolution as a fused op (baracuda `causal_conv1d_*_run` hook on GPU; CPU
fallback contracted here). A **depthwise** 1-D causal convolution with a per-channel bias and an
optional fused SiLU on the output store. Three inputs: `x [batch, channels, seq + (kernel − 1)]`
(the **caller** left-pads with `kernel − 1` zeros so the causal mask is satisfied,
`seq_in == seq_out + kernel − 1`), `weight [channels, 1, kernel]` (depthwise — one length-`kernel`
filter per channel; groups == channels in conv terms), and `bias [channels]`. For each `(b, c, t)`
it computes `y = bias[c] + Σ_k weight[c,0,k]·x[b,c,t+k]`, and if `use_silu` applies
`SiLU(y) = y / (1 + exp(−y))` on store. Output is `[batch, channels, seq_out]` where
`seq_out = x.dim[2] − (kernel − 1)` (kernel from `weight.dim[2]`), dtype = `x`. f32/f64 native;
bf16/f16 widen to f32 (accumulate + SiLU in f32), narrow on store. Forward is `NotDifferentiable`
(v1 inference-only); `decompose` panics (no `Op::Conv1D` primitive) — backends without a native
kernel use `cpu_fallback`. Known limitations: contiguous zero-offset only (caller owns the left-pad
and any contiguize); no in-place; the per-output cost is `O(kernel)` FMAs — bandwidth-bound for the
small Mamba kernel size (typically 4).

```fkc
kernel: causal_conv1d
fused_op: CAUSAL_CONV1D
blurb: "Fused depthwise causal 1-D conv (caller left-pads) + per-channel bias + optional fused SiLU; x[batch,channels,seq_in] -> out[batch,channels,seq_out]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::causal_conv1d_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [batch, channels, seq_in]; caller left-padded: seq_in = seq_out + kernel - 1
    - name: weight
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [channels, 1, kernel] depthwise
      shape_constraint: "dim[0]=x"         # channels matches x.dim[1]
    - name: bias
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [channels]
      shape_constraint: "dim[0]=x"         # channels matches x.dim[1]
  op_params:
    variant: CausalConv1d                  # FusedOpParams::CausalConv1d (fused namespace; §3.7)
    fields:
      use_silu: { kind: bool, note: "fuse SiLU(y) = y / (1 + exp(-y)) on the output store (in compute dtype)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(seq_out)     # [batch, channels, seq_out], seq_out = x.dim[2] - (weight.dim[2] - 1); symbolic seq_out preserved
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: conv }
    - { when: "use_silu == false", note: "no SiLU branch on store" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); the formula hints below are the derivable prior
  class: conv
  # Depthwise: kernel FMAs per output element over batch*channels*seq_out outputs => derivable FLOPs.
  # seq_out = x.dim[2] - (weight.dim[2] - 1); kernel = weight.dim[2].
  flops: "2 * x.dim[0] * x.dim[1] * (x.dim[2] - weight.dim[2] + 1) * weight.dim[2]"
  bytes_moved: "(x_elems + out_elems + weight_elems + x.dim[1]) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic nested loop; f32 accumulate for half, f64 for F64
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                          # 2026-07-03 maintainer flip (CireSnave): relocates the CAUSAL_CONV1D_CPU_PRECISION bit-stable claim onto the contract (same author, same guarantee); FKC import is now production — false would DOWNGRADE to UNAUDITED (§4.8/§12.4)
  notes: "depthwise conv + bias + optional fused SiLU; f32/f64 native, bf16/f16 widen to f32 then narrow. Deterministic; not bit-stable cross-hardware (FMA contraction may differ)."

determinism: same_hardware_bitwise
```

---

## selective_scan  (Mamba-1 selective state-space scan, fused, multi-output bundle)

Mamba-1's selective state-space scan forward pass, as a fused op. Five inputs: `u [B, L, dim]`,
`delta [B, L, dim]`, `a [dim, dstate]` (the log-A recurrence matrix), `b [B, L, dstate]`,
`c [B, L, dstate]`. Per timestep, per `(batch, dim)`: `d = softplus(delta)` when `delta_softplus`
else `delta` (stable softplus `max(x,0) + ln(1 + exp(−|x|))`), then the per-state recurrence
`h[b,i,j] = exp(d·a[i,j])·h[b,i,j] + d·b[b,t,j]·u[b,t,i]` and output
`y[b,t,i] = Σ_j h[b,i,j]·c[b,t,j]`. The recurrent hidden state `h [B, dim, dstate]` is allocated
internally, zero-initialized, and threaded across the time axis; it accumulates in **f64 regardless
of element dtype T** (the load-bearing precision invariant), narrowing to T on the stores. The
optional Mamba inputs (`d_skip` / `z` / `delta_bias`) are NOT exposed in v1.

**Multi-output bundle (Option C).** This op emits **one** `KernelRef` whose buffer is the
concatenation `[y_bytes ; last_state_bytes]`, declared via `output_views`: slot 0
`y [B, L, dim]` (= `u`'s shape, the `shape_rule`), slot 1 `last_state [B, dim, dstate]`, both in T,
both contiguous. Consumers project the slots via `Op::View` (§5.5). Forward is
`NotDifferentiable` (v1); `decompose` panics (the O(seqlen) recurrence has no primitive form) —
backends without a native kernel use `cpu_fallback`. Known limitations: contiguous zero-offset
only; no in-place; cost is `O(B·L·dim·dstate)` with the per-state `exp` dominating (sequential
recurrence over the time axis).

```fkc
kernel: selective_scan
fused_op: SELECTIVE_SCAN
blurb: "Fused Mamba-1 selective state-space scan (f64 state accumulator); inputs u,delta,a,b,c; bundled output [y; last_state]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::selective_scan_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: u
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [batch, seqlen, dim]
    - name: delta
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [batch, seqlen, dim]
      shape_constraint: "same_as=u"
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [dim, dstate] (log-A recurrence matrix)
      shape_constraint: "dim[0]=u"         # a.dim[0] (dim) matches u.dim[2]
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [batch, seqlen, dstate]
    - name: c
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [batch, seqlen, dstate]
      shape_constraint: "same_as=b"
  op_params:
    variant: SelectiveScan                 # FusedOpParams::SelectiveScan (fused namespace; §3.7)
    fields:
      delta_softplus: { kind: bool, note: "apply stable softplus(delta) before use" }

return:
  bundle:                                  # multi-output Option C: one buffer [y ; last_state] (output_views, §5.5)
    - { name: y,          dtype_rule: passthrough(u), shape_rule: same_as(u),               layout_guarantee: contiguous }   # [batch, seqlen, dim]
    - { name: last_state, dtype_rule: passthrough(u), shape_rule: from_params(last_state),  layout_guarantee: contiguous }   # [batch, dim, dstate]

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: attention }
    - { when: "delta_softplus == false", note: "no softplus branch per element" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); the formula hints below are the derivable prior
  class: attention
  # Sequential recurrence: per (batch, seqlen, dim, dstate) the inner step does ~1 exp + a few FMAs.
  # FLOPs scale with batch*seqlen*dim*dstate; the transcendental exp dominates => coarse hint only.
  flops: "8 * u.dim[0] * u.dim[1] * u.dim[2] * a.dim[1]"
  bytes_moved: "(2 * u.dim[0] * u.dim[1] * u.dim[2] + a.dim[0] * a.dim[1] + 2 * b.dim[0] * b.dim[1] * b.dim[2] + u.dim[0] * u.dim[1] * u.dim[2] + u.dim[0] * u.dim[2] * a.dim[1]) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "u.dim[0] * u.dim[2] * a.dim[1] * 8", disk_bytes: 0 }   # internal f64 hidden state h

precision:
  bit_stable_on_same_hardware: true      # deterministic sequential scan; f64 hidden-state accumulator
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                          # 2026-07-03 maintainer flip (CireSnave): relocates the SELECTIVE_SCAN_CPU_PRECISION bit-stable claim onto the contract (same author, same guarantee); FKC import is now production — false would DOWNGRADE to UNAUDITED (§4.8/§12.4)
  notes: "f64 hidden-state accumulator regardless of T (narrowed on store); stable softplus. Deterministic; not bit-stable cross-hardware (exp/FMA contraction may differ)."

determinism: same_hardware_bitwise
```

---

## ssd_chunk_scan  (Mamba-2 SSD chunked scan, fused, multi-output bundle)

Mamba-2's State-Space Duality (SSD) chunked scan forward pass, as a fused op. Five inputs:
`x [B, L, H, head_dim]`, `dt [B, L, H]`, `a [H]` (per-head scalar log-A), `b [B, L, H, state_dim]`,
`c [B, L, H, state_dim]`. Produces the scanned sequence `y [B, L, H, head_dim]` (= `x`'s shape, the
`shape_rule`) plus the final per-head state. `chunk_size` is the SSD block size (typically 256 in
Mamba-2): it is a **GPU parallelism granularity knob and does not affect the mathematical result** —
the CPU kernel runs a sequential scan regardless; any `chunk_size ∈ [1, seqlen]` that divides
`seqlen` is valid (`chunk_size > 0`, `seqlen % chunk_size == 0`). The per-head hidden state
accumulates in **f64 regardless of element dtype T**, narrowing to T on store.

**Multi-output bundle (Option C).** One `KernelRef` whose buffer is `[y_bytes ; last_state_bytes]`,
declared via `output_views`: slot 0 `y [B, L, H, head_dim]`, slot 1
`last_state [B, H, head_dim, state_dim]`, both in T, both contiguous (§5.5). Forward is
`NotDifferentiable` (v1); `decompose` panics — backends without a native kernel use `cpu_fallback`.
Known limitations: contiguous zero-offset only; no in-place; `seqlen` must be divisible by
`chunk_size`.

```fkc
kernel: ssd_chunk_scan
fused_op: SSD_CHUNK_SCAN
blurb: "Fused Mamba-2 SSD chunked scan (f64 per-head state accumulator); inputs x,dt,a,b,c; chunk_size is a GPU-parallelism knob; bundled output [y; last_state]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::ssd_chunk_scan_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [batch, seqlen, heads, head_dim]
    - name: dt
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [batch, seqlen, heads]
      shape_constraint: "dim[2]=x"         # heads matches x.dim[2]
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [heads] (per-head scalar log A)
      shape_constraint: "dim[0]=x"         # heads matches x.dim[2]
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [batch, seqlen, heads, state_dim]
    - name: c
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [batch, seqlen, heads, state_dim]
      shape_constraint: "same_as=b"
  op_params:
    variant: SsdChunkScan                  # FusedOpParams::SsdChunkScan (fused namespace; §3.7)
    fields:
      chunk_size: { kind: usize, constraint: "chunk_size > 0; x.dim[1] % chunk_size == 0", note: "SSD block size; GPU parallelism knob only — does NOT change the result (CPU runs sequential)" }

return:
  bundle:                                  # multi-output Option C: one buffer [y ; last_state] (output_views, §5.5)
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
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); the formula hints below are the derivable prior
  class: attention
  # Per (batch, seqlen, heads) the inner work scales with head_dim*state_dim (outer-product state
  # update + output contraction); the transcendental exp on the per-head decay adds a constant.
  flops: "8 * x.dim[0] * x.dim[1] * x.dim[2] * x.dim[3] * b.dim[3]"
  bytes_moved: "(x.dim[0] * x.dim[1] * x.dim[2] * x.dim[3] + x.dim[0] * x.dim[1] * x.dim[2] + x.dim[2] + 2 * b.dim[0] * b.dim[1] * b.dim[2] * b.dim[3] + x.dim[0] * x.dim[1] * x.dim[2] * x.dim[3] + x.dim[0] * x.dim[2] * x.dim[3] * b.dim[3]) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "x.dim[0] * x.dim[2] * x.dim[3] * b.dim[3] * 8", disk_bytes: 0 }   # internal f64 per-head state

precision:
  bit_stable_on_same_hardware: true      # deterministic sequential scan; f64 per-head state accumulator
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                          # 2026-07-03 maintainer flip (CireSnave): relocates the SSD_CHUNK_SCAN_CPU_PRECISION bit-stable claim onto the contract (same author, same guarantee); FKC import is now production — false would DOWNGRADE to UNAUDITED (§4.8/§12.4)
  notes: "f64 per-head state accumulator regardless of T (narrowed on store); chunk_size does not affect the result. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```
