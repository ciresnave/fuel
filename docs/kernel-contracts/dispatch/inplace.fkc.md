---
fkc_version: 1
provider:
  name: fuel-dispatch
  backend: Cpu                       # default backend for the blocks below (CPU is the always-built
                                     # fallback). Each in-place OpKind here also has an IDENTICAL
                                     # contiguous-only baracuda-CUDA registration (kernel_source
                                     # "baracuda", backend Cuda) — see the "CUDA sibling" note in
                                     # each section; per the §3.1 front-matter override rule the CU
                                     # twin is the same contract with backend/kernel_source/entry_point
                                     # swapped.
  kernel_source: "portable-cpu"      # the BindingEntry.kernel_source tag (CU twin: "baracuda")
  link_registry: fuel_dispatch::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-dispatch — in-place elementwise-unary / affine / clamp / powi kernel contracts

The `fuel-dispatch` registrations for the **in-place scalar-param** elementwise op family: the
in-place affine (`InplaceAffine`), the in-place clamp / powi (`ClampInplace`, `PowIInplace`), the
in-place unary activations (`ReluInplace`, `SiluInplace`, `GeluInplace`, `TanhInplace`,
`SigmoidInplace`), and the 16-op in-place unary family (`NegInplace`, `AbsInplace`, `SqrInplace`,
`SqrtInplace`, `RsqrtInplace`, `RecipInplace`, `ExpInplace`, `LogInplace`, `SinInplace`,
`CosInplace`, `SignInplace`, `FloorInplace`, `CeilInplace`, `RoundInplace`, `ErfInplace`,
`GeluErfInplace`). Every kernel in this bundle mutates a **single** buffer in place — the output
**IS** the target Storage (passed as `outputs[0]`), and the CPU wrapper **rejects a non-empty
`inputs` list** (the dispatch ABI for the in-place arm: no out-of-line input, `out[i] = op(out[i])`).

**Family-wide facts (each section overrides where its inventory entry differs):**

- **Both backends, both contiguous-only.** Each `OpKind` here is registered on **CPU**
  (`register_cpu_kernels`, `dispatch.rs`) *and* on **baracuda CUDA**
  (`register_baracuda_cuda_kernels`, `baracuda_dispatch.rs`). Unlike the non-in-place CUDA unary
  family (which registers `strided_input`), the **in-place** CUDA arm registers **contiguous-only**
  (no strided cap) — the executor rejects strided in-place targets up front, so neither backend
  walks strides/offsets here (inventory: "In-place affine … CU C (no strided cap — executor rejects
  strided in-place targets up front)"). The contract block in each section is the CPU twin; the
  CUDA sibling is the same contract with `backend: Cuda`, `kernel_source: "baracuda"`, and the
  baracuda `entry_point` (cited per section). The dispatch **key** (`[T, T]`) and `KernelCaps`
  (default / contiguous-only) are identical between the two.
- **`reverse_strides: rejected` everywhere in this file.** No in-place kernel walks a signed
  (negative) stride; a flipped view feeding one is normalized to a non-negative contiguous copy by
  an upstream movement kernel before dispatch (and, as below, every awkward layout is contiguized
  first regardless).
- **Contiguous-only, single buffer, full positional overwrite.** No kernel reads a `Layout` /
  strides / offset — the CPU wrappers take `_layouts` UNUSED and operate on raw bytes
  (`CpuStorageBytes`); the executor's auto-Contiguize pass realizes any strided / broadcast /
  non-zero-offset target into a contiguous, zero-offset buffer before these kernels run. NO kernel
  in this crate is offset-capable (inventory cross-cutting note), so the planner inserts an
  `Op::Contiguize` (itself an FKC kernel) for any non-contiguous target and **sums its cost** (§4.3
  / §4.4). The element count is carried implicitly by the buffer byte length (validated against the
  output Storage's `dtype` width); geometry comes from `OpParams`, never from a `Shape` argument.
- **Half via f32; numerics match the non-in-place cousins.** f32/f64 evaluate natively; bf16/f16
  widen to f32, do the math, narrow on store. CPU half routes through the f32-pivot blanket impls
  and **bit-matches the non-in-place kernel** (inventory: "CPU half routes through f32-pivot blanket
  impls (bit-matches non-inplace)").
- **dtypes: F32, F64, BF16, F16** for every kernel here (key `[T, T]`); one `entry_point` per
  `(op, dtype)` (the cited symbol is the f32 representative — §12.6).
- **In-place return-contract is uniform:** `dtype_rule: passthrough(out)`, `shape_rule:
  same_as(out)` (symbolic extents carry through, §5.2), `layout_guarantee: contiguous`, and
  `aliasing: in_place(out)` with `caps.in_place: true` (§4.6 / §5.4) — the output IS the input
  buffer, so the executor allocates **no** new output (`memory.device_bytes: 0`).
- **Cost is `judge_measured` for every kernel in this file** — the Judge bootstraps and refines the
  empirical coefficients (§4.4). The only genuinely op-derivable structure is recorded as a hint:
  these are **bandwidth-bound elementwise** ops touching `n` elements with **one read + one write of
  the single in-place buffer** (`bytes_moved ≈ 2·n·dtype_bytes`), with `flops ≈ n` (or `2·n` for the
  affine multiply-add). `overhead_ns` and any absolute timing are left null for the Judge.
  `provenance: judge_measured` is a first-class, visible marker (§4.4) — not a placeholder gap.
- **Precision (CPU):** deterministic positional loops, `bit_stable_on_same_hardware: true`,
  `audited: false` — the importer applies the CPU family default `PRIMITIVE_DETERMINISTIC_CPU`
  (§12.4 / `fill_unset_cpu_precision`). The CUDA sibling's precision is the per-op baracuda
  guarantee (left to the baracuda contract's own bulk-fill, not asserted here).

---

## inplace_affine  (out = mul*out + add, in place)

In-place affine over a single contiguous buffer: `out[i] = mul * out[i] + add`. The target is
passed as `outputs[0]` and the wrapper **rejects a non-empty `inputs`** list (in-place ABI); it
mutates the buffer in place. `mul`/`add` are scalar params from `OpParams::Affine` (carried as f64;
the half/f32 paths narrow to f32). Covers in-place AddScalar (`mul == 1`) and MulScalar
(`add == 0`). One multiply + one add per element, IEEE inf/NaN. Contiguous-only on both backends
(CU registers no strided cap — the executor rejects strided in-place targets). f32/f64 native;
bf16/f16 widen to f32, mul-add in f32, narrow on store. **CUDA sibling:** `backend: Cuda`,
`kernel_source: "baracuda"`, baracuda_dispatch.rs:2669 (key `[T, T]`, contiguous-only). Source:
dispatch.rs:4108 (CPU).

```fkc
kernel: inplace_affine
op_kind: InplaceAffine
blurb: "In-place affine out[i]=mul*out[i]+add over a single contiguous buffer; half via f32; rejects non-empty inputs."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::inplace_affine_f32"   # one per (op,dtype); §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: out                 # the SINGLE in-place buffer (outputs[0]); inputs list is empty
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Affine             # OpParams::Affine { mul: f64, add: f64 }
    fields:
      mul: { kind: f64, note: "half/f32 paths narrow the param to f32" }
      add: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)     # in-place: output dtype == target dtype
      shape_rule: same_as(out)         # shape preserved; symbolic extents carry through
      layout_guarantee: contiguous
      aliasing: in_place(out)          # output IS the target buffer (caps.in_place: true)

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true                # mutates its single buffer (§4.6)
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured    # the Judge bootstraps/calibrates the coefficients (§4.4)
  class: cheap_elementwise
  # FLOPs/bandwidth hint (op-derivable): bandwidth-bound elementwise — read+write the single buffer.
  flops: "2 * n"                # one multiply + one add per element
  bytes_moved: "2 * n * dtype_bytes"   # read out, write out (in-place)
  overhead_ns: ~                # judge_measured
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # in-place: no new alloc

precision:
  bit_stable_on_same_hardware: true    # deterministic positional loop
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                # CPU primitive: importer applies PRIMITIVE_DETERMINISTIC_CPU (§12.4)
  notes: "Native f32/f64 mul-add; bf16/f16 widen to f32 then narrow. IEEE inf/NaN. mul==1 ⇒ AddScalar; add==0 ⇒ MulScalar. CUDA sibling: baracuda_dispatch.rs:2669, contiguous-only."

determinism: same_hardware_bitwise
```

## clamp_inplace  (out = clamp(out, min, max), in place)

In-place clamp over a single contiguous buffer: `out[i] = clamp(out[i], min, max)`. The target is
`outputs[0]`; the buffer is fully overwritten in place. `min`/`max` are scalar bounds from
`OpParams::Clamp` (carried f64; half/f32 paths narrow to f32). `min > max` is a hard precondition
(the in-place cousin in `fuel-cpu-backend` returns a typed `Result` error rather than a clamp-flip
or panic). f32/f64 native; bf16/f16 widen to f32, clamp in f32, narrow on store. Contiguous-only on
both backends. **CUDA sibling:** `backend: Cuda`, `kernel_source: "baracuda"`,
baracuda_dispatch.rs:2679 (key `[T, T]`, contiguous-only). Source: dispatch.rs:4116 (CPU).

```fkc
kernel: clamp_inplace
op_kind: ClampInplace
blurb: "In-place clamp out[i]=clamp(out[i],min,max) over a single contiguous buffer; half via f32; rejects min>max."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::clamp_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Clamp             # OpParams::Clamp { min: f64, max: f64 }
    fields:
      min: { kind: f64, constraint: "min <= max", note: "min > max returns a typed Error (no panic); half/f32 paths narrow to f32" }
      max: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                   # one clamp (two compares) per element
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Native f32/f64 clamp; bf16/f16 widen to f32, clamp against f32-narrowed bounds, narrow on store. min>max ⇒ typed Error (no panic). CUDA sibling: baracuda_dispatch.rs:2679, contiguous-only."

determinism: same_hardware_bitwise
```

## powi_inplace  (out = out.powi(exp), in place)

In-place integer power over a single contiguous buffer: `out[i] = out[i].powi(exp)`. The target is
`outputs[0]`; the buffer is fully overwritten in place. `exp` is an `i32` from `OpParams::PowI`;
native `powi` semantics (repeated multiplication, `exp == 0 ⇒ 1.0`, negative `exp` ⇒ reciprocal,
IEEE inf/NaN). f32/f64 native; bf16/f16 widen to f32, raise to the power in f32, narrow on store.
Contiguous-only on both backends. **CUDA sibling:** `backend: Cuda`, `kernel_source: "baracuda"`,
baracuda_dispatch.rs:2684 (key `[T, T]`, contiguous-only). Source: dispatch.rs:4121 (CPU).

```fkc
kernel: powi_inplace
op_kind: PowIInplace
blurb: "In-place integer power out[i]=out[i].powi(exp) over a single contiguous buffer; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::powi_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: PowI              # OpParams::PowI { exp: i32 }
    fields:
      exp: { kind: i32 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  # powi cost scales with the exponent's bit-length (square-and-multiply); the per-element op
  # count is not a fixed constant, so only n (element count) is given.
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Native f32/f64 powi (i32 exponent); bf16/f16 widen to f32, powi, narrow on store. exp==0 ⇒ 1.0; negative exp ⇒ reciprocal; IEEE inf/NaN. CUDA sibling: baracuda_dispatch.rs:2684, contiguous-only."

determinism: same_hardware_bitwise
```

## relu_inplace  (out = max(out, 0), in place)

In-place ReLU over a single contiguous buffer: `out[i] = max(out[i], 0)`. The target is
`outputs[0]`; `op_params` is `None`. CPU half routes through f32-pivot blanket impls and bit-matches
the non-in-place `ReluElementwise`. f32/f64 native; bf16/f16 widen to f32, narrow on store.
Contiguous-only on both backends. **CUDA sibling:** `backend: Cuda`, `kernel_source: "baracuda"`,
baracuda_dispatch.rs:2693 (key `[T, T]`, contiguous-only). Source: dispatch.rs:4213 (CPU).

```fkc
kernel: relu_inplace
op_kind: ReluInplace
blurb: "In-place ReLU out[i]=max(out[i],0) over a single contiguous buffer; half via f32; matches non-in-place numerics."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::relu_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }   # OpParams::None — no auxiliary scalar params

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Exact (max with 0); bf16/f16 widen to f32, narrow on store. Bit-matches the non-in-place ReluElementwise. CUDA sibling: baracuda_dispatch.rs:2693, contiguous-only."

determinism: same_hardware_bitwise
```

## silu_inplace  (out = out * sigmoid(out), in place)

In-place SiLU (swish) over a single contiguous buffer: `out[i] = out[i] * sigmoid(out[i])`. The
target is `outputs[0]`; `op_params` is `None`. CPU half routes through f32-pivot blanket impls and
bit-matches the non-in-place `SiluElementwise`. f32/f64 native; bf16/f16 widen to f32, narrow on
store. Transcendental (a `sigmoid` per element) — not bit-stable across hardware, but deterministic
per hardware on CPU. Contiguous-only on both backends. **CUDA sibling:** `backend: Cuda`,
`kernel_source: "baracuda"`, baracuda_dispatch.rs:2693 (key `[T, T]`, contiguous-only). Source:
dispatch.rs:4213 (CPU).

```fkc
kernel: silu_inplace
op_kind: SiluInplace
blurb: "In-place SiLU out[i]=out[i]*sigmoid(out[i]) over a single contiguous buffer; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::silu_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                   # one transcendental sigmoid + one multiply per element
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x*sigmoid(x), f32 math; bf16/f16 widen to f32, narrow on store. Bit-matches the non-in-place SiluElementwise. CUDA sibling: baracuda_dispatch.rs:2693, contiguous-only."

determinism: same_hardware_bitwise
```

## gelu_inplace  (out = gelu_tanh(out), in place)

In-place GELU (**tanh** approximation) over a single contiguous buffer:
`out[i] = 0.5·out[i]·(1 + tanh(√(2/π)·(out[i] + 0.044715·out[i]³)))`. The target is `outputs[0]`;
`op_params` is `None`. **This is the tanh-approximated GELU, NOT erf** (the erf flavor is
`GeluErfInplace`, below) — matching the non-in-place `GeluElementwise` (inventory: "`GeluElementwise`
is the **tanh** approximation"; CU binds the baracuda `unary_gelu_tanh_*` family). CPU half routes
through f32-pivot blanket impls and bit-matches the non-in-place cousin. f32/f64 native; bf16/f16
widen to f32, narrow on store. Contiguous-only on both backends. **CUDA sibling:** `backend: Cuda`,
`kernel_source: "baracuda"`, baracuda_dispatch.rs:2693 (key `[T, T]`, contiguous-only). Source:
dispatch.rs:4213 (CPU).

```fkc
kernel: gelu_inplace
op_kind: GeluInplace
blurb: "In-place GELU (tanh approximation) over a single contiguous buffer; half via f32; NOT the erf flavor."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::gelu_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                   # one tanh-GELU evaluation per element
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "TANH-approximated GELU (NOT erf — see gelu_erf_inplace). f32 math; bf16/f16 widen to f32, narrow on store. Bit-matches the non-in-place GeluElementwise. CUDA sibling: baracuda unary_gelu_tanh_*, baracuda_dispatch.rs:2693, contiguous-only."

determinism: same_hardware_bitwise
```

## tanh_inplace  (out = tanh(out), in place)

In-place hyperbolic tangent over a single contiguous buffer: `out[i] = tanh(out[i])`. The target is
`outputs[0]`; `op_params` is `None`. CPU half routes through f32-pivot blanket impls and bit-matches
the non-in-place `TanhElementwise`. f32/f64 native; bf16/f16 widen to f32, narrow on store.
Contiguous-only on both backends. **CUDA sibling:** `backend: Cuda`, `kernel_source: "baracuda"`,
baracuda_dispatch.rs:2693 (key `[T, T]`, contiguous-only). Source: dispatch.rs:4213 (CPU).

```fkc
kernel: tanh_inplace
op_kind: TanhInplace
blurb: "In-place tanh out[i]=tanh(out[i]) over a single contiguous buffer; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::tanh_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "tanh, f32 math; bf16/f16 widen to f32, narrow on store. Bit-matches the non-in-place TanhElementwise. CUDA sibling: baracuda_dispatch.rs:2693, contiguous-only."

determinism: same_hardware_bitwise
```

## sigmoid_inplace  (out = 1/(1+exp(-out)), in place)

In-place logistic sigmoid over a single contiguous buffer: `out[i] = 1 / (1 + exp(-out[i]))`. The
target is `outputs[0]`; `op_params` is `None`. CPU half routes through f32-pivot blanket impls and
bit-matches the non-in-place `SigmoidElementwise`. f32/f64 native; bf16/f16 widen to f32, narrow on
store. Contiguous-only on both backends. **CUDA sibling:** `backend: Cuda`,
`kernel_source: "baracuda"`, baracuda_dispatch.rs:2693 (key `[T, T]`, contiguous-only). Source:
dispatch.rs:4213 (CPU).

```fkc
kernel: sigmoid_inplace
op_kind: SigmoidInplace
blurb: "In-place sigmoid out[i]=1/(1+exp(-out[i])) over a single contiguous buffer; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::sigmoid_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "logistic sigmoid, f32 math; bf16/f16 widen to f32, narrow on store. Bit-matches the non-in-place SigmoidElementwise. CUDA sibling: baracuda_dispatch.rs:2693, contiguous-only."

determinism: same_hardware_bitwise
```

## neg_inplace  (out = -out, in place)

In-place negation over a single contiguous buffer: `out[i] = -out[i]`. The target is `outputs[0]`;
`op_params` is `None`. Exact (sign flip, IEEE-correct on signed zero / NaN). f32/f64 native; bf16/f16
widen to f32, narrow on store (the narrow is exact for negation). Contiguous-only on both backends.
**CUDA sibling:** `backend: Cuda`, `kernel_source: "baracuda"`, baracuda_dispatch.rs:2719 (CU
in-place unary loop; key `[T, T]`, contiguous-only). Source: dispatch.rs:4239 (CPU in-place unary
loop).

```fkc
kernel: neg_inplace
op_kind: NegInplace
blurb: "In-place negation out[i]=-out[i] over a single contiguous buffer; exact."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::neg_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Exact sign flip (IEEE signed zero / NaN). bf16/f16 negation is exact through f32. Bit-matches the non-in-place NegElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## abs_inplace  (out = |out|, in place)

In-place absolute value over a single contiguous buffer: `out[i] = |out[i]|`. The target is
`outputs[0]`; `op_params` is `None`. Exact (magnitude, IEEE-correct on NaN). f32/f64 native; bf16/f16
widen to f32, narrow on store (exact for abs). Contiguous-only on both backends. **CUDA sibling:**
`backend: Cuda`, `kernel_source: "baracuda"`, baracuda_dispatch.rs:2719 (CU in-place unary loop;
key `[T, T]`, contiguous-only). Source: dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: abs_inplace
op_kind: AbsInplace
blurb: "In-place absolute value out[i]=|out[i]| over a single contiguous buffer; exact."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::abs_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Exact magnitude (IEEE NaN). bf16/f16 abs is exact through f32. Bit-matches the non-in-place AbsElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## sqr_inplace  (out = out*out, in place)

In-place square over a single contiguous buffer: `out[i] = out[i] * out[i]`. The target is
`outputs[0]`; `op_params` is `None`. One multiply per element, IEEE inf/NaN. f32/f64 native; bf16/f16
widen to f32, square in f32, narrow on store. Contiguous-only on both backends. **CUDA sibling:**
`backend: Cuda`, `kernel_source: "baracuda"`, baracuda_dispatch.rs:2719 (CU in-place unary loop;
key `[T, T]`, contiguous-only). Source: dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: sqr_inplace
op_kind: SqrInplace
blurb: "In-place square out[i]=out[i]*out[i] over a single contiguous buffer; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::sqr_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                   # one multiply per element
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x*x, f32 math; bf16/f16 widen to f32, narrow on store. IEEE inf/NaN. Bit-matches the non-in-place SqrElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## sqrt_inplace  (out = sqrt(out), in place)

In-place square root over a single contiguous buffer: `out[i] = sqrt(out[i])`. The target is
`outputs[0]`; `op_params` is `None`. IEEE `sqrt` (correctly-rounded; `sqrt(neg) = NaN`,
`sqrt(-0) = -0`). f32/f64 native; bf16/f16 widen to f32, sqrt in f32, narrow on store.
Contiguous-only on both backends. **CUDA sibling:** `backend: Cuda`, `kernel_source: "baracuda"`,
baracuda_dispatch.rs:2719 (CU in-place unary loop; key `[T, T]`, contiguous-only). Source:
dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: sqrt_inplace
op_kind: SqrtInplace
blurb: "In-place square root out[i]=sqrt(out[i]) over a single contiguous buffer; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::sqrt_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "IEEE sqrt (correctly rounded; sqrt(neg)=NaN, sqrt(-0)=-0). bf16/f16 widen to f32, narrow on store. Bit-matches the non-in-place SqrtElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## rsqrt_inplace  (out = 1/sqrt(out), in place)

In-place reciprocal square root over a single contiguous buffer: `out[i] = 1 / sqrt(out[i])`. The
target is `outputs[0]`; `op_params` is `None`. f32/f64 native; bf16/f16 widen to f32, rsqrt in f32,
narrow on store. Contiguous-only on both backends. **CUDA sibling:** `backend: Cuda`,
`kernel_source: "baracuda"`, baracuda_dispatch.rs:2719 (CU in-place unary loop; key `[T, T]`,
contiguous-only). Source: dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: rsqrt_inplace
op_kind: RsqrtInplace
blurb: "In-place reciprocal square root out[i]=1/sqrt(out[i]) over a single contiguous buffer; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::rsqrt_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "1/sqrt(x), f32 math; bf16/f16 widen to f32, narrow on store. Bit-matches the non-in-place RsqrtElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## recip_inplace  (out = 1/out, in place)

In-place reciprocal over a single contiguous buffer: `out[i] = 1 / out[i]`. The target is
`outputs[0]`; `op_params` is `None`. IEEE division (`1/0 = inf`, `1/inf = 0`). f32/f64 native;
bf16/f16 widen to f32, reciprocal in f32, narrow on store. Contiguous-only on both backends. **CUDA
sibling:** `backend: Cuda`, `kernel_source: "baracuda"`, baracuda_dispatch.rs:2719 (CU in-place
unary loop; key `[T, T]`, contiguous-only). Source: dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: recip_inplace
op_kind: RecipInplace
blurb: "In-place reciprocal out[i]=1/out[i] over a single contiguous buffer; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::recip_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "IEEE 1/x (1/0=inf, 1/inf=0). f32 math; bf16/f16 widen to f32, narrow on store. Bit-matches the non-in-place RecipElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## exp_inplace  (out = exp(out), in place)

In-place natural exponential over a single contiguous buffer: `out[i] = exp(out[i])`. The target is
`outputs[0]`; `op_params` is `None`. f32/f64 native; bf16/f16 widen to f32, exp in f32, narrow on
store. Contiguous-only on both backends. **CUDA sibling:** `backend: Cuda`,
`kernel_source: "baracuda"`, baracuda_dispatch.rs:2719 (CU in-place unary loop; key `[T, T]`,
contiguous-only). Source: dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: exp_inplace
op_kind: ExpInplace
blurb: "In-place exp out[i]=exp(out[i]) over a single contiguous buffer; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::exp_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "exp(x), f32 math; bf16/f16 widen to f32, narrow on store. Bit-matches the non-in-place ExpElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## log_inplace  (out = ln(out), in place)

In-place natural logarithm over a single contiguous buffer: `out[i] = ln(out[i])`. The target is
`outputs[0]`; `op_params` is `None`. IEEE log (`ln(0) = -inf`, `ln(neg) = NaN`). f32/f64 native;
bf16/f16 widen to f32, log in f32, narrow on store. Contiguous-only on both backends. **CUDA
sibling:** `backend: Cuda`, `kernel_source: "baracuda"`, baracuda_dispatch.rs:2719 (CU in-place
unary loop; key `[T, T]`, contiguous-only). Source: dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: log_inplace
op_kind: LogInplace
blurb: "In-place natural log out[i]=ln(out[i]) over a single contiguous buffer; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::log_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "ln(x), f32 math (ln(0)=-inf, ln(neg)=NaN); bf16/f16 widen to f32, narrow on store. Bit-matches the non-in-place LogElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## sin_inplace  (out = sin(out), in place)

In-place sine over a single contiguous buffer: `out[i] = sin(out[i])`. The target is `outputs[0]`;
`op_params` is `None`. f32/f64 native; bf16/f16 widen to f32, sin in f32, narrow on store.
Contiguous-only on both backends. **CUDA sibling:** `backend: Cuda`, `kernel_source: "baracuda"`,
baracuda_dispatch.rs:2719 (CU in-place unary loop; key `[T, T]`, contiguous-only). Source:
dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: sin_inplace
op_kind: SinInplace
blurb: "In-place sine out[i]=sin(out[i]) over a single contiguous buffer; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::sin_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "sin(x), f32 math; bf16/f16 widen to f32, narrow on store. Bit-matches the non-in-place SinElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## cos_inplace  (out = cos(out), in place)

In-place cosine over a single contiguous buffer: `out[i] = cos(out[i])`. The target is `outputs[0]`;
`op_params` is `None`. f32/f64 native; bf16/f16 widen to f32, cos in f32, narrow on store.
Contiguous-only on both backends. **CUDA sibling:** `backend: Cuda`, `kernel_source: "baracuda"`,
baracuda_dispatch.rs:2719 (CU in-place unary loop; key `[T, T]`, contiguous-only). Source:
dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: cos_inplace
op_kind: CosInplace
blurb: "In-place cosine out[i]=cos(out[i]) over a single contiguous buffer; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::cos_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "cos(x), f32 math; bf16/f16 widen to f32, narrow on store. Bit-matches the non-in-place CosElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## sign_inplace  (out = sign(out), in place)

In-place sign over a single contiguous buffer: `out[i] = sign(out[i])` with `sign(0) = 0`,
`sign(x>0) = 1`, `sign(x<0) = -1`. The target is `outputs[0]`; `op_params` is `None`. Exact (yields
−1/0/1). f32/f64 native; bf16/f16 widen to f32, narrow on store (exact). Contiguous-only on both
backends. **CUDA sibling:** `backend: Cuda`, `kernel_source: "baracuda"`,
baracuda_dispatch.rs:2719 (CU in-place unary loop; key `[T, T]`, contiguous-only). Source:
dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: sign_inplace
op_kind: SignInplace
blurb: "In-place sign out[i]=sign(out[i]) (sign(0)=0) over a single contiguous buffer; exact."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::sign_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Exact sign: sign(0)=0, +1 for x>0, -1 for x<0. bf16/f16 exact through f32. Bit-matches the non-in-place SignElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## floor_inplace  (out = floor(out), in place)

In-place floor over a single contiguous buffer: `out[i] = floor(out[i])` (round toward −inf). The
target is `outputs[0]`; `op_params` is `None`. Exact (integer-valued result, IEEE inf/NaN passthrough).
f32/f64 native; bf16/f16 widen to f32, floor in f32, narrow on store (exact). Contiguous-only on
both backends. **CUDA sibling:** `backend: Cuda`, `kernel_source: "baracuda"`,
baracuda_dispatch.rs:2719 (CU in-place unary loop; key `[T, T]`, contiguous-only). Source:
dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: floor_inplace
op_kind: FloorInplace
blurb: "In-place floor out[i]=floor(out[i]) (toward -inf) over a single contiguous buffer; exact."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::floor_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Exact floor (toward -inf); IEEE inf/NaN passthrough. bf16/f16 exact through f32. Bit-matches the non-in-place FloorElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## ceil_inplace  (out = ceil(out), in place)

In-place ceiling over a single contiguous buffer: `out[i] = ceil(out[i])` (round toward +inf). The
target is `outputs[0]`; `op_params` is `None`. Exact (integer-valued result, IEEE inf/NaN passthrough).
f32/f64 native; bf16/f16 widen to f32, ceil in f32, narrow on store (exact). Contiguous-only on both
backends. **CUDA sibling:** `backend: Cuda`, `kernel_source: "baracuda"`,
baracuda_dispatch.rs:2719 (CU in-place unary loop; key `[T, T]`, contiguous-only). Source:
dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: ceil_inplace
op_kind: CeilInplace
blurb: "In-place ceiling out[i]=ceil(out[i]) (toward +inf) over a single contiguous buffer; exact."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::ceil_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Exact ceil (toward +inf); IEEE inf/NaN passthrough. bf16/f16 exact through f32. Bit-matches the non-in-place CeilElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## round_inplace  (out = round_ties_even(out), in place)

In-place rounding over a single contiguous buffer: `out[i] = round_ties_even(out[i])` (banker's
rounding — round half to even, matching the non-in-place `RoundElementwise`: CPU `round_ties_even`,
CU `rint`). The target is `outputs[0]`; `op_params` is `None`. Exact (integer-valued result, IEEE
inf/NaN passthrough). f32/f64 native; bf16/f16 widen to f32, round in f32, narrow on store.
Contiguous-only on both backends. **CUDA sibling:** `backend: Cuda`, `kernel_source: "baracuda"`,
baracuda_dispatch.rs:2719 (CU in-place unary loop, `rint`; key `[T, T]`, contiguous-only). Source:
dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: round_inplace
op_kind: RoundInplace
blurb: "In-place banker's rounding out[i]=round_ties_even(out[i]) over a single contiguous buffer; exact."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::round_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Banker's rounding (ties to even); IEEE inf/NaN passthrough. CPU round_ties_even, CU rint. bf16/f16 exact through f32. Bit-matches the non-in-place RoundElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## erf_inplace  (out = erf(out), in place)

In-place error function over a single contiguous buffer: `out[i] = erf(out[i])` (plain error
function, CPU `libm::erf{,f}`, matching the non-in-place `ErfElementwise`). The target is
`outputs[0]`; `op_params` is `None`. f32/f64 native; bf16/f16 widen to f32, erf in f32, narrow on
store. Contiguous-only on both backends. **CUDA sibling:** `backend: Cuda`,
`kernel_source: "baracuda"`, baracuda_dispatch.rs:2719 (CU in-place unary loop; key `[T, T]`,
contiguous-only). Source: dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: erf_inplace
op_kind: ErfInplace
blurb: "In-place error function out[i]=erf(out[i]) over a single contiguous buffer; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::erf_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Plain error function erf(x) via libm; f32 math; bf16/f16 widen to f32, narrow on store. Bit-matches the non-in-place ErfElementwise. CUDA sibling: baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```

## gelu_erf_inplace  (out = gelu_erf(out), in place)

In-place GELU (**erf** flavor) over a single contiguous buffer:
`out[i] = 0.5·out[i]·(1 + erf(out[i]/√2))`. The target is `outputs[0]`; `op_params` is `None`. **This
is the erf-exact GELU, NOT the tanh approximation** (the tanh flavor is `GeluInplace`, above) —
matching the non-in-place `GeluErfElementwise` (inventory: "`GeluErf` = erf-flavored gelu"; CPU via
`libm::erf`). CPU half routes through f32-pivot blanket impls and bit-matches the non-in-place
cousin. f32/f64 native; bf16/f16 widen to f32, narrow on store. Contiguous-only on both backends.
**CUDA sibling:** `backend: Cuda`, `kernel_source: "baracuda"`, baracuda_dispatch.rs:2719 (CU
in-place unary loop, baracuda `unary_gelu_*` erf flavor; key `[T, T]`, contiguous-only). Source:
dispatch.rs:4239 (CPU in-place unary loop).

```fkc
kernel: gelu_erf_inplace
op_kind: GeluErfInplace
blurb: "In-place GELU (erf flavor) over a single contiguous buffer; half via f32; NOT the tanh approximation."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::gelu_erf_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                   # one erf-GELU evaluation per element
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "ERF-flavored GELU (NOT tanh — see gelu_inplace): 0.5*x*(1+erf(x/sqrt2)) via libm. f32 math; bf16/f16 widen to f32, narrow on store. Bit-matches the non-in-place GeluErfElementwise. CUDA sibling: baracuda unary_gelu_*, baracuda_dispatch.rs:2719, contiguous-only."

determinism: same_hardware_bitwise
```
