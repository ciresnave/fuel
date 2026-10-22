---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — RoPE (rotary position embedding) family kernel contract

The Vulkan backend's **rotary position embedding** primitive (crate `vulkan`, family `rope`):
`OpKind::Rope` applies the rotation defined by precomputed cos/sin tables to the input `x`
(pointwise per (seq, head-dim) rotation pair). This is the PRIMITIVE `OpKind::Rope` binding
production actually wires — a SEPARATE concern from the aspirational `fused_op: ROPE`
decompositions in `vulkan/conv-attn-rope.fkc.md`.

**As-built binding model — production truth.** RoPE registers as a PRIMITIVE binding keyed
`(OpKind::Rope, [x_dtype, cos_dtype, sin_dtype, out_dtype], Vulkan) + kernel_source` — a **4-slot**
key `[x, cos, sin, out]` (matching the CPU registration's operand shape). FOUR distinct per-dtype
wrappers (`attention::rope_{f32,f16,f64,bf16}` → `VulkanBackend::rope_*` paths). The single section
below fans the BASE `entry_point` over `[F32, F16, F64, BF16]`: `x`, `cos`, and `sin` ALL enumerate
that shared list, so the importer fans them TOGETHER (one fan over the shared list, §3.4 — not a
`FanoutDtypeMismatch`), keying `[T, T, T, T]` byte-for-byte the deleted hand-written
`register_with_caps_and_precision(OpKind::Rope, &[T, T, T, T], …, strided, …)` regs.

**Caps — STRIDE-AWARE (matches the as-built reg).** `rope.slang`'s Params struct carries per-dim `x`
strides + an `x_contiguous` fast-path flag and decomposes the per-thread index into per-dim
coordinates, so the production registrations carry `KernelCaps::strided_input()`. The single
`strided_input` bool the reg set is TRUE (it signals "any input may be non-contiguous"; for RoPE
specifically only `x` is walked strided — the wrapper forces cos/sin contiguous through its own path),
so every operand layout declares `strided: accepted, broadcast_stride0: accepted` to reproduce the
as-built `strided_input == true` (`caps_map` projects the AND across operands). Output is always
freshly-allocated **contiguous**, no aliasing, not in-place.

**Cost provenance.** The cost block is `judge_measured` (§4.4). The bandwidth `bytes_moved` hint is
retained (RoPE is bandwidth-bound pointwise); no overhead constant is fabricated. The imported
`unknown_cost` sentinel is upgraded to the shared OpKind cost fn by `fill_unset_cost_for_backend`.

**Determinism (conservative seed).** RoPE is a deterministic per-thread pointwise rotation (no
atomics, no cross-thread reduction), but it carries the conservative author-seed posture the
elementwise migration set for Vulkan pointwise arithmetic (`audited: true` ⇒
`PrecisionGuarantee::UNAUDITED`; the Judge audits the ULP bound later) rather than re-asserting the
retired hand-written `VULKAN_{FLOAT,HALF}_POINTWISE_PRECISION` consts.

---

## rope  (rotary position embedding; f16/f64/bf16; stride-aware)

Apply the RoPE rotation to `x` using precomputed `cos` / `sin` tables (pointwise per (seq, head-dim)
rotation pair). STRIDE-AWARE on `x` (`rope.slang` walks per-dim `x` strides; cos/sin forced contiguous
by the wrapper). FOUR distinct per-dtype wrappers (`attention::rope_{f32,f16,f64,bf16}`); this section
fans the BASE `entry_point` over `[F16, F64, BF16]` (x + cos + sin share the list; f32 is `rope_f32`
below, which carries a measured accuracy claim). Dispatch key
`(Rope, [T, T, T, T], Vulkan)`.

```fkc
kernel: rope
op_kind: Rope
blurb: "Rotary position embedding (cos/sin table rotation of x); f32/f16/f64/bf16; stride-aware on x; 4-slot [x, cos, sin, out] key."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::rope"   # BASE symbol; fans rope_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F16, F64, BF16]     # fans the per-dtype wrapper (§3.4); cos + sin share the list
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: accepted }
      rank: any
      shape_constraint: "[..., seq, head_dim]; head_dim even (rotation pairs)"
      notes: "stride-aware: rope.slang walks per-dim x strides with an x_contiguous fast path."
    - name: cos
      dtypes: [F16, F64, BF16]     # shares x's list ⇒ fans together (one fan, not a mismatch)
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: accepted }
      rank: any
      shape_constraint: "cos table; wrapper forces contiguous through its own path"
    - name: sin
      dtypes: [F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: accepted }
      rank: any
      shape_constraint: "sin table; wrapper forces contiguous through its own path"
  op_params:
    variant: Rope                 # OpParams::Rope (primitive namespace; §3.7)
    fields:
      seq_len:  { kind: usize, note: "sequence length" }
      head_dim: { kind: usize, note: "rotated head dimension (even)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)           # same dtype as x; key [T, T, T, T]
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: strided         # stride-aware on x; lazy views reach the kernel unmaterialized
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "6 * n"                       # per rotation pair: 4 mul + 2 add-sub (n = element count)
  bytes_moved: "2 * n * dtype_bytes"   # read x + write out (cos/sin table reads amortized)

precision:
  bit_stable_on_same_hardware: false   # author seed (UNAUDITED); pointwise FMA, Judge audits the ULP bound later
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "pointwise cos/sin rotation of x; deterministic per-thread (no atomics). f16/bf16 upcast through f32 for the arithmetic; ULP bound not yet Judge-audited."

determinism: same_hardware_bitwise
```

---

## rope_f32  (rotary position embedding; f32 only; measured accuracy claim)

The f32 binding of the section above, split out so it can carry its own precision block (a fanned
section shares one precision block across its dtypes, and a claim measured for f32 is not true for
f16/bf16). Dispatch key `(Rope, [F32, F32, F32, F32], Vulkan)`.

```fkc
kernel: rope_f32
op_kind: Rope
blurb: "Rotary position embedding (cos/sin table rotation of x); f32/f16/f64/bf16; stride-aware on x; 4-slot [x, cos, sin, out] key."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::rope_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]     # single dtype: the entry_point resolves as-is
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: accepted }
      rank: any
      shape_constraint: "[..., seq, head_dim]; head_dim even (rotation pairs)"
      notes: "stride-aware: rope.slang walks per-dim x strides with an x_contiguous fast path."
    - name: cos
      dtypes: [F32]     # single dtype
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: accepted }
      rank: any
      shape_constraint: "cos table; wrapper forces contiguous through its own path"
    - name: sin
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: accepted }
      rank: any
      shape_constraint: "sin table; wrapper forces contiguous through its own path"
  op_params:
    variant: Rope                 # OpParams::Rope (primitive namespace; §3.7)
    fields:
      seq_len:  { kind: usize, note: "sequence length" }
      head_dim: { kind: usize, note: "rotated head dimension (even)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)           # same dtype as x; key [T, T, T, T]
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: strided         # stride-aware on x; lazy views reach the kernel unmaterialized
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "6 * n"                       # per rotation pair: 4 mul + 2 add-sub (n = element count)
  bytes_moved: "2 * n * dtype_bytes"   # read x + write out (cos/sin table reads amortized)

precision:
  bit_stable_on_same_hardware: false   # pointwise cos/sin rotation, deterministic per thread; bit-stability is not earned in the ledger; repeatability is NOT claimed
  max_ulp: ~
  max_relative: 0.00001                 # MEASURED (board 240, option D); see notes
  max_absolute: 0.00001                 # MEASURED; see notes
  audited: true                         # accuracy claim backed by ledger records (seed_vulkan_matmul_accuracy)
  notes: "pointwise cos/sin rotation of x; deterministic per-thread (no atomics). [evidence: accuracy MEASURED, not derived. The registered kernel was driven through VulkanInvoker over 5 shapes (outer 1-16, seq 1-512, head_dim 64-128; decode seq=1, prefill seq=21), x uniform [-1,1), cos/sin of position*10000^(-2j/hd), against the registered portable-cpu Rope kernel (f32), the oracle the fixtures compare against. Worst |out-cpu| = 1.19e-7 (1 ulp at 1.0; 0 for the seq=1 decode cases); claim 1e-5 is ~84x headroom (ledger evidence, fuel-dev-tools seed-vulkan-accuracy, RTX 4070 Laptop). Not evidence about other inputs, other kernels in the family (f16/bf16/f64), or shapes beyond the listed ones. The same cases on the integrated AMD Radeon 610M adapter also stayed inside the claim (fuel-dev-tools check-vulkan-accuracy with FUEL_VULKAN_TEST_ADAPTER=AMD; not recorded in the ledger, which has no adapter field). Repeatability (bit_stable) is separately NOT claimed. The claim is what lets the default placement rule consider this kernel against the CPU kernel on cost.]"

determinism: same_hardware_bitwise
```
