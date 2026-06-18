---
fkc_version: 1
provider:
  name: fuel-dispatch
  backend: Cpu
  kernel_source: "portable-cpu"
  link_registry: fuel_dispatch::fkc::ENTRY_POINTS   # §12.6 symbol → KernelRef map
  revision_base: "git:f41137b4"                     # provider build id, folded into kernel_revision_hash
---

# fuel-dispatch — conv-attn family kernel contracts

The convolution + attention + structured-state-space ops Fuel's dispatch layer registers
directly onto the `KernelBindingTable` (the primitive `op_kind` surface — `OpParams` carriers,
not the fused-graph registry). Source of truth: `docs/kernel-contracts/_inventory/dispatch.md`
and the as-built `fuel-dispatch/src/{kernel.rs, dispatch.rs, baracuda_dispatch.rs}`.

Family-wide as-built facts (apply to every kernel below unless overridden):

- **Every CPU wrapper is contiguous-only and not offset-capable** — it takes `_layouts` UNUSED
  and operates on raw byte buffers (`CpuStorageBytes`), relying entirely on the executor's
  auto-Contiguize pass. Geometry comes from `OpParams`; spatial shapes never flow through the
  Layout. So every operand here declares `awkward_layout_strategy: requires_contiguous` and the
  full five-flag layout set is `contiguous: required` + everything else `rejected`. No kernel in
  this crate accepts `reverse_strides`, `broadcast_stride0`, or a non-zero `start_offset`.
- **Output Storage is ALWAYS pre-allocated by the executor; no kernel allocates.** The wrapper
  writes into the pre-allocated bytes (`layout_guarantee: contiguous` / `preallocated`). Output
  dtype = the last entry of the `(op, dtypes)` key unless the op pins a fixed dtype.
- **half precision (bf16/f16) accumulates in f32** on CPU and narrows on store; CPU precision is
  bulk-upgraded to `PRIMITIVE_DETERMINISTIC_CPU` (`bit_stable_on_same_hardware: true`).
- **Cost is `provenance: declared`** — author priors the Judge later refines/bootstraps. FLOPs /
  bytes formula hints are given only where genuinely derivable from the op; the rest carry the
  formula where derivable and `judge_measured`-grade refinement is left to the Judge. No cost is a
  hidden gap (§4.4 / §10.8a).

---

## conv2d  (2D cross-correlation, NCHW, asymmetric stride/pad/dilation, grouped)

Direct 2D convolution (cross-correlation, PyTorch convention). `x [N, Cin, Hin, Win]`,
`weight [Cout, Cin/groups, Kh, Kw]`, optional 3rd input `bias [Cout]`; output
`[N, Cout, Hout, Wout]`. Asymmetric `stride`/`padding`/`dilation` and `groups` are all carried in
`OpParams::Conv2D` (the spatial shapes too, since byte Storage holds no geometry). f32/f64
evaluate natively; bf16/f16 accumulate the inner-product over the receptive field in f32 and
narrow on store. Registered twice per dtype: no-bias `[x, w, out]` and with-bias `[x, w, bias, out]`.
CPU-only in the binding table. Known limitations: contiguous NCHW packed input only (planner
inserts `Op::Contiguize` for any strided view); no implicit channel-last; `groups` must divide
both Cin and Cout.

```fkc
kernel: conv2d
op_kind: Conv2D
blurb: "2D cross-correlation, NCHW, asymmetric stride/pad/dilation, grouped; contiguous; half accum f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::conv2d_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [N, Cin, Hin, Win]
    - name: weight
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [Cout, Cin/groups, Kh, Kw]
      shape_constraint: "same_dtype=x"
    - name: bias                    # optional; presence implicit in inputs.len()==3
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # [Cout]
      optional: true
  op_params:
    variant: Conv2D                 # OpParams::Conv2D (primitive namespace; §3.7)
    fields:
      x_shape:   { kind: "[usize; 4]" }
      w_shape:   { kind: "[usize; 4]" }
      out_shape: { kind: "[usize; 4]" }
      stride:    { kind: "(usize, usize)" }
      padding:   { kind: "(usize, usize)" }
      dilation:  { kind: "(usize, usize)" }
      groups:    { kind: usize, constraint: "x.dim[1] % groups == 0 && w_shape[0] % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: conv2d(params)        # [N, Cout, Hout, Wout] from OpParams::Conv2D geometry
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "groups == 1", note: "dense conv; no grouped channel split" }
    - { when: "depthwise", note: "groups == Cin == Cout fast path" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared                       # author prior; Judge refines (§4.4)
  class: conv
  # output elements * receptive-field MACs (×2 = mul+add), grouped: Cin/groups per output channel
  flops: "2 * out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * (x_shape[1] / groups) * w_shape[2] * w_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  overhead_ns: 200
  memory: { device_bytes: 0, host_bytes: "out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3] * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true          # deterministic nested loop; f32 accum for half
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32/f64 native; bf16/f16 accumulate receptive-field inner product in f32 then narrow on store. Cross-correlation (PyTorch convention)."

determinism: same_hardware_bitwise
```

---

## conv_transpose2d  (2D transposed convolution, NCHW, output_padding, grouped)

2D transposed convolution (gradient-of-conv / fractionally-strided conv). `x [N, Cin, Hin, Win]`,
`weight [Cin, Cout/groups, Kh, Kw]` (note the transposed channel order vs Conv2D), optional 3rd
input `bias [Cout]`; output `[N, Cout, Hout, Wout]`. Carries the extra `output_padding` parameter
that transposed conv needs to disambiguate the output spatial size, alongside asymmetric
`stride`/`padding`/`dilation` and `groups`. f32/f64 native; bf16/f16 accumulate in f32 and narrow
on store. Registered no-bias and with-bias per dtype. CPU-only. Limitations: contiguous NCHW
only; weight uses the `[Cin, Cout/groups, ...]` order — a Conv2D-ordered weight is the wrong shape.

```fkc
kernel: conv_transpose2d
op_kind: ConvTranspose2D
blurb: "2D transposed convolution, NCHW, output_padding/dilation/groups; contiguous; half accum f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::conv_transpose2d_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [N, Cin, Hin, Win]
    - name: weight
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [Cin, Cout/groups, Kh, Kw]  (transposed channel order)
      shape_constraint: "same_dtype=x"
    - name: bias                    # optional; presence implicit in inputs.len()==3
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # [Cout]
      optional: true
  op_params:
    variant: ConvTranspose2D        # OpParams::ConvTranspose2D (primitive namespace; §3.7)
    fields:
      x_shape:        { kind: "[usize; 4]" }
      w_shape:        { kind: "[usize; 4]" }
      out_shape:      { kind: "[usize; 4]" }
      stride:         { kind: "(usize, usize)" }
      padding:        { kind: "(usize, usize)" }
      output_padding: { kind: "(usize, usize)" }
      dilation:       { kind: "(usize, usize)" }
      groups:         { kind: usize, constraint: "x.dim[1] % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: conv_transpose2d(params)   # [N, Cout, Hout, Wout] from OpParams::ConvTranspose2D geometry
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "groups == 1", note: "dense transposed conv" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared                       # author prior; Judge refines (§4.4)
  class: conv
  # scatter form: each input element fans out over the kernel into the output; MACs ∝ input elems * kernel * Cout/groups (×2 mul+add)
  flops: "2 * x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3] * (w_shape[1]) * w_shape[2] * w_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  overhead_ns: 200
  memory: { device_bytes: 0, host_bytes: "out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3] * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32/f64 native; bf16/f16 accumulate in f32 then narrow on store. Weight order [Cin, Cout/groups, Kh, Kw]."

determinism: same_hardware_bitwise
```

---

## causal_conv1d  (depthwise causal 1D conv with optional fused SiLU)

Depthwise causal 1D convolution along the time axis (Mamba / SSM short-conv block).
`x [batch, channels, seq_in]` (pre-padded by the caller with `kernel - 1` left zeros),
`weight [channels, 1, kernel]`, `bias [channels]`; output `[batch, channels, seq_out]` where
`seq_out = seq_in - (kernel - 1)`. One weight row per channel (depthwise). `use_silu` fuses a SiLU
activation on the store (matching baracuda's `causal_conv1d_*_run` flag). All tensors share dtype;
half accumulates in f32. Registered `[x, weight, bias, out]` per dtype on **both** CPU and
baracuda CUDA (both contiguous-only). Limitation: the left zero-pad is the caller's responsibility
(the kernel does not pad); weight is one tap-vector per channel, not a full conv weight.

```fkc
kernel: causal_conv1d
op_kind: CausalConv1d
blurb: "Depthwise causal 1D conv (caller-padded), optional fused SiLU on store; contiguous; half accum f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::causal_conv1d_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                       # [batch, channels, seq_in]  (seq_in includes kernel-1 left zeros)
    - name: weight
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                       # [channels, 1, kernel]
      shape_constraint: "same_dtype=x"
    - name: bias
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # [channels]
      shape_constraint: "same_dtype=x"
  op_params:
    variant: CausalConv1d           # OpParams::CausalConv1d (primitive namespace; §3.7)
    fields:
      batch:    { kind: usize }
      channels: { kind: usize }
      seq_in:   { kind: usize }
      seq_out:  { kind: usize, constraint: "seq_out == seq_in - (kernel - 1)" }
      kernel:   { kind: usize }
      use_silu: { kind: bool }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(out)      # [batch, channels, seq_out]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "use_silu == false", note: "no fused activation branch" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared                       # author prior; Judge refines (§4.4)
  class: conv
  # depthwise: per output element, kernel taps (×2 mul+add)
  flops: "2 * batch * channels * seq_out * kernel"
  bytes_moved: "(batch*channels*seq_in + channels*kernel + channels + batch*channels*seq_out) * dtype_bytes"
  overhead_ns: 100
  memory: { device_bytes: 0, host_bytes: "batch * channels * seq_out * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32/f64 native; bf16/f16 accumulate taps in f32 then narrow. Optional fused SiLU on store. Caller supplies kernel-1 left zero pad."

determinism: same_hardware_bitwise
```

---

## flash_attn  (multi-head SDPA over a fixed-capacity KV cache; symbolic live prefix)

Fused multi-head scaled-dot-product attention. `q [B, Hq, Sq, D]`, `k`/`v [B, Hkv, Sk, D]` with
`Hkv ≤ Hq`, GQA-divisible (`Hq % Hkv == 0`); optional 4th input `alibi_slopes [Hq]`. `Sk` is the
**physical capacity** of the K/V axis (strides + byte-length checks key off it); the kernel
attends only the first `k_len ≤ Sk` rows (the live prefix from a fixed-capacity KV-cache) and
bottom-right-aligns the causal mask at `k_len - Sq` (Phase D symbolic extents). `k_len` is a
dynamic scalar resolved per token from the SymEnv; the static path sets `k_len == Sk` and is
byte-identical to a plain `0..Sk` loop with a `kj > qi` causal test. f32 accumulation;
`softmax_scale`, optional `softcap`, sliding-window `(left, right)`. Registered no-alibi
`[q, k, v, out]` and with-alibi `[q, k, v, alibi, out]` per dtype. **CPU-only in this binding
table** (no CU/VK FlashAttn binding here). Numerics: stable streaming softmax in f32; not
bit-stable cross-hardware, but the deterministic CPU loop IS bit-stable per hardware.

```fkc
kernel: flash_attn
op_kind: FlashAttn
blurb: "Fused MHSA over a fixed-capacity KV cache; attends live prefix k_len <= Sk; GQA; causal/window/softcap."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hkv, Sk, D]  (Sk = CAPACITY)
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
      fdx:
        symbolic_extent: required   # attends live k_len from SymEnv; stride keyed to Sk
        extent_kind: range          # single bounded SymId (k_len <= Sk); §4.5
    - name: v
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"
      fdx:
        symbolic_extent: required   # k_len ≡ v_len ⇒ SAME SymId (FDX unification)
        extent_kind: range
    - name: alibi_slopes            # optional; presence implicit in inputs.len()==4
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # [Hq]
      optional: true
  op_params:
    variant: FlashAttn              # OpParams::FlashAttn (primitive namespace; §3.7)
    fields:
      b:   { kind: usize }
      hq:  { kind: usize }
      hkv: { kind: usize, constraint: "hq % hkv == 0" }
      sq:  { kind: usize }
      sk:  { kind: usize, note: "physical K/V capacity" }
      d:   { kind: usize }
      k_len: { kind: DynScalar, note: "live attended length <= sk; rides SymEnv" }
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)        # [B, Hq, Sq, D]; symbolic Sq preserved
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "k_len == sk", note: "static path; byte-identical to 0..Sk loop" }
    - { when: "causal == false", note: "no mask branch" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared                       # author prior; Judge refines (§4.4)
  class: attention
  # QK^T + PV over live k_len; symbolic over k_len, v1 evaluates at CAPACITY (sk). Live-prefix re-eval is [consumer-ahead].
  flops: "2 * b * hq * sq * k_len * d * 2"
  bytes_moved: "b * (hq*sq*d + 2*hkv*sk*d + hq*sq*d) * dtype_bytes"
  overhead_ns: 500
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true          # deterministic CPU streaming-softmax loop; f32 accum
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU streaming softmax, f32 accumulate; bf16/f16 narrow on store. Deterministic per hardware; not bit-stable cross-hardware (f32 narrowing differs)."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_q  (FlashAttn gradient wrt Q)

Backward of `flash_attn` producing **dQ**. Inputs `q, k, v, do` (upstream grad of the attention
output) plus optional `alibi_slopes`; output `dq` matches `q` shape/dtype. Same GQA / symbolic-KV
contract as the forward (`sk` capacity vs `k_len` live, causal/window/softcap via
`OpParams::FlashAttn`). **As-built cost caveat (faithful to the inventory):** the CPU wrapper
recomputes ALL THREE gradients (dQ, dK, dV) on every call and copies out only the requested one —
so dispatching `FlashAttnBackwardQ`, `...K`, and `...V` separately does ~3× the necessary work;
the cost below reflects the full three-gradient recompute, not a single-gradient kernel. Registered
no-alibi `[q, k, v, do, out]` and with-alibi `[q, k, v, do, alibi, out]` per dtype. CPU-only.

```fkc
kernel: flash_attn_backward_q
op_kind: FlashAttnBackwardQ
blurb: "FlashAttn gradient wrt Q (dQ); CPU recomputes all 3 grads then copies dQ (~3x); GQA/symbolic KV."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_q_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hkv, Sk, D]  (Sk = CAPACITY)
      shape_constraint: "divisible(q.dim[1], k.dim[1])"
      fdx: { symbolic_extent: required, extent_kind: range }
    - name: v
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
      fdx: { symbolic_extent: required, extent_kind: range }
    - name: do                      # upstream gradient of the attention output
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]
      shape_constraint: "same_as=q"
    - name: alibi_slopes            # optional; presence implicit in inputs.len()==5
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # [Hq]
      optional: true
  op_params:
    variant: FlashAttn              # OpParams::FlashAttn (primitive namespace; §3.7)
    fields:
      b:   { kind: usize }
      hq:  { kind: usize }
      hkv: { kind: usize, constraint: "hq % hkv == 0" }
      sq:  { kind: usize }
      sk:  { kind: usize, note: "physical K/V capacity" }
      d:   { kind: usize }
      k_len: { kind: DynScalar, note: "live attended length <= sk; rides SymEnv" }
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dq
      dtype_rule: passthrough(q)
      shape_rule: same_as(q)            # [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "k_len == sk", note: "static path" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared                       # author prior; Judge refines (§4.4)
  class: attention
  # ~3x forward: CPU recomputes dQ+dK+dV every call (inventory). fwd ~ 2*B*Hq*Sq*k_len*D*2.
  flops: "3 * 2 * b * hq * sq * k_len * d * 2"
  bytes_moved: "b * (2*hq*sq*d + 2*hkv*sk*d + hq*sq*d) * dtype_bytes"
  overhead_ns: 500
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true          # deterministic CPU loop; f32 accum
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate; bf16/f16 narrow on store. CPU recomputes all 3 grads per call (~3x cost); deterministic per hardware."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_k  (FlashAttn gradient wrt K)

Backward of `flash_attn` producing **dK**, output matches `k` shape/dtype. Same inputs and
symbolic-KV / GQA contract as `flash_attn_backward_q`; the OpKind selects which gradient is copied
out of the CPU wrapper's all-three recompute. Same ~3× cost caveat. Registered no-alibi / with-alibi
per dtype. CPU-only.

```fkc
kernel: flash_attn_backward_k
op_kind: FlashAttnBackwardK
blurb: "FlashAttn gradient wrt K (dK); CPU recomputes all 3 grads then copies dK (~3x); GQA/symbolic KV."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_k_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: k
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "divisible(q.dim[1], k.dim[1])"
      fdx: { symbolic_extent: required, extent_kind: range }
    - name: v
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
      fdx: { symbolic_extent: required, extent_kind: range }
    - name: do
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      optional: true
  op_params:
    variant: FlashAttn
    fields:
      b:   { kind: usize }
      hq:  { kind: usize }
      hkv: { kind: usize, constraint: "hq % hkv == 0" }
      sq:  { kind: usize }
      sk:  { kind: usize, note: "physical K/V capacity" }
      d:   { kind: usize }
      k_len: { kind: DynScalar, note: "live attended length <= sk; rides SymEnv" }
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dk
      dtype_rule: passthrough(k)
      shape_rule: same_as(k)            # [B, Hkv, Sk, D]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "k_len == sk", note: "static path" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared                       # author prior; Judge refines (§4.4)
  class: attention
  flops: "3 * 2 * b * hq * sq * k_len * d * 2"
  bytes_moved: "b * (2*hq*sq*d + 2*hkv*sk*d + hq*sq*d) * dtype_bytes"
  overhead_ns: 500
  memory: { device_bytes: 0, host_bytes: "b * hkv * sk * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate; bf16/f16 narrow on store. CPU recomputes all 3 grads per call (~3x cost); deterministic per hardware."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_v  (FlashAttn gradient wrt V)

Backward of `flash_attn` producing **dV**, output matches `v` shape/dtype. Same inputs and
symbolic-KV / GQA contract as the dQ/dK siblings; the OpKind selects which gradient is copied out
of the CPU wrapper's all-three recompute. Same ~3× cost caveat. Registered no-alibi / with-alibi
per dtype. CPU-only.

```fkc
kernel: flash_attn_backward_v
op_kind: FlashAttnBackwardV
blurb: "FlashAttn gradient wrt V (dV); CPU recomputes all 3 grads then copies dV (~3x); GQA/symbolic KV."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_v_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: k
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "divisible(q.dim[1], k.dim[1])"
      fdx: { symbolic_extent: required, extent_kind: range }
    - name: v
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
      fdx: { symbolic_extent: required, extent_kind: range }
    - name: do
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      optional: true
  op_params:
    variant: FlashAttn
    fields:
      b:   { kind: usize }
      hq:  { kind: usize }
      hkv: { kind: usize, constraint: "hq % hkv == 0" }
      sq:  { kind: usize }
      sk:  { kind: usize, note: "physical K/V capacity" }
      d:   { kind: usize }
      k_len: { kind: DynScalar, note: "live attended length <= sk; rides SymEnv" }
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dv
      dtype_rule: passthrough(v)
      shape_rule: same_as(v)            # [B, Hkv, Sk, D]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "k_len == sk", note: "static path" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared                       # author prior; Judge refines (§4.4)
  class: attention
  flops: "3 * 2 * b * hq * sq * k_len * d * 2"
  bytes_moved: "b * (2*hq*sq*d + 2*hkv*sk*d + hq*sq*d) * dtype_bytes"
  overhead_ns: 500
  memory: { device_bytes: 0, host_bytes: "b * hkv * sk * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate; bf16/f16 narrow on store. CPU recomputes all 3 grads per call (~3x cost); deterministic per hardware."

determinism: same_hardware_bitwise
```

---

## paged_attn  (vLLM-style blocked / paged KV-cache attention)

Multi-head attention over a **paged (blocked) KV cache**. `q [B, Hq, Sq, D]`; `k_cache` and
`v_cache` are physical block pools `[num_blocks, block_size, Hkv, D]`. A per-sequence
`block_table [B, max_blocks_per_seq]` (U32 — physical block index per logical position) and
`context_lens [B]` (U32 — true live context length per sequence) are passed as **separate graph
inputs** (the as-built ABI: `OpParams::PagedAttn`, `KernelRef::PagedAttn` operand order
`[q, k_cache, v_cache, block_table, context_lens, alibi?]`). Optional 6th input `alibi_slopes [Hq]`.
Output `[B, Hq, Sq, D]`. f32 accumulation; `softmax_scale`, optional `softcap`. Registered
no-alibi `[q, kc, vc, U32, U32, out]` and with-alibi per dtype. CPU-only.

Paged residency is described in **FDX gather terms by symbol** (§3.9.1): the `k_cache`/`v_cache`
pools are honest contiguous `uint8`-class block pools re-interpreted via an `FDXIndexedResidency`
sidecar (`kind = FDX_GATHER_PAGED_BLOCKS`). The **single-place rule** holds: `block_table` and
`context_lens` are ordinary `accept.inputs` operands (the ABI takes them as their own inputs), and
the pool operand's `fdx.gather.block_table` / `fdx.gather.context_lens` name **those same input
roles** — never a duplicate copy. **[consumer-ahead]:** the FDX gather descriptor + the
`Capability::DlpackExtGather` admission token are the 2026-06-17 FDX addition (no code yet); a
backend without the gather capability gets an explicit dense materialize priced from the
materialize kernel's contract, and an importer reaching the `gather`-bearing pool before the FDX
gather codes land returns `GatherNotYetSupported` rather than fabricating a descriptor.

```fkc
kernel: paged_attn
op_kind: PagedAttn
blurb: "Paged/blocked KV-cache MHA; block_table + context_lens select live blocks; GQA; softcap."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::paged_attn_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]
    - name: k_cache
      dtypes: [F32, F64, BF16, F16] # TRUE per-token pool element type (FDX FDXDTypeExt)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # physical pool [num_blocks, block_size, Hkv, D]
      shape_constraint: "divisible(q.dim[1], k_cache.dim[2])"   # GQA: Hq % Hkv == 0
      fdx:
        requires_ext: true          # MEANING_REQUIRES_EXT mandatory for a paged pool (FDX gather V19)
        symbolic_extent: required    # per-seq live length is symbolic (context_lens)
        extent_kind: range           # live length is a data-determined bounded SymId per sequence
        gather:
          kind: paged_blocks         # FDX FDX_GATHER_PAGED_BLOCKS
          block_table: block_table   # role of the SEPARATE block-table accept.input (below)
          context_lens: context_lens # role of the SEPARATE context-lens accept.input (below)
    - name: v_cache
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [num_blocks, block_size, Hkv, D]
      shape_constraint: "same_as=k_cache"
      fdx:
        requires_ext: true
        symbolic_extent: required
        extent_kind: range
        gather:
          kind: paged_blocks
          block_table: block_table
          context_lens: context_lens
    - name: block_table             # SEPARATE graph input (single-place rule §3.9.1)
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                       # [B, max_blocks_per_seq]
    - name: context_lens            # SEPARATE graph input (per-seq live lengths)
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # [B]
      fdx: { symbolic_extent: required }   # per-seq live lengths (data-determined sym)
    - name: alibi_slopes            # optional; presence implicit in inputs.len()==6
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # [Hq]
      optional: true
  op_params:
    variant: PagedAttn              # OpParams::PagedAttn (primitive namespace; §3.7)
    fields:
      b:   { kind: usize }
      hq:  { kind: usize }
      hkv: { kind: usize, constraint: "hq % hkv == 0" }
      sq:  { kind: usize }
      d:   { kind: usize }
      block_size:         { kind: usize }
      max_blocks_per_seq: { kind: usize }
      num_blocks:         { kind: usize }
      softmax_scale:      { kind: f32 }
      softcap:            { kind: "Option<f32>" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)        # [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "sq == 1", note: "single-query decode step (common paged-attn case)" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared                       # author prior; Judge refines (§4.4)
  class: attention
  # per query, attends ~ block_size * max_blocks_per_seq KV positions (bounded by live context). v1 at capacity.
  flops: "2 * b * hq * sq * (block_size * max_blocks_per_seq) * d * 2"
  bytes_moved: "b * (hq*sq*d + 2*hkv*(block_size*max_blocks_per_seq)*d + hq*sq*d) * dtype_bytes"
  overhead_ns: 500
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true          # deterministic CPU loop; f32 accum
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU f32 accumulate; bf16/f16 narrow on store. Deterministic per hardware. Live blocks selected by block_table within context_lens."

determinism: same_hardware_bitwise
```

---

## fused_softmax_cross_entropy  (stable log-softmax + NLL + ignore_index, one pass)

Fused softmax cross-entropy loss. Inputs `logits [n_rows, vocab]` (T-typed, flattened from the
original `[..., V]`) and `targets [n_rows]` (I64 class indices). Computes stable log-softmax + NLL
+ `ignore_index` masking + the requested `reduction` in a single pass, allocating only an
`[n_rows]` per-row accumulator (plus a scalar for Mean/Sum). **Output dtype is ALWAYS F32**
regardless of input T; shape is scalar `[]` for `Mean`/`Sum`, `[n_rows]` for `None`. Logits T ∈
{F32, F64, BF16, F16}; half rows widen to f32 for the log-softmax. CPU-only. Limitation: logits
must arrive as a contiguous `[n_rows, vocab]` 2D slab (caller flattens leading dims).

```fkc
kernel: fused_softmax_cross_entropy
op_kind: FusedSoftmaxCrossEntropy
blurb: "Stable log-softmax + NLL + ignore_index in one pass; output always F32; scalar or [n_rows]."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::fused_softmax_cross_entropy_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: logits
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                       # [n_rows, vocab]  (caller flattens [..., V])
    - name: targets
      dtypes: [I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # [n_rows]
      shape_constraint: "dim[0]=logits.dim[0]"
  op_params:
    variant: FusedSoftmaxCrossEntropy   # OpParams::FusedSoftmaxCrossEntropy (primitive namespace; §3.7)
    fields:
      n_rows:       { kind: usize }
      vocab:        { kind: usize }
      reduction:    { kind: "fuel_graph::registry::Reduction" }
      ignore_index: { kind: i64 }

return:
  outputs:
    - name: loss
      dtype_rule: fixed(F32)            # output ALWAYS F32 (§5.1)
      shape_rule: from_params(reduction)  # [] scalar for Mean/Sum; [n_rows] for None
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "ignore_index < 0", note: "no masked rows; skip ignore test" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared                       # author prior; Judge refines (§4.4)
  class: normalization
  # two passes over the [n_rows, vocab] logits (max + exp/sum) plus NLL gather
  flops: "n_rows * vocab * 4"
  bytes_moved: "(n_rows * vocab) * dtype_bytes + n_rows * 8 + n_rows * 4"
  overhead_ns: 80
  memory: { device_bytes: 0, host_bytes: "n_rows * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true          # deterministic row loop; f32 log-softmax accumulator
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Stable log-softmax (subtract row max) in f32; half logits widen to f32. Output F32. ignore_index rows contribute 0 and are excluded from the Mean denominator."

determinism: same_hardware_bitwise
```

---

## selective_scan  (Mamba SSM selective state-space scan)

Mamba selective-scan (S6) primitive. Five inputs: `u [batch, seqlen, dim]`,
`delta [batch, seqlen, dim]`, `a [dim, dstate]`, `b [batch, seqlen, dstate]`,
`c [batch, seqlen, dstate]`; output `y [batch, seqlen, dim]`. Sequential recurrent scan over the
time axis with an input-dependent (selective) discretization; `delta_softplus` applies
`softplus(delta)` before discretization (matching baracuda's `selective_scan_*_run` flag). All
tensors share dtype; half accumulates the recurrence in f32. CPU-only here. Note: although the op
can logically return `(y, last_state)`, this binding registers a single `y` output (the
multi-output bundle path is wired but not exercised here — see inventory cross-cutting facts).
Limitation: contiguous-only; the scan is inherently sequential on CPU.

```fkc
kernel: selective_scan
op_kind: SelectiveScan
blurb: "Mamba selective state-space scan (u, delta, A, B, C) -> y; sequential; contiguous; half accum f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::selective_scan_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: u
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                       # [batch, seqlen, dim]
    - name: delta
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                       # [batch, seqlen, dim]
      shape_constraint: "same_as=u"
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                       # [dim, dstate]
      shape_constraint: "same_dtype=u"
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                       # [batch, seqlen, dstate]
      shape_constraint: "same_dtype=u"
    - name: c
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                       # [batch, seqlen, dstate]
      shape_constraint: "same_as=b"
  op_params:
    variant: SelectiveScan          # OpParams::SelectiveScan (primitive namespace; §3.7)
    fields:
      batch:          { kind: usize }
      seqlen:         { kind: usize }
      dim:            { kind: usize }
      dstate:         { kind: usize }
      delta_softplus: { kind: bool }

return:
  outputs:
    - name: y
      dtype_rule: passthrough(u)
      shape_rule: same_as(u)            # [batch, seqlen, dim]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "delta_softplus == false", note: "skip softplus(delta) pre-step" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared                       # author prior; Judge refines (§4.4)
  class: reduction
  # per (batch, seqlen, dim) step: dstate state updates (×~4 fma+disc), plus C-projection
  flops: "batch * seqlen * dim * dstate * 4"
  bytes_moved: "(2*batch*seqlen*dim + dim*dstate + 2*batch*seqlen*dstate + batch*seqlen*dim) * dtype_bytes"
  overhead_ns: 120
  memory: { device_bytes: 0, host_bytes: "batch * dim * dstate * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true          # deterministic sequential scan; f32 recurrence accum
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32/f64 native; bf16/f16 accumulate recurrence in f32 then narrow. Sequential scan order is fixed and deterministic per hardware."

determinism: same_hardware_bitwise
```

---

## ssd_chunk_scan  (Mamba-2 SSD chunk-scan)

Mamba-2 state-space-duality (SSD) chunk-scan. Five inputs: `x [batch, seqlen, heads, head_dim]`,
`dt [batch, seqlen, heads]`, `a [heads]`, `b [batch, seqlen, heads, state_dim]`,
`c [batch, seqlen, heads, state_dim]`; output `y` matches `x`'s shape. `chunk_size` is the SSD
block size (a GPU-parallelism knob); validation requires `chunk_size > 0` and
`seqlen % chunk_size == 0`. **As-built note (faithful to inventory): the CPU kernel runs a
sequential scan regardless of `chunk_size`** — the chunking is honored as a parameter but the CPU
path does not parallelize over chunks. All tensors share dtype; half accumulates in f32. Single
`y` output registered (multi-output bundle wired but not exercised). CPU-only. Limitation:
contiguous-only; sequential on CPU.

```fkc
kernel: ssd_chunk_scan
op_kind: SsdChunkScan
blurb: "Mamba-2 SSD chunk-scan (x, dt, A, B, C) -> y; CPU sequential regardless of chunk_size; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::ssd_chunk_scan_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [batch, seqlen, heads, head_dim]
    - name: dt
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                       # [batch, seqlen, heads]
      shape_constraint: "same_dtype=x"
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # [heads]
      shape_constraint: "same_dtype=x"
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [batch, seqlen, heads, state_dim]
      shape_constraint: "same_dtype=x"
    - name: c
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [batch, seqlen, heads, state_dim]
      shape_constraint: "same_as=b"
  op_params:
    variant: SsdChunkScan           # OpParams::SsdChunkScan (primitive namespace; §3.7)
    fields:
      batch:      { kind: usize }
      seqlen:     { kind: usize }
      heads:      { kind: usize }
      head_dim:   { kind: usize }
      state_dim:  { kind: usize }
      chunk_size: { kind: usize, constraint: "chunk_size > 0 && seqlen % chunk_size == 0" }

return:
  outputs:
    - name: y
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)            # [batch, seqlen, heads, head_dim]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []                        # CPU sequential regardless of chunk_size (no fast path)
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared                       # author prior; Judge refines (§4.4)
  class: reduction
  # per (batch, seqlen, head, head_dim) step: state_dim state updates (×~4 fma)
  flops: "batch * seqlen * heads * head_dim * state_dim * 4"
  bytes_moved: "(batch*seqlen*heads*head_dim + batch*seqlen*heads + heads + 2*batch*seqlen*heads*state_dim + batch*seqlen*heads*head_dim) * dtype_bytes"
  overhead_ns: 150
  memory: { device_bytes: 0, host_bytes: "batch * heads * head_dim * state_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true          # deterministic sequential scan; f32 recurrence accum
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32/f64 native; bf16/f16 accumulate recurrence in f32 then narrow. CPU runs a fixed sequential scan regardless of chunk_size; deterministic per hardware."

determinism: same_hardware_bitwise
```
