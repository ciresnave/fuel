# Fuel fused-op registry — kernel inventory

Crate: **`fuel-graph`** (registry module `fuel-graph/src/registry/`), joined at
runtime to the kernel-side payload in **`fuel-dispatch::fused`** /
`fuel-dispatch::dispatch::register_default_fused_kernels`.

This inventory enumerates every distinct kernel-level op the Fuel
`FusedOpRegistry` provides. Each registry entry (`FusedOpEntry`) is a
kernel-level op carrying `shape_rule` / `dtype_rule` / `decompose` /
`backward` / `pattern` / `output_views`. The *graph-side* entry carries
shape/dtype/decompose/backward; the *kernel-side* payload
(`BackendImpl`: `cost`, `precision`, `caps`, `revision`) lives in
`fuel-dispatch`.

## Cross-cutting facts (apply to every entry unless noted)

- **Registry plumbing**: `fuel-graph/src/registry.rs`. 24 `FusedOpId`s
  allocated (slot 0 = `UNASSIGNED` sentinel); 23 entries registered in
  `default_registry()` (FlashAttnBackward is one source file with three
  entries: `entry_q`/`entry_k`/`entry_v`).
- **Layout handling (LOAD-BEARING)**: The graph-side registry does **not**
  encode layout. The kernel-side CPU wrappers in
  `fuel-dispatch/src/dispatch.rs` take a `_layouts: &[Layout]` argument and
  **ignore it** (underscore-prefixed; verified in `rope_f32_cpu_wrapper`
  at dispatch.rs:1296-1332). They call `cpu_input()`
  (dispatch.rs:224-233) which returns the **raw byte buffer with no stride
  application**. So every fused CPU kernel is **contiguous-only, offset-0,
  row-major**.
- **No fused kernel advertises `strided_input`**: every `register_fused!`
  call in `register_default_fused_kernels` (dispatch.rs:5238+) omits the
  optional `caps = ...`, so caps default to `KernelCaps::empty()`
  (macro at fused.rs:368-400). Consequence: a non-contiguous input is
  materialized to contiguous by the executor's **auto-Contiguize** step
  before the kernel runs (see `StridedInputPreferenceFilter`,
  strided_input_pref.rs:1-47). Broadcast/strided/offset inputs are never
  seen directly by these kernels.
- **dtype monomorphization**: CPU coverage is registered per-dtype, almost
  always `{F32, F64, BF16, F16}` (QMatMul = F32 only; PagedAttn index
  inputs are U32). Listed as a dtype list on one entry, per the task.
- **Precision (CPU)**: all CPU fused kernels claim
  `bit_stable_on_same_hardware: true` with no static ULP/relative/absolute
  bound. Accumulate in F32 for BF16/F16 inputs (F64 for F64 input); FSCE
  and the Mamba scans accumulate in F64. GPU kernels declare precision
  separately when they register (none registered through this registry yet).
- **`decompose` semantics**: forward norm/softmax/rope/FSCE entries lower
  to a primitive subgraph; conv/attn/quant/scan/backward-helper entries
  **panic** in `decompose` (no primitive form) and rely on the executor's
  `cpu_fallback` to the always-built CPU kernel.

---

## 1. SoftmaxLastDim — `FusedOps::SOFTMAX_LAST_DIM` (id 1)
- Source: `fuel-graph/src/registry/softmax_last_dim.rs:25`
- Op kind: forward reduction (softmax over last dim). Family `Forward`.
- Inputs: 1 (`x`). dtypes: F32/F64/BF16/F16 (UNARY tuples).
- Layout: contiguous-only, offset-0, row-major (auto-contiguized).
- Params: none (`FusedOpParams::SoftmaxLastDim`).
- Shape rule: passthrough (= input 0). Dtype rule: passthrough.
- Output: single, same shape+dtype as input; fresh contiguous buffer.
- Backward: `Fused(SOFTMAX_LAST_DIM_BACKWARD)`.
- Decompose: 7-node `ReduceMaxTo→BroadcastTo→Sub→Exp→ReduceSumTo→BroadcastTo→Div`.
- Pattern: callable matcher recognizes that 7-node subgraph (single-consumer guards).
- Precision: `NORM_FAMILY_CPU_PRECISION` (bit-stable, F32 accum).

## 2. FusedLinear — `FusedOps::FUSED_LINEAR` (id 2)
- Source: `fuel-graph/src/registry/fused_linear.rs:27`
- Op kind: GEMM + bias epilogue, `(a @ b) + bias`. Family `Forward`.
- Inputs: 3 (`a`,`b`,`bias`). dtypes: F32/F64/BF16/F16 (`[in,in,in,out]`).
- Layout: contiguous-only (CPU kernel ignores layouts).
- Params: none (`FusedOpParams::FusedLinear`).
- Shape rule: matmul output `[..., M, N]` (a=`[...,M,K]`, b=`[...,K,N]`); a/b ranks must match, ≥2. Dtype rule: = a (all three agree).
- Output: single, `[...,M,N]`, dtype = a; fresh buffer.
- Backward: `NotDifferentiable` in the registry; actual grad handled by `Tensor::backward`'s `Op::Fused(FUSED_LINEAR)` arm (3-grad decomposition).
- Decompose: `MatMul → BroadcastTo(bias) → Add`.
- Pattern: matches `Add(MatMul(a,b), BroadcastTo(rank-1 bias))`; bias len == matmul last dim; inner MatMul single-consumer.
- Precision: `FUSED_LINEAR_CPU_PRECISION` (bit-stable; BF16/F16 accum in F32).
- Notes: only fused op with a populated kernel-side `BackendImpl` in `default_kernel_registry()` historically; CUTLASS/cuBLAS bias-epilogue alternatives register here.

## 3. RmsNormLastDim — `FusedOps::RMS_NORM_LAST_DIM` (id 3)
- Source: `fuel-graph/src/registry/rms_norm_last_dim.rs:29`
- Op kind: RMS norm `x / sqrt(mean(x²)+eps)` over last dim. Family `Norm`.
- Inputs: 1 (`x`). dtypes: F32/F64/BF16/F16.
- Layout: contiguous-only.
- Params: `RmsNormLastDim { eps: f64 }`.
- Shape rule: passthrough. Dtype rule: passthrough.
- Output: single, same shape+dtype; fresh buffer.
- Backward: `Fused(RMS_NORM_LAST_DIM_BACKWARD)`.
- Decompose: 7-node `Sqr→MeanDim→Reshape→AddScalar(eps)→Sqrt→BroadcastTo→Div`.
- Pattern: callable matcher (extracts eps from AddScalar; single-consumer guards).
- Precision: `NORM_FAMILY_CPU_PRECISION`.

## 4. LayerNormLastDim — `FusedOps::LAYER_NORM_LAST_DIM` (id 4)
- Source: `fuel-graph/src/registry/layer_norm_last_dim.rs:39`
- Op kind: layer norm `(x-mean)/sqrt(var+eps)` over last dim, no affine. Family `Norm`.
- Inputs: 1 (`x`). dtypes: F32/F64/BF16/F16.
- Layout: contiguous-only.
- Params: `LayerNormLastDim { eps: f64 }`.
- Shape rule: passthrough. Dtype rule: passthrough.
- Output: single, same shape+dtype; fresh buffer.
- Backward: `Fused(LAYER_NORM_LAST_DIM_BACKWARD)`.
- Decompose: 11-node mean/var/normalize chain.
- Pattern: **stub `None`** (no matcher; one-way builder→fused migration).
- Precision: `NORM_FAMILY_CPU_PRECISION`.

## 5. Rope — `FusedOps::ROPE` (id 5)
- Source: `fuel-graph/src/registry/rope.rs:30`
- Op kind: rotary position embedding with caller cos/sin tables. Family `Forward`.
- Inputs: 3 (`x`,`cos`,`sin`). dtypes: F32/F64/BF16/F16 (`[x,cos,sin,out]`).
- Layout: contiguous-only (`rope_f32_cpu_wrapper` ignores `_layouts`; kernel keyed by `outer_count/seq/head_dim` in `OpParams::Rope`). Requires rank≥2, even head_dim.
- Params: none in registry (`FusedOpParams::Rope`); seq/head_dim recovered from shapes.
- Shape rule: passthrough (= x). Dtype rule: passthrough (= x).
- Output: single, = x shape+dtype; fresh buffer.
- Backward: `NotDifferentiable` in registry; `Tensor::backward` emits another Rope with negated sin.
- Decompose: 12-node slice/neg/concat + cos/sin reshape+broadcast + mul/add.
- Pattern: **stub `None`**.
- Precision: `ROPE_CPU_PRECISION`.

## 6. Conv2D — `FusedOps::CONV2D` (id 6)
- Source: `fuel-graph/src/registry/conv2d.rs:54`
- Op kind: 2-D cross-correlation, stride/padding/groups. Family `Forward`.
- Inputs: 2 or 3 (`x`,`weight`,[`bias`]). x=`[N,Cin,H,W]`, w=`[Cout,Cin/g,Kh,Kw]`, bias=`[Cout]`. dtypes: F32/F64/BF16/F16, with/without-bias tuples.
- Layout: contiguous-only.
- Params: `Conv2D { stride:(usize,usize), padding:(usize,usize), groups:usize }`. Dilation always 1.
- Shape rule: `[N, Cout, (H+2ph−Kh)/sh+1, (W+2pw−Kw)/sw+1]`. Dtype rule: = x.
- Output: single rank-4; fresh buffer.
- Backward: `NotDifferentiable` in registry; real grad via `Tensor::backward` (dX=ConvTranspose2D, dW=transposed conv, dB=reduce_sum_to).
- Decompose: **panics** (no `Op::Im2Col` primitive). Backends without native kernel use `cpu_fallback`.
- Pattern: **stub `None`**.
- Precision: `CONV2D_CPU_PRECISION`.

## 7. SoftmaxLastDimBackward — `FusedOps::SOFTMAX_LAST_DIM_BACKWARD` (id 7)
- Source: `fuel-graph/src/registry/softmax_last_dim_backward.rs:45`
- Op kind: softmax backward `s·(g − sum(g·s, last, keepdim))`. Family `Backward`.
- Inputs: 2 (`y`=fwd output, `upstream`). dtypes: F32/F64/BF16/F16 (`BW_*` `[T,T,T]`).
- Layout: contiguous-only.
- Params: none (`SoftmaxLastDimBackward`).
- Shape rule: = input 0. Dtype rule: = input 0.
- Output: single, = y shape+dtype.
- Backward: `NotDifferentiable` (higher-order grads panic — MVP).
- Decompose: **panics** (no primitive form worth materializing).
- Pattern: **stub `None`** (autograd emits directly).
- Precision: `NORM_FAMILY_CPU_PRECISION` (cost via `cost_norm_family_cpu`).

## 8. LayerNormLastDimBackward — `FusedOps::LAYER_NORM_LAST_DIM_BACKWARD` (id 8)
- Source: `fuel-graph/src/registry/layer_norm_last_dim_backward.rs:21`
- Op kind: layer-norm backward; recomputes mean/var from x. Family `Backward`.
- Inputs: 2 (`x`,`upstream`). dtypes: F32/F64/BF16/F16.
- Layout: contiguous-only.
- Params: `LayerNormLastDimBackward { eps: f64 }`.
- Shape rule: = input 0. Dtype rule: = input 0.
- Output: single, = x shape+dtype.
- Backward: `NotDifferentiable`. Decompose: **panics**. Pattern: **stub `None`**.
- Precision: `NORM_FAMILY_CPU_PRECISION`.

## 9. RmsNormLastDimBackward — `FusedOps::RMS_NORM_LAST_DIM_BACKWARD` (id 9)
- Source: `fuel-graph/src/registry/rms_norm_last_dim_backward.rs:19`
- Op kind: rms-norm backward closed form `r_rms·(g − x·s/(n·(mean_sq+eps)))`. Family `Backward`.
- Inputs: 2 (`x`,`upstream`). dtypes: F32/F64/BF16/F16.
- Layout: contiguous-only.
- Params: `RmsNormLastDimBackward { eps: f64 }`.
- Shape rule: = input 0. Dtype rule: = input 0.
- Output: single, = x shape+dtype.
- Backward: `NotDifferentiable`. Decompose: **panics**. Pattern: **stub `None`**.
- Precision: `NORM_FAMILY_CPU_PRECISION`.

## 10. ReduceMaxToBackward — `FusedOps::REDUCE_MAX_TO_BACKWARD` (id 10)
- Source: `fuel-graph/src/registry/reduce_max_to_backward.rs:28`
- Op kind: backward of primitive `Op::ReduceMaxTo` — route upstream to argmax positions, fair-share ties. Family `Backward`.
- Inputs: 2 (`x`,`upstream`). dtypes: F32/F64/BF16/F16.
- Layout: contiguous-only.
- Params: none (`ReduceMaxToBackward`).
- Shape rule: = input 0 (grad_x has x's shape). Dtype rule: = input 0.
- Output: single, = x shape+dtype.
- Backward: `NotDifferentiable`. Decompose: **panics** (would need an equality primitive). Pattern: **stub `None`**.
- Precision: `REDUCE_MAX_TO_BACKWARD_CPU_PRECISION` (cost `cost_reduce_max_to_backward_cpu`).
- Note: not reached via `BackwardKind::Fused`; autograd reaches it directly from `Op::ReduceMaxTo`.

## 11. ConvTranspose2D — `FusedOps::CONV_TRANSPOSE2D` (id 11)
- Source: `fuel-graph/src/registry/conv_transpose_2d.rs:32`
- Op kind: 2-D transposed (fractionally-strided) convolution, no bias. Family `Forward`.
- Inputs: 2 (`x`,`weight`). x=`[N,Cin,H,W]`, w=`[Cin,Cout/g,Kh,Kw]`. dtypes: F32/F64/BF16/F16 (reuses CV tuples).
- Layout: contiguous-only.
- Params: `ConvTranspose2D { stride, padding, output_padding, dilation, groups }` (all (usize,usize) except groups).
- Shape rule: `Hout=(H−1)·s − 2·p + d·(K−1) + out_pad + 1` (saturating); Cout = (Cout/g)·groups. Dtype rule: = x.
- Output: single rank-4; fresh buffer.
- Backward: `NotDifferentiable` (forward arm panics in `Tensor::backward`). Decompose: **panics**. Pattern: **stub `None`**.
- Precision: `CONV_TRANSPOSE2D_CPU_PRECISION`.

## 12. FlashAttn — `FusedOps::FLASH_ATTN` (id 12)
- Source: `fuel-graph/src/registry/flash_attn.rs:46`
- Op kind: multi-head scaled-dot-product (FlashAttention) attention. Family `Attention`.
- Inputs: 4 or 5 (`q`,`k`,`v`,[`alibi`]). q=`[B,Hq,Sq,D]`, k/v=`[B,Hkv,Sk,D]`, alibi=`[Hq]`. dtypes: F32/F64/BF16/F16 (no-alibi 4-tuple / with-alibi 5-tuple).
- Layout: contiguous-only.
- Params: `FlashAttn { softmax_scale:f32, causal:bool, window_size_left:Option<usize>, window_size_right:Option<usize>, softcap:Option<f32>, k_len:Option<DynScalar> }`. `k_len=None` ⇒ full K extent; `Some` ⇒ runtime live-prefix over a capacity KV cache (FA2 bottom-right causal mask at `k_len−Sq`).
- Shape rule: = q (input 0). Dtype rule: = q.
- Output: single, = q shape+dtype.
- Backward: `NotDifferentiable` in registry (grads via the three FlashAttnBackward ids). Decompose: **panics** (would re-materialize the `[B,Hq,Sq,Sk]` score matrix). Pattern: **stub `None`**.
- Precision: `ATTN_CPU_PRECISION` (CPU naive reference bit-stable; GPU tiled-softmax differs).

## 13. PagedAttn — `FusedOps::PAGED_ATTN` (id 13)
- Source: `fuel-graph/src/registry/paged_attn.rs:40`
- Op kind: paged-cache attention (decode-only). Family `Attention`.
- Inputs: 5 or 6 (`q`,`k_cache`,`v_cache`,`block_table:U32`,`context_lens:U32`,[`alibi`]). q=`[B,Hq,Sq,D]`, caches=`[num_blocks,block_size,Hkv,D]`. dtypes: q/cache F32/F64/BF16/F16, block_table+context_lens U32.
- Layout: contiguous-only.
- Params: `PagedAttn { softmax_scale:f32, block_size:usize, softcap:Option<f32> }`.
- Shape rule: = q. Dtype rule: = q.
- Output: single, = q shape+dtype.
- Backward: `NotDifferentiable` (decode-only). Decompose: **panics**. Pattern: **stub `None`**.
- Precision: `ATTN_CPU_PRECISION`.

## 14. QMatMul — `FusedOps::QMATMUL` (id 14)
- Source: `fuel-graph/src/registry/qmatmul.rs:40`
- Op kind: quantized matmul `C = A @ dequant(W_Q)` (GGUF/llama.cpp block stream). Family `Quantized`.
- Inputs: 2 (`a`:F32 activations `[...,M,K]`, `w_q_bytes`:U32 packed block stream). dtypes: F32 only (`QM_F32 = [F32,U32,F32]`).
- Layout: contiguous-only.
- Params: `QMatMul { quant_type: QuantType, k:usize, n:usize }` (QuantType ∈ Q4_0..Q6K).
- Shape rule: `[...,M,N]` (M from a[-2], N from params). Dtype rule: = a (F32).
- Output: single, `[...,M,N]` F32; fresh buffer.
- Backward: `NotDifferentiable` (frozen weights). Decompose: **panics** (avoids dequant DRAM round-trip). Pattern: **stub `None`**.
- Precision: `QMATMUL_CPU_PRECISION` (cost `cost_qmatmul_cpu`).

## 15. PowIBackward — `FusedOps::POWI_BACKWARD` (id 15)
- Source: `fuel-graph/src/registry/powi_backward.rs:26`
- Op kind: backward of primitive `Op::PowI` — `grad_x = exp·x^(exp−1)·upstream`. Family `Backward`.
- Inputs: 2 (`x`,`upstream`). dtypes: F32/F64/BF16/F16.
- Layout: contiguous-only.
- Params: `PowIBackward { exp: i32 }`.
- Shape rule: = input 0 (x). Dtype rule: = input 0.
- Output: single, = x shape+dtype.
- Backward: `NotDifferentiable`. Decompose: **panics** (fallback PowI(n−1)→MulScalar→Mul lives in `Tensor::backward`). Pattern: **stub `None`**.
- Precision: `POWI_BACKWARD_CPU_PRECISION` (cost `cost_powi_backward_cpu`).
- Note: reached directly from `Op::PowI`, not via `BackwardKind::Fused`.

## 16. InplaceAffine — `FusedOps::INPLACE_AFFINE` (id 16)
- Source: `fuel-graph/src/registry/inplace_affine.rs:23`
- Op kind: in-place affine `x = mul·x + add`, mutating input 0. Family `Forward`.
- Inputs: 1 (mutated tensor). dtypes: F32/F64/BF16/F16.
- Layout: contiguous-only.
- Params: `InplaceAffine { mul:f64, add:f64 }`.
- Shape rule: = input 0. Dtype rule: = input 0.
- Output: **aliases input 0 by contract** (destructive on index 0 via `Op::destructive_input`; `derive_ordering` pins it after non-destructive readers). In-place / not a fresh buffer.
- Backward: `NotDifferentiable` (autograd integration is Phase 4). Decompose: **panics** (no non-destructive form). Pattern: **stub `None`**.
- Precision: `INPLACE_AFFINE_CPU_PRECISION` (cost `cost_inplace_affine_cpu`).

## 17. FusedSoftmaxCrossEntropy — `FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY` (id 17)
- Source: `fuel-graph/src/registry/fused_softmax_cross_entropy.rs:68`
- Op kind: fused softmax + NLL over class-index targets. Family `Forward`.
- Inputs: 2 (`logits` `[...,V]` F32; `targets` `[...]` I64). dtypes per cost: logits F32, targets I64.
- Layout: contiguous-only.
- Params: `FusedSoftmaxCrossEntropy { reduction: Reduction(Mean|Sum|None), ignore_index: i64 }`.
- Shape rule: Mean/Sum → scalar `[]`; None → targets.shape. Dtype rule: **always F32** regardless of input dtype.
- Output: single F32; scalar or per-row.
- Backward: **`Decompose`** (autograd lowers and runs the primitive backward; re-introduces `[...,V]` intermediates).
- Decompose: log-softmax (ReduceMax→Sub→Exp→ReduceSum→Log→Sub) + Cast(targets→U32)→Unsqueeze→Gather→Squeeze→MulScalar(−1)→reduce; ends with Cast to F32 if work_dtype≠F32. **`ignore_index` NOT honored in the lowered form** (forward CPU kernel does mask).
- Pattern: **stub `None`** (explicit builder opt-in).
- Precision: `FUSED_SOFTMAX_CROSS_ENTROPY_CPU_PRECISION` (stable log-sum-exp in F64, narrow to F32).

## 18. CausalConv1d — `FusedOps::CAUSAL_CONV1D` (id 18)
- Source: `fuel-graph/src/registry/causal_conv1d.rs:66`
- Op kind: depthwise 1-D causal conv + optional fused SiLU (baracuda `causal_conv1d_*_run`). Family `Forward`.
- Inputs: 3 (`x` `[batch,channels,seq+kernel−1]` caller-left-padded; `weight` `[channels,1,kernel]`; `bias` `[channels]`). dtypes: input dtype (F32 in v1; F32/F64/BF16/F16 cost-monomorphized).
- Layout: contiguous-only.
- Params: `CausalConv1d { use_silu: bool }`.
- Shape rule: `[batch, channels, seq]` where seq = x_seq − (kernel−1) (kernel from weight[2]). Dtype rule: = x.
- Output: single rank-3; fresh buffer.
- Backward: `NotDifferentiable` (v1 inference-only). Decompose: **panics** (no `Op::Conv1D`). Pattern: **stub `None`**.
- Precision: `CAUSAL_CONV1D_CPU_PRECISION` (F32 accumulator; SiLU `x/(1+exp(−x))`).

## 19. SelectiveScan — `FusedOps::SELECTIVE_SCAN` (id 19)
- Source: `fuel-graph/src/registry/selective_scan.rs:86`
- Op kind: Mamba-1 selective state-space scan (forward). Family `Forward`. **Multi-output.**
- Inputs: 5 (`u` `[B,L,dim]`, `delta` `[B,L,dim]`, `a` `[dim,dstate]`, `b` `[B,L,dstate]`, `c` `[B,L,dstate]`). dtypes: F32 (uniform v1; cost-monomorphized over F32/F64/BF16/F16).
- Layout: contiguous-only; `output_views` declare both output slots as `Layout::contiguous`.
- Params: `SelectiveScan { delta_softplus: bool }`.
- Shape rule: slot 0 `y: [B,L,dim]` (= input 0). Dtype rule: = u.
- Output: **bundled, 2 slots** via `output_views`: slot0 `y [B,L,dim]`, slot1 `last_state [B,dim,dstate]`, both = u's dtype, both contiguous. Consumers project via `Op::View`.
- Backward: `NotDifferentiable` (v1). Decompose: **panics** (O(seqlen) recurrence). Pattern: **stub `None`**.
- Precision: `SELECTIVE_SCAN_CPU_PRECISION` (F64 state accumulator, narrow to F32 on store).

## 20. SsdChunkScan — `FusedOps::SSD_CHUNK_SCAN` (id 20)
- Source: `fuel-graph/src/registry/ssd_chunk_scan.rs:75`
- Op kind: Mamba-2 SSD chunked scan (forward). Family `Forward`. **Multi-output.**
- Inputs: 5 (`x` `[B,L,H,head_dim]`, `dt` `[B,L,H]`, `a` `[H]`, `b` `[B,L,H,state_dim]`, `c` `[B,L,H,state_dim]`). dtypes: F32 (uniform v1; cost-monomorphized).
- Layout: contiguous-only; `output_views` both slots `Layout::contiguous`.
- Params: `SsdChunkScan { chunk_size: usize }` (GPU parallelism knob; CPU runs sequential; require chunk_size>0 and seqlen%chunk_size==0).
- Shape rule: slot 0 `y: [B,L,H,head_dim]` (= input 0). Dtype rule: = x.
- Output: **bundled, 2 slots**: slot0 `y [B,L,H,head_dim]`, slot1 `last_state [B,H,head_dim,state_dim]`, both = x's dtype, contiguous.
- Backward: `NotDifferentiable` (v1). Decompose: **panics**. Pattern: **stub `None`**.
- Precision: `SSD_CHUNK_SCAN_CPU_PRECISION` (F64 per-head state accumulator).

## 21. Nf4Matmul — `FusedOps::NF4_MATMUL` (id 21)
- Source: `fuel-graph/src/registry/nf4_matmul.rs:68`
- Op kind: bitsandbytes 4-bit NormalFloat quantized matmul. Family `Quantized`.
- Inputs: 3 (`activations` `[...,M,K]`; `w_packed` `[N,K/2]` U8, two NF4 codes/byte, K even; `absmax` `[N,K/block_size]` F32). dtypes: activations F32/F16/BF16 (v1), w_packed U8, absmax F32.
- Layout: contiguous-only.
- Params: `Nf4Matmul { block_size: usize }` (typ. 64; require K%block_size==0). NF4 16-entry LUT baked in kernel.
- Shape rule: `[...,M,N]` (N = w_packed[0]). Dtype rule: = activations.
- Output: single `[...,M,N]`, dtype = activations; fresh buffer.
- Backward: `NotDifferentiable` (frozen weights). Decompose: **panics** (avoids dequant round-trip). Pattern: **stub `None`**.
- Precision: `NF4_MATMUL_CPU_PRECISION` (F32 inner-product accumulator; F16/BF16 up-cast on load).

## 22-24. FlashAttnBackward Q / K / V — ids 22 / 23 / 24
- Source: `fuel-graph/src/registry/flash_attn_backward.rs` (`entry_q`:27, `entry_k`:42, `entry_v`:57). One file, three registry entries (three FusedOpIds).
- Op kind: FlashAttention backward producing dQ / dK / dV. Family `Attention`.
- Inputs: 4 or 5 (`q`,`k`,`v`,`do`,[`alibi`]). dtypes: F32/F64/BF16/F16 (no-alibi 5-tuple / with-alibi 6-tuple, `FAB_*`).
- Layout: contiguous-only.
- Params: shared `FlashAttnBackward { softmax_scale:f32, causal:bool, window_size_left:Option<usize>, window_size_right:Option<usize>, softcap:Option<f32> }`. The FusedOpId distinguishes Q vs K vs V.
- Shape rule: Q→input0 (q) shape; K→input1 (k) shape; V→input2 (v) shape. Dtype rule: = input 0 (all share dtype).
- Output: single per id, matching the respective forward input.
- Backward: `NotDifferentiable`. Decompose: **panics** (would re-materialize the score matrix; every backend must register a kernel). Pattern: `None`.
- Precision: `ATTN_BACKWARD_CPU_PRECISION` (cost `cost_attn_backward_cpu`); v1 each variant recomputes softmax independently (3× recompute on CPU reference).

---

## Coverage summary

- **23 registered entries** (24 ids, slot 0 reserved).
- Forward family: SoftmaxLastDim, FusedLinear, Rope, Conv2D, ConvTranspose2D, InplaceAffine, FusedSoftmaxCrossEntropy, CausalConv1d, SelectiveScan, SsdChunkScan.
- Norm family: RmsNormLastDim, LayerNormLastDim.
- Backward family: SoftmaxLastDimBackward, LayerNormLastDimBackward, RmsNormLastDimBackward, ReduceMaxToBackward, PowIBackward.
- Attention family: FlashAttn, PagedAttn, FlashAttnBackward{Q,K,V}.
- Quantized family: QMatMul, Nf4Matmul.
- **Multi-output** (bundled `output_views`): SelectiveScan, SsdChunkScan.
- **In-place / aliasing**: InplaceAffine (destructive on input 0).
- **decompose panics (no primitive form)**: Conv2D, ConvTranspose2D, FlashAttn, PagedAttn, QMatMul, Nf4Matmul, CausalConv1d, SelectiveScan, SsdChunkScan, all backward helpers, all FlashAttnBackward.
- **decompose lowers to primitives**: SoftmaxLastDim, FusedLinear, RmsNormLastDim, LayerNormLastDim, Rope, FusedSoftmaxCrossEntropy.
- **Live pattern matcher** (recognizes user-decomposed subgraph): SoftmaxLastDim, FusedLinear, RmsNormLastDim. All others stub `None`.
- **Every fused CPU kernel is contiguous-only / offset-0 / row-major; none advertise `strided_input`; strided inputs are auto-contiguized by the executor.**
