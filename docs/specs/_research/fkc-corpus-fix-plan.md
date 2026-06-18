# FKC corpus lint — authoritative per-defect FIX PLAN

Scope: `docs/kernel-contracts/` (68 `*.fkc.md` files / 708 `## ` sections). Lint = the
`V-FKC-*` battery in `fuel-dispatch/src/fkc/validate.rs` + the parse/extract rules in
`fuel-dispatch/src/fkc/parse.rs`, run standalone over the checked-in corpus.

Result baseline: 12 correct deferrals (MX / gather consumer-ahead — leave as-is) + **73 HARD
failures to fix** (this plan). Do NOT edit the corpus or the importer in this task; do NOT commit.
Fix agents consume their crate-dir section below.

## As-built enums (the source of truth used to classify every fix)

Read from the as-built code this round:

- **`OpKind`** — `fuel-core-types/src/dispatch.rs` lines 52–494. The lowering table that gates
  `op_kind:` is `fuel-dispatch/src/fkc/lower.rs::lower_op_kind` (lines 130–264), one arm per variant.
- **`OpParams`** variants — `fuel-dispatch/src/kernel.rs` lines 163–681; the validator's accepted set
  is `validate.rs::is_op_params_variant` (lines 776–821).
- **`FusedOpParams`** variants — accepted set `validate.rs::is_fused_op_params_variant` (lines 824–850);
  `FusedOps::*` token table is `lower.rs::fused_op_id_for_const_name` (lines 296–325).

Key facts that decide TYPO-vs-NO-OPKIND:

- `CumSum` EXISTS (dispatch.rs:269). The corpus wrote `Cumsum`. → **TYPO → rename `Cumsum`→`CumSum`.**
- `ClampInplace`, `PowIInplace` EXIST (dispatch.rs:445/448). `ClampElementwise`/`PowIElementwise` also
  exist. The in-place clamp/powi kernels wrote bare `Clamp`/`PowI` (NOT enum names). → **TYPO → rename
  to the in-place variant** (`ClampInplace`/`PowIInplace`) because those kernels are the in-place arm.
- `ConvTranspose2D` EXISTS as a real `OpKind` (dispatch.rs:127) **and** `OpParams::ConvTranspose2D`
  (kernel.rs:244). So `ConvTranspose2D` contracts are REGISTRABLE as-is (no fix).
- `ConvTranspose1D` does **NOT** exist as an `OpKind`. Only `OpParams::ConvTranspose1D` exists
  (kernel.rs:233) with **no matching `OpKind`** → NO-OPKIND → describe-only.
- `Im2Col`, `Im2Col1d`, `Col2Im1d`, `MaxPool2D`, `AvgPool2D`, `Conv2dSimple`, `UpsampleNearest2D`,
  `UpsampleBilinear2D`, `Transpose`, `Permute`, `ArgSort`, `RandUniform`, `RandNormal`,
  `AddAssignScaled`, `DequantizeQ4_0`, `DequantizeQ8_0`, `DequantizeQ4KM`, `QuantizeQ8_0` — **none is
  an `OpKind`** (verified absent from dispatch.rs / lower_op_kind). → NO-OPKIND → describe-only.
  (`DequantizeQ4_0`/`DequantizeQ8_0`/`DequantizeQ4KM`/`QuantizeQ8_0` are `Capability` tokens, not
  `OpKind`s.)
- `OpParams::Unary` / `OpParams::Binary` do **NOT** exist (elementwise binding-table key carries
  `OpParams::None`). `ConvShape` is a `fuel-conv` type, not an `OpParams` variant. `Permute` is not an
  `OpParams` variant either.

## The DESCRIBE-ONLY marker (added this round)

NON-REGISTRABLE documentation sections (chassis umbrellas + ops with no real dispatch `OpKind`) get a
new **describe-only marker** added to the FKC schema + validator THIS round:

- Schema: add `registrable: Option<bool>` to `FkcKernel` (`fuel-dispatch/src/fkc/schema.rs`), default
  `None` (≡ `true`).
- Validator: in `validate.rs::validate_kernel`, when `registrable == Some(false)`, **skip rule-2
  exactly-one-of-op_kind/fused_op and skip rule-7 op-param-namespace**; a describe-only section is NOT
  required to name a real `op_kind`/`fused_op` and may keep `op_kind: ~` / `op_params.variant: ~` or a
  forward-marker token. All OTHER rules (blurb, entry_point, ≥1 input, ≥1 output, cost, precision,
  determinism, layout coherence, dtype membership) STILL apply.
- Corpus sections in the describe-only set below add `registrable: false` to their ` ```fkc ` block.
- This is the documented, never-relax-a-validator-to-hide-a-defect path: describe-only is an explicit,
  visible classification, not a silenced check.

NEVER invent an OpKind/OpParams/FusedOpParams variant. NEVER relax a validator to mask a defect.

---

# Summary tables (consume these first)

## A. TYPO → exact-OpKind RENAMES (the OpKind genuinely exists under a different exact name)

| corpus token | exact real name | enum | files affected |
|---|---|---|---|
| `Cumsum` | `CumSum` | `OpKind` (dispatch.rs:269) | vulkan/norm-softmax (cumsum_f32/_f64/_f16/_bf16 — 4 sections) |
| `Clamp` (op_kind) | `ClampInplace` | `OpKind` (dispatch.rs:445) | cpu/inplace-unary-affine (clamp_inplace_f32/_f64/_bf16/_f16 — 4 sections) |
| `PowI` (op_kind) | `PowIInplace` | `OpKind` (dispatch.rs:448) | cpu/inplace-unary-affine (powi_inplace_f32/_f64/_bf16/_f16 — 4 sections) |

No `ConvTranspose1D`/`ConvTranspose2D` rename: `ConvTranspose2D` is already correct (real OpKind);
`ConvTranspose1D` has no OpKind (→ describe-only, below).

## B. DESCRIBE-ONLY set (NO real OpKind/OpParams — mark `registrable: false`, stay documentation)

| file | sections (count) | reason (no real OpKind/OpParams) |
|---|---|---|
| metal/conv-pool | im2col, im2col1d, col2im1d, upsample_nearest2d, upsample_bilinear2d, max_pool2d, avg_pool2d, conv_transpose1d (8) | no `OpKind::{Im2Col,Im2Col1d,Col2Im1d,UpsampleNearest2D,UpsampleBilinear2D,MaxPool2D,AvgPool2D,ConvTranspose1D}`; their `op_params.variant` (Im2Col/Conv1D/ConvTranspose1D/UpsampleNearest2D/UpsampleBilinear2D/MaxPool2D/AvgPool2D) are likewise not real `OpParams` |
| metal/sort-random | asort_asc, asort_desc, carg_block_sort, sort_mbsort, partition_mbsort, merge_mbsort, rand_uniform, rand_normal (8) | no `OpKind::{ArgSort,RandUniform,RandNormal}`; `OpParams::{ArgSort,RandUniform,RandNormal}` absent |
| reference/conv-pool | conv2d_simple, max_pool2d (2) | no `OpKind::{Conv2dSimple,MaxPool2D}` / matching `OpParams` |
| reference/matmul | transpose_2d, transpose_last_two, permute (3) | no `OpKind::{Transpose,Permute}` (zero-copy graph views, no dispatch carrier); `permute` also names non-variant `OpParams::Permute` |
| vulkan/quantized | dequant_q4_0, dequant_q8_0, dequant_q4_km, quantize_q8_0 (4) | `DequantizeQ4_0/Q8_0/Q4KM`, `QuantizeQ8_0` are `Capability` tokens, not `OpKind` |
| vulkan/elementwise | add_assign_scaled (1) | no `OpKind::AddAssignScaled`; `OpParams::AddAssignScaled` absent |
| metal/elementwise | const_set, const_set_strided, copy2d, powf_kernel, powf_kernel_strided, elu_kernel, elu_kernel_strided, fill (8) | already `op_kind: ~` / `op_params.variant: ~` — documentation chassis with no backing OpKind |
| cpu/elementwise-unary | unary (1) | chassis umbrella; `op_kind: ~` + `fused_op: ~`; the named per-op sections below it carry the real OpKind |

Describe-only total = **35 sections** across 8 files. Each adds `registrable: false`. The token in
`op_kind:` / `op_params.variant:` may stay as the forward-looking marker (or `~`); it is no longer
required to resolve.

Note (verify-but-likely-already-passing): `metal/elementwise` `affine_kernel`/`affine_kernel_strided`
use `op_kind: Affine` (real) and `cpu/inplace-unary-affine` `affine_inplace_*` use `op_kind: Affine`
(real) — these are NOT failures and need no change.

---

# Per-defect rows, grouped by crate dir

Notation: `file :: section -> fix`.

## cpu/

### cpu/inplace-unary-affine.fkc.md  (8 TYPO renames)
- `cpu/inplace-unary-affine :: clamp_inplace_f32` -> rename op_kind `Clamp` → `ClampInplace`.
- `cpu/inplace-unary-affine :: clamp_inplace_f64` -> rename op_kind `Clamp` → `ClampInplace`.
- `cpu/inplace-unary-affine :: clamp_inplace_bf16` -> rename op_kind `Clamp` → `ClampInplace`.
- `cpu/inplace-unary-affine :: clamp_inplace_f16` -> rename op_kind `Clamp` → `ClampInplace`.
- `cpu/inplace-unary-affine :: powi_inplace_f32` -> rename op_kind `PowI` → `PowIInplace`.
- `cpu/inplace-unary-affine :: powi_inplace_f64` -> rename op_kind `PowI` → `PowIInplace`.
- `cpu/inplace-unary-affine :: powi_inplace_bf16` -> rename op_kind `PowI` → `PowIInplace`.
- `cpu/inplace-unary-affine :: powi_inplace_f16` -> rename op_kind `PowI` → `PowIInplace`.
  (op_params stay `Clamp`/`PowI` — those ARE real `OpParams` variants, kernel.rs:362/368. `unary_inplace`
  + `affine_inplace_*` in this file already use real OpKinds `ReluInplace`/`Affine` — no change.)

### cpu/shape-ops.fkc.md  (2 YAML `:{` spacing fixes)
- `cpu/shape-ops :: flip` -> YAML field style: `dtype_size:{ kind: usize, ... }` (line 150) → `dtype_size: { ... }` (add space after the colon so serde reads a flow-map, not a scalar).
- `cpu/shape-ops :: roll` -> YAML field style: `dtype_size:{ kind: usize }` (line 222) → `dtype_size: { kind: usize }`.
  (Note `concat` line 297 `input_dim_sizes:{` and `batch_count:{` line 547 carry the SAME `:{` defect —
  apply the identical `: {` spacing fix to every `<field>:{` occurrence in this file while you are in it.)

### cpu/elementwise-unary.fkc.md  (1 declares-neither → describe-only)
- `cpu/elementwise-unary :: unary` -> mark `registrable: false`. This is the shared chassis umbrella
  (`op_kind: ~`, `fused_op: ~`); it deliberately binds no op. The per-op sections below it (relu, neg,
  sqr, … erf, gelu_erf) already carry real OpKinds and are fine.

### cpu/padding.fkc.md  (YAML `:{` — fix while in the file; not in the original 73 count if the lint stops at first error per section, but a genuine defect)
- `cpu/padding :: pad_constant / pad_reflect / pad_replicate / pad_backward` -> `fill_bytes:{ ... }`
  (lines 75/149/220/301) → `fill_bytes: { ... }`. (Defensive: align with the YAML rule applied
  cluster-wide; the fix is identical `:{`→`: {`.)

## reference/

### reference/matmul.fkc.md  (3 describe-only)
- `reference/matmul :: transpose_2d` -> mark `registrable: false` (op_kind `Transpose` has no OpKind;
  op_params already `None`). Keep `op_kind: Transpose` as the forward marker.
- `reference/matmul :: transpose_last_two` -> mark `registrable: false` (same).
- `reference/matmul :: permute` -> mark `registrable: false` (op_kind `Permute` has no OpKind; AND
  `op_params.variant: Permute` is not a real OpParams variant — describe-only exempts the rule-7 check).
  (matmul / matmul_2d use real `MatMul`; eval_qmatmul / dequantize_blocks / dequantize_q4_km_block use
  real `fused_op: QMATMUL` + `op_params QMatMul` — all pass, no change.)

### reference/conv-pool.fkc.md  (2 describe-only)
- `reference/conv-pool :: conv2d_simple` -> mark `registrable: false` (`Conv2dSimple` no OpKind;
  `op_params.variant: Conv2dSimple` not real).
- `reference/conv-pool :: max_pool2d` -> mark `registrable: false` (`MaxPool2D` no OpKind;
  `op_params.variant: MaxPool2D` not real).
  (conv2d / conv_transpose2d in this file use real `Conv2D`/`ConvTranspose2D` — no change.)

### reference/broadcast-binary.fkc.md  (4 layout-incoherence §10.4 fixes)
Kernel behavior (from prose + `ops.rs`): both input *buffers* are physically contiguous, zero-offset;
the broadcast is a LOGICAL relation expanded INTERNALLY (`broadcast_src_flat`). The kernel does NOT
read stride-0 broadcast inputs at the data layer. Per the decision rule → **drop the broadcast claim**:
set `broadcast_stride0: rejected` (keep `strided: rejected`), and KEEP
`awkward_layout_strategy: contiguize_internally`.
- `reference/broadcast-binary :: broadcast_add` -> on operands `a` and `b`: `broadcast_stride0: accepted` → `broadcast_stride0: rejected`. (op_kind `AddElementwise` is correct.)
- `reference/broadcast-binary :: broadcast_sub` -> same `broadcast_stride0: accepted`→`rejected` on `a`,`b`.
- `reference/broadcast-binary :: broadcast_mul` -> same `broadcast_stride0: accepted`→`rejected` on `a`,`b`.
- `reference/broadcast-binary :: broadcast_div` -> same `broadcast_stride0: accepted`→`rejected` on `a`,`b`.
  CAVEAT for the fix agent: `awkward_layout_strategy: contiguize_internally` per rule-5 requires
  `strided: accepted` on the operand. With broadcast dropped and `strided: rejected`, the per-operand
  rule-5 check (`contiguize_internally` ⇒ `strided: accepted`) will fail. Two coherent resolutions —
  pick per the kernel's truth: (a) since these buffers are ALWAYS physically contiguous and the
  expansion is purely internal index math (not a strided WALK), set the kernel-level
  `caps.awkward_layout_strategy: requires_contiguous` and REMOVE the per-operand
  `contiguize_internally` (there is no strided input to contiguize — the operands are contiguous, the
  broadcast is logical); OR (b) if the maintainers want to keep `contiguize_internally` semantics, set
  BOTH `strided: accepted` and `broadcast_stride0: accepted` (model it as a true strided/broadcast
  acceptor). Recommended: (a) — it matches the as-built "contiguous buffers, internal expansion"
  reality and keeps the §10.4 coherence trivially satisfied (`contiguous: required` alone).

### reference/shape-mask-pad.fkc.md  (YAML `:{` spacing fixes)
- `reference/shape-mask-pad :: pad_constant / pad_reflect / pad_replicate` -> `fill_bytes:{ ... }`
  (lines 310/382/453) → `fill_bytes: { ... }` (add space).

## vulkan/

### vulkan/norm-softmax.fkc.md  (4 TYPO renames)
- `vulkan/norm-softmax :: cumsum_f32` -> rename op_kind `Cumsum` → `CumSum` (op_params variant
  `Cumsum` → `CumSum` too: `OpParams::CumSum` is the real variant, kernel.rs:500 / is_op_params_variant
  lists `CumSum`).
- `vulkan/norm-softmax :: cumsum_f64` -> rename op_kind `Cumsum` → `CumSum`; op_params `Cumsum` → `CumSum`.
- `vulkan/norm-softmax :: cumsum_f16` -> rename op_kind `Cumsum` → `CumSum`; op_params `Cumsum` → `CumSum`.
- `vulkan/norm-softmax :: cumsum_bf16` -> rename op_kind `Cumsum` → `CumSum`; op_params `Cumsum` → `CumSum`.
  (All softmax/rms/layer-norm + their backwards in this file use real OpKinds + `SoftmaxLastDim`/
  `NormLastDim` OpParams — no change.)

### vulkan/elementwise.fkc.md  (9 op_params-variant fixes + 1 describe-only)
The `unary*`/`binary*` kernels carry `op_params.variant: Unary`/`Binary`, which are NOT real `OpParams`
variants — elementwise dispatch uses `OpParams::None` (the per-element params ride the wrapper, not an
OpParams variant). Fix: set the variant to `None` (and either drop the `fields:` block or leave it as
a documentation note — the validator only checks the variant token; `None` carries no fields).
- `vulkan/elementwise :: unary` -> set `op_params.variant: Unary` → `None`. (op_kind `ReluElementwise` ok.)
- `vulkan/elementwise :: unary_f16` -> `op_params.variant: Unary` → `None`.
- `vulkan/elementwise :: unary_f64` -> `op_params.variant: Unary` → `None`.
- `vulkan/elementwise :: unary_bf16` -> `op_params.variant: Unary` → `None`.
- `vulkan/elementwise :: binary` -> `op_params.variant: Binary` → `None`. (op_kind `AddElementwise` ok.)
- `vulkan/elementwise :: binary_f16` -> `op_params.variant: Binary` → `None`.
- `vulkan/elementwise :: binary_f64` -> `op_params.variant: Binary` → `None`.
- `vulkan/elementwise :: binary_bf16` -> `op_params.variant: Binary` → `None`.
- (That is 8 `Unary`/`Binary` variant fixes. The 9th op_params failure in the task tally is the
  `ConvShape` case in conv-attn/conv — see below; it is NOT in this file.)
- `vulkan/elementwise :: add_assign_scaled` -> mark `registrable: false` (op_kind `AddAssignScaled`
  has no OpKind; `op_params.variant: AddAssignScaled` not real). Describe-only exempts both rule-2 and
  rule-7. (affine/affine_f16/affine_f64/affine_bf16 use real `Affine`+`Affine`; clamp uses real
  `ClampElementwise`+`Clamp`; powi uses real `PowIElementwise`+`PowI` — all pass, no change.)

### vulkan/quantized.fkc.md  (4 describe-only)
- `vulkan/quantized :: dequant_q4_0` -> mark `registrable: false` (`DequantizeQ4_0` is a Capability
  token, not an OpKind; op_params already `None`). Keep `op_kind: DequantizeQ4_0` as the marker.
- `vulkan/quantized :: dequant_q8_0` -> mark `registrable: false` (`DequantizeQ8_0`).
- `vulkan/quantized :: dequant_q4_km` -> mark `registrable: false` (`DequantizeQ4KM`).
- `vulkan/quantized :: quantize_q8_0` -> mark `registrable: false` (`QuantizeQ8_0`).
  (qmatvec_q4_0 / matmul_q4_0_tiled use real `fused_op: QMATMUL` + `op_params QMatMul` — no change.)

### vulkan/data-movement.fkc.md  (YAML `:{` spacing fixes)
- `vulkan/data-movement :: <the 4 write-slice sections at lines 1252/1330/1408/1483>` -> `shape_buf:{ kind: "storage<u32>", ... }` → `shape_buf: { ... }` (add space after the colon) for each occurrence.

## metal/

### metal/conv-pool.fkc.md  (8 describe-only)
- `metal/conv-pool :: im2col` -> mark `registrable: false` (`Im2Col` no OpKind; `op_params.variant: Im2Col` not real).
- `metal/conv-pool :: im2col1d` -> mark `registrable: false` (`Im2Col1d` no OpKind; variant `Conv1D` exists but no matching OpKind — describe-only exempts rule-7 anyway).
- `metal/conv-pool :: col2im1d` -> mark `registrable: false` (`Col2Im1d` no OpKind; variant `ConvTranspose1D` exists but no matching OpKind).
- `metal/conv-pool :: upsample_nearest2d` -> mark `registrable: false` (`UpsampleNearest2D` no OpKind / OpParams).
- `metal/conv-pool :: upsample_bilinear2d` -> mark `registrable: false` (`UpsampleBilinear2D` no OpKind / OpParams).
- `metal/conv-pool :: max_pool2d` -> mark `registrable: false` (`MaxPool2D` no OpKind / OpParams).
- `metal/conv-pool :: avg_pool2d` -> mark `registrable: false` (`AvgPool2D` no OpKind / OpParams).
- `metal/conv-pool :: conv_transpose1d` -> mark `registrable: false` (`ConvTranspose1D` no OpKind;
  `OpParams::ConvTranspose1D` exists but no matching OpKind to key against).
  (conv_transpose2d uses real `ConvTranspose2D` OpKind + OpParams — REGISTRABLE, no change.)

### metal/sort-random.fkc.md  (8 describe-only)
- `metal/sort-random :: asort_asc` -> mark `registrable: false` (`ArgSort` no OpKind; variant `ArgSort` not real).
- `metal/sort-random :: asort_desc` -> mark `registrable: false` (`ArgSort`).
- `metal/sort-random :: carg_block_sort` -> mark `registrable: false` (`ArgSort`).
- `metal/sort-random :: sort_mbsort` -> mark `registrable: false` (`ArgSort`).
- `metal/sort-random :: partition_mbsort` -> mark `registrable: false` (`ArgSort`).
- `metal/sort-random :: merge_mbsort` -> mark `registrable: false` (`ArgSort`).
- `metal/sort-random :: rand_uniform` -> mark `registrable: false` (`RandUniform` no OpKind; variant `RandUniform` not real). NOTE: this section also has `accept.inputs: []` (no inputs — a random fill) which would fail rule-2 `≥1 input`; describe-only does NOT exempt the ≥1-input rule. Fix agent must EITHER extend describe-only to also relax the input/output-presence requirement for no-input fills, OR (preferred, smaller blast radius) keep the input-presence rule and confirm the lint currently fails this section on op_kind FIRST — if so, marking describe-only + a follow-up for the empty-inputs case. Flag to maintainers: random fills with zero graph operands need a describe-only carve-out for `accept.inputs` too.
- `metal/sort-random :: rand_normal` -> mark `registrable: false` (`RandNormal`); same empty-inputs caveat as rand_uniform.

### metal/elementwise.fkc.md  (8 declares-neither → describe-only)
These already carry `op_kind: ~` AND `op_params.variant: ~` (no `fused_op`), so they fail rule-2
exactly-one-of. They are documentation chassis kernels with no backing dispatch op.
- `metal/elementwise :: const_set` -> mark `registrable: false`.
- `metal/elementwise :: const_set_strided` -> mark `registrable: false`.
- `metal/elementwise :: copy2d` -> mark `registrable: false`.
- `metal/elementwise :: powf_kernel` -> mark `registrable: false`.
- `metal/elementwise :: powf_kernel_strided` -> mark `registrable: false`.
- `metal/elementwise :: elu_kernel` -> mark `registrable: false`.
- `metal/elementwise :: elu_kernel_strided` -> mark `registrable: false`.
- `metal/elementwise :: fill` -> mark `registrable: false`.
  (unary_kernel / unary_kernel_strided use `ExpElementwise`; binary_kernel(_strided) use
  `AddElementwise`; affine_kernel(_strided) use `Affine` — all real, no change.)

### metal/indexing.fkc.md  (YAML `:{` spacing fix)
- `metal/indexing :: index_add` (line 404) -> `base_dim_size:{ kind: usize, ... }` → `base_dim_size: { ... }` (add space).

## dispatch/

### dispatch/elementwise-binary.fkc.md  (1 multiple-fkc-blocks section → split into N)
Multiple `## ` sections each carry one ` ```fkc ` block per backend (CPU/CUDA/Vulkan), violating
parse.rs's "exactly one fkc block per section." Each block is already a complete, valid contract with
its own unique `kernel:` name (e.g. `add_elementwise_cpu` / `_cuda` / `_vulkan`). Fix: **split each
multi-block section into N separate `## ` sections**, one per backend block, titled from the block's
`kernel:` name (e.g. `## add_elementwise_cpu`, `## add_elementwise_cuda`, `## add_elementwise_vulkan`),
each containing exactly one ` ```fkc ` block. Apply to every multi-block section in the file
(`add_elementwise`, `sub_elementwise`, `mul_elementwise`, `div_elementwise`, `maximum_elementwise`,
`minimum_elementwise`, `pow_elementwise`, `rem_elementwise`, and any other 3-block section). The op_kinds
themselves (`AddElementwise`, … `RemElementwise`) and `op_params: { variant: None }` are all correct —
this is purely a section-splitting fix, no op_kind/op_params change.

### dispatch/reduce.fkc.md  (1 multiple-fkc-blocks section → split into N)
Same defect: `## SumReduce` / `## MaxReduce` / `## MinReduce` / `## MeanReduce` each carry 3 per-backend
` ```fkc ` blocks (e.g. one CPU + CUDA + Vulkan). Fix: split each into N single-block `## ` sections by
the block's `kernel:` name. The op_kinds (`SumReduce`/`MaxReduce`/`MinReduce`/`MeanReduce`/`ReduceSumTo`/
`ReduceMaxTo`/`ReduceMaxToBackward`/`ArgMaxDim`) and `op_params variant: Reduce`/`ReduceSumTo`/… are all
real and correct — section-splitting only.

### dispatch/shape-ops.fkc.md  (YAML `:{` spacing fix)
- `dispatch/shape-ops :: pad` (line 106) -> `fill_bytes:{ kind: "Vec<u8>", ... }` → `fill_bytes: { ... }` (add space).

## conv-attn/

### conv-attn/conv.fkc.md  (1 op_params-variant fix → describe-only)
- `conv-attn/conv :: im2col` -> `op_params.variant: ConvShape` is a `fuel-conv` type, NOT an `OpParams`
  variant (rule-7 `BadOpParamsVariant`); `op_kind: Conv2D` is real but im2col has no standalone Conv2D
  carrier (it is the lowering strategy). Fix: **mark `registrable: false`** (describe-only — im2col is a
  conv-lowering building block, not a dispatched op; describe-only exempts the rule-7 ConvShape variant
  check and the op_kind requirement). Keep the prose note that the carrier is `OpKind::Conv2D`'s im2col
  lowering. (conv2d_direct + conv2d_via_gemm use real `Conv2D` + `OpParams::Conv2D` — no change. This is
  the 9th "op_params variant not a real OpParams" failure in the tally: 8 `Unary`/`Binary` in
  vulkan/elementwise + 1 `ConvShape` here.)

## quantized/

### quantized/dequantize.fkc.md  (1 no-fkc-block section → remove)
- `quantized/dequantize :: to_float_q8_1` -> **remove the section** (or its lint-visibility). It has NO
  ` ```fkc ` block (parse.rs `MissingFkcBlock`) because `GgmlType::to_float` for Q8_1 is
  `unimplemented!()` and panics — there is no working kernel to contract. The section is prose-only by
  design. Fix: delete the `## to_float_q8_1` section so the parser sees no zero-block section. (Preserve
  the bundle-header note + the return-summary mention that Q8_1 has no dequant; move any prose worth
  keeping into the file's intro, not a `## ` section.) All to_float_q4_0…q8k sections use real `QMatMul`
  op_kind + `QMatMul` op_params and pass — no change.

---

# Defect-count reconciliation (73 HARD failures)

- 36 "unknown op_kind" → split as: **8 TYPO renames** (Cumsum×4 → CumSum; Clamp×… see below) + **28
  NO-OPKIND describe-only**. Precisely:
  - TYPO renames among the unknown-op_kind bucket: `Cumsum`×4 (vulkan/norm-softmax) +
    `Clamp`×4 + `PowI`×4 (cpu/inplace-unary-affine) = **12 renames** (table A; these 12 are the
    op_kind-rename rows, all in the unknown-op_kind tally because the lint reported the wrong token).
  - NO-OPKIND describe-only from the unknown-op_kind bucket: Im2Col, Im2Col1d, Col2Im1d, MaxPool2D
    (metal + reference), AvgPool2D, Conv2dSimple, ConvTranspose1D, UpsampleNearest2D,
    UpsampleBilinear2D, Transpose×2, Permute, ArgSort×6, RandUniform, RandNormal, AddAssignScaled,
    DequantizeQ4_0, DequantizeQ8_0, DequantizeQ4KM, QuantizeQ8_0 = the remaining unknown-op_kind rows
    → describe-only.
- 9 "op_params variant not a real OpParams" → 8 `Unary`/`Binary` → `None` (vulkan/elementwise) + 1
  `ConvShape` (conv-attn/conv → describe-only).
- 9 "declare neither op_kind nor fused_op" → 8 metal/elementwise chassis + 1 cpu/elementwise-unary
  `unary` → describe-only.
- 6 "op_params field YAML `:{` style" → dispatch/shape-ops (pad), reference/shape-mask-pad (×3 sections),
  cpu/shape-ops (flip/roll dtype_size), metal/indexing (index_add), vulkan/data-movement (×4 write-slice).
  (Plus cpu/padding `fill_bytes:{`×4 — same defect, fix cluster-wide while in the tree.)
- 4 "layout incoherence (§10.4)" → reference/broadcast-binary broadcast_add/sub/mul/div
  (`broadcast_stride0: accepted` with `strided: rejected` → drop broadcast + requires_contiguous).
- 2 "multiple fkc blocks in one section" → dispatch/elementwise-binary, dispatch/reduce (split per-backend).
- 1 "section with no fkc block" → quantized/dequantize :: to_float_q8_1 (remove the section).

The describe-only set is intentionally larger than the raw "unknown op_kind" subset because it also
absorbs the 9 declares-neither chassis sections, the 8 `Unary`/`Binary`-but-describe-only? — NO: the
`Unary`/`Binary` ones are FIXED to `None` (registrable), only `add_assign_scaled` + `ConvShape`-im2col
go describe-only. Net describe-only sections = 35 (table B).

# Net fix actions per the decisions

- RENAME op_kind to exact real OpKind: 12 sections (Table A).
- Set op_params to `None` (real variant): 8 sections (vulkan/elementwise unary/binary).
- Fix YAML `:{` → `: {` (+ keep bare-kind values quoted as already written): all `<field>:{`
  occurrences across dispatch/shape-ops, reference/shape-mask-pad, cpu/shape-ops, metal/indexing,
  vulkan/data-movement (+ cpu/padding defensively).
- Split-into-N-sections: 2 files (dispatch/elementwise-binary, dispatch/reduce) — each multi-block
  `## ` section becomes N single-block `## ` sections named by `kernel:`.
- Mark `registrable: false` (describe-only, NEW marker this round): 35 sections (Table B + conv-attn/conv
  im2col).
- Fix layout (§10.4): 4 sections (reference/broadcast-binary) — drop `broadcast_stride0`, use
  `requires_contiguous` (buffers are physically contiguous; broadcast is internal).
- Remove no-fkc-block stub: 1 section (quantized/dequantize to_float_q8_1).

No OpKind/OpParams/FusedOpParams variant is invented. No validator is relaxed to hide a defect — the
only validator change is the explicit, documented `registrable: false` describe-only carve-out (with a
flagged follow-up for zero-input random fills in metal/sort-random).
