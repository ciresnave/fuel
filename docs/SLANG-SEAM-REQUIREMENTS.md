# What the Slang side needs from the Frontend/Backend seam

Written 2026-10-28 by the Fuel lane, at the PM's request (CireSnave, first-hand, 2026-10-28: *"In
addition to Baracuda's CUDA parser and emitter, Fuel will be contributing a Slang parser and emitter to
Unpopped. How far along is Fuel on that? I ask because having Unpopped and Baracuda decide that seam
could make the seam very CUDA-centric."*). Addressed to the Unpopped lane and the FireLine lane.

**Docs only; no code.** Every count below is a measurement over Fuel `origin/main` at `2eed77789`
(147 files in `fuel-kernels-source/kernels/*.slang`, read from a `git archive` of that ref, not a
working tree), by a throwaway script that regexes the sources. Regexes are not a Slang parser:
a count says "this text appears in N files", and a comment can match, so I name the definition
beside each number. Where I could not measure something, it says so.

## 1. Where Fuel's P3 stands

ROADMAP row 101: **NOT STARTED.** No Slang parser or emitter code exists in Fuel. It waits on
Unpopped's IR design merging (the interface comes from that design doc). What I can say today:

- Fuel owns the corpus (147 `.slang` kernels, all post-fork Fuel work; Vulkane has none) and the
  host-side contract those kernels are dispatched against (`fuel-vulkan-backend`).
- Unpopped already ships an in-tree Slang **emitter** (`unpopped-slang` 0.29.0) and a Slang lifter
  (`Frontend` `convert::SLANG`). The CUDA side shaped the Frontend/Backend seam first. Unpopped's own
  measurement: its Slang front end accepts **0 of 147** (4/147 after a buffer-naming fix:
  `docs/slang-naming-convention-note.md` on Unpopped `origin/main`). I have **not** re-run that;
  I read their docs.
- So the honest state is: the seam exists, it was drawn against CUDA, and Fuel has not yet tested it
  against Slang's needs. This document is that test, done on paper, so it can happen before the IR
  design freezes rather than after.

**What Fuel needs from Unpopped to start:** (a) the IR design merged; (b) a decision on the
questions in section 4 (above all: does a generated kernel carry a structured ABI manifest, and is the
"Fuel binding layout" a declared input to the Frontend or something it guesses from names); (c) one
named owner on each side for the manifest type.

## 2. What I could and could not read of the current traits

Read (Unpopped `origin/main` `43dda06`, read-only): `crates/unpopped/src/backend.rs` (the `Backend`
trait: `name`, `provider`, `lower(&KernelPlan) -> GeneratedKernel`, `supports_dtype(ElementKind,
TargetId)`, `effective_count_width`, `lower_variants`; `GeneratedKernel { name, source: String }`,
`#[non_exhaustive]`, with a doc comment that a SPIR-V word stream plus an ABI/binding manifest
"can't ride a `String`" and was reserved for a minor version), the signatures in
`crates/unpopped/src/convert.rs` (`Frontend`, `SLANG`, `lift_elementwise/_reduction/_scan`),
the header of `crates/unpopped-slang/src/lib.rs`, `docs/slang-corpus-sizing.md` and
`docs/slang-second-blocker-sizing.md`.

**Not read:** `ir.rs`, `plan.rs`, the body of `convert.rs`, `docs/normative-seams.md`,
`docs/ir-union-surveys/slang.md`, and the not-yet-merged IR design. Anything below about those is
unknown to me, not asserted.

What the traits visibly assume, from the part I read: a kernel is one `String` of source text plus a
symbol name; the Slang emitter writes a contiguous scalar elementwise kernel only (`F32/F64/I32/I64`;
`F16`/`Bf16`/`U8`/`S8`/`U32` declined; any other schedule panics); buffers are named `output`/`inputK`;
there is no `[[vk::binding]]`, no parameter block, and no bounds guard (the harness is
"exact-dispatch"), so an emitted kernel cannot yet be dispatched by Fuel's backend as it stands.

## 3. Requirements, each with the kernels that show it

Corpus facts used throughout: **147 of 147** kernels declare every resource with an explicit
`[[vk::binding(n, 0)]]` in **descriptor set 0 only**; **0** use push constants, specialization
constants, `import`, or preprocessor directives; **147 of 147** end with a `ConstantBuffer<Params>` at
the highest binding index (checked: no kernel breaks the rule).

### R1. A binding/ABI manifest, not a source string

Fuel's Vulkan backend does not discover bindings. It builds fixed pipeline layouts keyed by shape
(`layout_2s1u`, `layout_3s1u`: N storage buffers then one uniform buffer;
`fuel-vulkan-backend/src/lib.rs`) and writes descriptors by index, then writes the parameter block at
the next index. A kernel is dispatchable only if its bindings are exactly that sequence.
- Evidence: `add_assign_scaled.slang` (`dst` RW at 0, `src` RO at 1, `Params` at 2). Bindings per
  kernel: 3 in 69 files, 4 in 64, 5 in 12, 6 in 1, 7 in 1 (count of `vk::binding` attributes).
- Needed in the artifact: per binding, index, role (read / read-write), storage-vs-uniform, element
  type and packing (R6), and the **byte layout of `Params`** (field order and offsets; today the host
  writes a `repr(C)` struct that must match the shader's `ConstantBuffer` layout by hand).
- CUDA contrast: a CUDA kernel takes positional pointer arguments plus by-value scalars; there is no
  descriptor index, no set, and no uniform/storage split. A seam that only says "name + source"
  loses all of this for Slang.

### R2. SPIR-V artifact, with the build flags part of its identity

Fuel commits the compiled `.spv` (177 lines mention `include_bytes!` in `fuel-vulkan-kernels/src/lib.rs`, one of them a doc comment; the embedded table is about that long),
built by `fuel-kernels-source/kernels/compile.sh` as
`slangc <src> -target spirv -profile glsl_450 -entry <entry>`, then an optional Python pass
(`hoist_extinst_imports.py`) that moves `OpExtInstImport` out of function bodies. **None of the 147
current sources uses `spirv_asm`**, so the pass is dormant, not exercised.
- Needed: the artifact is a SPIR-V word stream (plus the manifest), and the toolchain/flags that
  produced it are recorded, because they change device behaviour with the source unchanged.
  Concrete case (Unpopped's finding, `unpopped-slang` header, not re-measured by Fuel): without
  `-denorm-mode-fp32 preserve` an RTX 4070 Laptop flushes f32 subnormals. Fuel's `compile.sh` does
  **not** pass that flag, so Fuel's committed SPIR-V declares no denorm mode. I have not measured what
  that does to Fuel's kernels; it is a question this seam makes visible.
- CUDA contrast: PTX/cubin vs source text is the same artifact-kind question, but CUDA's flags
  (`-arch`) travel with the target; Slang's numerics flags do not live in the source text.

### R3. Entry points, workgroup size, and dispatch geometry

- 143 files have one entry named `main`. **4 files** (`autodiff_poc`, `triangular_b2`, `triangular_b4`,
  `triangular_b8`) have **two** entries, and the entry names live in a **comment directive**
  (`// entries: triu_b2, tril_b2`) read by `compile.sh`, not in the language. A parser that reads
  Slang will see the entries; a build that reads the directive will not agree unless the manifest
  carries the names.
- `[numthreads(...)]`: `(256,1,1)` in 138 files, `(64,1,1)` in 9, `(32,1,1)`, `(16,16,1)`,
  `(128,1,1)` in one each. The host computes the group count from a Rust constant
  (`workgroups(n) = n.div_ceil(256)` in `fuel-vulkan-backend`), so **the workgroup size is an ABI fact
  the host must be told**, not something it can read back at run time.
- Needed: entry names, workgroup size, and the rule that maps the problem shape to group counts.

### R4. Subgroup operations

- **27 files** use subgroup intrinsics: `WaveActiveSum` (92 uses), `WaveIsFirstLane` (70),
  `WaveGetLaneCount` (54), `WaveActiveMax` (14), `WaveActiveMin` (5). Examples:
  `layer_norm_last_dim*.slang`, `reduce_last_dim*.slang`.
- `WaveGetLaneCount` is used to size per-subgroup partials, so the kernel's correctness depends on a
  **device-dependent lane count**. CUDA's warp is the constant 32; a Vulkan subgroup is 32, 64, or
  another size, and may differ between adapters (this box has an NVIDIA and an AMD adapter; I have not measured either's subgroup size).
- Needed: subgroup size and subgroup-arithmetic support as **target properties** (the Backend's
  `supports_dtype(kind, target)` shape, extended to a capability query), and a schedule vocabulary
  that does not hard-code 32.

### R5. Shared memory, barriers, atomics

- `groupshared` + `GroupMemoryBarrier*` in **36 files** (families: `arg_reduce_last_dim`, `reduce`,
  `reduce_last_dim`, `softmax`, `softmax_last_dim_backward`, `layer_norm_last_dim` and its backward,
  `rms_norm_last_dim` and its backward, `flash_attention`, `matmul_q4_0_tiled`, `qmatvec_q4_0`);
  `Interlocked*` in **19** files (`index_add`, `scatter_add`, `pad_backward_reflect`,
  `pad_backward_replicate`, `concat_along_dim`, `reduce_last_dim`, `quantize_q8_0`).
  Unpopped's own run classes these as its "residue" bucket (53 files refused as Inexpressible).
- Needed: shared tiles, a barrier, and read-modify-write atomics (Unpopped says 16 files use
  compare-exchange, which is how Slang's float atomic add is written there) as first-class schedule
  elements, with the same decline discipline CUDA gets.

### R6. Storage type is not compute type; packed 16-bit and quantized blocks

- bf16 and f16 are stored **two per `u32`** in `StructuredBuffer<uint>` (**89 files**), widened in
  registers (`asfloat(b << 16u)` for bf16). `half` appears in 35 files and needs the 16-bit storage and
  `shaderFloat16` features; `double` in 23 (`shaderFloat64`); `uint16_t` in 4; fp8 conversions in 6.
- **6 files** read quantized blocks from `ByteAddressBuffer` (`dequant_q4_0`, `dequant_q4_km`,
  `dequant_q8_0`, `matmul_q4_0_tiled`, `qmatvec_q4_0`, `quantize_q8_0`).
- Needed: per-binding storage element and packing, separate from the arithmetic dtype; required device
  features per kernel; and a block-layout vocabulary for the quantized reads. Unpopped's emitter
  declines `F16`/`Bf16` today, so this is the largest single dtype gap against Fuel's corpus.

### R7. Strided and broadcast access

- **31 kernels** declare stride fields in `Params` (parsed: a `Params` field named `*stride*`, `*_sN`
  or `sN`); they spell a rank of at most **4** (26 of the 30 that name a rank spell four; the rest
  two), with explicit `shape0..3` and `*_s0..3` fields. **12** carry a `flags` bit 0
  "input contiguous" that selects a fast path at run time.
  Examples: `affine.slang`, `binary.slang`, `unary.slang`, `cumsum*`, `rope*`, `reduce*`,
  `scatter_add*`.
- This is the count I can stand behind. The PM message and Unpopped's note say "up to 70 of 147":
  that is a different, looser definition (I get 90 files if I count the word "stride" anywhere,
  comments included, and 31 by `Params` fields), and I did not reproduce 70. The difference is a
  definition, not a contradiction, but I have not reconciled it.
- Needed: an access pattern as a first-class IR fact (affine index map whose coefficients are
  **run-time `Params`**, plus broadcast as stride 0), and the contiguous fast path expressed as a
  **variant** the dispatcher chooses, not a data-dependent branch the lifter has to model. Unpopped's
  note already found the flat-index model cannot represent this.

### R8. Numerics that are Slang-specific

- `binary*.slang` implement the KISS-OPS §6.13 NaN half by hand over `max`/`min`
  (`isnan(x) ? x : isnan(y) ? y : max(x, y)`). **Fuel GAP-276** records that the ±0 half of §6.13
  is **not satisfied on Vulkan and not fixable at source level on one measured driver** (RTX 4070
  Laptop, NVIDIA 610.88): SPIR-V `FMax`/`FMin` canonicalize the ±0 tie, and two source-level
  rewrites (a float select, a bit-cast select) were both re-fused by the driver into a canonicalizing
  hardware min/max. The gap's own lesson applies to this seam: `spirv-dis` shows what is emitted,
  never what executes. So an emitter's `max`/`min` spelling cannot be accepted on inspection of its
  output; it needs a device-run conformance check (Unpopped's harness already runs such checks).
- Subnormal mode: see R2.

### R9. Things the corpus does NOT need (so the seam need not carry them for Fuel's sake)

- No specialization constants, push constants, imports, preprocessor, or generics (the four text
  matches for "generic/extension" are comments). All variation today is separate files per dtype.
  Nothing here argues against supporting them later; nothing requires them now.
- `[Differentiable]`/`bwd_diff` appears in **2** files (`autodiff_poc`, `rms_norm_last_dim`). The
  Frontend should decline them with a typed reason rather than mis-lift.

### R10. Outside the 147

27 further kernels are **GLSL**, not Slang (flash attention, cooperative-matrix matmul, `matvec*`,
`conv2d_im2col_bf16`). Cooperative matrix (`VK_KHR_cooperative_matrix`) has no analog in the current
Backend trait except through CUDA's tensor-core path. They are not in this count, but a seam designed
to carry "the Vulkan side" will meet them next.

## 4. Questions the seam has to answer before it freezes

1. Does `GeneratedKernel` carry a **structured manifest** (bindings, `Params` layout, entry names,
   workgroup size, required features) alongside a SPIR-V stream? (The existing doc comment reserves
   room; Fuel's view is that without it a Slang kernel cannot be dispatched by anyone.)
2. Is Fuel's binding layout a **declared input** to the Slang Frontend (read `[[vk::binding]]` and the
   `Params` struct) or a naming convention it infers? The 0/147 → 4/147 result says inference by
   buffer name is brittle; reading the declarations is the other path.
3. Is `supports_dtype(kind, target)` the model for **all** capability questions (subgroup size,
   16-bit storage, float64, cooperative matrix), or is there a capability-set type?
4. Where do numerics flags (denorm mode) live: in the artifact, the target, or the plan?
5. Who maintains the Fuel-layout dialect: Fuel (ROADMAP row 101) or Unpopped?

## 5. How this was measured (so it can be reproduced or refuted)

`git archive origin/main fuel-kernels-source` into a scratch directory; a Python script counted, per
`.slang` file, regex matches for the constructs named above (comments included unless stated). Positive
control: the script finds 147 `.slang` files (the same population as `git ls-files`) and all 147
`[shader("compute")]` entries. Known weaknesses: regexes over text, so `generic`/`stride` can match
comments; `Params` field parsing assumes a struct literally named `Params` (one file, `autodiff_poc`,
has none); rank is read from field names, not from index math. None of this was run through a Slang
parser, which is the thing P3 would build.
