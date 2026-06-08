# StableHLO → Fuel op map (and Fuel completeness audit)

**Status**: v0.1 (2026-06-08). Source of truth for StableHLO: <https://openxla.org/stablehlo/spec> (119 ops, fetched 2026-06-08). Fuel side: `fuel-graph/src/lib.rs` `Op` enum + the fused-op registry (`fuel-graph/src/registry/`). Companion to [architecture §13-interchange](../architecture/13-interchange.md) and the migration plan (`../session-prompts/model-interchange-import-export-plan.md`).

Two jobs: (1) the **StableHLO→Fuel import map** for later import work; (2) a **completeness audit** of Fuel's primitive vocabulary against the op set XLA's designers consider functionally complete.

## Disposition legend

Every op gets a disposition. Nothing is "out of scope" — ops Fuel doesn't model as IR nodes are handled by a *named mechanism at another layer or at import time*, and that mechanism is what the importer must implement.

| Code | Meaning |
|------|---------|
| **P** | **Primitive** — maps 1:1 to a Fuel `Op` primitive. |
| **D** | **Decompose** — maps to a short sequence of Fuel primitives. |
| **F** | **Fused** — maps to an existing Fuel `Op::Fused(...)` registry entry. |
| **L** | **Import-time lowering** — resolved by a graph transformation at import (unroll / constant-fold / inline region / real-pair emulate / elide). *Not* an IR op. The importer owns this. |
| **X** | **Other Fuel layer** — handled by an existing mechanism outside the primitive IR: multi-output bundle, scheduler/ordering, weight-interchange quant, cross-device `Copy`/`Move`, or inference orchestration. |
| **G** | **Gap** — in-scope but unmapped today: a candidate new primitive or fused op. *The actionable output of this audit.* |

### Principle: representation ≠ op

A capability can live **in Fuel's graph** without ever becoming an **`Op`**. A conditional is the clearest case: it is *expressible* in the DAG today — constant predicate folds to the taken branch; a data-dependent, side-effect-free conditional becomes **predication** (compute both branches, `Where`-select); a loop with a static bound **unrolls** into nodes. None of these adds a control-flow op; all of them are representable. So "Fuel has no `if`/`while`/`scan` op" must not be read as "Fuel can't import control flow" — it imports control flow by *lowering it to graph structure*. The `L` disposition is exactly this: the computation is representable, the op vocabulary stays minimal. Only a genuinely unbounded, data-dependent, side-effecting loop has no graph representation and earns a hard-reject.

---

## ELEMENTWISE_UNARY (21)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `abs` | P | `Abs` |
| `ceil` | P | `Ceil` |
| `convert` | P | `Cast(dtype)` |
| `cosine` | P | `Cos` |
| `exponential` | P | `Exp` |
| `floor` | P | `Floor` |
| `log` | P | `Log` |
| `logistic` | P | `Sigmoid` |
| `negate` | P | `Neg` |
| `round_nearest_even` | P | `Round` (Fuel `Round` is banker's/half-to-even) |
| `rsqrt` | P | `Rsqrt` |
| `sign` | P | `Sign` |
| `sine` | P | `Sin` |
| `sqrt` | P | `Sqrt` |
| `tanh` | P | `Tanh` |
| `exponential_minus_one` | D | `Exp` → `AddScalar(-1)` (expm1; minor precision loss near 0) |
| `log_plus_one` | D | `AddScalar(1)` → `Log` (log1p; minor precision loss) |
| `cbrt` | D | `Pow` by 1/3 (or `Exp(Log(x)/3)`) |
| `tan` | D | `Sin` ÷ `Cos` (no `Tan` primitive) |
| `round_nearest_afz` | D | floor(x + 0.5·sign(x)) — Fuel `Round` is half-to-**even**, so afz needs decomposition |
| `reduce_precision` | D | `Cast(narrow)` → `Cast(wide)` round-trip as approximation |

## ELEMENTWISE_BINARY (9)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `add` | P | `Add` |
| `subtract` | P | `Sub` |
| `multiply` | P | `Mul` |
| `divide` | P | `Div` |
| `maximum` | P | `Maximum` |
| `minimum` | P | `Minimum` |
| `power` | P | `Pow` |
| `remainder` | P | `Rem` (both PyTorch sign-of-divisor convention) |
| `atan2` | **G** | **no inverse-trig in Fuel.** Candidate: `Atan2` primitive (+ `Atan`/`Asin`/`Acos`). |

## COMPARISON (2)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `compare{EQ,NE,LT,LE,GT,GE}` | P | `Equal` / `Ne` / `Lt` / `Le` / `Gt` / `Ge` (→ U8) |
| `is_finite` | D | comparison-based: `Eq(x,x)` ∧ `abs(x) ≠ inf` via compare + `Mul` |

## BITWISE / INTEGER (9)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `and` | D / G | boolean-mask (U8): `Mul` or `Minimum`. **Integer bitwise: gap.** |
| `or` | D / G | boolean-mask: `Maximum`. **Integer bitwise: gap.** |
| `not` | D / G | boolean-mask: `1 - x` via `MulScalar(-1)`+`AddScalar(1)`. **Integer: gap.** |
| `xor` | G | **no bitwise op.** Candidate `BitXor` if a bit-manip model is imported. |
| `shift_left` | G | candidate `ShiftL` (rare in NN; RNG/quant-pack use it) |
| `shift_right_arithmetic` | G | candidate `ShiftRA` |
| `shift_right_logical` | G | candidate `ShiftRL` |
| `popcnt` | G | candidate `PopCount` |
| `count_leading_zeros` | G | candidate `Clz` |

→ The integer-bitwise family is a coherent **gap cluster**; add it only when a consumer model genuinely does bit manipulation (most NN graphs never touch these). Boolean-mask logic is already expressible.

## COMPLEX (3)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `complex` | L | **real-pair emulation**: pack real+imag into a trailing dim of 2 (or paired tensors) via `Concat`/`Unsqueeze` |
| `real` | L | `Slice`/`IndexSelect` the real lane |
| `imag` | L | `Slice`/`IndexSelect` the imag lane |

→ Fuel dtypes are real-only by design; complex is emulated as paired real tensors at import. Complex *arithmetic* expands to the standard real formulas. (Pairs with `fft` below — if `fft` lands as a fused op it consumes/produces this layout.)

## REDUCTION (2, region-carrying)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `reduce` | L → P/F/G | **inline + recognize the `body` combiner**: add→`SumDim`/`ReduceSumTo`, max→`MaxDim`/`ReduceMaxTo`, min→`MinDim`, mean is post-div. Unrecognized combiner (e.g. **product → no `ProdDim`: gap**) or arbitrary body → decompose or gap. |
| `reduce_window` | F / G | windowed reduction = pooling. **Candidate fused `Pool`** (max/avg) — baracuda ships a pool family; Fuel has no caller yet. |

## REGION_OP — non-reduction (4, region-carrying)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `map` | L | **inline the `computation` region** as an elementwise Fuel subgraph applied over the operands |
| `scatter` | P / D | add-combiner → `ScatterAdd` / `IndexAdd`; overwrite-combiner → decompose (mask + `Where`) or partial gap |
| `sort` | **G** | **no general sort.** Fuel has `ArgMaxDim`/`ArgMinDim` only. Candidate fused `Sort`/`TopK` (needed for top-k sampling). |
| `select_and_scatter` | D / G | max-pool backward; decompose via mask + `ScatterAdd`, or gap (training-time, rare) |

## CONTROL_FLOW (3, region-carrying)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `if` | L | **constant pred → fold to the taken branch**; data-dependent + cheap side-effect-free branches → predicate both via `Where`; else recognize-or-error |
| `case` | L | constant index → fold; else predicated `Where` chain or error |
| `while` | L | **static trip count → unroll into the DAG**; recognized recurrence (scan) → `Fused(SELECTIVE_SCAN/SSD_CHUNK_SCAN/CAUSAL_CONV1D)`; truly data-dependent unbounded → the one honest hard-reject (clear `Result` error) |

→ Fuel stays a DAG ([03-ir](../architecture/03-ir.md)). Control flow is an **importer responsibility**, never an IR op. Traced *inference* graphs arrive already unrolled, so this path mostly bites on un-unrolled JAX `lax.scan`/`while_loop`.

## DISTRIBUTED / collectives (8)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `all_gather` | L / X | single-replica → **identity (elide)**; multi-replica → scheduler + cross-device `Copy`/`Move`; true cross-process collective deferred to multi-node (a current non-goal, [09](../architecture/09-non-goals.md)) |
| `all_reduce` | L / X | single-replica → identity; else as above |
| `all_to_all` | L / X | single-replica → identity; else as above |
| `reduce_scatter` | L / X | single-replica → identity; else as above |
| `collective_broadcast` | L / X | single-replica → identity; else `Copy` |
| `collective_permute` | L / X | single-replica → identity; else scheduler transfer |
| `partition_id` | L | fold to `Const(0)` (single partition) |
| `replica_id` | L | fold to `Const(0)` (single replica) |

→ Importing a single-device shard makes collectives vanish. Genuine multi-process semantics belong to a future multi-node scheduler layer, not the IR.

## DYNAMIC_SHAPE (9)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `dynamic_slice` | P / L | static start known → `Slice`; runtime start → `IndexSelect`/`Gather` or per-shape monomorphization |
| `dynamic_update_slice` | P / L | → `WriteSlice` (static) / `WriteSliceRotating` (wrapping) / monomorphize |
| `dynamic_broadcast_in_dim` | L | resolve `output_dimensions` (usually constant-foldable) → `BroadcastTo` |
| `dynamic_reshape` | L | resolve `output_shape` → `Reshape` |
| `dynamic_pad` | L | resolve dynamic padding → `Pad` |
| `dynamic_iota` | L | resolve shape → materialize `Const` arange |
| `dynamic_gather` | L | resolve `slice_sizes` → `Gather` |
| `dynamic_conv` | L | resolve dynamic padding → `Fused(CONV2D)` |
| `get_dimension_size` | L | fold to `Const(size)` once the shape is concrete |

→ Mechanism: **shape specialization at import** (the dynamic operands are nearly always constant-foldable from the concrete inputs that tracing provides). Genuinely runtime-varying shapes → **monomorphize** (one static graph per concrete shape) at the model/runtime layer, which is how autoregressive decoding already extends graphs.

## DATA_MOVEMENT (13)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `constant` | P | `Const` |
| `reshape` | P | `Reshape` |
| `transpose` | P | `Transpose` / `Permute` |
| `concatenate` | D | fold of `Concat { dim }` (Fuel `Concat` is binary) |
| `pad` | P | `Pad` |
| `reverse` | D | per-dim fold of `Flip { dim }` |
| `select` | P | `Where` |
| `clamp` | P / D | scalar bounds → `Clamp`; tensor bounds → `Maximum(Minimum(x,hi),lo)` |
| `broadcast_in_dim` | D | `Unsqueeze`/`Reshape` + `BroadcastTo` |
| `iota` | L | materialize `Const` arange (static) |
| `slice` | P / D | contiguous → `Slice { dim, start, len }`; **strided (step>1) → gap-ish**: decompose via `Slice`+strided `Gather` (Fuel `Slice` has no step) |
| `gather` | P / D | simple index-gather → `Gather`/`IndexSelect`; fully-general offset/collapse machinery → decompose (possible partial gap) |
| `bitcast_convert` | D / G | same-bit-width reinterpret → view/`Cast`-like; differing width → gap (rare) |

## LINALG (5)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `dot_general` | D | `Reshape`/`Transpose` to align batch+contract dims → `MatMul` (the workhorse mapping) |
| `convolution` | F / G | 2-D → `Fused(CONV2D)` / `Fused(CONV_TRANSPOSE2D)`; 1-D/3-D/grouped-general → decompose or gap |
| `fft` | **G** | **no FFT.** Candidate fused `FFT` (audio/signal models; consumes the complex real-pair layout) |
| `cholesky` | **G** | **no Cholesky.** Candidate fused linalg op (rare in NN inference) |
| `triangular_solve` | **G** | **no triangular solve.** Candidate fused linalg op (rare) |

## QUANTIZATION (2)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `uniform_quantize` | X | weight-interchange quant interpreter + Cast-to-quant; feeds `Fused(QMATMUL)`/`Fused(NF4_MATMUL)` |
| `uniform_dequantize` | X | weight-interchange dequant path (Fuel's `QuantType` block formats) |

→ Quantization is **covered**, just not as primitive IR nodes — it lives in `fuel-interchange-weights` + the fused quant-matmul ops.

## RNG (2)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `rng` | X / L | inference dropout → **elide**; sampling RNG → inference-orchestration layer (`fuel-inference`; a known future sampling OpKind), not the graph |
| `rng_bit_generator` | X / G | as above; in-graph training RNG would be a gap if ever needed |

## BATCH_NORM (3)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `batch_norm_inference` | D / F | affine normalize via primitives, or a layer-norm-family fused op |
| `batch_norm_training` | D | reductions for mean/var + normalize |
| `batch_norm_grad` | D | training backward via primitives |

## MISC — tuples, tokens, async, I/O, escape hatches (12)

| StableHLO | Disp | Fuel target / mechanism |
|---|---|---|
| `tuple` | X | **multi-output bundle** (`OutputView`) |
| `get_tuple_element` | X | `Op::View { slot }` |
| `after_all` | X | scheduler ordering token (drop / ordering edge; `side_effect_roots`) |
| `optimization_barrier` | X | scheduler fence — identity that blocks fusion across it |
| `async_start` | L / X | **unwrap** to the inner op; async = pipelined-executor concern |
| `async_done` | L / X | join point; consumed by the unwrap |
| `send` | X | I/O boundary → graph output / `side_effect_roots` |
| `recv` | X | I/O boundary → graph input |
| `infeed` | X | → graph input boundary |
| `outfeed` | X | → graph output boundary |
| `custom_call` | F / **reject** | recognized `call_target` → map to a `Fused` op; unknown target → honest hard-reject (clear error) |
| `composite` | L / F | **inline its `decomposition`**, or recognize the composite name → `Fused`. The friendliest op — it ships its own lowering. |

---

## Actionable output: the gaps (G)

Ops within Fuel's scope with no current mapping. Each is a candidate primitive or fused op — add **only under real consumer pressure** ([no-consumer-is-not-a-reason cuts both ways](../architecture/02-layers.md#stopping-rule-for-new-crates); these are the genuinely cold tail until a model needs them):

1. **Inverse trig** — `atan2` (+ `atan`/`asin`/`acos`). Candidate unary/binary **primitives**. Appears in rotary variants, geometry, some positional encodings.
2. **Sort / TopK** — `sort`. Candidate **fused op**. Highest-value gap: top-k sampling and beam search want it.
3. **Pooling** — `reduce_window`. Candidate **fused `Pool`** (max/avg). baracuda already has the kernels; CNNs need it.
4. **FFT** — `fft`. Candidate **fused op**. Audio/signal models (consumes the complex real-pair layout).
5. **Product reduction** — `reduce` with a multiply combiner → no `ProdDim`. Candidate **primitive**.
6. **Integer-bitwise cluster** — `xor`/`shift_*`/`popcnt`/`clz` (+ true-integer `and`/`or`/`not`). Candidate **primitive cluster**, only if a bit-manipulating model is imported.
7. **Dense linalg** — `cholesky`, `triangular_solve`. Candidate **fused linalg ops**, rare in NN inference.
8. **Strided slice / general gather** — `slice` with step, fully-general `gather`. Likely **decompositions** rather than new ops, but worth a kernel if hot.

Everything else (≈100 of 119) is P, D, F, L, or X — i.e., already expressible or handled by a named mechanism. **Fuel's primitive vocabulary is functionally close to complete** relative to StableHLO's tensor-algebra core; the real holes are sort/topk, pooling, fft, inverse-trig, and product-reduce.

## The import-lowering toolkit (L) — what the StableHLO importer must implement

These are the graph transformations the importer owns. Building them once serves ONNX and ATen import too (they have the same shapes):

- **Unroll** — static-bounded `while`/`scan`/`case` → DAG.
- **Constant-fold** — constant predicates, dynamic-shape operands, `iota`, `get_dimension_size`, `partition_id`/`replica_id`.
- **Inline regions** — `map`, `reduce`/`reduce_window` bodies, `sort` comparators, `composite` decompositions → Fuel subgraphs; recognize known combiners (add/max/min) and named composites.
- **Recognize → fused** — composite-by-name, `custom_call`-by-target, scan→`SELECTIVE_SCAN`, conv→`CONV2D`, layernorm/softmax patterns → their `Fused` entries.
- **Real-pair emulation** — `complex`/`real`/`imag` → paired real tensors.
- **Elide / monomorphize** — single-replica collectives → identity; runtime-dynamic shapes → one static graph per concrete shape.

## What's handled at another Fuel layer (X) — not the importer's job to model

- **Multi-output** (`tuple`/`get_tuple_element`) → `OutputView` bundle + `Op::View`.
- **Scheduling/ordering** (`after_all`, `optimization_barrier`, `async_*`) → the pipelined executor's ordering + `side_effect_roots`.
- **Cross-device / collectives** (multi-replica) → `Op::Copy`/`Op::Move` today; a multi-node scheduler later.
- **Quantization** (`uniform_quantize`/`dequantize`) → `fuel-interchange-weights` + fused quant matmul.
- **Sampling RNG** → `fuel-inference` orchestration, not the graph.
- **I/O boundaries** (`send`/`recv`/`infeed`/`outfeed`) → graph input/output nodes.
