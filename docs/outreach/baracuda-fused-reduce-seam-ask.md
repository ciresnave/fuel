# Baracuda ask — fused RmsNorm/Softmax/LayerNorm adoptable via the §5 seam (inbound 2026-06-25)

**Fuel-side status (2026-06-26): RECEIVED, response QUEUED after the B0 retirement program.**
This is the verbatim inbound ask. The Fuel response (answers a–d + any grammar/matcher
change + the FKC-pattern-for-RowReduce work + the extent pre-condition) is a dedicated
unit on the kernel-seam/JIT thread (ROADMAP frontier item 4), sequenced after B0. It is
independent of B0 (it touches `fuel-kernel-seam-types` + `match_region`/`derive_pattern` +
the `cost_expr` core, none of which B0 moves).

Answering it requires code-grounded investigation of: the frozen grammar (`OpTag`/`OpAttrs`/
`Bind`/`MeanDim`/`SumDim`/`PatternNode`), `derive_pattern` (→ `NotElementwise` today),
`match_region`, the FKC contract emission, and the `cost_expr` core. Likely outputs: confirm
(a); add `MaxDim`/`MinDim` to the `#[non_exhaustive]` `OpTag` for (b); extend
`match_region`/`derive_pattern` to admit reduce→broadcast→elementwise for (c); pin cost vars
for (d); and enforce the `weight.extent[-1] == x.extent[-1]` pre-condition at the seam boundary.

---

Baracuda now synthesizes **fused RmsNorm, Softmax, LayerNorm, and weighted-RmsNorm**
(one block per row, warp-shuffle + shared-mem tree reduce, numerically validated on
sm_89). But they're **AOT-only** — NOT adoptable through the §5 JIT seam, because the
seam region path (`region_to_op`) and FKC contract emission only handle **elementwise**
regions; a region containing a reduction honest-misses (`derive_pattern` →
`NotElementwise` → no contract). To make these Fuel-fusable (region in → kernel + recipe
out → cost-gated adoption), agree on how a **fused-reduce region** is encoded in the
frozen grammar and matched. The codegen is done; this is purely the seam encoding.

## 1. Region encoding (proposed — no grammar change for RmsNorm/LayerNorm)

The frozen grammar already expresses reduce → broadcast → elementwise via
`MeanDim`/`SumDim` + `OpAttrs.axis` + a shared `Bind` (the node-identity guard):

- **RmsNorm** `x · rsqrt(mean(x², −1) + eps)`:
  `Mul( Bind0, Rsqrt( AddScalar{eps}( MeanDim{axis:-1}( Sqr( Bind0 ) ) ) ) )`
  — `Bind0` twice (shared `x`); `MeanDim` carries `axis: Some(-1)`; `eps` on `AddScalar`;
  broadcast-back implicit at the outer `Mul`.
- **LayerNorm** `(x − μ)·rsqrt(var + eps)·w + b`, two reductions:
  `Add( Mul( Mul( Sub(Bind0, MeanDim{-1}(Bind0)), Rsqrt(AddScalar{eps}(MeanDim{-1}(Sqr(Sub(Bind0, MeanDim{-1}(Bind0))))))), Bind1 /*w*/), Bind2 /*b*/ )`
  — inner `Sub(Bind0, MeanDim{-1}(Bind0))` is centered-x (shared); var = stable mean of its square.

**Ask (a): confirm** this encoding — `MeanDim`/`SumDim` with `axis=Some(-1)`, shared `Bind`,
`AddScalar` for eps, broadcast-back implicit at the consuming binary op.

## 2. Gap — Softmax's last-axis max

`OpTag` has `SumDim`/`MeanDim` (per-dim) but only `MaxAll` / `ReduceMaxTo` for max.
Stable Softmax needs a **last-axis max**:
`Div( Exp(Sub(Bind0, MAX_LASTDIM(Bind0))), SumDim{-1}( Exp(Sub(Bind0, MAX_LASTDIM(Bind0))) ) )`

**Ask (b): how to spell `MAX_LASTDIM`?** (i) `ReduceMaxTo` with `[…,1]` target via attrs; or
(ii) add `MaxDim`/`MinDim` to the (`#[non_exhaustive]`) `OpTag` mirroring `MeanDim`/`SumDim`.
Baracuda leans (ii) (cleanest mirror; `Access::RowReduce` already has `ReduceOp::Max`/`Min`) —
Fuel's call. RmsNorm + LayerNorm can go live without it.

## 3. Broadcast-back + `match_region`

The region grammar is shapeless/structural, so a reduction's `[…,1]` result broadcasting
into the surrounding elementwise op is implicit at the consuming node. **Ask (c):** does
`match_region` already match *reduce → (implicit broadcast) → elementwise* (a `MeanDim`
result feeding a broadcasting `Mul`/`Sub`/`Div`), or does the region need an explicit
`BroadcastTo` node between them? If the latter, Baracuda emits `BroadcastTo` in `pattern:`
and consumes it in `region_to_op`.

## 4. Baracuda's side, once (a)–(c) are pinned

- Extend `region_to_op`: recognize a fused-reduce region (last-axis `MeanDim`/`SumDim`/max
  feeding an elementwise epilogue) → lower to `Access::RowReduce { stages, epilogue }` (each
  reduction → a stage; rest → epilogue; each reduction result → a `Reduced(i)` leaf).
- Emit the FKC contract + `pattern:` for `RowReduce` ops (today they honest-miss).
- Advertise the capability so Fuel may issue fused-reduce JIT requests.
- **Extent pre-condition the seam CALLER must enforce:** `structure_key` carries broadcast
  masks but **no numeric extents** (structure-specialized), so the synthesizer can't verify a
  per-column weight/bias's feature extent equals `x`'s `k` — a too-short weight keys
  identically and reads OOB. Like `n_out`/`k`, this is a caller pre-condition: the boundary
  holding live `FdxOperandDesc`/`OperandDesc` extents (Fuel's `op_to_tag` / region-assembly)
  must assert `weight.extent[-1] == x.extent[-1]` before the request crosses. (On-device:
  compute-sanitizer-clean with extents present; the mismatch is only reachable via a
  mis-sized operand.)

## 5. Cost-gating

A fused-reduce kernel = one launch, ~2–3 passes/row; the primitive path = several
reduce+broadcast+elementwise kernels (each a full pass+launch). Baracuda emits a `cost` expr
over `n` (out elems) + row extent `k` reflecting fused pass count, for the `cost_expr` core to
gate adoption vs the primitive estimate. **Ask (d): preferred cost variables** for a row-reduce
op (we have `n`; is a per-row `k` binding available, or stay in `n`?).

## Scope

Last-axis reductions (transformer norm/softmax) first; multi-axis/arbitrary-axis later.
Single row-streamed input + per-column weight/bias (LayerNorm) built; a second row-streamed
input (fused residual-add LayerNorm) is a follow-up.

**Summary of asks:** (a) confirm `MeanDim`+axis+shared-`Bind` encoding; (b) pin Softmax
last-axis-max spelling (`ReduceMaxTo` vs new `MaxDim`); (c) confirm `match_region` handles
reduce→broadcast→elementwise (or specify explicit `BroadcastTo`); (d) cost-expr variables for a
row-reduce op.
