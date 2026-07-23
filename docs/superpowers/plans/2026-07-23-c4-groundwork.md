# C-4 shape-oracle frontier — governance split + Fuel-internal param threading (2026-07-23)

> **For agentic workers:** implement task-by-task with TDD; each task ends with an observed test run + commit. Worktree branched from `main` @ `af4b7dd4`. All code files are `fuel-dispatch/src/fkc/{return_check.rs, shape_expr.rs}` — nothing in `fuel-graph`. *(Superseded as-built — see §7 "Deviations": T3 also touched `fuel-graph/src/registry/conv_transpose_2d.rs`.)*

**Thread:** `c4-shape-frontier` · **Verdict: GO_REDUCED.** The Dims/WithDim tag activation is externally gated on a KISS extension-registry entry; the Fuel-internal slice (param threading + dtype-differential activation + hygiene + docs corrections) builds now.

## 1. The governance ruling (settled FIRST, per the thread brief)

KISS-OPS-6.20-0002 (`C:\Projects\KISS\spec\ops.md:1966-1968`): *"`Reduce(operand, axis, keepdim)`, `WithDim(operand, axis, DimExpr)`, and `Dims([DimExpr, …])` are **reserved** and MUST NOT be emitted by a producer at this vocabulary version (they enter through the extension registry, umbrella §6.4)."* §6.20-0005 adds *"an encoder MUST NOT emit a tag outside this set."* Umbrella §6.4 (`umbrella.md:276`): extensions enter **experimental → arbitrated → core**, PR-gated under ThinkersJournal; core promotion needs **two dissimilar implementations + a conformance test**, promoted by the sub-standard's editor.

**Ruling:** activating the reserved tags — even as a Fuel-text-DSL-only evaluation that never touches the wire — introduces constructors the closed vocabulary forbids, into an implementation (`shape_expr.rs`) that declares itself a byte-matching realization of §6.20. Propose-first applies in full. Fuel files the extension proposal (drafted, orchestrator sends); implementation of Dims/WithDim waits for acceptance of the experimental entry. Requested: **Dims (0x0B)** + **WithDim (0x0A)**; **Reduce (0x09) stays reserved** (no consumer — §6.20-0007 derives reduce shapes from attrs).

## 2. The per-op split (the design) — verified against the registry fns

| Op | Registry truth | Needs | Status |
|---|---|---|---|
| `conv2d` | rank-4 `[N, Cout, (H+2ph−Kh)/sh+1, (W+2pw−Kw)/sw+1]`; dilation fixed 1 (`conv2d.rs:79-99`) | `Dims` + `Param` (sh,sw,ph,pw) + `Extent` (Kh,Kw from weight) | **KISS-gated** |
| `conv_transpose_2d` | rank-4, `(H−1)·sh − 2ph + dh(Kh−1) + opad + 1` (9 params, `registry.rs:492-499`) | `Dims` + `Param` | **KISS-gated** |
| `qmatmul` | `a.dims[..last] ++ [n]`, n from `FusedOpParams::QMatMul` (`qmatmul.rs:56-71`) | `WithDim` (rank-poly) or `Dims` (per-probe-rank) + `Param` | **KISS-gated** |
| `nf4_matmul` | `[...,M,N]`, N = `Extent(w_packed,0)` — no param needed | `Dims` | **Double-gated**: only section is `registrable:false` until FDX AFFINE_BLOCK (`linear-quant.fkc.md:310-332`) — scope out entirely |
| `fused_softmax_cross_entropy` | reduction-**conditional**: Mean/Sum → `[]`, None → `targets.shape` (`fused_softmax_cross_entropy.rs:96-99`) | a **conditional constructor** — outside even the reserved vocabulary | **Permanent documented skip** (whole-shape rule); its `fixed(F32)` **dtype** check goes live NOW |
| `selective_scan` slot-1 `last_state` | `[u.0, u.2, a.1]` pure extents (`selective_scan.rs:144-149`) | `Dims` only | **KISS-gated** |
| `ssd_chunk_scan` slot-1 `last_state` | `[x.0, x.2, x.3, b.3]` pure extents (`ssd_chunk_scan.rs:149-163`) | `Dims` only | **KISS-gated** |

**Honest headline:** param threading alone flips **zero** of the 7 shape rules — each also needs a whole-shape constructor. The buildable-now value is (a) prerequisite infrastructure with a scheduled consumer, (b) the **dtype differentials** for the params-dependent variants, dead today only because `synth_probe_params` returns `None` for them (`return_check.rs:332-347` requires `Some(params)`), and (c) hygiene + doc-honesty fixes.

**Scan slot-1 is NOT premature:** the Phase-2/3 open item is the *decompose-path* view composer (`10-decisions-log.md:832`; `selective_scan.rs:80-90` — typed error, inert). The bundle differential's reference is `output_views` — live, allocator-wired, fused-kernel-written — and the §4 guardrail already forbids referencing decompose. Slot-1 declared rules join the gated batch.

**ROADMAP correction shipped in this thread (T5):** `ROADMAP.md:142-151` currently claims the reserved tags "cover" all ~7 ops — false for FSCE (conditional) and misleading for nf4 (double-gated).

## 3. Param-threading architecture (Fuel-internal, existing vocabulary only)

- `Dim::Param`/`TAG_PARAM=0x04` is **active core vocab**: `eval_dim` already evaluates it (`shape_expr.rs:216-221`), `parse_dim` already parses `param(N)` (`shape_expr_parse.rs:65-67`). The gap is the caller: `return_check.rs:66` passes `&[]`, so `param(N)` declines `ParamOutOfRange` → skip (pinned at `return_check.rs:590`).
- **Values are synthesized, not read from the contract** — `OpParamsSchema.fields` are constraint specs, not values (`schema.rs:174-185`), and probes are synthetic. New `synth_probe_param_points(variant, combo) -> Vec<(FusedOpParams, Vec<i64>)>`: the SAME values go to the declared-rule evaluator (ints) and the real registry fn (FusedOpParams). Shape-coupled fields derive FROM the combo (QMatMul `k = a.dims[last]`, `n = w.dims[0]`) — consistency by construction. Params-dependent ops get **≥2 points** (no single-point false-greens; the sabotage-calibration norm applied to params).
- **Flattening convention:** `param(N)` indexes `FusedOpParams::key().ints` (public, `registry.rs:417-428`): Conv2D → `[sh,sw,ph,pw,groups]`, ConvTranspose2D → 9 slots, QMatMul → `[quant_type_key,k,n]` (n = `param(2)`), FSCE → `[reduction.key(), ignore_index]`. Pinned by a differential test + per-variant tables in the corpus prose.
- **Never-panic evolution:** C-3's "evaluable ⇒ params-independent" coincidence (comment at `return_check.rs:238-255`) is retired; the mechanism is matching-variant synth (wrong-params panic unreachable) + `guard_rule` + `expected_min_inputs` gaining `Conv2D → Some(2)` (vulkan CONV2D fused sections exist: `vulkan/conv-attn-rope.fkc.md:49,123`). Rewrite the comment in the same change.

## 4. Tasks (each: red test observed → green → commit)

**T1 (S) — thread params through `eval_shape_rule`.** Signature gains a params slice; `param(0)` with `&[7]` evaluates to `Shape[7]`; composite `mul(extent(x,0), param(1))` works; `&[]` still declines to skip (updated pin at `:590`). Files: `return_check.rs`.

**T2 (M) — `synth_probe_param_points`.** Per-variant, per-combo points for Conv2D (2 pts), ConvTranspose2D (2 pts), QMatMul (1 pt, combo-derived), FusedSoftmaxCrossEntropy (2 pts: None + Mean), CausalConv1d (1 pt). ints pinned == `key().ints`; unknown variant → empty (never a foreign variant — extends the pin at `:536-540`). Files: `return_check.rs`.

**T3 (M) — cross-check loops param points; dtype differentials live.** Born-red flips: `:534` (Conv2D synth no longer None); **mutation test** — synthetic FSCE section declaring `dtype_rule: fixed(F16)` is REJECTED (real fn constant-F32), proving enforcement; `expected_min_inputs(Conv2D) == Some(2)`. Regression: corpus imports at `register.rs:1338-1340` stay green (their `conv2d(params)`/`from_params` shape rules remain *documented* skips — silent at import, no `ImportWarning`; "warned skips" here was wrong, corrected in review — their `passthrough`/`fixed` dtype rules now truly check). Rewrite the invariant comment. Files: `return_check.rs`.

**T4 (S) — reserved-tag named declines.** Explicit decoder arm naming `TAG_REDUCE`/`TAG_WITH_DIM`/`TAG_DIMS` (same typed `ReservedTag` decline — behavior-preserving), killing the dead_code warnings by reference, not `#[allow]`. This arm is the future activation point. Red test: constant-named declines for 0x0A/0x0B. Files: `shape_expr.rs`.

**T5 (S) — docs.** ROADMAP C-4 correction (FSCE conditional; nf4 double-gated; 5 ops KISS-gated); param-index tables + threading note in `fused/linear-quant.fkc.md` + `fused/conv-rope.fkc.md`; commit this plan. No cargo test (docs) — done-check is the table-vs-docs match.

## 5. Verification (whole increment)
- `cargo test -p fuel-dispatch --lib fkc` green, no regression; the FSCE-mutation rejection observed red first.
- `cargo build -p fuel-dispatch`: TAG_ dead_code warnings gone.
- **GPU: none** — import-path lib tests only.
- Adversarial review of the diff (the C-3 cadence): synth-value validity, no false-green at param points, the retired invariant's comment truthfulness.

## 6. Externally gated follow-up (queued, NOT this build)
On KISS acceptance of the experimental entry: implement `Dims`/`WithDim` (AST + wire byte-matching the minted goldens + eval with per-element Gap propagation + text-DSL parse) → rewrite the 5 gated ops' corpus rules (incl. both scan slot-1 rules; ~9 fused sections across `fused/conv-rope.fkc.md`, `fused/linear-quant.fkc.md`, `vulkan/conv-attn-rope.fkc.md`, `vulkan/quantized.fkc.md`) → oracle coverage ~16 → ~21 of 22 (FSCE stays the one honest skip). Peer-ask drafts live in the dispatch record (`external_coordination`); the orchestrator sends them.

## 7. Completion (ticked as built, branch `feat/c4-groundwork`)

- [x] **T1** — params threaded through `eval_shape_rule`; `&[]` still declines to skip (`daa54508`)
- [x] **T2** — `synth_probe_param_points`, ints pinned == `key().ints` (`2871dfe6`)
- [x] **T3** — cross-check loops param points; dtype differentials live; FSCE `fixed(F16)` mutation
  rejected; `expected_min_inputs(Conv2D) == Some(2)`; invariant comment rewritten (`c6f05a3b`)
- [x] **T4** — reserved-tag named declines (`TAG_REDUCE`/`TAG_WITH_DIM`/`TAG_DIMS`), dead_code
  warnings gone by reference (`0107c7c6`)
- [x] **T5** — ROADMAP C-4 correction; `param(N)` index tables + threading notes in both fused
  corpus files, pinned by the born-red `corpus_prose_pins_param_index_tables_matching_key_ints`
  doc-vs-code drift test; scan-slot-1 verdict recorded in the ROADMAP entry; outreach note
  `docs/outreach/kiss-dims-withdim-extension-registry-filed.md`. Plan committed at `1cf03f32`.
- **Post-plan external update (adopted in T5, newer than §1's "orchestrator sends"):** the
  Dims/WithDim §6.4 extension-registry proposal is **FILED** (KISS coordinator files the
  rfc-labeled issue on Fuel's behalf, attributed, per the #57 process; mechanics pre-verified
  against KISS main `c9153b2`); Baracuda: no objection + future consumer (Window/pooling + conv) +
  cosigns with the `dims(...)`/`with_dim(...)` functional-spelling pin in the same clause as the
  wire tags; kiss-ref: consistent with its §6.20 stake, the second dissimilar implementation,
  timing theirs. §6's follow-up remains queued on acceptance.

### Deviations from the plan text (as-built record, disclosed per the doc-vs-code-drift norm)

- **T3 touched `fuel-graph` (contradicting this plan's "nothing in `fuel-graph`" header).**
  `c6f05a3b` widens the arity `debug_assert`s in
  `fuel-graph/src/registry/conv_transpose_2d.rs` `shape_rule`/`dtype_rule` from exactly-2 to
  2-or-3 inputs (`x`, `weight`, `[bias]`). Cause: the contract declares an OPTIONAL bias operand,
  so the §3.5 probe combos carry 3 operands, and the exact-2 assert made the now-live dtype
  differential guard-catch (a debug-only skip). The change matches the op's documented arity and
  the conv2d precedent; pinned by `conv_transpose2d_dtype_differential_fires`; gate
  `cargo test -p fuel-graph` 349 green. The commit body disclosed it; this plan + the ROADMAP
  entry did not until the review pass — recorded here.
- **T4 widened `shape_expr` visibility `pub(crate)` → `pub`** (`0107c7c6`,
  `fuel-dispatch/src/fkc/mod.rs`) — a fuel-dispatch public-API expansion (exposes
  `Dim`/`ShapeExpr`/`eval_dim`/codec/`TAG_*`) this plan did not mandate. It IS load-bearing for
  §5's gate: with `pub(crate)` and no crate-internal codec consumer, the whole §6.20 codec chain
  is dead code, so ALL 11 `TAG_` constants warn and no decoder-arm reference can silence them
  without `#[allow]` (re-verified empirically in the review pass: flipping back to `pub(crate)`
  reintroduces all 11 warnings). Rationale stands (golden-verified KISS-interop surface + the
  future activation point); decision now recorded here rather than only in the commit body.
- **Review pass (post-T5) hardening, same branch:** (a) the T2 param points were
  order-DEGENERATE (Conv2D stride==padding at both points; ConvTranspose2D
  padding==output_padding, dilation==groups) — a `key().ints` slot reorder or a future
  `param(i)`/`param(j)` rule confusion evaluated identically at every point; points are now
  order-asymmetric, pinned by `synth_param_points_distinguish_every_ints_slot_pair`; (b) a
  params-dependent variant whose probe combo the shape-coupled synth can't read now WARNS
  instead of silently skipping every differential
  (`empty_param_points_for_params_dependent_variant_warns`); (c) the "warned skips" claims for
  the KISS-gated whole-shape rules were wrong (a non-evaluable rule skips silently, no
  `ImportWarning`) — reworded to "documented skips" in both corpus files + T3 above.

## 8. Evidence index
`ops.md:1966-1968,1985-1997` (reservation) · `umbrella.md:276` (lifecycle) · `rfcs/shape-expression-oracle.md` (§6.20-0002 reservation + Q1) · `shape_expr.rs:17-19,145,216-221,335` · `shape_expr_parse.rs:65-67` · `return_check.rs:32,66,188-236,238-255,332-347,534,590` · `schema.rs:174-185` · `registry.rs:417-428,458,492,555,572` · `conv2d.rs:79-99` · `qmatmul.rs:56-71` · `fused_softmax_cross_entropy.rs:85-110` · `selective_scan.rs:75-90,136-156` · `ssd_chunk_scan.rs:136-163` · `linear-quant.fkc.md:230-243,310-332,459-471` · `conv-rope.fkc.md:217-228,307-320,503-504,596-597` · `register.rs:1174-1178,1336-1340` · `10-decisions-log.md:822-841,875-881` · `ROADMAP.md:142-151` · base `af4b7dd4`.
