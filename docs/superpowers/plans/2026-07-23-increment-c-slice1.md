# Increment C slice 1 — recipe-interior migration foundations (2026-07-23)

**Thread:** `increment-c` · **Base:** `main` @ `af4b7dd4` · **Status:** design ratified for build, worktree execution
**Program:** ROADMAP "Recipe-grammar convergence" bullet (`ROADMAP.md:114-141`) — "the remaining Increment C is narrowed to the recipe interior."
**This document is slice 1 of an L-sized program.** The full-program roadmap (slices 2–5) is §9.

---

## §0 Verified ground truth (re-checked against `af4b7dd4`; supersedes stale context)

Every claim below was re-verified in this design pass; anchors are current.

1. **Crate dependency direction:** `fuel-dispatch` → `fuel-graph` → `fuel-kernel-seam-types`. `fuel-graph` can NEVER depend on `fuel-dispatch` (`fuel-dispatch/Cargo.toml` lists `fuel-graph = { workspace = true }`; `fuel-graph/Cargo.toml` lists `fuel-kernel-seam-types`). `fuel-kernel-seam-types` is deliberately dependency-free ("std-only POD grammar types so any synthesizer backend can depend on this crate", `fuel-kernel-seam-types/Cargo.toml:11-15`).
2. **`fuel-dispatch/src/fkc/shape_expr.rs` is 100% std-only** — zero `use` statements in the whole file. `Dim`/`ShapeExpr` AST (`:30-49`), §6.20 codec with **u16-LE child lengths** (`encode_binary`, `:79-87`), `LAST = 0xFF` (`:28`), `SYMBOLIC = i64::MIN` (`:22`), typed declines (`:89-101`), `eval_dim(d, operands: &[Vec<i64>], params: &[i64])` (`:205`), `resolve_axis` (`:180-187`), `reduce_shape`/`gather_shape`/`matmul_shape` (`:253-290`), goldens in-file. It moves crates verbatim.
3. **Registry inventory (22 submodules, re-verified):**
   - **4 basis-gap self-returns** (body = `id`): `conv2d.rs:127-129`, `conv_transpose_2d.rs:111-113`, `qmatmul.rs:100-102`, `inplace_affine.rs:67-69`. Unchanged; out of this program (need IR primitives — C-4/ROADMAP tracked).
   - **2 scans**: `selective_scan.rs`, `ssd_chunk_scan.rs` — **have LEFT the basis-gap set** (decompose totally onto `Op::Scan`; self-return only on impossible-params arms `:245/:261`, `:229/:246`) but stay outside first-order PatternNode migration: `Op::Scan`/`Op::ScanPlaceholder` are outside `OpTag` (`jit.rs:105,136`). Their data form is the flat-table scan layout (co-design Q5) — slice 4.
   - **5 fully slice-1-migratable** (this session): `softmax_last_dim` (7 nodes: ReduceMaxTo/BroadcastTo/Sub/Exp/ReduceSumTo/BroadcastTo/Div, `softmax_last_dim.rs:78-134`), `rope` (11 nodes incl. 2 rank-pad Reshapes, `rope.rs:83-178`), `rms_norm_last_dim` (7 nodes incl. `AddScalar(eps)`, MeanDim+Reshape-keepdim), `layer_norm_last_dim` (11 nodes, x-only input, MeanDim×2+Reshape×2+AddScalar(eps)), `softmax_last_dim_backward` (5 nodes: Mul/ReduceSumTo/BroadcastTo/Sub/Mul, `softmax_last_dim_backward.rs:96-120`).
   - **11 deferred with named blockers** (§9): `powi_backward` (PowI carrier gap, `runtime_fused.rs:363-366` honest miss + param-arithmetic `exp-1`), `reduce_max_to_backward` (MaskedFill carrier gap), `rms_norm_last_dim_backward` (`MulScalar(n)` where n = reduced extent — shape-derived scalar, `rms_norm_last_dim_backward.rs:142`), `layer_norm_last_dim_backward` (same family; re-check at slice 2), `fused_linear` (broadcast target = interior matmul's shape `[batch..,M,N]` — matches NO bind; needs flat-table node-indexed SameAs or reserved `Dims`), `causal_conv1d` (node count loops over kernel taps — param-shape-driven structure), `flash_attn` (32 pushes, k_len 3-case + causal/window/softcap conditional structure + nested `Op::Fused(SOFTMAX_LAST_DIM)` at `flash_attn.rs:330`), `flash_attn_backward` (variant-conditional), `paged_attn` (nested fused softmax `paged_attn.rs:225`, param-driven), `fused_softmax_cross_entropy` (reduction-conditional structure), `nf4_matmul` (block_size-driven, registrable:false).
4. **STALE-CONTEXT CORRECTION:** "emit/register_runtime_fused are v1 same-shape-elementwise only" is outdated. Increment A shipped full first-order emit parity: `emit` derives shape+dtype per node via `primitive_shape` (`runtime_fused.rs:527-533`); `tag_to_op` reconstructs the full first-order vocabulary — Slice from `axis+slice_start/len`, SumDim/MeanDim from `axis`, Unsqueeze from `dims`, Cast, MatMul, Iota (`runtime_fused.rs:290-366`). Honest misses: PowI/Clamp/MaskedFill (`:363-366`).
5. **Open scalar slots already work:** `scalar_slot_arity` (AddScalar|MulScalar = 1, `runtime_fused.rs:385-387`), `count_scalar_slots` (`:394-402`), pre-order cursor fill in `emit` (`:500-503`), validated as slot templates in `validate_representable` (`:411-413`). Static entries need only a per-entry `FusedOpParams → Vec<f64>` projection.
6. **`check_broadcast_compatible` is right-aligned + rank-raising** (`fuel-graph/src/lib.rs:10268-10284`: `src.len() <= dst.len()`, pad-left). Rope's two `Reshape`-to-1s-prefix nodes are semantically the right-aligned pad. We still emit them (see D4) to keep the emitted graph byte-identical to legacy.
7. **`OpTag` has NO `MaxDim`** (`fuel-kernel-seam-types/src/lib.rs:48` — reductions are `SumAll, MaxAll, MinAll, MeanAll, SumDim, MeanDim, ReduceSumTo, ReduceMaxTo, CumSum`) while `Op::MaxDim(usize)` exists graph-side with builders + backward (`fuel-graph/src/lib.rs:636, 5907-5909, 8683`). `primitive_shape` covers `SumDim|MaxDim|MinDim|MeanDim` in one arm (`shape.rs:175`) and `Unsqueeze` allows `dim == rank` append (`shape.rs:107-112`).
8. **KISS #67 (external, in flight):** (a) §6.8-0007 amends `u16`→`u32` TOWARD Fuel's shipped u32-LE op_attrs outer frame — Fuel's oracle is authoritative (`docs/outreach/fuel-recipe-grammar-kiss-design-input.md:99`); (b) the recipe-NODE/table wire envelope (`op_name` + `child_edges` framing) is **being defined in #67 — NOT pinned** (`:104, :168`); (c) **two distinct blobs, two widths — do not unify**: op_attrs outer = u32-LE byte-len; shape-expr child = u16-LE (`:101-102`). (d) Leaf-token byte layouts (`runtime_scalar{slot_index}`, `reduced_count{axes}`, `const{bits}`, `scan_placeholder{role,index}`) are pinned as TOKENS (`:43-49`) but have **no pinned byte arms** — propose-first.
9. **Matmul role vectors are LOCKED, mutual** (commit `b64aa1db`; `fuel-recipe-grammar-kiss-design-input.md:125-147`): roles `{Batch=0, FreeM=1, FreeN=2, ContractedK=3}` u8 each; INNER `u32_le(count) ++ roles`, lhs-then-rhs; OUTER `u32_le(body_len) ++ body`; rank-2 golden `0C000000|02000000|0103|02000000|0302`. Today MatMul falls through to the empty arm → `[00,00,00,00]` (`lib.rs:239-241`, test `:339`); `tag_to_op` accepts unconditionally (`runtime_fused.rs:316`). No `put_u8` helper exists (`lib.rs:146-153`).
10. **`base_map_hash` is process-local, never persisted** (`opt.rs:392-398`) — changing a base-map spelling has no ledger/cross-process blast radius; the jit_ingest recipe-identity verifier computes both sides live.
11. **Parity oracles that gate this work:** `emit_matches_softmax_last_dim_decompose` (`runtime_fused.rs:900`), `emit_matches_rope_decompose` (`:937`), `emit_matches_layer_norm_last_dim_decompose` (`:972`); fuel-core `lazy.rs` gap-posture tests (flash/nf4/selective — untouched by this slice); rope live consumer `fuel-core` `rope_with_tables_decomposed` (`fuel-core/src/lazy.rs:1300-1350`).

---

## §1 Design decisions

### D1 — Vocabulary home: MOVE `shape_expr` to `fuel-kernel-seam-types` (re-export shim in fuel-dispatch)

`fuel-graph` cannot depend on `fuel-dispatch` (§0.1), so the `Dim`/`ShapeExpr` vocabulary must live at or below `fuel-kernel-seam-types` for `OpAttrs` to carry it. Decision: **move `fuel-dispatch/src/fkc/shape_expr.rs` verbatim to `fuel-kernel-seam-types/src/shape_expr.rs`** (it is std-only — zero imports, §0.2); `fuel-dispatch/src/fkc/shape_expr.rs` becomes `pub use fuel_kernel_seam_types::shape_expr::*;` so every existing path (`crate::fkc::shape_expr::…`), test, and the KISS goldens stay working unchanged.

*Why not the alternatives:* a new crate is overhead with no second consumer distinct from seam-types' exact charter ("frozen kernel-seam wire types… shared across the Fuel↔backend-synthesizer seam"); `fuel-ir` is the logical-dtype home, not the seam-wire home, and bloats a crate every backend links; duplication violates the single-oracle rule (the codec byte-matches KISS goldens — two copies drift). This also does not touch the retiring crates (fuel-core/fuel-core-types), per the B0 program.
*Anti-entanglement guard (KISS #67 / `0a996e65` pin):* the moved module keeps its **u16-LE child length**; `OpAttrs::to_canonical_bytes` keeps its **u32-LE outer byte length**. A new seam-types test pins BOTH widths side-by-side so a future consolidation can't silently unify them. `shape_expr_parse.rs` (text-surface parser for FKC contract strings) STAYS in fuel-dispatch — it is contract-import machinery, not grammar.

### D2 — Shape-relative interior attrs = new optional `OpAttrs` fields, resolved at emit; NOT serialized this slice

New fields on `OpAttrs` (all `Default`-empty ⇒ zero behavior change for existing regions):

```rust
pub target_shape_rel: Option<shape_expr::ShapeExpr>, // SameAs{operand} — operand = BIND index
pub slice_start_rel:  Option<shape_expr::Dim>,       // DimExpr over bind shapes
pub slice_len_rel:    Option<shape_expr::Dim>,
pub axis_last:        bool,                           // this op's axis-carrier = its per-tag LAST
```

- **Reference convention:** in the recipe interior, `ShapeExpr::SameAs{operand}` / `Dim::Extent{operand,..}` index the region's **Bind space** (`Bind(i)` == a contract's role — the same convention the merged KISS RFC pins for contracts, so one convention across all three homes).
- **`axis_last` per-tag resolution table:** reduces/Slice/Concat/Flip/CumSum/etc. → `rank(operand[0]) − 1`; `Unsqueeze` → `rank(operand[0])` (append; `primitive_shape` permits `dim == rank`, §0.7). Squeeze → `rank − 1`.
- **Resolution point:** inside `emit` (fuel-graph/src/runtime_fused.rs). Reorder the body: emit children FIRST (scalar-slot cursor fill stays pre-order, before descending — unchanged), then `resolve_rel_attrs(attrs, bind_shapes, child_shapes) -> Result<OpAttrs, …>` produces a fully-concrete `OpAttrs`, then the **unchanged** `tag_to_op` → `primitive_shape` path runs. Resolution reuses `shape_expr::eval_dim`/`resolve_axis` (no second evaluator). Mutual-exclusion validation (rel XOR abs per field) + bind-range checks extend `validate_representable` using the existing slot-template probe pattern (`runtime_fused.rs:411-413`).
- **Totality:** a resolution failure in the registry-decompose path returns `id` (fixpoint, surfaced gap) — never panics. The raw `emit_region` entry keeps its current documented posture.
- **Wire form: deliberately NOT serialized this slice.** `to_canonical_bytes` continues to serialize only concrete fields; rel fields are in-memory recipe data. Rationale: the §6.19 arms for `broadcast_to`/`slice` are pinned as ABSOLUTE (`put_i64_list(target_shape)`, `u32(axis)++u64(start)++u64(len)` — `fuel-recipe-grammar-kiss-design-input.md:109,112`), and the node-envelope framing that would carry a relative alternative is being defined in KISS #67 (§0.8b). Serializing a rel form now would unilaterally extend a shared byte contract — propose-first says no. Rel-attr recipes never flow to `to_canonical_bytes` callers (those operate on emitted/graph nodes, which are always concrete post-resolution); documented on the fields.

### D3 — Keepdim strategy = the RATIFIED shrink-via-swap (adds `OpTag::MaxDim`), not the reserved `Reduce`/`WithDim` tags

`ReduceMaxTo/ReduceSumTo(keepdim_shape)` and `Reshape(keepdim_shape)` bake a whole shape that equals NO operand (the §6.20 constructors for it — `Reduce`/`WithDim`/`Dims` — are tag-reserved and reader-rejected, `shape_expr.rs:17-19`). Per the ratified decision ("most decomposes become polymorphic by swapping to shape-polymorphic primitives"), migrated recipes spell keepdim as **`{Max,Sum,Mean}Dim(axis_last)` + `Unsqueeze(axis_last=append)`**:

- softmax fwd: `ReduceMaxTo+Bcast / ReduceSumTo+Bcast` → `MaxDim+Unsqueeze+Bcast / SumDim+Unsqueeze+Bcast` (7→9 nodes).
- rms/layer_norm fwd: only `Reshape(keepdim)` → `Unsqueeze(append)` (node-type change, metadata-only, bit-exact).
- softmax bwd: `ReduceSumTo` → `SumDim+Unsqueeze`.

**Consequences, managed:** the migrated ops' base maps change → `base_map_hash` changes (process-local only, §0.10); the three `emit_matches_*` oracles are repointed at FROZEN copies of the legacy imperative builders with numeric (not structural) parity as the gate; re-fuse `canonical_pattern`s still match the OLD user-spelled subgraphs (no regression), and extending them to also match the new spelling is a slice-2 item unless an existing roundtrip test forces it sooner. Numeric risk: `MaxDim` is order-insensitive (exact); `SumDim` vs `ReduceSumTo` accumulation order could differ → parity tests assert bit-exact FIRST and fall back to a sabotage-calibrated tolerance only with the calibration evidence in the test (per the sabotage-test-calibration norm). This needs the additive **`OpTag::MaxDim`** (§0.7): `op_to_tag`/`tag_to_op` (from `axis`), §6.19 arm = `i64(axis) ++ u8(keepdim=0)` — inside the pinned "reduce (monoid rides op_name)" schema row, additive per `#[non_exhaustive]`; flagged to KISS/Baracuda in the coordination note.

### D4 — Broadcast rank-raising: the resolver MATERIALIZES the legacy `Reshape`-pad

When a resolved `BroadcastTo` target has rank > operand rank, the emit-resolver first pushes the exact legacy `Reshape` (1-padded left, right-aligned — provably identical to `rope.rs:99-104`'s hand-built prep since `check_broadcast_compatible` is right-aligned, §0.6). The rope RECIPE is 9 nodes of data; its EMISSION is the byte-identical legacy 11-node graph. This retires all backend risk (CUDA decode's live rope-decompose path sees an unchanged graph), keeps `emit_matches_rope_decompose` green against the existing oracle, and keeps recipe data free of rank-dependent node counts.

### D5 — Matmul role vectors: attrs-resident, empty-is-implicit, resolver validates the canonical cell

Per the locked reply-3 layout (§0.9): add `lhs_roles: Vec<u8>` / `rhs_roles: Vec<u8>` to `OpAttrs` + a pure `matmul_roles(lhs_rank, rhs_rank)` helper in seam-types (deriving `Batch×(r−2),FreeM,ContractedK` / `Batch×(r−2),ContractedK,FreeN`); `put_u8_list` helper (`u32_le(count) ++ u8s`); a named `MatMul` arm in `to_canonical_bytes` (empty roles → empty body `[00,00,00,00]`, preserving today's golden; set roles → the locked blob, rank-2 golden `0C…0302`); a resolver cell in `tag_to_op` (empty → `Op::MatMul` implicit-accept; set → validate exactly-one FreeM/FreeN/ContractedK at the pinned positions with leading Batch → `Op::MatMul`, else `None` = surfaced honest miss at registration — role POSITIONS, not extents: GQA-divisible batch stays all-Batch). Static recipes keep matmul implicit (role vectors bake rank; recipes are rank-polymorphic); concrete/ingested nodes get explicit roles. `attrs_match` does not consult the new fields this slice (documented). Independent of D2–D4.

### D6 — Registry migration mechanism: `decompose` fn stays; its body becomes emit-of-static-data

`FusedOpEntry.decompose: fn(&mut Graph, NodeId, &FusedOpParams) -> NodeId` (`registry.rs:112`) is untouched. Each migrated submodule replaces its imperative body with:

```rust
fn recipe() -> &'static PatternNode          // OnceLock-built static data
fn scalars(params: &FusedOpParams) -> Option<Vec<f64>>  // per-entry projection (eps, …)
pub fn decompose(g, id, params) -> NodeId    // = decompose_via_recipe(g, id, recipe(), scalars)
```

`decompose_via_recipe` (new, fuel-graph): reads the fused node's inputs as binds, projects scalars, calls the resolving emit; ANY failure (wrong params, resolution error, slot mismatch) → `return id` (fixpoint, surfaced gap, never panic — the exact G2 posture the imperative bodies have today).

### D7 — Leaf-token serialization (`runtime_scalar`/`reduced_count`/`const`/`scan_placeholder`): propose-first, code deferred

Byte arms are not pinned externally (§0.8d) and their import consumer (the #67 node envelope) doesn't exist yet. Drafted ask (§8) proposes: `runtime_scalar → u32(slot_index)`; `reduced_count → i64(axis)` (single-axis, byte-identical to the fold's axis field minus keepdim, growing to the list in lockstep — the §6.12 constraint; the resolver MUST reuse the fold's axis-resolution codepath verbatim when it lands); `const → u64(bits)` (dtype-agnostic bit pattern); `scan_placeholder → u8(role) ++ u32(index)`. Code lands in slice 2 after ack. (`iota` already serializes — Increment A.)

---

## §2 What slice 1 ships (scope)

CPU-only, one worktree branched from `main` @ `af4b7dd4`, serial task execution (T2–T9 share `fuel-kernel-seam-types/src/lib.rs` and `fuel-graph/src/runtime_fused.rs`).

1. **T1** shape-expr vocabulary home move + shim + two-width anti-entanglement pin.
2. **T2** rel-attr fields + `resolve_rel_attrs` (pure fn + unit tests).
3. **T3** emit integration (reorder + resolution + D4 pad + `validate_representable` rel-probe).
4. **T4** `OpTag::MaxDim` (additive).
5. **T5** `decompose_via_recipe` bridge + softmax_last_dim pilot migration.
6. **T6** rope migration (DimExpr slices + D4).
7. **T7** rms_norm + layer_norm migration (eps open slots).
8. **T8** softmax_last_dim_backward migration.
9. **T9** matmul role-vector serialize/resolve (locked layout).
10. **T10** docs: recipe-signature-reference §A/§C status banners, ROADMAP frontier, decisions-log check.

**Result:** 5 of the 22 registry decomposes are portable `PatternNode` data (polymorphic across shapes/ranks — the thing that is impossible today), the shape-relative attr machinery exists end-to-end with the KISS-consistent vocabulary in its permanent home, and the locked matmul byte contract is live in both directions.

## §3 Task list (TDD; every task starts born-red)

**T1 — Move `shape_expr` to `fuel-kernel-seam-types`; shim; width pin. (M)**
Red: new seam-types test module (ported `serialization_golden` incl. the 17-byte rope-half golden `08 0300 0200FF 0900 03 02…` + round-trips + typed-decline cases) fails to compile — module doesn't exist. New `two_blob_widths_stay_distinct` test: asserts a binary `Dim`'s child-length prefix is 2 bytes (u16) AND `OpAttrs::to_canonical_bytes` outer prefix is 4 bytes (u32) in one test body with a comment pinning KISS #67 do-not-unify.
Green: move file verbatim; `pub mod shape_expr;` in seam-types; fuel-dispatch `fkc/shape_expr.rs` → `pub use fuel_kernel_seam_types::shape_expr::*;` (keep the module doc). Full fuel-dispatch fkc suite must stay green through the shim (`return_check.rs`, `shape_expr_parse.rs` untouched).
Files: `fuel-kernel-seam-types/src/lib.rs`, `fuel-kernel-seam-types/src/shape_expr.rs` (new), `fuel-dispatch/src/fkc/shape_expr.rs`.

**T2 — Rel-attr `OpAttrs` fields + `resolve_rel_attrs` pure resolver. (M)**
Red: unit tests for `resolve_rel_attrs` (new fn, doesn't exist): `SameAs{0}` over bind shapes `[2,3]`→target `[2,3]` and `[4,5]`→`[4,5]`; `slice_start_rel=Div(Extent(0,LAST),2)` at d=4→2 and d=8→4; `axis_last` on SumDim at rank 2→axis 1, rank 3→axis 2; Unsqueeze `axis_last`→append (=rank); error cases: bind out of range, rel+abs both set, negative result → typed error (never panic).
Green: fields on `OpAttrs` (D2; check struct-literal construction sites across fuel-graph/fuel-dispatch — all use field-init or `..Default::default()`), resolver in fuel-graph reusing `shape_expr::eval_dim`/`resolve_axis`, per-tag LAST table.
Files: `fuel-kernel-seam-types/src/lib.rs`, `fuel-graph/src/runtime_fused.rs` (resolver + tests).

**T3 — Emit integration: children-first reorder + resolution + D4 pad + validation. (M)**
Red: the headline polymorphism test — a region with `BroadcastTo{target_shape_rel: SameAs{0}}` emitted twice at different shapes produces correct shapes BOTH times (impossible today: absolute attrs). Second red: `BroadcastTo` with rank-raising operand emits the legacy `Reshape`-pad node first (assert node sequence + shapes match a hand-built legacy graph). Third red: `validate_representable` accepts a rel-attr region (today `tag_to_op` returns `None` on empty absolute fields → rejected) and still rejects a region with rel+abs both set.
Green: reorder `emit` (children → resolve → `tag_to_op` → `primitive_shape`; scalar cursor fill stays pre-order — assert `decompose_fills_slots_from_the_node_scalars` (`runtime_fused.rs:813`) stays green), D4 pad insertion, `validate_representable` rel-probe (mirror of the `:411-413` slot-template fill).
Files: `fuel-graph/src/runtime_fused.rs`.

**T4 — Additive `OpTag::MaxDim`. (S)**
Red: round-trip test `op_to_tag(Op::MaxDim(1)) == Some(T::MaxDim)` + `tag_to_op(T::MaxDim, axis=1) == Some(Op::MaxDim(1))` (tag doesn't exist → compile-red); canonical-bytes golden `i64(axis) ++ u8(0)` matching the SumDim row shape.
Green: `OpTag::MaxDim` (+ arm in `op_to_tag` `jit.rs`, `tag_to_op`, `to_canonical_bytes` reduce arm, `scalar_slot_arity` 0 by default, `op_to_attrs` axis projection).
Files: `fuel-kernel-seam-types/src/lib.rs`, `fuel-graph/src/jit.rs`, `fuel-graph/src/runtime_fused.rs`.

**T5 — `decompose_via_recipe` bridge + softmax pilot. (L)**
Red: (a) polymorphic decompose test: lower `Op::Fused(SOFTMAX_LAST_DIM)` at `[2,4]` AND `[3,5,7]` via the registry; realize on CPU; numerics match a FROZEN copy of the legacy imperative builder (moved into the test module) — bit-exact first, sabotage-calibrated tolerance only with calibration evidence; (b) structural golden: the new 9-node spelling (MaxDim/Unsqueeze/Bcast/Sub/Exp/SumDim/Unsqueeze/Bcast/Div) op-sequence snapshot; (c) totality: wrong-params payload → `decompose` returns `id` (fixpoint); (d) regression: `canonical_pattern` still matches the OLD user-spelled 7-node subgraph.
Green: `decompose_via_recipe` (D6) + softmax_last_dim data recipe. Reconcile `emit_matches_softmax_last_dim_decompose` (`runtime_fused.rs:900`): repoint its oracle at the frozen legacy builder OR the new data path with the updated expected structure — state which in the commit.
Files: `fuel-graph/src/registry.rs` (helper), `fuel-graph/src/registry/softmax_last_dim.rs`, `fuel-graph/src/runtime_fused.rs` (tests).

**T6 — Rope migration. (M)**
Red: polymorphic rope decompose at `[2,4]` (rank 2) AND `[1,2,3,8]` (rank 4) — numerics vs frozen legacy builder, expect BIT-EXACT (D4 keeps emission byte-identical: 11 nodes incl. both Reshape pads; assert the node sequence equals legacy). Slice attrs in the recipe: `start=Const(0), len=Div(Extent(0,LAST),2)` and `start=Div(E,2), len=Sub(E, Div(E,2))` — the reference-doc worked example.
Green: 9-node data recipe (Bcast(SameAs0)cos / Bcast(SameAs0)sin / Slice×2 / Neg / Concat(axis_last) / Mul×2 / Add). Verify `emit_matches_rope_decompose` (`:937`) stays green unchanged; run fuel-core (`rope_with_tables_decomposed` consumer) in the gate.
Files: `fuel-graph/src/registry/rope.rs`, tests in `runtime_fused.rs`/`registry/rope.rs`.

**T7 — rms_norm + layer_norm migration (eps open slots). (M)**
Red: per-op polymorphic numeric parity vs frozen legacy at two shapes; PLUS an eps-wiring proof: two fused instances (eps=1e-5 vs 1e-6) lower to graphs whose realized outputs differ accordingly (proves the projection→slot path, not a baked constant). Wrong-params → `id`.
Green: recipes with `AddScalar` open slot (empty `scalars`) + projection `RmsNormLastDim{eps}→vec![eps]` / `LayerNormLastDim{eps}→vec![eps]`; `MeanDim(axis_last)` + `Unsqueeze(append)` swap (bit-exact — metadata-only change). Reconcile `emit_matches_layer_norm_last_dim_decompose` (`:972`) as in T5.
Files: `fuel-graph/src/registry/rms_norm_last_dim.rs`, `fuel-graph/src/registry/layer_norm_last_dim.rs`, tests.

**T8 — softmax_last_dim_backward migration. (S)**
Red: polymorphic numeric parity vs frozen legacy at two shapes, including through the `BackwardKind::Fused(SOFTMAX_LAST_DIM_BACKWARD)` autograd path (build a softmax, backprop, realize).
Green: 7-node recipe (Mul / SumDim(axis_last) / Unsqueeze / Bcast(SameAs0) / Sub / Mul).
Files: `fuel-graph/src/registry/softmax_last_dim_backward.rs`, tests.

**T9 — Matmul role-vector serialize/resolve. (M)** *(logic-independent of T2–T8; sequenced here for file-conflict avoidance)*
Red: (a) golden: `to_canonical_bytes(MatMul)` with roles `[1,3]/[3,2]` == the LOCKED 16 bytes `0C000000|02000000|0103|02000000|0302`; empty roles == `[00,00,00,00]` (existing golden `lib.rs:339` must NOT change); (b) `matmul_roles(2,2)==([1,3],[3,2])`, `matmul_roles(4,4)==([0,0,1,3],[0,0,3,2])`; (c) resolver: canonical roles → `Some(Op::MatMul)`; transposed `[3,1]` lhs / multi-K / FreeN-before-K → `None` (registration surfaces the miss, never crashes); empty → `Some(Op::MatMul)` (implicit-accept).
Green: `put_u8_list`, `lhs_roles`/`rhs_roles` fields, named `to_canonical_bytes` arm, `matmul_roles` helper, `tag_to_op` cell check per D5.
Files: `fuel-kernel-seam-types/src/lib.rs`, `fuel-graph/src/runtime_fused.rs`.

**T10 — Docs in the same change. (S)**
Update `docs/recipe-signature-reference.md` (Part II §A: migrated-set table 5/22 + rel-attr machinery as-built with anchors; §C: role vectors SHIPPED; the three-homes status), `ROADMAP.md` recipe-grammar bullet (slice-1 shipped scope + remaining-slices pointer), check `docs/architecture/04-optimization.md` claims (decompose spelling change for softmax/norms — add a `10-decisions-log.md` entry ONLY if a versioned claim changed).
Files: `docs/recipe-signature-reference.md`, `ROADMAP.md`, `docs/architecture/*` (conditional), this plan committed at `docs/superpowers/plans/2026-07-23-increment-c.md`.

## §4 Task ordering & worktree discipline

T1 → T2 → T3 → T4 → T5 → {T6, T7, T8 in any order, serial} → T9 → T10. All tasks share two hot files (`fuel-kernel-seam-types/src/lib.rs`, `fuel-graph/src/runtime_fused.rs`) — **serial execution in ONE worktree**, no parallel subagents on this thread. All cargo runs FOREGROUND (subagent bg-cargo deadlock lesson), `-p` scoped ALWAYS, one invocation at a time.

## §5 Gates (all CPU; no GPU required for this slice)

- `cargo test -p fuel-kernel-seam-types` — moved codec goldens + width pin + role-vector goldens.
- `cargo test -p fuel-graph` — resolver, emit, migrations, parity oracles, registry, opt.
- `cargo test -p fuel-dispatch` (default features, CPU) — fkc shape-expr through the shim + return_check + jit_ingest recipe-identity (live-computed hashes stay self-consistent).
- `cargo test -p fuel-core` (default features, CPU) — lazy.rs decompose parity/gap-posture + `rope_with_tables_decomposed` consumer.
- OPTIONAL post-merge, local, one-suite-at-a-time (GPU): `cargo test -p fuel-dispatch --features cuda -- --ignored` kernel-parity sanity — expected unaffected (no kernel or emitted-rope-graph changes; softmax/norm decode paths use fused kernels, decompose only on kernel-miss).

## §6 Risks

1. **SumDim/MaxDim vs ReduceSumTo/ReduceMaxTo numeric drift (CPU kernels' accumulation order).** Mitigation: bit-exact assertion first; if red, sabotage-calibrated tolerance with evidence in-test; if calibration exposes real drift, hold the SUM swap and land MaxDim-only + escalate (the swap is ratified but not at the cost of silent numeric change).
2. **`emit` reorder disturbs the scalar-cursor order or the malformed-region fallback.** Guard: existing tests `:813`, `:843`, `:740-756` must stay green untouched; the reorder keeps cursor-fill pre-order by construction.
3. **Hidden decompose→pattern roundtrip tests go red on the new softmax spelling.** If found: extend the `Callable` matcher to ALSO match the new spelling in the same task (small); do not delete the old-spelling match.
4. **`OpAttrs` field additions break exhaustive struct literals or a published-crate consumer.** `OpAttrs` derives `Default`; audit construction sites in T2. Additions are semver-minor on the published seam crates; `OpTag` is `#[non_exhaustive]` (compile-safe downstream). Flagged to Baracuda in the coordination note.
5. **KISS #67 lands mid-build and amends an arm we touch.** Our slice touches only LOCKED arms (matmul roles = closed mutual; reduce row = pinned; rel attrs = deliberately unserialized). Leaf arms deferred until ack (D7).
6. **CUDA decode regression via changed base maps.** Rope emission is byte-identical (D4) — risk retired. Softmax/norm decode uses fused kernels (decompose only on kernel-miss); residual risk is the optional §5 GPU sanity run.
7. **Parallel-thread worktree collision** (other threads this session touch fuel-graph). This thread's hot files: `runtime_fused.rs`, `registry/*.rs`, seam-types `lib.rs`, `fkc/shape_expr.rs` shim. Orchestrator must not co-schedule another thread on these.
8. **Frozen-legacy-oracle hygiene.** Each migration task copies the legacy imperative builder into the test module BEFORE replacing the live one, and the parity test must be OBSERVED running red then green (the never-ran-test failure mode is banned).

## §7 Explicitly out of scope (slice 1)

Flat-DAG-CSE table + node/table WIRE serializer (KISS #67-gated); leaf-token byte arms (ask-gated, D7); `reduced_count` executable leaf + shape-derived scalar slots (blocks rms/layer_norm BACKWARD); PowI/Clamp/MaskedFill carriers (blocks powi_backward, reduce_max_to_backward); param-conditional recipes (flash_attn, flash_attn_backward, paged_attn, causal_conv1d, fused_softmax_cross_entropy, nf4_matmul — stay imperative, which remains legal: `FusedOpEntry.decompose` is unchanged); fused_linear (interior-donor broadcast → flat-table `SameAs(node)`); scan flat serialization (selective_scan/ssd_chunk_scan data form); basis-gap 4 (IR primitives, C-4 program); reserved `Dims`/`WithDim` promotion; §6.19 import/decoder (M-3); matcher-predicate use of the new attr fields.

## §8 External coordination

Drafts live in the session's structured output `external_coordination` (KISS #67 note incl. leaf-arm byte proposals + `max_dim` token heads-up; Baracuda FYI on the role-vector binary arm + additive seam-crate surface; both reiterate the two-width do-not-unify pin). Orchestrator sends; positions get committed to `docs/outreach/` + pushed per the collaboration norms.

## §9 Roadmap for the rest of Increment C (slices 2–5)

- **Slice 2 — carriers + the remaining first-order migrations.** PowI carrier (serialize stays `put_f64_list` per the pinned arm; `tag_to_op` reconstructs with a lossless-integer check), Clamp (2-slot arity), MaskedFill (`Scalar` from `scalars[0]` + `cast_dtype`); per-entry projections computing param arithmetic (`exp−1`); shape-derived scalar slots (projection signature gains `&[Shape]` — covers `MulScalar(n)`; the DATA-portable form is the `reduced_count` leaf, flat-DAG-era) → migrate `powi_backward`, `reduce_max_to_backward`, `rms_norm_last_dim_backward`, `layer_norm_last_dim_backward`; leaf-token byte arms after KISS ack (D7) incl. the §6.12 fold-axis-resolver-reuse constraint; extend `canonical_pattern`s to the new spellings.
- **Slice 3 — flat-DAG in-memory representation.** Indexed node table (`child_edges` as table indices), `emit` memoization (`Vec<NodeId>` keyed by table index — shared interiors emit ONCE), settle the risk-2 maintainer choice from the reference doc (is the flat table *materially* CSE'd or only identity-CSE'd; decide for/against `by_pattern_hash`/`PatternHash` wiring, `registry.rs:750, :852-857`), migrate `fused_linear` via flat-table `SameAs(node_index)` (propose to KISS first if the wire form is affected).
- **Slice 4 — wire + import.** The node/table serializer once #67 pins the envelope; `SEAM_CAP_RECIPE_IMPORT` (FEAT bit 35); the §6.19 decoder with M-3 None-vs-zero rules; scan flat serialization (`child_edges = [init_carry, xs.., consts.., body_new_carry, body_y]`) → selective_scan/ssd_chunk_scan as data; ingest-side resolution of `runtime_scalar`/`reduced_count` leaves (honest-miss until executable).
- **Slice 5 / adjacent tracked programs.** Param-conditional recipe arms (flash_attn family — or a documented permanent-imperative posture); C-4 reserved `Dims`/`WithDim` + param threading (already ROADMAP-tracked); basis-gap IR primitives (`Im2Col`/`Col2Im`/GGUF-unpack/`AffineInplace`).
