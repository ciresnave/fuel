# Convergence-C · Increment C-3 — activate the shape-oracle cross-check across the registry

> **For agentic workers:** implement task-by-task with TDD; each task ends with a run + commit.

**Goal:** Turn on the FKC shape-oracle cross-check (declared §6.20 contract string vs. the registry `shape_rule` fn) for the ops that are expressible in the shipped C-1/C-2 vocab, taking the oracle from **9 → ~16 of 22** actively-validated fused ops.

**Architecture (the concrete meaning of "migrate the 22 decomposes onto the vocab"):** The registry `FusedOpEntry::shape_rule` fn (`fuel-graph/src/registry/<op>.rs`) is the **reference oracle** and is the single source of truth — it is **never rewritten**. `cross_check_fused_section` (`fkc/return_check.rs`) validates a contract's DECLARED shape_rule string against that fn, double-gated: **Gate A** = `eval_shape_rule` can evaluate the string; **Gate B** = `synth_probe_params` returns the op's `FusedOpParams` variant. C-3 opens gates for the expressible ops via three levers, ALL in `fuel-dispatch/src/fkc` + the fused contract docs — **NONE in `fuel-graph/src/registry`**.

**Tech stack:** Rust (edition 2024), `fuel-dispatch`, `fuel-graph::registry::FusedOpParams`.

## Global Constraints

- **§4 independence guardrail (load-bearing):** derive each op's shape rule DIRECTLY from the §6.20 vocab (`SameAs` / single `DimExpr` / role-woven `matmul`), keyed off role/index structure — **NEVER** by resolving `entry.decompose` to primitives and reading the terminal node's shape (that repoints the reference onto a shared decomposition table, exactly what §4 flags). The differential reference stays `(entry.shape_rule)(&in_shapes, p)` at `return_check.rs:~165`.
- **Never-panic synth invariant:** `synth_probe_params` returns `Some(params)` ONLY when the variant NAME matches, else `None`. A real registry `shape_rule`/`dtype_rule` fn is invoked ONLY when both the declared rule is evaluable AND synth returned `Some`. Every op added here has a params-INDEPENDENT `same_as`/`matmul` shape rule (verified by the scoping pass), so the wrong-params panic in `qmatmul`/`conv2d` stays unreachable.
- **No false rejects:** an op whose shape is inexpressible in the core vocab stays `Ok(None)` = skip (a coverage gap, never a false reject). Do not force it.
- **Build discipline (CLAUDE.md):** `-p fuel-dispatch` only, never workspace-wide; one cargo invocation at a time.
- **All work in:** `fuel-dispatch/src/fkc/{return_check.rs, shape_expr_parse.rs}` + `docs/kernel-contracts/fused/*.fkc.md` (+ the causal_conv1d CUDA contract). None in `fuel-graph/src/registry`.

## Synth values (params-independent placeholders — any valid value works; shape rules ignore them)

- `InplaceAffine { mul: 1.0, add: 0.0 }`
- `FlashAttn { softmax_scale: 1.0, causal: false, window_size_left: None, window_size_right: None, softcap: None, k_len: None }`
- `PagedAttn { softmax_scale: 1.0, block_size: 16, softcap: None }`
- `FlashAttnBackward { softmax_scale: 1.0, causal: false, window_size_left: None, window_size_right: None, softcap: None }`
- `SelectiveScan { delta_softplus: false }`
- `SsdChunkScan { chunk_size: 1 }`

---

### Task 1 (Tier 1a): `InplaceAffine` → live `same_as(x)` cross-check
**Files:** `fuel-dispatch/src/fkc/return_check.rs` (`synth_probe_params`, ~line 43-58; test module).
- `inplace_affine`'s CUDA FKC contract already declares `same_as(x)` (Gate A passes); it is skipped only because `InplaceAffine` is absent from `synth_probe_params`. Add `Some("InplaceAffine") => Some(FusedOpParams::InplaceAffine { mul: 1.0, add: 0.0 })`.
- **TDD:** add a test that `synth_probe_params(Some("InplaceAffine"))` returns the matching variant; and (integration) that the cross-check now fires + passes for a `same_as(x)` contract against the registry fn (mirror the existing `cross_check_fused_section` test in `register.rs:1059`). Run `cargo test -p fuel-dispatch --lib fkc::return_check` → green. Commit.

### Task 2 (Tier 1b): `FlashAttnBackward` Q/K/V → 3 live `same_as` cross-checks
**Files:** `return_check.rs` (synth); `docs/kernel-contracts/fused/attention.fkc.md` (or the FA-backward contract — author the declarations, currently FKC-UNTRACKED).
- Add `Some("FlashAttnBackward") => Some(FusedOpParams::FlashAttnBackward { softmax_scale: 1.0, causal: false, window_size_left: None, window_size_right: None, softcap: None })`.
- Author three per-variant declarations: `dQ = same_as(q)` (operand 0), `dK = same_as(k)` (operand 1), `dV = same_as(v)` (operand 2) — the registry fns `shape_rule_q/k/v` already return `input_shapes[0/1/2].clone()`. Additive (no existing string to reconcile).
- **TDD:** cross-check fires + passes for each of the three FusedOpIds. Run + commit.

### Task 3 (Tier 2 + Tier 1c): `flash_attn` + `paged_attn` doc fix + synth
**Files:** `docs/kernel-contracts/fused/attention.fkc.md`; `return_check.rs` (synth).
- Both registry fns return `input_shapes[0]` (= q), but the contract UNDER-declares `from_params(q)` (Gate A declines). Rewrite the `shape_rule` string `from_params(q)` → `same_as(q)` for both. Pin operand-role labels while editing (q = operand 0; bind to q, not k/k_cache/v_cache; covers the 5- and 6-input alibi forms with one `SameAs`).
- Add `FlashAttn` + `PagedAttn` to synth (values above).
- **TDD:** both cross-checks green (declared `same_as(q)` == registry fn output). Run + commit.

### Task 4 (Tier 3 — the one real code build): wire `matmul(a,b)` into the oracle
**Files:** `fuel-dispatch/src/fkc/shape_expr_parse.rs` (parse `matmul`); `fuel-dispatch/src/fkc/return_check.rs` (`eval_shape_rule` dispatch); test module.
- `fused_linear` already declares the canonical `matmul(a, b)` and is already in the Gate-B allowlist (`FusedLinear`), but `matmul(...)` declines at Gate A because nothing parses it. Wire it:
  - In `shape_expr_parse.rs` (or a sibling helper): recognize `matmul(role_a, role_b)`, resolve the two roles → operand positions.
  - In `eval_shape_rule`: when the rule is a `matmul(a,b)` form, build the two operands' shapes from the combo and call the **already-shipped** `shape_expr::matmul_shape(lhs, rhs)` → `Ok(Some(Shape))`. This is a WHOLE-shape (multi-dim) result, so it returns a `Shape` directly (unlike a `DimExpr` single-dim). Keep the §4 guardrail: `matmul_shape` is derived from operand shapes + the M/N/K role structure, not from `entry.decompose`.
  - The bias operand (rank-1 `[N]`) is not read by the shape rule — correct.
- **TDD:** `eval_shape_rule("matmul(a, b)", combo, "k")` returns the right shape for a combo (e.g. a=[8,4096], b=[4096,1024] → [8,1024]); and the `fused_linear` cross-check goes green vs `matmul_output_shape` (which agrees byte-for-byte). Confirm reduce/gather woven kinds are NOT needed (no current fused op requires them). Run + commit.

### Task 5 (scan slot-0): `SelectiveScan` + `SsdChunkScan` → slot-0 bundle validation
**Files:** `return_check.rs` (synth).
- Add both to synth (values above). Their slot-0 `y = same_as(u)` / `x` is already declared; adding synth turns on the bundle cross-check (arity + slot-0 rank). Slot-1 `last_state` stays `from_params` = skip (C-4).
- **TDD:** the slot-0 bundle cross-check fires + passes. Run + commit.

### Task 6 (hygiene): fix `causal_conv1d` contract drift + document the C-4 scope-out
**Files:** the `causal_conv1d` CUDA `.fkc.md` (locate it — declares `shape_rule: same_as(x)`); a doc note.
- `same_as(x)` is INACCURATE for `kernel>1` (out_seq shrinks by `kernel-1`; Mamba K=4 → off by 3). Since `CausalConv1d` is NOT in synth, the oracle never fires it today — but retarget the string to a non-evaluable `from_params(...)` form so a future synth expansion can't surface a mismatch (or a false-green at `kernel==1`). Add a one-line regression note.
- Document the intentional C-4 scope-out set (see below).
- Run `cargo test -p fuel-dispatch --lib fkc` (no regression). Commit.

---

## Verification (whole increment)
- `cargo test -p fuel-dispatch --lib fkc` → all green, no regression; new cross-checks fire + pass.
- Confirm the oracle now validates ~16 of 22 (was 9).
- Never-panic invariant preserved (synth matching-variant-or-None).
- Final: an **ultracode adversarial review** of the C-3 diff (correctness of the synth values, the §4 guardrail, no false-green risk, the matmul wiring).

## C-4 frontier (documented scope-out — intentional `Ok(None)` skips, NOT this increment)
The reserved `Dims`/`WithDim` tags (`TAG_DIMS=0x0B`, `TAG_WITH_DIM=0x0A`) + param-value threading into `eval_dim` are the C-4 successor. They cover the ~7 genuinely-inexpressible ops: `conv2d`, `conv_transpose_2d` (rank-4 + param threading), `qmatmul` (N from param + packed rhs), `nf4_matmul` (transposed-packed-weight matmul variant; also `registrable:false`), `fused_softmax_cross_entropy` (reduction-conditional rank-0 scalar), and the two scan `last_state` slots (multi-dim reweave). All correctly stay `Ok(None)` skips today — never false rejects.
