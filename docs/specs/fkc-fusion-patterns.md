# FKC Fusion Patterns — declarative subgraph patterns so a backend's fused kernel auto-wires on import

**Status: DRAFT for review (2026-06-19, rev 3), branch `feat/kernel-contracts-dlpack`.** Extension to
the Fuel Kernel Contract Format (FKC). Reviewed adversarially against Fuel's real fusion matchers
(rev 2) and against Baracuda's auto-generating-provider review (rev 3). Rev 2 corrected the op
vocabulary (graph `Op`, not `OpKind`) and added node-identity guards. **Rev 3** fixes the two §8
type-check bugs, makes **commutative-operand canonicalization normative** (§3a.2a — the one blocking
review item), unifies guard/extract auto-skip, adds the `input()` phasing rule, extends the dtype
list + resolves `Bool`→`U8`, pins the `Gelu`/`GeluErf` flavors, and re-sequences multi-output + an
import-time never-match lint ahead of cosmetic deferrals. Per-point resolutions: §11.

**Audience:** kernel providers who ship **fused** kernels and want Fuel to use them automatically on
contract import. This document specifies the new `pattern:` block; it references the base FKC spec
for the surrounding contract (`accept`/`return`/`op_params`/`cost`/`precision`/`entry_point`).

---

## 0. The problem this closes

Fuel imports an FKC contract and **registration = the kernel is dispatchable**. For a **primitive**
kernel (one op — a matmul, a gelu) that is the whole story: the contract carries the dispatch key
`(op_kind, dtypes, backend, kernel_source)` + an `entry_point`, and the op is live immediately. **A
primitive kernel a backend offers is picked up as soon as it is offered.**

Some **fused** kernels need more. The wrinkle is *how the fused op enters the graph* — and there are
**two kinds** (§1). For the kind that Fuel must **recognize** out of a primitive subgraph (a
`matmul→bias-add`, a softmax built from `max/sub/exp/sum/div`), the optimizer needs a **pattern**:
a description of *which primitive subgraph the fused kernel stands in for*. **Today the FKC contract
does not carry that pattern** — it lives as hand-written Rust inside Fuel (`canonical_pattern`
functions), or, for ops registered with the not-yet-implemented declarative form, it does not fire
at all (`PatternKind::Declarative => false`). So such a fused kernel is *registered but unused* until
someone hand-authors its matcher in Fuel. **This spec lets the contract carry that pattern
declaratively**, so the fused kernel auto-wires on import.

---

## 1. When a fused op needs a `pattern:` — and the recipe principle (every fused op should be decomposable)

| Kind | How it enters the graph | Auto-pickup on offer? | Needs `pattern:`? |
|---|---|---|---|
| **Builder-only / coarse op** — e.g. `FlashAttn`, `PagedAttn`, `Rope`, `Conv2D`, `CausalConv1d`, `SelectiveScan`, `SsdChunkScan`, `FusedSoftmaxCrossEntropy` | A model author calls the builder directly (`tensor.flash_attn(…)`); the node is a single coarse op from birth, so there is **no primitive subgraph in the graph** to recognize (these ops' `canonical_pattern` is `None` today). | **YES, already** — dispatched as a single `OpKind`, keyed `(op_kind, dtypes, backend, kernel_source)`. A backend's kernel for it auto-wires via the *primitive* path (§0). | **No** — not needed (the op is already in the graph), and for data-dependent ops not possible. *See the note below — this is a practical default, not a hard rule.* |
| **Pattern-recognized fusion** — e.g. `FusedLinear` (`matmul+bias`), `SoftmaxLastDim`, `RmsNormLastDim`, `LayerNormLastDim`, `LogSoftmaxLastDim` | The model is written with **primitive ops**; Fuel's optimizer **recognizes the subgraph** and rewrites it to the fused op. | Only if Fuel has the pattern to recognize the subgraph. | **Yes** — the `pattern:` block is exactly this. |

**So `pattern:` is for pattern-recognized fusions only.** If your fused kernel is a coarse op a caller
invokes directly (most attention/conv/scan kernels), ship it as an ordinary FKC contract — it
auto-wires already, no pattern. Author a `pattern:` only when you want Fuel to *discover* your fused
kernel inside graphs the user wrote with primitives.

> **Why two kinds — and a correction: a coarse-written op is NOT a dead end for fusion.** An earlier
> draft claimed that if a model is written with the coarse builder (`flash_attn(q,k,v)`) the op is "in
> the graph from birth, so a pattern is moot." **That is wrong.** Fuel already has the machinery to go
> the other way: **`LoweringRule`** (`fuel-graph/src/opt.rs:338`) matches `Op::Fused(id, _)` and
> **decomposes it to its primitive subgraph** via the entry's `decompose` fn, and the **`FusionRule`**
> then **re-fuses** that primitive graph — lowering runs *before* fusion in the pass order. So Fuel
> can take coarse model code, **lower it to primitives, and re-fuse to the best available kernels —
> including a provider's specialization the model author never wrote** (a larger fusion that absorbs
> an adjacent op, a tighter variant, a backend-specific cell). This is exactly the
> canonicalize-to-primitives-then-search-for-the-best-covering optimization (an e-graph already does
> the operand reordering this relies on). **So a `pattern:` is valuable for *any* op with a
> decomposition** — not only ops users happen to write with primitives — because it is what makes a
> fused kernel *discoverable* in that re-fusion.
>
> **The governing principle (stronger than "two kinds"): every fused op SHOULD have a primitive
> recipe.** A fused op is, by definition, a faster way to compute some composition of primitives —
> so it has a recipe, expressed in **two directions of the same thing**: a **`decompose`** (the
> *break-down* — fused op → its primitive subgraph, which *lowers* it onto the **base map**, the
> primitive-level form of the model) and a **`pattern:`** (the *build-up* — recognize that primitive
> subgraph and re-fuse to the op). **Both are required for the base-map optimization** the whole
> telemetry/specialization story rests on: you *generate* the base map by lowering every op to
> primitives, and you *optimize* by re-fusing the base map to the best available kernels. An op with
> no recipe is an **opaque island** — invisible to base-map analysis (the co-occurrence /
> missing-fusion telemetry can't see across or inside it) and impossible to re-fuse.
>
> So "can't be decomposed" is **not a fundamental category** — it is a gap, of one of two kinds:
> - **The primitive basis is incomplete.** Paged-attention *does* decompose — to `Gather` (the
>   block-table indexing is a normal data-dependent *value*, not a structural unknown) `→` attention;
>   selective-scan/SSM decomposes once Fuel has a higher-order **`Scan`** primitive (today it lacks
>   one), MoE once the control-flow primitives (`Branch`, which Fuel *has*) are used. The fix is to
>   grow the basis, not to declare the op un-recipe-able. (The recipe is the *math* definition; the
>   kernel is a faster numerically-close implementation, governed by the FKC `precision` tolerance —
>   e.g. flash-attn's online softmax vs its materialized `softmax(QKᵀ)·V` recipe.)
> - **The `decompose` is deliberately withheld.** Some builder-only ops `panic!` in `decompose`
>   today (`nf4_matmul.rs`: "no registry-layer decomposition… exposing that round-trip would defeat
>   the point") — because *unconditional* lowering can produce a worse graph. But **cost-guided**
>   lowering removes that objection: Fuel lowers an op to the base map only when re-fusion finds
>   something at least as good (else it keeps the coarse op), so *having* the recipe is always safe
>   and never strands a model on a slow primitive form.
>
> **Conclusion:** every fused op should carry both halves of its recipe; an op that can't be
> decomposed marks a basis gap to fill or a withheld decompose to wire, not a permanent class. This
> spec specifies the **build-up** half (`pattern:`); the **break-down** half (`decompose`) is its
> inverse and equally load-bearing.
>
> **The `decompose` contract is TOTAL and never `panic!`s** (Fuel's never-panic rule): a **primitive
> decomposes to *itself*** — the recursion's fixpoint, already the identity form
> `decompose = |_g, id, _p| id` used at `fuel-graph/src/registry.rs:823` — a fused op decomposes to
> its recipe subgraph, and the base map is the **fixpoint of `decompose` over every node** (lower
> until `decompose(x) == x` everywhere; this is exactly Fuel's `optimize_to_fixpoint` rewrite model,
> where a primitive is simply a node no lowering rule fires on). A `panic!` in `decompose` (today:
> `nf4_matmul`/`flash_attn`/`selective_scan`) is therefore always wrong — it is **either** a true
> primitive that should return itself **or** a non-primitive whose recipe is missing (a bug / basis
> gap). The two are distinguished by **basis membership**, never by the return value: a node in the
> declared primitive basis returning itself is correct; a *non-basis* op that fails to decompose is a
> **surfaced opaque-op gap** (a base-map flag, → the missing-fusion/inventory telemetry), not a crash
> and not silently masquerading as primitive.
>
> **For Baracuda:** authoring `pattern:` + `decompose` for your
> fused kernels is what lets Fuel lower a model's `flash_attn`/`softmax`/`conv` calls onto the base
> map and re-fuse them into your specialized kernels — the highest-leverage use of the feature, and
> why the base map (which your `structure_key` / co-occurrence / missing-fusion telemetry operates
> over) needs every op to carry a recipe.

---

## 2. The three things a pattern-recognized fused op needs

| Piece | Carried by |
|---|---|
| The **kernel** | FKC `entry_point` → `link_registry` (base FKC) |
| The **operand / return / cost / precision** contract (incl. the input arity **N** and the `op_params` variant) | FKC `accept`/`return`/`op_params`/`cost`/`precision` (base FKC) |
| The **pattern** — which primitive subgraph this fused op replaces | **NEW: `pattern:` (this spec)** |
| The **`decompose`** — the primitive lowering used when no fused kernel exists on a backend | provider-named fn in the `link_registry` (like `entry_point`); base FKC |

---

## 3. The `pattern:` block and the pattern-node grammar

A fused-op contract (`fused_op: <ID>`, never `op_kind:`) for a pattern-recognized fusion MAY carry:

```fkc
pattern:
  root: <PatternNode>     # the op that PRODUCES the fused output (the subgraph SINK)
```

`root` is a **PatternNode**. A PatternNode is exactly one of four kinds. The table fixes which
optional keys are legal on each kind (rev-2 correction — keys are no longer Op-only):

| kind | required keys | optional keys |
|---|---|---|
| **Op** | `op:` | `operands:`, `consumers:`, `guard:`, `extract:` |
| **bind** | `bind:` | `guard:` |
| **see_through** | `see_through:`, `then:` | `consumers:` |
| **any** | `any: true` | — |

### 3.1 `Op` node

```yaml
op: <Op>              # a graph Op name from §4 (e.g. Add, MatMul, MeanDim, AddScalar)
operands: [ <PatternNode>, … ]   # ordered, one per TENSOR input (positional, exact arity — §3a.2)
consumers: <1 | N | any>         # OPTIONAL guard (§3a.4). Default: `any` on the root, `1` on every
                                 #   interior Op/see_through node.
guard: { … }          # OPTIONAL constraints on THIS node (§5)
extract: { … }        # OPTIONAL params pulled from THIS node / its subtree (§6)
```

### 3.2 `bind` node — a leaf; **and node-identity guards** (rev-2)

```yaml
bind: <index>         # this position is a LEAF; bind the producer node as the fused op's input[index].
guard: { … }          # OPTIONAL constraint on the bound node (e.g. shape).
```

The bound nodes, by ascending `index`, become the fused op's input list (`inputs[0]` ← index 0, …).
**`N` (the input count) is the operand count declared in the contract's `accept` block; the set of
`bind` indices MUST equal `[0, N)`.** A `bind: i` that appears at **more than one position means
those positions must be the SAME graph node** (a node-identity constraint). This is required for
the matchers that re-use an input — e.g. RMSNorm's `x` feeds both the numerator and the `Sqr`
(`rms_norm_last_dim.rs:203`: `if sq.inputs[0] != x_id { return None }`); softmax's pre-max `x` and
its exp-numerator each appear twice. Without identity guards a pattern over-matches. (So the rule is:
every index in `[0,N)` appears **≥1** time; repeats are identity constraints.)

### 3.3 `see_through` node — optional transparent wrappers

```yaml
see_through: [ <Op>, … ]   # value-preserving movement ops to skip (the §4.3 set)
then: <PatternNode>        # match this after skipping zero-or-more wrappers
consumers: <1 | N | any>   # OPTIONAL; default 1 (a skipped wrapper is subject to the same
                           #   sole-consumer rule as an interior node — §3a.4).
```

Greedily matches the longest run of consecutive nodes whose op is in the `see_through` set, then
matches `then` beneath. **Greedy, not lazy:** if `then`'s op is itself in the `see_through` set, the
skip consumes it; author `then: { op: X }` with `X ∉ see_through` to anchor. Transparent ops are
pure layout/movement (§4.3). **Warning:** an op whose *attributes* are semantically load-bearing must
be matched with `op:` + a `guard:`, **not** `see_through` — `see_through` discards the wrapper's
identity. (RMSNorm matches its `Reshape` with `op: Reshape` + a keepdim shape guard, NOT
`see_through`, because the reshape's target shape is the correctness check.)

### 3.4 `any` node

```yaml
any: true             # matches any single node; does NOT bind it and does NOT impose a consumer guard.
```

---

## 3a. Matching semantics (what Fuel guarantees)

1. **Root = sink, topological, deterministic.** Fuel runs each fused op's matcher over graph nodes
   in **node-id order** (a deterministic topo-compatible order); `root` matches the candidate node
   (the op producing the fused output). The matched subgraph is replaced by one `Fused(<ID>, params)`
   node whose inputs are the `bind` nodes in index order.
2. **Positional, exact arity.** `operands[i]` matches the producer of tensor-input `i`; the node's
   tensor-input arity must equal `len(operands)`. **Scalar/attribute params are NOT operands** — e.g.
   `AddScalar` has tensor-arity 1 (the scalar is an attribute, read via `extract`, §6).
2a. **Commutative ops match operands order-independently (NORMATIVE — rev 3, resolving the blocking
   review question).** Before matching, **Fuel canonicalizes the operands of commutative ops
   (`Add`, `Mul`, `Maximum`, `Minimum`) into a deterministic order** — the same structural
   canonicalization `structure_key` uses: operands are sorted by a stable key
   `(producer.op rank, then the producer's own canonical key, recursively; ties by producer
   node-id)`. A pattern's `operands:` for a commutative op is matched against that canonical order, so
   **a provider emits operands in any one ordering and it matches regardless of the order the user's
   graph presents** (an e-graph / operand-reordering optimizer cannot desync the two — both sides
   canonicalize identically). Non-commutative ops (`Sub`, `Div`, `MatMul`, …) stay strictly
   positional. This removes the §9 "emit both orderings" escape hatch for v1.
3. **Auto-skip is symmetric across guards and extract.** Both `operand(j)` in a `guard:` (§5) and in
   an `extract:` (§6) resolve to the **first non-transparent producer** — they skip `see_through`-set
   wrappers (§4.3) identically. (`input(i)` resolves directly to the bound node.)
3b. **Bind phasing.** All `bind` leaves are resolved (and their node-identity constraints checked)
   **before** any `guard:` that references `input(i)` is evaluated; an `input(i)` that is unresolved
   when its guard runs makes the guard `false` (mirroring the out-of-range rule).
3c. **Conjunctive, first-fail.** All op/arity/`consumers`/`guard`/identity checks must hold or the
   match is `None` — never partial/greedy-on-failure.
4. **The consumer-count guard is the load-bearing safety rule.** An **interior** node (any matched
   Op or see_through node that is neither the `root` nor a `bind` leaf) defaults to `consumers: 1` —
   it must feed **only** the fusion, else fusing **duplicates** its computation and Fuel **declines**.
   Override with `consumers: N` (exact) or `consumers: any`. (Softmax legitimately sets `consumers: 2`
   on its shared `exp` node, which feeds both the sum and the divide.)
5. **Rule ordering across contracts.** When several imported fused patterns match at one node, the
   one matching the **most graph nodes** wins (largest fusion); ties broken by `fused_op` id order.
   So a `Gelu(Add(MatMul,bias))` pattern beats a `FusedLinear` pattern at the same `Add` only if it
   roots higher (`Gelu`); two patterns rooted at the same node prefer the larger.
6. **Recognition only, never forced.** A pattern governs *where the fused op may apply*; the planner's
   cost model (FKC `cost`) decides fuse-vs-not, and falls back to the primitive subgraph (or
   `decompose`) when no fused kernel exists on the chosen backend. Numerical equivalence of the
   pattern and the kernel (within the FKC `precision` tolerance) is the **author's** responsibility;
   Fuel does not verify it.

---

## 4. The op vocabulary — the graph `Op` enum

A pattern matches **graph `Op` nodes**, so `op:` and `see_through:` reference **graph `Op` variant
names** (bare names: `Add`, `MatMul`, `Gelu`, `AddScalar`, `MeanDim`), **not** the `OpKind`
dispatch-key names. (Note: the coarse ops `SoftmaxLastDim`/`RmsNormLastDim`/`LayerNormLastDim` are
**not** graph-`Op` variants — in a graph they appear either as their primitive subgraph or as
`Op::Fused(<ID>,…)`; a pattern matches the **primitive subgraph**, and its result is the fused node.)

### 4.1 Matchable computational ops (the `op:` vocabulary)

**Elementwise binary** (two same-shape tensor inputs): `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Rem`,
`Maximum`, `Minimum`.

**Elementwise unary (math):** `Neg`, `Sqr`, `Sqrt`, `Rsqrt`, `Recip`, `Abs`, `Exp`, `Log`, `Sin`,
`Cos`, `Tanh`, `Sigmoid`, `Silu`, `Gelu` (**tanh approximation**), `GeluErf` (**exact erf**),
`Erf`, `Relu`, `Step`, `Floor`, `Ceil`,
`Round`, `Sign`.

**Scalar-parameter ops** (carry an immediate read via `extract`, §6): `AddScalar` (`.value: f64`),
`MulScalar` (`.value: f64`), `PowI` (`.value: i32`), `Clamp` (`.min`, `.max: f64`).

**Comparison (→ U8 mask):** `Equal`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`.

**Select / mask:** `Where` (cond, a, b), `MaskedFill`.

**Reductions** (carry a reduced-axis attribute `.axis` where noted, read via `extract`/guarded via
§5): `SumDim` (`.axis`), `MaxDim` (`.axis`), `MinDim` (`.axis`), `MeanDim` (`.axis`), `SumAll`,
`MaxAll`, `MinAll`, `MeanAll`, `ReduceSumTo`, `ReduceMaxTo`, `CumSum`, `ArgMaxDim`, `ArgMinDim`.

**Dense linear algebra:** `MatMul`.

**Shape / movement** (also see §4.3 for the transparent subset): `Transpose`, `Permute`,
`BroadcastTo`, `Reshape`, `Unsqueeze`, `Squeeze`, `Cast`, `Flip`, `Roll`, `Pad`, `Triu`, `Tril`,
`Concat`, `Slice`.

**Indexing / scatter:** `IndexSelect`, `Gather`, `IndexAdd`, `ScatterAdd`.

**KV-cache writes (in-place):** `WriteSlice`, `WriteSliceRotating`.

**In-place elementwise** (mutate input 0; same math as the non-inplace cousin — legal in a pattern
but uncommon): `ReluInplace`, `SiluInplace`, `GeluInplace`, `GeluErfInplace`, `TanhInplace`,
`SigmoidInplace`, `NegInplace`, `AbsInplace`, `SqrInplace`, `SqrtInplace`, `RsqrtInplace`,
`RecipInplace`, `ExpInplace`, `LogInplace`, `SinInplace`, `CosInplace`, `SignInplace`,
`FloorInplace`, `CeilInplace`, `RoundInplace`, `ErfInplace`, `ClampInplace`, `PowIInplace`.

**Backward (autograd) ops** (rarely pattern targets; listed for completeness): `LogSoftmaxLastDim`,
`LogSoftmaxLastDimBackward`, `PadBackward`.

### 4.2 Non-matchable structural / IR ops (NOT legal as `op:`)

These are IR plumbing, not computation, and a pattern must not match them as compute nodes:
`Const`, `Fused`, `View`, `ViewOwned`, `Branch`, `Alloc`, `ZeroFill`, `Release`, `Move`, `Copy`,
`ScatterIntoSlot`. (`Fused` may appear as a *bound input* — a fused op can consume another fused op's
output — but is not matched as a primitive.)

### 4.3 Transparent ops (the `see_through` set)

Value-preserving movement, skippable via `see_through`: **`BroadcastTo`, `Reshape`, `Transpose`,
`Permute`, `Unsqueeze`, `Squeeze`, `Contiguize`, `View`.** (Use `op:` + `guard:` instead when the
movement op's attributes/shape are load-bearing — §3.3 warning.)

> An `op:` referencing a name not in §4.1, or a `see_through:` op not in §4.3, is a contract
> validation error at import (`#[non_exhaustive]` enum — unknown names are rejected, never guessed).

---

## 5. The `guard:` expression language

A guard is a boolean expression evaluated against the node carrying it; a false guard makes the whole
pattern not match (identical to an op mismatch).

```yaml
guard:
  shape: "<bool-expr>"       # over the node's OUTPUT shape and sibling operands' nodes
  dtype: "<bool-expr>"       # over the node's dtype
```

**Grammar.** Atoms: integer literals; `rank` (the node's output rank); `dim[i]` (the node's **output**
shape at axis `i`, negative indices from the end; out-of-range axis ⇒ the guard is `false`, never an
error); `self.<attr>` (an op attribute — `self.axis` for reduction ops; `self.target_shape.dim[i]`
for `Reshape`/`BroadcastTo`; `self.min`/`self.max` for `Clamp`; `self.start`/`self.len`/`self.dim`
for `Slice`); `operand(j).<…>` where `operand(j)` is the **j-th operand node of the node carrying the
guard** (node-relative; nestable: `operand(0).operand(1).dim[-1]`), exposing the same
`rank`/`dim[i]`/`self.<attr>` accessors on that node; and `input(i).<…>` — the node bound at `bind: i`
(the i-th fused-op input) — so a guard on one branch can **cross-reference a bound input on another
branch** (e.g. a keepdim `Reshape` vs the original `x`), exposing the same accessors. **Axis
attributes (`self.axis`) and `dim[i]` indices are compared in normalized negative-from-end form** —
`self.axis == -1` means the last axis at any rank, so no `rank - k` arithmetic is needed (rev-3 fix).
**Dtypes** name the logical Fuel `DType`: `U8, I8, U32, I16, I32, I64, BF16, F16, F32, F64, F8E4M3,
F6E2M3, F6E3M2, F4, F8E8M0` (packed sub-byte / quant payloads — `F4`/`I4`/`U4`/`B1` — appear on the
tensor via the FDX sidecar, not as a distinct base dtype). Fuel has **no `Bool` dtype**: comparison
ops produce a **`U8`** mask (`1`/`0`), so a provider's `Bool` maps to `U8` here. Operators, by
**precedence, loosest to tightest**: `or` < `and` < {`==`,`!=`,`<`,`<=`,`>`,`>=`} < `%`. So
`rank == 1 and dim[0] == operand(0).dim[-1]` parses as `(rank == 1) and (dim[0] == operand(0).dim[-1])`,
and `dim[i] % k == 0` parses as `(dim[i] % k) == 0`. RHS of `%` and `dim[]` indices are integer
literals or `operand(j).dim[i]`/`self.<attr>` (no general arithmetic in v1).

---

## 6. The `extract:` expression language

`extract` populates the fused op's `op_params` variant (declared in the base-FKC `op_params` block)
from the matched subgraph. Omit it for a unit-param fused op (`FusedLinear`, `SoftmaxLastDim`).

```yaml
extract:
  <param_field>: "<path>"    # one entry per field of the op_params variant
```

**Path grammar.** A path is anchored at the **node carrying the `extract:`** (an `Op` node, usually
the root) and walks operands: `self` (this node), `operand(j)` (j-th operand node — node-relative,
nestable), then a terminal attribute accessor. `operand(j)` **auto-skips `see_through` wrappers**
(it resolves to the first non-transparent producer), so a path need not spell out broadcasts/reshapes
the pattern saw through. Terminal accessors are the op's typed attributes: `AddScalar.value` /
`MulScalar.value` (f64), `PowI.value` (i32), `Clamp.min`/`.max` (f64), `<reduction>.axis` (usize),
`Reshape.target_shape`/`BroadcastTo.target_shape`. **Symbolic / non-scalar params** (a `DynScalar`
K-length, a `bool`/`Option` flag) are **out of scope for v1 patterns** — they belong to builder-only
coarse ops (§1), which do not use `pattern:`. So `extract` in v1 carries only static scalar/axis
attributes (e.g. RMSNorm's `eps`).

Example (RMSNorm `eps`, grounded in `rms_norm_last_dim.rs:177`): the root is `Div`; the eps lives on
the `AddScalar` reached via `operand(1)` (denominator) → `Sqrt` → `AddScalar`, with the `BroadcastTo`
between `Div` and `Sqrt` auto-skipped:

```yaml
extract: { eps: "operand(1).operand(0).value" }
```

---

## 7. End-to-end wiring (what import does)

Importing a fused-op contract with a `pattern:`:
1. Registers the **kernel** (`entry_point` → `link_registry`) under the fused op.
2. Compiles `pattern:` to a matcher and registers it as the op's `SubgraphPattern` so the optimizer's
   fusion pass recognizes the subgraph and emits `Fused(<ID>, params)`.
3. Wires the FKC `return` shape/dtype rules and the `op_params` variant (base FKC).
4. Uses the provider's `decompose` (the primitive lowering) when no fused kernel exists on a backend.

After import, a model graph containing the primitive subgraph is **auto-rewritten** to use the fused
kernel — no Fuel-side code per kernel. A provider that regenerates contracts ships new
pattern-recognized fused kernels by *adding contract files*.

---

## 8. Worked examples (both type-checked against §3, grounded in the real matchers)

### 8.1 `FusedLinear` — `Add(MatMul(a,b), broadcast(bias))` (grounds to `fused_linear.rs`)

```fkc
fused_op: FUSED_LINEAR
# accept declares N=3 operands: a, b, bias.  op_params: { variant: FusedLinear }  (unit — no extract)
pattern:
  root:
    op: Add                       # subgraph SINK (produces the fused output)
    operands:
      - op: MatMul
        consumers: 1              # MatMul must feed ONLY this Add, else fusing duplicates it
        operands:
          - bind: 0               # input[0] = a
          - bind: 1               # input[1] = b
      - see_through: [BroadcastTo, Reshape]
        then:
          bind: 2                 # input[2] = bias …
          guard: { shape: "rank == 1 and dim[0] == input(1).dim[-1]" }   # input(1)=b; bias len == b's last dim
```

### 8.2 `RmsNormLastDim` — parameterized + identity + attribute guards (grounds to `rms_norm_last_dim.rs`)

`Div(x, broadcast(Sqrt(AddScalar_eps(Reshape(MeanDim(Sqr(x)))))))`, where the **same `x`** feeds the
`Div` numerator and the `Sqr`, the `MeanDim` is along the last axis, the `Reshape` is keepdim, and
every interior node is sole-consumer:

```fkc
fused_op: RMS_NORM_LAST_DIM
# accept declares N=1 operand: x.   op_params: { variant: RmsNormLastDim, fields: { eps: f64 } }
pattern:
  root:
    op: Div
    extract: { eps: "operand(1).operand(0).value" }   # Div.op(1)→[skip BroadcastTo]→Sqrt.op(0)→AddScalar.value
    operands:
      - bind: 0                   # input[0] = x  (numerator)
      - see_through: [BroadcastTo]
        then:
          op: Sqrt
          operands:
            - op: AddScalar       # carries eps (extracted above)
              operands:
                - op: Reshape     # keepdim — matched (NOT see_through) so its shape can be guarded
                  guard: { shape: "rank == input(0).rank and dim[-1] == 1" }
                  operands:
                    - op: MeanDim
                      guard: { shape: "self.axis == -1" }   # reduce the LAST axis (axes normalize negative-from-end, §5)
                      operands:
                        - op: Sqr
                          operands:
                            - bind: 0    # SAME x as the numerator — node-identity guard (§3.2)
```

Every interior `Op`/`see_through` node defaults to `consumers: 1` (§3a.4), reproducing the matcher's
sole-consumer checks. The repeated `bind: 0` enforces RMSNorm's `sq.inputs[0] == x_id`.

---

## 9. What Fuel builds vs the provider provides; deferred

**Fuel implements (this spec):** the `pattern:` field + validation; the `PatternTree` type and the
declarative **matcher compiler** (today `PatternKind::Declarative` is a never-firing stub — this spec
makes it real); node-identity, consumer-count, guard, and extract evaluation; the rule-ordering
(§3a.5); the import wiring (§7).

**A provider supplies:** the `pattern:` tree, the fused `kernel` (`entry_point`), the `op_params`
variant + `return` rules + `decompose` + `cost`/`precision` (base FKC).

**Near-term (sequenced ahead of cosmetic deferrals, per the rev-2 review):**
- **Multi-output fused ops** — v1 roots at one sink and replaces with one `Fused` node, so a
  save-stats norm/softmax (LayerNorm/RMSNorm emitting `mean`+`rstd`, softmax emitting
  `max`+`logsumexp`) is **inference-only** as a pattern; the training form must ship coarse (§1) for
  now. This is the inference-vs-training line for fused norms, so it is prioritized above the
  cosmetic items below (it rides the multi-output bundle infrastructure).
- **An import-time "structurally-can-never-match" lint** — the worst failure mode for an
  "auto-wires on import" feature is a *valid* pattern that silently never fires (recreating §0's
  "registered but unused"). A static lint that rejects a pattern whose op-chain can never occur, plus
  a **typed match-failure reason** surfaced to the fusion-miss telemetry, are pulled ahead of the
  rest. (Structurally *malformed* patterns already error at import.)

**Deferred (not v1):** variadic operands (fixed arity in v1 — fits 100% of a fixed-`n_inputs`
provider); `chunk`/`split` → `Slice` canonicalization + a `Split`/`Chunk` op (gated activations —
SwiGLU/GeGLU — ship coarse until then, expressible later via `Slice` start/len guards, §5);
symbolic/`DynScalar`/`bool`/`Option` param extraction (belongs to builder-only coarse ops, §1);
**interior** node-identity (repeated `bind` pins shared *leaf* inputs in v1; pinning a shared
*interior* subtree as one node — needed once an e-graph does interior CSE — is deferred); composite/
parameterized activations with no §4.1 anchor (Mish/Softplus/Hardswish, runtime `LeakyRelu(α)`/`ELU(α)`
— ship coarse; a `Softplus` unary + a runtime scalar-param form would unblock auto-discovery later);
a fully-declarative `decompose` (rebuild the subgraph from the pattern, retiring the provider fn).
**Resolved in rev 3 (no longer deferred):** commutative-operand matching → §3a.2a (Fuel canonicalizes;
provider emits one ordering).

---

## 10. The ask

For a provider that already generates FKC contracts: for each **pattern-recognized** fused kernel
(one that replaces a primitive subgraph a user would otherwise write — fused-linear, activation+linear,
norm fusions), **emit a `pattern:` block** (§3) over the graph-`Op` vocabulary (§4). That makes Fuel
auto-recognize and dispatch your kernel on import, with no Fuel-side glue — the "offer it and it's
used" property primitives already have. Your **coarse/builder-only** kernels (attention, conv, scan)
need **no** pattern — they auto-wire as ordinary FKC contracts (§1). Review the grammar (§3, §5, §6)
and the vocabulary (§4); flag anything a pattern-recognized fused kernel of yours can't express.

---

## 11. Rev 3 — resolutions to Baracuda's rev-2 review

**A. Spec self-consistency (must-fix) — all fixed:**

- **A1** (`self.axis == input(0).rank - 1` used `-`, which §5 forbids) → §8.2 now `self.axis == -1`;
  §5 states axes/`dim[]` compare in **negative-from-end** form, so no `rank - k` arithmetic.
- **A2** (FusedLinear bias guard read `operand(0)` from a `bind` leaf, which has no operands) → §8.1
  now `dim[0] == input(1).dim[-1]` (`b` is `bind: 1`). Both §8 examples type-check.
- **A3** (auto-skip asymmetric: extract skipped `see_through`, guards didn't) → §3a.3: **both**
  `operand(j)` in guards and extract auto-skip the transparent set.
- **A4** (`see_through` `consumers` lacked the `N` form) → §3.3 now `<1 | N | any>`.

**B. Expressiveness asks:**

- **B1 — commutativity (BLOCKING) — RESOLVED normatively (§3a.2a).** Fuel canonicalizes the operands
  of `Add`/`Mul`/`Maximum`/`Minimum` by a deterministic structural key (the same one `structure_key`
  uses) before matching; emit **one** ordering and it matches regardless of how the user's graph (or
  your e-graph) orders them. No 2ᵏ blow-up.
- **B2 — multi-output** → §9 re-sequenced **ahead** of cosmetic deferrals; it is the inference-vs-
  training line for fused norms. v1 patterns are inference-only single-sink; save-stats forms ship
  coarse until multi-output lands.
- **B3 — interior node-identity** → added to §9 deferred (v1 pins shared *leaf* inputs via repeated
  `bind`; shared *interior* subtree identity is deferred, needed once you do interior CSE).
  **Confirmed:** repeated `bind: i` IS the shared-*input* mechanism (your reading is right).
- **B4 — `input(i)` phasing** → §3a.3b: all binds resolve before any `input()`-referencing guard;
  unresolved `input()` ⇒ guard `false`.
- **B5 — dtypes / `Bool`** → §5 now lists the full logical `DType` set; **Fuel has no `Bool` dtype —
  masks are `U8`**, so your `Bool` maps to `U8`. Packed sub-byte (`F4`/`I4`/`U4`/`B1`) ride the FDX
  sidecar, not a base dtype.
- **B6 — Gelu flavor** → §4.1: bare **`Gelu` = tanh approximation**, **`GeluErf` = exact erf**.
- **B7 — gated activations / Slice offsets** → §5 adds `self.start`/`self.len`/`self.dim` for `Slice`;
  `chunk`/`split`→`Slice` canonicalization + a `Split`/`Chunk` op is §9-deferred. **Ship SwiGLU/GeGLU
  coarse until then** (you noted this is fine).
- **B8 — composite activations (Mish/Softplus/…)** → no spec change; §9 records they ship coarse, and
  a `Softplus` unary + runtime scalar-param form would enable auto-discovery later if wanted.
- **B9 — silent non-firing** → §9: an **import-time "structurally-can-never-match" lint** + a typed
  match-failure reason are pulled ahead of cosmetic deferrals.

**E. Open questions — answers:**

1. **(blocking) Commutative canonicalization?** **Yes**, normatively — §3a.2a, with the canonical
   order stated (structural-key sort; ties by producer node-id).
2. **`Gelu` flavor?** `Gelu` = tanh-approx, `GeluErf` = exact erf (B6).
3. **`extract` for eps-like static scalars?** **Yes** — that is exactly its purpose (§6); build the
   body-scalar → `AddScalar`-attribute → `op_params` bridge once.
4. **`chunk`/`split` → `Slice` canonicalization / `Split` op?** Deferred (B7); gated activations
   coarse until then.
5. **dtype `…` open + spellings + `Bool`?** Full list now in §5; `Bool` ⇒ `U8` (B5).
6. **Unify `operand(j)` auto-skip across guard + extract?** **Yes** (§3a.3 / A3).
7. **`input(i)` phasing rule?** **Yes** (§3a.3b / B4).
8. **Multi-output ahead of cosmetic deferrals?** **Yes** (B2).
9. **Match-failure / never-match lint earlier?** **Yes** (B9).

**Confirmed on Baracuda's side (no action from Fuel):** the `derive_pattern(body)` elementwise-epilogue
emitter (zero new IR), the `(*Plan, *Kind, axis) → §4.1 op-name` mapping table, the per-contract
pattern-equivalence certification gate (Fuel verifies nothing, §3a.6 — so a separate pattern-divergence
precision bound, not a naive `PrecisionGuarantee` projection, is correct), and the IR-growth sequence
(`ScalarExpr::Const` → `Unary` → reductions/DAG/layout/`MatMul`) that unlocks the §8 targets on
Baracuda's roadmap.
