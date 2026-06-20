# FKC Fusion Patterns — declarative subgraph patterns so a backend's fused kernel auto-wires on import

**Status: DRAFT for review (2026-06-19, rev 2), branch `feat/kernel-contracts-dlpack`.** Extension to
the Fuel Kernel Contract Format (FKC). Reviewed adversarially against Fuel's real fusion matchers;
rev 2 corrects the op vocabulary (graph `Op`, not `OpKind`), scopes patterns to *decomposable*
fusions, and adds node-identity guards + the guard/extract expression grammars.

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

## 1. Two kinds of fused op — and which needs a `pattern:`

| Kind | How it enters the graph | Auto-pickup on offer? | Needs `pattern:`? |
|---|---|---|---|
| **Builder-only / coarse op** — e.g. `FlashAttn`, `PagedAttn`, `Rope`, `Conv2D`, `CausalConv1d`, `SelectiveScan`, `SsdChunkScan`, `FusedSoftmaxCrossEntropy` | A model author calls the builder directly (`tensor.flash_attn(…)`); the node is a single coarse op from birth — there is **no primitive subgraph** to recognize (these ops' `canonical_pattern` is literally `None`). | **YES, already** — dispatched as a single `OpKind`, keyed `(op_kind, dtypes, backend, kernel_source)`. A backend's kernel for it auto-wires via the *primitive* path (§0). | **No.** A `pattern:` is neither needed nor possible (no subgraph exists). |
| **Pattern-recognized fusion** — e.g. `FusedLinear` (`matmul+bias`), `SoftmaxLastDim`, `RmsNormLastDim`, `LayerNormLastDim`, `LogSoftmaxLastDim` | The model is written with **primitive ops**; Fuel's optimizer **recognizes the subgraph** and rewrites it to the fused op. | Only if Fuel has the pattern to recognize the subgraph. | **Yes** — the `pattern:` block is exactly this. |

**So `pattern:` is for pattern-recognized fusions only.** If your fused kernel is a coarse op a caller
invokes directly (most attention/conv/scan kernels), ship it as an ordinary FKC contract — it
auto-wires already, no pattern. Author a `pattern:` only when you want Fuel to *discover* your fused
kernel inside graphs the user wrote with primitives.

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
consumers: <1 | any>       # OPTIONAL; default 1 (a skipped wrapper is subject to the same
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
3. **Conjunctive, first-fail.** All op/arity/`consumers`/`guard`/identity checks must hold or the
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
`Cos`, `Tanh`, `Sigmoid`, `Silu`, `Gelu`, `GeluErf`, `Erf`, `Relu`, `Step`, `Floor`, `Ceil`,
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
error); `self.<attr>` (an op attribute — `self.axis` for reduction ops, `self.target_shape.dim[i]`
for `Reshape`/`BroadcastTo`, `self.min`/`self.max` for `Clamp`); `operand(j).<…>` where `operand(j)`
is the **j-th operand node of the node carrying the guard** (node-relative; nestable:
`operand(0).operand(1).dim[-1]`), exposing the same `rank`/`dim[i]`/`self.<attr>` accessors on that
node; and `input(i).<…>` — the node bound at `bind: i` (the i-th fused-op input) — so a guard on one
branch can **cross-reference a bound input on another branch** (e.g. a keepdim `Reshape` vs the
original `x`), exposing the same accessors. Dtypes: `F16`, `BF16`, `F32`, `F64`, `U8`, … (the FKC
dtype names). Operators, by **precedence,
loosest to tightest**: `or` < `and` < {`==`,`!=`,`<`,`<=`,`>`,`>=`} < `%`. So
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
          guard: { shape: "rank == 1 and dim[0] == operand(0).dim[-1]" }   # operand(0)=the MatMul
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
                      guard: { shape: "self.axis == input(0).rank - 1" }   # reduce x's last axis
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

**Deferred (flagged, not v1):** commutative-operand matching (v1 is positional — emit both orderings
or rely on Fuel canonicalization); variadic operands (fixed arity in v1); symbolic/`DynScalar`/`bool`/
`Option` param extraction (belongs to builder-only coarse ops, §1, which need no pattern); a
fully-declarative `decompose` (rebuild the subgraph from the pattern, retiring the provider fn);
multi-output fused ops (single-sink in v1); a typed match-failure reason surfaced to telemetry.

---

## 10. The ask

For a provider that already generates FKC contracts: for each **pattern-recognized** fused kernel
(one that replaces a primitive subgraph a user would otherwise write — fused-linear, activation+linear,
norm fusions), **emit a `pattern:` block** (§3) over the graph-`Op` vocabulary (§4). That makes Fuel
auto-recognize and dispatch your kernel on import, with no Fuel-side glue — the "offer it and it's
used" property primitives already have. Your **coarse/builder-only** kernels (attention, conv, scan)
need **no** pattern — they auto-wire as ordinary FKC contracts (§1). Review the grammar (§3, §5, §6)
and the vocabulary (§4); flag anything a pattern-recognized fused kernel of yours can't express.
