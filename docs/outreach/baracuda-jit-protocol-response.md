# Fuel → Baracuda — JIT-on-request protocol (§5): response + proposed wire types (2026-06-21)

To the Baracuda synthesizer team, on your **JitRequest/JitResponse proposal** (2026-06-21).
Short version: **we accept the architecture wholesale** — and the headline move (reuse the
FKC §3 `PatternNode` grammar as the region form) is exactly right; it's the choice that makes
the two halves share one node form instead of inventing a second. Below: our answer on each of
your eight decisions (§A), the **Fuel-side wire types** you asked for (§B), the transport choice
(§C), and **two honesty notes** about Fuel-side foundations this protocol stands on that we are
formalizing now (§D). We're matching your propose-first discipline: this is the contract we'll
build to, so push back before we freeze the Rust types.

---

## A. The eight decisions

**5.1 — Region = the FKC §3 `PatternNode` grammar. ACCEPTED, with the DAG line drawn.**
Yes: `JitRequest.region` is an FKC §3 `PatternNode` tree — `Op` nodes over the §4.1 graph-`Op`
vocabulary, `bind: i` leaves for external inputs, indices exactly `[0, n_inputs)`. A synthesized
op's `pattern:` is that region re-emitted; your `region_to_op` is the inverse of our
`derive_pattern` walk. One grammar, both directions.
- **Confirmed on the DAG caveat:** for **increment 1 our base-map regions are sole-consumer
  elementwise-epilogue trees**, so tree-ification (recompute of a shared interior) doesn't arise —
  and where it would, recompute of cheap elementwise interiors is acceptable, matching our §9
  deferral of interior node-identity. **The line we draw:** the tree form is correct **only while
  the recomputed interior is cheap**. The first region we'd ever hand you with an *expensive*
  shared interior (e.g. a shared `MatMul` feeding two epilogues) we will **not** tree-ify — we'll
  either split it into two requests or hold it until we've coordinated a DAG-preserving region
  form. So: tree for increment 1; DAG preservation is the first joint extension, exactly as you
  scoped it. Repeated `bind: i` (node-identity on a shared *input*) is already in the grammar and
  carries over unchanged.

**5.2 — Operands = the `FdxOperandDesc` projection. ACCEPTED (see §D-2).** `operands` is the
inputs-then-output list of the same minimal operand projection that keys `structure_key`, passed
verbatim, never re-derived. Confirmed it carries to JIT — with the honesty note that this
projection is a Fuel-side type we are formalizing as part of the telemetry/`structure_key` layer
(§D-2); the definition in §B is what we propose you map onto.

**5.3 — Scalar params → `op_params` via `extract`. ACCEPTED.** A region `AddScalar`/`MulScalar`
lowers to a runtime `Param` (launch arg) and the emitted contract's `extract:` pulls each scalar
back from its graph path into `op_params` — same as the AOT path, same `extract:` path grammar
(fkc-fusion-patterns §3a.4). That's exactly how we want the synthesized op's params bound; the
region carries the scalar attributes and they round-trip with no side channel.

**5.4 — Recipe `decompose:` = the region (interim). ACCEPTED as interim.** Since the synthesized
op is by construction equivalent to the region, its `decompose` **is** that region — emit
`decompose:` as the region's pattern-node subgraph (provisional header) and we'll consume it.
The declarative-decompose *format* is our §9-deferred item; we're formalizing it together with the
`PatternNode` grammar (§D-1), and when it lands we'll hand you the exact header so `pattern:` and
`decompose:` stay one canonical node with shared `extract:` routing. Until then your interim form
is what we read.

**5.5 — Budget = soft ceiling. ACCEPTED.** Soft is right — nvrtc has no compile-deadline API, and
the gate that matters is **ours**: synthesis is speculative, and we **cost-gate adoption after**
you return (we keep the primitive subgraph if the synthesized kernel doesn't win). So a soft
ceiling on optimization depth / e-graph iterations is sufficient; reject-zero is correct. No hard
wall-clock abort for increment 1. If a pathological compile ever shows up we'll ask for the
watchdog then, not now.

**5.6 — Target. ACCEPTED, with the backend selector.** Map as you propose: dtypes+shapes inside
`operands`, device folded into `ArchSku` for the schedule key, finer device identity refined by the
compiler when the artifact is SM-specific. We **do** want the small backend enum now (we'll add
Slang/Metal/CPU synthesizers later) — proposed as `JitBackend` in §B. No device-ordinal pinning for
increment 1 (CUDA, arch-keyed); we'll send a `DeviceTarget` when a multi-GPU adopt path needs it.

**5.7 — Honest-miss taxonomy. ACCEPTED — and miss IS non-fatal, by construction.** A typed error =
a miss = keep the primitive subgraph, no fused kernel. This is *the* honesty invariant
(decompose is always available; the fused kernel is the cost-win when present), so a miss is never
fatal on our side — it's the normal, expected branch for anything outside increment-1 scope. Your
set (`UnsupportedOp` / `UnsupportedDtype` / `MixedDtype` / `OperandArity` / `Budget` / `Compile`
with the nvrtc log) is exactly the granularity we want; we route every one to "stay decomposed"
and record it as a missing-fusion telemetry event (the §5.3 work-order feed).

**5.8 — Artifact + link row. ACCEPTED — we consume `link`, and we refuse `Stub`.** `kernel.kind`
tags the artifact and our loader **rejects `Stub`** (never feeds non-loadable bytes to the driver —
a miss instead). `JitResponse.link` is the `link_registry` row that resolves `entry_point` →
`KernelRef` at load (FKC §12.6); we **consume it** — it's the same mechanism we just stood up for
the built-in CPU provider (`CpuLinkRegistry`, a real `&[(symbol, KernelRef)]` resolver), so an
adopted JIT kernel's `entry_point` resolves through the returned `link` exactly like a contract's.

---

## B. The Fuel-side wire types (the byte form is ours — map yours onto these)

Proposed Rust types. As with FDX, **we own the byte form**; you map your native records onto these
at the boundary. These are *proposals to reconcile*, not frozen — react before we commit them.

```rust
// ── The region: the FKC §3 PatternNode grammar, in the input direction ──────
// Identical to the node form `pattern:` matching reads; `region_to_op` is the
// inverse of `derive_pattern`. (This is the SAME type as the declarative
// PatternTree we are formalizing — see §D-1.)
pub enum PatternNode {
    /// An op over the §4.1 graph-Op vocabulary, with one child per tensor input
    /// (ordered, exact arity). `attrs` carries the op's non-tensor attributes
    /// (e.g. the scalar of AddScalar) for `extract:` round-trip (5.3).
    Op { op: OpTag, operands: Vec<PatternNode>, attrs: OpAttrs },
    /// A leaf: bind the producer at this position as the fused op's input[index].
    /// Repeated index = node-identity guard on a shared input (§3.2).
    Bind { index: u8 },
    // `see_through` / `any` (§3.3/§3.4) exist in the matcher grammar but are not
    // emitted in a JIT region (a region is concrete) — included for one node form.
}

// ── The operand projection: the SAME minimal form that keys structure_key ───
pub struct FdxOperandDesc {
    pub dtype: DTypeTag,        // FDX dtype code (the shared §5-base vocabulary)
    pub shape: Vec<i64>,        // logical extents; -1 = symbolic (bound at launch)
    pub layout: LayoutFlags,    // contiguous / strided / broadcast (the FKC §3.3 five-flag set)
}

pub enum JitBackend { Cuda, Slang, Metal, Cpu }   // 5.6 — you set the synthesizer

pub struct JitRequest {
    pub region:      PatternNode,
    pub n_inputs:    u8,                  // bind indices are exactly [0, n_inputs)
    pub op_category: OpCategory,          // strategist-chosen schedule-legality key
    pub operands:    Vec<FdxOperandDesc>, // inputs then output (len == n_inputs + 1)
    pub backend:     JitBackend,
    pub arch:        ArchSku,             // target compute capability (schedule key)
    pub fused_op_id: String,             // stable identity to register the op under
    pub budget:      JitBudget,          // { max_compile_ms: u32 }  (soft; nonzero)
}

pub struct JitResponse {
    pub kernel:   JitKernel,             // { entry_point, source, artifact, kind }
    pub contract: String,               // full FKC contract block (accept/return/op_params/cost/precision/determinism)
    pub recipe:   JitRecipe,             // { pattern, decompose } — both from one canonical node (5.4)
    pub link:     LinkEntry,             // (entry_point, structure_key, revision_hash) — FKC §12.6
}

pub enum JitArtifactKind { Ptx, Cubin, Stub }   // 5.8 — loader refuses Stub
pub enum JitError {                              // 5.7 — every variant ⇒ "stay decomposed"
    UnsupportedOp(OpTag), UnsupportedDtype(DTypeTag), MixedDtype,
    OperandArity { got: usize, expected: usize }, Budget, Compile(String),
}
pub type JitResult = Result<JitResponse, JitError>;
```

Two reconciliation asks back to you:
- **`OpTag` / `DTypeTag` shared vocab.** These must be the same enumerations on both sides — they
  are the §4.1 graph-`Op` set and the FDX dtype table (FKC §10 rule 16: every token is in FDX's
  normative table). We'll publish ours as the canonical list; confirm yours matches 1:1 or send the
  delta.
- **`OpCategory` / `ArchSku`.** You named these as yours-to-key-on; we'll treat them as opaque tags
  we set (`OpCategory`) / derive (`ArchSku`) and pass through. Send your enumerations so our
  `JitRequest` constructor produces values your schedule cell accepts.

---

## C. Transport — **direct Rust for increment 1**, C-ABI deferred (§6)

We choose **direct Rust integration** for increment 1: Fuel constructs the `PatternNode` region +
the `FdxOperandDesc` projection and calls `synthesize(request, backend, compiler)` as a native Rust
call — exactly mirroring how `structure_key` is invoked. Rationale: both halves are Rust; the region
+ operands need no marshalling; and it gets the working loop end-to-end fastest (your half is built,
ours is being built — let's converge on live calls, not a C-ABI we'd shim twice). The **C-ABI
trampoline is the deferred generalization** — we'll add it when a non-Rust ecosystem needs the JIT
seam, and at that point the byte form is the §B types marshalled to C (the region + operands are the
only non-trivial part, and they're tree/array-shaped). The handshake (`baracuda_seam_hello`) stays
C-ABI as ratified; **JIT-on-request rides the Rust path until the seam is frozen.** If you'd rather
do C-ABI from day one, say so and we'll spec the marshalling before either side builds it.

---

## D. Two honesty notes — Fuel-side foundations this protocol stands on

So this is implementation-true on **our** side too: two of the types §B references are Fuel-side
foundations that are **spec'd but not yet formalized in code**, and we're building them now.

1. **The `PatternNode` grammar is currently a placeholder.** `SubgraphPattern::Declarative` carries a
   `PatternTree` that today is a placeholder newtype, and `PatternKind::Declarative` is a
   never-firing stub — i.e. our *declarative pattern engine* (fkc-fusion-patterns §3/§5/§6) isn't
   built yet. **Your proposal makes this the same work:** the `JitRequest.region` type **is** that
   `PatternNode`, and a synthesized op's `pattern:` is a region re-emitted. So formalizing the §3
   grammar as the concrete `PatternNode` enum (+ the matcher compiler) is now on our critical path,
   and it lands as one type both the JIT region and `pattern:` matching use. We'll send the frozen
   `PatternNode` definition as we cut it.

2. **The operand projection / `structure_key` path isn't wired yet.** Per kernel-seam-interop §7.3,
   the telemetry/`structure_key`/`DispatchRecord` emission is "not built" on our side — and §5.2's
   `FdxOperandDesc` projection lives in that layer. We'll define `FdxOperandDesc` (the §B shape) as
   part of standing that up, and call **your** `structure_key` to key the schedule (we never
   recompute it — that's the ratified division).

**Net:** the protocol shape is accepted as proposed; what gates a live first call is **our** two
foundations above (the `PatternNode` enum + the operand projection), not a disagreement on the wire.
We'll build both and send you the frozen Rust types; reconcile `OpTag`/`DTypeTag`/`OpCategory`/
`ArchSku` and the transport on your side, and we flip `SeamCapJitOnRequest` the moment both halves
call across the line.

— Fuel
