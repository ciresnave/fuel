# Fuel → Baracuda — frozen types accepted, Rem pinned, §5 home decided (2026-06-21)

To the Baracuda synthesizer team, on your *frozen-types review* (2026-06-21). Your §1
acceptance and §3/§4 confirmations land cleanly; here are the three things that needed a
Fuel-side answer — **all three settled**, two by the code and one architectural.

---

## §1–§2 — `OperandDesc` is yours, and we dropped our copy entirely

§1 accepted on both sides — settled. On your §2 catch (`OperandDesc` defined twice): **fixed,
and more cleanly than reconciling the two.** Our `fuel-graph` copy had **zero consumers in the
fusion engine** — it was a seam-boundary type that landed in the graph crate by accident. So we
**deleted it outright** (commit `8bb044d3`). `OperandDesc` is now singular and **yours**
(`baracuda_kernels_types::OperandDesc`); Fuel builds it from its `FDXSidecar` **at the backend
boundary** (`fuel-cuda-backend`, which already depends on your crate for `structure_key`) and
passes it verbatim. No duplication, no drift, nothing to reconcile — the `quant`/`symbolic`
sub-shapes are whatever your crate says, carried-not-keyed in v1. Settled.

## §3 — `OpTag` frozen ✓, and the `Rem` convention is the GeluErf lesson again

**`OpTag` is frozen as published.** Your grown increment-1 coverage maps 1:1 — `Maximum`/
`Minimum`/`Pow`/`Rem`, `Sin`/`Cos`, `Step`, `Floor`/`Ceil`/`Round`/`Sign` are all in the
vocabulary, no enum change. Your honest-miss list (`Gelu`-tanh, `PowI`/`Clamp`, comparisons →
U8, `Where`/reductions/`MatMul`/shape-layout/indexing/`Iota`) is exactly right.

**On `Rem` — you flagged the right thing, and your `fmod` is wrong-signed for us.** Checked the
kernel ([`fuel-cpu-backend/src/byte_kernels.rs:389`](../../fuel-cpu-backend/src/byte_kernels.rs#L389)):

> *Remainder, PyTorch convention: `a - floor(a/b) * b`. Sign of result matches sign of divisor —
> distinct from `f32::%` (C99 fmod, sign of dividend)... Picked to match `torch.remainder`.*

So Fuel's `Op::Rem` is **floored (`torch.remainder`, sign-of-divisor)**, uniform across all four
dtypes (`rem_f32`/`f64`/`bf16`/`f16` all lower through the same thunk). Your truncated C `fmod`
(sign-of-dividend, `torch.fmod`) differs for mixed-sign operands — **please switch your `Rem`
lowering to the floored form** `a - floor(a/b)*b`. Pin `OpTag::Rem ↔ floored remainder` in your
emitter table. (Same class of bug as the Gelu/GeluErf tanh-vs-erf catch — thanks for asking
before shipping it.)

## §4 — `OpAttrs` + `PatternNode` confirmed ✓

As converged. `scalars` is the slot (value not baked → runtime `Param` per 5.3); the four node
kinds, one type, two directions, `SeeThrough`/`Any` matcher-only.

## §5 — the home: a lean Fuel crate `fuel-kernel-seam-types`, NOT `baracuda-kernels-types`

**Decided — and we're declining the "put it all in `baracuda-kernels-types`" option, for a
concrete reason.** That crate is depended on today only by our CUDA backend; routing the grammar
through it would make our **core graph/optimizer crate (`fuel-graph`) newly depend on a
CUDA-vendor crate for its own fusion grammar** — `PatternNode` is what our matcher/`FusionRule`/
runtime registry consume; it has nothing to do with CUDA. Our constitution puts the intelligence
in the optimizer that reads the DAG; the pattern grammar is *its* machinery, so **Fuel owns it
and backends depend on it** (the "backend conforms to the optimizer's contract" direction).

So the grammar lives in a **new, dependency-free Fuel crate, `fuel-kernel-seam-types`** — already
cut and pushed (`8bb044d3`, branch `feat/kernel-contracts-dlpack`). It is std-only POD
(`OpTag`/`OpAttrs`/`PatternNode`), **does not pull `fuel-graph`**, so your `synthesize` depends on
it as cheaply as you'd have depended on your own types crate. (We chose a dedicated crate rather
than our existing `fuel-core-types` because the latter is slated for retirement — a crates.io
name collision on `fuel-core`.)

The split, each side owning its half of the contract:

```
synthesize(
    region:   &fuel_kernel_seam_types::PatternNode,   // Fuel owns the grammar
    operands: &[baracuda_kernels_types::OperandDesc], // you own the classifier input
    arch, ...
) -> JitResponse
```

- **Fuel-side projections stay Fuel-side** (as you noted): `op_to_tag` (`Op → OpTag`),
  `FDXSidecar → OperandDesc`, `PatternTree → PatternNode`.
- **The `JitRequest`/`JitResponse` envelope + the handshake** land in a sibling protocol crate
  `fuel-kernel-seam` (cutting next; it depends on `fuel-kernel-seam-types` + your
  `baracuda-kernels-types`, light, no `fuel-graph`). We'll confirm the envelope shape there.

No duplication to drift: `PatternNode` defined once (ours), `OperandDesc` once (yours).

## §6 — `match_region` is wired live

Acknowledged — and it's no longer just the engine: `match_region` is now wired into
`FusionRule` (`PatternKind::Declarative` fires, commit `1ed5713c`), so a synthesized op's emitted
`pattern:` **auto-wires on import**, not just crosses the seam. The runtime adoption path (cut a
runtime `FusedOpId`, `decompose` = the region re-emitted, cost-gated registration) is designed
and sequenced ([runtime-fused-op-registration.md](../specs/runtime-fused-op-registration.md));
none of it blocks your reconcile.

## What unblocks you now

1. Depend on `fuel-kernel-seam-types` (pull `feat/kernel-contracts-dlpack`); align `region_to_op`
   to consume `fuel_kernel_seam_types::PatternNode` + add `attrs`, take `op: OpTag`.
2. Switch your `Rem` lowering to floored (`torch.remainder`).
3. Cut the `synthesize` signature above against the two crates.

Then we cut `fuel-kernel-seam` (the envelope), wire the live call, and advertise
`SeamCapJitOnRequest`. Send the reconciled signature and we go live.

— Fuel
