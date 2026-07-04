# Fuel → Baracuda — JIT seam integration: envelope aligned + budget answered (2026-07-04)

**Re:** your four §5.2 integration answers (direct-Rust surface, runtime link_registry,
concurrency, budget). All four land; the envelope is **revised to match your built
two-step handover**, and your one open question (budget axes) is answered. Two small
conform-asks on your side (§Q2). Grounded in the revised `fuel-kernel-seam` (branch
`jit-envelope-reconcile`, builds vs `baracuda-kernels-types` alpha.72).

---

## Q1 — Direct-Rust surface: **accepted, exactly as you built it**

You `impl Synthesizer for BaracudaSynthesizer` against our envelope trait; dep is
Baracuda → `fuel-kernel-seam` (+ `-types`); Fuel holds `&dyn Synthesizer` / `Box<dyn
Synthesizer>` and calls `.synthesize(&req)`; Fuel owns the instance
(`BaracudaSynthesizer::new(max_compile_ms)`) and depends on none of your types. Settled.

## Q2 — Runtime handover: **envelope revised to your `take_kernel` + `SynthArtifact` design** (+ 2 conform-asks)

Your built shape is **better than our §5.2 draft** and we adopted it: the wire response
stays light, the heavy artifact rides in the retained artifact and only crosses on adopt.
We revised the envelope so this is the *contract*, not just your internal impl:

```rust
pub enum JitResponse {
    Synthesized { entry_point: String },   // light handle (was: { entry_point, contract })
    Declined    { reason: String },
}

pub trait Synthesizer: Send + Sync {
    fn synthesize(&self, req: &JitRequest) -> JitResponse;
    fn take_kernel(&self, entry_point: &str) -> Option<SynthArtifact>;  // NEW — the handover
}

pub struct SynthArtifact {          // Fuel-owned (so Fuel stays type-decoupled)
    pub artifact: Vec<u8>,          // compiled bytes — Fuel loads as a module
    pub kind:     ArtifactKind,     // Ptx | Cubin | Source
    pub link:     LinkEntry,        // the runtime binding row (below)
    pub contract: String,           // full FKC contract markdown (Fuel's importer parses it)
}
pub struct LinkEntry { pub entry_point: String, pub symbol: String,
                       pub structure_key: String, pub revision_hash: u64 }
```

Fuel's adopt path is exactly your `:872-877` sketch: `synthesize` → cost-gate →
`take_kernel(entry_point)` → load `artifact` as a module, resolve `link.symbol`, wrap a
`KernelRef`, import `contract` via our existing FKC importer, `adopt_runtime_fused`.

**Two conform-asks** (small, on your side):
1. **Make `take_kernel` a *trait* method** (it's inherent on `BaracudaSynthesizer` at
   `jit.rs:894`). Fuel calls it through `&dyn Synthesizer`, so it must be on the trait.
2. **Return the *envelope* `SynthArtifact`** (`fuel_kernel_seam::SynthArtifact`), not a
   Baracuda type — so Fuel depends on nothing of yours (your own Q1 invariant). Convert
   your internal artifact at the trait boundary.

**One field we dropped — confirm:** your artifact listed `recipe` alongside `contract`.
We **omitted a separate `recipe`** because the re-fuse `pattern:` rides in the FKC
`contract`, and the `decompose` is the `JitRequest.region` Fuel already holds — so recipe
is derivable. If your `recipe` carries something *not* reconstructable from
(contract.pattern + the region we sent), tell us and we'll add the field back.

## Q3 — Concurrency: **accepted — sync trait, Fuel owns the threading (G7)**

`Send + Sync` + interior-mutable is exactly what we want. Fuel drives `synthesize` on a
**background / idle-time thread** (the G7 "JIT fusion is a background re-optimization
trigger" model), never on the realize hot path, and adopts via `take_kernel` when it
lands. **No async wrapper needed** — the sync fn + your thread-safety lets Fuel own the
concurrency. We'll ask for `synthesize_async` only if that ever changes (it won't for v1).

## Q4 — Budget: **coarse `max_compile_ms` is enough for v1; no hard watchdog, no extra axes yet**

Your honest three-part answer makes the call easy:

- **Coarse is sufficient.** Because synthesis runs **off the realize path** (Q3, a
  background/idle-time thread), a runaway nvrtc compile wastes background time but **never
  stalls a realize**. So we do **not** need the hard wall-clock watchdog for v1 — the
  validated budget + typed `Declined` (bounding your optimizer effort) is the right
  granularity for a background trigger. We'll ask for the watchdog only if JIT ever moves
  onto a latency-sensitive path (not planned).
- **No extra budget axes for v1.** Our **ranker cost-gates adoption** — a synthesized
  kernel enters the binding table as one more sibling and is adopted only if it *wins*.
  So a poor-occupancy or oversized kernel simply **loses the cost-gate and is dropped**;
  Fuel doesn't need you to pre-gate on regs/smem or op-count to stay correct. Pre-gating
  would only *save wasted synthesis*, which is cheap on idle time. Keep `budget` at
  `{ max_compile_ms }`.
- If wasted synthesis ever becomes measurable, the axis we'd want first is your **(a)
  register/shared-memory budget** (occupancy) — it's the one the cost-gate can't cheaply
  predict pre-compile. We'll ask then; not now.

## Summary

| Question | Resolution |
|---|---|
| Q1 surface | Accepted — you impl our trait; Fuel type-decoupled. |
| Q2 handover | Envelope **revised** to `Synthesized{entry_point}` + `take_kernel → SynthArtifact`. Asks: make `take_kernel` a **trait** method; return the **envelope** `SynthArtifact`. Confirm whether `recipe` is derivable (we dropped it). |
| Q3 concurrency | Accepted — sync trait; Fuel drives it on a G7 background thread; no async needed. |
| Q4 budget | Coarse `max_compile_ms` only for v1 (synthesis is off the realize path; ranker cost-gates). No watchdog, no extra axes yet. |

Nothing blocks your next release — the envelope shape is frozen on
`jit-envelope-reconcile`; when it merges we publish the bump and you build against it.

— Fuel
