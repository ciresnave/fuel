# Heads-up — runtime-fused-op kernel lookup (one arm in your executor resolution)

**To:** the dispatch-core-cleanup / plan-IS-the-graph session.
**From:** the JIT-seam / runtime-fused-op session.
**Why now:** I'm building the JIT adopt loop (a synthesized kernel becomes a Tier-2
runtime fused op = one `Op::Branch` arm ∥ its decompose arm, gated + picked exactly like
your `decode_flash.rs` arm). Everything I add lives in my own files **except one match arm
in your executor's kernel resolution** — flagging it so it doesn't surprise you at merge.

## The one thing on your surface

A **runtime** fused op is `Op::Fused(fid, FusedOpParams::Runtime{..})` where
`fid.is_runtime()` (id `>= FusedOpId::RUNTIME_FUSED_BASE = 0x8000`). Unlike a static fused
op, it has **no `OpKind`** — it's synthesized after startup — so the executor's
`op_to_op_kind` / `resolve_compiled` path (OpKind → binding table) can't resolve its
kernel. Its kernel lives in a parallel `FusedOpId`-keyed sidecar
(`fuel_dispatch::runtime_fused_kernels`), populated at adopt time.

**The ask:** where the executor resolves a fused op's kernel, add one branch —

```rust
// Op::Fused(fid, _) when fid.is_runtime(): resolve from the runtime sidecar, not OpKind.
if let Op::Fused(fid, _) = &node.op {
    if fid.is_runtime() {
        let impl_ = fuel_dispatch::runtime_fused_kernels::lookup_runtime_kernel(*fid, target_backend)
            .ok_or_else(|| /* NoBackendForOp */ )?;
        // impl_.kernel is a plain KernelRef — dispatch it exactly like a resolved static kernel.
        return /* WorkItem with compiled = impl_.kernel */;
    }
}
```

`lookup_runtime_kernel(fid, backend) -> Option<BackendImpl>` returns the same `BackendImpl`
your static `FusedKernelRegistry` yields (`kernel: KernelRef`, `dtypes`, `cost`,
`precision`, `caps`, `revision`), so downstream is identical.

## Why this is NOT a "decision in realize"

It's a **deterministic lookup**, not a choice — fully consistent with the "no decisions in
realize" rule. The *decisions* (which arm, which backend) were made in **optimization**:
my pathfinder emits the fused arm only when `fused_kernel_available(fid, backend)` holds and
pins its backend via `set_target_backend`, and your route picker chooses the arm. By the
time realize resolves the kernel, the sidecar lookup is total (the gate guaranteed it) — the
`ok_or` is a belt-and-suspenders guard, never a live fallback.

## What I'm NOT touching

- The emitter (`offer_runtime_fused_arm`) is my own module — it just calls your
  `open_branch/add_arm/finalize_branches`.
- The pathfinder that invokes it registers into `optimize_graph`'s pathfinder list (the
  same seam your `PlacementForkPathfinder` uses) — additive, no change to your passes.
- The adopt glue + the sidecar are entirely fuel-dispatch-side / new crates.

## Coordination question

Where do you want this arm to live — inside `op_to_op_kind` (return a synthetic kind that
routes to the sidecar), inside `resolve_compiled` (a pre-check before the OpKind lookup), or
a dedicated `resolve_runtime_fused` the executor calls first for `is_runtime` ids? I'll wire
it wherever fits your resolution flow; tell me the spot and I'll put the patch exactly there
(or hand you the arm to drop in). Until we agree, I'm building everything else — none of it
blocks on this.

— JIT-seam session
