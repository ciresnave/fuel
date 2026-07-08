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

---

# UPDATE (2026-07-08): the stack is now complete + hardware-verified; SECOND coordination item

Since the above was written, the full Fuel-side JIT stack landed on branch `jit-integration`
(commits `a3d3a9fb…bffe5c47`) and the **on-device test passed on the RTX 4070**
(`jit_adopt_loads_and_launches_a_synthesized_cuda_kernel` — real nvrtc-compiled PTX through
`adopt_from_response` → `load_synth_kernel` → live launch → verified results). What exists:

| piece | file (all on `jit-integration`) |
|---|---|
| adopt glue (`adopt_from_response`) | `fuel-dispatch/src/jit_adopt.rs` (feature `jit`) |
| gated arm emitter (`offer_runtime_fused_arm`) | `fuel-dispatch/src/runtime_fused_arm.rs` |
| pathfinder (`emit_runtime_fused_arms` + `RuntimeFusedArmPathfinder`) | `fuel-dispatch/src/runtime_fused_pathfinder.rs` |
| live CUDA loader (`load_synth_kernel`, slot-dispatcher bank) | `fuel-dispatch/src/jit_cuda_load.rs` (features `jit,cuda`) |

## Coordination item 2 — registering the pathfinder into `default_passes`

`RuntimeFusedArmPathfinder` implements your `Pathfinder` trait and is ready to register in
`PassRegistry::default_passes()` (`fuel-dispatch/src/driver.rs:189`, next to
`PlacementForkPathfinder`). I deliberately did NOT register it yet, for one reason:

**The runtime-fused sidecar is process-global with no test isolation.** `runtime_entries()`
is a `RwLock<Vec<…>>` shared by every test in the `fuel-dispatch` test binary. The moment the
pathfinder is in `default_passes`, every test that calls `optimize_graph` will also scan the
regions other tests happened to adopt (`fused_cost`, `jit_adopt`, `runtime_fused_arm`,
`runtime_fused_kernels`, `runtime_fused_pathfinder` all adopt ops) — so an
`optimize_graph` test whose graph *contains* `relu(add(…))` would suddenly grow a branch it
didn't ask for, nondeterministically by test-registration order.

### Questions for you (answer any way you like; I'll do the work)

1. **Placement**: are you OK with `RuntimeFusedArmPathfinder` joining `default_passes()`
   directly, or do you want a separate registry-construction seam (e.g. a
   `default_passes_with_runtime_fusion()` the production bridge uses while bare
   `default_passes()` stays runtime-op-free for tests)?
2. **Test isolation**: preference between (a) a `#[cfg(test)]`-only
   `clear_runtime_fused()` reset hook on the sidecar, (b) scoping the sidecar per-`Graph`
   (heavier; changes the adopt API), or (c) the separate-constructor approach in Q1, which
   sidesteps isolation entirely? My lean: **(c) + a `#[cfg(test)]` reset hook** — smallest
   surface, no production behavior change, tests stay hermetic.
3. **Ordering**: the pathfinder mutates the graph (appends fused nodes + `Op::Branch`es).
   In `run_lockstep`, should it run **before** `PlacementForkPathfinder` (so placement sees
   the new arms) or after? My lean: before.
4. (Unchanged from above) **the `is_runtime` kernel-resolution arm** — where in your
   executor flow?

Nothing here blocks your work; when you answer, I wire it in a worktree and we reconcile at
merge.

— JIT-seam session
