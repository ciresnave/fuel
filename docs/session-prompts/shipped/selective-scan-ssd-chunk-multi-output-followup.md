# Session prompt — SelectiveScan + SsdChunkScan `last_state` outputs (on top of multi-output infra)

## What this session is for

Add the `last_state` second output to `SelectiveScan` (Mamba-1 SSM
forward) and `SsdChunkScan` (Mamba-2 SSD forward), enabling
autoregressive resumption across calls. Both ops shipped 2026-05-30 in
single-output form (y-only); this session lights up their multi-output
form so a generator loop can pass `last_state` from step N into step
N+1 without re-running the full prefix scan.

This is the consumer-side session for the multi-output Option C
infrastructure — it should run **after**
`multi-output-nodes-option-c.md` lands.

## Background — why this was abandoned the first time

The original 2026-05-30 attempt added `return_state: bool` to
`FusedOpParams::SelectiveScan` / `FusedOpParams::SsdChunkScan` and
matching kernel signatures, then got mid-stream **reverted by the
parallel `fuel-dispatch` extraction session** (commit `206e9bbf`
extracted `fuel-storage/src/dispatch.rs` → `fuel-dispatch` and
clobbered the in-flight params changes). At that point we pivoted to
the NF4 file loader (isolated work in fuel-core/src/nf4.rs) and
documented the abandonment in
`project_cpu_opkind_followups_shipped.md`.

The redo path is different now: instead of bolting a `bool` flag onto
the params and forcing every kernel to handle two output shapes, this
session uses the multi-output infrastructure (`Op::View` /
`Op::ViewOwned` / bundled storage) to express the two outputs
naturally. No `return_state` flag — the op always produces both
outputs; consumers View whichever slots they want.

## Preconditions — verify before starting

1. **Multi-output infra is shipped.** `Op::View`, `Op::ViewOwned`,
   `Storage` bundle side-table, `FusedOpEntry::output_views`, and
   the planner View-vs-ViewOwned pass all exist and have tests. If
   any of those is missing, stop and finish that work first.
2. **Mamba eager-to-lazy migration state.** Check the active state of
   `docs/session-prompts/eager-tensor-retirement-master-plan.md` and
   `docs/session-prompts/shipped/mamba-eager-to-lazy-migration.md`. If Mamba
   has migrated to LazyTensor, this session can both add the
   capability AND wire it into the consumer. If Mamba is still on
   eager `fuel-core::Tensor`, this session ships the capability with
   the consumer migration deferred (same pattern as the original
   shipment).
3. **Parallel dispatch / picker work has settled.** Check
   `git log --since="3 days" fuel-dispatch/` for recent rewrites of
   `FusedOpParams` / `OpParams` / the kernel binding-table. If there's
   active churn, coordinate before editing those files.

## Scope

### Multi-output op authoring for SelectiveScan

In `fuel-graph/src/registry/selective_scan.rs`:

- Implement `output_views()` returning:
  - Slot 0: `y` — dtype = input dtype (F32/F64/BF16/F16), shape
    = `(batch, seqlen, d_inner)`.
  - Slot 1: `last_state` — dtype = F32 (always, regardless of input
    dtype — matches the F64-accumulator-narrow-to-F32 contract
    documented in the v1 shipment), shape = `(batch, d_inner,
    d_state)`.
- Update `decompose()` / shape-validation accordingly.
- Backward: y's backward path is the existing forward-mode backward;
  last_state's backward path is "scatter scalar zero" (last_state is
  not differentiable in the standard Mamba training setup — verify
  against the reference impl). If last_state IS used in a training
  loop (e.g., truncated BPTT across chunks), the backward needs real
  treatment; flag that case during implementation.

In `fuel-cpu-backend/src/byte_kernels.rs`:

- Extend the existing `selective_scan_kernel!` macro to write BOTH
  outputs into the bundled storage. The kernel already computes
  `last_state` internally as the final value of the SSM hidden-state
  accumulator — just need to add the `as_slice_mut::<f32>()[state_offset..]
  = h.to_f32()` write at the end.
- Per-dtype dispatch wrappers in `fuel-dispatch/src/dispatch.rs`
  receive the bundled output Storage + the bundle metadata so they
  know where each slot lives. Wrapper signature change is the
  multi-output authoring contract from infra session.

In `fuel-core/src/lazy.rs`:

- `LazyTensor::selective_scan(...)` returns a `(y: LazyTensor,
  last_state: LazyTensor)` tuple where each LazyTensor wraps an
  `Op::View` node pointing at the bundled output. Internal helper:
  `LazyTensor::selective_scan_bundle(...) -> LazyTensor` returns the
  raw bundle for callers who want manual View control.

### Multi-output op authoring for SsdChunkScan

Mirror the SelectiveScan changes in
`fuel-graph/src/registry/ssd_chunk_scan.rs` and the matching kernel
file. Multi-chunk lift (commit `93d9f3f9`) is already in place, so the
last-state output is the per-chunk final state aggregated across
chunks — the existing kernel already tracks this internally.

Slot 0: `y` — `(batch, seqlen, n_heads, head_dim)`, input dtype.
Slot 1: `last_state` — `(batch, n_heads, head_dim, d_state)`, F32.

### CUDA + Vulkan + baracuda — explicitly deferred

Same rule as the 2026-05-30 shipment: CPU multi-output for both ops
this session. CUDA/Vulkan multi-output added in a follow-up session
when:

- The parallel baracuda-CUDA-dispatch work that motivated the original
  parallel session is settled.
- Or the baracuda alpha bump exposes selective_scan + ssd_chunk_scan
  kernels with bundled-output signatures.

Whichever comes first.

### Tests

- 3 unit tests per op: bundled-output shape verification, y-output
  numerical match against the existing single-output kernel (when
  decoded via `Op::View`), last_state numerical match against a
  reference Python impl (or hand-computed for tiny cases).
- 1 integration test per op covering "step N produces last_state,
  step N+1 consumes it" — proves the resumption pattern works.
- 1 planner regression test: a bundled-output node whose y is
  consumed-then-dropped and last_state is retained across a barrier
  should result in `Op::View` for y + `Op::ViewOwned` for
  last_state (validate the planner is making the expected choice).

## After this op pair — remaining CPU OpKind work queue

The remaining CPU OpKind work is split by whether it depends on the
multi-output infra. **Items in §A can ship before multi-output infra
lands** — they're standalone single-output work and could be picked
up in a separate session right now, independent of this one. **Items
in §B share this session's precondition** — they need the infra
first, like the headline SelectiveScan/SsdChunkScan rewire above.

Pick the top item from whichever section the current infra state
allows. If §B is blocked, §A items still ship cleanly.

### §A. Independent of multi-output infra — pickable anytime

1. **CausalConv1d backward.** Deferred at original shipment because
   Mamba inference was the only consumer. If the Mamba eager-to-lazy
   migration session is bringing training online, the backward is
   suddenly needed. Pattern: mirror the Conv2D backward decomposition
   already in fuel-graph; CausalConv1d's backward is the same shape
   with a 1-D kernel and a left-padding mask. Single output — no
   multi-output dependency.
2. **F16/BF16/F64 expansion for FSCE.** Mamba trio got the dtype
   expansion 2026-05-30 evening; FSCE shipped F32-only and the same
   macro pattern (`fused_softmax_cross_entropy_native_kernel!` +
   `fused_softmax_cross_entropy_half_kernel!`) applies. Same
   single-output shape; just more dtypes.
3. **NF4 file loader v2.** Real-checkpoint validation against an
   actual bitsandbytes-quantized model file. The 2026-05-30
   `f64dd0dcc` shipment implemented to documented format without real
   fixtures; this session would download a representative bnb-quant
   checkpoint (Llama-2-7b-bnb-4bit or similar), validate the loader
   round-trips correctly, and patch any format mismatches that
   surface. Pure fuel-core/src/nf4.rs work.
4. **CausalConv1d / SelectiveScan / SsdChunkScan baracuda CUDA paths.**
   Contingent on baracuda exposing the relevant kernels. Verify
   alpha.60+ symbol surface; if missing, file the ask with baracuda
   team and defer. Adds dispatch for the existing y-only shape — no
   multi-output dependency. (If multi-output infra has landed by
   the time this fires, prefer waiting for the bundled-output baracuda
   surface so you don't ship single-output CUDA that needs immediate
   rework. Otherwise ship single-output CUDA now and bundled-output
   later.)
5. **Mamba-1 / Mamba-2 backward fused ops.** Real training need.
   Defer until the eager-to-lazy migration session signals a training
   workload is incoming. Single-output gradient computation — no
   multi-output dependency.
6. **`Op::Conv1d` as a first-class IR op (not a panic-decompose).**
   CausalConv1d currently panic-decomposes because fuel-graph has no
   `Op::Conv1D` — only `Op::Conv2D`. Adding Conv1D as IR would
   unblock several other 1-D conv-like ops cleanly. Substantial
   change; only worth it if multiple consumers materialize.
   Independent of multi-output.

### §B. Requires multi-output infra (gated by this session's preconditions)

1. **FSCE multi-output bundling.** FSCE currently returns loss only;
   `FusedOpId(17)` could be lifted to bundle loss + grad_logits in one
   forward pass (eliminates the redundant backward kernel launch
   pytorch-style). Same multi-output pattern as the headline
   SelectiveScan rewire; smaller slot 1 (grad_logits shape == logits
   shape == big). **Hard dependency:** multi-output infra shipped.
   Natural follow-on once §B headline work is in hand.

## What's NOT in scope

- **CUDA / Vulkan / baracuda multi-output dispatch** — see above; CPU
  this session, follow-up for accelerated backends.
- **Mamba-2 backward** — explicit non-goal; the inference path is
  what's being unlocked.
- **Generator-loop scaffolding in fuel-core** — the integration test
  proves the pattern works; productionizing a `MambaGenerator` struct
  belongs to a model-implementer session, not this one.
- **Cross-op fusion (SelectiveScan + LayerNorm or similar)** — the
  multi-output infrastructure enables it, but actual fusion rules are
  a separate session.

## Scope estimate

- SelectiveScan multi-output: ~2 hours (kernel write + wrapper +
  registry).
- SsdChunkScan multi-output: ~2 hours (mirror of SelectiveScan).
- Tests: ~2 hours.
- One follow-up CPU OpKind item from the queue above: 1-4 hours
  depending on which one.
- Total: 1 focused session if just the Mamba pair, 1.5 if folding in
  one queue item.

## Coordination

- **Multi-output infra session.** Hard dependency. Don't start until
  it lands.
- **Mamba eager-to-lazy migration.** Loop in — they're the natural
  consumer. If their session is shipping the multi-output capability
  themselves via the new infra, this work might already be done by
  the time you check.
- **baracuda alpha bumps.** If a recent alpha exposed
  selective_scan/ssd_chunk_scan kernels, CUDA multi-output becomes
  feasible — extend scope.

## References

- Memory: `project_cpu_opkind_followups_shipped.md` — full context on
  the original abandoned attempt, including the parallel-session
  conflict.
- Memory: `project_selective_scan_shipped.md`,
  `project_ssd_chunk_scan_shipped.md` — original shipment details,
  test patterns, kernel templates.
- Memory: `project_cpu_opkind_coverage_plan.md` — broader OpKind
  trajectory the queue items above pull from.
- Session prompt: `multi-output-nodes-option-c.md` — the
  infrastructure this session depends on.
- Session prompt: `mamba-eager-to-lazy-migration.md` — the consumer
  side; coordinate.
- Commits to study before starting:
  - `1e63ebd4` — SelectiveScan v1 shipment.
  - `e4ef8e10` — SsdChunkScan v1 shipment.
  - `93d9f3f9` — SsdChunkScan multi-chunk lift (touches the same
    kernel file).
  - `a1c9466b` + `1610e6d1` — SelectiveScan/SsdChunkScan dtype
    expansion (the macro pattern to extend for bundled outputs).
  - `206e9bbf` — the dispatch extraction that reverted the original
    attempt (study the change to avoid re-tripping the same conflict).
