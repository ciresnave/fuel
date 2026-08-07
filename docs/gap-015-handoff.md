# GAP-015 handoff — deadlock in `fuel-dispatch --features cuda` lib tests

**Status:** diagnosed, fix design ruled, **not built**. Everything below is
operational knowledge that is *not* in the `docs/gaps.md` row — the row carries
the findings; this carries how to reproduce them, and which instruments lie.

Diagnosed 2026-08-07. Instrument landed at `c9b9e991`.

---

## 1. The one-paragraph version

Readers and writers deadlock on the single `GLOBAL_BINDINGS` `RwLock`
(`fuel-dispatch/src/dispatch.rs`). std's `RwLock` blocks new readers once a
writer queues (anti-writer-starvation), so a thread holding a read guard that
requests a *second* read waits on a lock its own guard is blocking. It needs a
queued writer, therefore concurrency, therefore `-j1` is clean. It is **not** an
ABBA with `GLOBAL_REGISTRY` — that was the leading hypothesis of two independent
sessions and the trace falsified it.

The fix is ruled: `RwLock<Arc<KernelBindingTable>>`, accessor clones the `Arc`
and **drops the guard inside the accessor**, so no guard escapes and the
deadlock becomes unrepresentable. Writers clone-mutate-swap. No new dependency.

---

## 2. Reproduce it (exact recipe)

The reproducer is deterministic. Build once, then run.

**Build** — needs `vcvarsall` for nvcc; a bare shell fails with
`nvcc fatal: Cannot find compiler 'cl.exe'`:

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
cd /d <worktree>
cargo test -p fuel-dispatch --features cuda --lib --no-run
```

Invoke that `.bat` from **PowerShell**, not the Bash tool: Git Bash rewrites
`/c` → `C:/`, so `cmd /c foo.bat` silently becomes an interactive shell that
prints a banner and exits 0 having compiled nothing. That failure returns
**exit 0**, so check for positive artifacts (`Finished`, an `Executable` line),
never the exit code.

**Run** (all GPU-touching runs go through the machine-wide mutex):

```
pwsh scripts/gpu-run.ps1 -Project gap015 -- <cmd>
```

Bound the hang and kill it — do not sit on the global mutex while deadlocked.
A healthy full run is ~5-20s, so a 35-60s timeout is generous.

| `--test-threads` | result |
|---|---|
| 1, 2, 4, 8, 16 | PASS (743 passed, 17 ignored) |
| 24, 32 | **HANG**, always at exactly 343 of 760 completed |

`available_parallelism` on the box this was found on is **32**. The threshold is
between 16 and 24; it was not bisected more finely because the mechanism, not
the exact number, is what matters.

---

## 3. Instruments that LIE — read before measuring

Each of these cost a run. They are the real content of this document.

**`--nocapture` corrupts completion parsing.** Under parallelism it interleaves
test output with libtest's `test NAME ... ok` status lines. Use capture ON when
you need to parse completions; use `--nocapture` only when you need the lock
trace to reach stderr immediately (a hung test never flushes captured output).

**"Never reported" ≠ "blocked".** At the block, 401 of 760 tests had not
reported — but only ~44 *can* be blocked (that is the thread count). The rest
never **started**, because every worker was parked. `ranker::` alone contributes
170 never-started tests and they are pure-CPU logic tests with nothing to do
with the deadlock. Any per-module histogram of "not completed" is dominated by
never-started noise.

**Removal-based bisect is invalid here.** Skipping half the candidates and
asking "did it pass?" only localizes a *single* culprit. Every half-removal
still hung, which disproves the single-culprit assumption — and if you keep
walking you converge on an arbitrary last element whose individual removal
changes nothing. Use **keep-based** bisect: skip everything *except* the
candidate subset; HUNG means the culprit set is inside what you kept. Sanity-
check both ends (keep=ALL must hang, keep=NONE must pass) before trusting a
single iteration.

**Every ablation needs a control in the same run.** A passing ablation proves
nothing unless a no-skip run in the same session still hangs — the threshold is
close enough that "it passed" is otherwise unfalsifiable.

**The lock trace's blind spots** (also in `c9b9e991`'s message): it does **not**
log guard drops, so `got BINDINGS.read … want BINDINGS.read` is equally
consistent with two sequential non-overlapping acquisitions — it is *not* proof
of overlap. `want` is logged before the whole `get_or_init(...).read()`
expression, so "blocked in the initializer" and "blocked on the read lock"
print identically; they are distinguishable only via the init markers. It covers
the dispatch globals only.

---

## 4. Results already obtained — do not re-run these

Each module **alone** at `-j32` passes (topology 5.28s, optimize 1.87s,
pipelined 2.14s, ranker 0.44s, plan 2.08s, residency 2.23s, variant_bake 2.54s,
runtime_fused 2.52s). Nothing deadlocks against itself.

Ablations at `-j32`, each against a control that still hung:

| ablation | result |
|---|---|
| CONTROL (no skip) | HUNG (343 ok) |
| skip `fkc::` | HUNG (118 ok) |
| skip `dispatch::` | **PASS** 5.2s |
| skip `optimize::` | **PASS** 6.9s |
| skip `dispatch::`+`fkc::` | PASS 5.0s |
| skip `runtime_fused` | **HUNG** (342 ok) |
| skip single test `audit_multi_backend_coverage` | HUNG |
| skip single test `copy_from_cuda_wrapper` | HUNG |

Keep-based bisect over `dispatch::`'s 51 tests converged to 7, all
`global_bindings_registers_*_family_from_contract`, with the culprit **spanning**
the final split — several concurrent callers, not one test.

---

## 5. Hypotheses REFUTED (with why, so they are not re-derived)

1. **Pre-existing / load / probe fan-out** — the three recorded before this
   session. GAP-012's probe memoization is fixed and positive-controlled as *not*
   the cause.
2. **ABBA between `GLOBAL_BINDINGS` init and `GLOBAL_REGISTRY`** — held
   independently by two sessions, pinned to exact lines, and **false**: the trace
   shows zero threads inside the initializer and 20+ already past it. Init had
   completed; it is not the blocking point.
3. **`SystemTopology::build` holding a registry guard across the bindings call**
   — `topology.rs:419-427` scopes the guard and drops it before `:433`.
4. **ABBA between the two `OnceLock` *initializers*** — `default_cpu_caps()`
   does not reach `global_bindings()`. (Positive-controlled: the function exists
   at `dispatch.rs:6197`, so the empty result is a real absence.)
5. **`bump_topology_generation()` as a deadlock source** — it is a plain
   `fetch_add`.
6. **`pipelined_bridge.rs:795` guard held across `:877`** — different functions
   (`build_optimized_graph`@705 vs `dispatch_with_plan_retry`@807); the guard is
   dead by the second call. **Refutes only that PAIRING — not the site.** An
   earlier version of this document said "so that site is clean," which was
   broader than the evidence: see §6, the guard at `:795` *is* held across the
   whole optimize call at `:804`. Refuting one pairing does not clear a scope.
7. **`cast_fusion_predicate()` as the live second `.read()`** — the `Arc<dyn Fn>`
   really does re-acquire `global_bindings()` on every invocation, but it is
   **test-only**: it appears in `cast_fusion.rs` tests (`:187/200/221/251`) and
   `opt.rs`'s test region (`:4569-4737`). The production rule registry
   (`opt.rs:187-228`) wires `LoweringRule` + `FusionRule` only. Do not fix on the
   assumption that cast fusion is the culprit.
7. **Writers are `adopt_runtime_fused`/`clear_runtime_fused_for_tests`** —
   over-attribution from a doc comment. Skipping every `runtime_fused` test still
   hangs. The comment names those because they were *that test's* local problem;
   a comment is not an inventory. The blocked writers are in
   `extend_global_bindings`, the general per-backend registration path.

---

## 6. Open / unconfirmed

**The model's precondition is CONFIRMED; the exact second `.read()` is not.**

Confirmed (positive): `fuel-core/src/pipelined_bridge.rs:795` binds
`let bindings_guard = global_bindings();` and holds it across the **entire**
optimize call at `:804` —
`optimize_graph_with_runtime_fusion(&mut g, roots, &bindings_guard, &options)`.
The guard is passed by reference *precisely so the optimizer reuses it*. So a
site holding a guard across a large re-entrant region definitely exists, and the
recursive-read model has no hole.

Still open: **which callee under that guard ignores the passed `&bindings_guard`
and calls `global_bindings()` itself.** That callee is the second `.read()`.
Unchecked candidates, all reached under the held guard: `order_for`
(`pipelined.rs:2051` Streaming arm, and `:2215`), the `LoweringRule` /
`FusionRule` internals, the residency / layout passes, and any registration path
re-entered during optimize.

The reproducer plus the trace at `c9b9e991` can pin it in one run — add a
`lock_trace` marker at the suspected callee and see whether the blocked thread
passes through it.

Note the search rule: `RwLockReadGuard` has a `Drop` impl, so NLL does **not**
end it early — it lives to end of scope. "Held across" means "still in scope,"
not "still textually used."

**This does not block the fix.** Option 3 removes guard-holding entirely, so it
fixes any recursive-read site. But a *negative* sweep result would mean the model
has a hole, and that is worth knowing before a cross-crate signature change.

---

## 7. Building the fix

**Blast radius:** `global_bindings()` returns
`RwLockReadGuard<'static, KernelBindingTable>` today; returning `Arc<…>` is a
**type** change touching every caller across at least `fuel-dispatch` and
`fuel-core`. `--lib` is a **blind instrument** for this. Gate with
`--all-targets` on every consuming crate, and under `--features cuda` and
`--features vulkan` **separately** — some callers are feature-gated and a green
default build proves nothing about them.

Watch for callers that pass `&bindings_guard` by reference into helpers (e.g.
`pipelined.rs:2051` `order_for` → `lower_picked_route_streaming`); those signatures
change too.

**The real gate is the reproducer**: `-j32` must pass, and `-j1`/`-j16` must
still pass.

**Weigh, don't assume:** with snapshot semantics a reader sees a point-in-time
table, so a kernel adopted mid-optimize is invisible to that pass. This is
believed to be a *correctness improvement* — it makes a plan a pure function of
`(graph, table-version)`, where today a pass observing mid-flight mutation can
produce a plan not reproducible from its own inputs, which would break the
`base_map_hash` recipe-identity property Tier-2 convergence rests on. It has
**not** been exhaustively verified that no path depends on seeing a mid-pass
adoption. Check before relying on it.

**Second incident on this lock**, retired by the same change:
`runtime_fused_kernels.rs:415-424` records `finalize()`'s `.expect()` panicking
**while holding the write lock**, poisoning it for every later test in the
binary. With no long-lived write lock there is nothing to poison.
