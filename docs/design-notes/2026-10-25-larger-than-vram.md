# Larger-than-VRAM: what exists, what does not, and the build order (ROADMAP row 67)

Status: design note, 2026-10-25. CireSnave (board 282, 2026-10-25, relayed by the PM): "Build it now! I
thought that was already in place". It is not in place. Everything below is read from `origin/main@991b93702`
(static; nothing here was run on a GPU).

## 1. What exists

- `fuel-dispatch/src/residency.rs`: `ResidencyPlanner::analyze` (live ranges and peak bytes of a graph) and
  `insert_residency_evictions(graph, roots, budget_bytes, max_rounds, placement_of)`, which, while the peak
  exceeds the budget, inserts an `Op::Move{Cpu}` plus a reload `Op::Copy{device}` around the candidate with the
  largest `bytes x inactive_gap`. Unit tests drive it through `PipelinedExecutor` on small graphs.
- `optimize_graph` (`optimize.rs`) already runs a residency pass, `insert_residency_copies`, gated on
  `PlanOptions::pinned_device`; it stitches cross-device copies. It is a different thing from eviction.
- `ResidencyPlanner::analyze` has one production user: sizing the KV pool (`fuel-tensor/src/kv_block_pool_device.rs`).
- The ranker's `VramPressureSelector` / `ChainedSelector` read per-backend available and total VRAM when choosing a
  kernel route; that signal exists but nothing feeds it to the planner as a budget.

## 2. What does not exist

1. `insert_residency_evictions` has **no production caller**: only its re-export (`fuel-dispatch/src/lib.rs`) and
   tests in `residency.rs`.
2. **Nothing prefetches**: 0 occurrences of "prefetch" in Rust source. A reload `Op::Copy` runs where the
   consumer's ordering puts it, so the transfer is on the critical path.
3. **No budget**: `PlanOptions` has no device byte budget, and no policy turns the probed free VRAM into one.
4. **No weight streaming.** The eviction pass fits an intermediate that is live across a gap. A model whose weights
   alone exceed VRAM needs the opposite shape: the weights START on the host, and each layer's weights are copied to
   the device just before use and released after (double-buffered, so layer N+1 loads while layer N computes).
   The existing evict/reload chain cannot express that: a weight read once per decode step has one consumer and no
   gap, so `find_gap_consumers` returns `None` for it.

## 2b. What I1 found about the existing pass (2026-10-25, after writing sections 1-3)

Wiring the pass and testing it showed three defects in it, none visible while nothing called it:

1. **It measured a number it could not move.** `ResidencyPlanner::analyze` walks a plain topological order
   and counts every node's output. The staged host copy (`Op::Move{Cpu}` output) is not bytes of the device
   being planned, and the `Move` sits wherever the topological sort puts it, not where the executor runs it
   (right after the last non-destructive reader of its source). Fix: `analyze_on(graph, roots, device)`
   walks `execution_plan` and counts only nodes resident on `device`. `analyze` is unchanged (KV-pool sizing).
2. **A second run double-evicted.** The `evicted` set only covered one run; a second run put a second
   destructive `Op::Move` on the same source, and `execution_plan` panics "cycle in ordering edges".
   Fix: a candidate that already feeds a `Move` is skipped.
3. **The reload is eager, so one round does not lower the peak.** The ordering `execution_plan` uses is
   DERIVED from destructive ops (`derive_ordering`); there is no way to add a constraint of one's own. The
   reload `Op::Copy` therefore runs right after its `Move`, and on the test graph the analysed device peak is
   64 B before and after (pinned by `today_the_reload_runs_eagerly_so_the_device_peak_does_not_drop`).
   **I3 therefore needs a new primitive, not just a pass: a stored "run after" constraint on `Graph` that
   `execution_plan` merges with the derived edges** (so the reload can be held until just before its first
   post-gap consumer, or released `k` nodes earlier for prefetch).

## 3. Build order (each increment is a PR with its own measurement)

- **I1, wire the existing pass (DONE in the PR that carries this edit).** `PlanOptions::device_budget_bytes:
  Option<usize>`; in `optimize_graph`, after `insert_residency_copies`, call `insert_residency_evictions_in`
  (a `&mut Graph` form; the `SharedGraph` entry point wraps it) when a pinned device and a budget are both set,
  with `placement_of` read from the plan's resolved placements, plus the two fixes in section 2b (items 1 and 2).
  Tests: no budget, a roomy budget and no pinned device leave the graph untouched (negative controls); a budget
  below the peak inserts `Move`/`Copy` chains; `analyze_on` discounts the staged copy; a candidate is evicted at
  most once. Not claimed: any reduction of the device peak (section 2b item 3) and any GPU behaviour.
- **I2, the budget source (BUILT, `fuel-dispatch/src/device_budget.rs`; not wired by default, see the CHANGELOG).** Turn the probed total/free VRAM of the pinned device (the `BackendStreams` signal the
  selector already reads) into a default budget with a documented reserve fraction; an explicit
  `FUEL_DEVICE_BUDGET_BYTES` style override for tests and for a box with a shared GPU. The default must be
  conservative and visible in the plan report, never silent.
- **I3, ordering primitive and prefetch.** First the stored "run after" constraint on `Graph` (section 2b item 3),
  then order the reload `Op::Copy` a configurable number of nodes before its first post-gap consumer, priced
  with the probed `TransferCalibration` so the distance covers the transfer time. Measure on the 4070 (8 GB)
  with a model that does not fit; needs a GPU window and `scripts/gpu-run.ps1`.
- **I4, weight streaming.** For weights that do not fit, a planner transform that keeps them host-resident and
  inserts a per-layer prefetch `Copy` plus a release, double-buffered. This is the load-time planner's residency
  program that `residency.rs` already names as its owner. It needs I2 and I3, and the quantized weights from
  ROADMAP row 34 (native Q8_0 cut TinyLlama-1.1B's resident memory 13,750 -> 5,049 MB) to make the footprint small
  enough to be worth streaming at all.

## 4. Decisions needed from the PM (none block I1)

1. Default budget policy for I2: probed free VRAM times a reserve fraction (proposed 0.9), or an explicit opt-in
   only. Proposed: default ON for a pinned CUDA/Vulkan device, off for CPU.
2. Whether I4 may change where `Op::Const` weights are placed at load.
