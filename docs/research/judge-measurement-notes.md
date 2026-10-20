# Judge measurement notes: noise floor, measurement order, and the early-stop experiment

Written 2026-10-20 for ROADMAP item 17, row **H15** (Judge early-stop). Measured on this box (RTX 4070 Laptop 8 GB via Vulkan, AMD iGPU via Vulkan, CPU), `fuel` at `167ef3228` (0.20.20) plus an experimental patch kept on branch `judge-early-stop` (`23bcbb992`, **not for merge**), `cargo build --release -p fuel-lazy-examples --bin judge --features vulkan`, every run through `scripts/gpu-run.ps1`, one run at a time. `base3`, `base4` and every `ord*`/`var*` run had no builds or tests running beside them. `base1` ran while I was reading and editing sources and I cannot rule out a short cargo invocation overlapping it. A second early baseline, `base2`, was contaminated by my own concurrent builds and is excluded from every table.

## Result in one paragraph

**The hypothesis "early stop cuts full-baseline Judge wall time by at least 30% with the same winners" is falsified.** Early stop does cut the *timed* work by 43% against the same measurement order, but timed iterations are only about 12% of the Judge's wall time, so the wall-time effect is at most about 5% and is buried in a run-to-run wall-time spread of 35% between identical runs. The experiment also found two things worth more than the experiment: the Judge disagrees with itself on the winner in 13–16% of cells between identical runs, and **the order in which device classes are measured changes the measured Vulkan latencies by about 2x**.

## Method

- The Judge profiles 408 cells (op x dtype x size class) on 3 device classes (CPU, Vulkan NVIDIA, Vulkan AMD iGPU): 1,020 entries, 7 timed iterations and 3 warmups per entry.
- `base*` = unmodified Judge. `ord*` = Judge with `--early-stop-factor=1e9`: accelerator classes are measured before the CPU (the experimental patch's order) but no candidate is ever stopped. `var*` = `--early-stop` (stop a candidate after 3 timed runs when its fastest run exceeds 3x the best median already recorded for the cell; recorded `iterations` says how many runs it took).
- "Winner" per cell = lowest recorded latency among its entries. Agreement = cells with the same winning (backend, device, kernel) in both runs. Comparison tool: `scripts/judge_compare.py`.
- "Timed work" = sum over entries of `latency_ns x iterations`: the time spent inside timed iterations. Wall time is the whole process under `gpu-run`.
- n is small (3 base, 2 order-only, 3 early-stop). Ranges are reported instead of p-values; none of the comparisons below should be read as more than "the ranges do or do not overlap".

## Measurements

| run | kind | wall (s) | timed work (s) | timed iterations | CPU / Vulkan0 / Vulkan1 timed work (s) |
|---|---|---:|---:|---:|---|
| base1 | baseline | 820 | 100.5 | 7140 | 63 / 31 / 7 |
| base3 | baseline | 1056 | 122.3 | 7140 | 71 / 42 / 10 |
| base4 | baseline | 784 | 101.7 | 7140 | 71 / 23 / 8 |
| ord1 | order only | 679 | 85.1 | 7140 | 67 / 13 / 5 |
| ord2 | order only | 629 | 80.9 | 7140 | 63 / 13 / 5 |
| var1 | early stop | 590 | 45.9 | 6524 | 31 / 10 / 5 |
| var2 | early stop | 655 | 43.6 | 6552 | 29 / 9 / 5 |
| var3 | early stop | 658 | 53.0 | 6312 | 34 / 13 / 6 |

Winner agreement between runs (cells out of 408):

| comparison | agreement |
|---|---|
| baseline vs baseline (base1/base3/base4, three pairs) | 344, 354, 356 (84–87%) |
| order-only vs order-only | 395 (97%) |
| early-stop vs early-stop (var1/var2, var1/var3) | 393, 382 (94–96%) |
| early-stop vs its order-only counterpart (var1/ord1, var2/ord2) | 395, 391 (96–97%) |
| baseline vs order-only or early-stop (nine pairs) | 359–388 (88–95%) |

Per-entry latency drift between baseline runs: median 21–34%, p90 78–204%, max 10x–54x.

## What follows from it

1. **The Judge's own noise floor is 84–87% winner agreement between identical runs** (97% when the measurement order is reversed, see 2). Any claim that a Judge change "keeps the same winners" has to be read against that, not against 99%. The disagreements I inspected (the contaminated `base2` pair, the only one I read cell by cell) were mostly near-ties around 1 ms where launch overhead dominates and no winner is meaningful; the figure still bounds what the table can promise a consumer. This is the same lesson as Strata's run-the-engine-twice noise floor (ROADMAP item 16, S10), measured here for the Judge.
2. **Measurement order changes the measurement.** With the CPU class measured first (the current order) the two Vulkan classes' timed work was 23–42 s and 7–10 s; with accelerators first it was 13 s and 5 s in both order-only runs and 9–13 s and 5–6 s in the three early-stop runs, with CPU timed work 63–71 s in every baseline and order-only run. The cause is not established here (a laptop power/thermal budget shared with a CPU that has just run flat out is the obvious candidate; nothing in this experiment tests it). What is established: the same code measures the GPUs about 2x slower when it measures the CPU first, so a Judge table built in the current order is biased against the accelerators on this machine. Order-only runs also agree with each other far more (97%) than baselines do (84–87%), which fits the same explanation.
3. **Early stop is a small win on the wrong cost.** Against its own order (ord vs var): 83 s -> 47 s of timed work (-43%) and 7,140 -> 6,312–6,552 timed iterations (-8% to -12%: 15–20% of entries were truncated to 3 of 7 iterations; where I split them, 105–112 were Vulkan and 42 CPU per run). It added no winner disagreement beyond the order-only noise (96–97% agreement with the order-only counterpart). But wall time is 590–658 s against 629–679 s order-only: the ranges overlap. Timed iterations are 81–122 s of 784–1056 s of baseline wall (about 12%); the other 88% is outside the timed loop (3 warmups per candidate, per-cell graph builds and realizer creation, the 204 "skipping" cells).
4. **Wall time between identical runs varies 35% (784–1056 s).** A single A/B of wall time on this box cannot see an effect of 5%.

## Decision

- **The early-stop patch is not proposed for merge.** It adds public API (`Judge::early_stop`, `EarlyStop`) for a gain bounded at about 5% of wall time and truncates measurements (3 iterations) that downstream consumers read as medians. It stays on `judge-early-stop` for reproducibility.
- **Follow-ups this points at, in order of expected value:**
  1. Randomise or fix a *documented* measurement order and say which in the table's provenance (item 2 above). Cheapest to test: reverse the order in the default Judge and re-measure the existing dispatch decisions; a change in which backend wins at the large size classes would be the real result.
  2. Cut the non-timed 88%: warmups for clear losers, realizer/graph setup shared across iterations and classes. This is where the wall time is.
  3. Record the Judge's noise floor with every table (two runs, agreement and drift) the way Strata reports its self-noise floor, and give the dispatch consumer a tie band instead of a hard winner where cells are within the floor.
- ROADMAP item 17's H15 keeps its roofline-prior and "restore inputs for in-place kernels" parts unchanged; the second is settled as not a bug today (the Judge never times `*_inplace` kernels; PM was told 2026-10-20).

## Reproduce

```text
# build once per variant, copy the exe, then for each run:
pwsh scripts/gpu-run.ps1 -Project fuel -- ./judge-<variant>.exe [--early-stop | --early-stop-factor=1e9] out.json
python scripts/judge_compare.py a.json b.json
```

Branch `judge-early-stop` (`23bcbb992`) holds the patch and the extra `judge` binary flags. Raw reports are not committed (about 540 KB each).
