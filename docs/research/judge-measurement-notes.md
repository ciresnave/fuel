# Judge measurement notes: noise floor, discrete-GPU variability, and the early-stop experiment

Written 2026-10-20 for ROADMAP item 17, row **H15** (Judge early-stop). Measured on this box (RTX 4070 Laptop 8 GB via Vulkan, AMD iGPU via Vulkan, CPU), `fuel` at `167ef3228` (0.20.20) plus an experimental patch kept on branch `judge-early-stop` (`23bcbb992`, **not for merge**), `cargo build --release -p fuel-lazy-examples --bin judge --features vulkan`, every run through `scripts/gpu-run.ps1`, one run at a time. `base3`, `base4` and every `ord*`/`var*` run had no builds or tests running beside them. `base1` ran while I was reading and editing sources and I cannot rule out a short cargo invocation overlapping it. A second early baseline, `base2`, was contaminated by my own concurrent builds and is excluded from every table.

## Result in one paragraph

**The hypothesis "early stop cuts full-baseline Judge wall time by at least 30% with the same winners" is falsified.** Early stop does cut the *timed* work by 43% against the same measurement order, but timed iterations are only about 12% of the Judge's wall time, so the wall-time effect is at most about 5% and is buried in a run-to-run wall-time spread of 35% between identical runs. The experiment also found two things worth more than the experiment: the Judge disagrees with itself on the winner in 4–16% of cells between identical runs, and **the discrete GPU's readings are bimodal from run to run (a 2–3x swing in its median latency) for reasons unrelated to measurement order**.

> **Correction (2026-10-20, later the same day).** The first version of these notes (merged in #409) concluded that measuring the CPU class before the Vulkan classes makes the Vulkan classes read about 2x slower, and that today's tables are order-biased. **That conclusion was wrong.** It rested on baseline runs that happened to land in the discrete GPU's slow mode and order-only/early-stop runs that happened to land in its fast mode. Seven further runs in interleaved orders (section "Measurement order: the follow-up") show the same bimodality inside every order, and no order effect on winners beyond run-to-run noise. The passages below were rewritten; the claim is withdrawn, not caveated.

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

1. **The Judge's own noise floor is 84–96% winner agreement between identical runs** (84–87% across the first three clean baselines, 90–96% across the seven order-follow-up runs). Any claim that a Judge change "keeps the same winners" has to be read against that, not against 99%. The disagreements I inspected (the contaminated `base2` pair, the only one I read cell by cell) were mostly near-ties around 1 ms where launch overhead dominates and no winner is meaningful; the figure still bounds what the table can promise a consumer. This is the same lesson as Strata's run-the-engine-twice noise floor (ROADMAP item 16, S10), measured here for the Judge.
2. **The discrete GPU's Judge readings are bimodal from run to run, independent of measurement order.** Its class (Vulkan0, the RTX 4070) shows a median per-entry latency of about 4 ms in some runs and 9–12 ms in others, and timed work of 13–15 s versus 22–42 s (untruncated runs), while the CPU (58–71 s in the untruncated runs) and the iGPU (5–10 s) stay put. Across the 15 runs in this document the slow mode occurred in order `sorted` (base1/3/4, o-sorted1), `reversed` (o-rev2) and `random` (o-rand1) alike, as did the fast mode. Whatever flips it (the card's power state, another process on the GPU, thermal state) is not controlled by the Judge. A table built in a slow-mode run says the discrete GPU is 2–3x slower than one built in a fast-mode run. On this box it does not change what wins, because the discrete GPU almost never wins a cell (0–4 of 408) in either mode, probably because the Judge's cells include a host round trip the iGPU does not pay (not tested here).
3. **Early stop is a small win on the wrong cost.** Against its own order (ord vs var): 83 s -> 47 s of timed work (-43%) and 7,140 -> 6,312–6,552 timed iterations (-8% to -12%: 15–20% of entries were truncated to 3 of 7 iterations; where I split them, 105–112 were Vulkan and 42 CPU per run). It added no winner disagreement beyond the order-only noise (96–97% agreement with the order-only counterpart). But wall time is 590–658 s against 629–679 s order-only: the ranges overlap. Timed iterations are 81–122 s of 784–1056 s of baseline wall (about 12%); the other 88% is outside the timed loop (3 warmups per candidate, per-cell graph builds and realizer creation, the 204 "skipping" cells).
4. **Wall time between identical runs varies 35% (784–1056 s).** A single A/B of wall time on this box cannot see an effect of 5%.

## Measurement order: the follow-up

PM asked for the order question to be settled before anything else (fixed, documented order plus a re-measure of which backend wins at large sizes). Seven runs of the same binary (`judge-order-exp`, branch `judge-order-exp` at `b28f05017`, experiment hooks `FUEL_JUDGE_ORDER` and `FUEL_JUDGE_SETTLE_MS`, **not for merge**), interleaved so a drifting machine state cannot line up with one order: `sorted` (the current CPU, Vulkan0, Vulkan1 order) x2, `reversed` x2, `random:<seed>` per cell x2 (seeds 1 and 2), and `sorted` with a 300 ms pause before every class measurement.

| run | order | wall (s) | timed work CPU / Vulkan0 / Vulkan1 (s) | median entry latency CPU / Vulkan0 / Vulkan1 (us) |
|---|---|---:|---|---|
| o-sorted1 | sorted | 768 | 63 / 27 / 5 | 1334 / 11721 / 1224 |
| o-rand1 | random:1 | 640 | 60 / 22 / 5 | 1071 / 9403 / 1105 |
| o-rev1 | reversed | 665 | 63 / 14 / 5 | 1040 / 4213 / 1176 |
| o-rand2 | random:2 | 632 | 62 / 15 / 6 | 1068 / 3915 / 1147 |
| o-sortedS | sorted + 300 ms pause | 1296 | 58 / 13 / 5 | 992 / 4085 / 1140 |
| o-sorted2 | sorted | 786 | 61 / 13 / 5 | 1047 / 4032 / 1166 |
| o-rev2 | reversed | 794 | 64 / 25 / 9 | 1213 / 9062 / 1207 |

- **No order effect.** Vulkan0 is in its slow mode (median 9–12 ms) in `sorted`, `reversed` and `random` runs, and in its fast mode (about 4 ms) in all three as well. The earlier 2x "order effect" was this bimodality landing unevenly across the earlier runs.
- **Winner agreement does not depend on order.** Across all 21 run pairs agreement is 90–96%: `sorted` vs `sorted` 92%, `reversed` vs `reversed` 92%, `random` vs `random` 95%, `sorted` vs `reversed` 90–96%. Order does not stand out from run-to-run noise.
- **Large sizes are stable.** For the 27 cells at or above 2^18 elements the winners are CPU 8–10 and the iGPU 17–19 in all seven runs (and in the eight earlier runs). Whole-table counts move by about 5% (CPU 281–307 wins, iGPU 101–127), all in near-tie cells.
- **The pause buys nothing visible and costs 65% wall time.** `sorted` + 300 ms: 1296 s against 768–794 s, with timed work in the fast mode like several unpaused runs. So there is no evidence for carry-over between classes that a pause would cure.
- Not tested: whether the slow mode is the card's power state, contention from another process on the GPU, or thermal. All 15 runs went through `gpu-run`, which serialises fuel sessions only; other programs and other projects' lanes can use the card.

## Decision

- **The early-stop patch is not proposed for merge.** It adds public API (`Judge::early_stop`, `EarlyStop`) for a gain bounded at about 5% of wall time and truncates measurements (3 iterations) that downstream consumers read as medians. It stays on `judge-early-stop` for reproducibility.
- **Follow-ups this points at, in order of expected value:**
  1. Make the Judge robust to the discrete GPU's modes (item 2 above): run a short fixed reference kernel on each class at the start and end of a profile and record the pair in the table's provenance, so a table built in a slow mode is recognisable; or profile twice and keep the per-entry minimum. This is the open problem the follow-up found; the order question it replaced is closed (no order effect, below).
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
