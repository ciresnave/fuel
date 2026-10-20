# N-gram (prompt-lookup) drafting: what a proxy measurement shows

ROADMAP item 17, row **H24**. Measured 2026-10-20 with `fuel-examples/examples/ngram_draft_probe` against `fuel_inference::ngram_draft::NgramDrafter`, on real Qwen3 tokens (`Qwen/Qwen3-0.6B` `tokenizer.json`, Apache-2.0, not committed). One run per cell, release build, this box's CPU, no other work running.

## What was and was not measured

**Measured:** how often a longest-recent-n-gram lookup over the token history proposes the right continuation, and what the lookup costs per call.

**Not measured, and not claimed:** tokens per second, quality, or the cost of a target step. There is no target model in this experiment. The "target" is ground truth: the output B is assumed to be reproduced token for token, so a draft token is accepted exactly when it equals the next token of B. Every acceptance figure below is therefore an **upper bound for the scenario**: a real model copying a file diverges more often than ground truth does. The proxy also cannot show the cost the n-gram drafter adds to a *real* step (verifying K extra positions), which is what decides whether tokens per step turns into speed.

## Scenarios

The history starts as the tokens of file A (the "prompt"). The output B is:
- `edit-N%`: A with N% of its lines deleted, replaced by, or preceded by a line of unrelated file C (seeded, deterministic). This is the code-edit shape: re-emit most of the input with local changes.
- `control`: file C alone, which shares no text with A. As the output grows, B's own earlier tokens enter the history, so a control still finds repeats *inside* B.

Each step: the drafter proposes up to `max_draft` tokens; the leading proposals equal to B are accepted; the step also emits one token of the target's own (the bonus or correction), as `verify_draft` does. `tok/step` = B's length / steps.

## Results (n-gram sizes max,min; draft = most tokens proposed per step)

Code: A = `fuel-kv-pool/src/lib.rs` (11,950 tokens), C = `fuel-inference/src/speculative.rs` (532 lines).

| scenario | (4,2,3) tok/step | (4,2,5) | (4,2,10) | (4,2,5) steps with a draft | (4,2,5) accepted/proposed |
|---|---:|---:|---:|---:|---:|
| edit-2% | 2.44 | 2.87 | 3.41 | 81% | 0.46 |
| edit-10% | 2.24 | 2.57 | 2.97 | 74% | 0.42 |
| edit-30% | 2.08 | 2.33 | 2.62 | 70% | 0.38 |
| control | 1.38 | 1.42 | 1.44 | 47% | 0.18 |

Prose: A = `docs/research/ml-library-harvest.md` (7,615 tokens), C = `docs/research/strata-ideas.md`.

| scenario | (4,2,3) tok/step | (4,2,5) | (4,2,10) | (4,2,5) steps with a draft | (4,2,5) accepted/proposed |
|---|---:|---:|---:|---:|---:|
| edit-2% | 3.45 | 4.76 | 7.34 | 89% | 0.85 |
| edit-10% | 2.72 | 3.36 | 4.31 | 66% | 0.72 |
| edit-30% | 2.70 | 3.34 | 4.26 | 66% | 0.71 |
| control | 1.17 | 1.18 | 1.19 | 31% | 0.12 |

Shorter minimum n-grams are cheaper but accept less: (2,2,5) gives 2.17-2.57 tok/step on code edits against 2.33-2.87 for (4,2,5). Full 20-row tables for both corpora are reproducible with the command below.

**Drafter cost.** 60-490 microseconds per call at a 7-12k-token history in the runs taken with the box otherwise quiet (four runs; the figures are timing, so they move with load: one run taken while other builds were running measured 0.9-2.4 ms for the same cells) (the naive scan is O(history x n); (2,2,5) is the cheapest at about 80-125 us, (4,2,x) about 125-490 us). That is small beside a multi-millisecond model step at these lengths but grows linearly with context: at 100k tokens expect milliseconds per call, which would matter. A rolling index (n-gram hash to last position, updated per emitted token) removes that; it needs a stateful drafter and is the obvious follow-up, not part of this change.

## Reading it

- The hypothesis in the roadmap row ("at least 1.3 accepted tokens per target step on repetitive workloads") is met by a wide margin as an upper bound: 2.1-3.4 tokens per step on code edits and 2.7-7.3 on prose edits. How much of it survives a real model is not measured.
- On the unrelated control the drafter still finds repeats inside B (1.17-1.44 tokens per step), so "no match, no gain" is not quite right; boilerplate repeats within one output.
- Tokens per step cannot go below 1 in this proxy because the bonus token is always emitted. The real downside of a bad draft is the extra positions the target must verify and the lookup; neither is measured here. Prompt-lookup engines usually gate drafting on measured acceptance for exactly that reason (Strata, ROADMAP item 16 S3); the stats type `SpeculativeStats` already provides a windowed acceptance rate to gate on.

## Reproduce

```text
cargo build --release -p fuel-examples --example ngram_draft_probe
./target/release/examples/ngram_draft_probe.exe <tokenizer.json> fuel-kv-pool/src/lib.rs fuel-inference/src/speculative.rs
./target/release/examples/ngram_draft_probe.exe <tokenizer.json> docs/research/ml-library-harvest.md docs/research/strata-ideas.md
```
