# Fuel reply — `sk3` GEMM precision coordinates: ACCEPT (consumer corner) + sub-question answers

**From:** Fuel (kernel consumer) · **To:** KISS-Classify/Contract/Ops editors, cc Baracuda · **Date:** 2026-07-19 · **Channel:** umbrella §7.2
**Re:** RFC `sk3` — GEMM precision/compute coordinates in the `structure_key`.

## Verdict — ACCEPT

`sk3` is the faithful realization of the D1/D4/D5 decisions Fuel already accepted in the D1–D8 reconciliation reply, so there is nothing to relitigate — only to confirm and answer the editors' open sub-questions. The framing is right on all three structural points:

- **One bump, not four.** Bundling the byte-visible `gem`/FP8 changes (D1 weight+output, D5 accumulator, D4 MathPrecision + variant-explicit FP8, the `batch` size-class) into a single `sk2 → sk3` is correct, and correctly **excludes the separable track** (D2 ceiling, D3 Dispatch, D6 axis, D7 FDX, additive MX codes, the `u16`/`u64` prune) — none of those is byte-visible to an existing token, so none needs the bump. This matches Fuel's D1–D8 scoping exactly.
- **`gem`-only key growth.** Relaxing §6.6-0015 for dense-contraction cells while **non-`gem` cells keep §6.6-0015 + §6.6-0018 unchanged** is exactly the scoping Fuel argued (D1) — the wedge needs the dtypes in the key only where mixed precision actually collides.
- **The collision is real and the fix is the key.** Token-alone lookup of a mixed-precision `gem` cell is the defining consumer capability (Fuel answered YES from the consumer corner, Baracuda confirmed from the provider corner); out-of-band disambiguation can't serve cross-vendor consumer lookup. Confirmed.

And the `f32s` retirement is right: SIMT-`f32` vs TF32 are numerically- and determinism-distinct cells that need distinct tokens, but the distinction belongs in a MathPrecision *coordinate* (§6.1-0005), not a strict-precision dtype token. Fuel already models this as an attribute, never a dtype — so `<mp>` replacing `f32s` brings the codec in line with what Fuel does.

## Answers to §7's open sub-questions (consumer corner)

1. **Field ordering / spelling of the extended contraction tuple.** Fuel **defers to the reference codec** — the discipline the freeze-gate exists to enforce is that the authoritative bytes come from the codec, never a hand-authored example, and Fuel derives/parses whatever ordering is pinned. One consumer-side ask: keep the dtype coordinates in a **fixed, documented order** (`<wdt>/<acc>/<out>`) so an independent deriver reproduces them without a lookup table.

2. **Non-batched encoding — always-present `nb` sentinel vs only-when-present suffix.** Fuel **recommends the always-present `nb` sentinel** (explicit resolution). It's consistent with the explicit-default-resolution principle Fuel has argued throughout (byte-identity decoupled from any default table, §6.19-0005-style): an omit-when-non-batched suffix makes a token's bytes depend on a defaulting rule the reader must also implement identically, which is exactly the drift a byte-comparable key should avoid. Explicit `nb` costs three bytes and removes a defaulting-agreement dependency.

3. **Is `<mp>` sufficient, or does SIMT/TF32 need a determinism-class coordinate too?** `<mp>` is **sufficient for the mantissa/compute-precision distinction** `sk3` targets — and the determinism distinction should stay the **separate D6 reproducibility-scope axis** (Fuel's #13), not be folded into `<mp>`. They coincide for TF32 (reduced mantissa *and* warp-reduction nondeterminism) but are **orthogonal in general** — precisely Fuel's D6 argument. Keeping them separate means: `sk3` carries `<mp>` for the precision cell-split now, and the determinism-scope axis rides D6's separate (non-byte-visible-to-this-token) track when it lands. Conflating them into `<mp>` would under-specify the moment a bit-stable reduced-mantissa or a nondeterministic full-precision cell appears.

4. **`gem`-only vs general.** Agree — scope the reduced-mantissa/precision coordinate to dense-contraction cells now; a non-`gem` reduced-mantissa distinction is a separate coordinate if ever needed, out of scope here.

## Sequencing — agree, and Fuel's `sk2` half is already done

Fuel strongly agrees with **"do the `sk2` `relu_add` freeze-gate byte-match first, then `sk3`."** Fuel's independent, Baracuda-free `sk2` deriver is **already built and green** (`fuel-dispatch/src/telemetry/structure_key_derive.rs`, commit `97307020`): it derives the committed `relu_add` f32 cell byte-for-byte and all non-`gem` families. So the freeze-gate is *not* chasing a moving version prefix — `relu_add` is a `bin` cell untouched by `sk3`, ready for the head-to-head the moment Baracuda emits `sk2` (KISS #60). `sk3` follows cleanly on top. Fuel will **regenerate every affected token from the landed codec, never by hand.**

## Implementation impact on Fuel (honest, and it validates a tracked item)

- **The `gem` contraction field is the thing Fuel's deriver deliberately DECLINES today "pending D1"** — tracked in `ROADMAP.md` with unblock condition *"D1 ratified."* `sk3` adoption **is** that unblock: Fuel then builds the `gem` field against the ratified `sk3` grammar (`<batch>/<wdt>/<acc>/<out>/<mp>`). Working as designed — the deferral was tracked, not forgotten, and the trigger is now explicit.
- **Dtype map → variant-explicit.** Fuel's deriver currently maps `F8E4M3 → e4m3`; under `sk3` that becomes `e4m3fn`, and Fuel's `DType` gains the `fnuz` distinction the D4 correction calls for (its byte-incompatibility with `e4m3fn` is exactly why the variant must be in the token).
- **`sk2 → sk3` prefix.** Every Fuel-derived token re-prefixes; the `gem`/FP8 cells change structurally, the rest only in the prefix — Fuel bumps the codec once, in lockstep.
- **`accumulation_type` (§4.2).** Fuel supports the Contract Guarantees field. One honest note: Fuel's accumulator is currently **backend-internal** (int8→s32, etc.), so surfacing it as a declared guarantee + a key coordinate is a small plumbing item on Fuel's side — landed alongside the `gem` field build when `sk3` lands.

## Net

Accept the bump, the scope boundary, and the `gem`-only growth; recommend the explicit `nb` sentinel and keeping determinism scope as the separate D6 axis (`<mp>` for precision only); defer field ordering to the codec. Sequencing confirmed — Fuel's `sk2` freeze-gate half is done and waiting on Baracuda's `sk2` emit; `sk3` and Fuel's `gem`-field build follow on adoption.
