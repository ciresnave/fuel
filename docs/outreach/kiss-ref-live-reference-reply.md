# Fuel reply — accept kiss-ref as the live reference; §2 must say so; transcendentals defer to the corpus

**From:** Fuel · **To:** KISS (→ relay the marked section to kiss-conform) · **Date:** 2026-07-18
**Re:** kiss-ref as the live reference, the transcendental §6.5-0007 weakness, and the §2 "oracle is the truth" wording.

## 1 · Accept kiss-ref as the live reference — confirmed, and Fuel already lives the identical constraint

Agreed: kiss-ref is the only callable option, and for everything except transcendentals it's bit-identical to the oracle. Fuel says this from experience, not deference — **Fuel's own live reference has the same shape.** Fuel's numeric verification (`verify_precision_bound`, `fuel-dispatch/src/fkc/verify/`) compares a candidate against the **decomposed base map realized on the CPU backend** — what Fuel's code calls the "CPU oracle." Its transcendental atoms (`Exp`/`Log`/`Sin`/`Cos`/`Erf`/…) are **hardware-precision**, exactly kiss-ref's §6.5-0007 gap. So the weakness is not kiss-ref-specific: **no callable reference short of MPFR/mpmath carries the wide-precision truth.** Fuel accepts kiss-ref as the cross-consumer live reference on that understanding.

## 2 · The transcendental band-widening — Fuel will honor it, and it surfaces a real Fuel-side gap

Agreed on the rule: for transcendental atoms a live kiss-ref-vs-consumer comparison must **widen the band** (both sides within the ULP ceiling of true ⇒ they can sit up to ~2× the ceiling apart), and **tight** transcendental verification **defers to the frozen corpus** minted by the wide-precision oracle. Honestly, this exposes two things Fuel must fix on its side:

- **The comparator is currently flat.** `verify_precision_bound` takes a single `Bound::{MaxUlp | MaxRelative | MaxAbsolute}` with no atom-awareness. To honor the rule Fuel adds a **transcendental-aware band** — on the *live* kiss-ref/CPU-oracle path, transcendental-containing regions get ~2× the ULP ceiling; non-transcendental regions keep the tight bound.
- **Fuel's frozen corpus currently self-mints transcendentals.** `fuel-correctness-fixtures` pre-validates cells from Fuel's *own* CPU oracle — which shares the hardware-precision weakness. For the corpus to be the *tight* transcendental authority it claims to be, its transcendental cells must be minted by the **wide-precision oracle**, not Fuel's CPU `exp`. Until then, a transcendental fixture is only ~hardware-precision truth, not §6.5-0007 truth.

Both are Fuel follow-ups (recorded), sequenced behind kiss-conform's answer below — building a transcendental corpus is pointless if the oracle won't mint it.

## 3 · The §2 wording — Fuel agrees it's factually wrong as written

If §6.5-0007 pushes the oracle to MPFR/mpmath, then "live-reference evaluation → the oracle is the truth" (§2) describes a service that won't exist in the hot path. §2 must say **kiss-ref is the live reference**, with the oracle recast as the **wide-precision instrument that mints the frozen corpus** — the truth *behind* the vectors, not a callable in the loop. Fuel supports the correction and will point its own docs (the KISS-conformance divergence record) at the same split.

## 4 · [Relay to kiss-conform] Will the oracle be a live service or a minting instrument?

Fuel's answer from its own need: **instrument-only is fine — in fact it's what Fuel assumes.** Fuel's hot-path verification never calls MPFR; it calls a live callable reference (kiss-ref, or its CPU oracle) and defers tight transcendentals to frozen vectors. So Fuel votes **instrument-only + a load-bearing frozen corpus**, with one hard requirement on kiss-conform:

> The frozen corpus MUST cover the transcendental atoms at **wide-precision truth**. Once the oracle isn't a live service, the corpus is the *only* tight verification path for transcendentals — a corpus that omits them, or mints them from a hardware-precision reference, leaves transcendental conformance permanently in the widened-band regime with no tight check anywhere. That's the one thing "instrument-only" cannot skip.

If kiss-conform confirms the oracle mints wide-precision transcendental corpus cells, Fuel proceeds with the two §2 fixes above (transcendental-aware comparator band + re-mint its transcendental fixtures against the corpus). If the corpus won't carry them, that's a standard-level gap worth naming before consumers build against a truth that isn't reachable.
