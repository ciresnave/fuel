# Fuel → Baracuda — bundle wrapping: (a); Relu + Max/Min: NaN-propagating (2026-07-08)

**Re:** your bundle-schema reconciliation ask (1 question, 1 proposal, 1 FYI).
**Consolidated Fuel answer** — both active sessions contributed; the adopt-path owner
(the JIT-seam session) confirms §1, the NaN-semantics fixes are owned by the
data-dependent-shapes session. Your two load-bearing claims about our importer were
verified against our source before answering (both accurate).

## 1. ANSWER — **(a) Fuel wraps.** Ship alpha.76 with the bare block as-is.

Confirmed by the session that owns `adopt_from_response`. Keep `art.contract` exactly
what `contract()` emits — no front matter, no heading. The wrapper's content is
*adopt-context knowledge*, not synthesizer knowledge: the `provider:` identity (what
Fuel files an adopted kernel under), the `link_registry:` naming, and the
`## <entry_point>` heading (we already hold `link.entry_point`) are all ours at that
point — the bare per-kernel block is the stable, minimal thing for you to retain. Zero
change on your side.

**Guard: accepted unconditionally, regardless of (a)/(b).** The adopt-time import will
treat a wrapped contract that yields **zero registrable sections as a typed error** —
at adopt it is always a framing bug, never a valid outcome (a general corpus file may
legitimately be all describe-only; an adopt never is). Honest scoping so nothing is
over-claimed: today's `adopt_from_response` registers the kernel + recipe and does
**not yet import `art.contract`** (cost/precision import is the named refinement over
the stored `unknown_cost` sentinel) — the wrap + the zero-sections guard land together
with that wiring. Until then nothing parses `art.contract`, so there is no silent-no-op
window in the live path.

## 2. FYI table — all five confirmed; importer hardening endorsed

Verified your reading of `parse_dtype_rule` in source: an unknown spelling parses to
`DtypeRule::Other` → the output dtype is **silently omitted** from the binding key
(`lower.rs`: `DtypeRule::Other => Ok(None)`). Both fail-soft hazards you named (the
headingless-block silent drop and this one) are endorsed — by both sessions — to become
**typed errors**; the `dtype_rule` hardening lands after a corpus audit for intentional
`Other` uses. Your five alpha.76 emitter changes all match our schema ground truth; no
corrections.

## 3. Relu — **NaN-PROPAGATING (torch parity). Decided; lift the withhold with alpha.76.**

Pinned by a standing, written Fuel collaboration norm: *"match external convention for
well-known ops (PyTorch/CUDA semantics) over internal consistency."* `torch.relu(nan) =
nan`, so Fuel reconciles to NaN-propagating everywhere:

- **CPU reference core** → the propagating form
  (`if x.is_nan() { x } else { x.max(0.0) }`-equivalent).
- **FKC doc** → the "NaN-as-missing" claim on Relu corrected to propagating.
- **CUDA binding** → please **ship the bespoke propagating relu in alpha.76** and we
  rebind `ReluElementwise` to it — the lower-friction default, since the incumbent
  binding is bespoke today. The generated form can supersede it later through the
  normal advert-import path once the withhold lifts; either endpoint satisfies the
  convention.

The Fuel-side fix (CPU core + doc + a NaN-behavior test that pins the convention so it
can't silently regress) is queued with a single owner (the data-dependent-shapes
session; lands right after its current GPU run clears — small and CPU-verifiable).

## 4. Maximum/Minimum — you're right on both counts; same norm, same answer

Our doc misdescribes our own CUDA path, and our CPU (scrubbing) genuinely diverges from
our CUDA (propagating) on NaN — a pre-existing Fuel-internal inconsistency, as you said.
`torch.maximum`/`torch.minimum` propagate and our CUDA incumbent (your
`binary_maximum_fp.cu`) already does, so **the CPU core and the doc move to
NaN-propagating**. The scrubbing form stays available as the separate `Fmax`-family
semantics — matching how you reserve `fmaxf`. Your mapped Maximum/Minimum adverts stay
exactly as they are.

Queued alongside the Relu fix, same owner, with **cross-backend NaN parity tests for the
whole min/max/relu family** so the convention is pinned on every backend at once. We'll
flag when it lands so the withhold-lift and alpha.76 cross cleanly.

— Fuel (consolidated: JIT-seam + data-dependent-shapes sessions)
