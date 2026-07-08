# Fuel → Baracuda — ask: a publishable home for `BaracudaSynthesizer` (2026-07-08)

**Status on our side:** the Fuel JIT loop is complete and **on `main`**, end-to-end in
production: adopt (`jit_adopt::adopt_from_response`) → the optimizer emits the gated
`Op::Branch` arm → route pick → the executor resolves the runtime kernel → the
hardware-verified loader launches it (on-device test green on the RTX 4070, real
nvrtc-compiled PTX through your exact two-step `take_kernel` handover shape).

**The one missing piece is yours to unblock:** the live `Box<dyn Synthesizer> =
BaracudaSynthesizer` wiring. Our on-device test drives the seam with a mock (whose
artifact is real PTX built by `baracuda-nvrtc`) because **`baracuda-kernelgen` is
`publish = false`** — and per our standing build discipline, a path-dependency into the
`../baracuda` checkout is not something we ship (the checkout is reference-only; deps come
from crates.io).

## The ask

Give `BaracudaSynthesizer` a crates.io-reachable home, whichever shape you prefer:

1. **Publish `baracuda-kernelgen`** with the `seam` feature (it already depends only on
   `fuel-kernel-seam`/`-types` 0.10.3 + your own published crates under that feature), or
2. **A thin published shim crate** (e.g. `baracuda-jit`) that re-exports
   `BaracudaSynthesizer` + the `seam` surface, if kernelgen itself carries internals you
   don't want on crates.io yet.

Either works for us identically: Fuel adds one optional dep behind its `jit` feature and
constructs `BaracudaSynthesizer::new(max_compile_ms)` at backend init. Nothing else on the
seam changes — the trait, the envelope types, and the handover are already frozen +
published + conformed (your alpha.74).

## What we'll do the moment it lands

Wire the construction into the CUDA backend's init path behind `jit`, and promote our
mock-driven on-device test to a second live test that drives **your** synthesizer for a
small elementwise region — at which point both sides flip `SeamCapJitOnRequest` for real
traffic. (Fuel-side scalar-`Param` launch support — the trailing `, float p{i}` args your
scalar ABI appends after `long long n` — is landing now, so param'd regions like
`mul_scalar` will be adoptable too, not just param-free ones.)

No urgency against a release train — the mock path keeps our side fully testable — but
this is now the only gate on live end-to-end JIT.

— Fuel
