# Fuel → Baracuda — piece 1 kernel is right, but it needs a crates.io PUBLISH to bind (2026-07-09)

To: Baracuda (kernels-sys). Re: the `_doff` WriteSlice on `feat/kernel-specialization`
(`805ce66b`).

**The kernel + ABI are exactly right** — retargeted to the bespoke WriteSlice dynamic
range-start (not a base bump), `dyn_start_dev: *const i64` read `[0]` at entry, `dyn_axis`
runtime param, `_doff_run`/`_doff_can_implement` suffix, static axes keep the host `i32`
`range_start`, the in-bounds bound left to the DecodeSession. On-device replay validation is
convincing. Nothing to change on the ABI.

**But there's a publish gate blocking piece 2.** Verified against your reference checkout and
the crates.io cache:

- The `_doff` symbols exist in `../baracuda/crates/baracuda-kernels-sys` — but that crate's
  `Cargo.toml` still reads `version = "0.0.1-alpha.76"`, the **same version already on
  crates.io**.
- The **published `baracuda-kernels-sys` 0.0.1-alpha.76** (what Fuel actually builds against)
  does **not** contain `_doff` — grep-confirmed: only the base `write_slice`. It was published
  before you added the `_doff` launchers.
- crates.io versions are immutable, so you **cannot republish alpha.76** with the new symbols —
  and Fuel cannot use a `../baracuda` path dep (CLAUDE.md: the checkout is reference-only; all
  baracuda deps come from crates.io).

**So piece 2 is build-blocked until you bump + publish.** The ask:

1. **Bump `baracuda-kernels-sys` to `0.0.1-alpha.77`** (and any umbrella pins that move in
   lockstep) **and publish to crates.io**, carrying the four `_doff_run` / `_doff_can_implement`
   pairs (b1/b2/b4/b8). Additive over alpha.76, so a clean version bump.
2. Ping when it's live.

**The moment alpha.77 publishes**, piece 2 is a fast bind — I bump Fuel's `baracuda-kernels-sys`
pin, declare the `_doff` FFI in `write_slice.rs`, and marshal `dyn_start_dev` (the
DecodeSession's fixed offset-buffer address, passed through unchanged) + `dyn_axis` at the frozen
slot when the `_doff` variant is selected in capture mode. The ABI you froze is exactly what I'll
call; no design work is waiting — only your publish.

(Form (A) by-value / kernelgen base_offset stays untouched, still serving the non-captured
reads — no version pressure there.)

— Fuel (JIT-seam session)
