# Baracuda bug report: cutlass-kernels-sys build failure under CUDA 13.3 (+ swallowed nvcc stderr)

**Status:** ready to relay (2026-07-03). Two findings from Fuel's side, one repro.
**Severity for Fuel:** blocks every `--features cuda` build of the workspace — the
FKC cost-unification Part A cuda-gated line remains uncompiled/unverified, and the
Baracuda-backed `StructureKeyProvider` impl (the last piece of the pinned
dispatch-record wire loop) can't be built or tested until this clears.

## Environment

- Windows 11, MSVC host toolchain, RTX 4070
- **CUDA 13.3** — `nvcc` reports `release 13.3, V13.3.33`
  (`Build cuda_13.3.r13.3/compiler.37862127_0`)
- `baracuda-*` from crates.io at **0.0.1-alpha.72** (the fuel workspace pin)

## Repro

```
cd fuel
cargo check -p fuel-dispatch --features cuda
```

Dependency compilation proceeds through `baracuda-cutlass-sys`, `baracuda-forge`,
`baracuda-cutlass-kernels-sys`, `baracuda-kernels-sys`, then:

```
error: failed to run custom build command for `baracuda-cutlass-kernels-sys v0.0.1-alpha.72`
  baracuda-cutlass-kernels-sys: nvcc build failed: CompilationFailed {
      path: "kernels/gemm_batched_rcr_sm80.cu",
      message: "nvcc error:\n\n"
  }
```

## Finding 1 — the actual failure: `gemm_batched_rcr_sm80.cu` won't compile under CUDA 13.3

`nvcc` errors on the CUTLASS SM80 batched-GEMM kernel. We can't see *why* (see
Finding 2), so the suspects are the usual new-toolkit ones, in rough likelihood
order:

1. **CUTLASS version × CUDA 13.3 incompatibility** — CUTLASS releases typically
   need bumping for a brand-new major toolkit; whatever CUTLASS the kernels-sys
   vendored may predate 13.x.
2. **Removed/deprecated CUDA APIs** — CUDA 13 removed several long-deprecated
   textures/APIs; anything the SM80 path touches transitively could now be a hard
   error.
3. **MSVC host-compiler interaction** — new nvcc versions occasionally tighten
   host-flag handling on Windows.

If it reproduces on your side, a CUTLASS bump (or a `-arch`/guard so the sm80
CUTLASS path degrades gracefully on toolkits it can't build under) both work for
us — Fuel routes through `baracuda-kernels-sys`'s FFI surface and doesn't
specifically need the CUTLASS batched path to exist on every toolkit.

## Finding 2 — the build script swallows nvcc's stderr (fix this regardless)

The `CompilationFailed` message is literally `"nvcc error:\n\n"` — the actual
compiler diagnostics are dropped, so every downstream consumer diagnoses blind.
Wherever the build script invokes nvcc and formats `CompilationFailed`, capturing
and including (at least the tail of) stderr would have turned this report from
"suspects, in rough likelihood order" into the actual error line. That's worth
fixing even if Finding 1 turns out to be environment-specific.

## What Fuel does meanwhile

- `--features cuda` builds are parked; the blocker is recorded in Fuel's
  `CLAUDE.md` (sibling-deps section) so sessions don't rediscover it.
- The dispatch-record/miss telemetry loop is fully built on Fuel's side against
  the pinned schema; the cuda-gated Baracuda `structure_key` provider slot is the
  piece waiting on this build.
