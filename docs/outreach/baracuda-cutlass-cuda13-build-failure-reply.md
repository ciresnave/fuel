# Baracuda reply — CUTLASS build-failure report (RECEIVED 2026-07-03)

**Received via CireSnave; filed with Fuel-side takeaways. Both findings
root-caused and fixed on their `feat/kernel-specialization` (commit `297a3aa`,
ships next alpha). The blocker clears TODAY on alpha.72 with a one-line
workaround.**

## Finding 2 (swallowed stderr) — confirmed, theirs, fixed

Root cause: all four child invocations in `baracuda-forge`'s builder (nvcc
compile, nvcc PTX, nvcc `--lib` link, `lib.exe` archive) used `.spawn()` +
`wait_with_output()`. `spawn()` inherits the parent's console handles, so the
"captured" stdout/stderr were **empty by construction** — every failure
reported a bare `nvcc error:` regardless of what nvcc said. Fixed with
`Command::output()` (pipes both streams); errors now carry the exit status
plus full stdout/stderr.

## Finding 1 — NOT a CUTLASS x CUDA 13.3 incompatibility

With diagnostics visible, Fuel's exact environment reproduces as:

```text
nvcc fatal   : Cannot find compiler 'cl.exe' in PATH
```

Mechanism: **rustc locates MSVC by itself, but nvcc requires `cl.exe` on
PATH** — which only a VS Developer shell provides. From a plain terminal the
pure-Rust crates compile fine, then the first build script that runs nvcc dies;
`gemm_batched_rcr_sm80.cu` was simply the first file in that crate's parallel
compile set. CUTLASS was never reached. Positive confirmation on their side:
**CUTLASS v4.2.0 compiles all 22 curated sm80 kernels clean under CUDA 13.3 +
MSVC 14.51** (dev shell and plain shell both, with the fix). Also noted:
`baracuda-cutlass-sys` fetches CUTLASS itself (sparse checkout of pinned v4.2.0
into `~/.baracuda-cutlass-sys/checkouts/`, `CUTLASS_DIR` to override).

## The permanent fix (shipped, next alpha)

`baracuda-forge` now resolves nvcc's host compiler itself (`resolve_ccbin`):
(1) `NVCC_CCBIN` env override; (2) `cl.exe` already on PATH; (3) otherwise
locate `cl.exe` via `vswhere` and pass `-ccbin`, with a cargo warning naming
the chosen compiler. `cargo build` of every nvcc-compiled Baracuda crate works
from any shell on Windows.

## Unblock now, on alpha.72

Either of these makes Fuel's `--features cuda` build work immediately:

- Run the build from a **VS x64 Native Tools / Developer shell**, or
- Set **`NVCC_CCBIN`** to the cl.exe path, e.g.
  `C:\Program Files\Microsoft Visual Studio\<ver>\<edition>\VC\Tools\MSVC\<toolset>\bin\Hostx64\x64\cl.exe`

---

## Fuel-side takeaways (recorded 2026-07-03)

1. **The CUDA build blocker is DISSOLVED** — it was never CUDA 13.3, CUTLASS,
   or removed APIs; it was `cl.exe` absent from PATH in plain shells, masked by
   the (now-fixed) swallowed stderr. Fuel's `CLAUDE.md` sibling-deps note is
   updated accordingly (their explicit ask).
2. **Un-parked today** (pending an exclusive cargo slot — the cuda dep build is
   long): the FKC cost-unification Part A cuda-gated line verify, and the
   Baracuda-backed `StructureKeyProvider` build/test (the last piece of the
   pinned dispatch-record wire loop).
3. Workaround of record on this machine: `NVCC_CCBIN=<path-to-cl.exe>` in the
   environment of any `--features cuda` cargo invocation (until the next alpha
   lands `resolve_ccbin`).
4. Process note: the swallowed-stderr fix means future baracuda build failures
   arrive with real diagnostics — no more suspect-list bug reports.
