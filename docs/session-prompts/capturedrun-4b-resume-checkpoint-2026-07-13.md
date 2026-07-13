# CapturedRun 4b — resume checkpoint (2026-07-13)

**This supersedes `capturedrun-4b-paused-pending-fkc-verification.md` for the post-Steps-1-2
state.** That older doc's blocker ("rope has ZERO CUDA candidates; baracuda `rope_apply` never
registered") is **now closed** — Step 2 registered it. Read this file first; the old pause doc
remains valid for the acceptance-test iteration mechanics (its `eprintln` diagnostic recipe in
`capture_decode` still applies).

The FKC gap-closure program that this depends on is **done and merged to main** (`cf0c3ee2`);
77 CPU + 1 CUDA (`rope_apply_f32`) impls are really GPU-verified in the ledger.

---

## Exact state

- **Branch:** `capturedrun-4b-resume`, clean tree, HEAD `3be28ab1`.
- **Merge-base with main:** `cf0c3ee2` (i.e. this branch = FKC-closure-on-main **+** the two
  CapturedRun commits below).
- **Ledger:** `docs/kernel-contracts/.fkc-verified-ledger.json`, **78 entries** (77 CPU + 1 CUDA
  `rope_apply_f32`). `include_str!`-embedded → editing it recompiles fuel-dispatch.
- **Commits on top of main:**
  - `a7a4d223` — merge: reconciled the paused executor worktree (α/β/γ/ζ + δ wiring + 85 CUDA
    audit flips) with the FKC verifier. (Step 1.)
  - `3be28ab1` — **first fused-CUDA registration in the codebase.** Registered baracuda
    `rope_apply_{f32,f16,bf16,f64}_into` as the CUDA impl of `FusedOps::ROPE` via a real FKC
    contract. (Step 2.)

### What Step 2 built (all compiles under `--features cuda`, exit 0, verified 2026-07-13)
- `docs/kernel-contracts/cuda/rope-apply-fused.fkc.md` — `fused_op: ROPE`, backend `Cuda`,
  `kernel_source: "baracuda"`. `x` fans `{F32,F16,BF16,F64}`; `cos`/`sin` pinned `[F32]`
  **full-width** `[seq, head_dim]` (NOT the half-width the *primitive* `rope-apply.fkc.md`
  declares — that half-width candidate is permanently unreachable by the real graph; see the ABI
  note below).
- `fuel-cuda-backend/src/baracuda/attention.rs:661` — `narrow_rope_table_f32` (one
  `cuMemcpy2DAsync` D2D narrow-copy, modeled on `mamba.rs::strip_prepad_d2d:170`) +
  `rope_apply_fused_{f32,f16,bf16,f64}_into` drivers (narrow cos/sin → forward to existing
  `rope_apply_<dt>_into`).
- `fuel-dispatch/src/baracuda_dispatch.rs:3128` — `register_cuda_rope_apply_fused_from_contract` +
  `register_baracuda_cuda_fused_kernels:3151` (new public entry) + the fused-KernelRef wrappers.
- `fuel-dispatch/src/fkc/cuda_link.rs:766` — `CudaLinkRegistry::resolve_fused` **rewritten from a
  permanent `None` stub to a real lookup** (the first `fused_op` it ever resolves).
- `fuel-dispatch/src/dispatch.rs` — `register_default_fused_kernels` now calls
  `#[cfg(feature="cuda")] register_baracuda_cuda_fused_kernels(r)`.
- Default build: `cargo test -p fuel-dispatch --lib` = **670 pass / 2 ignored** (up from 669;
  new test `parses_and_lowers_real_rope_apply_fused_contract`).

### Why it's an ABI full-width→half-width narrow (derived, not assumed)
baracuda's `rope_apply_<dt>_run` wants HALF-WIDTH cos/sin `[seq, head_dim/2]`. Fuel's
`Tensor::rope_with_tables` (`fuel-graph/src/lib.rs:6423`) **hard-asserts FULL-WIDTH**
`[seq, head_dim]`. From `rope_with_tables_decomposed` (`:6486`) the shared-angle identity forces
`cos[j] == cos[j+half]` and `sin[j] == sin[j+half]` for all `j` → Fuel's full-width table **is by
construction** the half-width table duplicated across both halves, so the first `head_dim/2`
columns are byte-for-byte baracuda's half-width table. Full derivation in the contract and in the
module note at `attention.rs:625-659`.

---

## The remaining critical path (ordered) — and the entanglement that reorders it

> **CRITICAL FACT discovered this session (this changes the naive ordering in the worklist):**
> **CUDA ledger seeding is a *prerequisite* for the acceptance test, not a follow-on.**
> The V-FKC-9 gate downgrades every CUDA kernel's `audited:true` → `UNAUDITED` until it has a
> ledger `pass` entry. `BitStablePreferenceFilter` then deprioritizes the unaudited CUDA
> candidate, and it **loses placement to the CPU alternative** (which the CPU seeding *did*
> verify → audited). Net: with an unseeded CUDA ledger, the decode runs on **CPU** and never
> captures. So the decode-path CUDA kernels (incl. the new fused rope) **must be seeded before**
> the acceptance test can place the decode on CUDA at all.

**Step 2b — make the fused rope capture-safe** (`in_progress` when checkpointed; compile-check
DONE).
- The narrow-copy at `attention.rs:667` allocates a fresh device buffer **every call**
  (`device.alloc_zeros` at `:691`) → violates CapturedRun's zero-alloc-during-capture invariant.
  The KNOWN GAP is flagged in-code at `attention.rs:652-659`.
- **Fix design:** give the narrowed `cos_half`/`sin_half` a **grow-only per-device scratch cache**,
  modeled directly on `fuel-cuda-backend/src/baracuda/scratch.rs::WorkspaceCache` (grow-only,
  `Arc`-shared across `CudaDevice` clones, first consumer = flash_decoding; see its module doc
  at `scratch.rs:18-28`). A stable-capacity decode loop sizes cos/sin identically every step, so
  after step 1 the cache converges to one allocation and every later step reuses it → the D2D
  copy targets a fixed buffer, zero alloc during capture. Add a second cached buffer (or a
  2-slot cache) since cos and sin are narrowed independently and must not alias.
- GPU-verify numerically: the `cuMemcpy2DAsync` narrowing + the rope math have **never run**
  (Step-2 flag #1). Verify the fused CUDA rope output equals the CPU decompose reference before
  trusting it in the acceptance test.

**Step 3 — CUDA-seed the ledger** (GPU-heavy; the CUDA analog of the 77-CPU 4.5b seeding).
- Build a CUDA seeding harness mirroring `fuel-dispatch/src/fkc/verify/harness.rs` (which already
  seeds `rope_apply_f32` via `CudaInvoker` + `verify_bit_stability` + `ledger.upsert`). The
  `upsert` (replace-by-key, idempotent) is at `fuel-dispatch/src/fkc/verify/ledger.rs`.
- **Scope for greening the acceptance test = the decode-path subset only**, not all 85: IndexSelect
  (embed), RmsNorm, MatMul, Softmax, Silu, Add/MulElementwise, primitive Rope `[F32,F32]`, the
  fused ROPE, Affine, WriteSlice(`_doff`), Copy, Contiguize, Concat (~15 kernels). Seed the full
  85 afterward for completeness (the user's "seed all 85" ask).
- Key match is by construction: `revhash::compute_revision` (deterministic FNV-1a, no seed) makes
  the seeding-harness key `(backend, dtypes[format!("{d:?}")], kernel_revision_hash, claim)` match
  the gate key. Same recipe as the rope acceptance.
- **Duplicate-entry trap (already hit + fixed once):** the ledger is `include_str!`-embedded, so
  the first seeding write recompiles → re-embeds → a naive re-run appends a second identical entry.
  Always use `ledger.upsert`, never `ledger.push`. Regenerate a clean single-entry-per-key ledger.

**Step 4 — the acceptance test.** `forward_with_kv_context_captured_matches_persistent` in
fuel-core, `--features cuda --ignored`, **cuDNN on PATH** (else `STATUS_DLL_NOT_FOUND`). Must be
byte-exact captured == persistent. The old pause doc says expect further blockers past rope; iterate
with `eprintln` in `capture_decode` per its recipe. NOTE both halves must place on CUDA → depends
on Step 3 seeding the whole decode path, not just rope.

**Step 5 — 4b-ε bench.** Add the captured-replay leg to `run_persistent_decode_bench`
(third leg alongside uncaptured + persistent), median-of-≥8, nvidia-smi logging. See
`capturedrun-4b-real-decode-worklist.md` §ε.

---

## Build / run recipes (Windows, RTX 4070, this machine)

- **CUDA build needs vcvars64** (so nvcc finds `cl.exe`). The batch that worked this session
  (`/c/Windows/Temp/step2_cuda.bat`): `call ".../vcvars64.bat"` → `cd /d C:\Projects\fuel` →
  `cargo build -p fuel-dispatch --features cuda`. Run it via `cmd //c`, **in the background**
  (2-min tool timeout otherwise), then grep the redirected log for `BUILD_EXITCODE`.
- **fuel-core cuda *test exe* also needs cuDNN + CUDA on PATH** to launch (it imports
  `cudnn64_9.dll` + `cublas64_13.dll`): prepend `C:\Program Files\NVIDIA\CUDNN\v9.23\bin\13.3\x64`
  and the CUDA `bin`. (`fuel-dispatch` cuda tests do NOT need cuDNN.)
- **Discipline:** always `-p <crate>`, one cargo at a time, one live-GPU suite at a time (8 GB).

---

## Step-2 flags still open (from `.superpowers/sdd/step2-rope-fused-report.md`)
1. Narrow-copy numerical correctness DERIVED but UNTESTED on GPU → Step 2b verify.
2. Capture zero-alloc gap NOT closed → Step 2b scratch cache (design above).
3. cos/sin pinned F32 regardless of `x` dtype (baracuda ABI). Moot for the F32 decode; a
   hypothetical F16/BF16 model whose cos/sin are *also* cast to that dtype would miss this key and
   fall back to decompose (safe, just not accelerated).
4. `audited:true` downgraded to UNAUDITED on import until seeded — intended; Step 3 fixes it.

## Pointers
- Old pause doc (acceptance-iteration mechanics): `capturedrun-4b-paused-pending-fkc-verification.md`
- Worklist (6 gaps / 5 increments α–ε): `capturedrun-4b-real-decode-worklist.md`
- Step-2 research + flags: `.superpowers/sdd/step2-rope-fused-report.md`
- CPU seeding template: `fuel-dispatch/src/fkc/verify/harness.rs` (the `#[ignore]`'d
  `fkc_verify_rope_apply_writes_a_pass_ledger_entry`).
