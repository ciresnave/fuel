# CapturedRun 4b: PAUSED pending FKC contract verification + automatic kernel integration

**Status (2026-07-11): PAUSED, by explicit user decision.** Do not resume any work toward
finishing task 4b-δ (fuel-core wiring of `CapturedDecodeSession` into the real Llama decode
loop), 4b-ε (the captured-decode bench leg), or the broader "audit the next unaudited kernel
the trace hits" pattern **until the FKC contract-verification + automatic-kernel-integration
program described below exists**. This is not a suggestion — it is the resume condition. If
you are picking this file up, read it fully before touching `fuel-core`, `fuel-dispatch`, or
any `docs/kernel-contracts/**/*.fkc.md` file in service of finishing 4b-δ.

## Why we stopped here, not at some other point

This session (CapturedRun executor build-out, continuing from `docs/session-prompts/
capturedrun-4b-real-decode-worklist.md`) made real, verified progress — see "What shipped"
below — by repeatedly finding a CUDA kernel the real decode graph needed, discovering it was
either (a) marked `audited: false` in its FKC contract, unconditionally losing placement to a
CPU alternative regardless of cost or capability, or (b) never wired into Fuel's dispatch table
at all despite the kernel already existing and shipping in baracuda. Each time, the fix was:
read baracuda's actual kernel source by hand, reason about determinism, hand-edit the FKC
markdown, and (for the wiring gaps) hand-write a new dispatch wrapper + registration. This
worked — six real fixes landed, GPU-verified, see below — but it does not scale and it is not
the system's own designed process for this exact seam.

**The precise trigger for stopping**: tracing the sixth gap (`rope_apply` — see "Where we
stopped" below) led to the discovery that its FFI symbol exists in baracuda specifically
because Fuel asked for it ("Fuel ask Gap 2" in baracuda's own source comment) — baracuda built
and shipped it — and it was **never wired into Fuel at all**: no FKC contract section, no
dispatch registration, nothing. It just sat there, unused, until this session went looking for
why a decode graph couldn't capture. That is a much sharper failure than "a kernel is
unaudited" — it is "a kernel Fuel itself requested was delivered and then never connected."

Investigating why led to `docs/session-prompts/kernel-contract-adoption-plan.md` (FKC's
original design doc, already in this repo) and a direct comparison against the actually-shipped
validation code. The finding, precise and code-verified, not inferred:

- The design doc specifies **V-FKC-9**: *"a non-reference contract may not ship `UNAUDITED`
  precision or `unknown_cost` cost"* (`kernel-contract-adoption-plan.md:532-535`), and describes
  a **"ship → verify" gate** (`kernel-contract-adoption-plan.md:599-606`) where a provider's
  contract claims get checked against real behavior before the contract-import path replaces
  hand-written registration.
- The **actually-implemented** `V-FKC-9` (`fuel-dispatch/src/fkc/validate.rs:1080-1107`,
  `validate_precision_coherence`) is much narrower than the design doc's prose: it only checks
  that a `determinism: nondeterministic` declaration agrees with `bit_stable_on_same_hardware:
  false` + `audited: true` in the *same contract* — an internal-consistency check between two
  hand-authored fields. A contract that simply declares `audited: false` (no `determinism:
  nondeterministic` claim) sails through this check untouched — `fuel-dispatch/src/fkc/
  precision.rs:78-82`'s `lower_precision` treats `audited: false` as a **valid, successfully
  lowered** case (→ `PrecisionGuarantee::UNAUDITED`), not an error. `FkcError::
  PlaceholderPrecision` (the error type whose doc comment cites V-FKC-9) is only raised when a
  precision block is **entirely absent** (no `audited` field, no bounds, no `notes` at all) —
  never for an explicit, well-formed `audited: false`.
- The "ship → verify" **equivalence test** (adoption plan step 6) that gated each provider's
  migration onto the FKC-import path only checked that the imported registration **reproduced
  the pre-existing hand-written registration's values** — a migration-safety check, not an
  independent empirical test against the kernel's real behavior. Since the pre-existing
  hand-written registrations were themselves plain `register(...)` calls with no real precision
  claim (confirmed: every `audited: false` seed touched this session carries the identical note
  *"author-declared UNAUDITED seed (byte-for-byte the deleted plain register default)"*), the
  equivalence test trivially passed by reproducing the same never-audited default. Nothing
  independently verified the claim was true; nothing flagged that it hadn't been.

**Net: the two things FKC was designed to guarantee — (1) a provider's contract may not ship
placeholder precision, and (2) claims get verified before becoming load-bearing — are not what
the shipped code actually enforces.** This is a real, evidenced implementation gap relative to
the system's own stated design, not a new idea introduced this session. The user's framing,
verbatim, is the resume condition:

> "FKC contract handling was designed to allow kernel providers like Baracuda to create a
> kernel, make claims about it in an FKC contract, pass both of those to a kernel consumer like
> Fuel and have Fuel automatically test as many of the claims in the contract as possible and
> automatically put that kernel into full rotation using *all* of its abilities as if it had
> been there from day one."

Continuing to fix CapturedRun's decode-graph blockers one hand-audited kernel at a time is
solving the same problem this system already exists to solve, by hand, kernel by kernel, with
no mechanism to prevent the next kernel (baracuda ships more of these regularly) from landing
in the exact same unverified state. That is why we stop here rather than continuing to
"rope_apply" and whatever comes after it.

## What shipped this session (real, GPU-verified, all committed on `worktree-capturedrun-executor`)

Base: `1e8fc057` (origin/main at session start). Current HEAD: `68eed195`. Ten commits, each
independently tested:

1. `81660c2e` — **4b-γ**: `ContiguizeOf` + 2-input `Concat` write-into (extends the
   `PersistentOutputs` capture mechanism to two allocation sites the real decode graph hits).
2. `8e6e9e3d` — **4b-δ (partial)**: `CapturedDecodeSession` wired into `LlamaModel`'s decode
   loop (`forward_with_kv_context_captured`, `DecodeSession::logits_node()`/`base_cache()`
   getters, fixed per-token input buffers). Committed BLOCKED — the fuel-core wiring itself is
   done and correct; the end-to-end test does not pass yet (see "Where we stopped").
3. `c3e2b033` — **4b-ζ**: same-device CUDA→CUDA `Op::Copy` write-into (the pattern
   `insert_safety_copies` needs for GQA/KV-write aliasing hazards). Bonus: a real `fuel-graph`
   fix (`insert_safety_copies`' `target_location` fallback wrongly defaulted to `Cpu` when
   `placement` was unset).
4. `8d40f297` — cuBLAS `gemm_dense` (`MatMul`) determinism audit: `audited: true` with real
   evidence (NVIDIA docs + ~4500 empirical repeat-calls under contention + process restarts).
   The heaviest, most rigorous audit — appropriate since cuBLAS is a vendor black box.
5. `2b285921` — decisions-log correction: the real 4b-δ blocker after (4) was `MulElementwise`
   UNAUDITED, not a cost-based `BroadcastTo` placement as first suspected.
6. `eb2d13f9` — **`scatter_add_f32`/`f64` false-claim fix**: these two FKC sections claimed
   `bit_stable_on_same_hardware: true` while baracuda's own kernel source discloses genuine
   `atomicAdd` nondeterminism. Not an audit — an active, pre-existing wrong claim, corrected.
7. `afcce809` — **80-kernel simple-elementwise/memory-movement tier** audited to `audited:
   true`, via a 14-family Workflow (draft + independent adversarial-verify agent per family).
   Caught and recovered from two real process failures: 2 families fabricated their completion
   reports (claimed edits that were never applied — caught by the verify step reading files
   fresh); several more landed in the wrong git worktree (the main checkout instead of this
   one) and had to be recovered and relocated before commit. See the 2026-07-11 decisions-log
   entry for the full account — worth reading before running another such workflow.
8. `0d5e58e4` — **moderate-tier decode-relevant audit**: `softmax`, `log_softmax`, `rms_norm`
   (done directly by the controller, not delegated — small bounded scope). Both share one
   pattern: a host launcher dispatches between a legacy per-thread kernel and an SMEM
   block-reduction fast path; eligibility is a pure function of shape/stride/dtype, and the
   SMEM path's reduction is a fixed warp-shuffle butterfly (no atomics) — genuinely
   deterministic, confirmed by reading `baracuda_norm.cuh`/`baracuda_softmax.cuh` directly.
9. `6203a76b` — reverted 4b-δ's test fixture to this file's usual tiny dims, since the audits
   proved the earlier "bump the fixture to avoid cost-based CPU placement" theory was never the
   real lever (every blocker found was the audited-flag gate, independent of tensor size).
10. `68eed195` — **`Rope` capture-safety**: audited `rope.fkc.md`'s single-input kernel
    (`OpKind::Rope` at dtype key `[F32,F32]`) to `audited: true`, and added `OpKind::Rope` to
    `op_kind_is_capture_writeinto` (the write-into wrapper, `cuda_rope_baracuda_wrapper!` →
    `rope_*_into`, already existed — it had simply never been added to the predicate, since
    earlier analysis wrongly assumed decode's rope always decomposes to elementwise ops).

Total: **85 CUDA kernel precision claims corrected or newly audited** (1 cuBLAS + 80 simple +
3 moderate + 1 rope, minus the 2 scatter_add corrections counted separately), **2 real
capture-safety mechanism extensions** (4b-γ, 4b-ζ), and **1 real fuel-graph bug fix**
(`insert_safety_copies` placement fallback), all independently GPU-verified, zero regressions
in the full `fuel-dispatch --features cuda --lib` suite (633 passed throughout) or the 8-test
`capture_decode_*_cuda` regression suite.

## Where we stopped (the exact technical state, for whoever resumes)

`cargo test -p fuel-core --features cuda --lib forward_with_kv_context_captured_matches_persistent
-- --ignored --nocapture` (prepend `PATH="/c/Program Files/NVIDIA/CUDNN/v9.23/bin/13.3/x64:$PATH"`
or the test binary dies `STATUS_DLL_NOT_FOUND`) still fails. The blocker chain this session
traced, in order, each confirmed by a temporary `eprintln!` diagnostic in `capture_decode`'s
validation loop (added, run, reverted — never committed):

MatMul (audited) → MulElementwise (audited) → RmsNormLastDim (audited) → Softmax/LogSoftmax
(audited) → `Op::Rope` at dtype key `[F32,F32]` (audited + wired) → **`Op::Rope` at dtype key
`[F32,F32,F32,F32]`, the fused/table-based variant (`x, cos, sin → out`, `fuel-graph/src/
registry/rope.rs`'s `FusedOps::ROPE` / `Op::Fused` emission, matching `Tensor::
rope_with_tables_decomposed`'s non-decomposed sibling). Zero CUDA candidates are registered for
this key at all** — confirmed via a temporary diagnostic in `fuel-dispatch/src/ranker/
enumerate.rs::enumerate_candidates` dumping every candidate's backend/precision for
`OpKind::Rope` (added, run, reverted — not committed). Only `backend=Cpu` entries appear.

Baracuda already ships the exact kernel needed: `rope_apply_f32`/`f16`/`bf16`/`f64`
(`baracuda_kernels_rope_apply_<dt>_run`, FFI declared via `BARACUDA_KERNELS_ROPE_APPLY_INSTANTIATE`
in `baracuda/crates/baracuda-kernels-sys/kernels/include/baracuda_attention.cuh:1703`, host
launcher `launch_rope_apply_fp` at line 1105, kernel `rope_apply_fp_kernel` at line 1017) — "RoPE
apply variant with caller-supplied precomputed cos/sin tables," built specifically in response
to a Fuel request ("Phase 36 (Fuel ask Gap 2)" per the source comment). **It has never been
registered in `fuel-dispatch`'s CUDA binding table, has no FKC contract section, and has no
dispatch wrapper.** This is the concrete instance that triggered the pause — see "Why we
stopped" above.

**Do not hand-wire `rope_apply` to unblock this test.** That would repeat exactly the pattern
this pause exists to stop. The correct fix is: build the FKC verification + auto-integration
program below, have baracuda submit (or Fuel author with baracuda's direct input) a real
contract for `rope_apply`, let the new automated pipeline verify + integrate it, and 4b-δ's
test should then pass without further hand debugging — that itself is the acceptance test for
the new program.

## The prerequisite: FKC contract verification + automatic kernel integration

This is a **design sketch**, not a finished plan — properly scoping this is real
architecture work deserving its own session (the `superpowers:brainstorming` skill, most
likely, given "we build genuinely new capability" is exactly its trigger condition), ideally
with the same rigor `kernel-contract-adoption-plan.md` itself was written with. Captured here
so the next session has a concrete starting point instead of a blank page.

**The goal, in the user's words**: a kernel provider (baracuda, or a future third provider)
ships a kernel *and* makes claims about it in an FKC contract; Fuel automatically tests as many
of those claims as it can; a kernel whose claims pass goes into full rotation using all of its
declared abilities, as if it had been there from day one — no hand-wiring, no manual audit
archaeology, no silent unaudited/unwired kernels sitting in baracuda's tree unconnected.

**What already exists to build on** (do not rebuild these):
- The FKC schema + parser + `register_into` path (`fuel-dispatch/src/fkc/*`) — contract →
  `KernelBindingTable`/`FusedKernelRegistry`, already unconditional core infrastructure.
- `PrecisionGuarantee::REFERENCE` / `PRIMITIVE_DETERMINISTIC_CPU` — CPU kernels already have a
  trusted reference tier other claims could be diffed against.
- The `determinism:` field + `validate_precision_coherence` — a real, if narrow, coherence
  check; a template for what a broader validator function looks like.
- This session's cuBLAS audit (`fuel-cuda-backend/src/baracuda/gemm_dense.rs::
  determinism_audit`) — a hand-written instance of exactly the kind of check (repeat-call
  bit-exactness, under contention, across process restarts) a generic verifier should be able
  to run for any kernel claiming `bit_stable_on_same_hardware: true`, automatically, from the
  contract alone (input shapes/dtypes are already declared in `accept:`).

**What's missing, concretely** (from the conversation that led to this pause):
1. **A per-claim-type automated verifier.**
   - `bit_stable_on_same_hardware: true` → synthesize inputs matching a declared `accept:`
     combination, call the kernel N times, hash/diff outputs, fail on any divergence. Directly
     generalizes the cuBLAS audit's protocol. Highest value, lowest lift — this is the claim
     that gated every blocker this session hit.
   - `max_ulp`/`max_relative`/`max_absolute` → diff against a `PrecisionGuarantee::REFERENCE`
     implementation (CPU, where one exists) on the same inputs, check the claimed bound holds.
   - `accept:` layout/dtype/shape coverage → smoke-test every declared combination: does the
     kernel run without erroring, does the output shape/dtype match the `return:` rule.
   - `cost:` claims → weaker guarantee achievable (order-of-magnitude benchmark sanity check).
2. **A real V-FKC-9**: reject (or at minimum loudly flag, gate behind an explicit escape hatch
   with a human-reviewable reason) a non-reference contract that ships `audited: false` — the
   design doc already specifies this; the implementation needs to actually do it.
3. **A genuine ship→verify gate**, distinct from the existing migration-equivalence test: newly
   submitted or edited contract claims must pass the automated verifier *before* `audited: true`
   (or any claim) becomes load-bearing for placement decisions — not just structurally valid
   YAML that happens to parse.
4. **Automatic integration**: once a contract's claims verify, the kernel should enter the
   normal binding table / placement rotation without a human writing a dispatch wrapper by
   hand for the common cases (the write-into vs allocate-and-return distinction from this
   session's earlier increments may still need per-shape-family templates, but the
   registration + FKC-authoring + audit steps should not require a human archaeology session
   per kernel).
5. **Process, not just code**: does baracuda submit contracts alongside kernels going forward
   (the `rope_apply` case argues yes), and if so, in what format / through what channel — this
   needs to go through the existing "propose-first" Baracuda↔Fuel correspondence, not be
   decided unilaterally by Fuel.

**Explicit non-goals for this sketch**: this is not a call to re-litigate FKC's schema, nor to
retroactively force every existing kernel through the new verifier in one pass (that's its own
follow-on, likely large, effort once the mechanism exists) — scope the first version to (a) the
automated bit-stability verifier (item 1's first bullet) plus (b) wiring one real
provider-submitted contract (`rope_apply` is the natural acceptance test) through it end to end.

## Resume checklist

When FKC contract verification + automatic integration lands (or a deliberate decision is made
to proceed without it, which should itself be a recorded decision, not a silent default):

1. Re-read this file in full.
2. Confirm `rope_apply` (or whatever the new pipeline's first real integration is) actually
   unblocks `forward_with_kv_context_captured_matches_persistent` — that's the natural
   acceptance signal that the new mechanism works, not just that it exists.
3. If it doesn't fully unblock the test, resume the same diagnostic technique used throughout
   this session (temporary `eprintln!` in `capture_decode`'s validation loop or
   `enumerate_candidates`, run, revert before committing) rather than guessing.
4. Once 4b-δ's test passes: 4b-ε (the captured-decode bench leg, `run_persistent_decode_bench`
   at `fuel-core/src/lazy.rs:11963`, third leg + median-of-≥8 protocol + nvidia-smi logging) is
   the next and final increment of the original CapturedRun worklist.
5. Final whole-branch review + `superpowers:finishing-a-development-branch` once 4b-ε lands.

## Related documents

- `docs/session-prompts/capturedrun-4b-real-decode-worklist.md` — the worklist this session
  continued (gaps α–ζ, all now landed).
- `docs/session-prompts/kernel-contract-adoption-plan.md` — FKC's original design; §10 (the
  V-FKC-* validators) and §11 step 6 (the "ship → verify" equivalence gate) are the sections
  most relevant to the gap this pause is about.
- `docs/architecture/10-decisions-log.md`, 2026-07-11 entries — the cuBLAS audit, the
  80-kernel audit (including the fabrication/wrong-worktree process failures), and this pause
  decision itself.
- `.superpowers/sdd/cuda-kernel-audit-inventory.md` (this worktree, gitignored scratch — not
  committed) — the full 99-kernel inventory; 85 are now resolved (audited or corrected) by
  this session, 14 remain (the "moderate" non-decode-relevant tier — `reduce`/`arg_reduce`/
  `reduce_to`/`cumsum`/`causal_conv1d` — and the 2 "complex" kernels, `gemm_int`/
  `flash_decoding`). Worth regenerating fresh rather than trusting this stale copy if picked up
  much later, since the underlying `.fkc.md` files will have moved.
