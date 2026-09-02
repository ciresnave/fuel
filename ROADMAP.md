# Fuel — Roadmap

This document describes the current state of this project, the structural and ergonomic
problems it aims to solve, and the planned order of work.

---

## Authoritative architecture: see `docs/architecture/`

The architecture documents in [`docs/architecture/`](docs/architecture/00-index.md)
are the constitutional description of fuel — what fuel is, how it's structured, what
makes it competitive, and the boundaries it commits to. **When this ROADMAP and the
architecture set conflict, the architecture set wins**; this ROADMAP is updated to
match. The architecture set was established at v1.0 on 2026-05-09 and captures 24
foundational architectural decisions in
[`docs/architecture/10-decisions-log.md`](docs/architecture/10-decisions-log.md).

This ROADMAP describes *the path* — phases, work items, sequencing, current state.
It anchors to architecture sections by cross-reference rather than by restating them.
Most phase entries below were drafted before the architecture set existed; where they
diverge from the architecture, the architecture is the source of truth and the phase
entry will be updated next time it's actively worked on.

---

## Current frontier (last full pass 2026-08-26; item 11 added 2026-09-02)

**Vision anchor.** Fuel is a DAG-first, lazy-only ML framework: the DAG is the source of truth for every decision, and the optimizer reading it is where the intelligence lives. The full statement is [`docs/architecture/01-identity.md`](docs/architecture/01-identity.md); this ROADMAP is *the path* toward it. Every item on the active frontier below moves at least one of the four [identity-enforcement checks](docs/architecture/01-identity.md#how-this-identity-is-enforced) *more* true and none less.

**The "plan IS the graph" redirection (2026-06-14, sharpened 2026-06-22).** The optimizer writes its decisions *into the graph* — chosen backend as a `Graph` side-table stamp, alternative execution paths as arms of `Op::Branch` decision-point nodes — and the executor picks among arms at runtime by live device load. `ExecutionPlan` is transitional scaffolding being removed; build new dispatch/fusion infrastructure into the **graph + the runtime-mutable kernel registry**, never into a side "plan." See [`docs/foundational-types.md`](docs/foundational-types.md) and the memory `plan-is-the-graph-architecture`.

**Crate-name glossary — read before trusting any crate name in this file (added 2026-07-30).** This document spans a long history, and several crates it names have been renamed or never existed. Corrections were previously recorded *locally* (the as-built note at "Current state (as built)" below), which does not help a reader who lands mid-document. Verified against the workspace manifest on 2026-07-30:

| Name you may read here | Reality |
|---|---|
| `fuel-core-types` | **Renamed → `fuel-ir`** (B0.1). Mentions inside item 2 below correctly describe *the rename itself*; elsewhere, read `fuel-ir`. |
| `fuel-storage` | **Renamed → `fuel-memory`** (B0.5). |
| `fuel-nn` | **Never existed.** The NN surface lives in `fuel-core` as `lazy_nn/` + `lazy_nn_varbuilder` / `lazy_nn_varmap`. `kv_cache` and `sampling` are in `fuel-core`, not here. |
| `fuel-reference-backend`, `fuel-graph-cpu`, `fuel-graph-executor`, `fuel-graph-router`, `fuel-loaders`, `fuel-autograd`, `fuel-model`, `fuel-tensor`, `fuel-flash-attn-cuda`, `fuel-internal` | **No such crate on disk.** These are target/aspirational decompositions from earlier plans. Treat any file path under them (e.g. `fuel-reference-backend/tests/attention.rs`) as *not a location you can open*. |

The 40 crates that actually exist are the directories with a `Cargo.toml` at the repo root; `fuel-tensor-tools` is a real crate and is *not* the aspirational `fuel-tensor`. Historical and checklist entries naming a since-renamed crate are correct **as history** and are deliberately not rewritten — this table is the resolution mechanism instead.

### Active frontier (critical-path order)

1. **Dispatch-core cleanup** — move every strategic decision out of the realize-time bridge into `optimize_graph`, then have the executor read the graph. Step A (backend-stamping → optimizer) + **Step B (layout-fixup + residency cross-device-copy passes → `optimize_graph`, 2026-06-27)** + **Step C (arm-selection → the executor, 2026-06-28)** shipped — the bridge no longer makes any placement/residency/layout decision *or pre-computes the branch route*; the graph arrives at the executor fully stamped, copy-stitched, and fixed up (cache residency reaches the optimizer via the `PlanOptions::input_residency` provider), and the executor itself runs `pick_route` at dispatch (`realize_with_optimized_picking_env`) — the bridge only builds + hands over the Device/Judge-derived selector + live lookup. **Step D (delete the threaded `ExecutionPlan`, 2026-06-28)** shipped — `optimize_graph` returns only `OptimizedGraph`; the plan is now an optimizer-INTERNAL accumulator (`compile_plan` + the stamp/residency/layout passes + rankers), never returned or threaded. The executor re-derives each `Op::Branch` arm's candidate from the graph + the runtime binding registry (`enumerate_fork_set`); attribution + generation read the graph/registry. **Step E (live-load arm re-picking)** is **scoped as its own program** (2026-06-28): investigation found it gated on prerequisites that don't exist — execution is synchronous (no varying queue depth to react to) and there's no load telemetry. Sequenced **A (async/concurrent execution foundation — the real gate, fuel-internal, large) → B (queue-depth signal: B1 a fuel-internal per-device in-flight counter is the primary signal; B2 optional baracuda/vulkane cross-process telemetry) → C (streaming run-walk + `DeviceLoadSelector` — the actual per-decision-point re-pick)**. Design + outcomes in [`docs/session-prompts/step-e-async-execution.md`](docs/session-prompts/step-e-async-execution.md) (+ the `docs/outreach/*queue-depth*` ask/response pairs). **Progress (2026-06-29): A1 (completion-handle seam, `06cf3fbf`) + A2 (Vulkan async — lazy `flush_pending` + `force_flush` + executor force-flush guards; live-GPU-verified) + A3 (CUDA async — stream-ordered alloc/free via baracuda alpha.72, all 59 per-op `synchronize` removed, `alloc_zeros`→`zeros_async`, mem-pool retention; live-CUDA-verified 8/8 on RTX 4070) shipped; B2 telemetry resolved with both siblings (baracuda alpha.69 `Stream::is_complete()`/NVML; vulkane `device_identity()` join-key).** A4 (concurrent multi-device): **A4a (the placement mechanism) is ALREADY BUILT — verified live 2026-06-29.** Two read-only probes (and the A4 design premise) misread plan.rs:187-192's doc comment as a graph-global "one device" constraint; a max-effort Opus agent that read the prune *code* found it's a **per-node** prune (each node's alternative set → that node's own device) = exactly the per-node placement wanted, plus an `Op::Branch` arm-safety invariant. A live test (`fuel-core/tests/cuda_multidevice_realize_live.rs`) realizes a mixed **CPU+CUDA** graph in ONE pass — per-node placement honored, cross-device `Op::Copy` auto-inserted both directions, byte-exact, green on the RTX 4070 (independently re-run). The only change was correcting the misleading doc comment (zero executable lines). A4a-2 auto-placement also confirmed wired (priced placement DP). **A4c-prereq (multi-vendor CUDA↔Vulkan CORRECTNESS) shipped 2026-06-29 (`54e7043b`)**: two-hop CUDA↔Vulkan residency (host-staged via `SystemTopology::transfer_path`, CSE-shared CPU intermediate) + dual-device-seed (`realize_one_as_multi_device`); a CUDA(4070)+Vulkan(AMD iGPU) graph realizes byte-exact in one pass (`cuda_vulkan_multidevice_realize_live.rs`); 7/7 live tests green, single-device byte-identical. **A4b (CONCURRENCY) SHIPPED 2026-06-29** — independent CUDA + Vulkan sub-DAGs now run CONCURRENTLY in one realize. 5 PRs: A4b-1 CUDA `Pending(Event)` (`51d43e93`) → A4b-2 Vulkan `Pending`/`submit_batch` (`056ca786`) → A4b-3 finer cross-device wait (`4150ad7a`) → A4b-4 eager Vulkan submit + A4b-5 dual-GPU overlap benchmark (`16aefc28`). **Overlap confirmed: ~0.50–0.66 efficiency** (combined ~1.10s vs sequential ~1.26s, RTX 4070 + AMD iGPU); single-device byte-identical (the `multi_backend` gate keeps the eager path unreachable single-device); in-flight Vulkan-batch data-buffer lifetime UAF-guarded; 40× stress-clean. The key insight that landed it: the overlap enabler is *eager Vulkan submission*, not the handle (Vulkan deferred its batch to realize-end, so the iGPU never ran while CUDA did). Design + the 5-PR detail: [`docs/session-prompts/step-e-a4b-async-completion.md`](docs/session-prompts/step-e-a4b-async-completion.md). **CAVEAT (carries into C):** A4b is the overlap *mechanism + proof*; the topo scheduler emits each chunk's D2H right after its producer, so *general* automatic overlap of arbitrary independent sub-DAGs is the **Phase C** `DeviceLoadSelector`/scheduler frontier (dispatch independent sub-DAGs adjacently). **Phase C (the load-aware scheduler) SHIPPED 2026-06-30 — Step E core COMPLETE.** C-0 residency-all-arms fix (`5cf57516`, caught a real latent bug) → B1 per-device in-flight counter + the constitutional `BackendStreams` trait (`d3d21e20`) → C1 streaming run-walk (lazy per-branch resolution, `aed217d7`, streamed==one-shot) → C2 `DeviceLoadSelector` (`337a4b77` — the **live-load arm re-pick**, the original Step E goal: a 2-device branched graph picks the UNLOADED arm under load, VRAM still outranks load, no-load byte-identical) → C3 auto-overlap reorder (`1ad3c32d` — device-alternating topological reorder, a byte-identical hint, makes A4b overlap automatic for arbitrary graphs). Design: [`docs/session-prompts/step-e-phase-c-design.md`](docs/session-prompts/step-e-phase-c-design.md). **Follow-ons — 2 resolved + 1 in flight (2026-06-30):** (1) operand-independent wall-clock auto-overlap — **RESOLVED, a misdiagnosis**: a max-effort investigation (`c48ddfa3`) proved it was MEASUREMENT NOISE, not a code path — C3's reorder sorts runs by downstream compute weight (operand-position-independent), so both reconverge operand orders lower to a byte-identical (op,device) dispatch order and the executor already eager-submits all Vulkan before any wait (already operand-insensitive); the earlier "0.39 vs 0.0" flipped run-to-run on IDENTICAL code (thermal + CUDA mem-pool growth + iGPU OOM ceiling). Proven by `c3_reorder_is_operand_order_invariant` (CPU) + the both-orders benchmark. The full ready-set Kahn pump stays a *deferred* end-state for bounded-lookahead / CUDA-graph replay — not for operand-independence. (2) the error-path UAF — **FIXED** (`1eb3f515`): `SubmittedBatch` self-waits its fence on Drop when `!consumed` (never-panic, no double-wait on the normal path). (3) **A2.1** (Vulkan deferred-deletion of evicted buffers, throughput-only) — **SHIPPED** (`ebce71f5`): the multi-backend eviction path retains the evicted buffer on the in-flight batch (freed post-fence via (2)) instead of host-blocking on a drain; single-device keeps the blocking `force_flush` (byte-identical). All three Step E follow-ons resolved. **Lesson (recorded): test, don't read — structural code-reading misjudged A4 thrice; a fresh agent that ran a live test found the truth each time.** *Unblocks Phase D integration + the §10 runtime/JIT refactor.*

2. **fuel-core / fuel-core-types retirement → `fuel-ir` + `fuel-hardware` + `fuel-backend-contract`** (foundation refactor; scoped, sibling-safe). `fuel-core-types` (the vocabulary) → `fuel-ir` (B0.1); hardware discovery (the `SystemTopology` split: discovery vs dispatch-overlay) → a new `fuel-hardware` crate (B0.2); the **backend-contract traits** (the `DynBackendStorage`/`DynBackendDevice` dyn-backend pair, `HostStorage`/`BackendStorage`/`BackendRuntime`/`BackendCapabilityProvider`, the quantized `DynQuantizedStorage`/`QuantizedDeviceKernels`, and `InplaceOp1/2/3`) + the type-erased `Storage` handle → a new `fuel-backend-contract` crate (B0.3, done 2026-06-27 — sits above fuel-ir, below the backends; capability *data* types stay in fuel-ir); the CPU SIMD primitives (`VecOps`/`vec_dot`/`erf`) → a new `fuel-cpu-kernels` leaf crate with the `WithDType: VecOps` supertrait **dropped** (B0.5, done 2026-06-27 — fuel-ir is now a pure-vocabulary leaf); the dispatch overlay stays in `fuel-dispatch`. **B0.1–B0.5 COMPLETE.** The Storage-unification (merge the closed-enum `fuel_memory::Storage` with the moved `Box<dyn DynBackendStorage>` handle) is carved out as **blocked on eager-dispatch retirement (B6)**, not B0. Sequenced inside the cleanup (Step B0) because both names collide on crates.io before publish. See the memory `fuel-core-retirement`.

3. **Phase D — symbolic extents + persistent decode. PERSISTENT DECODE SHIPPED 2026-06-30.** Foundation (SymId/SymEnv/DynScalar/Extent; WriteSlice runtime offset; FlashAttn runtime `k_len`) was already on main + survived the dispatch-core cleanup. The ~1.8x/token plan-once decode is now landed + CPU-verified byte-exact: **D1** input-independent LlamaModel decode graph (`4585b194`) → **D2a** optimize-skip bridge seam `realize_one_prebuilt_env` (`2c19a3b7`) → **D2b** held `DecodeSession` + per-token data re-bind (`ca518525`) → **D2c** wired into the generate loop (`9abdaf50`) → **D4** PhiModel plan-once port (`d8f0bf3e`) → **D3** concurrency isolation proven (`28636185`). Both LlamaModel + PhiModel plan ONCE per generation and re-realize per token (optimize runs once, each token bit-exact vs the replan path, concurrency-isolated per-session; fuel-core --lib 1291/0). Remaining: the ~1.8x **GPU wall-clock benchmark** (manual, realistic model + live GPU — CPU bench showed 1.87x indicative); D2d (optional per-token `insert_safety_copies` skip, profile-gated); **step 2 = the CUDA flash decode arm** (via baracuda `flash_decoding` / FD++ — the FA2 `B·Hkv==1` constraint is OBSOLETE in alpha.72; see memory) + optional fuel-internal Vulkan FlashDecoding, both gated after this. Spec: `docs/session-prompts/symbolic-extents-and-persistent-decode.md` + `docs/session-prompts/phase-d-persistent-decode.md`; memory `phase-d-symbolic-extents`. **⚠️ UPDATED 2026-08-14 — THIS ITEM DESCRIBED A TWO-MODEL CAPABILITY FOR SIX WEEKS AFTER IT BECAME AN ELEVEN-FAMILY PROGRAM. Persistent decode is no longer `LlamaModel` + `PhiModel`: GAP-029 generalised it into a shared seam and ported the causal-LM families onto it, and the RULED SCOPE IS NOW COMPLETE.** `fuel-core/src/persistent_decode.rs` holds the shared rebind driver (`build_decode_graph` / `DecodeBackbone` / `MaskPlan` / `RopePlan`), parameterised over D1-rebuild and D2-persistent tails. **Ported: Qwen2, Qwen3, Qwen3Moe, Glm4, Phi3, SmolLm3, Gemma3 — seven causal-LM families, plus the pre-trait `LlamaModel`/`PhiModel` pair that still uses `rebind_and_realize_prebuilt` directly. `impl PersistentDecodeModel` is exactly 7 workspace-wide.** **⚠️ THE TWO COUNTS RANGE OVER DIFFERENT CONSTRUCTS AND BOTH ARE TRUE — QUOTE THE CONSTRUCT WITH THE NUMBER: 8 of 11 on the `lazy_quantized_*` denominator (`docs/session-prompts/gap-029-persistent-decode-trait.md:5`), and 7 of 7 on the causal-LM scope the ruling actually resolved to. The remaining three are ALL carve-outs, not backlog:** LFM2 (per-layer STATE KIND — attention KV interleaved with ShortConv rolling windows — blocked on BOTH a per-layer state-kind axis on `DecodeBackbone` AND the multi-output/option-C node that gates Mamba autoregressive resume; **(b) gates (a), so building the axis first yields an axis nothing can use**), and T5/Whisper (encoder-decoder). **So "8 of 11" is not three-quarters of a queue — it is a finished scope plus three deliberately-deferred shapes.** **"step 2 = the CUDA flash decode arm" above is LIVE AGAIN and its premise changed: the arm, the spec and the graph CAN express a window — `DecodeFlashSpec` carries `window_size_left`/`window_size_right` as DISQUALIFIERS and `flash_decode_admissible` declines on them, tested. What cannot express one is the capacity-K kernel. The Fuel-side defect is that the OFFER SITE hardcodes `None`, which is true for `LlamaModel` and false for any windowed family — so all four windowed families currently lose the arm on their DENSE layers too. See docs/gaps.md GAP-194.** Registry: GAP-029 (program), GAP-098 (LFM2), GAP-194 (flash arm), GAP-195 (`sliding_window: None` is DENSE). **⚠️ AND A NEW FRONTIER ITEM, AUTHORISED 2026-08-15 BY CIRESNAVE EXPLICITLY AS A CAPABILITY WITH NO CURRENT CONSUMER: THE DERIVED-PRECISION PROGRAM — a precision bound COMPUTED from a verified primitive basis rather than declared by a contract author.** Fuel is unusually positioned for it because the prerequisites already exist for unrelated reasons: a **build-time-closed primitive `Op` basis (~36 variants)** and a **total, never-panic `decompose` (31 impls)** — the recipe principle. Verify the primitives against a higher-precision oracle, DERIVE a composition's bound by propagating those through the recipe, then bound a fused kernel against its own decomposition plus that derived bar. **NOT FROM SCRATCH — MEASURED 2026-08-15: the propagation model EXISTS (`jit_ingest.rs::advisory_ulp_band`: single exact op -> exact; exact-only region -> `n_ops-1`; transcendental region -> Sum of per-op ceilings + `(n_exact-1)`), and the per-op ceiling table is NOT Fuel-authored (`kiss_ops_vocab::Op::ulp_ceiling`, pinned). MISSING: a higher-precision oracle (no f64 `KernelInvoker`; `fuel-cpu-backend` has 60 `F64` refs so f32-vs-f64 on a primitive is the cheapest credible one) and a `GraphInvoker` to run a decomposition (`KernelInvoker` takes a `BindingEntry`; `decompose_via_recipe` returns a `NodeId`) — which must route through fallible `realize_one_as::<T>()`, never GAP-186's panicking accessors.** **⚠️ THE BLOCKING DECISION IS NOT A WIRING CHANGE: `advisory_ulp_band` is ADVISORY BY DESIGN, and its own doc pins why — *"linear ULP addition is a first-order model; cancellation-heavy regions can exceed the band and flag spuriously."* Promoting it to a VERDICT is a decision about what a derived guarantee may CLAIM.** Two further limits recorded so they are not rediscovered: fusion deliberately changes arithmetic and is often MORE accurate than its decomposition, so a large divergence does not say which side is worse without the derived bar; and this route can NEVER produce a `max_ulp: 0` claim, so byte-movers and exact ops still need direct verification. **SEQUENCED BEHIND BLOCKING WORK, and its true scope is unknown until GAP-207 lands — what remains unverified after the bit-stable seeding IS the set that needs an oracle.** Registry: GAP-096 (answered), GAP-207 (the seeding program); design detail in `docs/architecture/10-decisions-log.md` 2026-08-15.

   - **Plan-once *prefill* (exploratory; filed 2026-07-30 at CireSnave's direction via the Lightbulb consumer — file, don't schedule).** Decode gets plan-once free (`seq==1` → the `cached_len`-independence above). Prefill re-optimises per call because `seq` = prompt length varies per request (the persistent path explicitly drops the held `DecodeSession` and rebuilds for any `seq != 1`); Fuel has no prefill session and no shape-keyed plan cache today. Fixed-size **chunked prefill** would make `seq` constant (`=C`), so ONE cached plan could serve every chunk of every request. *Hypothesis*: a win only at **high request-rate × short-prompt** serving, where prefill re-optimise recurs per request and per-call compute is small enough for optimise to matter — negligible for long single prompts (the inverse of decode's profile). *Coupling (load-bearing)*: a chunked path that inherited Phase D's full-capacity-read + tail-mask shape would compute over `max_seq_len` for **every** chunk, multiplying the masked waste by chunk count — likely swamping the optimise it saves. So it wants **the runtime-`k_len` flash arm** (the Phase D tradeoff follow-up / step 2 above) to land FIRST — a sequenced pair, not independent; don't build the chunked path onto the wasteful shape. *Test-before-build gate*: measure optimize-time as a fraction of prefill wall-clock across (prompt length × request rate); build only where that fraction is material. *Consumer note* — **CORRECTED 2026-08-07 by the Lightbulb owner; the original claim was true of the project and false of the path that matters.** It read: "no new consumer-side machinery — Lightbulb already has fixed-size chunked prefill (`model/chunked_prefill.rs`) for TTFT bounding, so this is a second justification for it, not a new ask." Lightbulb does have that code, but it hangs off `ParallelModelManager` and is **candlelight-path only** — `model_runner.rs` passes `None` for its config at **both** call sites, so **the Fuel path has no chunked prefill at all**. The "no new consumer-side machinery" premise therefore fails: if plan-once prefill is built, the consumer needs chunking on the Fuel path first. Reading a capability at *project* granularity when the coupling is *path*-specific is what produced the error. *Status*: **held unscheduled pending measurement**, at the consumer's insistence — Lightbulb declined to estimate whether prefill re-optimisation is material against compute without paired-arm numbers (absolute values per arm, build profile, commit, hardware, window definition, not a ratio). Do not build on the hypothesis until those land.

4. **Tier-2 runtime fused-op registration (the JIT loop).** Envelope crate landed (`fuel-kernel-seam` / `fuel-kernel-seam-types`); FKC declarative patterns + structural matcher + CPU/Baracuda cost trampoline complete. The §10 runtime/JIT integration SHIPPED — via a different mechanism than this line once anticipated: the runtime-fused kernel sidecar was **FOLDED into the generalized binding key** (2026-07-08, [10-decisions-log](docs/architecture/10-decisions-log.md) — not a graph annotation and not an executor-time sidecar lookup), and runtime adoption landed as the Spec-B ingestion service + Increment-1 recipe-identity verification (item 6 below). Memories `fusion-recipe-principle`, `kernel-contracts-dlpack-program`.

5. **Self-describing storage + kernel contracts.** SType/Encoding/ScaleSpec + DLPack view + FDX sidecar complete; kernel-boundary frozen. Next: finalize Baracuda/Vulkane coordination replies, land in a coordinated session. Memory `self-describing-storage`. **FKC cost unification — Part A (2026-07-01; since landed — the cost model completed honest + contract-sourced 2026-07-04, see the Shipped ledger):** the placement cost model priced uncapped backends at ZERO — only the CPU auto-registered `BackendCapabilities`, so `ranker::cost::compute_static_costs` skipped any GPU candidate (`capabilities_for(gpu)` → `None`), leaving `static_cost` at the default zero; a GPU could then out-rank a real CPU cost and a CPU-pinned realize could spuriously spill onto an unseeded GPU (`Op::Alloc/Copy` "no CUDA/Vulkan storage in input cache" crash). Part A adds `dispatch::derive_backend_caps` (any-backend analogue of `default_cpu_caps` — derives `op_dtype_support` from the backend's registered kernels via `KernelBindingTable::iter_keys`, keyed on the OUTPUT dtype) + `register_derived_gpu_caps` (registers every non-CPU backend's derived caps at the `global_bindings` init boundary, after kernels register) + a CUDA `fill_unset_cost_for_backend` pass (Vulkan already had one). GPU candidates now price at flops + inbound-transfer (Layer-1's backend-agnostic FLOPs; per-backend throughput is Part C). Parts B (transfer-cost honesty) + C (per-backend throughput) landed with the 2026-07-04 completion ([10-decisions-log 2026-07-04](docs/architecture/10-decisions-log.md)). **FIRST PRODUCTION FKC CONSUMER (2026-07-02; since landed and generalized — the binding table is contract-sourced end to end across CPU/Vulkan/CUDA as of 2026-07-04, see the Shipped ledger):** the CPU elementwise-binary family (8 ops × 4 dtypes = 32 bindings) is now registered by IMPORTING its kernel contract (`docs/kernel-contracts/cpu/elementwise-binary.fkc.md`) in `register_cpu_kernels` — the hand-written `table.register(...)` calls for the family are DELETED. The importer resolves each `entry_point` through the production `CpuLinkRegistry` to the exact same wrapper fn-pointers (behavior-preserving: identical kernels + caps + cost; `kernel_source` now the contract's `"portable-cpu"` tag, precision the contract's audited claim). To make this safe, the `fkc` cargo feature was **REMOVED** — FKC is now unconditional core infrastructure (serde/serde_yml always compiled, `pub mod fkc` unconditional): once a family's hand-written path is deleted, a build that could disable the importer would silently lose the family, so no such gate is allowed to exist. Born-red test `global_bindings_registers_binary_family_from_contract`. (The family migration subsequently completed — all three real backends register from `docs/kernel-contracts/**`, 2026-07-04; see the Shipped ledger.)

6. **FKC contract verification + automatic kernel integration (2026-07-11, blocks CapturedRun 4b).** A CapturedRun executor build-out session (4b-γ through 4b-ζ, `81660c2e`..`68eed195`) got the CUDA-graph decode-capture mechanism working and, chasing its real-decode capture test through six consecutive placement blockers, found and fixed 85 CUDA kernel precision claims that had shipped `audited: false` (never verified) — plus one kernel, baracuda's `rope_apply_f32` (built specifically in response to an earlier Fuel request), that was never wired into Fuel's dispatch table at all despite already existing and shipping. Investigating why traced to a real, code-confirmed gap between `docs/session-prompts/kernel-contract-adoption-plan.md`'s design (§10 V-FKC-9: a non-reference contract may not ship placeholder/`UNAUDITED` precision; §11 step 6: a "ship → verify" claim-verification gate) and what `fuel-dispatch/src/fkc/validate.rs`/`precision.rs` actually enforce (a much narrower internal-field-coherence check that never rejects `audited: false`, and a migration-equivalence test that only ever checked new imports against old never-audited hand-written defaults — never against real kernel behavior). **CapturedRun 4b-δ/4b-ε are PAUSED, by explicit decision, until this is built**: a mechanism where a provider (baracuda) submits a kernel + FKC contract claims, Fuel automatically tests as many claims as it can (starting with `bit_stable_on_same_hardware`, generalizing this session's cuBLAS `determinism_audit` repeat-call protocol), and a kernel whose claims verify enters full rotation without a hand-wiring session per kernel. Full account + design sketch: `docs/session-prompts/capturedrun-4b-paused-pending-fkc-verification.md`. Decisions: `docs/architecture/10-decisions-log.md` 2026-07-11 entries. **INGESTION SERVICE SHIPPED (Spec B, 2026-07-14, merged to main via `capturedrun-4b-resume` @ `cbb2e289`).** The verify→adopt/reject mechanism this item called for now exists: `fuel-dispatch/src/jit_ingest.rs` — a source-agnostic `IngestionService` (bounded queue + one idle-aware verify worker so it never swamps live inference) wrapping `verify_candidate` (probe synth → candidate CUDA invoke → bit-stability → reference realized from **Fuel's OWN registered recipe** via `reference_from_registered_recipe` (Increment 1, 2026-07-15 — supersedes Spec B's original "reference realized from the candidate's `decompose` via `reference_output`") → declared-precision-claim compare → fresh in-memory ledger) → `adopt_runtime_fused` on pass / `RejectionReport` + `ProviderFeedback::on_rejected` on fail. Landed the Task-1 lock-nesting fix with it (the pathfinder's availability gate reads the threaded binding table, so a background adopt's write can't deadlock the optimizer's read). 8 TDD tasks, whole-branch-reviewed; all 17 ingest tests green on the RTX 4070 (incl. a live e2e that drives the whole service). Design/plan: `docs/superpowers/specs/2026-07-13-jit-candidate-kernel-ingestion-spec-b-design.md` + `docs/superpowers/plans/2026-07-13-jit-candidate-kernel-ingestion-spec-b.md`; memory `spec-b-ingestion-complete`. **Carried limitations — one remaining, one RESOLVED:** (a) `verify_candidate` realizes its reference through Fuel's OWN backend and probes only `[-0.5, 0.5)` — exactly the oracle-independence + edge-corpus gap the "KISS interop-standard alignment" goals 3/4/5 target — **REMAINS, and is now registry-tracked as GAP-236 (Tier A) with the scope measured 2026-08-26.** **STRUCTURAL infidelity IS caught** — (b) below made the reference Fuel's own registered recipe plus a `recipe_identity_matches` base-map pre-check, and the interleaved-rope rejection is the shipped proof. **What survives is NUMERICAL infidelity inside a MATCHING structure:** `fmaxf` mis-lifted as a NaN-propagating `Max` claims `Max`, lowers to `Max`, agrees with Fuel's `Max` recipe on every input a `[-0.5, 0.5)` probe can reach, and passes. ⚠️ **LATENT TODAY — measured: nothing outside `fuel-dispatch` constructs a `CandidateKernel` or starts an `IngestionService`. It goes LIVE when a provider ships candidates, and CireSnave has APPROVED the Unpopped kernel-handback, so this is a PRECONDITION for that, not parallel work.** **Fix is coverage, not authorship: Fuel already vendors the KISS corpus (`fuel-dispatch/fixtures/kiss-corpus/`, schema `kiss-op-manifest-v1`, bit-exact vectors already `tags`-classified) — it holds 5 vectors over 1 op of 106 and ZERO NaN vectors, so the ask is a coverage extension on KISS plus consuming it here (which needs the documented `corpus_verdict` seam correction).** (b) **RESOLVED by Increment 1 (2026-07-15, merged to main @ `afc6ff32`)**: the reference is no longer the candidate's self-`decompose` (a self-certification hole) but Fuel's OWN registered recipe — `reference_from_registered_recipe` builds `Op::Fused(claimed_op)`, lowers via `lower_to_base_map`, and realizes the primitives (`3c10505e`/`a8c7b201`); recipe identity = `base_map_hash` equality (`f35e8e99`); `adopt_runtime_fused` is idempotent (`f4a43565`). The rope oracle now exists — a candidate claiming ROPE is verified against Fuel's rotate-half recipe and baracuda's interleaved `rope_apply` is REJECTED (`e0cf3c45`); the registered-recipe path IS the non-`PatternNode` reference path this item asked for. Memory `increment-1-recipe-identity-complete` (resolves `rope-not-patternnode`).

7. **Judge/ranker prerequisites for a high-breadth kernel supply — two additive changes that must land BEFORE the kernels do (2026-08-26, at CireSnave's direction).** ⚠️⚠️ **SUPERSEDED — THIS ITEM HAS LANDED (`d5ad733b`). READ NO FURTHER FOR SCHEDULING PURPOSES; the text below is the ORIGINAL ARGUMENT, kept because the reasoning is reusable.** **Measured at `origin/main` (`20657292`) by Fuel 3 and re-verified: `fuel-ir/src/dispatch.rs:84` has `pub const PROFILE_REPORT_VERSION: u32 = 5`, where this item's own scheduling argument reasons from *"already at 4"*; `kernel_revision_hash` is present with `#[serde(default)]` and documented as variant-granular with `0` = untracked; and the hardware-identity field carries the profiled device's `EquivalenceKey`.** **The `pre-v5` wording in those field docs is the giveaway: the bump this item argues FOR has already happened.** ⚠️ **WHY THIS MARKER IS INLINE AND NOT APPENDED: the item opens *"ONE LANE, BOTH, NOW"*, so a reader scheduling off its face value meets an imperative before any status.** The portfolio PM was holding it as schedulable and a lane was minutes from rebuilding shipped work. **A frontier document's most dangerous failure is not being wrong — it is being STALE AT ITS OPENER, where dispatch decisions are actually made.** Unpopped is scoping an expansion to a kernel per Nvidia compute capability, then the same breadth across recent Vulkan-capable GPUs. **The comparison machinery to choose among them ALREADY EXISTS and is wired into the live realize path**, at three layers: the **Judge** (`fuel-core/src/judge/`, the empirical racer — it enumerates the sibling kernels registered at a binding key via `direct_call_alternatives` and times each one **directly, bypassing dispatch**, skipping any that panic or return `Err` in warmup, recording latency *and* max relative error against a reference backend) → **Picker 1**, the plan-time ranker (`fuel-dispatch/src/ranker/`, ~8.8k lines: enumerate → hard-filter on precision → rank by composite cost = Layer-1 static estimate + Layer-2 `JudgeOracle::measured_latency_ns` → keep top-N) → **Picker 2**, the runtime selector (`pipelined.rs`'s `StreamingPick::selector`, documented in-file as *“the production `ChainedSelector` (VRAM-pressure guard + load tier + Judge rank)”*, consulted at **every branch** of the streaming walk, constructed from `judge::cached_oracle()` at `pipelined_bridge.rs:1631`). Criteria are `Fastest` / `MostAccurate` / `Balanced` with a tunable accuracy penalty. **So this item is not “build a racer.” Both changes below are the SAME defect at two different identifiers: the information is PRESENT at measurement time and DROPPED at the write.** **And both fail SILENTLY, in the direction that produces a plausible answer rather than an error — which is the whole argument for landing them before the kernels arrive rather than after.**

   - **(a) A persisted `ProfileReport` must identify the hardware it describes.** Measured at `98b6ecce`: the type is `ProfileReport { version, entries }`, and the load path validates **only `version`** (`fuel-ir/src/dispatch.rs:895`). `ProfileEntry` identifies its device as `backend` + `device_index: u32` — **an ordinal, not an identity.** Meanwhile the `DeviceDescriptor` the Judge is *holding at measurement time* carries `hardware_sku`, `vendor_id`, `device_id`, `compute_capability`, `subgroup_width` and `driver_version`, and CUDA really populates it (`fuel-cuda-backend/src/probe.rs:126` reads `dev.compute_capability()`). A report measured on the 4070 Laptop loads on the Radeon VII or the Arc B50 without complaint. ⚠️ **AND THE ORDINAL IS WORSE THAN NO IDENTIFIER AT ALL — Unpopped's point, from their own `structure_key`/`TargetId` scar: `device_index` is stable only within one machine's CURRENT enumeration order, so it rebinds across boots and under `CUDA_VISIBLE_DEVICES`, silently, to a different physical device. A MISSING identifier makes you go look; a REBINDING one answers confidently and wrongly.** **Scope:** stop dropping the descriptor's identity at the write; validate on load beyond `version`; refuse or re-profile on mismatch. **A born-red is cheap — a report saved under one hardware stamp must be REFUSED under another — and it must be RETAINED as a sabotage sibling, so the discrimination is re-proven on every run rather than once at authoring time.**

   - **(b) The Judge must key siblings on `kernel_revision_hash`, which every layer BUT the Judge already carries.** ⚠️ **THIS ITEM WAS SCOPED WRONG FIRST AND THE CORRECTION IS THE VALUABLE PART.** The original scoping said *“`kernel_source` is provider-granular; invent a variant-granular kernel identity”* — which would have invented a naming scheme (`unpopped-matmul-tile128x128-stage3`) that needs a registry, needs maintenance, will drift, **and overloads a field whose current meaning is correct**. Unpopped refused it and named the existing discriminator; measuring their claim against Fuel's tree made it smaller again. **As built:** Fuel's canonical kernel identity is `ImplId` — `(BackendId, op, dtypes, kernel_source, kernel_revision_hash)`, FKC §4.11, `fuel-dispatch/src/telemetry/impl_id.rs:21`, whose own doc says *“no new identifier is invented … every field already exists on the dispatch surface”*. `kernel_source` is **correctly** provider-granular on both sides (Unpopped emits `backend.provider()`; Fuel's production tags are `portable-cpu`, `cublas`, `baracuda`, `baracuda-generic-strided`, `mkl`, `slang`, `fuel-vulkan-kernels`). The sibling discriminator is `kernel_revision_hash`, **computed over the EMITTED SOURCE** (Unpopped `contract.rs:1793`, pinned by their `revision_hash_is_source_sensitive` test) — so forty tilings emit forty distinct source strings and get forty distinct hashes **automatically, content-derived, collision-free by construction, with nothing to maintain**. **And `BindingEntry` ALREADY HAS IT** (`fuel-dispatch/src/kernel.rs:838`, field `kernel_revision_hash: u64`), as does the FKC ledger key `(kernel_source, backend, dtypes, kernel_revision_hash)` (`fkc/register.rs:366`). **The one place it is missing is `ProfileEntry` — and `impl_id.rs`'s own module doc states the defect without flagging it: *“a telemetry record's impl id and the Judge's measurement key are the same `kernel_source` axis, by construction”*. Five fields of identity, one field of measurement key.** So the Judge reads binding entries that already carry the hash and drops it when it writes the measurement; forty variants collapse into one cell, the table keeps one, and the other thirty-nine are **invisible with no error** — the injectivity failure this repo has met at three other seams (*a distinction stated in prose must be carried in the type*). **Scope:** thread the existing `kernel_revision_hash` through `direct_call_alternatives` → `ProfileEntry` → `Pick` → `JudgeOracle::measured_latency_ns`. Additive, no new vocabulary, no registry. **Note `0` currently means “untracked” on that field, so the born-red must prove two DISTINCT NON-ZERO revisions produce two distinct cells — a test that only shows `0` vs non-zero would pass while siblings still collapse.**

   **Three adjacent findings, recorded so they are not re-derived.** **(i)** The profiled matrix is **40 distinct ops** and **does** cover `FlashAttn` — the module doc calling FlashAttn *“not yet profiled … slated for Phase 7.6”* is **STALE and was nearly relayed as current fact**. Still unprofiled: `PagedAttn`, `SoftmaxLastDim`, `RmsNormLastDim`, `LayerNormLastDim`, `Rope`, `FusedLinear`, `Conv2D`/`ConvTranspose2D`, `QMatMul` and the shape-movers. **Unpopped's read of that list is the strategically important one: those are where a generator wins OUTRIGHT — not by beating a hand-tuned kernel, but because no hand-tuned kernel exists for the cell. So “expand the profiled matrix” and “admit Unpopped candidates” are NOT two projects competing for the same time; the ops where racing buys least today are the ops where it would buy most once there is something to race.** **(ii)** Judge cost is ops × dtypes × sizes × devices × **siblings**, and that last factor is **~1 today**. There is no `incremental`/`resume`/`prune`/`budget` vocabulary anywhere in the Judge runner or `scheduling.rs` (**control: `latency_ns` returns 16 hits in the same file, so the zero is genuine absence rather than a broken query**). **Unpopped has taken bounding it as THEIR obligation rather than Fuel's, with a three-stage pipeline whose expensive stage is deliberately last — spec-legal enumeration (static, free) → compile-measured pruning (`ptxas -v` registers/spills/occupancy, cheap, no nvcc) → raced (survivors only) — and a committed target of 4–6 survivors per cell, not 40.** **Fuel should therefore NOT build budget/prune vocabulary defensively; if a cell ever needs more than that it is a conversation, not a default.** **(iii)** Neither (i) nor (ii) blocks (a)/(b).

   **Scheduling — PM-priced 2026-08-26: ONE LANE, BOTH, NOW, and the leading reason is the PERSISTED SCHEMA rather than the handback.** `ProfileReport` is persisted, and `PROFILE_REPORT_VERSION` is **already at 4** (`fuel-ir/src/dispatch.rs:67`); load rejects on mismatch (`:895`) and `judge/oracle.rs:60` documents that load *“already filters stale schemas”*. **The migration mechanism is proven, not theoretical — it has been bumped three times, and `oracle.rs` names one of those bumps as being for EXACTLY this ambiguity: *“pre-v2 reports were ambiguous about which kernel sibling they timed.”* A version bump for sibling ambiguity has already happened here once; (b) is the same bump finished.** ⚠️ **So (a) is RETROACTIVE, not merely preventive: every profile written to date is keyed by an ordinal, so any written on a multi-GPU box or on a different machine MAY ALREADY BE WRONG and is indistinguishable from a right one. Bumping the version DISCARDS them, which is the correct outcome.** **That argument does not depend on Unpopped's schedule at all — it would hold with no provider in the picture — which makes it stronger than the sequencing argument it replaces.** The sequencing argument remains the second-best one, and is worth keeping in the provider's own words: *“I would rather wait for both than be the first provider through a gate that cannot tell my forty variants apart.”*

   **No contention with the restructure, and this is counter-intuitive enough that nobody should accept it without looking.** All three racer layers live in crates the restructure moves — `fuel-core/src/judge/` (3 files), `fuel-dispatch/src/ranker/` (20 files), `fuel-dispatch/src/pipelined.rs`. **But the SCHEMA does not: `PROFILE_REPORT_VERSION`, `ProfileEntry` and `ProfileReport` are all in `fuel-ir`, the vocabulary crate the restructure KEEPS.** **The racer's CODE is entirely in the moving half and its TYPE is not**, so the two fields land in a crate that is staying and the consumers merely relocate around them.

   ⚠️ **Two sizing corrections, both in the direction of MORE work than “two fields” suggests.** **(1) Price it as 24 sites, not 2 fields.** `ProfileEntry {` has **24 construction sites** at head — `scheduling.rs` 15, `judge/cache.rs` 4, `judge/mod.rs` 2, `judge/oracle.rs` 2, `fuel-ir/src/dispatch.rs` 1. Most are test fixtures, which is cheap but not free. ***“Two fields” is true of the STRUCT and false of the CHANGE.*** **(2) Scope (a) as ADDITION beside `device_index`, NOT replacement — and say so in the allocation, or it gets discovered mid-flight.** `device_index` has **95 references across 20 files**, including both probe crates and `fkc/verify/seed_vulkan_ledger.rs` (GAP-236 territory). **A lane handed “fix the device identity” could reasonably read it as the much larger job of retiring the ordinal. Name the boundary.**

   ⚠️ **Both born-reds MUST drive the PRODUCTION path, not the validator directly — and the natural way to write them is the wrong way.** **(a)** write a profile under hardware descriptor X, load it under Y, and require the DECLINE to come out of `ProfileReport::load`. **(b)** two variants differing only in `kernel_revision_hash` must occupy TWO cells **via the Judge's real cache path**. **The trap is live and freshly worked: KISS #344 carried nine controls that called `check_rfc_collisions` directly, none of which proved the wiring — deleting the two-line dispatch loop left ALL NINE GREEN.** Same shape is available here, on two failure modes that are both silent. **Report the red before the green on both.**

   ⚠️⚠️ **CORRECTION TO THIS ITEM'S OWN LEADING ARGUMENT — “the migration mechanism is PROVEN, not theoretical” IS TRUE ONLY FOR ADDITIVE-WITH-DEFAULTS CHANGES, and I asserted it to two projects before checking (found by `77d0a1cx` while implementing, 2026-08-26).** **DESERIALIZATION RUNS BEFORE THE VERSION CHECK** (`fuel-ir/src/dispatch.rs:893-896`): `serde_json::from_slice::<Self>(&bytes).map_err(…)?` — **an `Err`** — sits above `if report.version != PROFILE_REPORT_VERSION { return Ok(None) }`. So the doc's promise (*“Returns `Ok(None)` on a missing file or schema-version mismatch”*) **holds only for reports that still deserialize**: any non-additive change makes an old report `Err` at parse and **never reaches the gate that exists precisely to handle it.** **The three prior bumps worked because the fields they added carry `#[serde(default)]` — the version gate was not doing the work, the defaults were.** **What survives:** *“(b) is the v2 bump finished”* (unaffected) and *“(a) is retroactive”* (unaffected). **What weakens: the stated BASIS.** It is proven for the shape of change this item makes — both new fields are additive and carry defaults — and it is **not** the general reassurance it was offered as. **FOURTH INSTANCE OF THIS ITEM'S SHAPE, and this one is the architect's own: a true justification attached to a wider claim than it supports.**

   ⚠️ **AND THE `serde(default)` FIX REOPENS THE SENTINEL AT A PATH NOBODY EXAMINED — required test, do not drop it.** `#[serde(default)]` on `kernel_revision_hash` means **a v4 report DESERIALIZES with `kernel_revision_hash: 0`** — the untracked sentinel, arriving through defaults rather than through a parse failure. **It is safe ONLY because the version check then returns `Ok(None)`, and nothing asserts that.** **Required:** pin that **a v4 report produces NO oracle entries — not entries carrying `0`** — asserted on the `load`/oracle result, never on the deserialized struct. **The natural future cleanup — *“check the version before deserializing, it's cheaper”* — turns those defaults into real `0`s in the measurement key and silently collapses every v4 sibling into the untracked bucket.** Note the same reasoning forbids a blanket default on `device: EquivalenceKey`: a zero-valued hardware identity would match nothing and refuse everything — loud rather than silent, but still a fabricated identity. **Apply `serde(default)` per field, having looked at what the default MEANS, never uniformly.** **This is Unpopped's sentinel principle firing at the seam this item had already declared clean two hours earlier: *a sentinel is safe only if EVERY path that can produce it is deliberate* — we both read that as being about parse failures, and it was also about DEFAULTS.**

   ⚠️ **(a) DOES NOT ADD A PRECONDITION — IT RESTORES ONE THE RANKER ALREADY ASSUMES, and that is the accurate commit message (found by `77d0a1cx` while scoping, 2026-08-26).** `judge/oracle.rs:34-39`: *“The trait key carries no device axis; when a report holds the same cell measured on multiple devices … the adapter keeps the **MINIMUM** latency — **‘the best this backend has demonstrated’** — deterministic regardless of entry order.”* **That semantics is documented, deliberate and SOUND — under an invariant nothing currently enforces: that every entry was measured on THIS machine.** Once a report can arrive from elsewhere, *“the best this backend has demonstrated”* silently becomes *“the best this backend demonstrated anywhere, on hardware you may not own.”* **It also settles the load-time design question: whole-report-REFUSE, not filter-to-matching.** A `ProfileReport` is the output of ONE Judge run on ONE machine — if any device identity in it is foreign, the run happened elsewhere, so the entries that DO match local hardware are not a valid subset but **a foreign report with coincidental overlap.** (Filtering is in fact safe for the min; what it leaves is a partial matrix whose holes are silent — permanently degraded ranking with no signal, which is the degraded-window failure made permanent.) **THIRD INSTANCE OF THIS ITEM'S SHAPE, and the pattern is now the item's real subject: `EquivalenceKey`'s own doc calls it *“the key the Judge uses to share a profile across identical devices”* (`fuel-ir/src/probe.rs:172`) and THE JUDGE DOES NOT USE IT FOR LOOKUP AT ALL. Identity exists; the measurement key does not carry it — exactly as with `kernel_revision_hash`. The `judge` binary has been printing `probe.equivalence_classes().len()` on every run the whole time.** **Design confirmed for the lane:** embed `EquivalenceKey` (same crate as `ProfileEntry`, already `Hash`/`Eq`/serde, relation already tested BOTH directions at `probe.rs:232`/`:256`; the duplicated `backend` field is worth keeping because the redundancy is then CHECKABLE); gate inside `ProfileReport::load` itself rather than a separate `validate_against` a caller must remember; return `Ok(None)` like the stale-schema path, because a foreign-hardware report is benign rather than erroneous; compare against `equivalence_classes()` (`fuel-hardware/src/probe.rs:103`) rather than raw descriptors, so an identical replacement GPU does not invalidate. **Note `driver_version` is IN the equivalence key by an explicit prior ruling — *“invalidating on driver upgrade is cheap insurance”* — so re-profiles are more frequent than they look; that is settled, not open.** **Re-profile entrypoint: `fuel-lazy-examples/src/bin/judge.rs` (`cargo run --release --bin judge -- <path>`). DEFAULT FEATURES ARE CPU-ONLY — no GPU, no `gpu-run`. `--features cuda` touches the 4070 (so: `gpu-run`) AND links, i.e. a full ~56-minute baracuda forge, detached with a self-written exit-code marker. Recommended split is by COST, not correctness: CPU baseline re-profile in the same PR (it proves the write→load→refuse→accept round-trip on real output); CUDA/Vulkan re-profile as a separate scheduled run.**

   ⚠️ **THE BUMP HAS A DEGRADED-RANKING WINDOW, AND NOTHING ABOUT IT IS AN ERROR — raised by Unpopped, and it follows from two facts already in this item rather than from anything new.** A version mismatch produces an **EMPTY oracle** (`judge/oracle.rs`: *“every lookup misses → Layer-1 static costs stand”*), and the ranker's composite cost is **Layer-1 static + Layer-2 Judge measurement**. **So bumping `PROFILE_REPORT_VERSION` removes the Layer-2 term until the Judge re-runs.** Between the bump and the first re-profile, selection falls back to static estimates alone — *correct* behaviour, correctly discarding suspect data, **but a window of degraded ranking quality that reports nothing.** Whoever lands this must plan the re-profile as part of the change, not as a follow-up, and say in the commit that the window exists. **AND THE SAME OBSERVATION IS AN INDEPENDENT ARGUMENT FOR DOING IT NOW, distinct from both the retroactive one and the sequencing one: the forced re-profile is a BASELINE run — no providers, siblings ≈ 1 — which is the cheapest that run will ever be. Every month it waits, and certainly once candidate kernels exist, the same unavoidable re-profile costs strictly more.** **That reason depends on no provider's schedule, which by this item's own standard makes it better-shaped than the sequencing argument it supplements.**

   **Sentinel check on `kernel_revision_hash: 0` — RAISED by Unpopped, MEASURED CLOSED at Fuel's boundary, principle RETAINED because threading into `ProfileEntry` could reintroduce it.** Their concern: the value crosses the repo boundary as `u64 → String → u64` (their wire form is a `String`), and `0` sits at the bottom of every failed conversion on the return leg — so *“untracked”* and *“I could not read it”* would share a bucket, the `DeclinedOp::Access` shape this workspace spent a release removing. **Measured at head, Fuel's boundary is already fallible and `absent` never becomes `0`:** `fkc/revhash.rs::compute_revision` takes `None | Some("auto")` ⇒ **compute** FNV-1a over `entry_point ++ revision_base ++ canonical-block`, and `Some(hex)` ⇒ `u64::from_str_radix(cleaned, 16).map_err(|_| FkcError::Yaml(…))` — **a hard error, not a default**. **No `unwrap_or(0)` exists on any revision path** (*control: `unwrap_or(0)` appears in 60 files, so the query finds it where it exists*), and **all 12 sites stamping a literal `0` are test fixtures, zero in production** (*instrument caveat, stated rather than smoothed: a site is classified test if any `#[cfg(test)]` appears earlier in the file — conventional in Rust, not a proof*). **The general form is worth keeping regardless of this instance: a sentinel is safe only if EVERY path that can produce it is deliberate. `0 = untracked` plus an `unwrap_or(0)` anywhere in a parse path silently merges “untracked” with “conversion failed” — `unwrap_or(0)` is a warning nobody reads, spelled as a number.**

8. **The model-construction layer is missing, and every model pays for it — measured 2026-08-27, surfaced by CireSnave asking how a developer loads a NEW model.** **02-layers v0.5 ratifies exactly this layer as `fuel-model-core` (the `Model` trait, the `model_type`/`general.architecture` → builder registry, `AutoModel::from_path`, link-time distributed registration). THE CRATE DOES NOT EXIST**, and neither does any part of what it specifies. **Measured at `90cb5265`, with controls:** `trait Model` **0** · `AutoModel` **0** · registry / `inventory` / `linkme` **0** *(control: 100 files contain `trait `, so the query finds traits where they are)*. **What exists instead is the hand-rolled cross-product:** **399** distinct `*Weights` structs · **148** distinct `*Config` structs · **27** separate definitions of `from_hf_json_str`, all in `fuel-core`. **`LlamaFullConfig` + `build_llama3_model` are not an EXAMPLE of the pattern — they ARE the pattern, 148 times over.** ⚠️⚠️ **SUPERSEDED 2026-09-02 AT `292ef3df` — THE CENTRAL CLAIM ABOVE IS NOW FALSE AND EVERY NUMBER IN IT IS STALE. `fuel-model-core` EXISTS** (built at `96e1c5de`): `trait Model`, `AutoModel::from_path`, and an `inventory` registry are all present. **The premise moved TWICE — the crate was built, and Stage 2 then relocated the duplication it was written about.** Re-measured at head, constructs named: **`pub struct *Weights` DEFINITIONS 439** (was 399) — **`pub struct *Config` DEFINITIONS 169** (was 148) — **`fn from_hf_json_str` DEFINITIONS 34** (was 27), of which **30 are now in `fuel-transformers` and 4 in `fuel-core`**, because Stage 2 moved 146 model files out. **A number attached to `fuel-core` in the sentence above is a number about a crate that no longer holds the subject.** ⚠️ **AND THE CRATE EXISTING IS NOT THE CRATE WORKING: `impl Model for` yields **3 TYPES, AND THEY ARE SCAFFOLDING BY MEASUREMENT RATHER THAN BY CHARACTERISATION: `LocalModel` (`fuel-model-core/tests/registry.rs:43`), `FixtureModel` (`fuel-model-fixture/src/lib.rs:26`), `UmbrellaModel` (`fuel-model-umbrella-fixture/src/lib.rs:29`)** — **one lives in a `tests/` directory and two in crates whose names END IN `-fixture`.** **ZERO production models implement it and the zoo is not wired.** **TRIAL, run by Fuel 1 2026-09-02 (throwaway branch, nothing landed): `impl Model` + a `build(&ModelSpec)` for Qwen2, trait and model unmodified. IT COMPILES CLEAN, AND THAT IS THE TRAP RATHER THAN THE RESULT.** ⚠️ **`trait Model` is ONE method — `architecture(&self) -> &str` — so the impl is a hardcoded string literal and it compiles for ANY model. A test that supplies the value it claims to check tells you nothing about what production computes.** ⚠️⚠️ **THE LOAD-BEARING FINDING: `build()` COMPILES AND IS SEMANTICALLY BROKEN.** `ModelSpec` is `{path, architecture}` only. `Qwen2Config` derives `Debug, Clone, PartialEq` — **no `Deserialize`, no `from_hf_json_str`** — and is constructible ONLY from hardcoded presets (`qwen2_7b()`). **So the builder cannot parse `spec.path/config.json`; it is forced to a preset, IGNORING THE ARTIFACT IT WAS HANDED — correct for a 7B checkpoint and wrong for every other Qwen2. It is GREEN AND UNABLE TO HONOUR ITS OWN INPUT.** *(Confirmed independently; note a path glob `*qwen2*` hits `lazy_qwen2_moe.rs`, a SIBLING that does have a parser, and reads as a refutation. Control: bert returns 4 hits.)* **NON-UNIFORM: ~30 of 146 models have a parser; ~116 including Qwen2 do not.** ⚠️⚠️ **AND THE FACT THAT DECIDES THE PROGRAM — FUEL HAS TWO MODEL TRAITS WITH OPPOSITE DEFECTS.** `fuel-model-core::Model` is **UNIVERSAL AND EMPTY** (1 method, 3 fixture impls). **`fuel-inference::multi_session::DecodeModel` (`:79`) is CAPABLE AND NARROW** — it carries `n_layers`, `layer_state_specs`, **`forward_with_kv_context_persistent`**, `supports_batched_decode`, `build_batched_decode_logits` — and **`impl DecodeModel for` yields 5 TYPES against a zoo of 434.** **`Box<dyn Model>` is INFERENCE-DEAD: `forward` lives on the concrete type by design (*"forward IS the architecture"*), so `AutoModel::from_path` returns a handle you can only ask `architecture()` of, and running it requires a downcast — which is the match statement the registry was meant to remove, relocated into a type.** ⚠️ **Universal-and-capable is NOT AVAILABLE, because model inputs genuinely vary across encoder / vision / multimodal. That is a constraint, not a gap.** **RULED 2026-09-02. (I) IS NOT THE SEAM — item 8 no longer proposes wiring the zoo to it.** **(II) CONFIG-FROM-PATH IS THE WORK AND IT STANDS ALONE:** the `from_hf_json_str` convention is proven on 30 models, the remaining ~116 are mechanical and gateable, and the value does not depend on `Model` existing. **Build it as a capability of THE CONFIG TYPES, never as a service to a registry, so it survives whatever happens to (I).** **(III) `WeightSource` sequences behind (II).** **The born-red is Fuel 1's own broken case: a builder that ignores `spec.path` and returns a 7B preset must FAIL ON VALUE against a real non-7B `config.json`** — a POSITIVE golden, not a `≠ preset` relative oracle, since the latter passes for garbage that merely differs. **Assert `num_key_value_heads` explicitly: a parser defaulting it to `num_attention_heads` yields a structurally different model while every derived accessor still returns something plausible.** **EMPIRICAL POSITIVES, measured rather than assumed:** `Model: Send + Sync` **HOLDS** for a Tensor-holding model — the uncertain break did not occur; and the **orphan rule forces the impl into the model's own crate**, so wiring lands in `fuel-transformers` or a per-model leaf, never a standalone wiring crate (`E0117`). **NOT VERIFIED and stated as such: the bert (parser-present) case, and encoder / vision / multimodal / GGUF fit.** **The `96e1c5de` linkage defect — registration needs a link-forcing line — is orthogonal and stands.**

   **Question 1 — how does a developer instantiate a NEW model, with no weights yet?** The **primitives** exist: `fuel-nn/src/varmap.rs` (`VarMap`), `fuel-nn/src/modules/init.rs` (`xavier_uniform`, `kaiming_uniform`), tensor-level `randn`. **The entry point does not.** A model does not take a *weight source* — it takes a **concrete `*Weights` struct**, and those are built by `from_safetensors_bytes` / `from_safetensors_view`. **So training from scratch means hand-constructing your `*Weights` struct field by field with freshly initialised tensors; there is no `Weights::init(&config)` convention.** *(Scope of the check, stated: this is the aggregate — no init/random constructor convention across the population. All 399 structs were NOT inspected individually, so a one-off may exist.)*

   **Question 2 — what should be generic?** Three of the four layers, and the fourth is precisely what the authoring surface is for. **(i) Config loading — YES.** The 27 parsers read mostly the same HF names (`num_hidden_layers`, `hidden_size`, `num_attention_heads`, `vocab_size`, `rms_norm_eps`, `rope_theta`). A common core plus a per-model extension is **serde work, not architecture work**. **(ii) Weight binding — YES, the MECHANISM.** 399 structs each hand-map `model.layers.N.self_attn.q_proj.weight`. **The names are per-model; the prefix-scoped-lookup mechanism is not** — that is what a `VarBuilder` is for, and Fuel has one but does not use it at the model boundary. **(iii) Registry + `AutoModel` — YES, entirely.** Pure boilerplate, already specified in 02-layers. **(iv) `forward()` — NO.** That is the architecture, and it is the thing the authoring surface addresses.

   **THE ONE CHANGE THAT ANSWERS BOTH QUESTIONS: a model must take a weight SOURCE, not a weight STRUCT.** `LlamaModel::new(&cfg, LlamaWeights::from_safetensors_view(…))` — concrete, file-shaped, 399 of them — becomes `LlamaModel::new(&cfg, source)`, where a source is a safetensors view **or** GGUF **or** fresh initialisation **or** (once it exists) a `.fuel` file. **From-scratch training then costs nothing extra — same builder, different source — which turns Question 1 from a per-model chore into a non-question.** **And it is the prerequisite for the authoring surface to be worth building:** a language that generates model definitions needs a stable layer to generate *against*, and 399 bespoke structs is not one. **`399 + 148 + 27` is the measured price of not having this layer**, and it is the strongest concrete justification for the authoring work that has been produced so far.

   ✅ **RATIFIED BY CIRESNAVE 2026-08-27 — *“I agree with you. Let's get this on Fuel's roadmap and work with the PM about when to schedule it.”* This is no longer an architect's finding; it is agreed work awaiting a slot.** ⚠️ **AND A MEASUREMENT THAT MAKES IT CHEAPER THAN THE PROPOSAL READS.** The recommendation was *“use a `VarBuilder`, which Fuel has but does not use at the model boundary.”* **Measured, the situation is better than that:** `VarBuilder` lives at `fuel-nn/src/varbuilder.rs` and is consumed by `fuel-nn/src/modules/linear.rs` — and the **only** mentions inside model files are **three doc comments**, e.g. `lazy_snac.rs:738` *“Tensor names mirror the eager `Model::new` **VarBuilder tree**”*, `lazy_mimi_transformer.rs:354`, `lazy_mimi_quantization.rs:447`. **So the models ALREADY CONFORM to VarBuilder's naming discipline and document that they do — they simply re-implement the prefix-scoped lookup by hand, 399 times.** This is therefore **not “adopt a new abstraction”; it is “wire up the one whose convention the corpus already follows.”** The hard part of such a migration — agreeing a naming scheme across 399 structs — **is already done and is load-bearing in the doc comments.**

   **DECOMPOSITION — three increments, independently landable, cheapest gate first.** **(I) The registry + `AutoModel` = create `fuel-model-core`.** Pure boilerplate, fully specified in 02-layers v0.5 (link-time distributed registration via `inventory`/`linkme`). **Independent of the other two, unblocks nothing and is unblocked by nothing** — the cheapest way to make the ratified layer stop being fictional. **(II) The config core.** 27 hand-rolled `from_hf_json_str` → a common core (`num_hidden_layers`, `hidden_size`, `num_attention_heads`, `vocab_size`, `rms_norm_eps`, `rope_theta`) + a per-model extension. **Mechanical serde work with an unusually good gate: a DIFFERENTIAL against the existing 27** — every real `config.json` in the corpus must parse to the same values through both paths. **Born-red is free: perturb one field name and watch the differential fail.** **(III) The weight-source abstraction.** The keystone, and the one that makes from-scratch training free. `WeightSource` + impls for safetensors-view / GGUF / fresh-init, then model constructors migrate from concrete `*Weights` to the source. **399 structs, so this is incremental by construction — introduce the source, adopt one model, then sweep.** Do **not** attempt it as one change.

   ⚠️⚠️ **CORRECTED PRE-BUILD BY `nduiwcsu`, 2026-08-27 — THE TARGET IS 7, NOT 27, AND MY QUERY WAS THE LOOSE ONE.** **The decomposition sums exactly, which is what proves it rather than fits it:** `grep 'fn from_hf_json_str'` → **27** *(what I ran)* · `grep 'fn from_hf_json_str('` → **13** *(actual definitions)* · `grep 'fn from_hf_json_str_'` → **14** *(TEST functions, named after what they test — `..._parses_canonical_gemma2_2b_fields`, `..._bails_on_qk_layernorm_true`)*. **13 + 14 = 27.** **This is the delimiter trap from this repo's own working agreement, at the end that keeps being forgotten — the boundary AFTER the identifier.** **Then a second halving: of the 13, SIX are already `#[derive(Deserialize)]` with a 4-LOC body** (`serde_json::from_str::<Self>(s).map_err(…)`) — **verified individually, 6/6.** **HAND-ROLLED (`serde_json::Value` + `get_*` closures), 7 parsers / 365 LOC — the real target:** `DebertaV2Config` · `Gemma2Config` · `Llama2cConfig` · `LlamaConfig` · `LlamaFullConfig` · `PhiConfig` (lazy.rs) · `PhiConfig` (lazy_phi.rs). **ALREADY SERDE, 6 / 25 LOC:** `BertConfig` · `ClipTextConfig` · `ConvNextConfig` · `HFLlavaConfig` · `Qwen2MoeConfig` · `WhisperConfig`. **THIS MAKES THE INCREMENT BETTER, NOT MERELY SMALLER: the six ARE the destination pattern.** The work is not *“invent a config core”* but *“make the remaining seven look like the six that already work this way”* — **with in-tree precedent and a 25-LOC exemplar.**

   **AND THE OTHER TWO NUMBERS WERE TESTED FOR THE SAME DEFECT AND PASS — which is the part that makes this diagnosable rather than a reason to re-derive the item.** `399` *(438 definition lines → **399 DISTINCT NAMES**)* · `148` *(150 lines → **148 distinct**)* · `147 lazy_* files` *(**147**)*. **The text says “distinct” and the measurement used `sort -u` with a `\b` boundary; only the `from_hf_json_str` query was built differently.** *(Corollary worth keeping: 438 lines against 399 names means ~39 `*Weights` names are defined more than once — the same shape as the duplicate `PhiConfig` below.)*

   ⚠️ **THE PROPOSED CORE FIELD LIST WAS 4-OF-6 RIGHT, and the measured one is different in both directions.** Against the **7 hand-rolled** parsers: `num_hidden_layers` **7/7** · `hidden_size` **7/7** · `num_attention_heads` **7/7** · `vocab_size` **7/7** · **`rms_norm_eps` 4/7 — NOT common** (absent from `DebertaV2Config`, `PhiConfig`) · **`rope_theta` 6/7 — NOT common** (absent from `DebertaV2Config`). **The keys actually read by ALL SEVEN:** `hidden_size` · **`intermediate_size`** · `num_attention_heads` · `num_hidden_layers` · `vocab_size`. **`intermediate_size` is genuinely common and was NOT in the proposal; `rms_norm_eps` and `rope_theta` are not common and WERE.** Written to the proposed list, two of six core fields would be `Option` or per-model from day one and a field that could have been core would not be. **Use the measured list.**

   ⚠️ **THE GATE'S CORPUS DOES NOT EXIST AS FILES — `find . -name config.json` → ZERO.** The de-facto corpus is **inline JSON inside the 14 test functions**, covering 8 of 12 types. **RULED: differential over what exists, with the boundary stated in the gate's OWN OUTPUT and the uncovered types named there** — not invented fixtures. **A fixture invented for an uncovered type is evidence that the new core agrees with the old parser ON THE AUTHOR'S GUESS, not on a real config** — an oracle that certifies a guess, which is the vacuous-fixture failure with extra steps. **A gate that names its own uncovered set is honest; one that fabricates coverage is not.**

   ⚠️ **`PhiConfig` EXISTS TWICE — `lazy.rs:12439` and `lazy_phi.rs:375`, 52 and 56 LOC.** Whether that is one type with two parsers or **two types sharing a name** must be answered before either is folded into a core. **This is GAP-209's exact shape — two functions, one signature, potentially different domains — surfaced inside the increment whose sequencing that very incident dictated.** **RULED: read the CALLERS, not the signatures.** Per the standing rule: two functions with one signature and different domains are either an accidental divergence (dedup to the superset) or **two distinct obligations sharing a shape, which must not be merged at all** — and nothing in the code distinguishes them; only the callers' requirements do. **If resolving it is its own question, it is out of scope for (II): defer it, record it, and fold neither until it is answered.**

   ⚠️ **SEQUENCING AGAINST THE RESTRUCTURE — do NOT combine this with migration Stage 2.** Stage 2 moves **146 of the 147** `lazy_*` files into the Models tier (`lazy_latent_cache` stays; see item 11) and is **mechanical** (change crate, fix imports, path-set unchanged). Item 8 is **semantic** (change what a constructor accepts). **Combining them produces one enormous change in which a reviewer cannot tell a move from a rewrite** — and this session has the worked example: a “pure move” refactor that rebound a call to a same-signature helper with a narrower domain, compiled green, and orphaned 228 ledger records (GAP-209). **A move and a semantic change in the same diff is exactly the shape that defeats review.** **Order: Stage 2 first, then item 8** — so the semantic work happens in a crate that is not simultaneously moving, and each model is touched once by each kind of change rather than once by both at the same time. **Increments (I) and (II) do not touch the model files at all and can proceed in parallel with anything.**

   **How this composes with the two adjacent gaps, because all three are faces of one thing.** A "model" in Fuel today is a **triple**, and only two parts are files: **hyperparameters** (HuggingFace `config.json`) · **weights** (`*.safetensors` / GGUF, mmapped) · **the architecture** (**compiled-in Rust — not a file at all**). **`fuel-formats/src/` contains six modules — `ggml`, `gguf`, `imatrix`, `pickle`, `safetensors`, `lib` — every one a FOREIGN format. Fuel has no native model file.** 11-persistence v1.3 ratifies `.fuel` as *“the primary artifact … base map + storage + optimized paths”*; **measured, it does not exist**: 0 base-map serialization sites *(control: 12 files mention `base_map`)*, 0 `derive(Serialize)` in `fuel-graph` *(control: 14 `derive(` in `lib.rs`)* — **`Graph`, `Node` and `Op` are not serializable at all, so there is nothing to write even if a writer existed.** That is doc-vs-code drift on a core claim. **The consequence, and it is the real cost: a model is a COMPILE-TIME DEPENDENCY of everyone who runs it.** “Add a model to Fuel” means write Rust, compile it in, and every consumer rebuilds. **So: the authoring surface decides how the architecture gets WRITTEN · this item decides what it is written AGAINST · and the native format decides whether it can be SHIPPED AS DATA rather than linked.** Only the third has a ratified name (`.fuel`) and none of the three has an implementation.

9. **The constitution names surfaces that were never built, and treats some of them as SHIPPED — measured 2026-08-27, and this is the direct answer to CireSnave's question *“are we detouring around things the constitution already decided?”*** **A mechanical doc-vs-code audit of all 15 `docs/architecture/` files** (`10-decisions-log` excluded — it is a log, so a dated status line there is the log working) **enumerated every artifact the docs name and checked existence at head.** **Raw 103 absences → 69 after correcting a ~33% false-positive rate**, with the false positives named by construct rather than silently dropped: adjectives (*“the format is fuel-specific”*), file extensions (`model.fuel-tolerance`), a template placeholder, a doc filename, **glob stems** (`fuel-format-*` stripped to `fuel-format`, which is satisfied by its leaves rather than missing), and hyphenated prose phrases. **Positive controls passed throughout (`Tensor`/`Graph`/`DType`/`NodeHandle`/`LlamaConfig` all EXIST), and the extractor's ground truth was re-derived at the start of each run** — necessary, because the architect's hand-written control set went stale within two hours when a dispatched lane created `fuel-model-core`.

   ⚠️⚠️⚠️ **RETRACTED WITHIN THE HOUR, BY THE LANE THAT FOUND IT, BEFORE ANY WORK WAS FILED — AND THE RETRACTION IS THE ITEM'S REAL RESULT.** **The `FusionMissRecord` headline below is WRONG. The docs declare it unbuilt, in bold, twice:** `08-pattern-harvest:29` — ***“No missing-fusion signal exists today”***, followed by *“they are **unbuilt stubs, not shipping behavior**”* and a pointer to canonical sequencing in `docs/session-prompts/baracuda-telemetry-plan.md` §9. `14-lifecycle:191-193` — ***“None of this exists today”***, naming its two prerequisites (a graph-layer hook, and the base-emission seam where `structure_key` is a stub and no `DispatchRecord` is emitted). **The doc names the thing, states it is unbuilt, names its prerequisites, and points at a plan. That is the constitution WORKING CORRECTLY, and filing it would have duplicated an existing plan.** **The misreading: *“v1 headline”* means FIRST IN BUILD ORDER, not *shipped and consumed*; and *“its consumer already exists”* refers to the `BindingEntry` table being the fix, not to a consumer of the record.** ⚠️ **THE MECHANISM IS THE REUSABLE PART AND IT IS THE SIXTH SHAPE APPLIED TO ITS OWN DISCOVERER'S FLAGSHIP CLAIM:** the lane identified the satisfied-non-goal class **in the second pass**, swept it FORWARD over the remaining 11 docs, **and never swept it BACKWARD over the priority 4 where this claim was already formed.** In their words: **“Discovering a defect class does not retroactively re-audit your existing findings — a claim formed before the lens exists never meets the lens unless you deliberately re-run it.”** ⚠️ **AND THE ARCHITECT'S HALF: I FILED IT AS THIS ITEM'S HEADLINE AND ASKED FOR THE PASSAGES AFTERWARDS.** The passages I requested are the passages that refute it. **Requesting them first would have cost one message and prevented a wrong headline on the answer to CireSnave's own question.** **THE CORRECTED RESULT: this audit's best finding is the SIXTH SHAPE ITSELF, and the retracted headline is its second worked example.** The instrument found the class; the people using it failed to apply it — once forward-only, once by filing ahead of the evidence. **WHAT SURVIVES AS REAL, all present-tense with NO unbuilt marker anywhere in their sections (polarity-checked across the whole group, not just the retracted item):** the four `fuel-format-*` leaves (`02-layers:79`) · `cargo add fuel-model-llama` (`02-layers:102`) · the interchange tier (`13-interchange:56`) · `fuel-codegen` (`13-interchange:77`) · `fuel-loaders` (`11-persistence:189`). **Five, not seven.** ⚠️⚠️ **RULING RETRACTED 2026-08-28 — IT HOLDS FOR 1 OF 3 AND IS WRONG FOR 2 OF 3. THE INFERENCE DID NOT SURVIVE ITS OWN GATING CHECK, WHICH IS WHAT LABELLING IT AS INFERENCE WAS FOR.** **Measured at `8c8c8fbc` by extracting each `####` heading's ```rust``` block, taking its declared `fn` names, and counting `fn <name>` definitions at head:** `BackendIdentity` **2/3** — `BackendPressureSignals` **0/2** — `BackendDiagnostics` **0/4**. **`BackendPressureSignals` AND `BackendDiagnostics` ARE GENUINELY UNBUILT — FILE THEM** (Tier 2 and Tier 3, so at their own tiers' priority, nowhere near Tier-1 urgency). **`BackendIdentity` IS category 4, but not by the guessed mechanism, AND IT IS THE HIGHEST-SEVERITY DOC DEFECT IN THE CORPUS:** `backend_id` has **zero `fn` definitions — it is a STRUCT FIELD** on `BackendCapabilities`, and `device_location`/`same_device` are **INHERENT methods on concrete types**, not on any shared trait. **It sits under *Tier 1 — Mandatory*, in the document external backend authors are pointed at: someone implementing to it searches for `trait BackendIdentity`, finds nothing, and cannot tell whether they missed a mandatory requirement or the doc is stale** — the `use fuel_nn::VarBuilder` injury, in the MANDATORY tier. **Doc fix, not new work.** All three are now marked in `05-backend-contract` v0.7. ⚠️ **HOW THE WRONG RULING WAS SUPPORTED, and the lane diagnosed it themselves: `BackendId` at 2109 non-comment occurrences and `BackendCapabilities` at 155 were cited as evidence the surface existed — OCCURRENCE COUNTS, which this file's own rule says cannot separate DEFINITION from CONSTRUCTION from MENTION.** The loose instrument we warn others about, pointed at our own disposition. **The check that settled it counts `fn <name>` DEFINITIONS against the doc's own declared method names — and that is the instrument the mechanical guard should use on doc-declared `pub trait` bodies first: they are enumerable and a missing trait name is a clean binary.** The traits are absent BY NAME; the functionality is not — `BackendId` **2109** non-comment occurrences, `BackendCapabilities` **155**, `BackendRuntime` **36**, and `pub trait BackendRuntime` really exists. The `pub trait Backend*` set at head is `BackendCapabilityProvider`/`BackendFactory`/`BackendProbe`/`BackendRuntime`/`BackendStorage`/`BackendStreams` — **a different trait decomposition than 05 specifies.** **Supporting detail: `05:3` says v0.6 was a *“crate-location refresh only”*, so the file was maintained for LOCATIONS while its TRAIT NAMES went unreconciled.** **LABELLED AS INFERENCE, NOT MEASUREMENT by the lane — the trait bodies have NOT been checked for semantic match, and that check gates the disposition.** **The original headline is retained below, struck, because the ERROR is the finding and deleting it would erase the second worked example.** ~~⚠️⚠️ **THE HEADLINE, AND IT IS A WHOLE TELEMETRY SURFACE: `FusionMissRecord` IS NAMED IN FOUR DOCS AND HAS NEVER EXISTED IN CODE.** `00-index` · `05-backend-contract` · `08-pattern-harvest` · `14-lifecycle`. **`08-pattern-harvest:26` calls it *“(closed-world, **v1 headline**)”* and `:83` gives it *“a **second, lower-latency consumer** than the maintainer loop.”*** **Its companion reason code `NoBackendKernel` — named in three docs — likewise never existed.** *(`git log -S` over `*.rs`: zero introducing commits, against a control set of names the same query correctly finds.)* **So the constitution describes a surface as shipped AND consumed, with a stated consumer, at v1-headline weight, and there is no implementation.** That is the clearest instance in the corpus of the thing CireSnave was asking about — **not doc rot, but a ratified decision nobody scheduled.** **ADJACENT AND WORSE IN FORM: `05-backend-contract.md` carries `####` SECTION HEADINGS for `BackendIdentity` (`:239`), `BackendPressureSignals` (`:383`) and `BackendDiagnostics` (`:426`) — none of which ever existed.** **A heading reads as a defined contract surface far more strongly than prose does**, and this is the document external backend authors are pointed at. *(`MemoryPressureSelector`/`LoadAwareSelector` at `:605` are described as what components *“all become”* — explicitly future, and correctly so.)*~~

   **SIX DISPOSITION SHAPES, and the last two were discovered BY the audit rather than supplied to it.** The architect handed the lane four; the corpus contained six, **and the taxonomy gap was in the direction that costs work rather than misses it.** **(1) A roadmap item nobody filed** — `FusionMissRecord`, the interchange tier, `fuel-loaders`. **(2) A decision walked back and never struck** — `PlanStore`, which `02-layers` states as a *current* boundary while **two `.rs` prose sites record its deletion**; the code answered and the constitution did not hear. **(3) Aspirational prose** — future-tense, correctly so. **(4) A rename the doc missed.** **(5) PARTIALLY REALISED — a tier whose CORE EXISTS and whose LEAVES DO NOT.** `fuel-formats` EXISTS, all four `fuel-format-*` leaves ABSENT; `fuel-transformers` EXISTS, only `fuel-model-core` of its tier does. **The doc is not wrong — the work is unfinished and nobody is tracking the remainder**, which the original four categories force into either “roadmap item” (losing that the substrate shipped) or “aspirational” (which is false). ⚠️⚠️ **(6) A SATISFIED NON-GOAL — an absence that CONFIRMS the constitution, reported by the instrument as a violation.** `Custom` is flagged ABSENT in three docs; the passages read *“there is **no** generic opaque / `Custom` node”* and *“an external op must decompose into this basis, **never become** a `Custom` node.”* **The doc asserts the thing MUST NOT EXIST.** **A grep for “named but missing” cannot distinguish an unkept promise from a KEPT PROHIBITION, and the failure direction is the expensive one: acting on it means “fixing” a decision that was made deliberately.** **`09-non-goals.md` is 177 lines of exactly this, so EVERY absence in it is presumptively CORRECT until the passage is read.** **Any section whose job is to say what Fuel does NOT have inverts this audit's polarity.**

   **CATEGORY 4 WAS HIDING, and finding it needed a second instrument.** Zero renames among the first 21 absences was suspicious — a tree that has done `Tensor`→`NodeHandle` and `fuel-core-types`→`fuel-ir` does not have zero. **`git log -S` over `*.rs` separates *never existed* from *existed and went away*:** **EXISTED, now gone** — `OpEntry` (34 commits, from the Phase 7.6 fused-op work) · `DimExpr` (12, from the FKC shape-expr AST) · `PlanStore` (6) · `PrecisionFloor` + `StridedInputPref` + `BitStablePref` (**all three from ONE commit**, the ranker's Phase 1.3 precision-and-caps filter) · `SoftmaxLastDimLowerRule` (3) · `MemGetInfo` (2) · `Relieved` (1). **NEVER existed** — `OptimizationMap` · `FusionMissRecord` · `NoBackendKernel` · `NodeKind` · `BackendIdentity` · `BackendPressureSignals` · `BackendDiagnostics` · `MemoryPressureSelector` · `LoadAwareSelector` · `WholeGraph`. ⚠️ **`SoftmaxLastDimLowerRule` is PINNED, NOT STALE, and belongs with the retirements rather than the defects:** `04-optimization:344` describes what a **named commit** shipped, and that commit is the one the pickaxe identifies as introducing it. **The doc is a historical statement; renaming it would make a true sentence false.** ⚠️ **AND THE LANE WITHDREW TWO OF ITS OWN VERDICTS BEFORE THEY WERE USED: `git log -S` matches a SUBSTRING ANYWHERE IN A DIFF, so it is unreliable on GENERIC names.** `Strict` came back *EXISTED(24)* with a top hit about JIT dtype mappings; `Concurrent` *EXISTED(1)* on the `SystemTopology` commit. **Both re-marked INCONCLUSIVE rather than renamed.** Distinctive multi-word names are trustworthy under the pickaxe; single common words are not.

10. **The five surviving doc-vs-code absences, FILED with measured dispositions — and the headline is that only ONE of them is unstarted work.** Item 9 named the survivors; this item disposes of them. **Every name below was re-measured at head 2026-08-28 with a positive control, because item 9's citations are LINE-NUMBERED into `02-layers.md`, which went v0.5—v0.8 in a single day — the line numbers in that item are already stale, and that is worth noting as its own small lesson: a citation into a file you are actively editing rots faster than the claim it supports.**

    **MEASURED (`ls -d`, controls in the right column):**

    ```
    fuel-codegen        ABSENT      fuel-formats       PRESENT   (control)
    fuel-loaders        ABSENT      fuel-transformers  PRESENT   (control)
    fuel-model-llama    ABSENT      fuel-model-core    PRESENT   (control)
    fuel-format-*       NO MATCH    fuel-model-*       3 MATCHES (control: the
                                    glob mechanism works, so the empty result is
                                    real absence and not a broken query)
    ```

    ⚠️ **THE DISPOSITION SPLIT IS THE ACTIONABLE RESULT, AND IT IS NOT "FIVE NEW ITEMS".** **ONE is unstarted work · ONE is downstream of an item already filed · THREE are CRATE-BOUNDARY SPLITS OF CODE THAT ALREADY EXISTS (category 5, partially realised).** **Reporting them as five roadmap items would have overstated the outstanding work by roughly a factor of five**, which is the counting defect this registry has been correcting all day, arriving in the audit's own output.

    **(a) `fuel-codegen` — CATEGORY 1, THE ONLY GENUINELY UNSTARTED ONE. UNSCHEDULED, tier B.** `13-interchange` describes a dev-time **scaffolder**: from source AST + `config.json` (optionally a trace as oracle), emit a *draft parametric* `fuel-model-*` crate — `Config` struct, `new()` skeleton, `forward()` stub with recognised ops and `TODO` markers for the ~20% that is genuinely novel. **Nothing like it exists.** ⚠️ **CireSnave raised this surface directly on 2026-08-27 and drew the distinction the doc does not: `fuel-codegen` is *"a code generator that builds model files that use the authoring language as the language they're built in"* — i.e. it EMITS in the authoring surface rather than BEING it. So its dependency is item 8's model-construction layer: a generator cannot emit against a target that does not exist.** **SEQUENCING RULING: (a) is blocked on item 8 and must not start before it.**

    **(b) `fuel-model-llama` — NOT A NEW ITEM. Downstream of item 8, cross-referenced only.** `02-layers` shows `cargo add fuel-model-llama` as the consumer-facing gesture. **That is item 8's model-construction layer seen from the outside** — the per-architecture crate is what item 8 makes possible, so filing it separately would double-count the same work. **Recorded here so a future audit does not re-file it as a sixth absence.**

    **(c) `fuel-format-*` leaves — CATEGORY 5, and the SUBSTRATE SHIPPED.** `fuel-formats` EXISTS and holds six modules — `ggml`, `gguf`, `imatrix`, `pickle`, `safetensors`, `lib` — **every one a FOREIGN format.** The doc claims a per-format leaf split; the code has one crate. **So this is a crate-BOUNDARY question, not missing functionality, and it is the same shape as `fuel-core`'s pending dissolution: the contents already imply the split line.** **Ruling: DO NOT schedule as feature work. It belongs to the restructure programme (`docs/restructure-migration-design.md`) and should ride whichever stage moves `fuel-formats`, or not at all.** **A split with no consumer asking for it is cost without benefit** — the standing test from the `fuel-dispatch`/`fkc` decision.

    **(d) the interchange tier (`fuel-format-interchange-*`) — CATEGORY 5, PARTIALLY REALISED.** `fuel-onnx` EXISTS (and needs `protoc`, per CLAUDE.md). The named leaves do not. **`13-interchange` is explicit that node↔weight binding stays FORMAT-LOCAL — *"the right and only job of a `fuel-format-interchange-*` leaf"* — so the doc describes a boundary the ONNX path already honours informally.** **Same ruling as (c): a naming/boundary question for the restructure programme, not feature work.**

    ** ⚠️ **DONE 2026-08-28 — the doc half is corrected.** `docs/architecture/11-persistence.md` is at **v1.4** with the AS-BUILT correction reframing the present-tense `fuel-loaders` sentence as an INTENDED surface. **The CRATE remains unbuilt and unallocated; what closed is the FALSE PRESENT TENSE — the half that could cost an external reader time today.** *(Verified at `origin/main` `20657292`.)*** `11-persistence:189` states it in the **PRESENT TENSE** — *"fuel-loaders uses the existing `hf-hub` Rust crate for HF Hub; GitHub is HTTPS GET on `raw.githubusercontent.com`"* — **which is the stronger form of the defect than a future-tense promise, because a reader has no cue to doubt it.** **Measured, and the two halves diverge:**

    ```
    hub download        EXISTS, SCATTERED   hf-hub is a workspace dep consumed
                                            directly by fuel-core, fuel-datasets,
                                            fuel-examples -- no loader crate
    the URI schemes     ESSENTIALLY ABSENT  hf://  1 file
                                            github://  0 files
                                            raw.githubusercontent  1 file
                                            (control: safetensors, 246 files)
    ```

    **So the CAPABILITY the sentence describes is real and the ABSTRACTION it names is not, and a consumer reading that line reasonably concludes there is a loader API to call.** ⚠️ **This is the `fuel-nn` injury verbatim — a reader wrote `use fuel_nn::VarBuilder` and hit a wall (`02-layers`, the only recorded consumer injury in the corpus) — and it is the ONE of the five that can cost an external reader time TODAY.** **RULING: the doc fix is not optional and does not wait for the crate.** Mark the sentence as describing an intended surface, or restate it against what exists (`hf-hub` used directly). **The CRATE is a genuine open question and stays unallocated; the FALSE PRESENT TENSE is a defect to be corrected on sight.**

    ⚠️⚠️ **COUNT RECONCILED 2026-08-28, AND THE TWO FIGURES ARE IN DIFFERENT UNITS — STATE THE DENOMINATOR OR THEY READ AS A CONTRADICTION.** The polarity sweep this item's five came from is now COMPLETE, and its number is **16** — ⚠️⚠️ **PROVISIONAL AND KNOWN-STALE AS OF 2026-08-28, AND THIS ENTRY OVERSTATED IT WITHIN THE HOUR OF WRITING IT.** **The sweep ran at `5ecba5ce`, 22 commits behind head; `6865f889` and `d2116ce1` are NOT ancestors of it, and BOTH HALVES of the instrument were on the stale checkout — docs from the working tree AND code ground-truth from `git ls-files` on the same tree.** So a doc fixed in those 22 commits still reads as broken, and a type added in them still reads as absent. **Being re-run at a NAMED SHA on both sides.** ⚠️ **HOW THE OVERSTATEMENT HAPPENED, because it is the rule this registry applies to everyone else: the architect measured THREE names at head (`PlanStore`, `fuel-tensor`, `fuel-autograd`), found the claims still standing, and read that as CLOSING the tree-currency question.** It closed **their own** question — *"did my dispositions reduce the count?"*, answered no — **and said nothing about the other thirteen.** **AN OUTPUT LICENSES ONLY ITS OWN QUESTION.** Caught by the portfolio PM, who had asked the sweep's ref directly instead of inferring it from a related result. **Carry the number with the SHA or not at all; `origin/main` is not a ref, it diverges as fast as anyone commits.**  **`5` counts ARTIFACTS** (the four `fuel-format-*` leaves grouped as one, `fuel-loaders` once); **`16` counts SITES, deduplicated by (name × doc)** — so the leaves are 4 and `fuel-loaders` is 2. **Same corpus, different questions, and the portfolio PM relayed both to CireSnave ten minutes apart with no denominator on either** *(caught and corrected by them)*. **The `5` above is not superseded; it is the disposition grouping and stays the scheduling unit.**

    **THE SWEEP'S CONSTRUCT, verbatim, because a bare figure is what caused the whole episode:** *16 present-tense claims naming an artifact absent at head, across the 4 priority architecture docs, deduplicated by (name × doc), polarity-checked site-by-site, with 7 measured false-positive classes removed.* Full corpus **28 across 9 of 15 docs**; **partition checksum asserted in the script (31 removed + 2 unresolved + 28 surviving = 61)** so a site cannot vanish between buckets. **2 unresolved (`Concurrent`, `WholeGraph`) are reported as their own bucket rather than folded in.**

    ⚠️ **THE `21 − 1` AUTOPSY, AND IT WAS WRONG IN BOTH THE OPERATION AND THE OPERAND.** The `−1` subtracted the `FusionMissRecord` retraction from a population **it was never in**; meanwhile **four items the architect had already dispositioned *LEAVE ENTIRELY* hours earlier** (`fuel-storage`, `fuel-core-types`, `fuel-graph-router`, `fuel-graph-executor`) **were still inside the number being reported upward.** **The disposition happened and the count never moved** — a different defect from a stale count, because the WORK was done and only the LEDGER lagged.

    ⚠️⚠️ **AND THE MIRROR-IMAGE DEFECT, WHICH WEARS THE SAME SYMPTOM AND HAS THE OPPOSITE REMEDY — THE ARCHITECT'S OWN, CAUGHT ONLY BY GOING TO THE ARTIFACT.** The sweep also listed `PlanStore`, `fuel-tensor` and `fuel-autograd`, which the architect had *believed* dispositioned in `02-layers` v0.7/v0.8 and queried as stale. **Measured at head: the diagram still named all three, plain and unmarked, as current entries. THE COUNT DID NOT MOVE BECAUSE THE CLAIM DID NOT MOVE** — the disposition was recorded ADJACENT to the artifact the reader and the extractor actually read. **Ledger-lagged and remedy-too-weak both produce "a number that will not go down", which is why they collapse together; only reading the artifact separates them.** **`02-layers` v0.9 (`a6b5476d`) marks the absent names INSIDE the diagram.** **The file had already diagnosed this exact failure — v0.8 exists because the `fuel-nn` as-built note went stale, and its own text says *"the remedy for a stale diagram entry was itself a diagram-adjacent prose claim"* — written the same day the same weak remedy was applied to four more names in the same diagram.**

    **WHAT THIS LEAVES FOR SCHEDULING, which is the answer to CireSnave's *"is our roadmap driving Fuel in the right direction"*:** **one new item blocked behind an existing one (a); one doc correction owed immediately (e); two boundary questions that belong to the restructure programme and should NOT be scheduled as feature work (c, d); one non-item (b).** **The constitution is not, on this evidence, steering the project somewhere the roadmap is not going. Its worst failure mode here is PRESENT-TENSE PROSE ABOUT UNBUILT ABSTRACTIONS, which costs external readers rather than misdirecting internal work** — and the durable remedy is item 9's extractor running in CI, not another round of notes.

11. **The crate restructure — Stage 1 SHIPPED 2026-09-02, Stage 2 in flight; and this program was MISSING from this frontier entirely until now, which is an instance of the very class items 9 and 10 catalogue.** The design is [`docs/restructure-migration-design.md`](docs/restructure-migration-design.md); §5 pins the four-stage ordering, and 02-layers v0.5 already ratified the destination, so nothing here re-opens it. **Recorded because ROADMAP is *the path*: a multi-stage structural program with a landed stage and no line on the frontier is doc-vs-code drift of exactly the kind item 10 was filed to catalogue.**

    **Stage 1 — make `fuel` real. SHIPPED `a2027651` (PR #35, 2026-09-02T06:50:48Z, five files).** `fuel` was a manifest alias (`fuel = { path = "./fuel-core", package = "fuel-core" }`); it is now a real facade crate re-exporting `fuel_core`’s public surface — `pub use fuel_core::*` **plus an explicit `bail` re-export, because a glob does not carry `#[macro_export]` macros.** Feature forwarding is 1:1 and gated by `fuel/tests/feature_forwarding.rs`, **born-red in BOTH directions — a crossed pair and an omitted one.** `Cargo.lock` gained only the new member, no smuggled version bumps. Green on 5 Check platforms and 3 Test Suite platforms.

    ⚠️ **WHAT THAT GATE DOES AND DOES NOT ESTABLISH, because the distinction is citable and the wrong half is the quotable one: coverage is 10 textual / 2 effect / 8 effect-unverified — a 1:1 MANIFEST assertion plus `#[cfg]` effect tests, NOT an item diff.** An item diff is **impossible** here (nothing expands the glob) **and would be circular if it were possible** — for a glob re-export the facade’s items *are* `fuel-core`’s, so `cargo public-api` describes the facade’s own few lines rather than the surface behind them. **Anyone citing "the public API is verified identical" is citing something that was never measured. What IS measured is that the FEATURE surface is 1:1.**

    **Stage 2 — move the model zoo out. IN FLIGHT.** **`147 − 1 = 146` `lazy_*` files** (of **135,742 of `fuel-core`'s 186,350 lines, 72.8%**) into the Models tier. ⚠️ **THE MINUS ONE IS `lazy_latent_cache`, AND IT IS FORCED RATHER THAN CHOSEN:** `fuel-nn` reaches it through the facade, `fuel-transformers` already depends on `fuel-nn`, so moving it into the Models tier would close a cycle. **The carve-out criterion is MECHANICAL — a `lazy_*` module STAYS if any crate at or below the Models tier references it — NOT a judgement about which modules are "really models", which would be 147 classification calls and would make a mechanical move semantic.** ⚠️ **AND THE CRITERION DIVERGED FROM THE PREDICTION IT WAS GLOSSED WITH: I said it would "keep the cache modules"; measured, `lazy_kv_cache` is NOT a stay-forcer and MOVES with the zoo. One cache module stays, not the family.** **Reuniting them is a Stage 3 question. A falsifiable criterion produced a surprising answer, which is what an unfalsifiable one can never do.** Mechanical (change crate, fix imports, path-set unchanged) and the largest single reduction available. **The reference partition is measured and lives in the design doc, not here** — and it is a measurement over a CORPUS, so it must be re-run after the final rebase rather than before: new `use fuel::…` lines merge cleanly and invalidate it silently.

    ⚠️ **SEQUENCING — do NOT combine Stage 2 with item 8’s model-construction work.** This is already stated under item 8 and is repeated here because **item 8 is where it will be missed**: Stage 2 is *mechanical*, item 8 is *semantic*, and combining them makes a 147-file move unreviewable.

    **Stages 3 (fission the ~50k remainder) and 4 (retire `fuel-core`) are UNSTARTED** and both depend on Stage 2 landing first.

    ⚠️ **STAGE 3 CARRIES THE SAME DEFECT CLASS STAGE 2 HIT, AND THE PRE-FLIGHT IS ONE QUERY.** Stage 2 was planned as a mechanical move and turned out to be blocked by an UNSTATED DEPENDENCY-GRAPH CONSTRAINT — moving `lazy_latent_cache` into the Models tier would have made `fuel-nn` depend on `fuel-transformers`, which already depends on `fuel-nn`. **Nothing in the plan predicted it; a cargo cycle did, at execution time.** **BEFORE STAGE 3 MOVES ANYTHING, RUN THE SAME CRITERION PER DESTINATION: does any crate at or BELOW the destination's layer reference the thing being moved? If yes it cannot go there.** **A dependency-graph query, reproducible, and it fails loudly — unlike "is this really a model", which has no instrument.**

    ⚠️ **TWO KNOWN HOLES IN THE STAGE 3 TABLE, both recorded now rather than discovered later:** **(a) `serving / decode` (8.1k) has destination "open"** — the only genuinely unresolved row, plausibly `fuel-inference`, and it should be settled BEFORE Stage 3 starts rather than during. **(b) `lazy_latent_cache` HAS NO STAGE 3 DESTINATION AND THE TABLE DOES NOT NAME IT** — it exists as a named stay-behind only because Stage 2's carve-out created it, so it falls into the table's catch-all "the rest ... distribute". **A decision in one stage created an item in the next and nothing recorded it; that is the pattern to watch for, not the module.** **Its sibling `lazy_kv_cache` moves with the zoo under the mechanical criterion, so Stage 3 also inherits the question of whether to reunite them.**

12. **TEN crates depend on the CONSUMER FACADE — THREE of them are LAYER VIOLATIONS, TWO cannot be classified because the layer model never placed them, and one of them is a hard cargo CYCLE that blocks Stage 2 — measured 2026-09-02 at `a744417e`.** The facade `fuel` exists so *consumers* have one import. **A Models-tier or Foundation-adjacent crate depending on it is a layering inversion**, and `fuel-transformers` depending on `fuel` makes it **structurally impossible for the facade to re-export the 147 models Stage 2 is about to move into it** — `facade → fuel-transformers → facade`. **No re-export trick avoids it; a `pub use` needs the dependency edge.**

    **AUDITED 2026-09-02 (owner: Fuel 3), by `cargo metadata` target kinds and reverse-dependency edges — NOT by crate name:**

    ```
    CRATE                |  SRC  s-doc  s-grp | TEST  t-doc | REAL REVERSE-DEPS
    ---------------------+--------------------+-------------+---------------------------------
    fuel-nn              |  219      6      2 |    0     0  | fuel-training, fuel-transformers
    fuel-inference       |   92     17      0 |    0     0  | (none)
    fuel-training        |   30     11      0 |    0     0  | (none)
    fuel-onnx            |   18      5      4 |    3     0  | fuel-examples
    fuel-parallel        |   16      4      4 |    4     1  | (none)
    fuel-datasets        |   13      8      2 |    0     0  | fuel-examples
    fuel-transformers    |    1      0      1 |    2     0  | fuel-examples, fuel-inference

    CONSUMERS, correct   fuel-examples  fuel-lazy-examples  fuel-tensor-tools
    CONTROL              fuel-vulkan-backend depends on fuel-core DIRECTLY
    ```

    **SRC/TEST** = `fuel::` path-token occurrences · **s-doc/t-doc** = those on `///`/`//!` lines · ⚠️⚠️ **THAT COLUMN WAS LABELLED "COMPILER-BLIND" AND THE LABEL COLLAPSED TWO POPULATIONS WITH DIFFERENT GATES. CORRECTED 2026-09-02 after an independent measurement on the `fuel-datasets` pilot:** doc-comment references split into **FENCED** (executable code inside a doctest, compiled by `cargo test --doc` and **NOT** by `cargo check --all-targets`) and **PROSE** (compiled by nothing, genuinely blind). **Measured, and the ratio VARIES PER CRATE so no single label is available:** `fuel-datasets` 8 doc = 8 fenced + 0 prose · `fuel-training` 11 = 7 + 4 · `fuel-nn` 6 = 1 + 5. **The other four are unsplit.** ⚠️ **THE ERROR RAN IN THE COSTLY DIRECTION: "compiler-blind" says a repoint verified by `--all-targets` is as good as it gets, when for the fenced portion there IS a red available under a DIFFERENT COMMAND.** Demonstrated on one tree: reverting a single fenced reference left `cargo check -p fuel-datasets --all-targets` **GREEN with a genuine `Checking` line** and turned `cargo test --doc -p fuel-datasets` **RED with `E0433`**. ⚠️ **AND IT LANDS HARDEST ON RESTRUCTURE STAGE 2, WHERE THREE INSTRUMENTS ARE BLIND AT ONCE: doctests compile as EXTERNAL crates, so one inside `fuel-core/src/lazy_bert.rs` writes `fuel_core::`, not `crate::`** — invisible to a `crate::lazy_*` sweep, invisible to a code sweep (it is inside a `///` fence), and not compiled by `--all-targets`. **`cargo test --doc` is therefore a REQUIRED gate for that move.**

    ⚠️⚠️ **AND THE CORRECTION ABOVE OVERSHOT, MEASURED ACROSS ALL SIX CRATES RATHER THAN THE ONE PILOT. THE FENCED HALF IS NOT THE DOMINANT HALF:**

    ```
    crate            fenced   prose      which gate catches it
    ---------------+--------+--------+-------------------------------
    fuel-datasets        8        0     cargo test --doc
    fuel-training        7        4     mixed
    fuel-inference       6       11     PROSE MAJORITY -- nothing
    fuel-nn              1        5     PROSE MAJORITY -- nothing
    fuel-parallel        1        4     PROSE MAJORITY -- nothing
    ```

    **PROSE CARRIES THE MAJORITY IN THREE OF SIX, so the gate that catches NOTHING is the dominant one more often than not.** ⚠️ **AND THE PILOT THAT PRODUCED THE FENCED FINDING WAS THE LEAST REPRESENTATIVE CRATE IN THE SET FOR THIS EXACT QUESTION — `fuel-datasets` is 8/8, the only 100%.** **It was chosen as the pilot for being the CHEAPEST (13 src refs, smallest in the set), and cheapness selected the sample where the claim it produced was MAXIMALLY TRUE.** **A PILOT CHOSEN FOR BEING CHEAPEST IS NOT THEREBY REPRESENTATIVE** — and the failure is invisible from inside the pilot, because everything it measured was correct. **I made that selection.** ⚠️ **THE STANDING FORM: a doc-reference count that is NOT SPLIT says nothing about which gate would catch it. My original column was one number over a population with two halves and two different fates; the first correction was one number over the same population weighted by the wrong crate.** ·  **s-grp** = count of `use fuel::{…}` STATEMENTS (each expands to many names, so it is a statement count, not a name count).

    ⚠️ **THE CLASSIFICATION BELOW IS THE FOURTH AND THE FIRST ONE THAT CAN FAIL. The three before it were EIGHT (by crate NAME), SEVEN (by layer, without applying the criterion to the list), and THREE-PLUS-TWO (by layer, applied). ⚠️ **NONE OF THOSE THREE WAS FALSIFIABLE BY RUNNING ANYTHING** — *"inversion by name"*, *"seven crates"*, *"correct by layer"* are all judgements, so each restatement had to be caught by a person. **Not carelessness three times; the same missing instrument three times.**

    **THE INSTRUMENT IS A REVERSE-DEPENDENCY QUERY, and it asks the question that actually matters: DOES A FACADE DEPENDENCY PROPAGATE INTO ANOTHER LIBRARY?** *(Measured excluding the ROOT manifest, which counts as a dependent and inflates every leaf to one.)*

    ```
    PROPAGATES INTO A REAL LIBRARY  -- the load-bearing pair, fix these
      fuel-transformers   <- fuel-examples, fuel-inference
      fuel-nn             <- fuel-training, fuel-transformers
    TERMINATES IN A CONSUMER        -- tidiness; propagates only into fuel-examples
      fuel-datasets       fuel-onnx
    LEAF, nothing depends on them   -- arguably not violations at all
      fuel-inference   fuel-training   fuel-parallel   fuel-tensor-tools
    ```

    ⚠️ **THE ORDERING NOW FALLS OUT OF THE MEASUREMENT RATHER THAN OUT OF COST.** `fuel-transformers` is exactly the one Stage 2 hit as a hard cycle, and `fuel-nn` is the other crate whose facade dependency reaches a real library — which is why those two are the expensive pair and the four leaves are arguably fine as they stand. **A leaf whose facade dependency propagates nowhere is an application, and consuming the facade is what the facade is FOR.**

    ⚠️ **AND NOTE WHAT THE EARLIER LAYER-BASED ANSWER GOT RIGHT FOR THE WRONG REASON: it cleared `fuel-inference` and `fuel-training` by calling them "Use-Case Orchestration", a judgement about intent. The rev-dep query clears them on evidence — nothing depends on either.** **Same verdict, and only one of the two can be re-run by the next person.**

    **The name-based error is still worth recording: THE AUDIT FOUND ONE ERROR AND IT WAS THE CRATE WHOSE NAME MOST SOUNDS LIKE A LIBRARY: `fuel-tensor-tools` HAS NO `lib.rs`** — `{bin: 1}`, one `main.rs`, **zero crates depend on it.** A leaf CLI consuming the facade is exactly what the facade is FOR. **Seven inversions and three consumers, not eight and two.**

    ⚠️ **AND THE COUNTS RECONCILE ONLY WHEN `src` IS SPLIT FROM `tests`: `fuel-transformers` reads ONE or THREE depending on the population, and both figures were reported by different lanes on the same day.** Neither was wrong. **The split is kept in the table because it is the construct that makes the numbers comparable at all.**

    ⚠️ **THE CYCLE QUESTION HAS A UNIFORM ANSWER, SO IT DOES NOT DISCRIMINATE.** I asked which crates would cycle if the facade re-exported from them, framing that as tidiness-versus-blocker. **All seven would, structurally: each depends on `fuel` today, so ANY facade dependency on ANY of them closes a loop.** **And there is NO cycle today — `fuel/Cargo.toml` depends only on `fuel-core`.** **The real discriminator is whether the facade PLANS to re-export from it: `fuel-transformers` is a LIVE blocker (Stage 2 re-exports the 147 models); the other six are LATENT, zero cost today, and cheapest to repoint precisely while nothing depends on the outcome.**

    **SEQUENCING BY MEASURED COST:** `fuel-transformers` with Stage 2 (1 src line) → `fuel-datasets` 13 / `fuel-parallel` 16 / `fuel-onnx` 18 as cheap independents → `fuel-training` 30 → `fuel-inference` 92 → **`fuel-nn` 219 as its OWN item, folded into nothing** — it is an order of magnitude above everything else and 213 of its references are compiler-visible.

    **Stage 2 repoints EXACTLY ONE** (`fuel-transformers` → `fuel-core`), and the cost is measured, not estimated: **one `fuel::` import in existing code, one manifest edit, five feature forwards.** ⚠️ **Those forwards are safe ONLY BECAUSE Stage 1’s forwarding is 1:1** — `fuel/cuda` *is* `fuel-core/cuda` — **so `fuel/tests/feature_forwarding.rs` is the artifact that licenses the edit. On any tree without that gate it would be a silent feature-surface change.**

    ⚠️ **THE OTHER SEVEN ARE EXPLICITLY OUT OF STAGE 2’S SCOPE.** A mechanical 147-file move plus a workspace-wide layering correction is two changes and the combined diff is unreviewable — the same reason item 8 must not be combined with Stage 2. **But they are not dormant: the cycle REAPPEARS the moment the facade re-exports anything from `fuel-nn` or `fuel-inference`, so this is a blocker in waiting rather than tidiness.**

**Blockers**: none on the CapturedRun critical path — **CapturedRun 4b is COMPLETE (2026-07-13, commits `a127c190`..`9b7a5d1c`, merged to main).** The FKC contract-verification prerequisite (item 6) shipped (`cf0c3ee2`); this session then seeded the CUDA verification ledger (`seed_cuda_ledger.rs`, 219 records) so the decode places on CUDA, and the acceptance test `forward_with_kv_context_captured_matches_persistent` is byte-exact GREEN on the RTX 4070. The 4b-ε bench measures a **10.4× captured-replay speedup, byte-exact** on TinyLlama-1.1B F32 (D2 plan-once ~267 ms/tok → D3 captured `cuGraphLaunch` replay ~25.8 ms/tok median). One correctness finding along the way: baracuda's `rope_apply` is INTERLEAVED but Fuel's `FusedOps::ROPE` is ROTATE-HALF, so the Step-2 fused registration was reverted and rope runs DECOMPOSED on CUDA (auto-re-fuses once a rotate-half fused kernel + a real rope pattern matcher land). Multi-GPU work (Phase 6c D2D, 6d MoE placement) is parked pending hardware.

### Adaptive runtime fusion (2026-06-20)

A locked architectural decision set the destination for an **adaptive
runtime-fusion loop** — Fuel detects fusion opportunities a model author
never wrote, asks a trusted backend (Baracuda first) to JIT-synthesize a
kernel during idle time, and adopts the result cost-gated, **without**
surrendering the constitution (the optimizer that reads the DAG holds the
strategy; backends synthesize a Fuel-chosen region but never find
opportunities). The eight decisions — the recipe principle (every fused op
ships a total `decompose` + a `pattern`), the build-time-closed primitive
basis, two-tier runtime extensibility (binding table already extensible;
trusted declarative fused-op registration is the new goal), missing-fusion
telemetry (closed-world `FusionMissRecord` first, open-world co-occurrence
deferred), the narrow/non-monotonic megakernel, the Fuel-strategist /
backend-synthesizer closed loop, and loses-everywhere kernel-cache pruning —
are canonical in the [2026-06-20 decisions-log entry](docs/architecture/10-decisions-log.md).
The FKC declarative-fusion spec
([`docs/specs/fkc-fusion-patterns.md`](docs/specs/fkc-fusion-patterns.md))
and the telemetry plan
([`docs/session-prompts/baracuda-telemetry-plan.md`](docs/session-prompts/baracuda-telemetry-plan.md)
§9) implement it. The ROADMAP touch-points are flagged in place: Phase 7.5
optimizer (recipe principle + total `decompose`), Phase 7.6 (two-tier
extensibility + the declarative engine as the Tier-2 prerequisite), the
"Online Judge cost feedback" addendum (missing-fusion telemetry), the
"Opportunities baracuda unblocks" list (megakernel ordering + kernel-cache
pruning), and Phase 10 (10a as the closed-loop seed).

---

### Deferred backlog (behind the critical path)

**DEPENDENCY BUMPS — five dependabot PRs, queued 2026-08-27 at CireSnave's direction.** His standing instruction: *"I prefer to be on the newest versions wherever possible. You don't need my okay to merge those"* — and, on seeing them red: *"add those to the queue … If they're not urgent, they can be back burnered behind what is."* ⚠️ **The authorization covers the DECISION, not overriding CI. Three of five red NINE jobs, which makes them integration work rather than merges — a migration wearing a version bump's clothes, and dependabot's framing is what makes it look routine.** **Order, cheapest first:** `#17` safetensors 0.7→0.8 (1 of 12 failing) · `#16` prost-build 0.13.5→0.14.4 (1 of 12) · `#18`/`#19` kiss-ref-core & kiss-classify-vocab 0.2.3→0.3.0 (9 of 12 — **ask the KISS architect first; these may interact with the sk4/conformance work**) · `#15` vulkane 0.9.0→0.13.0 (9 of 12). **And CI went green for the first time in 995 runs on the day these were queued, so the sequencing matters more than usual.**

   **`#15` vulkane 0.9.0 → 0.13.0 — measured by the vulkane author on request, recorded here so it survives the lane that reads it not being the architect.** **(a) THE SEMVER TRAP DOES NOT APPLY, and Fuel's memory of it needed the condition attached.** vulkane is 0.x, so cargo treats the first nonzero component as the major — `0.9` and `0.13` are different majors and `^0.9` does not match `0.13`. **Fuel's recorded trap is that `[patch.crates-io]` cannot cross a semver major and fails SILENTLY — but that needs a patch entry to be silent, and `[patch.crates-io]` holds ONLY `fuel-kernel-seam` + `fuel-kernel-seam-types` (measured).** vulkane is a plain crates.io dependency, so the requirement change is ordinary. **The trap is CONDITIONAL ON PATCHING and the note did not say so.** **(b) MOST OF THE CHANGELOG DOES NOT REACH FUEL.** All four of 0.13.0's breaking headings are KISS `vulkan:` token vocabulary behind the `kiss-target` feature, and **Fuel references no `vulkane::kiss` path.** *(Limit stated by the author: measured by grepping `vulkane::` paths, so a glob or re-export would be invisible — read it as "no direct path reference".)* ⚠️ **(c) THE ONE THAT DOES IS A SOUNDNESS FIX, AND IT INVERTS THE FRAMING: *driver-written enum fields could hold an invalid discriminant*.** Reading a Rust enum whose memory holds an out-of-set discriminant is UB, and every `returnedonly="true"` struct is driver-filled — **so the UB is reachable by upgrading a graphics driver, with no application change.** 81 fields became raw `i32`; safe accessors now return `Option<T>` with a `_raw` sibling. **`CooperativeMatrixProperties` is one of them and Fuel uses it — six `VkComponentTypeKHR` sites are where the adaptation goes.** **`None` means the driver reported a component type newer than the pinned `vk.xml`; it is NOT an error, and `_raw` exists so an unknown can be REPORTED rather than discarded** — which is the part a consumer gets wrong by reading the type. **So this is not "should we take four minor versions" but "we are currently exposed to driver-triggered UB", and 0.10.1 as a smaller step is off the table: it is additive and carries none of the fix, i.e. it leaves the UB in place deliberately.** **(d) 0.13.0 IS THE ONLY RUNG ABOVE 0.10.1** — 0.10.2, 0.11.x and 0.12.x were never published; 0.10.2 is documented but did not ship and its fixes arrive inside 0.13.0. ⚠️ **(e) A PREDICTION WITH A CAUSE AND A DETECTOR, FOR THE BUMP AFTER THIS ONE: every published vulkane including 0.13.0 declares `rust-version = "1.85"`, but vulkane `main` declares 1.88 and genuinely requires it** (let-chains in `vulkan_gen`, a BUILD-dependency and therefore on Fuel's compile path, plus `libloading 0.9` declaring 1.88.0). **This bump costs no MSRV movement. The next one will.** **Standing offer from the author: paste the first two compiler errors and they will say quickly whether the nine reds are the enum change or something unaccounted for — they read Fuel's dependency surface, not Fuel's failures.**

    ⚠️⚠️ **CORRECTED 2026-09-02 — THE CAUSE IS REAL AND THE CONSEQUENCE DOES NOT REACH FUEL.** The prediction *"this bump costs no MSRV movement; THE NEXT ONE WILL"* was recorded in good faith and its cause holds: vulkane `main` did raise its floor to 1.88, and every published vulkane through 0.13.0 declares `rust-version = "1.85"`. **But Fuel has no MSRV to move.** Measured independently by two people: `rust-toolchain.toml` pins `channel = "1.98.0"`, and **ZERO** workspace crates declare `rust-version` at all *(control: 41 declare `edition`, so the zero is absence and not a broken query)*. **A dependency raising its floor to 1.88 cannot move a floor Fuel does not declare, and CI builds on the pin regardless.**

    ⚠️ **THE EXPOSURE THAT REMAINS IS REAL AND UNGATED: a downstream consumer building Fuel on a toolchain older than a dependency's floor. Nothing in this repo checks that, and nothing would report it.** **So the prediction has a CAUSE and no DETECTOR on our side. What would make it live: Fuel declaring a `rust-version`, or a CI leg building on a floor rather than the pin. Until one exists, state this as UNMEASURED rather than as ABSENT.**

    **Pin arithmetic, because it pre-empts the inference people actually draw: `vulkane = "0.13.0"` is a caret requirement resolving `>=0.13.0 <0.14.0`, so it EXCLUDES the 0.14.0 published 2026-09-02.** That bump is a deliberate act, not something `cargo update` performs — and because 0.x minors are BREAKING under cargo semver it is an ADOPTION TASK of the same shape as the 0.9.0 → 0.13.0 one, not a version-string edit. **DEFERRED: its benefit is documentation correctness on a dependency's docs.rs page, weighed against a live-GPU verification cycle on a suite GAP-259 has just measured INTERMITTENT. The re-examination trigger already exists — GAP-260's "RE-EXAMINE AT THE NEXT VULKANE BUMP" — so no second clock was invented.**

Retained in detail below under Planned Work: Phase 7.5 C–F (graph-rewrite autograd, in-place-as-optimization, crate fission, layout contracts) + B3–B6 op-method sweep; Phase 7.6 steps 4/5/7/8/10 (fused-op migration sweep, Op-variant drops, PrecisionGuarantee/cost population, Comparison family); Phase 8 (FlashAttention tiers), 8.5 (activation sparsity), 9 (agentic extension hooks), 10 (equivalence-rewrite search); the eager-retirement follow-ups (binary re-migrations, test fixups). Sequenced *after* the active frontier; none is on the current critical path. One open design gap not yet phased: the **RNG / generator seam** — where a `Generator` lives (per-backend / per-device / per-graph), how it threads through realize and autograd, and how backends participate — which blocks dropout, sampling-as-a-graph-op, and stochastic training ops. Another backlog candidate: an **Apache Arrow `Tensor`** import/export leaf — *tensor*-level interchange (not a model format) for the columnar / data-engineering ecosystem (Arrow Flight distributed loading, polars/DuckDB feature pipelines, columnar feature stores). It is the host/serialization boundary, **complementary** to DLPack/FDX (which owns the device/kernel zero-copy boundary, `docs/specs/dlpack-extension.md`), not a competitor — sequence behind a real consumer, and lean on the existing DLPack↔Arrow-`ArrowDeviceArray` bridge first (FDX support already gives partial Arrow reach). Arrow's sparse layouts (COO/CSR/CSF) + `dim_names` are useful references if/when Fuel does sparse / named axes. See [13-interchange](docs/architecture/13-interchange.md) §Format posture. **A third backlog candidate, added 2026-08-14 and surfaced by MLMF's trim-down rather than by a Fuel review: IMATRIX GENERATION.** Fuel **consumes** llama.cpp importance matrices today — `load_imatrix`, `quantize_imatrix` (6 sites), `quantize_imatrix_onto` (6), `from_float_imatrix` (8), across `fuel-formats`, `fuel-backend-contract`, `fuel-cpu-backend`, `fuel-cuda-backend` and `fuel-core/src/quantized/` — but **cannot produce one**: `fuel-formats/src/imatrix.rs` exposes only `parse` / `parse_bytes` / `load_path`, **with no writer.** So the quantization workflow is completable only by leaving Fuel to run llama.cpp and coming back. **Scope is three pieces, and only the middle one is novel: (i) an imatrix WRITER (the read side already pins the format); (ii) per-tensor activation-statistics collection during a forward pass over a calibration corpus — the `CapturedRun` machinery (35 files) is the obvious substrate but has NOT been measured against this use, so treat that as a hypothesis; (iii) a calibration driver.** **⚠️ WHY THIS IS FUEL'S AND NOT SOMEBODY ELSE'S, WHICH IS THE WHOLE ARGUMENT: it requires forward passes. MLMF explicitly carries this as *unimplementable* rather than out-of-scope — they have no compute backend by design — and their trim deletes `quantization_simple.rs`/`quantization.rs` (1,752 lines) on exactly that ground. ⚠️ AND THERE IS NOTHING TO PORT: MLMF's `calibrate()` binds its `calibration_data` parameter and NEVER USES IT, iterating `model.raw_tensors` — the WEIGHTS — and storing the result in a field named `activation_stats`. It is weight statistics wearing an activation-statistics API, so a future reader should not go looking for an implementation to adapt.** **SEQUENCING, per the standing rule that "no consumer" sequences behind rather than skips: this HAS a consumer argument — every imatrix-quantized model Fuel loads was calibrated by another tool — but it is not on the critical path and sits behind the active frontier.**

- **Kernel dtype-coverage expansion (longer-term; the durable answer to mixed-precision op support).** Per-op kernel coverage today is the **hand-authored cross-product** — each backend registers only the `(op, per-operand-dtype)` combos someone wrote, with no generic type handling. So the graph layer can build a valid node (e.g. the mixed-precision `[F32, BF16, F32]` matmul the `matmul` builder + `apply_linear` docstring explicitly permit for bf16-weight serving) that only one backend implements natively (that combo: Vulkan only; CPU/CUDA are uniform `[T,T,T]`). The **immediate fix shipped** is an optimizer **dtype-reconciliation pass** (`optimize::insert_dtype_fixups`) that inserts a promoting `Op::Cast` (lossless upcast to the output dtype) when no backend serves a node's per-operand key — the "backends advertise; the optimizer adapts" contract applied to dtype, sibling of the layout/residency fixups; it makes the builder's mixed-precision promise executable everywhere today. **The longer-term work is to widen native per-op dtype coverage, starting with CPU kernels**, so hot combos run without the reconciliation upcast + the extra `Cast` node — e.g. a native CPU mixed-precision `[F32, BF16, F32]` matmul (read the BF16 weight, accumulate in F32) for bf16-weight serving, then sweeping the elementwise/reduction/etc. families across the full `DType` set (incl. F16/BF16 and the FP8/MX formats). Sequence behind real consumers; prioritize the combos the reconciliation pass reports firing on. Also a candidate: a per-backend "I can serve this via an internal upcast" capability advertisement, so a backend that *can* cheaply upcast isn't always forced through a graph-level `Cast`. Root diagnosis + the reconciliation pass: the 2026-07-29 mixed-matmul investigation. **Reconciliation-pass follow-ups** (surfaced by the first consumer, Lightbulb — the pass reports promotions via `tracing` today, but): (1) a promoting cast is **not numerically neutral** (it accumulates the upcast in higher precision, ≠ a native mixed kernel, so the same graph diverges by arm and already differs CPU-vs-CUDA), so make it **forbiddable under a C-5 determinism/tolerance constraint** — a byte-exact-oracle or CPU-vs-CUDA-parity consumer should get the native path or a hard fail, not a silent upcast; (2) **C-4 cost accounting** for the promotion's extra resident bytes (BF16→F32 doubles a weight — the dimension an inference host budgets C-1 admission on); (3) richer per-node promotion **telemetry** beyond the current summary line. **DX sweep:** the "tensors must live on the same graph" assert appears at **9 builder sites** (matmul [improved], qmatmul, conv2d ×2, conv_transpose2d, flash_attn ×3) — give all the "use `const_*_like` (each `from_*` mints a new graph)" hint.

### Frontier-architecture gaps (research-edge capture, 2026-07-04)

A frontier-readiness audit (six-track sweep against a survey of the 2025–26 research edge:
hybrid SSM/Transformer, MLA/attention compression, hyper-sparse MoE, test-time compute,
GRPO/verifiable post-training) cataloged the capabilities Fuel needs to run that frontier.
Full per-item status + consumers + citations: **[`docs/frontier-architecture-gaps.md`](docs/frontier-architecture-gaps.md)**.

*Already phased* (no new tracking needed, cross-referenced in the catalog): data-dependent
shapes → Phase 8.5 `Op::NonZeroIndices` + [`data-dependent-shapes-design.md`](docs/session-prompts/data-dependent-shapes-design.md); agentic/search-on-generation hooks → Phase 9;
graph-rewrite autograd → Phase 7.5 C; symbolic-`k_len` flash remains a documented,
never-crash basis gap in [10-decisions-log 2026-07-03](docs/architecture/10-decisions-log.md);
the `Scan` gap is **CLOSED** — `Op::Scan` Phase 1 shipped, see
[10-decisions-log 2026-07-15](docs/architecture/10-decisions-log.md).

*Newly tracked orphans* (had no planning-doc home — lived only in source comments or
nowhere; now captured so they are not forgotten):

- **Higher-order `Scan` `Op` — Phase 2 SHIPPED (2026-07-16); G3 CLOSED (Phase 1, 2026-07-15).**
  `Op::Scan` / `Op::ScanPlaceholder` (Fuel's first sub-graph-carrying primitive, Phase 1) landed;
  `selective_scan` + `ssd_chunk_scan`'s `decompose` emit it instead of surfacing the G3 basis gap —
  `decompose` is total over genuine primitives for the whole registered fused-op set. **Phase 2
  shipped: early-exit + differentiability + Hopfield consumer.** (1) `early_exit` is a built
  realize-barrier mechanism — a scalar-`U8` predicate over the carry carried as a trailing input
  (`Tensor::scan_until`), evaluated by a host step driver (`drive_scan_until_final_f32`) that stops
  at the runtime convergence step under the static-capacity `bound`. (2) BPTT differentiability via
  a `lower_scans_for_backward` pre-pass (decompose SSM recipes → `Op::Scan` → `unroll_scan` → node-
  general autograd, truncated to the static `bound`); `selective_scan`/`ssd_chunk_scan` are now
  differentiable (`BackwardKind::Decompose`, no bespoke `*_BACKWARD`). (3) First non-SSM consumer:
  `hopfield_retrieve` (Modern Hopfield associative memory), converges early + is BPTT-differentiable.
  **Still no `Op::Scan` native kernel** (kept out on purpose — the slot-1/`last_state` OOB blocker
  stays a typed error, not a live silent read); multi-carry, an `emit=All` early-exit valid-count
  buffer, and equilibrium/implicit-diff gradients remain deferred. See
  [10-decisions-log 2026-07-16](docs/architecture/10-decisions-log.md).
- **Reconstructive ("after-image") memory — DESIGNED (2026-07-30), SCHEDULED, not started.**
  A stateful memory in which a stored item is a **residual against the model's own prediction**
  (`r = x − p(cue; θ)`), reconstructed rather than replayed — so the same cue reconstructs
  *differently* as the weights drift. Full design:
  [`docs/superpowers/specs/2026-07-30-reconstructive-memory-design.md`](docs/superpowers/specs/2026-07-30-reconstructive-memory-design.md).
  **This is composition, not new substrate** — `hopfield_retrieve`, `Op::Scan`, `lazy_nn::lora`,
  `KvBlockPool` (`Fidelity`/`Externalized`/`restore`/`block_refcount`) and
  `Encoding::AffineBlock` are all already on main; the one genuine gap is that
  `hopfield_retrieve` takes `patterns` as a frozen **const** — there is no write/evict path.
  Classification: buckets **A** (residual path, generic-pool extraction) + **B** (precision-ladder
  rungs) + **F** (consolidation/migration scheduling = consumer policy, per §15) —
  **no bucket-E work; the primitive basis is untouched.** Load-bearing decisions: residual
  coupling adopted, query-side addressing deferred-but-hooked; **never fully evict** — precision
  decay with a floor set by the drift-*detection* requirement; a θ-stamp is a **LoRA adapter
  delta, not the weights** (which makes the frozen base a hard requirement, since full
  fine-tuning would make every checkpoint a whole model copy); migration is a copying collector
  (always-to-current-checkpoint, oldest-first, refcount-before-free, crash-safe via per-memory
  stamps); **staleness is free** (stamped reconstruction is exact at any age) so back-pressure
  belongs on live-checkpoint count, not staleness; `Δ‖r‖` falls out of migration as a per-memory
  measurement of catastrophic forgetting; shape is data (`MemoryGeometry` + a replaceable
  `MemoryPolicy` trait, mirroring the `KvGeometry` precedent), **not** a type parameter.
  **Sequenced after** multi-session inference and the RNG/generator-seam decisions.
  *(Corrected 2026-07-31: this entry originally also listed **B0.3 + B0.5** as blockers. They
  were already complete on 2026-06-27 — see "B0.1–B0.5 COMPLETE" above — so the gate count is
  two, not three. The constraint they stood in for still holds: the generic block-pool core
  must not land in the retiring `fuel-core`; that is now an available placement decision
  (`fuel-memory` or `fuel-backend-contract`) rather than a wait. The still-open item people
  mistake for B0.5 is the **Storage-unification**, blocked on B6 eager-dispatch retirement.
  Note the extraction also moves `KvBlockPool` out of `fuel-core`, so coordinate — the paged
  plan-once increment is actively editing that area.)*
  Increment 1 is the round-trip born-red gate (reconstruct via the *stamped* adapter
  succeeds; via the *current* adapter fails — the overshoot bug as a test), CPU-only, no training
  loop. **Precision ladder — RESOLVED 2026-07-31, favourably:** `DType::F4` is not the only
  sub-byte code. `fuel-ir/src/dtype.rs:14` carries five rungs — 32 (`F32`), 16 (`F16`/`BF16`),
  8 (`F8E4M3`, plus `F8E8M0`/`I8`/`U8`), 6 (`F6E2M3`/`F6E3M2`), 4 (`F4`) — all with handling
  beyond the enum arms, so precision decay has real resolution and the detection-derived floor
  is expressible. One small gap found instead: **there is no bit-width accessor** —
  `size_in_bytes()` (`dtype.rs:110`) returns **0** for the three sub-byte rungs, so mapping a
  bit budget onto a rung wants a `DType::bit_width()`; additive, bucket-B, trivially testable.
  **Still open:** whether `AffineBlock`'s `packed` has kernel support beyond `F4` (NF4 is the
  proven path; increment 1 stores F32 and does not decay, so it is not gated); and the
  `fuel-training` inference-time-update path is unverified.
- **Recipe-grammar convergence — Increment A SHIPPED (2026-07-16); shape-oracle SHIPPED as
  Convergence-C (2026-07-20/21); remaining Increment C narrowed to the recipe interior.**
  Increment A realized the pinned Fuel↔Baracuda recipe grammar's canonical form as machinery:
  `primitive_shape` (`fuel-graph/src/shape.rs`) as the single-source shape/dtype rule for the
  primitive basis (called by BOTH the `Tensor` builders and `emit`, so no builder-vs-`emit`
  drift), full first-order `emit`/`tag_to_op`/`validate_representable` (`Op::MatMul` now
  representable), and `OpAttrs::to_canonical_bytes` = the KISS §6.19 canonical positional-blob
  serialization — cross-checked byte-for-byte against the `rope`/`softmax`/`layer_norm`
  `decompose` oracles. **The gating shape-expression grammar co-design CONVERGED externally**
  (KISS RFC `rfcs/shape-expression-oracle.md` merged @ KISS `3bd6d2d`, 2026-07-19; superseded
  banner on the Fuel-side ask @ `9f8b8347`), and the Fuel-side implementation **SHIPPED on main
  as Convergence-C** (series `6dfc3011`..`f87fd401`, merge `9156e178`): **C-1** the shape-expr
  AST + §6.20 wire codec (byte-matches the KISS golden) + typed-decline decoder +
  `eval_shape_rule` evaluating the full `DimExpr` vocabulary
  (`6dfc3011`/`ddd76207`/`ae6b6300` — `OutputDesc.shape_rule` was `same_as`-only before);
  **C-2** the role/index-woven kinds — reduce/gather/matmul + the §6.4-0011 `shape_consistent`
  tie (`8d8338e9`); **C-3** cross-check ACTIVATION — the oracle now validates ~16 of the 22
  registered fused ops (was 9), plus the adversarial-review fixes (`80c20a47`
  warn-not-silent-skip + arity pre-check, `9c96a0f8` GQA probe shape-distinctness,
  `f87fd401` bundle-slot-vs-`output_views` differential). **Increment C slice 1 —
  recipe-interior FOUNDATIONS — SHIPPED (2026-07-23, branch `feat/increment-c-slice1`,
  T1–T10 `fbe96f0d`..HEAD; plan `docs/superpowers/plans/2026-07-23-increment-c-slice1.md`):**
  the `shape_expr` vocabulary moved to its permanent dependency-free home
  `fuel-kernel-seam-types` (`fkc/shape_expr.rs` is now a `pub use` shim); shape-**relative**
  `OpAttrs` interior fields (`target_shape_rel`/`slice_{start,len}_rel`/`axis_last`) + a pure
  `resolve_rel_attrs` resolver + a children-first resolving `emit` (D2/D3/D4); the additive
  `OpTag::MaxDim` (the D3 keepdim shrink-via-swap); a `decompose_via_recipe` bridge; **5 of the
  ~16 migratable registry `decompose` fns migrated to portable, shape- AND rank-polymorphic
  `PatternNode` data** (`softmax_last_dim`, `rope`, `rms_norm_last_dim`, `layer_norm_last_dim`,
  `softmax_last_dim_backward`); and **the locked matmul role-vector `op_attrs` serialize/resolve
  live in both directions** (the rank-2 golden `0C000000|02000000|0103|02000000|0302` is the
  Baracuda-confirmed cross-producer contract). Gates green: `fuel-kernel-seam-types` (18),
  `fuel-graph` (396). **Increment C slices 2–3 — carriers + first-order backward migrations —
  SHIPPED (2026-07-24):** slice-2 (`layer_norm_last_dim_backward` + `fused_linear`, the first live
  `WithDim` driver; on `origin/main`) then slice-3 (branch `feat/incc-reemit-carriers`, plan
  `docs/superpowers/plans/2026-07-24-increment-c-decompose-migration.md`) migrated
  `rms_norm_last_dim_backward` (`72eb481c`), `reduce_max_to_backward` (`98194c48`), and
  `powi_backward` (`fbe746e9`), each unblocked by a reusable re-emit carrier — the
  `OpAttrs.scalar_rel` shape-derived scalar (`reduced_count`/`MulScalar(n)`), the MaskedFill
  fill-`Scalar` carrier, and the PowI i32-exponent carrier — bringing the running total to **10 of
  22 `decompose` migrated / 12 remain** (4 of which are permanent basis-gap self-returns). Gates
  green: `fuel-graph --lib` (425), `fuel-dispatch` (712), `fuel-core --lib` (1385).
  **Still narrowed to the recipe interior (slices 2–5, §9 of the plan):**
  the remaining ~11 first-order `decompose` migrations (carriers: PowI/Clamp/MaskedFill,
  shape-derived scalar slots), the **flat-DAG-CSE recipe-node/table WIRE serializer** + the
  `reduced_count` leaf's **graph wiring**, the scan flat form (`selective_scan`/`ssd_chunk_scan`),
  and the §6.19 import decoder — node-envelope consolidation in-flight via **KISS #67**.
  *(Update 2026-07-23: the four **source-op leaf BYTE ARMS** — `const{bits}` with the MBZ
  narrow-dtype rule, `runtime_scalar{slot_index}`, `reduced_count{axis}`,
  `scan_placeholder{role,index}` — shipped in `to_canonical_bytes` on the KISS editor's
  four-leaf ack ([KISS #67 comment 5061571967](https://github.com/ThinkersJournal/KISS/issues/67#issuecomment-5061571967),
  acking Fuel's proposal comment 5060303085). Each rides **carrier (a)** (`u32`-LE `op_attrs`
  outer), distinct from the §6.8-0007 region-table (`u16`-LE) and §6.20-0005 shape-expr child
  (`u16`-LE) framings. Wire tokens only: `op_to_tag` emits none of them and `tag_to_op`
  declines all four as honest misses, so the leaves' graph wiring stays slices 2–5 work.)* See
  [10-decisions-log 2026-07-16 + 2026-07-21](docs/architecture/10-decisions-log.md);
  `docs/recipe-signature-reference.md` (Part II §A/§C, as-built); memories
  `recipe-grammar-codesign`, `shape-oracle-rfc-accepted`.
- **Increment C — Op::Scan recipe form SHIPPED (2026-07-24, branch `feat/incc-opscan-recipe`):**
  the **scan flat form** called out as pending in slice-1's "still narrowed" list now round-trips
  through a `PatternNode` **data** recipe. Added the re-emit carriers `OpTag::Scan`/`OpTag::View`
  (+ Fuel-internal `scan_*`/`view_slot` `OpAttrs`, off the §6.19 wire) in `tag_to_op`, a
  `ScanPlaceholder` body-shape carrier, and the 2-slot `output_views` bundle re-compose/re-attach
  for `Op::Scan`/`Op::View` in `emit` (both graph-resolved structural terminals, operand[0]
  fallback, never a panic; `Op::Scan` stays a base-map terminal — still no native kernel).
  `selective_scan` (two `delta_softplus` recipe variants) + `ssd_chunk_scan` (`chunk_size` a baked
  CPU no-op) migrated as data recipes, node-for-node identical to their frozen-legacy imperative
  decompose across all `batch`/`dim`/`dstate` (toy-interpreter parity + `base_map_hash` guards).
  Migrated registry `decompose` count → **9 of 22** (5 slice-1 + 2 slice-2 mechanical + 2 scan);
  **13 remain** (9 needs-extension carriers + 4 basis-gap). Plan:
  `docs/superpowers/plans/2026-07-24-increment-c-decompose-migration.md` ("Op::Scan recipe form —
  SHIPPED").
- **Increment C — `qmatmul` (Q4_0) decompose SHIPPED → 22 of 22 migrated (2026-07-28, branch
  `feat/qmatmul-q4_0-decompose`).** The last fused op leaves the opaque-island set. The old
  "basis gap needing 3 missing primitives" was FALSE (scoped 2026-07-25; the KISS §7.3-0002
  necessity test came back negative from Fuel/Baracuda/kiss-ref, so NO `Op::Bitcast` — see
  `docs/outreach/bitcast-basis-token-design-input-ask.md`). `qmatmul.rs` now carries a **total
  primitive recipe** for Q4_0, contained entirely to `fuel-graph` (no builder/loader/kernel/
  dtype change): byte-extract the U32 stream by **exact F64 arithmetic** (`Cast(U32→F64)` then
  `⌊·/256ⁱ⌋ mod 256`; F32 would round above 2²⁴), recover the embedded f16 block-scale by
  **arithmetic IEEE-754-half reconstruction** (5-bit binary-decomposition power-of-two, bit-exact
  to `f16::to_f32` for every finite half — proven over all 63488 finite bit patterns), nibble
  unpack + per-block broadcast + GEMM. Real-backend CPU-realize parity vs exact `BlockQ4_0::
  to_float` dequant at `rel<1e-5` (`fuel-core/tests/incc_qmatmul_q4_0_oracle.rs`, sabotage-
  calibrated); the fused dequant-in-kernel arm stays the cost-preferred cover (the lowering is
  the optimizer/basis-map alternative, never the executed path where the kernel exists).
- [ ] **`qmatmul` per-format decompose build-out (backlog).** Q4_0 is the only format the live
  loader produces today; the other **ten** `QuantType`s (`Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`,
  `Q2K`, `Q3K`, `Q4_K_M`, `Q5K`, `Q6K`) `decompose` as **surfaced gaps** (self-return, never a
  crash — the `flash_attn` concrete-vs-symbolic precedent). Each is its own recipe over the SAME
  technique (F64 byte-extract + the shared arithmetic f16 decode); the deltas are per-format block
  layout (`k_quants.rs` `#[repr(C)]` structs) and scale structure: flat single-scale (`Q8_0`),
  scale+min (`Q4_1`/`Q8_1`), scale+high-bit (`Q5_0`/`Q5_1`), and the hierarchical super-block
  6-bit sub-scales of the `*_K` family (a second arithmetic unpack over the sub-scale bytes).
  Sequence behind consumers: wire a format into the loader first (or a test that exercises it),
  then add its recipe + a `to_float` real-backend oracle. Grouping: the flat/near-flat formats
  (`Q8_0`/`Q4_1`/`Q8_1`/`Q5_0`/`Q5_1`) are near-mechanical reuses; the four K-quants are the
  harder sub-scale tier. Reference impl + oracle: `qmatmul.rs::recipe_q4_0` +
  `incc_qmatmul_q4_0_oracle.rs`.
- **Shape-oracle C-4 — Fuel-internal groundwork SHIPPED (2026-07-23); `Dims`/`WithDim`
  activation KISS-gated (extension-registry proposal FILED, cosign-tracked).** The C-4
  successor to Convergence-C, built as `feat/c4-groundwork` (plan
  `docs/superpowers/plans/2026-07-23-c4-groundwork.md`): `eval_shape_rule` threads
  per-variant synthesized param values (`param(N)` indexes the `FusedOpParams::key().ints`
  flattening — index tables pinned in `fused/{conv-rope,linear-quant}.fkc.md` + a
  doc-vs-code drift test), the return cross-check loops per-combo param POINTS (≥ 2 for
  variants with a free field, and order-ASYMMETRIC — no two `key().ints` slots agree at
  every point, so a flattening reorder or a `param(i)`/`param(j)` rule confusion cannot
  false-green: the sabotage-calibration norm applied to params), the params-dependent
  variants' `passthrough`/`fixed` dtype rules are now genuinely cross-checked (previously
  dead: synth returned `None` for them; a probe combo the shape-coupled synth can't read
  now surfaces an `ImportWarning`, never a silent skip), and the reserved-tag declines are
  NAMED at the decoder (`TAG_REDUCE`/`TAG_WITH_DIM`/`TAG_DIMS` → typed `ReservedTag`, the
  future activation point; `shape_expr` went `pub` — required to make the golden-verified
  §6.20 codec API-reachable so the `TAG_` dead_code warnings die by reference, not
  `#[allow]`). Scope disclosure (deviates from the plan's "nothing in `fuel-graph`"):
  `fuel-graph/src/registry/conv_transpose_2d.rs` arity `debug_assert`s widened 2 → 2-or-3
  (`x`, `weight`, `[bias]` — the contract's documented optional-bias arity; the exact-2
  assert made the now-live dtype differential guard-catch on the 3-operand probe); see the
  plan's §7 Deviations. **Correction — the prior entry's "reserved
  tags cover the ~7 skipped ops" overclaimed; the honest split is:** (a) **5 rules
  KISS-gated** — `conv2d` + `conv_transpose_2d` (rank-4 `Dims` + `Param`), `qmatmul`
  (`WithDim`/`Dims` + `Param`), and the two scan slot-1 `last_state` bundle rules
  (`Dims`-only pure extents; **verdict: NOT premature** — the Phase-2/3 open item is the
  *decompose-path* view composer, whereas the bundle differential's reference is the live,
  allocator-wired `output_views`, and the cross-check guardrail already forbids referencing
  decompose — so the slot-1 declared rules join the gated batch); (b)
  **`fused_softmax_cross_entropy` = PERMANENT documented whole-shape skip** — its rule is
  reduction-*conditional* (Mean/Sum → `[]`, None → `targets.shape`), outside even the
  reserved vocabulary (needs a conditional constructor); its `fixed(F32)` dtype check is
  live NOW at both reduction points; (c) **`nf4_matmul` double-gated** — its only corpus
  section is `registrable: false` until FDX `AFFINE_BLOCK` lands, out of the oracle's
  reach regardless of tags. Param threading alone flips ZERO whole-shape rules (each gated
  rule also needs its whole-shape constructor). **External state (2026-07-23):** the
  `Dims (0x0B)` + `WithDim (0x0A)` KISS §6.4 extension-registry proposal is **FILED** (the
  KISS coordinator filed the rfc-labeled issue on Fuel's behalf, attributed, per the #57
  process; `Reduce (0x09)` stays reserved — no consumer); Baracuda: no objection +
  declared future consumer + cosigning with a functional-spelling pin; kiss-ref: the
  second dissimilar implementation, timing theirs. On acceptance: implement the two
  constructors + rewrite the 5 gated rules (~9 fused sections) → oracle coverage
  ~16 → ~21 of 22 (FSCE the one honest skip). See
  `docs/outreach/kiss-dims-withdim-extension-registry-filed.md`.
- **`structure_key` independent deriver — D8 freeze-gate; Fuel half DONE (2026-07-19).**
  Fuel's second, **Baracuda-free** implementation of the KISS `structure_key`
  (`fuel-dispatch/src/telemetry/structure_key_derive.rs`) derives the `relu_add` f32 cell
  byte-for-byte AND all non-`gem` families: full §6.1 dtype map, the §6.5-0006 17-family
  op-code set, the reduce field (`rall`/`rlast`/`x<hh>`), rank-N / strided / broadcast /
  vec-width generality. **Remaining, TRACKED (not a forgotten TODO):** (1) the **`gem`
  contraction field** — the deriver DECLINES `gem` until decision **D1** settles its format
  (weight/accumulator/output dtypes + batch); *unblock = the **`sk3`** RFC adopted (D1's
  concrete realization: contraction tuple `<batch>/<wdt>/<acc>/<out>/<mp>`, `f32s`→MathPrecision,
  variant-explicit FP8) → then build the `gem` field + `sk2→sk3` bump + `accumulation_type`*. (2) the **op→category
  classifier** (Fuel `Op`/`OpKind` → `FuelOpCategory`) — the deriver takes the category;
  wiring Fuel's ops into it is the caller-side piece; *unblock = a dispatch-site consumer*.
  (3) a **different-namespace deriver** (CPU/Vulkan-driven) for the strict §6.4-0004 two-impl
  gate — the current deriver is same-namespace `cuda` (proves byte-reproduction, not
  cross-namespace). (4) the live **head-to-head** — waits on Baracuda's `sk1`→`sk2` /
  `cuda:sm89` emit. (5) **MX-primary cells** — the deriver DECLINES an MX-dtype-primary cell (its
  dtype not in the KISS §6.1 closed set) rather than guess a token; *unblock = KISS #9/#32 pin the
  MX keyable (element + block-attribute) form*. Memory `kiss-standard-vs-fuel`.
- **KISS-conformance verify deferrals (in-prose → tracked 2026-07-19).** (a) **Fix #2 — re-mint the
  transcendental correctness-fixtures against the wide-precision corpus:** `fuel-correctness-fixtures`
  self-mints transcendental cells from Fuel's own CPU oracle, which shares the §6.5-0007 hardware-
  precision weakness, so they are not the tight authority they claim; *unblock = kiss-conform ratifies
  + publishes the wide-precision transcendental corpus* (Fix #1 — the live transcendental-band
  comparator, `fkc/verify/ulp.rs` — already shipped `b45565c8`). (b) **`accumulation_type` surfacing:**
  Fuel's accumulator is backend-internal today; D5/`sk3` want it as a declared Contract guarantee + a
  key coordinate — *unblock = `sk3` adopted* (lands with the `gem` field build). Memory
  `kiss-standard-vs-fuel`.
- **SSM autoregressive decode** (`Op::SelectiveScanWithInitState`, feed `last_state` back) +
  **GPU scan dispatch** (wire the ported baracuda mamba kernels to `OpKind`) — the SSM
  long-context-decode payoff; plus the **GraniteMoEHybrid** Mamba branch (currently bails).
- **MoE sparse per-token dispatch** (`Op::TopKRoute` + gather-compute-scatter, the **MoE
  consumer** of the data-determined-shape primitive — today all MoE models route *densely*,
  ~32× over-compute) + **MoE load-balancing / aux-loss** + **soft-MoE / dual-softmax**.
- **MLA decode-time compressed KV cache** + **KV-cache container generalization** (both
  caches hardwire a symmetric K/V pair — the structural blocker for latent/pruned caching) +
  **MLA weight-absorption** + **two-projection attention / QKV pruning**.
- **Batched multi-sequence decode** + **forkable/copy-on-write KV cache** — the substrate a
  downstream search-on-generation (MCTS/beam) wrapper needs; search orchestration itself
  stays a Phase-9 downstream concern by design. Also the substrate a related-but-distinct
  2026-07-10 proposal needs — **cross-branch KV content splicing** for persistent parallel
  "trains of thought" (copy KV/residual content between concurrently-decoding branches, not
  a fork-to-one-winner search) — recommended as a host-level `KvCache` method, not a
  graph/`Op::Branch` change. The block-pool allocator both would benefit from is confirmed
  **absent** (no allocator/refcounting exists behind `Op::PagedAttn` today) — real but
  deferred, now that multi-agent/multi-session serving is a confirmed near-term personal
  roadmap goal (get Fuel's basics working first). Full detail + citations + the
  reevaluation-cost menu: `docs/frontier-architecture-gaps.md` §4. **Increment 1 of the
  serving substrate shipped** (`fuel-core/src/multi_session.rs`): host-side `SessionState`
  + `SessionScheduler` (serial arm = byte-exact oracle, T1 no-cross-session-contamination)
  + a live `BatchedDecode` arm (shared `[K,…]` KV buffer + `flash_decoding` batch, lockstep-
  only; a separate batch=K plan-once graph, so ε-close (logits within 1e-4) and token-identical
  to serial on the tested CPU shapes, not bit-exact; the CUDA bf16 flash-arm parity test is
  local/`#[ignore]`).
  No IR op, no kernel — host orchestration over the existing persistent-decode machinery.
  **Increment 2 — the block-pool allocator — SHIPPED 2026-07-29** (the confirmed-absent keystone
  is now present): pure host-side core `fuel-core/src/kv_block_pool.rs` (`KvBlockPool` — free list +
  refcounts + per-session block tables + refcount-aware evict/splice, model-agnostic, move-ready for
  the Q2 `fuel-inference` move) + the device-backed layer `fuel-core/src/kv_block_pool_device.rs`
  (`DeviceKvPool` — real `n_layers × 2` `[num_blocks, block_size, Hkv, D]` K/V pool buffers,
  `block_table`/`context_lens` materialization for `Op::PagedAttn`, `write_block`/`read_block` byte
  movement, and C-3 device evict/restore that round-trips block bytes device↔host). It delivers, **as
  mechanism**, the three clauses the Increment-1 audit flagged: **C-1** (`free_blocks` +
  `blocks_required` + `capacity`), **C-3-lossy** (`evict`/`restore`/`discard`/`splice`, `Fidelity`
  discriminator keeping the future Exact arm expressible), **C-4** one bite (`kv_bytes_resident`).
  Gated by a pool-routed `paged_attn` parity test (permuted physical layout vs a dense reference) +
  a byte-exact evict→restore round trip. Design-of-record:
  `docs/superpowers/plans/2026-07-29-kv-block-pool-allocator-serving-inc2.md`. Follow-ups: wire it
  under a real consumer (`fuel-inference` policy layer / Lightbulb); byte/dtype-generic block IO +
  live-GPU parity for the CUDA bf16 pool; the deferred named refcounted block-group handle.
  KV-content sharing/splice between concurrently-decoding branches (the residual-stream-donation
  path) rides on `splice` but stays a later increment.
  **Increment 2+ is now scoped by [15-consumer-contract](docs/architecture/15-consumer-contract.md)**
  (new 2026-07-28; annexes + as-built audit in `docs/fuel-consumer-seam.md`): Fuel owns mechanism,
  the consumer owns policy — Fuel never decides *whose* work runs. Clause status (updated 2026-07-29
  as Increment 2 landed): **C-1** capacity advertisement **WIRED** — `SessionScheduler::new` takes a
  `KvBudget` and admits sessions against a `KvBlockPool`; `add_session` reserves ⌈max_seq_len/block_size⌉
  blocks and rejects (typed, total) before building any cache when they don't fit, and `kv_free_blocks`/
  `kv_capacity`/`kv_blocks_required` are the caller's pre-check + `reap_finished` reclaims (`05dc98a3`);
  **C-2** bounded quantum + cancel *absent*; **C-3** state externalization — **mechanism SHIPPED** in the
  block-pool allocator (`evict`/`restore`/`discard`/`splice` + device byte movement, §4 above), not yet
  wired into the scheduler's contiguous-cache path (that rides the paged-storage integration); **C-4**
  measured cost *partial→wired for KV* (`kv_bytes_resident` on the scheduler; `StepReport` still says what
  happened, not what it cost);
  **C-5** constraint admission *absent* as a consumer control — note the ε-close batched arm means a
  logprob-returning consumer has a different requirement from a token-only one. `SchedulePolicy` is
  confirmed correctly Fuel's (equivalent arms = arm selection, not fairness); `run_to_completion`,
  the implicit `Vec`-order FIFO, and `add_session`'s name are consumer-policy shapes to keep out of
  the interface. **Tracked defect — RESOLVED 2026-08-07 (GAP-014, slot-pool KV contamination):** a
  held decode plan was welded to the KV **allocation** it was built against — `base_cache` holds the
  KV storage `Arc`s and neither rebind path ever re-binds them — while the validity key named only
  geometry + model identity. So the serving happy path (retire request A, admit B on a fresh
  same-shaped cache, reuse the plan for speed) had B decoding over A's KV at full speed with a
  plausible distribution and nothing to report. Closed by `decode_shape::KvAllocId`, a
  never-recycled per-allocation id now in BOTH `is_valid_for`s, across all three carriers
  (`KvCache`, `LatentKvCache`, `DeviceKvPool` — the hole was 3× its filed site). The id names the
  ALLOCATION, not the conversation: `truncate_to` preserves it, `clear`/`set_layer` re-mint. Follow-up
  GAP-028 **CLOSED 2026-08-08**: a swap that changes only the allocation now RE-BINDS the held
  plan under a guard instead of rebuilding it, so an admission costs nothing rather than one
  re-optimise. Scoped to the contiguous + latent carriers; the paged half is deliberately DECLINED
  (GAP-046) — one pool serves many sequences and per-request variation already rides
  `block_table`, so a guard there would be pure proof-obligation with no win paying for it.
  This makes [14-lifecycle](docs/architecture/14-lifecycle.md) v0.11 admit a THIRD disposition for
  baked state — re-bind under a guard — alongside "in the validity key" and "cannot change by
  construction", and unlike those two it is dynamic, so it can be wrong and carries a proof
  obligation they do not. Last of the four
  [14-lifecycle](docs/architecture/14-lifecycle.md) Stage-5 invariant violations.
  **Tracked defect — RESOLVED 2026-07-29 (Q2):** `multi_session.rs` moved from
  `fuel-core` to `fuel-inference` (`57abafb4`). The move forced the concrete-model coupling into a
  model-agnostic `DecodeModel` trait (`n_layers`/`n_kv_heads`/`head_dim` +
  `forward_with_kv_context_persistent` + `build_batched_decode_logits`; `LlamaModel` is the first
  impl) — Q2.1 decoupled in place (`e75c7525`), Q2.2 relocated (`57abafb4`). `SamplingStrategy` stays
  a fuel-core type (sheds rather than moves — sampling is consumer policy); the full sampling-location
  reshape (Q5) is still deferred. The scheduler is now consumer-side orchestration where it belongs,
  ready to wire the KV block-pool allocator under.
- **GRPO** + **RLVR** verifiable post-training — greenfield on the existing `fuel-training`
  stack (SGD/AdamW + autodiff + `cross_entropy`); needs the RNG/generator seam (above) for
  group sampling.

**Keystone:** SSM decode, MoE sparsity, and MLA cache all gate on the *data-determined*
half of symbolic extents (per-op-produced runtime counts over fixed-capacity buffers) —
sequence that first; it is also already required by Phase 8.5.

### KISS interop-standard alignment (2026-07-14)

Fuel's kernel seam (FDX + FKC + `fuel-kernel-seam*` + the `SeamHello` handshake) is the
named reference seed for the public **KISS — Kernel Interface Standards Suite**
(github.com/ThinkersJournal/KISS, CC0). A full dimension-by-dimension comparison
(2026-07-14) confirmed deep alignment-by-construction and surfaced a modest set of genuine
deltas — most are small concrete adoption gaps, not design conflicts. Canonical record +
per-item evidence: [`docs/outreach/kiss-conformance-and-divergences.md`](docs/outreach/kiss-conformance-and-divergences.md).
The comparison also corrects two stale internal beliefs: FKC cost expressions **are** wired
to the ranker (`ranker/cost.rs`), and the verification ledger **does** check correctness
(not determinism-only).

**Adopt-from-KISS goals (upcoming work — KISS is genuinely ahead of Fuel here).** These
raise Fuel's structural discipline and are sequenced behind the active frontier but ahead
of the first outside kernel provider (which Fuel's multi-agent-serving roadmap wants):

1. **Clause↔test traceability + build-fail gate** — give every FKC §10 rule + FDX V1–V21
   validator a stable append-only ID wired 1:1 to a named test, with a CI checker that
   fails the build on any untested normative MUST (port KISS's `tools/kiss_trace.py`).
   Directly targets the "shipped-never-wired validator" class (the FKC-verification-gap memory).
2. **Conformance + foreign-reader freeze gate** — raise the bar for stamping a wire type
   "RATIFIED/frozen" from same-author cosigner agreement to: a structurally-dissimilar
   second implementation + a foreign / cross-endian byte read, signed by a role distinct
   from the author. This is the gate that would have caught the `SEAM_MAGIC` byte-order bug.
3. **Reference/decomposition semantic oracle for pinned op edge-cases** — a
   decomposition-derived differential oracle over the primitive activations/atoms so
   NaN-propagation, `-0.0` preservation, integer wrapping, and fmax-vs-max cannot drift
   per backend (the root fix for the whole relu/-0.0, MKL-fmax, gelu-naming family below).
4. **Determinism-class-selected verify comparators** — select the verify comparator from
   the op's declared determinism class + a declared-ULP ceiling, instead of taking `Bound`
   as a free caller argument (which invites a trivial-pass `MaxAbsolute(1000.0)`).
5. **Oracle independence + edge-case corpus in verification** — require the differential
   oracle to share no lowering module with the impl under test (machine-checkable), and a
   deterministic edge corpus (±0/±inf/NaN+payload/subnormals/dtype-extremes). Today
   `verify_candidate` realizes its reference through Fuel's own backend and probes only
   `[-0.5, 0.5)`. (The ULP-distance total-order fix below is the first increment.)
6. **`MathPrecision` reduced-mantissa axis** — an orthogonal `{bit-stable,
   reduced-mantissa-permitted}` attribute so a DAG-first optimizer can offer a
   TF32/bf16-accumulate matmul arm distinctly from a merely loose-ULP kernel (today smeared
   into `max_relative` + free-text notes).
7. **Named `reference_function` + derived `audited_status`** — a precision bound must NAME
   its reference (gelu-erf vs gelu-tanh), and `audited_status` derives from (determinism,
   reference, ULP tier) rather than an authored boolean. Keep Fuel's empirical ledger gate
   (stronger than KISS's syntactic derivation) but adopt the naming.

**Verify-seam repoint follow-up program (kiss-ref advisory cross-check SHIPPED
2026-07-22; follow-ups (i)–(iii) implemented 2026-07-23, branch
`feat/kiss-ref-verdict-integration`).** `verify_candidate` runs a kiss-ref advisory
diff over the `fuel-kiss-ref-backend` adapter crate (git dep on ThinkersJournal/kiss-ref),
advisory-only `kiss_ref_advisory` ledger record — flag-not-verdict per KISS-CONFORM
§6.6-0007 (kiss-ref flags/escalates, never verdicts; recipe-realize stays the interim
verdict authority). Follow-ups now implemented (CPU-verified; live-GPU e2e legs written
`#[ignore]`, run under the exclusive `--features "jit cuda"` gate):

- **(i) `classify_floor_verdict` live wiring** — `verify_candidate_impl` now produces
  `VerifyVerdict::Inconclusive` (→ `IngestOutcome::Flagged` → `ProviderFeedback::on_flagged`)
  when a kiss diff exists but the recipe reference is unusable (realize-Err arms, coverable
  non-f32 numeric claims); the advisory region derives from the registry
  (`decompose.or(runtime_region(claimed_op))` — no static table). A compact kiss summary
  threads into `FlagReport.diff_summary`.
- **(ii) f64/f16/bf16 + multi-node advisory coverage** — the advisory block is region-based
  (adapter `PatternNode → Expr` row-wise `eval_expr`; elementwise, attrs-default, mapped-ops
  only — `SeeThrough`/`Any` decline) and dtype-dispatched (F32/F64/F16/BF16). The comparison
  band follows the kiss-ref tolerance refinement (2026-07-23): single exact op → exact;
  multi-node exact → `Ulp(n_ops−1)`; transcendental region → `Ulp(Σ per-op §6.8 ceilings over
  the transcendental ops + (n_exact−1))`, with raw `max_ulp` always recorded (linear-ULP
  addition is first-order — advisory-only, can flag cancellation-heavy regions spuriously).
- **(iii) IngestionService-level `Flagged` e2e** — service routing `Flagged → on_flagged`
  pinned (CPU) + a live-GPU f64-add escalate e2e (`#[ignore]`, gate 5).

**Seam-migration follow-on (2026-07-23, branch `feat/kiss-ref-expr-migration`).** kiss-ref
promoted the composition Fuel's adapter had been hand-rolling to a **first-class** seam
(`reference_expr` / `diff_expr` + the `_f32`/`_f16`/`_bf16` mirrors it minted for this
consumer). Fuel bumped the pinned rev `b75a748 → e8ae0b5` in lockstep (both `Cargo.toml`
pins — adapter + `fuel-dispatch`'s `jit`; inert — `kiss-ops-vocab` byte-unchanged, `eval_expr`
untouched, only the mirrors added), then migrated **all four float lanes**
(`reference_region_{f32,f64,f16,bf16}` / `diff_region_*`) onto that seam instead of the
verbatim local copy of kiss's diff loop — numerically inert (same `eval_expr` engine), pinned by
migration-equivalence tests keeping the old loop as a bit-exact oracle. The **advisory band
stays Fuel-owned** (`PatternNode → Expr` translation + `region_advisory_tolerance` + typed
declines): kiss-ref supplies the reference numerics, Fuel decides the tolerance — the §6.6-0007
mechanism-vs-verdict line. The **cancellation caveat is unchanged** (linear-ULP addition is
first-order; raw `max_ulp` always recorded). The §6.8 band formula's two hand-maintained copies
(adapter reference-only + live `jit_ingest::advisory_ulp_band`) were consolidated onto one shared
drift-pinning fixture (`fuel_kernel_seam_types::advisory_band_reference_cases()`).

Residual gaps (named, tracked): **static-op advisory** still declines — a static `FusedOpId`
carries no `PatternNode` region until the pinned recipe-grammar migration lands (the
Convergence Tier-2 PatternNode-data migration; static claims aren't elementwise anyway);
**non-f32 recipe VERDICT stays f32-only** — a kiss-coverable non-f32 numeric claim *escalates*
to Inconclusive rather than earning a numeric recipe verdict; **`corpus_verdict` stays
DORMANT** — KISS's v1 exact-byte corpus now EXISTS and is vendored
(`fuel-dispatch/fixtures/kiss-corpus/`, KISS `main` @ `c9153b2`) with a never-panic reader
(`kiss_corpus.rs`, `(add, f32)` populated), but the `corpus_verdict(op, dtype, seed)` seam
carries no candidate output and its `seed` selects a random probe disjoint from the corpus's
fixed inputs, so it cannot be authoritative without evaluating the candidate on the corpus's
own vectors (seam widening — see
[`docs/design-notes/2026-07-23-kiss-corpus-verdict-seam-mismatch.md`](docs/design-notes/2026-07-23-kiss-corpus-verdict-seam-mismatch.md)).

**Latent-bug fixes SHIPPED this pass (KISS's discipline caught these; TDD-verified green):**
`SEAM_MAGIC` byte-order (emitted "MAES", now "SEAM" per KISS-ANNOUNCE §6.1-0004), the
unmanaged `reserved1` alignment padding in `SeamHello` (now an explicit, zeroed, validated
field), the raw-bits ULP distance (now an IEEE total-order mapping, correct across the
sign/zero boundary, de-duped across `ulp.rs` + `seed_cuda_ledger.rs`), and `relu` `-0.0`
preservation (`select(x<0,0,x)`, not `max(x,0)` — the NaN-propagation half was already pinned
2026-07-08). See the Shipped ledger + the 2026-07-14 decisions-log entry.

**Deferred/tracked latent-bug items (real, but need coordination or reachability
confirmation — not safely fixable inline):**

- **`OpTag::Gelu → GeluTanh` seam rename** — KISS pins `gelu`=exact-erf / `gelu_tanh`=tanh
  (PyTorch-aligned); Fuel's bare `Gelu`=tanh inverts the token. The CPU chassis already uses
  a `GeluTanh` marker internally; the collision is at the frozen `OpTag` seam vocabulary
  (~20 cross-crate sites + `op_to_tag`). Coordinate the rename with baracuda + a KISS RFC.
- **`op_to_attrs` load-bearing-attr projection** — Gather axis/OOB, Cast target-dtype,
  Slice/Concat/Pad params fall through to `OpAttrs::default()` (a matcher wildcard), so a
  fusion recipe can bind the wrong node. Needs new fields on the **frozen** `fuel-kernel-seam-types::OpAttrs`
  (a coordinated schema change) + the projection + matcher guard.
- **Integer wrapping vs saturating** — confirm the *live* CPU integer add/sub/mul path
  (the float chassis is float-only; the `dyn_impl` f64-compute-then-saturate path's
  reachability is unconfirmed) before switching to wrapping two's-complement.
- **MKL `vs_max`/`vd_max` NaN-suppression guard** — `binary_op!(vs_max, f32, vsFmax)` binds
  IEEE-fmax (NaN-suppressing) but is currently **dormant** (not wired to any `Op::Maximum`
  dispatch — Maximum flows through the correct NaN-propagating scalar chassis). Guard or
  re-route before `vs_max` is ever wired, so it can't silently violate the pinned convention.

### Shipped ledger

Phases 0–7 and the shipped portions of 7.5/7.6 + Phase C are complete; full detail is in git history (verbose phase blocks condensed 2026-06-25). Highlights:

- **Phases 0–4** — ecosystem compatibility, docs/clarity, use-case crate separation, model-area organization, ergonomics.
- **Phase 5 (Tier 1–3)** — backend modularity + pluggable empirical dispatch (CPU/CUDA/Vulkan/AOCL/MKL binding tables; the Judge).
- **Phase 6 (a–d)** — lazy frontend + single/multi-backend + multi-device routing; paged attention, symbolic autograd, kernel fusion, scheduler integration.
- **Phase 7 / CUDA restructure** — storage-hierarchy refactor; AOCL + oneMKL CPU backends; baracuda CUDA stack (CUTLASS B1–B4, flash, byte-storage fan-out).
- **Phase 7.5 A/B1/B2/G/G2** — graph owns Storage; `Op::Const` unit variant; lazy factories; realize() interface.
- **Phase 7.6 steps 1–3/6/9a–9c(A–E.3.0)** — FusedOpRegistry skeleton + `Op::Fused` arm; binding-table planning-time refactor; KvCache/InferenceContext; Vulkan runtime Device; multi-target realize; pipelined executor unification.
- **Phase C** — runtime route picker + command-buffer capture/replay (bounded per-device Pareto frontier).
- **Adaptive runtime fusion / FKC / self-describing storage** — recipe principle, two-tier extensibility, kernel-seam, declarative fusion engine, SType/Encoding + DLPack/FDX. **FKC now 100% across all three real backends (2026-07-04)**: CPU + Vulkan (13 families) + CUDA (31/31 families, ~429 keys) register from `docs/kernel-contracts/**` via per-backend `LinkRegistry`s — the binding table is contract-sourced end to end (all one-time deferrals resolved). Cost model completed honest + contract-sourced: GPU caps + per-backend throughput + fused-op cost-from-decompose + the **cost-trampoline** (`cost.cost_fn` names a registered `CostFn`, e.g. `flash_decoding`'s infeasibility gate). Baracuda dispatch/miss telemetry schema pinned + emission built + structure-key provider wired. See [10-decisions-log 2026-07-04](docs/architecture/10-decisions-log.md).
- **Judge Layer-2 decode coverage + optimize-time kernel-variant bake (2026-07-04)** — the Judge profiles f16/bf16 + decode-shaped ladders + `OpKind::FlashAttn`; a matmul `SizeClass` reconciliation fixed a latent bug (non-square matmul Judge cells were unreachable/poisoned). `variant_bake` collapses a same-device `Op::Branch` to the arm that wins on **measured** latency at optimize time (the picker resolves placement only) — the mechanism (flash-arm emitter → decode-builder wiring → bake → Judge) is proven end to end; a live CUDA flash win now needs only a bf16 decode path + a live profile.
- **Dispatch-core cleanup Steps A/B/C/D** — backend-stamping + residency/layout-fixup → optimizer; `Op::Branch` arm-selection → the executor (`pick_route` at dispatch); `ExecutionPlan` deleted as a threaded type (`optimize_graph` returns only `OptimizedGraph`; the plan is optimizer-internal; the executor re-derives arm candidates from the graph + registry).
- **KISS seam/verify latent-bug fixes (2026-07-14)** — four defects KISS's wire-discipline caught, each TDD-verified: `SEAM_MAGIC` "MAES"→"SEAM" byte-order (`fuel-kernel-seam-announce` + the C-header mirror in `kernel-seam-interop.md`); explicit zeroed+validated `reserved1` padding on `SeamHello`; IEEE total-order ULP distance (`fkc/verify/ulp.rs`, shared with `seed_cuda_ledger.rs`); `relu` `-0.0` preservation (`fuel-cpu-backend` chassis). See the KISS-alignment subsection for the goals/deferred items and [`docs/outreach/kiss-conformance-and-divergences.md`](docs/outreach/kiss-conformance-and-divergences.md).
- **`Op::Scan` Phase 1 shipped (2026-07-15) — G3 CLOSED.** Fuel's first sub-graph-carrying
  primitive (`Op::Scan` / `Op::ScanPlaceholder`, body encoded as the node's own `inputs`,
  single-carry v1, always a 2-slot bundle); `selective_scan` + `ssd_chunk_scan`'s `decompose`
  now emit it instead of surfacing the G3 basis gap — `decompose` is total over genuine
  primitives for the whole registered fused-op set. `Op::Scan` is a base-map terminal with no
  native kernel; the fused CPU/CUDA kernels remain the executed path — the payoff is
  optimizer-basis closure, not new runtime capability. **Phase 2 shipped (2026-07-16):**
  early-exit (predicate-over-carry trailing input + `Tensor::scan_until` + the
  `drive_scan_until_final_f32` realize-barrier step driver), BPTT differentiability
  (`lower_scans_for_backward` decompose+unroll pre-pass; SSM ops now `BackwardKind::Decompose`,
  truncated to the static `bound`), and the first non-SSM consumer (`hopfield_retrieve`, Modern
  Hopfield associative memory — converges early + is BPTT-differentiable). Still no `Op::Scan`
  native kernel (the slot-1/`last_state` OOB blocker stays out of scope on purpose). See
  [10-decisions-log 2026-07-15](docs/architecture/10-decisions-log.md) +
  [2026-07-16](docs/architecture/10-decisions-log.md).

- **Missing-fusion telemetry (none today; build closed-world first).** The Judge feedback
  above measures fusions Fuel *performed*; it says nothing about fusions Fuel
  *wanted-but-lacked*. There is **no missing-fusion signal at all** today — "no rule fired" is
  identical across every primitive node, so Fuel can't tell a deliberate primitive from a
  fusion it would have wanted but had no kernel for. Closing this needs a **new graph-layer
  hook** plus the base-emission seam (`structure_key` is still a stub; no `DispatchRecord` is
  emitted yet). Sequencing per the 2026-06-20 decision ([G5 in
  10-decisions-log](docs/architecture/10-decisions-log.md); canonical in
  [`docs/session-prompts/baracuda-telemetry-plan.md`](docs/session-prompts/baracuda-telemetry-plan.md)
  §9): the **closed-world `FusionMissRecord`** — a recognized fusion-eligible chain realized as
  N primitives because the kernel was absent (reason `NoBackendKernel`, against **known**
  `FusedOpId`s) — is the v1 **headline**, built **first**, because its consumer already exists
  (append a `BindingEntry`, Tier 1). The **open-world co-occurrence `SequenceRecord{fused_as:
  None}`** (a frequent realized chain matching *no* known identity, found by observation not
  subgraph enumeration) is **deferred**, because its consumer is the Tier-2 trusted declarative
  registration. Fuel never enumerates the subgraph space and never searches for a whole-model
  fusion.

---

## Benchmarking

*Gated on Fuel running inference end-to-end without major speedbumps (the obviously
incomplete parts finished). Required as proof — not belief — before any non-alpha
release. This is the out-of-repo enforcement of the lazy-only performance bet stated in
[01-identity](docs/architecture/01-identity.md) and [09-non-goals](docs/architecture/09-non-goals.md):
since the in-repo eager path was retired in Phase 7.5, the comparator is now external.*

The thesis is that the lazy DAG should keep up with or **outperform every eager
framework**, because it picks the best available implementation of each op and adapts to
live device state — things eager code largely cannot do. That claim is currently
unproven; this program makes it falsifiable.

- **First yardstick — Candle eager** (fuel's near-unchanged fork parent; near-zero
  porting cost): same checkpoint, same machine, fuel lazy-realize vs Candle eager.
  Target: Candle's eager looks slow by comparison. This is the floor.
- **Beyond Candle**: apples-to-apples against llama.cpp (GGUF/quantized), PyTorch, ONNX
  Runtime, and Burn/CubeCL + tch-rs on the parts each does well — per-op where
  meaningful and end-to-end tokens/sec.
- **Instrumentation the program needs (build alongside the harness)**:
  - Per-token cost breakdown on a real anchor (graph build / topo / plan / dispatch /
    kernel wall) on CPU and GPU — the honest test of whether the planner overhead is
    amortized. Capture as a non-`#[ignore]` perf artifact, not folklore on stderr.
  - A roofline floor (model bytes ÷ measured memory bandwidth) so "DRAM-bound" claims
    are checkable rather than asserted.
  - A quantized end-to-end tokens/sec number (Q4_0 path) — on a bandwidth-bound system
    this is the largest single lever and the number every external evaluator asks first.
- **Make a budget constitutional once measured**: e.g. "plan + topo + coverage cost ≤ X%
  of measured decode-step time on the anchor suite," enforced as a perf gate, so the
  lazy-only bet has a numeric falsifier in-repo.

---

## Identity

Fuel is a **layered Rust ML framework**.

It aims to feel small at the bottom and powerful toward the top, without forcing
any particular use case on the layers below it. Someone doing tensor math should
not carry inference orchestration. Someone implementing a model architecture should
not need a runtime. Someone building a complete inference pipeline should have the
building blocks readily available.

The ecosystem should be easy to exit early. You should be able to stop at exactly
the layer you need.

---

## Layer Model

The ecosystem is organized into six conceptual layers. Dependencies within the
stack flow downward only. No lower layer may depend on a higher one.

```text
┌────────────────────────────────────────────────────────────────────────────┐
│  Use-Case Orchestration                                                    │
│  fuel-inference, fuel-training  (leaf crates — nothing depends on     │
│  either of these)                                                          │
│                                                                            │
│  fuel-inference: sampling, logits processing, KV-cache policy,          │
│  token generation loops, speculative decoding, batching, streaming         │
│  decode, cancellation, inference session abstractions                      │
│                                                                            │
│  fuel-training: training loops, gradient accumulation, LR scheduling,   │
│  gradient clipping, mixed precision policy, run-time checkpointing,        │
│  training session abstractions                                              │
├────────────────────────────────────────────────────────────────────────────┤
│  Models                                                                    │
│  fuel-transformers  (will be restructured internally)                   │
│  Architecture config structs, layer composition, forward passes,          │
│  weight name mapping. No serving logic, no decode loops, no sessions.     │
├────────────────────────────────────────────────────────────────────────────┤
│  IO                                                                        │
│  fuel-core (safetensors, npy, pickle), fuel-onnx                      │
│  Bidirectional data exchange across any boundary: files, network,         │
│  memory buffers. Checkpoint load and save, format translation, ONNX       │
│  import/export, HF Hub integration glue, config normalization,            │
│  tokenizer glue. To be consolidated.                                       │
├────────────────────────────────────────────────────────────────────────────┤
│  NN                                                                        │
│  fuel-nn                                                                 │
│  Layers, losses, optimizers, parameter utilities, initialization,         │
│  VarBuilder, VarMap. No model-family assumptions. No serving abstractions.│
├────────────────────────────────────────────────────────────────────────────┤
│  Foundation                                                                │
│  fuel-core                                                               │
│  Tensors, devices, dtypes, shapes, layouts, base ops, autograd,           │
│  storage backends, error types. No tokenization. No model-level concepts. │
├────────────────────────────────────────────────────────────────────────────┤
│  Backends / Kernels                                                        │
│  fuel-cuda-kernels, fuel-metal-kernels, fuel-flash-attn, fuel-ug  │
│  Hardware and runtime targets (CPU, CUDA, Metal) plus the concrete        │
│  mathematical kernel implementations for each: matrix multiply, conv,     │
│  flash attention, quantized dot products, SIMD/BLAS dispatch. This layer  │
│  knows tensors as shaped memory regions and operations as mathematical     │
│  functions over those regions. It has no concept of layers, models,       │
│  losses, tokens, or any other ML abstraction.                              │
│                                                                            │
│  Foundation: `BackendDevice` and `BackendStorage` traits already exist    │
│  in fuel-core. CUDA and Metal are behind feature flags. Phase 5         │
│  formalizes these as a published plugin contract and opens the type for   │
│  third-party backends.                                                     │
└────────────────────────────────────────────────────────────────────────────┘
```

> **As-built note (2026-07-29) — this diagram and the "Current State" analysis below are a
> target/historical snapshot, not the present workspace.** Verified against the crate list:
> **`fuel-nn` does not exist** — no directory, no manifest entry; the NN surface lives in
> `fuel-core` as `lazy_nn_varbuilder::VarBuilder`, `lazy_nn_varmap::VarMap`, and
> `lazy_nn/` (linear, embedding, norm, activation, lora, quantizable_linear, moe, conv,
> sequential). **`fuel-core-types` no longer exists** — it is now `fuel-ir`; `fuel-hardware`,
> `fuel-memory` (ex-`fuel-storage`), and `fuel-backend-contract` all exist; `fuel-core` itself
> has not yet dissolved. The "inference and training concerns are scattered" item below is
> **partly resolved**: `kv_cache` and `sampling` moved to `fuel-core`, and **`fuel-inference`
> now exists** as a real crate (6.2k LOC, 153 tests — though with zero consumers as of this
> date). Recorded because a consumer-facing port survey found these sections naming crates the
> workspace no longer has. See [`docs/architecture/02-layers.md`](docs/architecture/02-layers.md)
> §Retirement trajectory for the authoritative status.

---

## Current State

### What is working well

- Dependency direction between published crates is already mostly correct.
  `fuel-core` does not depend on `fuel-nn`, which does not depend on
  `fuel-transformers`. The early-exit property is structurally present.
- `fuel-core` has a reasonable backend abstraction (CPU, CUDA, Metal).
- Quantization has a meaningful home in `fuel-core::quantized`, better
  centralized than most frameworks at a comparable stage.
- The breadth of model implementations in `fuel-transformers` is genuinely
  impressive and is a key asset.

### Identified problems

**Documentation**
The primary way users currently learn non-trivial usage patterns is by reading
examples. Examples are useful but they are poor architecture documentation. Most
public API items across all crates lack doc comments or runnable examples in the
documentation itself.

**Ergonomics / developer experience**
Using Fuel non-trivially requires understanding `Var`, `VarBuilder`, `VarMap`,
device management, and dtype handling simultaneously before anything works. There
is no convenience path for common cases. Error messages often carry the right
information but do not always present it in a form that immediately tells you what
went wrong and how to fix it.

**Inference and training concerns are scattered**
`fuel-nn` currently contains `kv_cache.rs` and `sampling.rs`.
`fuel-transformers` contains `generation/` with `LogitsProcessor` and the
`Sampling` enum, as well as a `pipelines/` directory intended for orchestration
logic. These are inference-specific tools with no natural home below the
orchestration layer. The consequence is that `fuel-nn` carries inference
weight that pure layer-building users never need.

**`fuel-transformers` is a flat namespace with no internal structure**
Over 100 model files coexist in a single `models/` directory alongside their
quantized variants, shared utilities, object detection helpers, and generation
logic. There is no enforced separation between architecture definitions and
runtime glue. This will worsen as more models are added.

**No top-level guide**
There is no document that routes a new user to the right crate based on their
intent. New users are expected to infer the architecture from the repository
layout and the README example list.

---

## Planned Work

Work is organized into ten phases. Later phases depend on earlier ones being
stable but phases within a group can proceed in parallel. Phase 9 is
extension hooks for downstream consumers (specifically: an out-of-tree
agentic library); not gated on the others, just gated on a real consumer
asking for them. Phase 10 is equivalence-rewrite search; gated on the
eager-retirement program finishing and the picker accumulating real
Judge telemetry.

---

### Phases 0–7 + CUDA restructure — ✅ shipped (condensed)

The detailed per-phase blocks for Phases 0–7 and the CUDA stack restructure were condensed on 2026-06-25; they are summarized in the **Shipped ledger** under "Current frontier" above, and the full original text is in git history. The live and deferred work begins at Phase 7.5 below.

#### Opportunities baracuda now unblocks (future work items)

These aren't blocking anything; they're capabilities baracuda exposes
that cudarc didn't, pitched for later roadmap consideration.

- [ ] **CUDA Graph capture + replay** for Phase 6's `realize()` hot
      path ([`baracuda-driver/src/graph.rs`]). Decode-heavy LLM
      inference runs the same attention + MLP sequence per token —
      prime territory for `cuGraphCapture`. Expected payoff: cuts
      per-token kernel launch overhead. Non-trivial to integrate
      because it changes the executor hot path; needs its own
      design pass. **This is the cheaper step *below* whole-model
      fusion** (captured-run replay captures most of the
      launch-overhead win without fusing compute, per [G6 in
      10-decisions-log](docs/architecture/10-decisions-log.md) and
      [11-persistence](docs/architecture/11-persistence.md)).
      Whole-model / **megakernel** fusion is a real technique above
      it but **narrow, last, and the highest-risk target — never the
      default**: it wins *something* even over an ideal CUDA Graph
      (inter-kernel scheduling bubbles + cross-boundary pipelining),
      but the benefit curve **turns over** (fixed launch geometry;
      kernel-global register allocation imposes the worst region's
      footprint everywhere — internal sub-kernels do **not** fix
      this; per-shape JIT combinatorics). "Bigger fusion = better"
      is **not monotonic** — do CUDA-Graph replay first.
- [ ] **Stream-ordered mempool allocation**
      ([`baracuda-driver/src/mempool.rs`]). Fuel today allocates
      a fresh `DeviceBuffer` per op output. A `CUmemoryPool` with
      trim / release policies would recycle within a stream. Needs
      buffer-lifetime analysis vs stream semantics — not a
      mechanical swap.
- [ ] **CUDA ↔ Vulkan P2P zero-copy** via
      `ExternalMemory::import` (baracuda) + `DeviceMemory::get_win32_handle`
      (vulkane). Was estimated ~2-3 days before baracuda existed;
      now closer to ~1-2 days since both sides expose the primitives.
      Vulkane also exports **semaphore** handles
      (`Semaphore::get_win32_handle`/`get_fd`), so cross-API *sync* is
      available too — not only memory sharing.
      **Do not read the gate as merely unmet — on this hardware it is
      UNMEASURABLE (2026-07-31).** The stated gate ("someone runs a real
      multi-device model and finds the PCIe round-trip is the bottleneck")
      cannot become true on the dev rig: it is an RTX 4070 + AMD **iGPU**,
      and an iGPU's memory *is* system memory, so host staging is already
      near-optimal and eliminating it would buy close to nothing. The
      payoff is **dGPU↔dGPU**, and there is no dGPU pair on the machine to
      measure it on. Stays backlogged until such hardware exists — this is
      "not yet tried", not "tried and not worth it". (Analysis from the
      Vulkane session; re-derived and confirmed Fuel-side.)
- [ ] **Launch attributes** — cluster dims, programmatic stream
      serialization, priority ([`baracuda-driver/src/launch_attr.rs`]).
      Opportunistic tuning for specific kernels; measure before
      applying.
- [ ] **nvJitLink** for runtime kernel specialization. Matters when
      Fuel starts doing LoRA fusion, per-shape attention kernels,
      or other "generate a kernel for this exact problem" flows.
      Speculative. This is the same surface the adaptive runtime-fusion
      loop ([G7 in 10-decisions-log](docs/architecture/10-decisions-log.md))
      uses when a trusted backend JIT-synthesizes a Fuel-chosen region.
- [ ] **Kernel-cache pruning policy** (gates the JIT loop's disk
      growth; [G8 in 10-decisions-log](docs/architecture/10-decisions-log.md),
      canonical home [11-persistence](docs/architecture/11-persistence.md)).
      Once Fuel starts persisting JIT-synthesized kernels, the cache
      grows. The policy: **prune rarely** — likely not at all at first.
      Evict only kernels that **lose across *every* model** (a
      "never-useful no matter which model considers it" proof), and
      only **under space pressure**, gated behind a **developer-set
      max-kernel-drive-space cap** — so a currently-*shadowed* kernel
      that might win under a different cover or in a different model
      **stays** while there is room. **Never prune on a single model's
      loss** ("winning" is relative to the current kernel set, i.e.
      shadowing). Speculative; lands with the JIT loop, not before.

#### Post-Phase-6 dead-op audit

- [ ] **Audit candle-heritage ops for unused code paths** once the
      Phase 6 anchor suite (Llama 3, Whisper, ConvNeXt, SD 1.5,
      YOLOv8, BERT, Qwen2-MoE) runs against the CUDA backend. Ops
      that no anchor exercises (suspected: `upsample_nearest1d`,
      `index_add`, `elu`, `const_set`) can be removed. Not done
      pre-emptively — too easy to delete something a model actually
      needs.

---

### Phase 7.5 — Core simplification: lazy-only execution, graph-rewrite autograd, and crate fissioning

> **Status (2026-06-25):** Shipped — A (fuel-formats), B1 (realize stubs), B2 (graph-owned factories), G (Graph owns Storage), G2 (`Op::Const` unit variant). Deferred (backlog, behind the active frontier) — B3–B6 (remaining op-method → lazy sweep), C (graph-rewrite autograd), D (in-place-as-optimization), E (crate fission), F (layout contracts). The detail below is retained as the backlog spec.

*Structural cleanup that follows naturally from the now-complete
backend-agnostic refactor (the 15-step plan, branch tip f0c00233,
2026-05-01). Not urgent in the sense of blocking other work, but
high-leverage: each piece removes a tax that every consumer pays
today and unlocks downstream phases. Best done after Phase 7
stabilises and before Phase 8 lands new kernel-layer code that
would otherwise have to absorb the changes mid-flight.*

#### Why Phase 7.5 exists

The backend-agnostic refactor proved the architecture: `fuel-core`
no longer names any backend, every backend interacts through
`DynBackendStorage`, and the lazy stack (Phase 6) is the substrate
for empirical dispatch (Phase 6b), scheduler-driven residency
(PRs #1–#4), and multi-backend Router (PR #5). Three structural
debts remain that the previous architecture left in place:

1. **Two execution paths (eager + lazy) that do the same job.**
   `Tensor::matmul` runs immediately via `Storage::matmul`; the
   lazy stack builds a graph and dispatches via Router/Executor.
   The lazy path is strictly more capable (Judge dispatch,
   ResidencyEvictionRule, ConstLoweringRule, future fusion). Every
   op currently has to work in both modes — compile-time tax,
   test-matrix tax, source of subtle drift between paths.
2. **Autograd entangled with `Tensor` and the `Op` enum.**
   `Tensor_` carries an `op: BackpropOp` field that every
   inference path pays for, and `Op` does double duty as forward
   IR (used by the lazy graph) and backward tape entry (used by
   `.backward()`). Inference consumers (Lightbulb, embeddings,
   retrieval, oracle test runners, quantized-only paths) inherit
   the autograd cost for nothing.
3. **Inplace ops are a user-facing decision rather than an
   optimization concern.** `InplaceOp1/2/3` in `fuel-core-types`
   forces users to choose `relu_inplace` vs `relu` based on a
   correctness model they have to track manually, and in
   differentiated regions inplace can silently produce wrong
   gradients.

These three sit on top of a fourth pressure: `fuel-core` itself is
becoming a kitchen sink. `fuel-quantized` and `fuel-conv` already
fissioned out under real consumer pressure; the remaining
contents (Tensor + autograd + eager dispatch + loaders +
custom_op extension hook + indexer) split along clean
consumer-boundary lines.

#### Architectural decisions

**Single execution path: lazy-only + explicit `.realize()`.**
Drop eager mode. `Tensor::matmul` and every other op build a
graph node; values are produced when the user calls
`.realize()` / `.materialize()` / `.item()` / similar. The lazy
stack already has every capability eager has plus residency,
empirical dispatch, and (future) fusion. The cost of the change
is ergonomic — print-debug, dynamic control flow on tensor
values, and interop with non-Fuel code all need an explicit
materialisation call. JAX has demonstrated this idiom is
learnable. Single path also collapses the autograd story to
"is this graph differentiated?" — no "is autograd active and is
the op eager?" matrix.

**Option 2 with the lazy graph as the tape.** Autograd becomes a
graph rewrite over the forward IR, not a separate tape data
structure. The lazy graph already has every property a tape needs
(ordered nodes, input dependencies, op metadata). Backward is a
graph transformation that walks the forward graph in reverse and
emits backward nodes, then the unified graph is executed via the
same `fuel-graph-executor`. Backward implementations live in
`fuel-autograd` (or co-located per-op alongside their forward
definitions in their owning crate — `fuel-conv`, `fuel-quantized`,
etc.). `Tensor_` drops the `op: BackpropOp` field and the
`is_variable` flag. `Op` becomes pure forward IR; the lazy stack
and autograd both consume it.

This choice has strong synergy with what is already shipped:
- Phase 6b probe/judge/dispatch — backward ops are ordinary ops,
  dispatch through the same Judge/DispatchTable. Backward of
  `matmul(A, B)` is just `matmul(grad, Bᵀ)` and `matmul(Aᵀ, grad)`.
- PRs #1–#4 scheduler-driven residency — unified forward+backward
  graph means the scheduler sees full activation lifetimes and
  computes correct eviction. The destructive-input metadata on
  `Op::Release` already prevents forward eviction of tensors
  needed for backward; activation checkpointing falls out almost
  for free.
- P5 tiered residency — activations evicted during forward can
  be `fault_back`'d when backward consumes them; the planner has
  the dependency visible.
- ConstLoweringRule — backward graphs are also const-foldable.
- Higher-order gradients (`grad(grad(f))`) work because the
  backward graph is itself differentiable.

**Inplace as an optimization concern, not a user concern.** A
graph optimizer pass runs liveness analysis on the unified
forward+backward graph and rewrites in both directions:
- *Inplace-IN*: a non-inplace op whose input has no remaining
  consumers (no other forward use, no backward dependency) is
  swapped to its inplace variant. Free buffer reuse.
- *Inplace-OUT*: an inplace op whose input is needed elsewhere
  is swapped to its non-inplace variant. The original inplace
  marker becomes a hint the optimizer is free to ignore.

User consequence: the same source code is correct in inference
and training. Inplace is a perf hint, not a semantic constraint.
Inference paths get every inplace win the analysis can find;
differentiated paths get correctness for free; mixed regions
handled by the same liveness pass with no special-casing. This
generalises JAX's `donate_argnums` from a user annotation to an
optimizer-inferred property.

**Fissioning `fuel-core` along consumer boundaries.** Each split
is justified by a class of consumer that uses one side and not
the other:
- `fuel-tensor`: `Tensor` + eager-dispatch methods (now
  graph-builder methods) + indexer + custom_op + scalar helpers.
  Consumer: anyone who wants the tensor surface without autograd
  (Lightbulb, embedding/retrieval pipelines, oracle runners).
- `fuel-autograd`: tape-as-graph-rewrite + backward registration
  machinery + `.backward()` API. Consumer: training pipelines.
- `fuel-formats`: pure parsers for safetensors, pickle, GGUF,
  GGML, and imatrix wire formats. Operate on `impl Read` /
  `&[u8]` / `Cow<[u8]>` — knows about format structure, knows
  nothing about `Tensor`, `Device`, or `Storage`. Depends only
  on `fuel-core-types` (`DType`, `Shape`, `GgmlDType`).
  Consumer surface: anyone who needs to read or write these
  formats over *any* transport — file, mmap, HTTP, S3, Unix
  socket, shared-memory, network IPC. Splitting parsers from
  transport is the structural prerequisite for streaming weight
  load, inter-process tensor exchange (Fuel ↔ Lightbulb ↔ mlmf
  using safetensors as the wire schema), `RemoteHostStorage`
  (Phase 7c), and HF-ecosystem interop without bolting on
  adapters.
- `fuel-loaders`: file-transport adapters built on `fuel-formats`
  — `from_path`, `from_mmap`, `MmapedSafetensors`, etc. Builds
  `Tensor` / `QTensor` from parsed format output. Depends on
  `fuel-tensor` (post-E) and `fuel-formats`. Consumer: model-
  conversion tools and the initial-load path; not needed by
  inference-with-pre-loaded-weights or by network/IPC consumers
  that go directly through `fuel-formats`.
- `fuel-net` / `fuel-ipc` (out of scope for 7.5, natural
  follow-ons): same shape as `fuel-loaders` but over network /
  IPC transports respectively. Mentioned only to make clear that
  the `fuel-formats` / transport split is doing real work
  beyond breaking a circular dependency.
- `fuel-core` retains the umbrella facade role — re-exports the
  common API for ergonomics, like `tokio` re-exporting from
  `tokio-*`. Most users keep depending on `fuel-core` directly;
  internal consumers depend on the leaf crates.

The stopping rule: a crate boundary is justified only when there
is a class of consumer that uses one side and not the other.
Indexer and scalar helpers have no consumer asking for them
without `Tensor`, so they stay folded into `fuel-tensor`.

**Graph optimizer architecture: transactional rewrites on a single
primary graph.**

Optimization is a pipeline of rule-driven graph rewrites. Two rule
families:

- *Lowering*: high-level op → primitive subgraph (exposes fusion
  opportunities to later passes).
- *Fusion*: recognized primitive subgraph → fused op (recovers or
  improves on the original-flavour kernel).

Lowering and fusion are two halves of one machine. Rules ship as
`(matcher, rewriter)` in one registry. The lowered form is
intermediate IR, not an execution form — runs see the
post-optimization graph.

*The recipe principle — both directions are mandatory, the recipe
always ships.* Per the 2026-06-20 adaptive-runtime-fusion decision
([G1/G2 in 10-decisions-log](docs/architecture/10-decisions-log.md)),
every fused op carries a primitive recipe in **both** inverse
directions — a `decompose` (fused → primitive subgraph) **and** a
`pattern` (recognize that subgraph, re-fuse) — and **both are
mandatory**. A fused op with no recipe is an **opaque island**:
invisible to base-map analysis (the missing-fusion / co-occurrence
telemetry can't see across or inside it) and impossible to re-fuse.
So the recipe **always ships with the op** — it is never deferred
"until intermediates fit." That earlier framing (a lowering rule may
ship before its fusion partner, withholding `decompose` for the
memory-blowup ops below) is **withdrawn**: `decompose` is **total**
and **never-panic** (a primitive decomposes to itself — the
recursion's fixpoint), the **base map is the fixpoint of `decompose`
over every node**, and optimization itself = *lower-to-base-map +
find-best-cover*. An op that won't decompose therefore **breaks the
optimizer** (it leaves a blind island in the base map the cover
search can't enter), not merely a downstream JIT feature. The three
current panicking decomposes (`nf4_matmul`, `flash_attn`,
`selective_scan`) are **bugs to fix**, not a permanent "wait for
fusion partner" category. *(Update 2026-07-03: this is superseded —
none actually panicked (a prior G2 pass had already converted them to
self-returns), and the residual work was supplying the recipe, not
stopping a crash. `nf4_matmul` + concrete-`k_len` `flash_attn` now
carry total recipes; symbolic-`k_len` `flash_attn` and `selective_scan`
are the two remaining **documented basis gaps**, each with a named
missing primitive (a `DynScalar`-length slice; a higher-order `Scan`
op) — surfaced never-crash gaps, not "bugs to fix". See the
[2026-07-03 decisions-log entry](docs/architecture/10-decisions-log.md)
and [`docs/frontier-architecture-gaps.md`](docs/frontier-architecture-gaps.md).)*

- *Recipe ships now*, primitive intermediates linear in input:
  SoftmaxLastDim, RmsNorm, LayerNorm, NormLastDim, RoPE,
  FusedLinear, Affine, Clamp, PowI.
- *Recipe still ships now even though intermediates blow up* (the
  base map prefers the fused form via cost, but `decompose` must
  exist and be total so the op is on the base map at all): MatMul →
  outer-product-then-reduce (×K), Conv2D → im2col+matmul
  (×Kh×Kw), FlashAttn → softmax(QKᵀ)V (materializes [N,N]
  attention matrix), QMatMul → dequant+matmul (eats the
  quantization memory win). The lowered form being memory-expensive
  is a *cost-ranking* fact (the optimizer keeps the fused path), not
  a license to omit `decompose` — omitting it is what produces an
  opaque island.

*Transaction model.*

- One primary graph in steady state.
- A working copy exists only during open transactions or briefly
  during commit-with-drain.
- Transaction = unit of consistency: at commit, all touched nodes
  are in a runnable state. No half-applied rules ever visible.
- Default granularity: one rule application. Coarser (per-pass,
  whole-pipeline) allowed when the optimizer can prove correctness
  across the larger atomic unit.
- Commit triggers: fixpoint (no more rules apply at current rule
  set) or budget exhaustion (deadline hit; used for cold-start
  TTFT).

*Switching semantics on commit.*

- New runs always start on the most-optimized version.
- In-flight runs switch at the next node-execution boundary if and
  only if the optimization is entirely ahead of the run's frontier.
  Otherwise the run finishes on the old graph; the optimization is
  preserved for subsequent runs.
- The conservative-ahead-of-frontier rule isn't just for
  approximate optimizations — lowering and fusion change node
  count and identity, so cached storage from already-executed
  nodes can't be remapped to the new graph in general. Switching
  backward across the frontier would require re-running upstream
  nodes to rebuild missing storage, which negates the in-flight
  optimization win.
- Multi-node device-queue case: when a backend queues N nodes'
  worth of ops asynchronously, the run's "currently executing"
  set is N nodes, not 1. Switching is gated on optimization being
  downstream of all queued nodes.
- Old graph lifetime ≤ max(longest queued-node duration,
  transaction duration) post-commit. Dropped once all in-flight
  runs have switched or finished.

*Concurrency.* Active graph as `Arc<Graph>` for lock-free runner
reads. Optimizer mutates a working copy uncontended. Commit =
atomic store on the active reference. No hand-rolled lock-free
machinery needed beyond `Arc` swap.

*Memory model.* Full-clone-then-mutate per transaction. CoW
between graph versions is profile-driven future work — only
attractive when graphs are 100K+ nodes and transactions touch
few-node deltas, neither of which fuel's typical inference graph
(low thousands of nodes) satisfies.

*Out of scope.* Approximate optimizations — mixed-precision
lowering (F32→BF16 hotspots), FP reassociation `(a+b)+c →
a+(b+c)`, fast/approximate intrinsics. These require explicit
approximation-budget semantics and don't fit the
strict-equivalence transaction model. Deferred until that
semantics layer exists.

*Phasing.*

- **PR 3 (next)**: rule-registry framework + first lowering/fusion
  rule pair (SoftmaxLastDim ↔ 7-node primitive subgraph) +
  synchronous "optimize-to-fixpoint, single graph" loop. No
  transactions, no snapshots, no concurrent optimization. Entry
  point factored cleanly so wrapping it in transactions later is
  mechanical.
- **Subsequent PR**: transaction snapshots, in-flight switching
  with ahead-of-frontier rule, multi-queued-node frontier
  accounting.
- **Later PR**: budget-exhaustion mode + cold-start TTFT path.
- **Future**: hot-path re-optimization triggered by execution-count
  or profiling; per-node optimization-tier tracking if needed for
  finer-grained scheduling decisions.

Work items D (inplace-as-optimization) and F (layout-tracking
pass) become rule families on this framework once it exists.

*Forward reference*: the rule registry's hand-written rules
(SoftmaxLastDim's lower/fuse pair) will become *auto-generated*
once Phase 7.6 (FusedOpRegistry) lands. Each FusedOpEntry's
`decompose` + `pattern` produce a lowering rule and a fusion rule
declaratively — the same data viewed in opposite directions (the
recipe principle, [G1 in 10-decisions-log](docs/architecture/10-decisions-log.md)).
Because both directions are mandatory and `decompose` is total, the
registry can derive the lowering/fusion pair for *every* fused op
one-to-one; an entry that supplied only one direction would be an
opaque island. The hand-written form remains as an escape hatch.
See [Phase 7.6](#phase-76--fusedopregistry-open-registry-for-fused-ops-closed-enum-for-primitives)
for the registry refactor that consumes this framework.

#### Work items

**A. `fuel-formats` extraction — transport-independent format
parser layer** (ships first, has zero `Tensor` coupling, unlocks
streaming / IPC / network use cases independent of the rest of
7.5).

The original framing here was "fission loaders for compile-time
leanness." Inspection in 2026-05-02 revealed the bigger seam:
loader files today couple format-parsing (header layout, block
decode, opcode interpretation) to transport (file path, mmap,
`Read`) to construction (`Tensor` / `QTensor` from parsed
metadata). Cutting only the construction join — what work item
A originally described as a `fuel-loaders` crate — would create
a circular dependency on `fuel-core` (loaders need `Tensor`;
`fuel-core` would re-export loaders for back-compat). Cutting
the parse-vs-construct join instead lifts a transport-agnostic
parser layer that has standalone value. See "Fissioning
fuel-core" above.

- [x] Create `fuel-formats` crate. Pure-Rust parsers for
      safetensors, pickle, GGUF (file + mmap), GGML, imatrix.
      API operates on `impl Read` / `impl Seek` / `&[u8]` /
      `Cow<[u8]>` and returns format-typed structs. Depends only
      on `fuel-core-types` (`DType`, `Shape`, `GgmlDType`).
      *Shipped 2026-05-02 (commits be7066f8 → 8f2614bb on branch
      refactor/step-11-quantized-kernels). Module surfaces:
      `imatrix::parse`, `ggml::{Header, RawTensor, read_one_raw_tensor}`,
      `gguf::{Content, TensorInfo, Value, ValueType, VersionedMagic}`,
      `pickle::{OpCode, Object, Stack, TensorInfo, read_pth_tensor_info}`,
      `safetensors::{SafeTensors, TensorView, MmapedFile}` (re-exports
      from upstream + the mmap convenience).*
- [x] Migrate the parser bodies out of
      `fuel-core/src/safetensors.rs`, `pickle.rs`,
      `quantized/gguf_file.rs`, `gguf_mmap.rs`, `ggml_file.rs`,
      `imatrix_file.rs`. Leave thin Tensor-construction wrappers
      in `fuel-core` (today's `safetensors::load(path, device)`,
      `pickle::read_all(path)`, etc.) that call `fuel-formats`
      to parse and then build Tensors. Public API of `fuel-core`
      unchanged.
      *Shipped — fuel-core's loader files now thin orchestrators
      that re-export format types and add Device-aware tensor
      construction. 126 fuel-core unit tests pass throughout.*
- [x] Add `fuel-formats` to the workspace. Verify the parser
      surface is complete by removing every byte-level read
      from `fuel-core` and confirming the wrappers don't
      reach for `byteorder` / `safetensors-rs` / etc. directly.
      *Shipped — workspace registration in commit be7066f8.
      fuel-core's remaining `byteorder`/`safetensors` imports are
      legitimate: NPY format (separate, not migrated), GGUF write
      path (Tensor-aware), and lazy_* materializers (Tensor-aware).
      No dead deps to remove from fuel-core/Cargo.toml.*
- [~] Round-trip test against the Phase 6 anchor weight set
      (BERT, ConvNeXt, Whisper, SD CLIP, SD VAE, Qwen2-MoE,
      YOLOv8) — same loaded tensors, byte-equivalent buffers,
      across both file and `Cursor<&[u8]>` paths.
      *Partial — `fuel-formats/tests/transport_independence.rs`
      exercises all 5 parsers with synthetic in-memory buffers and
      proves zero-filesystem operation. Real anchor-weight round-trip
      is gated on having those binary fixtures available in-tree;
      defer until Phase 6 anchor weights land in a test-data crate.*
- [x] Document the streaming / IPC / network use cases in
      `fuel-formats/README.md` so consumers know the parser
      surface is *the* public seam (file path is just one
      transport). *Shipped — README covers HTTP body parsing,
      inter-process tensor exchange via safetensors-on-the-wire,
      KV-cache handoff, RemoteHostStorage foundation, hot reload,
      and the pattern new transport adapters should follow.*

**A2. `fuel-loaders` finalization (post-E).** Once `Tensor`
lives in `fuel-tensor` (work item E below), the file-transport
wrappers currently in `fuel-core` migrate into a small
`fuel-loaders` crate that depends on `fuel-tensor` +
`fuel-formats`. `fuel-core` re-exports for back-compat. This
becomes ~one afternoon of mechanical extraction.

- [ ] Move `safetensors.rs` / `pickle.rs` Tensor-construction
      wrappers + `quantized/{gguf_file,gguf_mmap,ggml_file,
      imatrix_file}.rs` (now thin Tensor builders calling
      `fuel-formats`) into `fuel-loaders`.
- [ ] Decide whether `custom_op` extension hook stays with
      `fuel-tensor` (likely) or splits separately. If split,
      move to `fuel-custom-op`.
- [ ] Update `fuel-transformers` and `fuel-examples` to depend on
      `fuel-loaders` directly where weight loading is the only
      `fuel-core` API in use.

**B. Drop eager mode, introduce `.realize()`.**

Internal sub-phases (B1-B6) tracked in memory plan. B1 is shipped;
B2-B6 land *after* work items G + G2 below, both shipped
2026-05-02. G provides graph-owned Storage; G2 makes `Op::Const`
a slot-rooted unit variant. Together they're the substrate that
B's factory migration plugs into.

- [x] **B1.** Add `.realize()` / `.materialize()` / `.is_realized()`
      stubs to `Tensor`. Identity clones today; gain real semantics
      after G + B3.
      *Shipped 2026-05-02 (commit a8e192ff). 3 unit tests verify
      today-identity contract; full fuel-core test suite green.*
- [x] **B2.** Factories (`zeros`, `ones`, `from_slice`, `from_vec`,
      `from_iter`, `arange`, `arange_step`, `eye`, `full`, `rand`,
      `randn`, `meshgrid`) produce graph-rooted Tensors backed by
      `Op::Const` nodes whose Storage lives in the graph's
      storage map. *Shipped per `project_phase_7_5_work_item_b2_complete.md`:
      fuel-core eager `Tensor` factories produce node-handle tensors;
      8 view ops bridged through `realized_storage()`.*
- [ ] **B3.** Migrate every `Tensor::*` op method to build a graph
      node instead of calling `Storage::*` directly. One op family
      per commit (unary, binary, binary-scalar, cmp, reduce,
      reshape/transpose, slice, matmul, conv, qmatmul, misc).
      Dispatch becomes the lazy-stack's `realize_*` entry points,
      with a fast-path for one-node graphs to amortise per-op
      overhead.
- [ ] **B4.** Update `to_vec*`, `to_scalar`, `Display` impls, and
      any other "force value" entry points to call `.realize()`
      implicitly so users don't have to.
- [ ] **B5.** Migration pass through `fuel-nn`, `fuel-transformers`,
      `fuel-examples`: most code remains unchanged because op
      methods retain their signatures; only "inspect a value"
      sites need `.realize()`.
- [x] **B6.** Drop eager dispatch entirely. **The eager `Tensor` is
      DELETED** (2026-08-01) — `fuel-core/src/tensor.rs`, `op.rs`,
      `backprop.rs`, `custom_op.rs`, `variable.rs`, `indexer.rs`,
      `streaming.rs`, `convert.rs`, `npy.rs`, `pickle.rs`, `conv.rs`,
      `display.rs`, `hopfield.rs`, `sampling.rs`, `sort.rs`,
      `tensor_cat.rs`, `scalar.rs`, plus the `Module`/`ModuleT` traits and
      the eager halves of `storage.rs`, `test_utils.rs`, `shape.rs`,
      `safetensors.rs` and `quantized/`. `fuel-core` and every
      default-member compile without it.

      Sequencing that actually mattered — the naive "sever one island"
      plan was wrong in two ways: there are **five** carve-outs, not two,
      and one hard break sat in a *default-member* (`fuel-inference`
      re-exported `fuel::sampling::*`). The methodological trap that
      produced the wrong plan: the root manifest aliases the crate
      (`fuel = { path = "./fuel-core", package = "fuel-core" }`), so
      consumers write `use fuel::…` and **any grep keyed on `fuel_core::`
      returns near-zero by construction** — scanning both aliases moved
      the external hit count from ~5 files to 137.

      B6's residual is now CLOSED (2026-08-14). `fuel-onnx` was fully
      lazy-ported — the eager `eval.rs`/`simple_eval` was deleted
      (`67c5a2b3`, `1c94dfe4` "eager consumers gone") and it compiles
      clean; `fuel-book` (fork-inherited Candle book, whose snippets
      still used the deleted eager API) was deleted wholesale rather than
      ported, since it had no live consumer and its content had diverged
      from Fuel — recoverable from git if the book is recreated later.
      The two dead feature-gated examples were handled too
      (`reinforcement-learning` deleted, `mnist-training` ported). The
      `if tensor.item::<f32>() > 0.5` idiom question is moot for Rust
      callers — there is no eager value left to inspect — but resurfaces
      if Python bindings are revived.

**C. Sever `Op`-as-IR from `BackpropOp`-as-tape-entry; move
backward to `fuel-autograd`.**

- [ ] Confirm `Op` lives in `fuel-core-types` with no autograd
      coupling (already mostly there post-Phase 6).
- [ ] Drop `BackpropOp` and `is_variable` from `Tensor_`. Add a
      `Variable` concept that's just "a graph input the autograd
      pass differentiates with respect to" — data, not a type
      distinction.
- [ ] Create `fuel-autograd` crate. Define the
      `BackwardRule<Op>` registration trait and the
      `grad(graph, output, wrt)` graph-rewrite entry point.
- [ ] Move every existing backward closure into a `BackwardRule`
      impl. Co-locate per-op backward rules with their forward
      `Op` definitions in the owning crate where possible
      (`fuel-conv` owns Conv backward, `fuel-quantized` owns
      QMatMul backward). `fuel-autograd` provides only the
      traversal/transform machinery and the public API.
- [ ] Add a compile-time check that every `Op` variant has a
      registered `BackwardRule` (or is explicitly marked
      non-differentiable) — closes the "open enum" problem
      Option 2 normally has.
- [ ] Validate higher-order gradients work end-to-end on a small
      test case (`grad(grad(f))` for a simple function).

**D. Inplace-as-optimization graph rewrite.**

- [ ] Add `opt::inplace_rewrite` pass running before executor
      dispatch. Walks the unified graph, computes per-tensor
      liveness (forward consumers + backward dependencies),
      swaps non-inplace → inplace where the input has no
      remaining consumers, and swaps inplace → non-inplace
      where the input is needed.
- [ ] For each op that has both inplace and non-inplace forms,
      ensure the optimizer can pick freely. This is the
      shape-stable case; ops where inplace requires a different
      output shape don't qualify and the optimizer leaves them
      alone.
- [ ] Document that `*_inplace` op variants are now perf hints,
      not correctness primitives. Recommend users write the
      non-inplace form; the optimizer adds inplace where safe.
- [ ] Once the optimizer is shown to find every inplace win the
      hand-written `*_inplace` callers were getting, consider
      retiring the user-facing `*_inplace` API entirely and let
      the optimizer be the sole source of inplace decisions.

**E. Crate split: `fuel-tensor` and the umbrella facade.**

- [ ] Extract `Tensor`, eager-API methods (now graph builders),
      indexer, scalar helpers, and `custom_op` (if not split
      separately) into `fuel-tensor`.
- [ ] Reduce `fuel-core` to: re-export facade over
      `fuel-core-types`, `fuel-tensor`, `fuel-autograd`,
      `fuel-loaders`, `fuel-graph-*`, and the registered
      backends. Most public-API surface stays accessible via
      `fuel-core::*` for back-compat.
- [ ] Internal callers (`fuel-nn`, `fuel-transformers`,
      `fuel-examples`) keep depending on `fuel-core`. New
      lightweight consumers can depend on the smaller leaf
      crates directly.

**G. Graph owns Storage; `Tensor` becomes a thin handle.**

*Architectural prerequisite added 2026-05-02 between B1 (shipped)
and B2. Inserted after the design pass on B's design question
("how does `Tensor_` represent a graph-attached state?") concluded
that the long-term answer is "Tensor doesn't own Storage — the
Graph does," and that landing this before B2-B6 is cheaper than
migrating every consumer twice (once to add an `Option<GraphLink>`,
again to drop the Storage field).*

The model after G:

- `Graph` owns a `HashMap<NodeId, StorageSlot>` keyed per device.
  Each slot holds a `Box<dyn DynBackendStorage>` plus its realized
  `Layout`. Multi-device graphs (CPU↔Vulkan↔CUDA Router) keep
  working — each NodeId's slot lives on the device its placement
  side-table entry specifies.
- `Tensor` shrinks to `{ graph: SharedGraph, id: NodeId }`. The
  `Arc<RwLock<Storage>>` field on `Tensor_` goes away. (The
  `op: BackpropOp` field stays for now — its removal is work
  item C.)
- The executor's existing NodeId→Storage cache moves *as-is*
  into the Graph rather than living in executor scratch space.
  Residency machinery (`Op::Release`, `ResidencyEvictionRule`,
  `evict_from_candidates`) keeps working unchanged — it already
  operates by NodeId, so the cache's new home doesn't change its
  interface.
- `Op::Const` is a unit variant (post-G2). Bytes live in the
  graph's storage_map slot, populated when the constructor is
  called (`Tensor::from_f32`, `const_f32_like`, etc.). The
  executor's slot-first dispatch returns the slot's Arc on
  realize — no host-side payload rides on the node itself.
  Const-pool cache is liveness-witnessed via
  `Weak<RwLock<Storage>>` so slot Arc recycling can't produce
  stale cache hits.

Migration tactic — parallel-introduction-then-drop:

1. Add `StorageMap` to `Graph`. Add a "node-handle" mode to
   `Tensor` where the `storage` field is `Option<Arc<RwLock<Storage>>>`
   — `None` means "ask the graph." Existing eagerly-constructed
   Tensors stay as-is at first.
2. Migrate factories first (B2's actual work). The graph-side
   primitive (`fuel_graph::NodeHandle::from_storage`) is in place;
   B2 routes fuel-core's `Tensor::ones` / `::zeros` / `::from_slice`
   / etc. through it instead of the legacy `from_storage` (eager-
   mode) path. ~13 factory functions in `fuel-core/src/tensor.rs`
   plus a few callsites that use them; structural work, not
   trivially simple but not large either.
3. Migrate op methods family-by-family (B3 work, post-G). Each
   migrated family produces node-handle Tensors and removes one
   pin holding old-mode Tensors alive.
4. Once nothing produces old-mode Tensors, drop the `Option`
   wrapper and the legacy field. Tree compiles green throughout.

Sub-tasks (initial substrate 2026-05-02 + 5-commit fix-up sequence
that brought G into alignment with what was originally agreed):

- [x] Move `Storage` struct to `fuel-core-types`. *Shipped fix-up
      1/5 commit ffa9076e. Eager-dispatch methods that need
      `CustomOp1/2/3` (which transitively reference `Tensor`) stay
      in fuel-core via the `StorageApplyOps` trait extension; all
      other inherent methods moved with the struct. `Storage::device()`
      now returns `Arc<dyn DynBackendDevice>`; fuel-core wraps as
      `Device { inner: ... }` at use sites.*
- [x] `fuel_graph::Graph` owns the storage map directly:
      `HashMap<NodeId, Arc<RwLock<Storage>>>`. Sidecar
      (`fuel-core::graph_storage`) deleted. *Initial sidecar
      shipped 2026-05-02 commit 07691b97; fix-up 2/5 commit
      8c32b535 moved the map onto `fuel_graph::Graph` and dropped
      the fuel-core module entirely.*
- [x] Migrate `fuel_graph::SharedGraph` from `Rc<RefCell<>>` to
      `Arc<RwLock<>>` so `fuel_core::Tensor` retains Send+Sync
      after gaining `Option<fuel_graph::NodeHandle>`. *Shipped 2026-05-02
      commit e6c31614. ~100 mechanical borrow→read/write
      replacements across fuel-graph + fuel-graph-cpu/executor/router
      + cuda-backend + reference-backend + fuel-core
      lazy/scheduling. cudnn.rs's thread-local cache unrelated and
      unchanged.*
- [x] `Tensor_::link: Option<fuel_graph::NodeHandle>` — reuses the
      existing graph handle as the link payload (no separate
      `GraphLink` wrapper). *Initial commit 3c042bf8 introduced
      a separate `GraphLink`; fix-up 2/5 commit 8c32b535 dropped
      it in favor of `fuel_graph::NodeHandle` directly.*
- [x] `Tensor::realized_storage()` mode-agnostic read seam plus
      `has_graph_link()` / `graph_link()` accessors. *Initial commit
      3c042bf8; fix-up 4/5 commit f0f0df1d revised the seam to
      enforce the `(storage, link)` exactly-one-of invariant.*
- [x] Migrate every storage read in fuel-core + downstream
      (fuel-nn, fuel-flash-attn-cuda, …) through
      `realized_storage()`. ~85 sites bound the returned Arc into
      a named local + take `.read().unwrap()` /
      `.write().unwrap()`. *Shipped fix-up 3/5 commit 6e1e10db.*
- [x] `Tensor_::storage` becomes `Option<Arc<RwLock<Storage>>>`;
      "exactly one of `storage`, `link` is `Some`" invariant
      enforced at construction. `from_storage` produces
      legacy-mode tensors; new `from_link` constructor produces
      node-handle tensors (reads dtype/device/shape from the
      slot, errors cleanly when the slot is unpopulated). *Shipped
      fix-up 4/5 commit f0f0df1d.*
- [x] Smoke test: construct a node-handle Tensor end-to-end and
      verify `realized_storage()` returns the slot Arc.
      *Shipped 2026-05-02 commit 42a94c74; rewritten in fix-up 4/5
      to use the `from_link` constructor.*
- [x] Multi-device parity: parametric helper + gated CUDA/Metal
      tests verifying the slot mechanism preserves device
      identity. *Shipped 2026-05-02 commit ae87d92c — CUDA
      verified live on RTX 4070. Vulkan parity holds by
      construction (same trait inheritance) — re-enable an
      explicit Vulkan test once a device-construction shortcut
      for tests is added.*
- [x] Document the new model in `GUIDE.md` (architecture seam)
      and `PATTERNS.md` (runnable example). *Initial commit
      530cd371; fix-up 5/5 commit 56e109ca rewrote both to match
      the corrected architecture.*

Follow-on (post-G, ahead of CE):

- [x] **G2. Move `Op::Const` payload into graph-Storage.**
      Shipped 2026-05-02 as a 3-step sequence:
      1. Substrate (commit a4b836c9): `Op::Const(Option<ConstData>)`
         wraps the legacy host payload alongside a new slot-only
         `Op::Const(None)` mode; `Tensor::from_storage` primitive
         for slot-only construction; slot-first dispatch in
         fuel-graph-executor / fuel-graph-cpu / fuel-reference-backend's
         realize loops.
      2. Sweep (commit f0062c4f): public factories take an explicit
         `&Device` (`fuel_graph::NodeHandle` takes `&Arc<dyn DynBackendDevice>`,
         `fuel_core::Tensor` takes `&Device`); `const_*_like`
         methods stay 2-arg and derive device from `self`'s graph.
         ~700 callsites swept across ~50 files.
      3. Cleanup (commit a00e6738): `ConstData` enum dropped;
         `Op::Const` becomes a unit variant; gradient seeder
         `build_filled_const` slot-populates via
         `pick_device_from_graph`; `eval_const` arms in every
         backend become `unreachable!`; const_pool restored
         with `Weak<RwLock<Storage>>` liveness witness so slot
         pointer recycling across realize calls (fresh-graph-per-
         training-step pattern) can't cause stale cache hits.
         fuel-cuda-backend gained `try_adopt_slot_cuda` slot-first
         dispatch in all three realize loops.

Estimated scope: 1-2 focused weeks for G itself; G2 was about a
week (estimated half a week, plus the const_pool liveness fix and
the cuda slot-first wiring that surfaced during the work).

**F. Declared layout contracts and layout-tracking optimizer pass.**

*Placeholder — needs design-pass planning before sub-tasks are
written. Listed here so the idea isn't lost.*

The high-level idea: each op-on-each-backend declares the input
`Layout`s it can accept and the output `Layout` it produces. The
graph optimizer reads those contracts, matches consumer-input
against producer-output, and either inserts layout-conversion
ops where there's a mismatch or selects op variants whose
contract consumes the existing layout. Same kind of reasoning
XLA does for HLO sharding/layout, MLIR's linalg dialect does for
layout assignment, and cuDNN's plan-graph does for tensor format
selection.

Open design questions to resolve before this becomes actionable:

- Layout space is bigger than contiguous-vs-strided. NHWC vs
  NCHW for conv, blocked formats (cuDNN's `nchw_vect_c`, NHWC8),
  interleaved quant block layouts (Q4_0's 32-element packing
  isn't expressible as strides at all). Which axes of layout
  space does the optimizer reason about? A small closed set of
  named layouts plus an `Any` fallback for stride-aware kernels
  is the pragmatic answer, but the choice needs to be made
  explicitly.
- Most of Fuel's ops today implicitly accept any stride-aware
  Layout — their contract is `Any → Any`, which carries no
  signal for the optimizer. The ops where layout-contracts pay
  off are the rigid ones: cuBLAS gemm's lda/ldb/ldc rules, conv
  kernel format preferences, Q4_0 matmul's block-aligned input.
  Maybe 15-20 ops out of ~140. The cost-benefit of declaring
  contracts on the rest is real and needs a deliberate answer.
- Multi-device interaction: layout-on-device-A doesn't mean the
  same thing as layout-on-device-B. Per-device layout reasoning
  vs unified abstract layouts is itself a design choice.
- Interaction with G's storage slots: each slot already records
  a realized Layout. F's contract-reasoning operates on this
  metadata. F is gated on G having shipped.

Estimated scope: deferred — depends entirely on the design
choices above. Likely 2-4 weeks once scoped.

#### Sequencing

Revised after the 2026-05-02 design pass (see work item A
preamble for context). The original sequence put A first as a
cheap independent ship; closer inspection showed A's "loaders
fissioning" framing required a parse/construct seam workaround
because of the Tensor-coupling cycle. Re-framing A as
`fuel-formats` (parser layer) plus A2 (loaders finalization
after E) lets the parser layer ship now without compromise and
defers the Tensor-coupled file-transport extraction to where it
is mechanical.

Order (revised 2026-05-02 after G was added):

1. **A (`fuel-formats`)** ✅ shipped 2026-05-02. Parallel-safe with
   B because it touches the byte-decode bodies of loader files,
   not the Tensor construction call sites B is rewriting. No
   `Tensor` / `Storage` / `Device` coupling.
2. **B1 (`.realize()` stubs)** ✅ shipped 2026-05-02 (commit
   a8e192ff). Identity-clone stubs that stabilise the public API
   so downstream code can opt into the lazy idiom early.
3. **G (Graph owns Storage)** — architectural prerequisite for
   the rest of B. Lands the `(graph, NodeId)`-handle Tensor model
   and moves Storage ownership into the Graph. 1-2 focused weeks.
4. **G2 (`Op::Const` payload moves into graph-Storage)** ✅ shipped
   2026-05-02 across commits a4b836c9 / f0062c4f / a00e6738.
   Public factories (`Tensor::from_f32`, etc.) take an explicit
   `&Device`, slot-populate at construction, and emit `Op::Const`
   as a unit variant. ConstData is gone.
5. **B2-B6 (factories, op methods, force-value entry points,
   downstream migration, drop eager dispatch)** — much simpler
   on top of G. Each B sub-phase is an independently shippable
   landing; B3 ships op-family-by-op-family.
6. **C and E together** — once `Tensor_`'s `op: BackpropOp` and
   `is_variable` come out (C), `Tensor` is small enough that
   extracting it to `fuel-tensor` (E) is the same motion. C
   cannot finish without touching every site E needs to touch,
   and doing them together avoids a transitional state where
   `Tensor_` is half-shrunken.
7. **A2 (`fuel-loaders` finalization)** — afternoon of work once
   E lands. File-transport wrappers move from `fuel-core` to
   `fuel-loaders`; `fuel-core` re-exports for back-compat; no
   parse/construct seam to maintain.
8. **D (inplace-rewrite optimizer)** — depends on C+E producing
   the unified forward+backward graph to do liveness on.
9. **F (declared layout contracts)** — deferred, awaiting design
   pass. Gated on G being in place because F operates on the
   per-slot Layout metadata G introduces.

B and C/E are tightly coupled but should ship as separate
landings rather than one mega-PR — B first (so the eager-vs-
lazy duality is collapsed before autograd refactor), then CE.

Total estimated scope: A is one week (parser extraction is
self-contained); G is 1-2 weeks plus G2 about a week (estimated
half a week, plus the const_pool liveness fix and the cuda slot-
first wiring that surfaced during the work); B2-B6 is
two-to-three weeks of factory + op-method migration on top of
G; CE together is six-to-eight weeks (every op constructor
touched, plus mechanical Tensor extraction); A2 is half a day;
D is one-to-two weeks of optimizer-pass work; F is deferred.
Roughly three months end-to-end excluding F.

#### Success criteria

- `fuel-core` no longer carries byte-level format-parsing code;
  parser surface lives in `fuel-formats` and operates on
  arbitrary `Read` / `&[u8]` sources. File-transport wrappers
  live in `fuel-loaders` (post-E); `fuel-core` re-exports
  remain for back-compat.
- A streaming weight-load smoke test reads a safetensors file
  through `fuel-formats` directly off a network-style
  `impl Read` (e.g., `Cursor<&[u8]>`) without touching the
  filesystem — proves the parser surface is genuinely
  transport-independent.
- A new `fuel-tensor`-only program (no autograd, no loaders)
  builds and runs a forward pass with measurably smaller compile
  times than the current `fuel-core`-equivalent.
- `Tensor` has no `op: BackpropOp` field; inference paths show
  measurable reduction in per-op overhead and per-tensor memory
  vs. the pre-7.5 baseline.
- A training program written against `fuel-autograd` produces
  bit-equivalent gradients to the current in-tree autograd on
  the regression suite (CPU + at least one accelerator).
- Higher-order gradient test (`grad(grad(f))`) passes end-to-end.
- Inplace ops work correctly in differentiated regions without
  user intervention; benchmark suite shows inference paths
  picking up inplace wins on at least the activation functions
  in the Phase 6 anchor suite.
- Eager mode is removed; `.realize()` is the documented
  materialisation point; `GUIDE.md` and `PATTERNS.md` reflect
  the lazy-only idiom.

#### Honest caveats

This is the largest single phase since Phase 6 itself. The
biggest risk is C (autograd refactor): every op constructor in
the codebase is touched and any subtle change to gradient
semantics shows up as a training divergence that's expensive
to debug. Mitigation: bit-equivalence testing against the
pre-7.5 autograd at every step, on the CPU reference backend
where determinism is highest. The second risk is B: dropping
eager mode is a user-visible API change even if signatures
remain the same — anyone relying on "matmul executes now"
semantics has to learn `.realize()`. Mitigation: documentation,
an opt-in `FUEL_EAGER=1` env-flag during transition that forces
`.realize()` after every op, and a deprecation cycle before the
flag is removed.

The third risk is G (Graph owns Storage): every read path that
touches `tensor.storage()` today changes shape. Mitigation is
the parallel-introduction-then-drop tactic — old-mode and
node-handle Tensors coexist throughout the migration window so
the tree compiles green at every step, and a single mode-agnostic
read API (`tensor.realized_storage()`) gives consumers a stable
seam. The residency machinery (`Op::Release`,
`ResidencyEvictionRule`, `evict_from_candidates`) is purely
NodeId-based and rides the change without code edits, which is
a meaningful piece of evidence that the architectural cut is in
the right place.

This phase should not be attempted concurrently with Phase 8
(FlashAttention) or Phase 8.5 (sparsity); both add new
kernels/ops and would have to absorb the autograd-rewrite mid-
flight. Phase 9 (agentic hooks) is gated on a real consumer
and not in conflict.

---

### Phase 7.6 — FusedOpRegistry: open registry for fused ops, closed enum for primitives

> **Status (2026-06-25):** Shipped — steps 1–3 (registry skeleton + `Op::Fused` arm + SoftmaxLastDim), step 6, steps 9a–9c phases A–E.3.0 (binding-table planning-time refactor, KvCache/InferenceContext, Vulkan runtime Device, multi-target realize, pipelined_bridge). Deferred (backlog) — steps 4, 5, 7, 8, 10 (fused-op migration sweep, Op-variant drops, PrecisionGuarantee/cost population, Comparison family), gated on the dispatch-core cleanup landing; step 9c E.3 remainder (`forward_with_cache_on`, `generate_*`, spec decoding) + E.4. The detail below is retained.

*Architectural refactor that adds an open registry of fused ops accessible
through one arm of the closed `Op` enum (`Op::Fused(id, params)`). Enables
cross-backend cost-based placement and is the substrate the cost-aware
scheduler will consume. Touches every backend's kernel registration and
every consumer that pattern-matches on `Op`. ~2-3 weeks of focused work
against the architecture v1.0 design.*

**Architecture-set anchor**: this phase implements the commitments in
[`docs/architecture/03-ir.md`](docs/architecture/03-ir.md) (Op-shape A, fused-op registry, pre-resolved KernelRef),
[`docs/architecture/04-optimization.md`](docs/architecture/04-optimization.md) (per-decision-point alternatives, OptimizationMap),
and [`docs/architecture/05-backend-contract.md`](docs/architecture/05-backend-contract.md) (per-kernel `PrecisionGuarantee`).

**Phase design doc**: [`docs/fused-op-registry.md`](docs/fused-op-registry.md)
(refreshed against architecture v1.0). The design doc carries implementation
detail; this ROADMAP entry carries the work plan.

#### Why Phase 7.6 exists

PR 3 (rule registry) and PR 3.5 (Op::ReduceMaxTo, Unsqueeze, ReduceMaxToBackward)
surfaced a structural question: every fused op fuel adds today requires a new
`Op` variant + executor arms in every backend + autograd entry + op_short_name +
op_key + binding-table registration + hand-written lowering/fusion rules. Each
fused op multiplies the plumbing cost; the Op enum becomes the bottleneck.

The architectural answer (per [03-ir](docs/architecture/03-ir.md)): the `Op` enum has primitive variants
plus exactly one `Op::Fused(id, params)` arm. The `id` indexes a registry of
fused ops; the registry is open at build-time. Adding a new
fused op is a registry entry + a kernel function — no `Op` enum edit, no
autograd edit, no per-backend executor arm.

**Two-tier runtime extensibility (re-scoped 2026-06-20, [G4 in
10-decisions-log](docs/architecture/10-decisions-log.md)).** The earlier
"frozen at startup, no runtime extensibility" framing conflated two
distinct surfaces and one security boundary; it is re-scoped, not
deleted:

- **Build-time-closed (stays frozen):** the primitive `Op` enum
  (per [G3](docs/architecture/10-decisions-log.md) — no generic
  opaque / `Custom` node; an external op must decompose into the
  existing basis or prompt a build-time `Op`-enum extension) **and**
  untrusted user-installable rules/ops (the [09-non-goals](docs/architecture/09-non-goals.md)
  rejection of untrusted code in the optimizer holds).
- **Tier 1 — already runtime-extensible today:** the **kernel
  binding table** (implementations). `extend_global_bindings`
  (`fuel-dispatch/src/dispatch.rs:5098`) write-locks the table,
  appends (append-only, multi-sibling), re-`finalize()`s, and calls
  `bump_topology_generation()` to invalidate cached routes.
  JIT-ing a kernel for an **existing** op identity lands here — this
  was never the frozen part.
- **Tier 2 — the new goal:** trusted, **Fuel-orchestrated,
  cost-gated** runtime registration of a **new fused-op identity**
  (the metadata registry becomes append-only with **stable,
  never-reused** `FusedOpId`s). Its **mechanism is the declarative
  form** — a runtime fusion can *only* be declarative (pattern +
  recipe + shape/dtype/cost carried as **data**), because Rust `fn`
  pointers and enum variants can't be added at runtime — so the
  stubbed declarative pattern engine (`PatternKind::Declarative =>
  false` at `fuel-graph/src/opt.rs:434`) is the **prerequisite** for
  Tier 2. This is the trusted/untrusted reconciliation: *Fuel*
  chooses the region to fuse (strategy stays in the optimizer), a
  *trusted* backend synthesizes the kernel, the result arrives as a
  declarative recipe (`decompose` + `pattern` over existing
  primitives, per the recipe principle above), and the route picker
  cost-gates adoption — no untrusted code, no new primitive.

The cross-backend payoff: the cost-aware scheduler (downstream phase work)
needs every backend's fusion catalog visible *before* placement decisions to
compare "matmul+bias+relu costs X on CUDA fused, Y on Vulkan unfused." A
registry is the natural shape for that visibility; backend-internal fusion
(XLA's model) couldn't satisfy this.

#### Architectural decisions (anchored to v1.0)

These decisions live in the architecture set; this phase implements them.

- **Op-shape A**: closed `Op` enum with primitive variants + one `Op::Fused(id, params)` arm. No separate `NodeKind` discriminator. Per [03-ir §How nodes carry their op identity](docs/architecture/03-ir.md#how-nodes-carry-their-op-identity).
- **Pre-resolved `KernelRef` per node** at planning time. The binding table is a planning-time catalog; the executor calls function pointers directly. Resolves audit Q-A. Per [03-ir §The optimized form](docs/architecture/03-ir.md#the-optimized-form-the-multi-path-graph-the-plan-is-the-graph).
- **Lazy KernelRef resolution** at decision-point pick time + mmap'd cache. Per [11-persistence §Re-resolution on use](docs/architecture/11-persistence.md#re-resolution-on-use-lazy-not-at-load).
- **Fused-op registry crate location**: metadata in fuel-graph; `BackendImpl` payload (which carries `KernelRef`) in fuel-storage. Avoids a circular dependency.
- **Per-kernel `PrecisionGuarantee` structure** on the registration surface, replacing the OracleGrade flag concept. Per [05-backend-contract §Per-kernel precision guarantees](docs/architecture/05-backend-contract.md#per-kernel-precision-guarantees).
- **PR 3's hand-written rules become auto-generated** from `FusedOpEntry.decompose` + `FusedOpEntry.pattern`. Hand-written remains an escape hatch. **Both directions are mandatory and `decompose` is total** (the recipe principle, [G1/G2 in 10-decisions-log](docs/architecture/10-decisions-log.md)): an entry that ships only one direction is an opaque island, and a `decompose` that panics is a bug (a primitive returns self; a non-basis op that fails to decompose is a surfaced opaque-op gap, not a crash) — this is load-bearing for the optimizer, which is *lower-to-base-map + find-best-cover*.

#### Sub-tasks (revised against architecture v1.0)

- [x] **Step 1: registry skeleton.** *Shipped per [`project_phase_7_6_step_3_shipped.md`](MEMORY.md). `FusedOpId`, `FusedOpEntry`, `FusedOpParams`, `FusedOpRegistry` in fuel-graph; `BackendImpl`, `PrecisionGuarantee` in fuel-storage. See [`docs/fused-op-registry.md`](docs/fused-op-registry.md) v3 for the crate-split detail.*
- [x] **Step 2: extend `Op` enum with `Op::Fused(FusedOpId, FusedOpParams)` arm.** *Shipped (same memory). Coexists with legacy fused-op variants during migration; `op_short_name`/`op_key` handle the new arm.*
- [x] **Step 3: migrate first fused op (SoftmaxLastDim) end-to-end.** *Shipped (same memory). Auto-generated `LoweringRule` + `FusionRule` from the registry entry; PR 3's hand-written rules retired; live CUDA equivalence test green.*
- [~] **Step 4: migrate remaining 12 fused ops.** *Partial: FusedLinear shipped via [`project_phase_7_6_fused_linear_and_step_6_shipped.md`](MEMORY.md). RmsNormLastDim, LayerNormLastDim, Rope, Conv2D, ConvTranspose2D, FlashAttn, PagedAttn, QMatMul, plus the 4 backward-helper fused ops remain. Each is its own commit; ~half-day per op.*
- [ ] **Step 5: drop the per-fused-op `Op` variants.** Once nothing emits `Op::SoftmaxLastDim` etc., remove them from the enum. Mechanical: update `op_short_name`, `op_key`, autograd's match arms. Gated on Step 4.
- [x] **Step 6: backend registrations adopt `BackendImpl` shape.** *Shipped per [`project_phase_7_6_fused_linear_and_step_6_shipped.md`](MEMORY.md). `register_fused!` macro + `default_kernel_registry` populate `FusedOpEntry` → `BackendImpl` mapping; 4 CPU `FusedLinear` impls registered.*
- [ ] **Step 7: populate `PrecisionGuarantee` per registered kernel.** Bit_stable kernels (the always-built backend's coverage commitment) get the `bit_stable_on_same_hardware: true` flag; others declare what they can characterize. Per [05-backend-contract §Per-kernel precision guarantees](docs/architecture/05-backend-contract.md#per-kernel-precision-guarantees).
- [ ] **Step 8: populate cost estimates.** Each `BackendImpl`'s `cost` function gets a real implementation per backend. Initial: FLOP-counting + bandwidth model. Static-only for v1; community-aggregated empirical refinement (per [04-optimization §Cost model](docs/architecture/04-optimization.md#cost-model-static-annotations-refined-by-empirical-judge-data-accounting-for-parallelism)) follows when telemetry pipeline lands.
- [~] **Step 9: binding-table planning-time refactor.** *Steps 9a + 9b Track A shipped per [`project_phase_7_6_step_4_in_progress.md`](MEMORY.md). 9a: `KernelBindingTable` multi-impl alternatives per `(OpKind, dtypes, BackendId)` (commit `b9828f13`). 9b Track A: `NodeKernelBinding` + `compile_plan` + `resolve_kernel` + `TolerancePolicy` (commits `d60febc7`, `1251bb73`, `5b9f7ca3`, `700bb948`). Step 9c (typed-storage executor migration → see [Phase 7.6 step 9c](#phase-76-step-9c--typed-storage-retirement) below) is the next gate.*
- [ ] **Step 10: comparison family** (Equal/NotEqual/Less/LessEqual/Greater/GreaterEqual) added to `Op` as primitive variants. Bit-exact equality on floats; non-differentiable backward (panic stub, ArgMaxDim precedent). Lands here because primitive-set completion belongs with this architectural cleanup. **Note**: also tracked in the [`fill-op-primitive-set.md`](docs/session-prompts/fill-op-primitive-set.md) session prompt which audits the broader missing-primitive surface.

#### Success criteria

- `Op` enum is primitive variants + one `Op::Fused(id, params)` arm. ~85 primitive variants. No per-fused-op variants.
- `FusedOpRegistry` populated with 13-14 entries (the migrated fused ops). Adding a new fused op is one entry + one kernel function.
- PR 3's hand-written SoftmaxLastDim rules deleted; auto-generated rules from registry entries produce equivalent behavior. Round-trip identity test still passes.
- Live CUDA equivalence test (`cuda_executor_matches_cpu_on_softmax_via_lowering`) still passes through the registry-dispatched path.
- `cost_estimate(SOFTMAX_LAST_DIM, [B, N, M], CUDA)` query returns a plausible estimate via the registry surface.
- Every registered kernel carries a `PrecisionGuarantee`; the always-built backend's coverage commitment (one `bit_stable` kernel per primitive op) is testable as a CI lint.
- All existing tests green throughout the migration. CSE / op_key handles `Op::Fused(id, params)` correctly.
- ROADMAP and architecture decisions-log updated post-migration.

#### Honest caveats

This refactor touches the deepest layer of fuel. Backends, autograd, executor, op_short_name, op_key, dispatch wrappers, CSE — all match on `Op`. The migration uses parallel-introduction-then-drop: existing variants and the new `Op::Fused` arm coexist through the migration window; per-fused-op variants drop in step 5. Each fused-op migration in step 4 is independently shippable.

The architecture's pre-resolved KernelRef commitment (step 9) is a meaningful refactor on its own — it changes where the binding table is consulted (planning time, not execution time). Lands in this phase because Phase 7.6's executor work is the natural place to also restructure the executor's per-node dispatch path.

Cost estimates registered with `BackendImpl`s are advisory; the cost-aware scheduler that consumes them is downstream. Initial cost models can be coarse; the community-aggregated empirical refinement framework (per [11-persistence §Cache generation and distribution](docs/architecture/11-persistence.md#cache-generation-and-distribution)) tightens them over time.

This phase should not run concurrently with Phase 8 (FlashAttention) or Phase 8.5 (sparsity); both add new fused ops mid-flight that would have to absorb the registry refactor. Phase 7.5 work items B/C/E (Tensor/autograd/fission refactor) are orthogonal — they can run before, after, or in parallel.

#### Phase 7.6 step 9c — typed-storage retirement

*Audit + multi-session plan to swap the legacy `GraphExecutor<B>` (typed-storage shape) for `PipelinedExecutor` (dispatch-erased shape) across all callers.*

**Full audit**: [`project_phase_7_6_step_9c_parity_audit.md`](.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_phase_7_6_step_9c_parity_audit.md) (memory). 242 call sites across 34 files; ~12 PipelinedExecutor feature gaps; estimated 6-8 sessions / 30-50 commits.

**Status (2026-05-22)**:

- ✅ Phase A — multi-target realize (`realize_many` shipped 2026-05-19 in commit `c5ed169a`).
- ✅ Phase B — side-effect roots + destructive-input cleanup (shipped 2026-05-19 in commits `db89a283` + `f9ad93d0`).
- ✅ Phase C — CPU fallback shape decided: fail-fast (binding-table `lookup` returns `None` → typed error). Documented 2026-05-19.
- ✅ Phase D — optimization + rule-registry plumb-through: caller composes. Documented 2026-05-19.
- ✅ Phase E.1 + E.2 — fuel-core `pipelined_bridge` module shipped 2026-05-19 in commit `32d712f7`. `Tensor::realize_f32` migrated for CPU + CUDA; CUDA executor gained output allocation, auto-contiguize, layout-vs-storage-bytes mismatch handling.
- ✅ Phase E.3.0 — `InferenceContext` + `KvCache` primitives shipped 2026-05-20 in commit `a405e7c0`.
- ✅ Vulkan runtime `Device` wiring shipped 2026-05-22 (this session): `VulkanBackendDevice` + bridge module + parity test against CPU through `forward_with_kv_context`.
- ⏳ Phase E.3 remainder: pre-allocated buffers + `Op::WriteSlice` (E.3.2), `forward_with_cache_on` migration (E.3.3), `generate_*` + spec decoding (E.3.4).
- ⏳ Phase E.4: train.rs + factories.rs migration.
- ✅ Phase F + H — **executor-unification Session 7 (2026-06-15)**: the `GraphBackend` trait + all surviving impls (Cpu/Cuda/Vulkan/Mkl/Aocl), the `fuel-graph-executor` crate (`GraphExecutor<B>`), and the whole `fuel-graph-cpu` crate (`realize_any`, the typed third evaluator) are deleted. `PipelinedExecutor` (`fuel-dispatch`) is now the sole executor on every realize path. MKL/AOCL retain only their binding-table registration surface; the CUDA FA2 launcher (`fuel-cuda-backend::flash_attn::launch`) is preserved (`#[allow(dead_code)]`) for the queued FA2 eager-wrapper session; legacy-executor diff/oracle tests (`cpu_vulkan_diff.rs`, `conv2d_oracle.rs`, `flash_attn_cuda.rs`, `flash_attn_vulkan.rs`) retired with the trait. `fuel-reference-backend::exec::realize_f32` stays as the correctness oracle.
- ✅ Phase G — `GraphBackend` retain-vs-retire decision: **retired** (above). `fuel-graph-router`'s own crate disposition tracked with Session 6.
- ⏳ Remaining: executor-unification Session 8 (eager `Tensor` + `BackpropOp` tail).

#### Bridge-retirement trajectory post-9c

The Vulkan Device-wiring shipped this session uses a bridge pattern: `VulkanBackendDevice` wraps `Arc<VulkanBackend>` and implements `DynBackendDevice`, but the storage-returning `*_dyn` methods stub to errors because Vulkan storage lives on the byte-shape `VulkanStorageBytes` substrate, not on `DynBackendStorage`. The bridge is mirror-shaped across CUDA + CPU + Vulkan and follows the established pattern, but is not the architecture v1.0 destination.

The destination per [01-identity](docs/architecture/01-identity.md) + [05-backend-contract](docs/architecture/05-backend-contract.md) is: `Device` is a thin tag, storage allocation + transfer happen via graph-level `Op::Alloc` + `Op::Copy` primitives that the optimizer plans and the executor dispatches through the binding-table, and `DynBackendStorage` retires entirely.

Path from bridge to destination (each phase ~1 session, in dependency order):

1. ✅ **D2H through `Op::Copy`** — *shipped 2026-05-22.* `OpKind::Copy` registered in the binding table; CPU/CUDA/Vulkan each provide a `copy_to_cpu` wrapper at the canonical `[dt, dt]` key. `realize_one_as<T>` / `realize_many_as<T>` splice `Op::Copy { target: Cpu }` at every realize root (CPU + GPU) so the spliced node's `auto_contiguize` honors view-op layouts uniformly — the pre-9c "ignore strides, return full source bytes" CPU bug is fixed alongside. Executor uses a dedicated `WorkItemKind::Copy { target_location }` arm so output allocation goes on the target while the kernel lookup keys on the source backend. **Deleted**: the per-variant match in `BackendStorage::read_to_cpu_bytes` (including the Vulkan branch from `7a95001a`) — first deletion of bridge code from `7a95001a`. See [`project_phase_7_6_step_9c_parity_audit.md`](.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_phase_7_6_step_9c_parity_audit.md) "Bridge-retirement Phase 2 shipped" follow-up.
2. **H2D + zero-alloc through `Op::Alloc` + `Op::Copy`** — split into Phase 3a (zero-alloc) + Phase 3b (H2D Const upload).
   - **Phase 3a — `Op::Alloc` (uninit) + `Op::ZeroFill` (explicit fill)** ✅ *shipped 2026-05-22 (initial) + 2026-05-23 (follow-up refactor).* `Op::Alloc { target: DeviceLocation }` is a new graph primitive (0 inputs, source op like `Op::Const`) producing **uninit** device memory; `Op::ZeroFill` is a destructive in-place fill primitive (paired in current callers). The executor's `WorkItemKind::Alloc` arm dispatches per-backend (CUDA `alloc_uninit` via baracuda alpha.30's raw `cuMemAlloc`; Vulkan `alloc_bytes_handle` truly uninit). The executor's `WorkItemKind::ZeroFill` arm calls baracuda `DeviceBuffer::zero_async` (cuMemsetD8Async, in-place) on CUDA and `VulkanBackend::fill_bytes_zero` (vkCmdFillBuffer, device-side, ~2× the bandwidth of the old host-staged zeros) on Vulkan. `KvCache::with_capacity` emits 2N pairs of `Op::Alloc → Op::ZeroFill`. **Deleted**: `alloc_zeroed_on` (50-line per-`DeviceLocation` match in `fuel-core/src/inference_context.rs`). **Residual**: `device_seed_storage` (~15-line 0-byte-seed allocator per backend). **baracuda bumped to alpha.30** (workspace-wide) to pick up `DeviceBuffer::zero_async`.
   - **Phase 3b — H2D Const upload through `Op::Copy { target: device }`** ✅ *shipped 2026-05-23.* Extended `copy_from_cpu_wrapper` (renamed from `copy_to_cpu_cpu_wrapper`) to switch on output variant — CPU→CPU memcpy, CPU→CUDA via `CudaStorageBytes::write_from_host`, CPU→Vulkan via `VulkanBackend::write_bytes` (new helper: staging buffer + `vkCmdCopyBuffer`). Executor's `WorkItemKind::Copy` arm extended to allocate non-CPU output (Cuda via `alloc_uninit`, Vulkan via `alloc_bytes_handle` uninit). `build_const_cache` (in `pipelined_bridge`) for non-CPU realizes now builds a transient graph with `Op::Const → Op::Copy { target: device }` pairs (one per user Const) plus a device-handle anchor, realizes via `PipelinedExecutor::realize_many`, and writes results back to the user StorageCache at the original Const NodeIds. The transient graph isn't observable from the user's graph. **Deleted**: `upload_host_buffer` (~60-line per-`DeviceLocation` match in `fuel-core/src/pipelined_bridge.rs`) — third deletion of bridge code from `7a95001a`.
3. **`*_dyn` storage methods removed from `DynBackendDevice` trait** once nothing calls them from byte-storage callers. Trait shrinks to advertisement-only (`location_dyn`, `same_device_dyn`, `synchronize_dyn`, `set_seed_dyn`, `get_current_seed_dyn`, `as_any`, `supports_bf16`, `as_quantized_kernels`). Gated on the typed-storage retirement (audit Phases F + H) being complete. **Deletes**: ~6 stub error-bodies per backend's `DynBackendDevice` impl, including the stubs in `VulkanBackendDevice`.
4. **Trait renamed** (`DynBackendDevice` → `BackendAdvertise` or merge into [`BackendCapabilityProvider`](docs/architecture/05-backend-contract.md#static-capability-advertisement-registered-at-startup)). Doc-only.
5. **`Device` becomes a tag, backend handles move to a registry**. `Device { backend_id, location }` is a pure value type. Backend handles live in a process-wide registry consulted by `Device::synchronize` / `Device::set_seed`. **Deletes**: `From<VulkanBackend> for Device` + `as_device(&Device) -> Arc<VulkanBackend>` helper in `fuel-core/src/vulkan_backend.rs`; matching CUDA/CPU/Metal equivalents; the `VulkanBackendDevice` newtype itself. *The bridge built this session is gone at this point.*
6. **`DynBackendStorage` trait retired entirely** once all callers migrate to byte-storage. Significant cleanup in `fuel-cpu-backend/src/dyn_impl.rs` (1365 LOC), `fuel-cuda-backend/src/dyn_impl.rs` (587 LOC), `fuel-metal-backend/src/dyn_impl.rs` (503 LOC).
7. **Router migration** (audit Phase G). `fuel-graph-router` consumes `BackendCapabilities` from the registry; no `Arc<dyn DynBackendDevice>` dependency.

**Architecture-alignment check**: every step makes [01-identity](docs/architecture/01-identity.md#how-this-identity-is-enforced) more true (decisions move to the DAG-level optimizer; cost data flows through binding-table). No step requires revisiting an architecture v1.0 commitment — this is implementation catch-up to the architecture, which is the expected shape since the architecture was drafted ahead.

---

### Phase 8 — FlashAttention tiered implementation

*Affects only the Backends/Kernels layer. Was gated on two external
prerequisites: the new Vulkane release (adds external-memory /
handle-export primitives among other things) and Baracuda (Fuel-owned
CUDA FFI crate replacing cudarc, exposing functionality cudarc omits).*
**Both prerequisites are MET as of 2026-07-31 — this gate is no longer a
blocker.** Vulkane 0.8.0 shipped `DeviceMemory::get_win32_handle`/`get_fd`
plus `Semaphore::get_win32_handle`/`get_fd`, and the workspace pins **0.8.2**;
Baracuda is at **alpha.77** and integrated. Phase 8 is now gated only on
sequencing against the active frontier, not on external work. (Stale-gate
correction reported by the Vulkane session, re-verified against the tree.)

#### Why Phase 8 exists

FlashAttention reduces attention from O(N²) HBM traffic to O(N·d)
via tile-based online softmax. The math is backend-agnostic; the
kernel implementation is decidedly not. v3 and v4 in particular lean
hard on vendor-specific hardware (Hopper TMA + WGMMA, Blackwell
cooperative TMA + 5th-gen tensor cores). A direct port of the
upstream Dao-AILab kernels would be CUDA-only and leave Vulkan users
stuck on naive attention. The right shape is a tiered implementation
that shares the algorithm across backends and specializes only where
the perf justifies it.

A further question worth investigating within this phase: v4's
published speedups are partly algorithmic (warp specialization,
deeper pipeline depth, block-scaled low-precision) and partly raw
matrix-unit throughput. The algorithmic concepts extract cleanly and
can be re-expressed for non-Blackwell architectures; the throughput
component is hardware-gated. Tier 4 below is the place to ask "which
v4 ideas buy us something on Ampere / RDNA3 / Apple M / Intel Arc
and which don't."

#### Tier 0 — Audit existing FlashAttention crates

Fuel's workspace already contains `fuel-flash-attn` and
`fuel-flash-attn-v3` crates (see the Backends/Kernels layer box
above). Before writing anything new, determine what they contain,
which backends they target, and whether they can be refactored into
the tiered structure below.

- [x] Survey `fuel-flash-attn` — list the op surface, target
      backend(s), and parity-test coverage. *Shipped: see
      `docs/phase8_tier0_audit.md`.*
- [x] Survey `fuel-flash-attn-v3` — same. *Shipped in same audit.*
- [x] Decide whether Tier 2/3/4 below refactor these crates in place
      or supersede them. Document the decision. *Decision: rename
      to `fuel-flash-attn-cuda` / `fuel-flash-attn-v3-cuda`, extract
      `-sys` siblings to break the dep cycle, refactor in place.*

#### Tier 1 — CPU reference implementation

- [x] Pure-Rust FlashAttention forward in `fuel-flash-attn` (or
      wherever the audit in Tier 0 lands it). ~100 LOC. Slow by
      design; its job is to be the correctness oracle for every
      other tier. *Shipped as `fuel_reference_backend::attention::
      attention_flash` (~270 LOC; bigger than 100 because it also
      handles GQA, causal mask, sliding window, ALiBi, softcap —
      same surface the kernels target).*
- [x] Backward pass via recomputation — same approach as the
      upstream reference, matches the tier-2/3 kernels' expectations.
      *Shipped as `attention_flash_backward`.*
- [x] Parity tests against a naive-attention reference on small
      shapes (seq ≤ 256, head_dim ≤ 128, batch × heads ≤ 8). Tight
      tolerance (1e-5 in f32) — this tier has no excuse for drift.
      *Shipped: 7 parity tests + 1 finite-difference gradcheck in
      `fuel-reference-backend/tests/attention.rs`.*

#### Tier 2 — Portable GPU implementation in Slang

- [x] Single Slang source for FlashAttention v2 (tile-based,
      workgroup-parallel, online softmax, no warp specialization).
      Compile to SPIR-V for Vulkan; Slang's experimental CUDA PTX
      backend is a free bonus if it works, not a requirement.
      *Shipped as `fuel-kernels-source/kernels/flash_attention.slang`
      → `fuel-vulkan-kernels/spv/flash_attention.spv`.*
- [~] Targets VK_KHR_cooperative_matrix when the device advertises
      it; falls back to plain workgroup-shared-memory tiling on
      devices that don't. *Plain tiling shipped; coop_matrix path
      is a follow-up.*
- [x] Parity tests against Tier 1 across a matrix of
      (batch, heads, seq, head_dim, dtype) shapes. Start narrow
      (f32, contiguous, seq ≤ 1024) and widen once green.
      *Shipped: 4 parity tests in `fuel-core/tests/flash_attn_vulkan.rs`,
      green on RTX 4070 within 5e-4 of reference.*
- [ ] Performance notes: record ms/token on a handful of anchor
      shapes on the dev rig's Vulkan iGPU and on an RTX 4070. This
      tier's ceiling is roughly FA v2 perf on Hopper+ — v3/v4
      pipelining depth needs primitives Slang can't abstract.

#### Tier 3 — Hand-tuned backend kernels (opt-in per arch)

Only write these when Tier 2 benchmarks show meaningful perf left on
the table for a specific architecture.

- [x] **CUDA / Ampere (sm80)**: Dao-AILab FA-v2 kernels via
      `fuel-flash-attn-cuda-sys`. Wired through
      `CudaBackend::flash_attn` behind the `flash-attn` Cargo
      feature; validated on RTX 4070 within F16 precision (max
      abs 4.2e-5) of `attention_naive`.
- [x] **CUDA / Hopper (sm90)**: Dao-AILab FA-v3 kernels via
      `fuel-flash-attn-v3-cuda-sys`. Symbol renamed to `run_mha_v3`
      so both -sys crates link together cleanly. Behind the
      `flash-attn-v3` Cargo feature; dispatch chain prefers v3 and
      falls back to v2 on Err. Rust wiring complete; live-Hopper
      validation deferred to first user with sm90a hardware.
- [ ] **CUDA / Hopper+**: FA v2 or v3-equivalent using CUTLASS or
      hand-written PTX (would supersede the above port-only Tier 3
      entries with Fuel-native kernels). Requires Baracuda exposing
      `CUtensorMap`/`CUwgmma`/`cuTensorMemAcc` primitives.
- [ ] **AMD / RDNA3+**: WMMA + LDS prefetch, wavefront-specialized
      pipeline. Blocked on whatever Rust FFI we settle on for ROCm.
- [ ] **Apple Silicon**: simdgroup_matrix + AMX via Metal. Likely
      lives in `fuel-metal-kernels`.
- [ ] **Intel Arc / Xe-HPG+**: XMX via SYCL or a direct Level Zero
      binding. Lowest priority; revisit if an anchor model
      materially benefits.

Each arch lands independently. The Router + Tier 2 fallback means
users on untuned hardware still get FlashAttention, just not the
peak form.

#### Tier 4 — Extract v4 concepts for non-Blackwell architectures

This tier is research-flavoured and should be sized AFTER Tiers 1-3
are stable. Per-arch experiments to validate which v4 ideas
transfer:

- [ ] Deeper pipeline depth (3-4 stages vs v2's 2) on Ampere using
      `cp.async` + `__syncwarp()` — measure vs Tier 3 CUDA baseline.
- [ ] Warp specialization (producer / consumer split) on RDNA3 LDS
      prefetch — measure vs Tier 3 AMD baseline.
- [ ] Block-scaled low-precision path (MXFP4/MXFP6) — format is
      generic, so this lands as a Tier 2 Slang extension once the
      dtype plumbing exists in Fuel.

Honest caveat: on hardware with weaker matrix units the algorithmic
gains will not close the wall-clock gap with Blackwell; v4's
headline numbers are partly algorithm and partly hardware.

#### Success criteria for Phase 8

- Every backend Fuel targets runs FlashAttention (at minimum
  Tier 2-quality) on every model in the Phase 6 anchor suite.
- Parity with the CPU reference is verified per backend and included
  in the regression gate.
- The tiered structure is documented well enough that a contributor
  with a new backend (SYCL, WebGPU, whatever) can plug in at Tier 2
  without having to touch Tiers 1, 3, or 4.

---

### Phase 8.5 — Dynamic activation sparsity (research-flavoured)

*Affects only the Backends/Kernels and IR layers. Not urgent.
Research effort with model-specific calibration; queue after
Phase 6/7/8 are stable. Primarily benefits CPU inference; GPU
gains are model-dependent.*

#### Why Phase 8.5 exists

Older transformer FFN layers (ReLU-MLP, original Llama / OPT /
BLOOM) produce highly sparse intermediate activations — typically
70-90% of values are zero or below a meaningful threshold. Naive
dense compute on the down-projection wastes that work. Modern
SwiGLU/GeGLU models still have ~30-50% effective sparsity. The
**gather-compute-scatter** technique — extract active rows into a
dense subset, run the dense kernel on the subset, scatter back to
a zero-filled output — captures this win when the sparsity ratio
is high enough to amortize the predicate overhead.

The published name is *dynamic activation sparsity*. Production
references: DejaVu (Tri Dao et al., 2023), PowerInfer, TurboSparse.
DejaVu's headline result: ~2× on Llama 7B at 80% sparsity, with
negligible quality loss.

#### Where it wins (and doesn't)

- **Wins**: CPU inference (where dense GEMM is bandwidth-bound and
  sparsity directly saves work), older models with ReLU activations,
  FFN down-projection (the biggest dense matmul in the layer).
- **Marginal**: modern SwiGLU/GeGLU models — still positive, but
  smaller gain. Need higher sparsity ratios on GPU.
- **Doesn't apply**: attention output (no sparsity-producing
  activation), embedding lookup (already index-gather),
  normalization layers (no sparsity).

GPU dense GEMM is brutally hard to beat — cuBLAS sgemm hits ~98%
of peak on A100. Sparse alternatives need >70-80% real sparsity
*plus* cheap predicate overhead before they win on GPU. The
target hardware reality is: this technique should dominate on CPU
backends (AOCL, MKL, OpenBLAS) and earn its keep on GPU only for
very large FFN dims.

#### Building blocks (status today)

| Need | Status |
| --- | --- |
| Element gather (read indices → dense) | ✅ `Op::IndexSelect`, `Op::Gather` |
| Element scatter back | ✅ `Op::IndexAdd`, `Op::ScatterAdd` |
| Threshold→indices op (data-dependent count) | ❌ no `NonZero` / `Where` / `TopK` |
| Sparse-shaped matmul (variable batch dim) | ⚠️ `Op::MatMul` accepts variable `M`, but the IR's static-shape contract makes data-dependent shapes awkward |
| Gather-compute-scatter graph-rewrite pass | ❌ |

The two missing pieces are the work. The gather/scatter primitives
are already there from the lazy-graph IR.

#### Phase 8.5 work items

- [ ] **Add `Op::NonZeroIndices { threshold: f32 }`** to the IR.
      Returns `[active_count]` u32 indices. Data-dependent shape
      means the IR needs either a "ragged tensor" representation or
      a padded representation with a separate count. Padded is
      simpler; the down-projection sees padded zeros and the cost
      is bounded.
- [ ] **`opt::sparsify_ffn_down_projection`** rewrite pass.
      Detects `Activation(x) → MatMul(W_down)` (FFN down-projection)
      and rewrites to
      `IndexSelect(x, indices) → MatMul(IndexSelect(W_down, indices))
      → ScatterAdd(zeros, indices)`.
      Conservative single-consumer rule (don't fuse if the activation
      is consumed elsewhere) similar to `fuse_linear`'s.
- [ ] **Calibration harness**: per-layer sparsity profile for the
      Phase 6 anchor suite. Pick the threshold per layer per model.
      Offline, run once, stored as model metadata.
- [ ] **Quality gate**: token-equivalence vs the dense reference on
      each anchor model within a tolerance. Too-aggressive
      thresholds degrade output; this gate catches them.
- [ ] **Per-backend native sparse kernels**. Once the IR pattern
      stabilizes, hand-write CSR/dense gemv variants where the
      generic gather + dense matmul is leaving perf on the table.
      AOCL has sparse BLAS; oneMKL has IE-Sparse; cuSPARSE on CUDA;
      hand-written Slang on Vulkan.

#### Success criteria for Phase 8.5

- At least one anchor model (the ReLU-MLP archetype, e.g. original
  Llama 7B) shows ≥30% wall-clock speedup on CPU decode with
  sparsity enabled, without quality regression.
- The pass is opt-in via a flag/feature, never on by default —
  the threshold calibration is per-model.
- Modern SwiGLU models show neutral-to-positive perf (no regression
  even when the sparsity isn't there).
- Documentation explains *which* models benefit and how to find
  the threshold.

#### Honest caveats

This is a research effort, not a routine engineering task.
~2-3 weeks of focused work, ~60% of which is calibration and
benchmarking rather than IR plumbing. Not worth interrupting
current Phase 6/7/8 work for; do not pull forward.

---

### Phase 9 — Extension points for downstream agentic libraries

*Not urgent. Future-facing. Gated on a real downstream consumer
existing — i.e. when a separate agentic / cognitive-architecture
library on top of Fuel is far enough along to need these hooks. Do
not pre-build before that consumer exists.*

#### Why Phase 9 exists

A downstream "AGI library" — built on top of Fuel, not as part of
it — needs to make scheduling and execution decisions that are
*content-conditioned* (route based on tensor uncertainty, divert
work on prediction error, persist "inner monologue" state across
realize calls, distinguish self-state from sensory inputs). Today,
Fuel's surface lets you set placement hints and define custom ops,
but doesn't expose enough for an agent runtime to live cleanly above
without monkey-patching.

The right architectural cut: Fuel provides **theory-neutral
primitives** (metadata slots, runtime callbacks, persistent values).
The downstream library defines what GWT / IIT / Active-Inference /
hybrid semantics mean on top of those primitives. Fuel never ships
`enum SensoryBus` or `Op::DivertOnPredictionError` — those are the
agent library's concern, not Fuel's.

#### What this is NOT

- Fuel does not become an AGI framework. The AGI semantics live
  one layer up.
- No within-graph cycles. AGI's "inner monologue" is multi-step
  *streaming*, not a directed cyclic graph. Streaming is implemented
  by a Rust-level realize loop reading and writing persistent values
  between realize calls; the graph itself stays acyclic.
- No `Op::If` / `Op::Branch` in the graph. Conditional execution is
  Rust-level control flow above the realize loop, not a graph-level
  primitive. (Same reason every mature ML framework that experimented
  with in-graph control flow ended up regretting it.)

#### Three deliverables

**9a. Per-tensor user metadata slot.** A small additive change.
`Tensor::with_metadata(Arc<dyn Any + Send + Sync>)` builder +
`metadata() -> Option<&Arc<...>>` accessor. Metadata travels with
the lazy graph node, survives optimization passes (canonicalization
must preserve it), and is observable via `SchedulerRule` callbacks
and on the realized output. Fuel itself never reads or interprets
the contents — they're an opaque user payload. Sized: ~1-2 days
including survival-through-optimizer testing.

**9b. Runtime executor hooks.** Today's `SchedulerRule` runs at
plan time on the static graph. Add a sibling `RuntimeHook` trait
that fires after each node's realize:

> **Stale signature (flag, low):** the `output: &dyn DynBackendStorage`
> parameter below predates Phase 7.6 step 9c's bridge-retirement,
> which retires `DynBackendStorage` entirely (see [Bridge-retirement
> trajectory post-9c](#bridge-retirement-trajectory-post-9c), step 6).
> When 9b is actually built, this surface should be re-typed against
> the byte-storage substrate the executor uses by then, not the
> retired trait.

```rust
pub trait RuntimeHook: Send + Sync {
    fn on_node_realized(
        &self,
        id: NodeId,
        op: &Op,
        output: &dyn DynBackendStorage, // STALE: DynBackendStorage retires in 7.6 step 9c
    ) -> HookAction;
}

pub enum HookAction {
    Continue,                 // proceed with the planned next node
    Skip(NodeId),             // jump execution to a later node
    Inject(GraphFragment),    // splice new nodes into the plan
}
```

Lets the agent library steer execution mid-realize without
rewriting the executor. Output observation has to handle GPU
residency cleanly — either lazy host-readback on demand, or
shape-only metadata for the cheap path. Sized: 1-2 weeks for design
plus careful implementation. Real value beyond AGI: debugging,
tracing, checkpointing.

**9c. Named persistent values across realize calls.** Generalize
KVCache's "pre-populate + survive across realizes" pattern to
arbitrary user-named handles. `PersistentStore::write(name, tensor)`
plus `Graph::read_persistent(name)`. Each realize can read and write
across step boundaries; the agent library's outer realize loop
threads state by name. Covers "inner monologue," "world model
across observations," and any other multi-step state pattern.
Sized: ~1 week — KVCache is most of the work, generalization is
mostly API shape.

#### Success criteria for Phase 9

- A downstream cognitive-architecture library exists that uses 9a,
  9b, and 9c to implement at least one published theory of cognition
  (GWT, IIT, Active Inference, or a hybrid) without modifying Fuel
  source. The library's behaviour can be reasoned about purely in
  terms of those three primitives plus normal Rust control flow.
- Fuel's anti-goals (above) still hold: no AGI semantics, no theory
  of cognition, no agent abstractions inside Fuel itself.
- The hooks have at least one non-AGI consumer too (debugging,
  tracing, checkpointing) — pure single-purpose hooks tend to drift
  toward the consumer's specific needs over time, which we want to
  avoid.

#### Order of delivery

9a first (small, additive, immediately useful for diagnostic
metadata even pre-AGI). 9c next (KVCache generalization is its own
internal cleanup). 9b last (biggest investment, biggest design
risk, depends on the executor staying stable for the rest of Phase
6/7/8). Total: ~3-4 weeks across all three when a consumer needs
them.

---

### Phase 10 — Equivalence-rewrite search: device-shaped graph alternatives (research-flavoured)

*Not urgent. Future-facing. Do not pull forward. Sequenced after the
eager-retirement program completes and after the picker arc has
accumulated real Judge telemetry in production use. Builds entirely
on existing seams — rule registry, AlternativeSet, Judge, copy
insertion, SystemTopology — composing them rather than laying new
foundation. Inference-first; rewrites on the backward path are
explicitly out of scope for v1 (they can change training convergence
even inside tolerance).*

#### Why Phase 10 exists

The picker (fuel-dispatch ranker + Judge + selector chain) answers
"which kernel should run this node, on which device?" — but it takes
the graph's *shape* as given. For most op families that's correct:
GPU-optimal and CPU-optimal differ inside the kernel, not in the
graph. For a meaningful minority, the best *algorithm* differs per
target, and the graph shape should change with the placement:

- **Convolution**: im2col+GEMM vs direct vs Winograd — same math,
  opposite device preferences.
- **Attention**: FlashAttention's fusion is a GPU memory-hierarchy
  optimization; a cache-blocked CPU path may prefer the decomposed
  softmax chain it replaced.
- **Matmul reassociation**: `(AB)C` vs `A(BC)` — identical result,
  wildly different FLOPs/traffic depending on dims.
- **LoRA**: `Wx + B(Ax)` vs `(W+BA)x` depending on batch size.
- **Fuse-vs-stay-lowered**: fusion always wins today
  (`optimize_to_fixpoint` gives it the last word), but whether the
  fused form is best depends on whether the target backend has a
  good fused kernel — which the binding table already knows.

Generalizing the picker from "choose a kernel for this node" to
"choose among mathematically equivalent subgraphs for this region"
lets Fuel automatically retarget parts of a model GPU-shape ↔
CPU-shape (or NPU/TPU later) wherever the cost model — transfer
costs included — says it wins.

**Prior art**: this is graph-substitution search — TASO (SOSP '19),
PET (OSDI '21), Tensat (MLSys '21; equality saturation via the Rust
`egg` crate), and Unity (OSDI '22; joint rewrite + placement, the
closest analogue to Fuel's topology-aware version). Published wins:
1.3-3× on real models. Discovery of *new* equivalence rules is an
offline / out-of-the-hot-path tool (TASO-style enumeration +
verification), never part of the per-realize dispatch loop. (This
does **not** rule out the **idle-time** Fuel-strategist JIT loop of
[G7 in 10-decisions-log](docs/architecture/10-decisions-log.md):
that loop also runs *off the realize hot path*, whole-machine
resource-aware, and feeds results back through background
re-optimization at safe boundaries — it is a non-realize-time
discovery activity, exactly the category this line permits.)

**Relationship to the closed-loop adaptive optimizer ([G7](docs/architecture/10-decisions-log.md)).**
Phase 10 is the *static / offline* face of the same machine the
2026-06-20 adaptive-fusion decision describes; 10a (fuse-vs-lower as
a cost-gated picker decision) is the **seed** of the closed loop.
The shared substrate is the **base map** (total per the recipe
principle): both Phase 10's rewrite search and the JIT loop read it
and look for a better cover. The loop adds **explore/exploit**:
**co-occurrence telemetry** (frequency-counted realized chains, the
missing-fusion signal above) is the **exploration prior** ordering
which regions to JIT first; empirical **winning** (a kernel/path
entering an optimized plan under cost-gated selection) is the
**exploit posterior** — ground-truth fitness — and **win-rate
flattening is the STOP signal**. And it keeps the constitution
intact via the **Fuel-strategist / backend-synthesizer** split:
**Fuel** chooses *which* sub-base-map region to fuse and *when*
(idle-time, resource-aware), sending **partial** base maps and
making the cost-gated adopt/reject call; a **trusted backend**
(Baracuda) synthesizes the best kernel for that Fuel-chosen region —
**no backend-side opportunity-finding** (this is not backend-internal
fusion, [09-non-goals](docs/architecture/09-non-goals.md)).

#### Phase 10 building blocks (status today)

| Need | Status |
| --- | --- |
| Rewrite engine | ✅ `fuel-graph::opt` rule registry; `RuleFamily::Algebraic` exists |
| Equivalent forms in the IR | ✅ every lowering/fusion pair IS two equivalent designs — chosen globally + statically today |
| Per-device empirical cost | ✅ Judge measures real kernels per (op, dtype, size class, backend, device) |
| Transfer-aware placement cost | ✅ `insert_cross_device_copies` + SystemTopology transfer paths |
| Precision bookkeeping | ⚠️ `PrecisionGuarantee` is per-kernel; rewrite rules need per-rule deltas |
| Choice mechanism | ❌ `optimize_to_fixpoint` is one-way greedy, first-match-wins; nothing lets two equivalent forms coexist while a cost model picks |
| Equivalence rule library | ❌ one resident (cast fusion) |
| Offline rule discovery + verification | ❌ |

#### Work items (in delivery order)

- [ ] **10a — fuse-vs-lower as a picker decision.** The minimal
      version of the whole idea, **and the seed of the closed-loop
      adaptive optimizer** ([G7 in 10-decisions-log](docs/architecture/10-decisions-log.md)).
      Instead of fusion firing unconditionally, the fused form and the
      lowered composition become per-subgraph alternatives ranked by
      Judge data (it already profiles fused kernels against composed
      primitives). Reuses everything; no new theory. This alone
      captures the "backend lacks a good fused kernel" case — and that
      case is exactly the closed-world missing-fusion signal
      (`FusionMissRecord`, reason `NoBackendKernel`) whose consumer is
      a binding-table append; the JIT loop extends the same
      cost-gated fuse-vs-lower choice to *synthesize* the missing
      kernel for a Fuel-chosen region rather than only fall back to the
      lowered form.
- [ ] **10b — curated algebraic equivalence library.** Grow
      `RuleFamily::Algebraic` from 1 rule to dozens (hand-curated;
      this covers most of the published win). Every rule carries a
      declared precision delta feeding the existing precision-floor
      filters, and is cost-gated by Judge data instead of
      always-fire.
- [ ] **10c — search + joint placement.** Replace greedy rule
      application with search over rule applications, extracted
      against Judge costs + transfer costs jointly with device
      assignment. Escalation order: cost-gated greedy → backtracking
      over windows (TASO-scale graphs are fine) → e-graphs + ILP
      extraction (`egg`) only if backtracking hits combinatorial
      limits. Runs at `compile_plan` time with the plan cached —
      never per-realize.
- [ ] **10d — offline rule discovery.** TASO-style: enumerate small
      candidate graphs over the OpKind vocabulary, verify equivalence
      against `fuel-reference-backend` + `fuel-correctness-fixtures`
      as the oracle, emit rules into 10b's library. A build-once
      tool run per op-vocabulary change, not a runtime feature. This
      is the "automatically discover alternative designs" endpoint.

#### Hard constraints

- **Floating-point equivalence is tolerance-bounded, never exact.**
  Reassociation, Winograd, and factoring all change low bits. The
  gelu erf-vs-tanh incident (fixed `9b53da38`) is the cautionary
  tale: a ~1e-4 flavor divergence hid inside the 1e-3 consensus
  epsilon for two weeks. Per-rule precision deltas are mandatory,
  not decorative — the tolerance story is the actual hard part of
  this phase; the search is the easy 80%.
- **Transfers gate cross-device wins.** A CPU-shaped rewrite only
  wins if the subgraph amortizes 2× PCIe; the extraction objective
  must always include copy costs (it can — the pieces exist).
- **No rewrite fires without a cost-model win.** Equivalence alone
  is never sufficient justification.

#### Success criteria for Phase 10

- At least one anchor model where a device-conditional rewrite
  (fuse-vs-lower or a conv-algorithm choice) measurably beats the
  always-fuse pipeline on at least one backend, with the win
  attributable in the plan trace.
- Every rule in the library carries a verified precision delta; a
  rewritten graph's end-to-end tolerance is computable from the
  rules applied.
- Search cost is invisible in steady state (plan-cache hit) and
  bounded at cold compile.

#### Phase 10 honest caveats

This is a research-flavoured effort with strong prior art, not a
routine engineering task. 10a is days and worth doing early once the
gates clear; 10b-10c are multi-session; 10d is its own multi-week
tool. The phase exists in this document so the idea survives — the
2026-06-10 design discussion that produced it concluded Fuel is
unusually well-positioned (4 of 5 pieces already built,
device-agnostic and empirical by design), but that none of it should
interrupt the eager-retirement program or the picker arc.

---

### Phase 11 — Model modernization: inexact substitution as a declared capability trade (research-flavoured)

*Not urgent. Future-facing. Do not pull forward. Explicitly LOWER priority
than Phase 10, which it depends on conceptually. Recorded from a 2026-08-26
design discussion with CireSnave so the shape survives; nothing here should
interrupt any current program.*

#### Why Phase 11 exists, and how it differs from Phase 10

Phase 10 searches among **mathematically equivalent** subgraphs — same math,
different shape, chosen by cost. Its entire safety argument is that the result
is unchanged.

Phase 11 is the other half: substitutions that **change what the model
computes**, adopted deliberately. Two motivations, and the second corrected the
framing:

1. **Modernization.** A technique that did not exist when a model shipped may
   compute *nearly* the same thing far more cheaply.
2. **CAPABILITY TRADE — the direction Phase 10 cannot express.** A user may
   want a substitution that is *worse* on speed or memory because it is
   *better* on something they care about (context length, memory ceiling,
   batch behaviour). **Substitution is not always optimization.**

⚠️ **THOSE TWO NEED DIFFERENT VERIFICATION SHAPES, and conflating them is the
first trap.** An optimization must prove **nothing changed**. A trade must
prove **the expected thing changed, and ONLY that**. A single
"variance-within-budget" gate answers the first question and is meaningless for
the second.

#### The classifier already exists: base-map equality

`decompose` is total and the primitive basis is build-time closed, so every
fused op has a canonical primitive form and `lower_to_base_map` dissolves any
fused op back to it. Recipe identity is already defined as **base-map-hash
equality against Fuel's own recipe**.

So *"is this replacement the same function?"* is **mechanically decidable
here** — which most frameworks cannot do — and it is the right authority
boundary:

- **base maps agree** → exact rewrite → Phase 10 / the existing optimizer. No
  consent needed; verified by FKC precision contracts.
- **base maps differ** → a different function → Phase 11, requiring an explicit
  per-model opt-in.

**Use the structural test as the boundary, not a tolerance.**

#### The axis that matters on the inexact side

Not exact-vs-approximate. **BOUNDED A PRIORI vs OBSERVED ON A SAMPLE.**

- **Bounded a priori** — the error is a property of the *encoding* and holds for
  every input. **Quantization is the shipped precedent** (`qmatmul`,
  `nf4_matmul`, QuantizedLlama): Fuel already accepts a function-changing swap
  on numerical grounds, because the bound is known before anything runs.
  KV-cache quantization is the obvious next member.
- **Observed on a sample** — the error is input-dependent and unbounded.
  "Within budget across the runs we did" is a **sample statistic, not a bound**.
  Two sequences agreeing on an eval set can diverge arbitrarily on an input
  nobody tested.

⚠️ **THE FAILURE MODE IS NOT ELEVATED AVERAGE ERROR — IT IS RARE CATASTROPHIC
DIVERGENCE ON SPECIFIC INPUT CLASSES**, and the cheap metric cannot see it.
Linear attention's published weakness is *long-range recall*; a variance check
over general prompts looks excellent while the model has lost exactly that.
**The metric that would pass is the metric that cannot see the failure** — the
vacuous-fixture defect one level up, with the *eval corpus* collapsing the axis
under test.

#### Weight compatibility is a first-class field, not a footnote

*Recalled, NOT measured here — verify before building on it:* post-hoc
linearization of a softmax-trained model does not degrade gracefully, it
collapses. Published approaches that make it work (SUPRA, LoLCATs,
Mamba-in-Llama) all include a distillation or finetuning step. The weights
encode softmax attention's inductive bias.

So each substitution declares:

- **WEIGHT-COMPATIBLE** — applies to existing weights (RoPE scaling / YaRN,
  sliding-window, quantization). Fuel can perform it.
- **REQUIRES CONVERSION** — needs training Fuel does not do. Fuel's honest
  output is to **say so and name what it would take**, not to attempt it.

#### The output shape is a MENU WITH COSTS, not accept/reject

For *"I want longer context on this model"*:

| route | weight-compatible | cost |
|---|---|---|
| RoPE scaling / YaRN / LongRoPE | **yes** — already in-tree (`rope_scaling`) | quality loss at extended lengths |
| sliding-window attention | **yes** | loses long-range recall by construction |
| linear attention | **no** — needs distillation | not a substitution Fuel can perform alone |

**Fuel telling a user "the thing you asked for needs a training step, but here
are two that do not" is a better product than Fuel attempting it.**

#### Consent is capability-level, and the object does not exist yet

⚠️ **MEASURED 2026-08-26: Fuel has NO model-level variance budget.** The
precision contracts are **per-kernel, per-dtype**, and their own evidence clause
reads *"not evidence about other inputs, OTHER PARAMETER CONFIGURATIONS, other
machines, or other compilers."* **A per-kernel `max_ulp` cannot authorize an
architectural swap** — that is justification-scope mismatch, the GAP-166 shape.

The consent object must be **built**, and should name a **capability** rather
than a scalar: *"I accept up to X% degradation on long-context retrieval"* is
something a user can mean. *"1e-3 variance"* is a number whose relationship to
what they care about is unknown.

#### Verification: adversarial to the DECLARED weakness

Every approximate substitution's registry entry carries **what it is known to
degrade**, and its acceptance suite probes *that*, not aggregate variance.
`linear-attention: degrades long-range recall` → a retrieval probe at full
context. `kv-quant: degrades rare-token precision` → probe that. Structurally
the same obligation GAP-228 made mandatory for precision attestations: **a claim
carries its basis AND its coverage.**

#### The substitution-effects registry

Store **effects**, not verdicts — a capability/cost profile per
`(model, substitution)`. Architecturally the same object as the precision ledger
(`(kernel, dtype) → earned record with basis and coverage`); reuse that
machinery rather than inventing one. Three requirements, each learned
expensively elsewhere in this repo:

1. **Every effect carries its coverage.** An effect measured at 4k context says
   nothing about 32k.
2. ⚠️ **EFFECTS MEASURED IN ISOLATION DO NOT COMPOSE — and this will bite the
   re-attempt engine hardest, because combining is its whole job.** If an
   attention swap costs 3% retrieval and KV-quant costs 2%, **you cannot
   conclude the pair costs 5%.** They interact. A profile built from isolated
   measurements is a set of per-item claims standing in for a claim about the
   whole.
3. **A rejection needs a re-check trigger or it is permanent by inertia.**
   *"Rejected for Llama-2"* was true against one Fuel version, one kernel set,
   one hardware target. Tie it to a checkpoint that will occur.

And **make optimization goals a declared object**: then the re-attempt set when
goals change is a **query over the profile**, not a re-derivation.

#### Phase 11 honest caveats

Research-flavoured, and unlike Phase 10 it **does need new foundation** — the
consent object, the adversarial probe suites, and the effects registry do not
exist. The population of genuinely weight-compatible modernizations is **much
smaller than the framing suggests**, and is dominated by the exact cases Phase
10 already covers plus the a-priori-bounded ones (quantization) Fuel already
ships. **The honest near-term value is probably KV-cache quantization, and the
menu-with-costs answer for context extension.** Everything past that is genuine
research.
---

## Eager-retirement follow-ups (post-Phase γ)

Phase γ (the Eager Tensor retirement program) shipped the bulk of the migration
off `fuel_core::Tensor` to `Tensor`, but a handful of items got quarantined
or deferred along the way rather than block the main sweep. Each bullet below
captures one such item with the minimum context needed to pick it up cold in a
future session. Group ordering mirrors the rough cost ladder — the binaries
need lazy ports of missing model families; the WASM crates need a workspace-
wide swap; the fuel-core integration tests are small mechanical fixes; the
fuel-book work was documentation (now closed by deleting the crate — see 4-5);
the lazy-side gaps are net-new primitives.

### 0. Closed (deleted)

These follow-ups were resolved by deleting the underlying binary directory
outright. Captured here so future readers see the decision rather than
wondering where the entry went.

- **mamba-minimal** — deleted 2026-06-07. The `_mamba-minimal_retired/` directory was supplanted by the full `lazy_mamba` + `lazy_mamba2` ports (both already migrated with working binaries), so the legacy minimal demo had no remaining consumer. No workspace member referenced it; the directory held only a stale README.
- **llama_multiprocess** — deleted 2026-06-07. The `_llama_multiprocess_retired/` directory was already emptied of `.rs` sources in Phase H, and `docs/session-prompts/lazy-multi-process-inference.md` explicitly recommends deferring a lazy multi-process driver until a real Fuel consumer needs multi-GPU tensor-parallel inference. Until that demand lands, there's no point keeping the empty directory around — when the work is picked up, the session prompt has everything needed to recreate `fuel-examples/examples/llama_multiprocess/{main.rs,model.rs}` from scratch against the lazy substrate.

### 1. Re-migrate the 10 quarantined `fuel-examples` binaries

> **The retired eager sources these ports referenced were DELETED 2026-07-31.**
> `fuel-transformers/src/_models_retired/` (208 files, ~3.1 MB) plus
> `_fused_moe_retired.rs`, `_quantized_nn_retired.rs` and
> `_quantized_var_builder_retired.rs` were never declared as modules in
> `lib.rs`, so they had not been compiled since Phase H — they were dead
> weight that inflated every grep and made the eager surface look ~5455
> references larger than it was (the live figure was 2). Same precedent as
> the deletions recorded in §0 above.
>
> **They remain the porting reference — retrieve them from git, not from the
> working tree.** Last commit containing them: **`19365b07`**. For example:
> `git show 19365b07:fuel-transformers/src/_models_retired/audio/metavoice.rs`
> or `git checkout 19365b07 -- fuel-transformers/src/_models_retired/<path>`
> to restore one temporarily while porting it.

Each binary was set aside because its target model family doesn't yet have a
lazy port. Restoring each one means landing the lazy port called out, then
doing the standard binary swap (lazy_X imports + lazy weight loader +
Tensor signatures).

- **debertav2** — needs `ForMaskedLM` + `ForSequenceClassification` heads in `lazy_debertav2` (encoder body already ports cleanly; the two task heads are the missing piece).
- **xlm-roberta** — needs `ForMaskedLM` + `ForSequenceClassification` heads in `lazy_xlm_roberta` (same shape as debertav2 — encoder ready, heads missing).
- **csm** — needs the autoregressive generation loop driver in `lazy_csm`; the underlying transformer blocks are already there, the AR decode harness is what's missing.
- **metavoice** — needs a `lazy_encodec` port (MetaVoice's neural audio codec dependency); the MetaVoice text-to-speech model itself can land once Encodec is available.
- **stable-diffusion-3** — needs the full `lazy_sd3` family: the triple-CLIP text-encoder composer (CLIP-L + CLIP-G + T5-XXL), the SD3 VAE, and the flow-match Euler sampler with SLG (Skip Layer Guidance).
- **llava** — needs `HFLLaVAConfig` + `LLaVAConfig` + `utils::select_best_resolution` in `lazy_llava` (the multi-resolution image preprocessing helper that picks the closest supported grid); the underlying CLIP + LLaMA ports are already lazy.
- **paddleocr-vl** — still needs an `HFConfig` helper in `lazy_paddleocr_vl` to bridge HuggingFace-style config JSON to the fuel-internal config struct; layer code is already lazy.
- **quantized-lfm2** — needs the base `lazy_lfm2` port to land first (LFM2 currently has no fp32/bf16 lazy port at all, so the quantized variant has nothing to specialize over).
- **rwkv** — needs the RWKV tokenizer ported (~95 LOC, inline-able) into `lazy_rwkv5` or a sibling tokenizer module; the model layers themselves are already lazy via `lazy_rwkv5` / `lazy_rwkv7`.
- **trocr** — needs `vit` + `trocr` ported into `lazy_vit` / `lazy_trocr` internals (the OCR-specific decoder is the trocr-specific part; the ViT encoder body is shared with the broader ViT port).

### 2. ~~Re-migrate the 10 fuel-wasm-examples crates + fuel-wasm-tests~~ — WASM RETIRED 2026-08-14

> **⚠️ WASM SUPPORT IS RETIRED. HARD BREAK, CireSnave's decision 2026-08-14.** The
> `fuel-wasm-examples` tree (11 crates, 100 files), the WASM SIMD128 kernels
> (`fuel-cpu-kernels/src/simd128.rs`, `fuel-quantized/src/simd128.rs`, and their
> `k_quants` dispatch arms), the `with_simd128()` capability probe, the
> `[target.wasm32-unknown-unknown]` cargo config block and gemm's
> `wasm-simd128-enable` feature are all **deleted**. Retrievable from git history.
>
> **WHY, so it is not relitigated from the code's absence.** Two of Fuel's three
> optimizer premises do not hold on `wasm32-unknown-unknown`: the cost model
> assumes *"wall-clock under expected concurrency, not strict-serial sum"* and the
> placement DP needs more than one device — a single-threaded, single-backend
> target switches both off. **And weight storage is required to support OS-level
> page sharing and lazy residency (mmap), which is what makes MULTI-SESSION
> SERVING memory-viable — WASM has no analogue.** So WASM was never a smaller
> Fuel; it was a Fuel with the optimizer's premises removed.
>
> It had also been **unbuildable for its own target for an unknown period** and
> nobody noticed, because no gate built for `wasm32`. The tree type-checked on
> HOST, which is what kept it looking alive.
>
> **Not deleted:** `#[cfg(not(target_arch = "wasm32"))]` guards scattered across
> `fuel-ir`, `fuel-cuda-backend`, `fuel-metal-backend` and `fuel-core`. Those are
> guards *against* wasm, now permanently true; removing them is cosmetic churn
> across several crates and is deliberately left undone.


> **ARCHIVED 2026-07-31 — `fuel-pyo3` and `fuel-wasm-tests` are removed from the
> workspace and deleted from the tree.** Both were unmaintained, neither is needed to
> get Fuel working, and both **blocked the eager-`Tensor` retirement (B6)**:
> `fuel-pyo3` was the sole remaining consumer of `fuel_onnx::eval::simple_eval`, and
> `fuel-wasm-tests` was the only consumer keeping `QTensor`/`QMatMul`/`Module` alive.
> `fuel-wasm-tests` additionally **did not compile** (`unresolved import
> fuel::quantized::k_quants`) while being a *default-member*, which is what made a bare
> root `cargo check` fail — see the corrected note in CLAUDE.md.
>
> **Retrieve from git, not the working tree.** Last commit containing them:
> **`087109cf`** — e.g. `git checkout 087109cf -- fuel-pyo3`.
>
> **Also archived 2026-08-01, for the same reason:** `fuel-cublaslt` and
> `fuel-tensor-tools`. Both were 100% eager with no lazy half, and neither is a
> default-member.
> - `fuel-cublaslt` had **zero consumers** — it appeared only in `[workspace.members]`.
>   Its whole public surface (`fused_matmul`, `fused_batch_matmul`) took and returned
>   the eager `Tensor` via `CustomOp2`/`CustomOp3`, both deleted by B6. The cuBLASLt
>   fused-epilogue *idea* is still live as a future FusedLinear competitor
>   (`docs/session-prompts/baracuda-cutlass-alpha-13-integration.md`), but it would
>   need a lazy rewrite anyway.
> - `fuel-tensor-tools`, the GGUF quantize/inspect CLI, was archived and is now
>   **RESTORED (2026-08-01)** — back in `[workspace.members]` and ported to host types
>   exactly as the archive note predicted: `fuel-formats` readers → host `Vec<f32>` →
>   `fuel_quantized`'s `QuantizedType` (`cpu_zeros` + `from_float` /
>   `cpu_from_data` + `dequantize`) → a raw-bytes GGUF writer replacing the deleted
>   `gguf_file::write`. It never needed a tensor: the eager `QTensor` was doing
>   nothing a `Vec<f32>` could not, which is why this was a rewrite rather than a port.
>
>   **One capability did NOT come back: npz.** `fuel-core`'s `npy.rs` was deleted in B6
>   and no npy reader survives anywhere in the tree (`fuel-formats` has ggml, gguf,
>   imatrix, pickle, safetensors — no npy), so restoring it means writing a reader.
>   `pth` is `ls`-only for a related reason: `fuel_formats::pickle` exposes
>   `read_pth_tensor_info` (metadata) but not tensor payloads.


>
> Python bindings are deliberately deferred rather than ported: porting them now means
> porting them twice, and the `tensor.item::<f32>() > 0.5` question that B6 flags as
> "the most user-visible difference from PyTorch eager" is a Python-surface design
> decision best made when the bindings are actually wanted.
>
> **This section's claim that the WASM example tree is "quarantined out of the
> workspace" is STALE:** `fuel-wasm-examples/*` is present in both `members` and
> `default-members`, the 11 crates are real (100 files), and the sampled ones import no
> retired crate. Whether they build is UNVERIFIED.

The entire WASM example tree is currently quarantined out of the workspace
(removed from `[workspace.members]`) because every crate depends on
`fuel_transformers::models::*` (retired in Phase H) and `fuel_nn::*` (retired
in Phase β4). Restoring the tree mirrors the `fuel-examples` program: per
crate, swap to the corresponding `lazy_X` module, and inline small helpers for
the handful of API points that don't have a 1:1 lazy equivalent yet (notably
`ops::softmax`, the `Linear` layer wrapper, and the VarMap-style loaders the
WASM binaries use for compact safetensors loading).

- `fuel-wasm-examples/bert`
- `fuel-wasm-examples/blip`
- `fuel-wasm-examples/llama2-c`
- `fuel-wasm-examples/moondream`
- `fuel-wasm-examples/phi`
- `fuel-wasm-examples/quant-qwen3`
- `fuel-wasm-examples/segment-anything`
- `fuel-wasm-examples/t5`
- `fuel-wasm-examples/whisper`
- `fuel-wasm-examples/yolo`
- `fuel-wasm-tests`

### 3. Fuel-core integration test fixups (pre-existing breakage, not retirement-related)

These three integration tests were already broken before Phase γ started, but
were left untouched during the sweep so the retirement diffs stayed focused.
They are small mechanical fixes against the current `Tensor` + storage-seam
API, not architectural follow-ups.

- `tests/phase6b_cuda_anchor.rs` — the `realize_f32_*` methods are now `Result`-returning; needs `?` insertion at the call sites. Separately, `ClipTextConfig` gained a required `activation` field that this test's fixture doesn't populate.
- `tests/cuda_composed_bisect.rs` — same `Result`-vs-`Tensor` mismatch across `realize_f32`, `matmul`, and `rms_norm_last_dim` call sites.
- `tests/tensor_tests.rs` — the storage seam now returns `Arc<RwLock<Storage>>` instead of a `RwLockReadGuard`; the test reaches through the old guard shape and needs to be retargeted at the `Arc<RwLock<...>>` API.

### 4-5. fuel-book (doctest cleanup + markdown docs) — CLOSED by deletion (2026-08-14)

Both follow-ups were about porting `fuel-book` content off the eager API
(`fuel-book/src/simplified.rs` was `mod`-gated on retired `fuel_nn`; five `.md`
files under `guide`/`inference` referenced `fuel_nn` in prose + inline code).
`fuel-book` was fork-inherited Candle book content whose snippets and prose
had diverged from Fuel and still used the API retired in B6. Rather than port
it (no live consumer), the whole crate was deleted — closing both items and
B6's residual. Recoverable from git if a Fuel book is written later.

### 6. Lazy-side primitive gaps surfaced during retirement (defer to follow-up)

These three primitives were tagged during Phase γ as "would have been nice to
have during the binary migrations" but were not load-bearing for any binary
that actually shipped — each one was worked around at the call site. They
warrant first-class lazy implementations when a downstream consumer needs them.

- **General-axis softmax on Tensor** — currently only `softmax_last_dim` is exposed; a general `softmax(axis)` would close a recurring port-time papercut for models that softmax over a non-trailing axis (typical in attention rewrites and some vision heads).
- **`max_pool2d` with `-inf` padding** — only the zero-padded variant is exposed today; some segmentation and detection heads expect `-inf` padding (so padded positions can never win the max). The shape is the same as the existing zero-padded kernel with a different fill constant; the lazy fanout is the work item.
- **`Conv2d::absorb_bn` helper** — for inference-time folding of a following BatchNorm into the conv's weight + bias (the standard "fuse BN" optimization). Small algebraic helper; deferred because no current lazy binary needs it at the API surface (the folding ports that do exist do it ad-hoc at load time).

---

## Anti-goals by layer

These are explicit rules. When a proposed addition fits one of these descriptions,
the answer is always no for that layer — find the right layer instead.

| Layer                                 | Will never contain                                                                                                    |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Foundation (`fuel-core`)              | Tokenization, model-family assumptions, serving abstractions, HF Hub client code                                      |
| NN (`fuel-nn`)                        | Model-architecture implementations, inference session management, decode loops, training loops                        |
| Models (`fuel-transformers`)          | Serving infrastructure, batching schedulers, streaming decode loops, session lifecycle, training policy               |
| IO (`fuel-core` IO + `fuel-onnx`)     | Runtime policy, model architecture logic, serving abstractions                                                        |
| Inference (`fuel-inference`)          | New tensor primitives, new dtypes, new backend dispatch, training policy, anything that redefines foundation concepts |
| Training (`fuel-training`)            | New tensor primitives, new dtypes, inference-specific concerns (KV caches, sampling, decode loops)                    |
| Backends/Kernels                      | ML concepts, model logic, layer abstractions, training or inference policy, anything above shaped memory and math     |

---

## What will not change

- Published crate names will not be renamed speculatively. Renaming happens only
  after the new shape has proven itself, per the sequencing principle: define →
  document → reorganize → extract → rename.
- The early-exit property. A user who only wants tensor math must never be
  required to carry inference infrastructure.
- The breadth of model implementations. `fuel-transformers` is a genuine asset.
  The goal is to give it structure, not reduce its scope.
- Minimum viable complexity. Simple programs should stay simple. The framework
  should feel small from the bottom and powerful from the top.

---

## Dependency graph (target state)

```text
fuel-inference ─────────────────────────────────────────────────┐
fuel-training  ─────────────────────────────────────────────────┤
       │                                                      leaf crates
       │  both depend on                                          │
       ▼                                                          │
fuel-transformers ──────────────────────────────────────────────┤
       │                                                          │
       │  depends on                                          IO layer
       ▼                                                          │
fuel-nn ────────────────────────────────────────────────────────┘
       │
       │  depends on
       ▼
fuel-core  (eager path today; the lazy path in fuel_core::lazy
   │        wraps fuel-graph + fuel-graph-cpu + fuel-reference-backend)
   │
   │  depends on  [feature flags select which backend crates are compiled]
   ▼
fuel-cpu-backend          fuel-cuda-backend         fuel-metal-backend
    (always)                [feature = "cuda"]          [feature = "metal"]
                                   │                           │
                                   ▼                           ▼
                       fuel-cuda-kernels         fuel-metal-kernels
                       fuel-flash-attn
```

### Phase 6 sub-graph (the lazy layer)

```text
fuel-lazy-examples ─────────────────────┐
                                        │
                                      runnable
                                        │
                                        ▼
fuel-core::lazy (Tensor, LlamaModel, LlamaTokenizer, generate)
    │
    │  builds on
    ▼
fuel-graph-cpu (gemm-backed fast executor; `realize_*` entry points)
    │
    │  depends on, for non-matmul ops
    ▼
fuel-reference-backend (textbook-correct oracle; also provides RefTensor)
    │
    │  depends on
    ▼
fuel-graph (Op enum, Graph arena, Tensor handle, topo_order, backward)
    │
    │  depends on
    ▼
fuel-core-types (Shape, DType, Layout, BackendStorage trait, errors)
```

### Phase 7 sub-graph (vendor-optimized CPU backends)

```text
                Phase 6b dispatch table  (empirical per-op winner)
                              │ picks
       ┌──────────────────────┼──────────────────────────┐
       ▼                      ▼                          ▼
 fuel-aocl-cpu-backend  fuel-graph-cpu / fuel-cpu-     fuel-mkl-cpu-backend
       │                  backend (pure Rust;                │
       ▼                   gemm under                        ▼
   aocl-blas               the hood)                      onemkl
       │                                                    │
       ▼                                                    ▼
  AOCL BLIS runtime                                  Intel oneMKL runtime
   (external crate                                    (external crate
    aocl-blas-sys)                                     onemkl-sys)

All CPU backends implement the same GraphBackend trait surface and share
AnyRefTensor storage (so switching among them is a vtable swap, not a
transfer). Consumers enable backends via Cargo features (aocl, onemkl,
later accelerate / armpl / openblas). Phase 6b's Judge profiles each
loaded backend; the Router's pick_for_op consults the dispatch table per
op. No raw-cpuid heuristic picker — empirical wins on data, not on
vendor brand.
```

The backend crate split is the Tier 2 target state from Phase 5. Before that
landmark, the graph is the same but the backend code lives inside `fuel-core`
modules rather than separate crates.

Side dependencies: `fuel-onnx` and `fuel-datasets` depend downward as needed.
`fuel-pyo3` wraps whichever layers it needs without influencing them.
`fuel-dispatch` (Phase 5 Tier 4, long-term) sits between `fuel-core` and
new user-facing op-sequence APIs, with no effect on any layer above or below.
