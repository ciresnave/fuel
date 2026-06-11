# Executor unification — re-audit 2026-06-11

Refresh of the **Phase 7.6 step 9c parity audit** (2026-05-19, memory:
`project_phase_7_6_step_9c_parity_audit.md`; ROADMAP §"Phase 7.6 step
9c — typed-storage retirement"), merged with the **eager-Tensor
retirement** state (`eager-tensor-retirement-master-plan.md` +
`eager-retirement-phase-h-plan.md`). Read-only audit against
`main` @ `91c0afd5` (2026-06-11); all evidence is file:line or commit
hash in this tree.

**Method note on counts.** The 2026-05-19 audit reported "242 call
sites across 34 files". Today the same grep
(`GraphExecutor|fuel_graph_executor`) finds **221 occurrences across
41 files — ~212 in code across 36 files** (9 are docs/ROADMAP
references). The raw delta understates progress: the count includes
doc-comments, the fuel-graph-executor crate itself, and ~150
occurrences in test files. The production-path survivors are a much
smaller set, clustered below.

---

## 1. Headline state

- **PipelinedExecutor is feature-complete for everything the
  production realize paths need.** `WorkItemKind` now has 13 arms
  (`fuel-dispatch/src/pipelined.rs:62-234`): ConstAdopt, ViewOf,
  ContiguizeOf, Kernel, ReleaseMarker, WriteSlice,
  WriteSliceRotating, Copy, Alloc, ZeroFill, InplaceKernel, SlotView,
  SlotOwn. Of the original ~12 feature gaps, **9 are closed, 1 is
  closed-by-supersession, and 2 remain open** (GraphBackend
  disposition + Router rewiring) plus one narrow technical gap
  (`Op::Move` has no executor arm).
- **The legacy `GraphExecutor<B>` survives in 4 production seams**
  (Judge profiling via `factories::LazyRealizer`; the
  `judge::cached()` Router branch inside `LazyTensor::realize_f32`;
  the `*_gpu_on` generate/spec-decode family on LlamaModel/PhiModel/
  Llama2c; `train.rs`) **plus test/example callers**.
- **A third evaluator exists** that the original audit didn't track:
  `fuel-graph-cpu::realize_any` (typed CPU recursive evaluator,
  `fuel-graph-cpu/src/lib.rs`, 2798 LOC). It still backs
  `LazyTensor::realize_f64/_bf16/_f16` (`fuel-core/src/lazy.rs:1316-1328`)
  and the `GraphBackend` CpuBackend impl. Its `AnyTensor` enum covers
  only F32/F64/BF16/F16/U32 — **no U8** — which is consistent with
  the standing `lazy_encodec` "legacy-executor U8 gap" failure label.
  Full unification retires this surface too.
- **Eager-Tensor retirement is much further along than the master
  plan's status table records.** Phases β1–β4, γ, and the
  Workflow-C-shaped lazy phases 1–7 shipped (commits `1ab1d0c9`,
  `f95a6b29`, `565f83b7`, `34fb6190`, `5fb0ee6e`…`7c50b221`):
  `fuel-transformers/src/models` → `_models_retired`, fuel-nn →
  `_fuel_nn_retired`, LogitsProcessor takes `&[f32]`, public eager
  Tensor API hidden. What remains of eager is **internal to
  fuel-core**: the eager `Tensor` + `BackpropOp` tape (91 occurrences
  across 7 live files, concentrated in `fuel-core/src/tensor.rs` (61)
  + `op.rs`/`conv.rs`/`custom_op.rs`/`tensor_cat.rs`/`safetensors.rs`/
  `quantized/mod.rs`) and `train.rs`'s legacy-executor Trainer.

---

## 2. Original gap disposition

| # | Gap (2026-05-19 audit) | Status | Evidence | Remaining effort |
|---|---|---|---|---|
| 1 | Multi-target realize (`realize_many`, `realize_split`) | **closed** | `realize_many` shipped `c5ed169a`; lives at `fuel-dispatch/src/pipelined.rs` (realize_many path, lines 590-630). `realize_split` deferred-by-design: expressible as `realize_many` + caller-side selective download. | — |
| 2 | Side-effect root inclusion | **closed** | `extend_with_side_effect_roots` (`fuel-dispatch/src/pipelined.rs:283`) called in both `realize` (441/456) and `realize_many` (606/618). Shipped `db89a283`. | — |
| 3 | Destructive-input cleanup + ordering | **closed** | `WorkItem.destructive_input` + cache eviction (`pipelined.rs:483` region); `ReleaseMarker` arm (`pipelined.rs:979`). Ordering half: `insert_safety_copies(&mut g, …)` + `execution_plan(&g, …)` wired into BOTH realize paths (`pipelined.rs:442/457` and `607/619`) by commit `2ff321cd` (2026-05-30) — `execution_plan` internally runs `derive_ordering` (`fuel-graph/src/opt.rs:1853`), satisfying the "cleanup ≠ ordering" check. 2 regression tests per `project_pipelined_executor_ordering_shipped`. | — |
| 4 | CPU fallback on backend Err | **closed** (decision + upgrade) | 2026-05-19 decision: fail-fast binding-table contract. Superseded upward by the picker arc: commit `582c55a0` makes missing-impl ops **off-device fallback candidates** — a picker decision producing Op::Copy-bracketed execution on a backend that does have the kernel, instead of either fail-fast or hidden executor-level fallback. Architecturally the end-state the 2026-05-19 note predicted ("graph-level dispatch insertion"). | — |
| 5 | Optimization pass + rule-registry plumb-through | **closed-by-decision, structural passes wired** | Decision (2026-05-19): caller composes (`registry.optimize_to_fixpoint` then realize; demonstrated at `fuel-core/src/lazy.rs:1682-1688`). Since then the *structural* passes run on the production path: `insert_layout_fixups` wired by `95950ea2` (`fuel-core/src/pipelined_bridge.rs:262`), `insert_cross_device_copies` by `efef2836` (`pipelined_bridge.rs:1171`). Full lowering/fusion registry on the default realize path is **deliberately not wired** — that is the load-time incremental planner's scope (program item 3, `docs/session-prompts/load-time-incremental-planner.md`). | — (moves to planner program) |
| 6 | Pre-populate API | **closed** | `realize_*_as_with_initial` + `InferenceContext` persistent StorageCache (Phase E.3.0, commit `a405e7c0`; `fuel-core/src/inference_context.rs`). | — |
| 7 | Const pool with byte budget | **still-open** | `InferenceContext` holds a plain `HashMap<NodeId, Arc<RwLock<Storage>>>` — no LRU, no byte budget (grep for budget/limit/lru in `inference_context.rs`: none). The legacy `with_const_pool_limit` LRU remains the only larger-than-VRAM mechanism, and nothing on the pipelined path replaces it. | ~1 session. Recommend folding into the load-time planner's residency planning (Op::Release/Op::Move scheduling) rather than re-growing an executor-side LRU; decide there. |
| 8 | Typed `realize_f32` etc. | **partially-closed** | f32 entries migrated (Phase E.2, `32d712f7`): `realize_f32`/`realize_f32_cuda`/`realize_many_f32{,_cuda}` + `realize_u32` (`lazy.rs:880`) go through `pipelined_bridge::realize_one_as::<T>`. **But `realize_f64/_bf16/_f16` still call `fuel_graph_cpu::realize_*`** (`lazy.rs:1316-1328`) — the third evaluator, which panics on dtype mismatch and lacks U8. | ~0.5 session: switch the three entries to `realize_one_as::<T>` (already generic), delete the AnyTensor dependence from the public API. Likely also structurally fixes the `lazy_encodec` U8 class of failure. |
| 9 | Eval-node panic context | **partially-closed (acceptable shape)** | Compile/dispatch errors carry NodeId + op context as typed `Result`s (`pipelined.rs:1008, 1057, 1102, 1223, 1334, 3046`) — consistent with the no-panics policy, *better* than the legacy panic-prefix wrapper. Kernel-internal panics still surface raw (no catch_unwind by design). | Optional polish only; fold node-context into kernel-Err wrapping opportunistically. Not a port blocker. |
| 10 | Placement validation | **closed-by-supersession** | Explicit `validate_placements` exists only on the legacy executor (`fuel-graph-executor/src/lib.rs:773`). The pipelined path validates stronger and earlier: `compile_plan` resolves per-node alternatives at plan time and errors typed when no binding exists; per-node `target_backend` checks at `pipelined.rs:1334` etc. Matches "validate at graph-build time". | — |
| 11 | `GraphBackend` trait disposition | **still-open — the load-bearing decision** | Trait: 33 methods (`fuel-graph-executor/src/lib.rs:130-562`). 6 impls survive: CpuBackend (`fuel-graph-cpu/src/backend.rs:13`, 611 LOC file), CudaBackend (`fuel-cuda-backend/src/backend.rs:20`, 242 LOC), VulkanBackend (`fuel-vulkan-backend/src/lib.rs:10093`), MklBackend (`fuel-mkl-cpu-backend/src/lib.rs:154`), AoclBackend (`fuel-aocl-cpu-backend/src/lib.rs:101`), Router (`fuel-graph-router/src/lib.rs:979`). The 2026-05-19 default was "retain as Judge profiling surface" — and that is exactly what holds it in place today: the Judge measures through `factories::factory_for → LazyRealizer → GraphExecutor<B>` (`fuel-core/src/judge/mod.rs:736-758`, `factories.rs:45-50`). **The retain rationale has expired**: the pipelined path can now realize on CPU/CUDA/Vulkan uniformly, the Judge data model already carries `kernel_source` per alternative (commit `1ba99650`), and MKL/AOCL are kernel_source extensions of Cpu (commit `92c0251b`) that the trait-shaped Judge can no longer even distinguish properly. **Recommend: retire.** | ~1 session to re-point Judge/probe onto `pipelined_bridge` (see Session 2 below); trait deletion itself rides Phase F/H. |
| 12 | Router rewiring | **still-open — includes a live production branch** | `impl GraphBackend for Router` (`fuel-graph-router/src/lib.rs:979`) + `ResidencyEvictionRule`. Most surprising survivor: `LazyTensor::realize_f32` has a **production branch** — when `crate::judge::cached()` returns a dispatch table, it constructs `GraphExecutor::new(router)` and realizes through the legacy executor (`fuel-core/src/lazy.rs:1292-1303`). Every user who has run `populate_dispatch_table` gets the legacy executor on the hottest API in the crate. The picker (compile_plan + JudgeOracle Layer-2, commit `899d725e`/`130d2db2`) already consumes Judge data on the pipelined path, so this branch is now a *worse* duplicate of what the picker does. | ~1 session to delete the branch (Session 3); Router crate disposition + cross_device tests ride Phase G (Session 6). |
| 13 | `Op::Move` executor dispatch (deferred at Phase B) | **still-open** | `WorkItemKind` has no Move arm; `Op::Move` appears in `pipelined.rs` only as an exclusion from the InplaceKernel path (`pipelined.rs:995`) and `op_to_op_kind` has no mapping → a graph containing `Op::Move` fails to compile on the pipelined executor. Only emitter: `ResidencyEvictionRule` (fuel-graph-router). | ~0.5 session: reuse `WorkItemKind::Copy` machinery (Move = Copy to target + destructive release of source — both halves already exist). Prerequisite for migrating residency eviction in Session 6. |

### Closures since the audit that exceeded its scope

- **In-place ops through the production executor** — `InplaceKernel`
  arm (`pipelined.rs:204`) + auto `insert_safety_copies`
  (2026-05-30, `2ff321cd`).
- **Multi-output Option C** — `94fa2e47`: SlotView/SlotOwn arms +
  output bundles; unblocked SelectiveScan/SsdChunkScan dual-output.
- **Rotating KV** — `WorkItemKind::WriteSliceRotating`
  (`pipelined.rs:121`): eager-retirement master-plan Phase C shipped.
- **Picker fully load-bearing** — compile_plan → AlternativeSet →
  RuntimeSelector chain (Phase 4 arc, 2026-06-07) + per-node winner
  stamping (`0f8eded0`); prepare() no longer pins backends.
- **Bridge retirement** — Op::Copy (D2H + H2D), Op::Alloc,
  Op::ZeroFill graph-level primitives; all 150 LOC of `7a95001a`
  bridge code deleted.

---

## 3. Execution-surface census (the "three executors" problem)

| Surface | LOC | Role today | End state |
|---|---|---|---|
| `fuel-dispatch::PipelinedExecutor` | (in fuel-dispatch) | Production: all f32/u32 realize, KvCache, Op::Copy/Alloc/ZeroFill, picker, in-place, multi-output | **THE executor** |
| `fuel-graph-executor::GraphExecutor<B>` | 2147 | Judge profiling; `judge::cached()` Router branch; `*_gpu_on` generate family; train.rs; tests/examples | retire (Phase H) |
| `fuel-graph-cpu::realize_any` | 2798 | `realize_f64/_bf16/_f16`; backs CpuBackend's GraphBackend impl | retire with CpuBackend impl; `fuel-reference-backend::exec::realize_f32` stays as the test oracle |

---

## 4. Legacy caller clusters (port order within each tier)

Counts = grep occurrences of `GraphExecutor|fuel_graph_executor`,
code only (~212 total).

**Tier 0 — non-load-bearing (no port needed, delete with crate):**
doc-comments in `fuel-graph/src/registry/*` (5), `fuel-dispatch` (2),
`inference_context.rs`/`pipelined_bridge.rs` (2),
`fuel-cuda-backend/src/lib.rs` `CudaGraphExecutor` (a distinct legacy
const-pool struct, 3).

**Tier 1 — test-only callers (mechanical, port first):**
- `fuel-core/tests`: `g2_from_storage.rs` (1), `cuda_composed_bisect.rs`
  (1) — plain swap to `realize_f32`/`realize_f32_cuda`.
- `fuel-cuda-backend/tests/recip_abs_realize_live.rs` (4) — swap to
  binding-table dispatch test shape.
- `fuel-core/tests/flash_attn_cuda.rs` (4) + `flash_attn_vulkan.rs`
  (3) — deliberately pin the legacy FA2 trait launcher; they retire
  with the queued FA2 eager-wrapper retirement session
  (`docs/session-prompts/fuel-flash-attn-cuda-eager-retirement.md`),
  not before.
- `fuel-core/src/lazy.rs` `#[cfg(test)]` spec-decode tests
  (lines 8486-8705, ~10 occurrences) — port together with Session 4.
- `fuel-graph-router/src/residency_eviction.rs` tests (5) +
  `fuel-graph-router/tests/cross_device.rs` (43) — **pinned to the
  Router's fate**; port in Session 6, not earlier.
- `fuel-vulkan-backend/tests/cpu_vulkan_diff.rs` (44) +
  `conv2d_oracle.rs` (3) — oracle-diff tests over the legacy eager
  Vulkan wrappers; retire with the V.4
  `VulkanBackendDevice`/GraphBackend-impl cleanup (Session 7).

**Tier 2 — single-model / example paths:**
- `fuel-lazy-examples` bins (6): `llama-lazy-vulkan`,
  `phi-lazy-vulkan`, `llama-finetune-vulkan` — each migrates by
  swapping executor for `Device` + `forward_with_kv_context`
  (pattern already documented in the 9c audit's Vulkan section).
- `fuel-core/src/lazy_llama2c.rs` (8): the migrated
  `*_with_kv_context` family (lines 180-244) already coexists with
  the legacy `forward_with_cache_gpu_on`/`generate_streaming_gpu_on`/
  `generate_streaming_spec` family (lines 245-341) — port = delete
  the `_gpu_on` family after consumers move.
- `fuel-core/src/lazy.rs` production (~25): LlamaModel (impl at 6001)
  + PhiModel (impl at 6825) `_gpu_on` generate/spec-decode methods
  (E.3.3/E.3.4); `realize_f32_vulkan` (1363, Judge parity — falls out
  of Session 2); `realize_f64/_bf16/_f16` (1316-1328, Session 1);
  the `judge::cached()` Router branch (1292-1303, Session 3).
- `fuel-core/src/lazy_kv_cache_device.rs` (5): `KVCache<B:
  GraphBackend>` — superseded by `inference_context::KvCache`;
  retires with the `_gpu_on` family.

**Tier 3 — train.rs (12):** `Trainer` is generic over
`GraphBackend` and holds `&mut GraphExecutor<B>` through
`train_step`/`param_to_host` (lines 230-329) + 7 test constructions.
Phase E.4; intersects eager-retirement Phase G (optimizer/Var
mutation, BatchNorm EMA) — the last consumer of the eager
`BackpropOp` tape.

**Tier 4 — Judge/factories + Router (the two open decisions):**
- `fuel-core/src/factories.rs` (8): `LazyRealizer` =
  `GraphExecutor<B>` adapter consumed by `judge/mod.rs:736-778` and
  `probe.rs:63`.
- `fuel-core/src/transfer_cost.rs` (1):
  `measure_round_trip_via_backend<B: GraphBackend>`.
- `fuel-graph-router` (lib 1 + impl): Router-as-GraphBackend +
  residency eviction (`Op::Move` — gap 13).

**Tier 5 — the trait + impls + crate:** 6 GraphBackend impls
(§2 gap 11), the `fuel-graph-executor` crate (2147 LOC incl. legacy
const-pool LRU + `validate_placements`), `fuel-graph-cpu::realize_any`.

---

## 5. Eager-retirement state merge

Done (verify-level evidence):
- `fuel-transformers/src/models` → `_models_retired`;
  `quantized_nn`/`quantized_var_builder`/`fused_moe` retired
  (directory listing; Phase-H plan steps 1-2 executed).
- fuel-nn retired → `_fuel_nn_retired` (commit `565f83b7`).
- LogitsProcessor on `&[f32]` + public eager Tensor API hidden
  (commit `34fb6190`, Phase γ).
- Lazy phases 1-7 (`5fb0ee6e`…`7c50b221`) closed the lazy primitive
  gaps + binary migrations the master plan called Phases A/D/E/F.

Remaining eager remnants:
1. **Eager `Tensor` + `BackpropOp` tape inside fuel-core** — 91
   occurrences across 7 live files (`tensor.rs` 61, `custom_op.rs` 7,
   `op.rs` 6, `conv.rs` 5, `safetensors.rs` 3, `quantized/mod.rs` 3,
   `tensor_cat.rs` 2). Held in place by `train.rs` (Tier 3) and
   internal tests. Deletes after Session 5.
2. **`GraphBackend` impl blocks in backend crates** — the eager
   compute wrappers in `fuel-cuda-backend/src/backend.rs`,
   `fuel-vulkan-backend/src/lib.rs:10093` (large: the eager method
   bodies are a major chunk of the 11.6k-LOC file), CpuBackend,
   MKL/AOCL. These are *legacy-executor* remnants rather than
   eager-Tensor remnants; they delete in Session 7.
3. **FA2 eager wrapper** in fuel-cuda-backend — own queued session
   (memory: `project_fa2_lazy_launcher_migrated`).

Cross-reference to the standing fuel-core failures being fixed by
the ship lanes: `lazy_encodec` ("legacy-executor U8 gap"),
`lazy_quantized_qwen2` ("Q4_0 bake gate"), `lazy_sd3_text_encoder`
("concat graph identity") are all in model paths that still touch
legacy/typed-evaluator surfaces — evidence that the legacy seams are
actively decaying, which is the argument for sequencing this program
now rather than after the planner work.

---

## 6. Recommended session order

Ordering principle (user-stated): test-only first, then single-model
paths, then train.rs, then the trait itself — adjusted where a test
cluster is pinned to a later structural decision (Router tests,
Vulkan oracle tests, FA2 tests).

1. **Session 1 — small gaps + free test callers.** Switch
   `realize_f64/_bf16/_f16` onto `pipelined_bridge::realize_one_as`
   (gap 8); add `WorkItemKind::Move` reusing the Copy + destructive-
   release halves (gap 13); port `g2_from_storage`,
   `cuda_composed_bisect`, `recip_abs_realize_live`. Low risk;
   removes the third evaluator from the public API and likely
   structurally fixes the U8-class failures.
2. **Session 2 — Judge/probe re-point (decides gap 11).** Rebuild
   `factories::LazyRealizer` on `pipelined_bridge` + `Device`
   (CPU/CUDA/Vulkan); migrate `transfer_cost.rs`; delete
   `realize_f32_vulkan`'s legacy signature. This expires the only
   architectural reason to retain `GraphBackend`, and gives the
   Judge per-`kernel_source` measurement through the same dispatch
   path production uses (fixes the structural half of the
   probe/judge Reference-staleness test failures).
3. **Session 3 — delete the `judge::cached()` Router branch** in
   `LazyTensor::realize_f32` (gap 12a). The picker + JudgeOracle
   Layer-2 already consume the same data on the pipelined path;
   updating-the-expectation is the fix. After this,
   PipelinedExecutor is THE executor on every realize entry point.
4. **Session 4 — generate/spec-decode migration (E.3.3/E.3.4).**
   Port LlamaModel/PhiModel/Llama2c `*_gpu_on` families to
   `forward_with_kv_context`; retire
   `lazy_kv_cache_device::KVCache<B>`; migrate the 3
   fuel-lazy-examples bins; port the lazy.rs spec-decode tests.
5. **Session 5 — train.rs (E.4).** Trainer onto
   `realize_many` + InferenceContext. Intersects eager Phase G
   (Var/optimizer/BatchNorm-EMA decisions) — largest design surface;
   surface decisions early, don't bundle with Session 4. Unblocks
   deleting the eager `BackpropOp` tape.
6. **Session 6 — Router/Phase G.** Residency eviction onto the
   pipelined path (uses Session 1's Op::Move arm); decide Router
   retire-vs-rewire-to-picker; port `cross_device.rs` +
   `residency_eviction.rs` tests; const-pool byte-budget decision
   lands here or in the load-time planner (gap 7).
7. **Session 7 — Phase F + H.** Delete the 6 `GraphBackend` impls
   (incl. the big eager Vulkan wrapper block), retire the
   `fuel-graph-executor` crate, retire `fuel-graph-cpu::realize_any`
   (keep `fuel-reference-backend` as the oracle), migrate-or-retire
   `cpu_vulkan_diff.rs`/`conv2d_oracle.rs`. FA2 eager wrapper
   retirement can ride here or its own queued session.
8. **Session 8 — eager tail.** Delete eager `Tensor` + `BackpropOp`
   (the 7 fuel-core files), drop `_retired` trees after a final
   audit, master-plan Phase H step 7 cleanup.

Sessions 1-3 are each independently shippable and none blocks the
ship lanes' current test-failure work. The load-time planner program
(item 3) should start no earlier than after Session 3, when there is
exactly one executor for it to plan for.
