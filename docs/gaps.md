# Fuel gap registry

**Every deferred, incomplete, or declined path in Fuel lives here — one greppable list, not scattered across 40 files.** This exists because the JIT launcher's scalar-only limitation was acknowledged in a code comment ("documented loader follow-up") but never on any schedule, so it rotted for months and shipped an all-zero output. A gap acknowledged in prose but not tracked is a leak.

## Rules

1. **Every** `todo!()` / `unimplemented!()` / `panic!(` on a non-test path, and every prose hedge ("for now", "not yet", "shortcut", "bails", "would need"), carries a `// GAP(GAP-NNN)` reference to a row below.
2. Adding such a path **without** a `GAP(...)` ref is a defect (a CI check, pending, will enforce this — see GAP-141).
3. Close a gap → strike its row (`~~GAP-NNN~~ CLOSED <commit>`) and remove the `GAP(...)` ref from the code in the same change.
4. Owners: **A** = launcher / KISS-alignment / FKC-dispatch; **B** = cuda-backend / concurrency / gate; **C** = decode / models / fuel-core; **—** = pooled / unassigned. (Owners are roles, not sessions; claim by editing the row.)

## Tiers (completion order)

- **A — Correctness:** panics on production paths, silently-wrong results, missing gradients, races. Do these first; several violate the "never panic on production paths" hard rule.
- **B — Roadmap commitments / frontier:** architectural work other things depend on.
- **C — Feature completeness:** missing capability that currently **declines cleanly** (typed error / CPU fallback) — honest, lower-stakes.
- **P — Performance:** optimization only; no correctness stake.

---

## Tier A — Correctness (do first)

| ID | File:Line | Owner | Gap | Status |
|----|-----------|-------|-----|--------|
| GAP-014 | fuel-core/src/lazy.rs:9531-9570 (`rebind_and_realize_prebuilt`) | C | **SILENT cross-request KV contamination.** A held decode plan is welded to the `KvCache` it was built against; the validity key has ZERO `kv_nodes` references, so it cannot see a cache swap. Slot-pooled server happy path: retire request A, admit B with a fresh same-shaped cache → B decodes over A's KV, full speed, no error. Blocks multi-session serving (stated near-term priority). 4th instance of the `docs/architecture/14-lifecycle` Stage-5 invariant @ `ff8f61a9`; found by decode-owner. **Ranked #2 in Tier A** (silent wrong answer on the happy path). | OPEN |
| GAP-001 | fuel-dispatch/src/jit_cuda_load.rs:51 | A | Launcher marshals only the `_scalar` ABI → all-zero output ("all zeros, no error"). **Root-cause sub-form UNRESOLVED** (2026-08-07): the "declines `_co_v{w}` vectorized kernels" framing does NOT explain the peer bisect (kernelgen .76/.77 PASS, .78 FAIL on the *scalar* test, byte-identical symbol) — that test runs the scalar variant Fuel *does* marshal, so it is likelier a scalar launch-contract change .77→.78 that Fuel's .77-era marshaling mishandles. Byte-identical symbol rules out identity, NOT body-vs-contract. **First diagnostic:** dump Fuel's computed launch params vs the .78 header/metadata-declared BEFORE running — agree ⇒ body, disagree ⇒ contract. Fix: marshal per FKC `count_unit` (scalar + `_co_v{w}`), assert `n>0`. cuda-build-blocked here. | IN PROGRESS |
| GAP-002 | fuel-ir/src/scalar.rs:38,58,85 | A | `Scalar::zero/one/from_f64` `panic!` on sub-byte dtypes (F4/F6/F8E8M0/F8E6M2). Never-panic violation. Design "scales real, packed honest" — spec+plan committed on `feat/scalar-dtype-completion` (Result ctors; real F8E8M0/F8E6M2 scale variants; `Err` for packed F4/F6*; build-time MaskedFill guard). | IN PROGRESS |
| GAP-003 | fuel-graph/src/lib.rs:~2256 | B | `Tensor::from_*` uses `.expect()` on a production path (CLAUDE.md standing never-panic violation). Return `Result`. | OPEN |
| GAP-004 | fuel-graph/src/lib.rs:296-308 | B | In-place Relu/Silu/Gelu/Sigmoid "not yet wired through autograd" → silently wrong gradients. | OPEN |
| GAP-005 | fuel-graph/src/lib.rs:10105 | B | A gradient is "non-trivial and not yet implemented". | OPEN |
| GAP-006 | fuel-core/src/lazy.rs:6902 | C | Mixed (activation-dtype, F32 weight) matmul NOT supported (BF16-decode adjacent). | OPEN |
| GAP-007 | fuel-cuda-backend/src/storage.rs:3534 | B | Q4_0 matvec only `m=1` (decode); prefill `m>1` bails. | OPEN |
| GAP-008 | fuel-cpu-backend/src/mkl.rs:188-398 | — | MKL `vs_*/vd_*` wrappers `panic!` on length mismatch (~15 sibling sites). Never-panic. | OPEN |
| GAP-009 | fuel-cpu-backend/src/system_memory.rs:198 | — | `panic!` on an inconsistent OS memory snapshot. Never-panic. | OPEN |
| GAP-010 | fuel-quantized/src/k_quants.rs:40 | — | `from_float_imatrix` default trait method `panic!`s "unimplemented" for most dtypes. Never-panic. | OPEN |
| GAP-011 | fuel-core/src/judge/mod.rs:61 | C | `build_input_graph` "would panic!" for un-profiled ops (currently guarded by a catch-all arm; lower risk). | OPEN |
| GAP-012 | fuel-cuda-backend/src/probe.rs | B | `probe.rs` had NO memoization while the Vulkan twin has had it since `9bb68e6b`: re-entering the driver on every `enumerate_devices()` is real waste. FIXED with an `OnceLock` + non-vacuous test (counter in the *uncached* fn). **NOT the stall cause** — the suite still hangs with the fix compiled in (positive-controlled). (This row previously said "32 tests still hang"; that figure came from GAP-015's original symptom text and is measured-wrong — see GAP-015.) | FIXED (waste only) |
| GAP-015 | `fuel-dispatch/src/dispatch.rs:6554` (`global_bindings`) ×; `GLOBAL_REGISTRY` | B | **COLD-START LOCK DEADLOCK on a PRODUCTION path — reclassified 2026-08-07, was filed as a test-harness hang.** Reproduced + classified by the owner: a true user-space lock deadlock, not contention (CPU **flat** 25.38s→25.39s over 135s; 45 threads all `ThreadState=5`, **44 on `WaitReason=37` `WrAlertByThreadId`** = the SRWLOCK/condvar/`OnceLock` primitive; 45th is the main thread joining). **Deterministic with a concurrency threshold:** passes `-j2/4/8/16`, hangs `-j24/32`, and at both hanging values **exactly the same 343 of 760 tests complete — byte-identical sets** (a race would jitter). **Two-party:** removing either `dispatch::` or `optimize::` clears it (each against a still-hanging control); keep-based bisect on `dispatch::` converged to 7 `global_bindings_registers_*_family_from_contract` tests, and the culprit *spans* the final split ⇒ several concurrent callers, not one test. **Suspect:** `global_bindings()` is a `OnceLock::get_or_init` whose initializer calls `register_derived_gpu_caps`, taking a **write** lock on the *separate* `GLOBAL_REGISTRY`; the in-code comment at `:6607-6608` ("separate lock … no re-entrancy") rules out self-deadlock but **not lock ordering** — cross-lock work inside a `OnceLock` initializer is where ABBA inversions live. **Refuted, not assumed:** `SystemTopology::build` does NOT hold a registry guard across the bindings call (`topology.rs:419-427` scopes and drops before `:433`). **Why this is Tier A and not a gate nuisance (architect, code-grounded 2026-08-07):** `global_bindings()` is a **production** entry point, not test-only — `fuel-core/src/pipelined_bridge.rs:795,877` inside `build_optimized_graph` (the optimizer path) and `judge/mod.rs:1233,1255,1274` and `cast_fusion.rs:51` all call it, and every one is above its file's `#[cfg(test)]` boundary (1949 / 2252 / 164). Its own doc comment names "production callers picking up the global table". So the test suite at `-j24/32` is not the defect — it is the **first workload that fans out enough concurrent cold-start callers to expose it**, and concurrent cold start is precisely the multi-session serving profile (stated near-term priority). Treat as a latent production deadlock. **Symptom text corrected:** the old "32 tests stall (18 `optimize::*` + 14 `pipelined::*`)" is measured-wrong and understates it — 343/760 complete and 401 never report, but most of those **never started** (all workers parked; only ~44 can block at once). `ranker::` alone contributes 170 never-started, and those are pure-CPU. The suite stops dead. Interim gate: `--test-threads=1`. **Next:** code-read `register_derived_gpu_caps`'s lock usage to pin the opposing order (no GPU needed). | **IN PROGRESS** — owner xe3ch8hr (fuel-persistent-default); reproduced + classified, not fixed |
| GAP-013 | fuel-core placement (candidate-scope 42-fix) | A | VERIFY the shipped candidate-scope fix (`5358f596`) addresses the original 42 `Op::Copy on Cuda: no CUDA storage` **ERRORS** at the right layer: revert + re-run. NOTE: DECOUPLED from GAP-015 — the 42 were an error (suite ran to report), GAP-015 is a hang; likely separate. Do NOT treat the probe fix as the operative variable. | OPEN |

---

## Tier B — Roadmap commitments / frontier

| ID | File / area | Owner | Gap | Status |
|----|-------------|-------|-----|--------|
| GAP-020 | fuel-ir/dtype + fuel-dispatch/fkc/lower + structure_key | A | **sk4 token regen** — regenerate FKC/structure-key tokens + invalidate Judge caches once the six-way KISS cosign lands (dtype-spelling rename + MX + acc/mp coordinate). | BLOCKED on cosign |
| GAP-021 | fuel-core/src/lazy.rs (selective_scan) | — | **selective_scan / G3** basis gap: needs a higher-order `Op::Scan` (CumSum closed-form overflows for `a<0`). Keystone for non-transformer paradigms. | OPEN |
| GAP-022 | fuel-graph registry (flash_attn symbolic) | C | Symbolic (`Sym`) `k_len` flash-attn decode: no `DynScalar`-length `Slice`; registry-layer gap. | OPEN |
| GAP-023 | fuel-cpu-kernels/src/philox.rs + Op enum | A | **RNG generator seam** increments 2-4 (`Op::RandomBits`; inc 1 shipped; Op enum gated on a KISS RFC). | OPEN |
| GAP-024 | QMatMul / Nf4Matmul / FusedLinear | A | **SType-unification** (workstream E): collapse to MatMul-over-encoded-storage + bias epilogue (scale as sibling operand). KISS alignment. | OPEN |
| GAP-025 | ~24 in-place op variants (WriteSlice/Rotating/Doff…) | A | **In-place-variant collapse** → base-op + in-place-eligibility attribute. KISS alignment. | OPEN |
| GAP-026 | fuel-ir/src/dispatch.rs:191 | B | Fused 3-output FlashAttn-backward needs multi-output infra that "doesn't exist yet". | OPEN |
| GAP-027 | fuel-core AR loops (multi-output nodes) | C | Autoregressive loops (lazy_csm, lazy_lfm2) blocked on multi-output graph nodes. | OPEN |

---

## Tier C — Feature completeness (declines cleanly)

### fuel-ir
| ID | File:Line | Owner | Gap |
|----|-----------|-------|-----|
| GAP-040 | fuel-ir/src/dummy_dtype.rs:4 | — | safetensors-spec dtypes defined but "not yet fully implemented". |
| GAP-041 | fuel-ir/src/stype.rs:47 | A | NF4 has no distinct FDX code in v1 (deferred). |
| GAP-042 | fuel-ir/src/stype.rs:61 | A | SType is shape-only, "NOT wired in v1". |
| GAP-043 | fuel-ir/src/probe.rs:72 | — | fuel-metal-backend "not yet probe-wired". |
| GAP-044 | fuel-ir/src/capability.rs:40 | B | cross-device copy: concrete backends bail (Router-only). |

### fuel-graph
| ID | File:Line | Owner | Gap |
|----|-----------|-------|-----|
| GAP-050 | fuel-graph/src/lib.rs:525 | B | Pad Reflect/Replicate modes return "not yet implemented". |
| GAP-051 | fuel-graph/src/lib.rs:7006 | B | backward through fused `rms_norm_last_dim` not implemented. |
| GAP-052 | fuel-graph/src/lib.rs:7623 | B | Step/Where/Sign op "not yet in the catalog". |
| GAP-053 | fuel-graph/src/registry/flash_attn.rs:34 | B | FlashAttn backward is a panic stub (NotDifferentiable). |
| GAP-054 | fuel-graph/src/registry/conv_transpose_2d.rs:71 | B | backward "for now NotDifferentiable". |
| GAP-055 | fuel-graph/src/registry/conv_transpose_2d.rs:414 | B | with-bias conv_transpose form "not yet in the recipe". |
| GAP-056 | fuel-graph/src/registry.rs:695 | A | payload recovery via `extract:` layer is a follow-up. |
| GAP-057 | fuel-graph/src/opt.rs:1278,1282 | A | forward-placement follow-up + fuse-two-passes (Phase 4). |

### fuel-dispatch
| ID | File:Line | Owner | Gap |
|----|-----------|-------|-----|
| GAP-060 | fuel-dispatch/src/jit_cuda_load.rs:98 | A | only PTX loadable; Cubin loading "would need" more. |
| GAP-061 | fuel-dispatch/src/baracuda_dispatch.rs:413 | A | Bias/BiasRelu/BiasGelu/BiasSilu epilogues "not yet wired". |
| GAP-062 | fuel-dispatch/src/baracuda_dispatch.rs:2108 | A | decode attention: window/softcap/alibi "not implemented". |
| GAP-063 | fuel-dispatch/src/baracuda_dispatch.rs:1719 | A | returns Unsupported for pad tags other than the supported set. |
| GAP-064 | fuel-dispatch/src/vulkan_dispatch.rs:3515 | A | Vulkan quant matmul Q4_1/Q5_*/Q2K/Q3K/Q5K/Q6K/Q8_1 "not yet wired" → CPU. |
| GAP-065 | fuel-dispatch/src/vulkan_dispatch.rs:2543 | A | Vulkan MeanReduce deferred (no scalar-divide pass). |
| GAP-066 | fuel-dispatch/src/vulkan_dispatch.rs:2592 | A | non-f32 reduces only last-dim; other dims → CPU. |
| GAP-067 | fuel-dispatch/src/vulkan_dispatch.rs:2049 | A | Vulkan float add "not yet wired" → CPU. |
| GAP-068 | fuel-dispatch/src/vulkan_dispatch.rs:3948 | A | Vulkan conv2d "doesn't fuse bias yet". |
| GAP-069 | fuel-dispatch/src/vulkan_dispatch.rs:3783 | A | Vulkan flash-attn dQ bails on window/softcap. |
| GAP-070 | fuel-dispatch/src/pipelined.rs:9277 | A | kernel "doesn't yet handle negative strides". |
| GAP-071 | fuel-dispatch/src/fkc/mod.rs:31 | A | cost trampoline / V-FKC-* validators / CI lint "NOT yet implemented". |
| GAP-072 | fuel-dispatch/src/fkc/verify/harness.rs:47 | A | real cross-backend numeric check is a follow-on. |
| GAP-073 | fuel-dispatch/src/ranker/cost.rs:26 | A | picker "doesn't yet read live BackendCapabilities". |
| GAP-074 | fuel-dispatch/src/fkc/register.rs:120 | A | imported costs "not yet wired into a CostFn" (Judge bootstraps for now). |

### fuel-core (models)
| ID | File:Line | Owner | Gap |
|----|-----------|-------|-----|
| GAP-080 | fuel-core/src/judge/mod.rs:845 | C | many op families "not yet wired into the direct-call path". |
| GAP-081 | fuel-core/src/lazy_sam.rs:555 | C | `get_rel_pos` rel-pos interpolation "not yet supported". |
| GAP-082 | fuel-core/src/lazy_sam.rs:66 | C | ViT-L/H, bias-free variants, prompt-encoder/mask-decoder/TinyViT deferred. |
| GAP-083 | fuel-core/src/lazy_mmdit.rs:825 | C | MMDiT-X joint-block variant (SD3.5-medium) "not yet implemented" in v1. |
| GAP-084 | fuel-core/src/lazy_phi.rs:66 | C | `qk_layernorm=true` unsupported; bails at load time. |
| GAP-085 | fuel-core/src/lazy_granitemoehybrid.rs:7 | C | Mamba branch bails; attention-only layers implemented. |
| GAP-086 | fuel-core/src/lazy_recurrent_gemma.rs:38 | C | only prefill-from-zero-state supported (no state reset). |
| GAP-087 | fuel-core/src/lazy_mimi_resampler.rs:16 | C | `learnt=false` static resample not impl; streaming `step` unsupported. |
| GAP-088 | fuel-core/src/lazy_mimi_conv_wrappers.rs:16 | C | only `learnt=true` mode supported. |
| GAP-089 | fuel-core/src/lazy_nn_one_hot.rs:21 | C | negative-sentinel ("-1 ⇒ all-off") rows not supported. |
| GAP-090 | fuel-core/src/lazy_quantized_whisper.rs:25 | C | GGUF Whisper constructor errors, points to `from_f32_bake`. |
| GAP-091 | fuel-core/src/lazy_nn/conv.rs:10, lazy_dac.rs:14 | C | LazyTensor conv primitives don't yet accept dilation. |
| GAP-092 | fuel-core/src/lazy_efficientnet.rs:69 | C | lazy conv2d only supports symmetric padding. |
| GAP-093 | fuel-core/src/lazy_wuerstchen.rs:764 | C | conv_transpose2d "does not yet plumb the bias". |
| GAP-094 | fuel-core/src/lazy_sd_samplers_unipc.rs:109 | C | stochastic `SdeDpmSolverPlusPlus` is a TODO. |

### fuel-cpu-backend
| ID | File:Line | Owner | Gap |
|----|-----------|-------|-----|
| GAP-100 | fuel-cpu-backend/src/host_storage/mmap.rs:32 | — | bool/enum/ref dtypes "not supported" for mmap. |
| GAP-101 | fuel-cpu-backend/src/conv2d.rs:41 | — | "other cases" not handled. |

### fuel-cuda-backend
| ID | File:Line | Owner | Gap |
|----|-----------|-------|-----|
| GAP-105 | fuel-cuda-backend/src/flash_attn.rs:30 | B | FA2 `launch` module staged, "not yet wired" into `Op::FlashAttn` (dead code). |
| GAP-106 | fuel-cuda-backend/src/flash_attn.rs:141 | B | head_dim 40/80 not in FA2 supported set → fallback path. |
| GAP-107 | fuel-cuda-backend/src/baracuda/attention.rs:1296 | B | `rope_apply` NOT wired as CUDA `FusedOps::ROPE` (staged). |
| GAP-108 | fuel-cuda-backend/src/capture.rs:9 | B | CUDA-graph capture a capability, "not yet a wired-in optimization". |
| GAP-109 | fuel-cuda-backend/src/device.rs:601,647 | B | F16/BF16 `rand_uniform`/`normal` (needs upstream cudarc). |
| GAP-110 | fuel-cuda-backend/src/ug.rs:60 | B | "support more dtypes". |

### fuel-vulkan-backend
| ID | File:Line | Owner | Gap |
|----|-----------|-------|-----|
| GAP-115 | fuel-vulkan-backend/src/lib.rs:9907,9975 | — | custom weight pool NOT integrated with weight allocation; no defrag/rebind machinery. |
| GAP-116 | fuel-vulkan-backend/src/lib.rs:4804 | — | flash-attn naive; bails on `Sk>4096`/`D>256`. |
| GAP-117 | fuel-vulkan-backend/src/lib.rs:111 | — | weights "not yet emitted" to VRAM before compute ops. |
| GAP-118 | fuel-vulkan-backend/src/capture.rs:27 | — | graph capture a capability, "not a wired-in" optimization. |

### fuel-metal-backend / kernels (macOS; unbuildable here)
| ID | File:Line | Owner | Gap |
|----|-----------|-------|-----|
| GAP-120 | fuel-metal-backend/src/storage.rs:142 | — | per-dtype "not implemented" bails across affine/powf/elu/reduce/to_dtype/unary/where_cond/conv1d/col2im1d (~20 sites). |
| GAP-121 | fuel-metal-backend/src/ug.rs:62 | — | "support more dtypes". |
| GAP-122 | fuel-metal-kernels/src/kernels/sdpa.rs:19 | — | non-bf16 final type: template the kernel. |

### fuel-quantized / fuel-memory / fuel-onnx / fuel-formats
| ID | File:Line | Owner | Gap |
|----|-----------|-------|-----|
| GAP-125 | fuel-quantized/src/k_quants.rs:760 | — | `to_float` `unimplemented!("no support for vec-dot on Q8_1")`. |
| GAP-126 | fuel-memory/src/dlpack_view.rs:233 | A | GPU device-pointer extraction returns typed error ("later comm-layer slice"). |
| GAP-127 | fuel-memory/src/dlpack_view.rs:547 | A | gather sidecar (`FDX_FLAG_HAS_GATHER`) deferred — needs consuming-op geometry. |
| GAP-128 | fuel-onnx/src/lazy_eval.rs:501 | — | unsupported ONNX ops error out; sub-port-1 catalog limited. |
| GAP-129 | fuel-onnx/src/lazy_eval_conv.rs:105-782 | — | Conv/ConvTranspose/Pad/Pool: dilations≠1, negative/asymmetric pads, ceil_mode, SAME_* pooling, indices output unsupported (~15 sites). |
| GAP-130 | fuel-onnx/src/lazy_eval_ops.rs:349 | — | CumSum exclusive/reverse, ArgMax `select_last_index`, Slice `step≠1` unsupported. |
| GAP-131 | fuel-onnx/src/lazy_eval_norm.rs:89 | — | BatchNorm `training_mode=1` + scalar-input Softmax unsupported. |
| GAP-132 | fuel-formats/src/pickle.rs:7,384 | — | partial protocol-2 pickle parser; hard-coded around `_rebuild_tensor_v2`; ordered/default dict. |

---

## Tier P — Performance (no correctness stake)

| ID | File:Line | Owner | Gap |
|----|-----------|-------|-----|
| GAP-135 | fuel-ir/src/dispatch.rs:179; fuel-cpu-backend/src/byte_kernels.rs:6685,7481 | — | FlashAttn / PagedAttn are naive (non-tiled) reference impls on CPU. |
| GAP-136 | fuel-dispatch/src/vulkan_dispatch.rs:3576 | — | Vulkan flash-attn naive single-pass (not tiled). |
| GAP-137 | fuel-dispatch/src/pipelined.rs:7092 | — | "V.1.B stopgap": D2H→CPU-contiguize→H2D round-trip. |
| GAP-138 | fuel-cuda-backend/src/baracuda/scratch.rs:11 | — | per-stream scratch pool "an obvious optimization but not yet". |
| GAP-139 | fuel-cpu-backend/src/ops.rs (7×), utils.rs (2×) | — | specialized kernels / avoid redundant copies / double layout traversal. |
| GAP-140 | fuel-quantized/src/k_quants.rs:2303; neon.rs:577; fuel-metal-kernels/src/metal/commands.rs:265 | — | pre-allocate; NEON dotprod; avoid redundant metal alloc. |

---

## Meta

| ID | Owner | Gap | Status |
|----|-------|-----|--------|
| GAP-141 | A | CI check: fail on any `todo!`/`unimplemented!`/`panic!`/prose-hedge on a non-test path lacking a `GAP(...)` ref. The enforcement that keeps this registry honest. | OPEN |
