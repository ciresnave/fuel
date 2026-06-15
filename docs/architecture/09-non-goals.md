# Non-goals

**Status**: v0.2 (2026-06-14). Reconciled to the "plan is the graph" redirection ([10-decisions-log](10-decisions-log.md)): the e-graph non-goal is narrowed to the per-realize hot path (offline `optimize_graph` path-search is in-bounds), and the bundled-cache non-goal is distinguished from the in-bounds bundled Judge baseline.

What fuel deliberately doesn't try to be. Each rejection is a real architectural decision — not a "we didn't get to it yet" but a "we examined this direction and chose against it because of how it would change fuel's center of gravity."

This section is the negative-space companion to [01-identity](01-identity.md). The identity says what fuel is; this section sharpens that by pinning what it isn't. Together they bound the architecture's scope.

---

## Not eager-first

Eager execution mode (immediate evaluation as ops are constructed) hides the DAG and prevents every optimization the rest of this architecture commits to. Fuel's only execution path is lazy + explicit `.realize()`.

The performance target is **external, not internal**. Because fuel picks the best available implementation of every op and re-adapts to the live state of the visible devices — things eager code largely cannot do — lazy-realize should keep up with or outperform *every* eager ML framework, not merely fuel's own retired eager path. The first concrete yardstick is Candle (fuel's fork parent: near-unchanged eager Rust, near-zero porting cost): **lazy-realize fast enough that Candle's eager execution looks slow by comparison.** Beating Candle is the floor, not the goal — the same comparison is owed against llama.cpp, PyTorch, ONNX Runtime, Burn/CubeCL, and tch-rs on the parts each does well. This target lost its in-repo comparator when the hybrid eager path was retired in Phase 7.5, so it is enforced by an **out-of-repo benchmark suite** (see ROADMAP §Benchmarking) rather than "eager is forbidden because we say so." Before any non-alpha release the claim must be demonstrated with measurements against the popular frameworks and inference engines — proof, not belief.

What this means concretely:

- `Tensor::matmul()` returns a graph node, not a computed result.
- Materialization happens at explicit calls: `.realize()`, `.materialize()`, `.item()`, similar.
- Print-debug, dynamic control flow on tensor values, and interop with non-fuel code all need an explicit materialization call. JAX has demonstrated this idiom is learnable at scale.

Pre-Phase-7.5 fuel had a hybrid eager+lazy model. Phase 7.5 retires eager. The architecture commits to lazy-only.

## Not backend-internal fusion

Backends don't choose what to fuse. The optimizer does, with full visibility into every backend's fused-kernel catalog. Backend-internal fusion would be invisible to:

- **Cross-backend placement decisions** (the optimizer can't compare "fused-on-CUDA vs unfused-on-Vulkan" if Vulkan's fusion happens internally).
- **Algebraic-equivalence rewrites** (subgraph patterns that span ops the backend would never combine).
- **Tolerance-budget reasoning** (whether a fused kernel is admissible under a budget depends on its error contribution, which has to be visible to the optimizer).

The architecture commits to the FusedOpRegistry being the single source of truth for fused ops. Any backend that internally fuses kernels is in violation of [05-backend-contract](05-backend-contract.md).

## Not framework-agnostic dispatch

Fuel doesn't try to be a generic DAG executor that works with arbitrary user-defined ops at runtime. Specifically:

- The `Op` enum is closed (primitive variants are exhaustively defined; the `Op::Fused` arm delegates to a registry frozen at startup).
- The fused-op registry is populated at backend init and immutable thereafter.
- New op kinds require recompiling fuel.

The benefit fuel gives up: hot-loading new ops at runtime (TVM-style, ONNX-runtime-style). The benefit fuel gets: a closed, analyzable, statically-checked op vocabulary the optimizer can reason about exhaustively.

If a downstream consumer needs runtime-extensible ops, fuel may not be the right framework. The architecture optimizes for the ML-deployment use case, not the ML-research-experimentation use case.

## Not multi-dialect IR (MLIR-style)

Fuel's IR is two layers: primitive `Op` variants + the `Op::Fused(id, params)` arm. It's not a multi-level dialect framework where users define new dialects, lower between them, and apply per-dialect optimization passes.

Two layers cover fuel's needs:

- **Primitives** for the canonical decomposition the optimizer reasons over.
- **Fused ops** for the registered higher-level abstractions the optimizer collapses primitive subgraphs back into.

MLIR's expressiveness — multiple intermediate dialects, custom dialect interactions, dialect-specific verifiers — is overkill for fuel's scope. The cost of MLIR-style flexibility is conceptual surface area; fuel commits to two layers and moves on.

## Not a Python interop layer

Fuel is Rust-native. The user-facing API, the IR, the optimizer, the backends — all Rust. Python interop, if it ever happens:

- Lives in a separate, leaf-layer crate (e.g., `fuel-py`).
- Uses pyo3 or similar at the boundary.
- Doesn't shape the architecture below it.

The architecture documents in this set don't anticipate Python. If `fuel-py` ships, it'll wrap fuel's Rust API; the wrapping happens at the orchestration layer (per [02-layers](02-layers.md)), not at the Foundation layer.

## Not e-graph saturation *on the per-realize hot path*

The boundary here moved with the 2026-06-14 "plan is the graph" redirection ([10-decisions-log](10-decisions-log.md)), so state it precisely. What is **out**: e-graph saturation (egg-style equality saturation) **on the per-realize hot path** — exponential in pathological cases, building redundant representations a hot-path optimizer can't afford.

What is **in-bounds** (and was sharpened by the redirection):

- **`optimize_graph` is offline multi-path path-search, which is e-graph-*adjacent***: it explores alternative paths (algebraic rewrites, fusions, placements) and keeps a bounded Pareto frontier. This is fine because it runs **at load/import, not per realize**, and is bounded by construction (per-device Pareto + crowding cap, [04-optimization](04-optimization.md)). It may legitimately use e-graph techniques internally.
- **E-graphs as an offline rule-discovery tool** — fed harvested workload data ([08-pattern-harvest](08-pattern-harvest.md)) to find new algebraic equivalences and surface them as suggested OptimizationMap rules.

So the rejection is narrow: not "no equality-saturation-style search anywhere" (the offline optimizer is exactly that), but "no e-graph saturation in the per-realize dispatch loop." The runtime picks among already-pruned paths; it does not saturate.

## Not autotuning-search-style optimization

TVM-style autotuner search (try thousands of kernel configurations, pick the empirically best) produces excellent results but is operationally heavy:

- Tuning runs take hours to days per model+hardware combination.
- Tuning data is hard to share across users (very hardware-specific).
- The framework needs to manage tuning data lifecycles (when does a tuning result expire? How do you re-tune incrementally?).

Fuel's optimizer is heuristic + cost-driven, not search-driven. The empirical Judge fills the gap autotuning would otherwise fill — it measures actual per-(op, dtype, size_class, backend, device) latency and feeds the cost model. The Judge is incremental, lighter-weight, and produces shareable profile data.

If a use case really needs full autotuning, fuel's persistence layer (per [11-persistence](11-persistence.md)) provides the substrate for it: an external tuner could populate the cache with extreme-effort plans. But it's not in fuel's box.

## Not user-installable optimization rules at runtime

The OptimizationMap is populated at startup, frozen thereafter. Runtime rule extension would let users hot-load optimizations but introduces:

- Security surface (untrusted code in the optimizer).
- Stability surface (a rule that misbehaves can corrupt the optimization output).
- Debugging surface (every optimization failure now requires "did a user-rule cause this?").

Users who want custom optimizations contribute them via the open-source rule library, with code review and community testing. The architecture supports custom rules at compile time; not at runtime.

## Not global-optimization-passes-that-aren't-rule-based

Every transformation goes through the rule machinery. "Special pass that's not a rule" is forbidden:

- Makes the optimizer's behavior unanalyzable (you can't reason about all optimizations from a single registry).
- Makes commits unreproducible (special passes have side effects the rule registry doesn't track).
- Makes auto-generation impossible (auto-generated rules from FusedOpEntry decomposition + pattern is a real load-bearing capability).

If a transformation is needed, it's expressed as a rule (declarative or callable). The rule registry is the single optimization surface.

## Not a silent `fast_math` flag

Tolerance is hierarchical and explicit, never a global mode that silently changes results everywhere. There is no flag that makes a strict graph behave loosely; tolerance is set per-graph, per-subgraph, per-op, or per-call (see [07-tolerance §Hierarchical specification](07-tolerance.md#hierarchical-specification)). The architecture is committed to this rejection — making tolerance silent would invalidate the optimizer's correctness reasoning.

## Not automatic tolerance learning during inference

The optimizer doesn't decide what tolerance is acceptable for a user's use case; the user does (manually or via opt-in calibration per [07-tolerance §Tolerance discovery and calibration](07-tolerance.md#tolerance-discovery-and-calibration)). The optimizer reasons under whatever budget it's given; it doesn't try to infer "this user probably won't notice" from observed inputs.

This is a deliberate restraint. Auto-inferred tolerance would require the framework to model "what the user notices," which is use-case specific in ways no framework can capture.

## Not mandatory telemetry

Fuel never harvests without explicit opt-in. Production deployments that don't opt in are first-class supported. Pattern harvest, tolerance-recipe sharing, hardware-fingerprint reporting, and kernel-stat-summary sharing are contributor benefits, not taxes. The architecture commits to honoring this — privacy commitments are part of [08-pattern-harvest](08-pattern-harvest.md) and apply equally to any future telemetry feature.

## Not opt-out telemetry

The first-use prompt (per [08-pattern-harvest §How harvest is enabled](08-pattern-harvest.md#how-harvest-is-enabled)) is the primary opt-in mechanism. It is *opt-in*, not opt-out — the prompt asks the user to choose; no data is collected unless the user explicitly enables. Headless environments where prompting isn't possible default to disabled.

This is a deliberate rejection of the "telemetry on by default; users disable explicitly" pattern that several other open-source projects use. Reasons:

- **Legal exposure (GDPR, CCPA, similar):** opt-out telemetry has substantive consent-requirement risk in regulated jurisdictions; opt-in does not.
- **Trust:** telemetry-by-default has triggered backlash in many open-source ecosystems (VS Code, Homebrew, NPM controversies). Going from opt-in to opt-out is a one-way door for community trust.
- **Architectural alignment:** silent telemetry contradicts the "explicit, user-controlled" posture the rest of fuel commits to (see [01-identity §How this identity is enforced](01-identity.md#how-this-identity-is-enforced)).
- **The data gain is marginal:** the most valuable contributors (production deployments) typically block outbound telemetry regardless of default; opt-out catches mostly less-valuable contributors at the cost of legal/trust risk.

The first-use prompt captures the "people who don't care" segment that opt-out would target without the downsides — industry precedent (Homebrew analytics prompt, rustup installer telemetry question, mise first-run config) shows prompt-based opt-in achieves meaningful contribution rates without the privacy/trust costs.

## Not bundled distribution of compiled-for-specific-hardware artifacts

The optimization cache (per [11-persistence](11-persistence.md)) is hardware-fingerprint-keyed. Distributing `model.safetensors + model.fuel-cache` works only when the recipient's hardware matches the producer's. This is fine; it's the explicit semantics of the cache. What fuel doesn't try to do:

- Auto-distribute caches keyed by some abstracted "hardware class" that aliases incompatible hardware together.
- Translate a CUDA-cached plan to an equivalent Vulkan-cached plan automatically.
- Maintain a fleet-of-fingerprints cache that ships compiled artifacts for many target environments in one bundle.

Each of these would require either dropping the strict-fingerprint-match invariant (risky) or building substantial cross-target translation infrastructure (out of scope). Production deployments that need hardware diversity build per-target caches separately.

This non-goal is about the *hardware-keyed optimization cache* — **not** the bundled Judge baseline that ships in-package ([06-runtime](06-runtime.md), [10-decisions-log](10-decisions-log.md) 2026-06-13). The baseline is workload-agnostic statistical *priors* the local Judge falls back to before any local measurement exists, not a compiled-for-specific-hardware plan. It degrades gracefully on mismatched hardware (a wrong prior is corrected by the first local measurement), whereas a fingerprint-mismatched cache would be silently wrong. Shipping priors is in-bounds; shipping locked plans is what this rejects.

## Not training-orchestration-flavored architecture decisions

Fuel-the-Foundation supports both inference and training (autograd is a graph-rewrite over the forward IR per Phase 7.5; both forward and backward are first-class). But the architecture's center of gravity is *inference* — competitive-edge claims, default tolerance models, persistence semantics, all assume inference-flavored workloads.

Training-specific concerns (checkpointing strategies, gradient accumulation policy, mixed-precision training recipes, distributed training) belong at the orchestration layer (`fuel-training`), not at Foundation. The architecture doesn't reject training; it just doesn't shape Foundation around training-specific concerns.

## What this section is

A list of choices fuel made *against*, with reasons. It exists so that future contributors who think "shouldn't fuel also do X?" can find the answer (or its absence — if a non-goal isn't here, it might be reconsidered).

Future contributors: if you encounter a real consumer pressure that one of these non-goals blocks, raise it explicitly. The architecture is iterable; non-goals can be re-examined when there's data. But until there's data, these stand.

---

## See also

- [01-identity](01-identity.md) — what fuel *is*; this section is the complement.
- [04-optimization](04-optimization.md) — where many of these non-goals are operationally enforced.
- [05-backend-contract](05-backend-contract.md) — backend-side non-features (no internal fusion, no internal placement).
- [07-tolerance](07-tolerance.md) — the tolerance non-features in detail.
- [10-decisions-log](10-decisions-log.md) — when a non-goal is reconsidered, the decision is recorded there.
