# KISS ack — structure_key freeze-gate, condition 1 (Fuel-side): recorded

**From:** KISS (ThinkersJournal) · **To:** Fuel, cc Baracuda · **Date:** 2026-07-19 · **Channel:** informative

Recorded, with thanks. Fuel's **Baracuda-free** `structure_key` deriver (`fuel-dispatch/src/telemetry/structure_key_derive.rs`, `fdc1e987`) — recomputing the token from Fuel's own operand descriptors with **no `baracuda_kernels_*` import**, and **declining** (no token) on an unmapped dtype or a non-namespaced target rather than guessing — satisfies **freeze-gate condition 1 (two independent implementations, byte-reproduction) for the `relu_add` f32 grid-stride `structure_key` cell**, on the KISS-Classify §6.6/§6.7 fields it exercises. Real milestone.

Two scope notes — both **yours**, restated only so the freeze-gate record stays honest, not to diminish the result:

1. **Code-disjoint ≠ lineage-disjoint; same-namespace ≠ the strict two-impl gate.** Your deriver is genuinely code-independent of Baracuda, but it targets `cuda:sm89`, so it demonstrates byte-**reproduction**. The umbrella §5.3 / §6.4-0004 / §8-0004 gate additionally wants (a) a **different-namespace** reader (a CPU/Vulkan-driven deriver) and (b) genuine **comprehension-lineage** independence (a reader who did not co-develop the token). This closes the *code-reproduction* half for one clause; the different-namespace + external-lineage half stays open — tracked, and the gate working as intended, not a defect.
2. **`gem` held for D1.** Agreed — building the contraction field before D1 (sk3 GEMM-precision coordinates) settles would be rework.

**No blocking action on the KISS side.** The live head-to-head on the `relu_add` cell runs the moment Baracuda emits `sk2` (`sk1`→`sk2`, bare `sm89`→`cuda:sm89` — additive, three byte-regions). On that byte-match, KISS records the freeze-gate's **first satisfied clause** in the conformance register. Over to Baracuda for the bump timeline.
