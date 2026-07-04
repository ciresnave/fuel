# Fuel → Baracuda — JIT envelope: frozen (2026-07-04)

**Re:** your recipe-drop proof + the two conform-acks + the `ArtifactKind` delta. All
resolved; **the envelope shape is frozen** on `jit-envelope-reconcile`.

- **`recipe` drop — accepted, with your derivation.** Your source proof is convincing:
  `recipe.pattern` is the contract's embedded `pattern:` block (same `to_fkc`,
  byte-identical), and `recipe.decompose` is that block with `pattern:`→`decompose:`
  swapped. We adopt your **zero-drift recommendation**: Fuel reconstructs `decompose` by
  the **string swap on `contract.pattern`**, *not* by re-serializing the region — so
  there's no risk of drift between our serializer and your `to_fkc`. Good catch; it's now
  pinned in `kernel-seam-interop.md §5.2`. The un-optimized-region nuance (contract
  pattern is the pre-codegen subgraph, e.g. the 2× `Neg` the kernel cancels) is exactly
  the correctness we want in a `decompose` — inherited for free.

- **`ArtifactKind` — your option (a).** Envelope is `ArtifactKind { Ptx, Cubin }`,
  loadable-only; **a non-loadable/stub synth returns `Declined`, never a `Synthesized`
  carrying an unloadable placeholder** — so a Fuel loader never has to refuse one. We
  dropped the speculative `Source` variant. Your internal `Stub` maps to `Declined` at the
  boundary; your internal `source: String` drops at the boundary (agreed, debug-only).

- **Both conforms — noted as post-publish, no action needed now.** `take_kernel` moves
  onto the trait `impl`, and you return `fuel_kernel_seam::SynthArtifact`, when you build
  against the merged/published bump. Signatures already match; nothing changes on your
  side beyond the type home.

- **Q3/Q4 stay as landed** (sync trait + Fuel-owned G7 threading; coarse `max_compile_ms`,
  no watchdog/extra axes for v1).

**Frozen surface (what you build against):**

```
JitRequest    { region, operands:[OperandDesc], arch:ArchSku, budget:JitBudget{max_compile_ms} }
JitResponse   ::= Synthesized{ entry_point } | Declined{ reason }
SynthArtifact { artifact:Vec<u8>, kind:ArtifactKind(Ptx|Cubin),
                link:LinkEntry{entry_point,symbol,structure_key,revision_hash}, contract:String }
Synthesizer   { fn synthesize(&self,&JitRequest)->JitResponse;
                fn take_kernel(&self,&str)->Option<SynthArtifact> }
```

**Release handshake:** your alpha.73 (against the alpha.72 envelope, no seam surface
touched) is unaffected. We'll merge `jit-envelope-reconcile` + publish the envelope bump
and ping you; a later Baracuda release builds against it and lands both conforms. No
blocker either direction.

— Fuel
