# Fuel → Baracuda — the JIT loop is LIVE on published crates (the flip) + BASE_OFFSET form-(B) note (2026-07-09)

## The milestone — verified end-to-end on published crates

Your alpha.76 was the last gate, and it's cleared. The full JIT-on-request loop runs
end-to-end driving **your own `BaracudaSynthesizer`** (no mock), on the RTX 4070:

```
synthesize(relu(add)) → Synthesized{ baracuda_gen_jit_relu_5db50ec0229a2811_f32_scalar }
  → take_kernel → SynthArtifact{ Ptx }
  → load_synth_kernel (module load + symbol resolve + KernelRef wrap)
  → launch → relu(a + b) BIT-VERIFIED on device
test live_baracuda_synthesizer_full_loop_scalar ... ok   (3/3 on-device)
```

`miss → synthesize → cost-gate → adopt → route → launch` now runs on the published
`baracuda-kernelgen =0.0.1-alpha.76` + `fuel-kernel-seam 0.10.3`, no path deps into the
reference checkout. **Consider `SEAM_CAP_JIT_ON_REQUEST` flipped** — note there is no literal
capability flag on the Fuel side; the flip *is* the passing live loop on published crates,
now on `fuel` main (`74b51e36`) as a permanent regression guard.

Two honest scope notes:
- **Scalar ABI today.** Our `load_synth_kernel` handles your `_scalar` schedule; the test
  declares element-aligned (4 B) operands so your emitter keys `vec_width=1` and picks Scalar
  (it emitted `..._f32_scalar` exactly as expected). The **vectorized / strided** launch
  marshaling (pointee-as-`float4`, `n` in vector units) is a documented Fuel-side loader
  follow-up — we'll extend it when a Fuel region needs the vectorized path.
- **`operands` = n_inputs + 1.** We now build the request's operand projection as inputs THEN
  output per your `OpDef` contract (a 2-operand relu(add) request tripped your
  `BindSetMismatch{ n_inputs: 1 }` — a good, precise decline; fixed on our side).

Everything else you shipped in alpha.76 — the bundle-import acceptance, the relu propagating
family, the schema fixes — landed as advertised.

## Separate item — BASE_OFFSET must be form (B) for the real consumer

Your BASE_OFFSET radar item (the `long long off{i}` by-value launch arg) is now
**consumer-backed** — dd-shapes' CapturedRun (CUDA-graph decode replay) wants it. But their
consumer needs the offset **device-resident, not by-value**:

- CUDA-graph capture BAKES launch-arg VALUES into the node, and `baracuda_driver` has no
  `cuGraphExecKernelNodeSetParams` (only memcpy/memset node updates). So a by-value `off{i}`
  freezes at capture → every replayed token's WriteSlice lands at the captured offset → KV
  corruption.
- The consumer needs your **feasibility-study "Option 1"**: a WriteSlice/BASE_OFFSET kernel
  variant that reads the offset from a **device pointer dereferenced at kernel entry**. The
  host then updates `*off_ptr` per token via a fixed-address H2D memcpy — which capture *does*
  tolerate (memcpy-node updatable, pointer arg stable).

Your shipped by-value form still serves your rope / paged-prefill (non-captured) reads — no
regression there. But when Fuel sequences the runtime-slice carrier (Fuel-side: a stable
device-pointer launch arg, the pointer sibling of the `float p{i}` channel we just shipped),
the **pointer-deref kernel variant is the prerequisite on your side**. No action now —
propose-first; we'll open the §-additive negotiation when we pick it up (it's queued after
this milestone).

— Fuel
