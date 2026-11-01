# Seams, cleanup and splitting: the order of work

ROADMAP rows 136 to 139. Written 2026-11-01 from a design discussion (CireSnave's questions and the
code touched during that day's work). It records a decision about ORDER and a method; it does not
decide any split.

## The rule that orders everything

1. **Cleaning up the code comes before we know for sure where the seams are.** A seam drawn across
   code that leaks (a generic layer that knows about specific backends, a crate that does two jobs)
   will be drawn in the wrong place, or will be wide because of the mess and not because of the
   problem. Fix the structure near each candidate seam first.
2. **Defining the seams comes before splitting anything.** A split moves code into its own project;
   without a defined, tested interface that only relocates the mess and adds a version map.
3. **Only then split**, one seam at a time.

The rows below are in that order, and each row names what it depends on.

## Why this is worth doing at all

Fuel is one workspace of about forty crates: a lazy DAG and optimizer, planner, executor, dispatch,
storage for each backend, KV cache, a large model zoo, loaders and converters, datasets, training
and post-training, probing and judging, hardware topology, kernels in Slang and Metal, and the
public facade (`fuel`). Kernel runners (Baracuda, Vulkane, AOCL) and several other parts are
already separate projects, joined to Fuel by seam crates so that neither side needs the other's
version.

Splitting more of Fuel could help consumers who want only part of it, let parts release at their
own pace, and lower the gate cost of changes that touch one part. It has costs too, measured on
2026-11-01: every PR already pays for ratchets, provenance regeneration and a manifest and
CHANGELOG merge-forward; a split multiplies that per project and adds a compatibility story. The
one-version-per-project rule means a split-off project needs its own answer to "which version works
with which" (the seam crate is that answer where it exists). These are reasons to do the order of
work carefully, not reasons to avoid it.

## When a split pays (criteria, to be tested by row 136)

A part is a split candidate when all of these hold. The first three are measurable.

- Its interface to the rest is narrow and changes slowly.
- It has, or could have, consumers that do not need the rest of Fuel.
- It changes on a different rhythm from its neighbours (it rarely changes in the same commit).
- A seam crate with a conformance corpus can carry it, so neither side needs the other's version.

Where the seam is wide but the two sides are logically separate, the width is itself a finding:
look at separation of concerns, modularity, encapsulation, abstraction, single responsibility,
loose coupling and high cohesion near that seam (row 137) before deciding anything.

## Observations that motivate row 137

Taken from code read on 2026-11-01. They are leads, not an audit; row 136 measures them properly.

- **Backend knowledge in the generic executor path.** `fuel-tensor`'s `pipelined_bridge` has
  per-backend arms; `device_for_location` special-cases Vulkan; multi-device seeding keyed on the
  kind of backend (`BackendId`) where it needed the identity of the device (the row 114 defect,
  fixed by keying on `DeviceLocation`). Seeding by pushing dummy "anchor" nodes into the graph so the
  executor can find a device handle points at a missing abstraction (a device registry).
- **Direction of dependencies.** `fuel-hardware` is documented as discovery only and must not depend
  on dispatch, yet it depends on the backend crates to enumerate, and rows 131 and 125 added live
  memory and identity queries there that open real GPU handles. Whether that belongs in
  `fuel-hardware` or behind one backend-reported trait is a design question.
- **Encapsulation.** `VulkanBackend` exposes a public `physical` field; adding one quantized kernel
  touched the backend `lib.rs` (about 14,000 lines), the dispatch file, a pipelines macro list, the
  kernels crate table and a registry-count test.
- **Identity versus index.** `DeviceLocation::{Cuda,Vulkan}.gpu_id` is a within-process ordinal and
  its order differs between builds on the development box (row 125); several places treat it as an
  identity.
- **Mixed responsibilities.** Crates that enumerate, measure, probe, judge and dispatch together;
  `fuel-tensor` hosting modules named after specific backends.

## Method for one seam (row 137, repeated per candidate)

1. List everything public that crosses the seam: types, functions, re-exports.
2. Count use sites on each side; find back-edges (the lower side naming the upper) and
   `cfg(feature)` branches that mention the other side.
3. Find leaking types: backend-specific handles or indexes in generic signatures.
4. Write down what each side needs to know about the other and the narrowest interface that
   carries it.
5. Compare with the existing public surface; the gap is the cleanup list.
6. Do the cleanup in small PRs, each with the usual red-first test; re-measure steps 1 and 2.

## Candidate seams (hypotheses for row 136 to confirm or reject)

Ordered by how narrow the boundary is expected to be. All are judgement until measured.

1. Format and interchange code (ONNX, GGUF, conversion tools) from the core: weights plus a graph
   description.
2. Hardware probe and judge profiles from dispatch: device descriptors and profiles.
3. Executor bridge from the backends: the leak list above; this is where the cleanup is most likely
   to pay first even if no split follows.
4. Training, post-training, datasets and the model zoo from the core: all sit on the lazy `Tensor`
   API, which is wide; a seam there means deciding to freeze that API.
5. Slang kernel sources and compiled artifacts as their own project.

Not a candidate: the IR, graph, optimizer, planner, dispatch and storage together. They share one
data model.

## What is NOT decided

- Whether any split happens. Row 139 needs its own go (CireSnave, through the PM), including what
  the one-version-per-project rule means for the new projects.
- Fireline's contract was not read when this note was written; row 136 reads it and the existing
  consumer-contract docs (`docs/fuel-consumer-seam.md`, architecture section 15) first, and reports
  which candidate boundaries already fit them.
- The candidate list and the criteria above.
