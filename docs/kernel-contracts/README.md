# Fuel kernel-contract corpus

This directory holds the **FKC (Fuel Kernel Contract)** files for Fuel's internal kernels —
the markdown + ` ```fkc ` structured-block files a provider ships so that *importing them
auto-registers every kernel* onto Fuel's dispatch surface, with zero hand-written
registration glue.

The format itself is specified in
[`../specs/kernel-contract-format.md`](../specs/kernel-contract-format.md) (FKC). Its
tensor-axis sibling is [`../specs/dlpack-extension.md`](../specs/dlpack-extension.md) (FDX).
Read the format spec before authoring or editing a contract; this README is only the index.

## What lives here

These bundle files together describe **all ~390 of Fuel's internal kernels** — every kernel
Fuel itself ships across the CPU backend, the GPU backends (Vulkan / Metal), the dispatch
layer's CUDA/PTX wrappers, the quantized path, and the fused-op registry — plus the reference
(CPU correctness oracle) family. They are organized **one subdirectory per provider/family**,
mirroring the nine per-crate kernel inventories under `_inventory/` (see "Provenance" below).

`_inventory/` is **not** part of the corpus: those are the as-built audits (one row per
distinct `(OpKind, dtypes)` binding, read straight from the dispatch/backend sources) that
*seed* the contracts. The `.fkc.md` files are the authoritative importable contracts; the
inventories are reference notes. The index below excludes `_inventory/`.

## Per-provider / per-family bundle files

| Provider / family | Directory | Bundle files |
|-------------------|-----------|--------------|
| **CPU backend** (`fuel-cpu-backend`) | `cpu/` | `affine-clamp-powi`, `attention`, `cast`, `compare-where`, `conv`, `elementwise-binary`, `elementwise-unary`, `indexing`, `inplace-unary-affine`, `matmul`, `norm`, `norm-backward`, `padding`, `quant-matmul`, `reduce`, `reduce-to`, `rope`, `shape-ops`, `ssm` |
| **Reference** (CPU correctness oracle) | `reference/` | `attention`, `broadcast-binary`, `cast`, `conv-pool`, `elementwise`, `indexing`, `matmul`, `norm-rope`, `reduce`, `shape-mask-pad` |
| **Vulkan** (`fuel-vulkan-kernels`, Slang) | `vulkan/` | `cast`, `conv-attn-rope`, `data-movement`, `elementwise`, `indexing`, `matmul`, `norm-softmax`, `padding`, `quantized`, `reduce` |
| **Metal** (`fuel-metal-kernels`) | `metal/` | `cast`, `conv-pool`, `elementwise`, `indexing`, `matmul-attn`, `quantized`, `reduce-norm-rope`, `sort-random` |
| **Dispatch layer** (`fuel-dispatch`, CPU + PTX/CUDA wrappers) | `dispatch/` | `cast-affine`, `conv-attn`, `elementwise-binary`, `elementwise-unary`, `indexing`, `inplace`, `matmul`, `norm-softmax`, `reduce`, `shape-ops` |
| **Quantized** (`fuel-quantized`) | `quantized/` | `dequantize`, `quantize`, `vec-dot-matmul` |
| **Conv / attention** (CUDA family) | `conv-attn/` | `conv`, `flash-attn-cuda` |
| **Fused ops** (`FusedKernelRegistry`) | `fused/` | `attention`, `conv-rope`, `linear-quant`, `norm-softmax` |
| **MKL / AOCL** (vendor BLAS) | `mkl-aocl/` | `matmul-conv` |

Every file uses the `.fkc.md` extension. Each file begins with YAML front-matter declaring the
provider-wide defaults (`backend`, `kernel_source`, `link_registry`, `revision_base`), then one
`## ` section per kernel, each carrying a prose blurb + long description followed by exactly one
` ```fkc ` block (FKC §3.1).

## The FKC import model — single-bundle vs globbed multi-file

A provider becomes dispatchable by **importing its contract file(s)** (FKC §9). Two equivalent
layouts are supported, and the importer treats them identically (FKC §3.1, §9.1–§9.2):

- **Single bundle file.** One `<provider>.fkc.md` with N `## ` sections — one file, N
  registrations. Front-matter supplies provider-wide defaults; each section overrides as needed.
- **Globbed multi-file layout.** Many `*.fkc.md` files, one-or-more kernels each, discovered by
  a glob (e.g. `cpu/**/*.fkc.md`). A bare `_provider.fkc.md` (front-matter only, no sections)
  supplies tree-wide defaults that per-file front-matter overrides. Files are processed in
  **sorted path order** for deterministic registration ordering (this determines the order of
  sibling alternatives at a dispatch key — FKC §12.5).

This corpus is authored in the **globbed multi-file** style: each provider/family is a
subdirectory of split-by-family `.fkc.md` files rather than one monolith. The import pipeline
(FKC §9.3) parses each file's front-matter, extracts each section's ` ```fkc ` block, validates
it (`Result`-returning, never panic — FKC §10), resolves the `entry_point` symbol against the
provider's `link_registry` to a `KernelRef`, and registers it: `op_kind` contracts via
`KernelBindingTable::register_full_with_source(...)`, `fused_op` contracts via the
`FusedKernelRegistry`. A workspace manifest (FKC §9.4) binds each provider's contract glob to
its link-registry symbol; startup imports them all and freezes the registry.

## Cost `provenance` matches the block's content (both provenances are first-class)

Every contract's `cost` block carries a **required `provenance:` field** with exactly one of two
**first-class** values (FKC §4.4, §10.8a). Neither is a quality ranking; each simply records where
the numbers in *that* block came from, and **must match the block's actual content**:

- `declared` — the block carries an **authored absolute constant** that is *not* a derivable
  formula: a literal launch/latency number (e.g. `overhead_ns: 40`, `overhead_ns: 4000`) or, for a
  genuinely free / no-op kernel, an authored true-zero (`class: free` with `overhead_ns: 0`, or a
  CPU host-call with no launch overhead). That constant is a legitimate **author prior** — a
  starting value the Judge later refines, not a final measurement and not a placeholder.
- `judge_measured` — the block carries **no authored absolute constant**: only derivable formula
  hints (`flops: 2*m*n*k`, `bytes_moved: n*elem`, or a structural hint in a comment) and/or `~`
  placeholders for the parts that cannot be derived (notably the non-derivable launch overhead,
  which is written `overhead_ns: ~` under this provenance, **never** a fabricated number). It records
  that the live coefficient is the Judge's to populate/calibrate, not the author's to assert.

So the rule for choosing `provenance` is **content-driven, not a lifecycle stage**:

- If the block authors any absolute numeric constant that isn't a formula → `provenance: declared`.
- If the block is formula-hints-and/or-`~` only → `provenance: judge_measured`, with every
  non-derivable absolute (launch overhead) written as `~`.
- **Derivable formula hints are always allowed under either provenance** — a `flops` / `bytes_moved`
  formula is a structural fact, not a fabricated constant, so it never forces the provenance.
- `overhead_ns: 0` for a genuinely free / no-op kernel is a legit `declared` **true-zero**, not a
  fabrication; do not demote it to `judge_measured`.

A **provenance token must never sit in a numeric field** (`flops: judge_measured` /
`overhead_ns: declared` and the like are malformed — the value belongs in `provenance:`, and the
numeric field takes a formula, a literal constant, `0`, or `~`). Likewise an authored absolute
constant must **never** sit under `judge_measured`. Both are lint failures: every cost the optimizer
reads carries an origin that matches its content, and **no cost is silently a placeholder** (the 01
visibility gate). A bare, sentinel, or origin-less cost is also a failure — a cost must be *either*
`declared` *or* explicitly `judge_measured`.

The Judge then **bootstraps** cost from there: it treats an author-`declared` constant as a prior
it refines, and when it refines one it flips the binding's recorded provenance to `judge_measured`
(FKC §4.4, §11). FKC stays **agnostic to the Judge's internals** — it depends only on the two facts
that the Judge exists and that it refines/bootstraps cost, never on *how* it measures (the Judge is
mid-rebuild). Contracts in this corpus accordingly carry **both** provenances side by side today:
blocks with an authored launch-overhead prior (or a free-op true-zero) are `declared`; blocks that
expose only derivable formula hints with `overhead_ns: ~` are `judge_measured` and await the Judge's
calibration.

## Relationship to FDX

FKC and FDX are the two halves of the kernel boundary, kept as separate concerns (the
13-interchange "weight ⊥ graph" principle):

- **FKC describes a *kernel*** — the **advertisement / capability-cost axis**: dispatch key,
  per-operand accept-contract, per-output return-contract, and the capability + cost + precision
  + determinism advertisement the optimizer uses to choose, cost, admit, and dispatch.
- **FDX describes a *tensor* handed to that kernel** — the **tensor / storage axis**: an honest
  standard `DLTensor` base plus an optional sidecar for sub-byte / quant / symbolic-extent /
  residency / multi-output-bundle facts.

**FDX is the single normative source for all shared codes** (dtype, quant `family`,
`granularity`, `pack_order`, substrate). FKC **never re-lists** their numeric values: a tensor
descriptor in an `.fkc.md` file names a dtype/quant/granularity token by its FDX symbol and the
contract cites the FDX section, so the two vocabularies cannot drift. The split shows up
throughout: FKC advertises a kernel's *tolerance* for, e.g., an affine symbolic extent or a
paged/gather operand; FDX *describes* the specific extent expression or block table on the
tensor itself (FKC §0, §3.2, §3.9, §4.5).

## Provenance — the `_inventory/` seed

The bundle files are derived from the nine per-crate kernel inventories in
[`_inventory/`](_inventory/): `cpu`, `vulkan`, `conv-attn`, `fused`, `quantized`, `reference`,
`metal`, `mkl-aocl`, `dispatch`. Those inventories are as-built audits (read directly from
`fuel-dispatch` / the backend crates / `fuel-graph`'s fused registry) and are the canonical
~390-kernel census the contracts above realize. When a kernel's as-built behavior changes, update
its inventory row **and** its `.fkc.md` contract in the same change.
