# Glossary

Domain terms used across Fuel's documentation and code, each defined once. Where a term is
specified in detail, the entry points at the authoritative document rather than restating it;
`docs/architecture/` is the constitution and wins any conflict with this file.

## B

**Backend** — a component (CPU, CUDA, Vulkan, Metal, AOCL, MKL, ...) that reports its kernels,
capabilities, measured cost data and telemetry to the optimizer and executes what the optimizer
hands it. Backends advertise; they never make strategic decisions (placement, routing, fusion,
dtype choice). See [`docs/architecture/01-identity.md`](docs/architecture/01-identity.md).

**Baracuda** — the sibling project that supplies Fuel's CUDA kernels and JIT synthesis. Fuel
consumes it as a published crate; edits to it go through its own project.

**Binding table** — the registry that maps each operation cell (op, dtype, size class, backend)
to the kernel variants registered for it, from which the dispatcher and route picker choose.
Registering a kernel adds an entry; choosing among entries is the optimizer's job, not the
backend's.

**`bit_stable_on_same_hardware`** — the strongest field of a `PrecisionGuarantee`: the kernel
produces bit-identical output for bit-identical inputs on the same hardware, with no
nondeterminism from scheduling or atomic ordering.

## C

**Constitution** — `docs/architecture/`: the authoritative architecture documents. It outranks
`ROADMAP.md`, the session prompts and every handoff document when they disagree.

## D

**DAG-first** — Fuel's central commitment: every model is a directed acyclic graph of
operations, and that graph is the source of truth for every decision (what runs, where, in what
form, when). The optimizer that reads the graph is where the framework's intelligence lives.

## F

**Facade** — the `fuel` crate: a thin crate that re-exports Fuel's public API from the crates
that own each item, so consumers write `use fuel::...` whatever the internal crate layout.

**FDX** — Fuel DLPack Extension: the specification for exchanging tensors across the
device/kernel boundary, and the single normative source for the numeric codes shared between
Fuel and its kernel sibling projects. See [`docs/specs/dlpack-extension.md`](docs/specs/dlpack-extension.md).

**FKC** — Fuel Kernel Contract: the declarative description of a kernel (what it computes, on
which dtypes and layouts, with what precision guarantee) from which backends register kernels
into the binding table. See [`docs/specs/kernel-contract-format.md`](docs/specs/kernel-contract-format.md).

**Fused op** — an operation that stands for a subgraph of primitive operations and is registered
in the fused-op registry. Every fused op ships a total, never-panicking `decompose` back to
primitives and a `pattern` that recognises its subgraph (the recipe principle).

## G

**GAP** — a row in [`docs/gaps.md`](docs/gaps.md), the source of truth for open defects and who
owns them. A change that closes a gap updates its row in the same change.

## J

**Judge** — the empirical profiler: it measures every (op, dtype, size class) on every
(backend, device) and produces the profile-driven cost data the optimizer and route picker
consume. Measurements override static cost estimates.

## K

**KISS** — Kernel Interface Standards Suite: the public (CC0) standard that Fuel's kernel seam
seeded and now conforms to, covering operation vocabulary, wire encodings and conformance
vectors. See [`docs/specs/kernel-seam-interop.md`](docs/specs/kernel-seam-interop.md).

## L

**Lazy-only** — Fuel has no eager execution mode. Operations build graph nodes, and nothing runs
until an explicit `realize`.

## O

**Optimizer** — the component that reads the graph and the cost data backends report, and decides
placement, fusion, algebraic rewrites, precision and routing. It preserves a bounded Pareto
frontier of competitive routes rather than committing to one plan.

## P

**PrecisionGuarantee** — the structured statement every kernel registration carries about its
numerical behaviour: `bit_stable_on_same_hardware` plus optional maximum ULP, relative and
absolute error bounds and a free-text qualifier. The optimizer uses it to keep results inside a
tolerance budget.

## R

**Realize** — the explicit step that executes a lazy graph (or a subgraph) and produces concrete
tensor data.

**Route picker** — the runtime component that chooses among the optimizer's preserved routes
using live backend telemetry (memory pressure, queue depth, resident weights, priority).

## S

**Structure key** — the token that identifies a kernel cell by its operation class, operand
structure and target architecture; produced by a deriver against a versioned scheme (`sk4`,
`sk5`). It is the identity KISS conformance vectors are written against.

## T

**Tolerance budget** — the amount of numerical deviation from the strict-equivalence result that
a user, developer or model definition accepts, per operation or per region. A non-zero budget
lets the optimizer use approximate optimizations such as mixed-precision lowering.
