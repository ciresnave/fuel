# Fuel Ecosystem Compatibility Matrix

This document records the known-good dependency set for the Fuel workspace
and the ecosystem crates that integrate with it. It exists so that version drift
is caught before it silently breaks builds.

---

## Workspace version

| Crate                 | Version  |
| --------------------- | -------- |
| `fuel-core`         | `0.10.2` |
| `fuel-nn`           | `0.10.2` |
| `fuel-transformers` | `0.10.2` |
| `fuel-datasets`     | `0.10.2` |
| `fuel-onnx`         | `0.10.2` |

Workspace `edition = "2024"`, Rust resolver `"2"`.

---

## Ecosystem crate status (audited 2025-04-06)

| Crate               | Version               | Fuel pin                              | Status                                                   |
| ------------------- | --------------------- | --------------------------------------- | -------------------------------------------------------- |
| `fuel-optimisers` | `0.10.0-alpha.2`      | `0.9.2-alpha.1` registry                | **Outdated** — pins older alpha, needs bump to `0.10.2`  |
| `fuel-bhop`       | git (upstream HF ref) | `db08cc0a…` huggingface/fuel          | **Outdated** — should point to fork `main`, not upstream |
| `fuel-layer-norm` | `0.0.3`               | `git = ciresnave/fuel, branch = main` | OK for local builds; no crates.io version                |
| `fuellight`       | `0.2.1`               | `git = ciresnave/fuel` (no rev pin)   | OK for local builds; floating ref                        |
| `fuel-cuda-vmm`   | `0.1.1`               | N/A — local path dep                    | Used locally; not yet on crates.io                       |

### Findings

#### `fuel-optimisers`

Pins `fuel-core = "0.9.2-alpha.1"` and `fuel-nn = "0.9.2-alpha.1"` from
the registry. The workspace has moved to `0.10.2`. Any project that pulls both
`fuel-optimisers` and workspace crates will see a version conflict.

**Action required**: Bump to `fuel-core = "0.10.2"` and publish to crates.io,
or switch to `git = "https://github.com/ciresnave/fuel-optimisers"` with the
`update-fuel-deps-for-cuda-13` branch.

#### `fuel-bhop`

Uses `git = "https://github.com/huggingface/fuel", rev = "db08cc0a…"` — this
is a specific commit in the upstream HuggingFace repo, not our fork. When new
APIs are added to our fork they will not be visible to `fuel-bhop`.

**Action required**: Fork `fuel-bhop` at `ciresnave/fuel-bhop` (already
forked per `fuellight` Cargo.toml) and update the fuel pin to
`git = "https://github.com/ciresnave/fuel"`.

#### `fuel-layer-norm`

References `git = "https://github.com/ciresnave/fuel.git", branch = "main"`.
Floating `branch = "main"` means it tracks our fork's latest. This is acceptable
for development but will produce non-reproducible builds.

**Action required** (deferred): Once workspace reaches a stable release tag,
lock `fuel-layer-norm` to a rev rather than a branch.

#### `fuellight`

All core crates pulled via `git = "https://github.com/ciresnave/fuel"` without
a rev pin. Same reproducibility concern as `fuel-layer-norm`. Ecosystem deps
(`fuel-einops`, `fuel-birnn`, `fuel-lstm`, `fuel-crf`, `fuel-approx`,
`fuel_embed`, `fuel-ext`, `border-fuel-agent`) are all pulled from
`ciresnave/*` forks.

**Action required** (deferred): Same rev-pinning strategy above.

## Key external dependency pins

The authoritative version matrix is `[workspace.dependencies]` in `Cargo.toml`.
The table below highlights the external deps most likely to cause conflicts when
integrating ecosystem crates:

| Dependency    | Pinned version | Notes                                               |
| ------------- | -------------- | --------------------------------------------------- |
| `cudarc`      | `0.19.4`       | CUDA driver bindings; must match toolkit version    |
| `half`        | `2.5.0`        | FP16/BF16; `num-traits` + `rand_distr` features on  |
| `safetensors` | `0.7.0`        | Model weight format; API-breaking on major bumps    |
| `tokenizers`  | `0.22.0`       | HuggingFace tokenizers; default-features disabled   |
| `rand`        | `0.9`          | Needed by generation/sampling; must match ecosystem |
| `serde`       | `1.0.171`      | Serialisation; `derive` feature on                  |
| `rayon`       | `1.7.0`        | Parallel CPU ops                                    |
| `thiserror`   | `2`            | Error derives; major version differs from upstream  |

Ecosystem crates that pin an older `rand` (`0.8.x`) will conflict with this
workspace. Check `fuel-optimisers` specifically before updating `rand`.

---

## `fuel-vmm` / `fuel-cuda-vmm` split (completed)

`fuel-vmm` (`0.1.0`) has been extracted as a standalone crate:

- `VmmBackend` trait — 8 methods, the only abstraction surface
- `VirtualMemoryPool<B: VmmBackend>` — generic elastic pool
- `SharedMemoryPool<B: VmmBackend + Clone>` — multi-model pool
- `VmmError` — backend-agnostic error type (`BackendError` replaces old `CudaError`)

`fuel-cuda-vmm` has been refactored to depend on `fuel-vmm`:

- `CudaVmmBackend` — the sole CUDA-specific `impl VmmBackend`
- `VirtualMemoryPool` / `SharedMemoryPool` — type aliases for backward compatibility
- `MemoryStats` / `GlobalMemoryStats` — re-exported from `fuel-vmm`

---

## Platform support matrix

| Configuration                  | Status                                                         |
| ------------------------------ | -------------------------------------------------------------- |
| Linux + CUDA 12.x              | Tested upstream                                                |
| Linux + CUDA 13.0              | Required by `fuel-cuda-vmm`; CUDA VMM APIs stable since 11.2 |
| Windows + CUDA 13.0 + clang-cl | Required by `fuel-cuda-vmm`; OpenBLAS needs clang-cl         |
| macOS + Metal                  | Tested upstream                                                |
| CPU-only                       | Tested upstream                                                |

### Known issues

- `fuel-layer-norm` does not build on Windows + MSVC (inline assembly). Requires
  clang or clang-cl. See ROADMAP Phase 0 for fix tracking.
- `fuel-cuda-vmm` CUDA VMM bindings require CUDA ≥ 11.2 and Compute Capability ≥ 6.0.
  Builds on Windows only with clang-cl (not MSVC `cl.exe`) and Ninja generator.

---

## Recommended build toolchain

Per `.github/copilot-instructions.md`:

| Component          | Requirement                            |
| ------------------ | -------------------------------------- |
| Generator          | Ninja only (`-G Ninja`)                |
| Compiler (Windows) | `clang-cl` only                        |
| Linker             | LLD (`-fuse-ld=lld`)                   |
| Rust edition       | 2024                                   |
| MSRV               | Rust 1.87+ (required for edition 2024) |

---

Last updated: 2026-04-07
