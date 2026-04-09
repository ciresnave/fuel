# Candle Ecosystem Compatibility Matrix

This document records the known-good dependency set for the Candle workspace
and the ecosystem crates that integrate with it. It exists so that version drift
is caught before it silently breaks builds.

---

## Workspace version

| Crate                 | Version  |
| --------------------- | -------- |
| `candle-core`         | `0.10.2` |
| `candle-nn`           | `0.10.2` |
| `candle-transformers` | `0.10.2` |
| `candle-datasets`     | `0.10.2` |
| `candle-onnx`         | `0.10.2` |

Workspace `edition = "2024"`, Rust resolver `"2"`.

---

## Ecosystem crate status (audited 2025-04-06)

| Crate               | Version               | Candle pin                              | Status                                                   |
| ------------------- | --------------------- | --------------------------------------- | -------------------------------------------------------- |
| `candle-optimisers` | `0.10.0-alpha.2`      | `0.9.2-alpha.1` registry                | **Outdated** — pins older alpha, needs bump to `0.10.2`  |
| `candle-bhop`       | git (upstream HF ref) | `db08cc0a…` huggingface/candle          | **Outdated** — should point to fork `main`, not upstream |
| `candle-layer-norm` | `0.0.3`               | `git = ciresnave/candle, branch = main` | OK for local builds; no crates.io version                |
| `candlelight`       | `0.2.1`               | `git = ciresnave/candle` (no rev pin)   | OK for local builds; floating ref                        |
| `candle-cuda-vmm`   | `0.1.1`               | N/A — local path dep                    | Used locally; not yet on crates.io                       |

### Findings

#### `candle-optimisers`

Pins `candle-core = "0.9.2-alpha.1"` and `candle-nn = "0.9.2-alpha.1"` from
the registry. The workspace has moved to `0.10.2`. Any project that pulls both
`candle-optimisers` and workspace crates will see a version conflict.

**Action required**: Bump to `candle-core = "0.10.2"` and publish to crates.io,
or switch to `git = "https://github.com/ciresnave/candle-optimisers"` with the
`update-candle-deps-for-cuda-13` branch.

#### `candle-bhop`

Uses `git = "https://github.com/huggingface/candle", rev = "db08cc0a…"` — this
is a specific commit in the upstream HuggingFace repo, not our fork. When new
APIs are added to our fork they will not be visible to `candle-bhop`.

**Action required**: Fork `candle-bhop` at `ciresnave/candle-bhop` (already
forked per `candlelight` Cargo.toml) and update the candle pin to
`git = "https://github.com/ciresnave/candle"`.

#### `candle-layer-norm`

References `git = "https://github.com/ciresnave/candle.git", branch = "main"`.
Floating `branch = "main"` means it tracks our fork's latest. This is acceptable
for development but will produce non-reproducible builds.

**Action required** (deferred): Once workspace reaches a stable release tag,
lock `candle-layer-norm` to a rev rather than a branch.

#### `candlelight`

All core crates pulled via `git = "https://github.com/ciresnave/candle"` without
a rev pin. Same reproducibility concern as `candle-layer-norm`. Ecosystem deps
(`candle-einops`, `candle-birnn`, `candle-lstm`, `candle-crf`, `candle-approx`,
`candle_embed`, `candle-ext`, `border-candle-agent`) are all pulled from
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
workspace. Check `candle-optimisers` specifically before updating `rand`.

---

## `candle-vmm` / `candle-cuda-vmm` split (completed)

`candle-vmm` (`0.1.0`) has been extracted as a standalone crate:

- `VmmBackend` trait — 8 methods, the only abstraction surface
- `VirtualMemoryPool<B: VmmBackend>` — generic elastic pool
- `SharedMemoryPool<B: VmmBackend + Clone>` — multi-model pool
- `VmmError` — backend-agnostic error type (`BackendError` replaces old `CudaError`)

`candle-cuda-vmm` has been refactored to depend on `candle-vmm`:

- `CudaVmmBackend` — the sole CUDA-specific `impl VmmBackend`
- `VirtualMemoryPool` / `SharedMemoryPool` — type aliases for backward compatibility
- `MemoryStats` / `GlobalMemoryStats` — re-exported from `candle-vmm`

---

## Platform support matrix

| Configuration                  | Status                                                         |
| ------------------------------ | -------------------------------------------------------------- |
| Linux + CUDA 12.x              | Tested upstream                                                |
| Linux + CUDA 13.0              | Required by `candle-cuda-vmm`; CUDA VMM APIs stable since 11.2 |
| Windows + CUDA 13.0 + clang-cl | Required by `candle-cuda-vmm`; OpenBLAS needs clang-cl         |
| macOS + Metal                  | Tested upstream                                                |
| CPU-only                       | Tested upstream                                                |

### Known issues

- `candle-layer-norm` does not build on Windows + MSVC (inline assembly). Requires
  clang or clang-cl. See ROADMAP Phase 0 for fix tracking.
- `candle-cuda-vmm` CUDA VMM bindings require CUDA ≥ 11.2 and Compute Capability ≥ 6.0.
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
