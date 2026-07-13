---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - dense FP matmul facade (gemm_dense at OpKind::MatMul) kernel contracts

CUDA (baracuda) packed row-major (wrapper validates byte lengths), so contiguous (default caps); OpParams::Matmul.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is the author-declared `audited: false` -> UNAUDITED seed.


---

## matmul  (MatMul - {F32, F16, BF16, F64}, contiguous-only)

out=lhs@rhs Backs `OpKind::MatMul`. packed row-major (wrapper validates byte lengths), so contiguous (default caps); OpParams::Matmul. Output: fresh, contiguous, no aliasing.

```fkc
kernel: matmul
op_kind: MatMul
blurb: "out=lhs@rhs (CUDA/baracuda) {F32, F16, BF16, F64}; contiguous-only; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::matmul"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: Matmul }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: >
    Audited 2026-07-11 (task-cublas-audit; see
    `.superpowers/sdd/task-cublas-audit-report.md` and
    `docs/architecture/10-decisions-log.md` 2026-07-11 entry for full
    evidence). Call mode: `cublasGemmEx` (batch==1) /
    `cublasGemmStridedBatchedEx` (batch>1) under `CUBLAS_GEMM_DEFAULT`
    heuristic algorithm selection, `CUBLAS_COMPUTE_32F` (true IEEE
    binary32, NOT TF32), via baracuda-kernels-sys's
    `gemm_dense_cublas_facade` (baracuda 0.0.1-alpha.77) and its
    per-CUDA-context pooled cuBLAS handle (one handle bound to one
    stream at a time via `cublasSetStream_v2`). NVIDIA's cuBLAS docs
    (https://docs.nvidia.com/cuda/cublas/#results-reproducibility)
    state bit-wise reproducibility across runs "by design" for a fixed
    toolkit version + GPU architecture + SM count, provided a single
    CUDA stream is active; the guarantee is explicitly NOT claimed
    across toolkit versions or under multiple concurrently-active
    streams sharing a handle's workspace (mitigated by
    `CUBLAS_WORKSPACE_CONFIG` / per-stream workspace, not applicable
    here — Fuel's decode graph and this audit both use one context/
    stream per matmul sequence). Empirically verified on this exact
    machine (RTX 4070, driver + CUDA 13.3 + baracuda alpha.77, git
    f41137b4-derived crate) at 5 real decode-graph matmul shapes (Q/O
    dim128 GEMV, KV-proj GQA-narrow, FFN up/down, batched attention
    scores): (a) 150 repeat calls per shape, byte-identical to the
    first call, zero deviations; (b) same, while a SECOND CUDA context/
    stream concurrently launched >600 GEMMs in a tight loop (genuine
    cross-stream contention, not same-stream serialization); (c) the
    same golden output reproduced across 3 separate `cargo test`
    process invocations (fresh cuBLAS handle each time). Every check
    passed with zero deviation. Not claimed: bit-reproducibility across
    a cuBLAS/CUDA/driver upgrade, or under >1 concurrently-active
    stream sharing a single pooled handle's default workspace pool (a
    real but narrower risk than the seed note implied — documented
    above, not observed to actually break in this audit's 2-context
    test, but NVIDIA's own docs decline to guarantee it, so this
    contract does not claim it either).

determinism: same_hardware_bitwise
```
