//! CUTLASS bridge — registers CUTLASS-backed GEMM kernels as alternative
//! implementations at the `(MatMul, *, Cuda)` and `(FusedLinear, *, Cuda)`
//! decision points.
//!
//! Per architecture v1.0, CUTLASS kernels are *siblings* to the cuBLAS
//! path, not replacements. The optimizer + route-picker rank them by
//! `PrecisionGuarantee` + empirical telemetry. cuBLAS provides bit-stable
//! coverage; CUTLASS provides throughput alternatives (TF32, Rrr-layout
//! GEMM, fused Bias/BiasReLU/BiasGELU/BiasSiLU epilogues).
//!
//! This module is currently the integration seam — kernel wrappers land
//! in subsequent commits.

#[allow(unused_imports)]
pub use baracuda_cutlass::*;
