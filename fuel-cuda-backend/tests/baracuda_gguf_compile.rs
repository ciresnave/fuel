//! Compile-time smoke test for the GGUF dequant + MMVQ kernel
//! wrappers. Doesn't execute any kernel — just proves the public
//! function signatures resolve and the FFI symbols link.
//!
//! Live-CUDA execution tests for GGUF land alongside the QMatMul
//! dispatch wiring in a follow-up commit (the dispatch wrapper
//! needs to thread `QuantType` from `OpParams::QMatMul` through to
//! the right format-specific function and loop over batch/m rows
//! for the matrix-matrix case — out of scope for the kernel-only
//! integration here).

use fuel_cuda_backend::baracuda::gguf;

#[test]
fn gguf_function_signatures_compile() {
    // Every shipped wrapper is named below — if any disappear
    // upstream the test stops compiling. The function values
    // themselves are unused.
    let _dequants: [fn(
        &fuel_cuda_backend::CudaStorageBytes,
        usize,
    ) -> fuel_ir::Result<fuel_cuda_backend::CudaStorageBytes>; 11] = [
        gguf::dequant_q4_0,
        gguf::dequant_q4_1,
        gguf::dequant_q5_0,
        gguf::dequant_q5_1,
        gguf::dequant_q8_0,
        gguf::dequant_q2_k,
        gguf::dequant_q3_k,
        gguf::dequant_q4_k,
        gguf::dequant_q5_k,
        gguf::dequant_q6_k,
        gguf::dequant_q8_k,
    ];
    let _mmvqs: [fn(
        &fuel_cuda_backend::CudaStorageBytes,
        &fuel_cuda_backend::CudaStorageBytes,
        Option<&fuel_ir::Layout>,
        i64,
        usize,
        usize,
    ) -> fuel_ir::Result<fuel_cuda_backend::CudaStorageBytes>; 11] = [
        gguf::mmvq_q4_0,
        gguf::mmvq_q4_1,
        gguf::mmvq_q5_0,
        gguf::mmvq_q5_1,
        gguf::mmvq_q8_0,
        gguf::mmvq_q2_k,
        gguf::mmvq_q3_k,
        gguf::mmvq_q4_k,
        gguf::mmvq_q5_k,
        gguf::mmvq_q6_k,
        gguf::mmvq_q8_k,
    ];
}
