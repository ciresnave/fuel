// SPDX-License-Identifier: MIT OR Apache-2.0
//! x86 AVX SIMD kernels.
//!
//! # `unsafe` block form (GAP-176, ratified at `605df851`)
//!
//! Every `unsafe fn` here wraps its body in ONE documented `unsafe {}` block
//! rather than one block per operation. In these kernels every operation in a
//! function shares a SINGLE obligation — the `#[target_feature]` availability
//! that the `unsafe fn` contract already promises, plus (where pointers are
//! taken) that the caller passed a readable/writable run of the stated length.
//! Per-operation blocks would repeat one identical SAFETY note up to nine times
//! inside a tight loop, and the `*y = …` stores cannot be wrapped per-operation
//! at all, since the dereference is an assignment LHS.
//!
//! Per-operation blocks remain the right default where operations carry
//! DISTINCT obligations. That is not this file.

use super::{Cpu, CpuBF16, CpuF16};
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use half::{bf16, f16};

pub struct CurrentCpu {}

const STEP: usize = 32;
const EPR: usize = 8;
const ARR: usize = STEP / EPR;

impl Cpu<ARR> for CurrentCpu {
    type Unit = __m256;
    type Array = [__m256; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;

    fn n() -> usize {
        ARR
    }

    unsafe fn zero() -> Self::Unit {
        // SAFETY: `avx` available (the `unsafe fn` contract); no further precondition.
        unsafe { _mm256_setzero_ps() }
    }

    unsafe fn zero_array() -> Self::Array {
        // SAFETY: `Self::zero` upholds the same `avx` target-feature contract.
        unsafe { [Self::zero(); ARR] }
    }

    unsafe fn from_f32(v: f32) -> Self::Unit {
        // SAFETY: `avx` available (the `unsafe fn` contract); no further precondition.
        unsafe { _mm256_set1_ps(v) }
    }

    unsafe fn load(mem_addr: *const f32) -> Self::Unit {
        // SAFETY: `avx` available, and the contract requires `mem_addr` to point
        // to `EPR` (8) readable, contiguous `f32`s. `loadu` needs no alignment.
        unsafe { _mm256_loadu_ps(mem_addr) }
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        // SAFETY: `avx` available (the `unsafe fn` contract); no further precondition.
        unsafe { _mm256_add_ps(a, b) }
    }

    #[cfg(target_feature = "fma")]
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        // SAFETY: `fma` available — this arm is `cfg`-gated on it and the
        // `unsafe fn` contract carries it.
        unsafe { _mm256_fmadd_ps(b, c, a) }
    }

    #[cfg(not(target_feature = "fma"))]
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        // SAFETY: `avx` available (the `unsafe fn` contract). Both operations
        // share that one obligation.
        unsafe { _mm256_add_ps(_mm256_mul_ps(b, c), a) }
    }

    unsafe fn vec_store(mem_addr: *mut f32, a: Self::Unit) {
        // SAFETY: `avx` available, and the contract requires `mem_addr` to point
        // to `EPR` (8) writable, contiguous `f32`s. `storeu` needs no alignment.
        unsafe { _mm256_storeu_ps(mem_addr, a) };
    }

    unsafe fn vec_reduce(mut x: Self::Array, y: *mut f32) {
        // SAFETY: `avx`/`sse3` available (the `unsafe fn` contract), and `y`
        // points to one writable `f32`. Every operation below — the pairwise
        // adds, the 128-bit fold, and the `*y` store — shares those two
        // obligations, and the store's dereference is an assignment LHS that
        // cannot carry its own block.
        unsafe {
            for i in 0..ARR / 2 {
                x[2 * i] = _mm256_add_ps(x[2 * i], x[2 * i + 1]);
            }
            for i in 0..ARR / 4 {
                x[4 * i] = _mm256_add_ps(x[4 * i], x[4 * i + 2]);
            }
            #[allow(clippy::reversed_empty_ranges)]
            for i in 0..ARR / 8 {
                x[8 * i] = _mm256_add_ps(x[8 * i], x[8 * i + 4]);
            }
            let t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]), _mm256_extractf128_ps(x[0], 1));
            let t1 = _mm_hadd_ps(t0, t0);
            *y = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
        }
    }
}

pub struct CurrentCpuF16 {}
impl CpuF16<ARR> for CurrentCpuF16 {
    type Unit = __m256;
    type Array = [__m256; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;

    fn n() -> usize {
        ARR
    }

    unsafe fn zero() -> Self::Unit {
        // SAFETY: `avx` available (the `unsafe fn` contract); no further precondition.
        unsafe { _mm256_setzero_ps() }
    }

    unsafe fn zero_array() -> Self::Array {
        // SAFETY: `Self::zero` upholds the same `avx` target-feature contract.
        unsafe { [Self::zero(); ARR] }
    }

    unsafe fn from_f32(v: f32) -> Self::Unit {
        // SAFETY: `avx` available (the `unsafe fn` contract); no further precondition.
        unsafe { _mm256_set1_ps(v) }
    }

    #[cfg(target_feature = "f16c")]
    unsafe fn load(mem_addr: *const f16) -> Self::Unit {
        // SAFETY: `avx`/`f16c` available (this arm is `cfg`-gated on `f16c`), and
        // the contract requires `mem_addr` to point to 8 readable, contiguous
        // `f16`s — exactly the 128 bits `loadu_si128` reads.
        unsafe { _mm256_cvtph_ps(_mm_loadu_si128(mem_addr as *const __m128i)) }
    }

    #[cfg(not(target_feature = "f16c"))]
    unsafe fn load(mem_addr: *const f16) -> Self::Unit {
        let mut tmp = [0.0f32; 8];
        // SAFETY: `avx` available, and the contract requires `mem_addr` to point
        // to 8 readable, contiguous `f16`s — so `add(i)` for `i < 8` stays in
        // bounds and each dereference is valid. `tmp` is a local 8-`f32` array,
        // so its pointer is trivially valid for the load.
        unsafe {
            for i in 0..8 {
                tmp[i] = (*mem_addr.add(i)).to_f32();
            }
            _mm256_loadu_ps(tmp.as_ptr())
        }
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        // SAFETY: `avx` available (the `unsafe fn` contract); no further precondition.
        unsafe { _mm256_add_ps(a, b) }
    }

    #[cfg(target_feature = "fma")]
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        // SAFETY: `fma` available — this arm is `cfg`-gated on it and the
        // `unsafe fn` contract carries it.
        unsafe { _mm256_fmadd_ps(b, c, a) }
    }

    #[cfg(not(target_feature = "fma"))]
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        // SAFETY: `avx` available (the `unsafe fn` contract). Both operations
        // share that one obligation.
        unsafe { _mm256_add_ps(_mm256_mul_ps(b, c), a) }
    }

    #[cfg(target_feature = "f16c")]
    unsafe fn vec_store(mem_addr: *mut f16, a: Self::Unit) {
        // SAFETY: `avx`/`f16c` available (this arm is `cfg`-gated on `f16c`), and
        // the contract requires `mem_addr` to point to 8 writable, contiguous
        // `f16`s — exactly the 128 bits `storeu_si128` writes.
        unsafe { _mm_storeu_si128(mem_addr as *mut __m128i, _mm256_cvtps_ph(a, 0)) }
    }

    #[cfg(not(target_feature = "f16c"))]
    unsafe fn vec_store(mem_addr: *mut f16, a: Self::Unit) {
        let mut tmp = [0.0f32; 8];
        // SAFETY: `avx` available, and the contract requires `mem_addr` to point
        // to 8 writable, contiguous `f16`s — so `add(i)` for `i < 8` stays in
        // bounds and each store is valid. `tmp` is a local 8-`f32` array. The
        // stores' dereferences are assignment LHS and cannot carry their own
        // blocks.
        unsafe {
            _mm256_storeu_ps(tmp.as_mut_ptr(), a);
            for i in 0..8 {
                *mem_addr.add(i) = f16::from_f32(tmp[i]);
            }
        }
    }

    unsafe fn vec_reduce(mut x: Self::Array, y: *mut f32) {
        // SAFETY: `avx`/`sse3` available (the `unsafe fn` contract), and `y`
        // points to one writable `f32`. Every operation below shares those two
        // obligations, and the `*y` store's dereference is an assignment LHS
        // that cannot carry its own block.
        unsafe {
            let mut offset = ARR >> 1;
            for i in 0..offset {
                x[i] = _mm256_add_ps(x[i], x[offset + i]);
            }
            offset >>= 1;
            for i in 0..offset {
                x[i] = _mm256_add_ps(x[i], x[offset + i]);
            }
            offset >>= 1;
            for i in 0..offset {
                x[i] = _mm256_add_ps(x[i], x[offset + i]);
            }
            let t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]), _mm256_extractf128_ps(x[0], 1));
            let t1 = _mm_hadd_ps(t0, t0);
            *y = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
        }
    }
}

pub struct CurrentCpuBF16 {}
impl CpuBF16<ARR> for CurrentCpuBF16 {
    type Unit = __m256;
    type Array = [__m256; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;

    fn n() -> usize {
        ARR
    }

    unsafe fn zero() -> Self::Unit {
        // SAFETY: `avx` available (the `unsafe fn` contract); no further precondition.
        unsafe { _mm256_setzero_ps() }
    }

    unsafe fn zero_array() -> Self::Array {
        // SAFETY: `Self::zero` upholds the same `avx` target-feature contract.
        unsafe { [Self::zero(); ARR] }
    }

    unsafe fn from_f32(v: f32) -> Self::Unit {
        // SAFETY: `avx` available (the `unsafe fn` contract); no further precondition.
        unsafe { _mm256_set1_ps(v) }
    }

    #[cfg(target_feature = "f16c")]
    unsafe fn load(mem_addr: *const bf16) -> Self::Unit {
        // BF16 is the upper 16 bits of f32, so zero-extend to 32-bit and shift left by 16
        //
        // SAFETY: `avx`/`avx2` available (the `unsafe fn` contract), and the
        // contract requires `mem_addr` to point to 8 readable, contiguous
        // `bf16`s — exactly the 128 bits `loadu_si128` reads. The widen and
        // shift operate on registers and add no precondition.
        unsafe {
            let bf16_data = _mm_loadu_si128(mem_addr as *const __m128i);
            let extended = _mm256_cvtepu16_epi32(bf16_data);
            let shifted = _mm256_slli_epi32(extended, 16);
            _mm256_castsi256_ps(shifted)
        }
    }

    #[cfg(not(target_feature = "f16c"))]
    unsafe fn load(mem_addr: *const bf16) -> Self::Unit {
        let mut tmp = [0.0f32; 8];
        // SAFETY: `avx` available, and the contract requires `mem_addr` to point
        // to 8 readable, contiguous `bf16`s — so `add(i)` for `i < 8` stays in
        // bounds and each dereference is valid. `tmp` is a local 8-`f32` array.
        unsafe {
            for i in 0..8 {
                tmp[i] = (*mem_addr.add(i)).to_f32();
            }
            _mm256_loadu_ps(tmp.as_ptr())
        }
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        // SAFETY: `avx` available (the `unsafe fn` contract); no further precondition.
        unsafe { _mm256_add_ps(a, b) }
    }

    #[cfg(target_feature = "fma")]
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        // SAFETY: `fma` available — this arm is `cfg`-gated on it and the
        // `unsafe fn` contract carries it.
        unsafe { _mm256_fmadd_ps(b, c, a) }
    }

    #[cfg(not(target_feature = "fma"))]
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        // SAFETY: `avx` available (the `unsafe fn` contract). Both operations
        // share that one obligation.
        unsafe { _mm256_add_ps(_mm256_mul_ps(b, c), a) }
    }

    unsafe fn vec_store(mem_addr: *mut bf16, a: Self::Unit) {
        let mut tmp = [0.0f32; 8];
        // SAFETY: `avx` available, and the contract requires `mem_addr` to point
        // to 8 writable, contiguous `bf16`s — so `add(i)` for `i < 8` stays in
        // bounds and each store is valid. `tmp` is a local 8-`f32` array. The
        // stores' dereferences are assignment LHS and cannot carry their own
        // blocks.
        unsafe {
            _mm256_storeu_ps(tmp.as_mut_ptr(), a);
            for (i, &v) in tmp.iter().enumerate() {
                *mem_addr.add(i) = bf16::from_f32(v);
            }
        }
    }

    unsafe fn vec_reduce(mut x: Self::Array, y: *mut f32) {
        // SAFETY: `avx`/`sse3` available (the `unsafe fn` contract), and `y`
        // points to one writable `f32`. Every operation below shares those two
        // obligations, and the `*y` store's dereference is an assignment LHS
        // that cannot carry its own block.
        unsafe {
            let mut offset = ARR >> 1;
            for i in 0..offset {
                x[i] = _mm256_add_ps(x[i], x[offset + i]);
            }
            offset >>= 1;
            for i in 0..offset {
                x[i] = _mm256_add_ps(x[i], x[offset + i]);
            }
            offset >>= 1;
            for i in 0..offset {
                x[i] = _mm256_add_ps(x[i], x[offset + i]);
            }
            let t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]), _mm256_extractf128_ps(x[0], 1));
            let t1 = _mm_hadd_ps(t0, t0);
            *y = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
        }
    }
}
