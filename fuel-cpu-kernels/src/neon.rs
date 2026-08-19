use super::Cpu;
#[cfg(target_arch = "arm")]
use core::arch::arm::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

pub struct CurrentCpu {}

const STEP: usize = 16;
const EPR: usize = 4;
const ARR: usize = STEP / EPR;

impl CurrentCpu {
    #[cfg(target_arch = "aarch64")]
    unsafe fn reduce_one(x: float32x4_t) -> f32 {
        // SAFETY: the `unsafe fn` contract guarantees the `neon` target feature is
        // available; `vaddvq_f32` (horizontal add) has no further precondition.
        unsafe { vaddvq_f32(x) }
    }

    #[cfg(target_arch = "arm")]
    unsafe fn reduce_one(x: float32x4_t) -> f32 {
        // SAFETY: the `unsafe fn` contract guarantees the `neon` target feature is
        // available. All four lane reads share that single obligation, so one
        // block documents it once — wrapping each `vgetq_lane_f32` individually
        // would repeat the same note four times inside one expression.
        unsafe {
            vgetq_lane_f32(x, 0)
                + vgetq_lane_f32(x, 1)
                + vgetq_lane_f32(x, 2)
                + vgetq_lane_f32(x, 3)
        }
    }
}

impl Cpu<ARR> for CurrentCpu {
    type Unit = float32x4_t;
    type Array = [float32x4_t; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;

    fn n() -> usize {
        ARR
    }

    unsafe fn zero() -> Self::Unit {
        // SAFETY: `neon` target feature available (the `unsafe fn` contract).
        unsafe { vdupq_n_f32(0.0) }
    }

    unsafe fn from_f32(x: f32) -> Self::Unit {
        // SAFETY: `neon` target feature available (the `unsafe fn` contract).
        unsafe { vdupq_n_f32(x) }
    }

    unsafe fn zero_array() -> Self::Array {
        // SAFETY: `Self::zero` upholds the same `neon` target-feature contract.
        unsafe { [Self::zero(); ARR] }
    }

    unsafe fn load(mem_addr: *const f32) -> Self::Unit {
        // SAFETY: the `unsafe fn` contract guarantees `neon` is available and that
        // `mem_addr` points to 4 readable, contiguous `f32`s.
        unsafe { vld1q_f32(mem_addr) }
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        // SAFETY: `neon` target feature available (the `unsafe fn` contract).
        unsafe { vaddq_f32(a, b) }
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        // SAFETY: `neon` target feature available (the `unsafe fn` contract).
        unsafe { vfmaq_f32(a, b, c) }
    }

    unsafe fn vec_store(mem_addr: *mut f32, a: Self::Unit) {
        // SAFETY: the `unsafe fn` contract guarantees `neon` is available and that
        // `mem_addr` points to 4 writable, contiguous `f32`s.
        unsafe { vst1q_f32(mem_addr, a) };
    }

    unsafe fn vec_reduce(mut x: Self::Array, y: *mut f32) {
        // SAFETY: the `unsafe fn` contract guarantees `neon` is available and that
        // `y` points to a writable `f32`. Every operation below — the pairwise
        // adds, the horizontal reduce, and the `*y` store — rests on that one
        // obligation, so a single documented block states it once. The `*y = …`
        // store's assignment-LHS dereference cannot be wrapped on its own anyway,
        // which is why per-operation blocking does not apply cleanly here.
        unsafe {
            for i in 0..ARR / 2 {
                x[2 * i] = vaddq_f32(x[2 * i], x[2 * i + 1]);
            }
            for i in 0..ARR / 4 {
                x[4 * i] = vaddq_f32(x[4 * i], x[4 * i + 2]);
            }
            *y = Self::reduce_one(x[0]);
        }
    }
}
