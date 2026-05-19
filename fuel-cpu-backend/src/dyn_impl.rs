//! `DynBackendStorage` and `DynBackendDevice` implementations for the CPU backend.
//!
//! `CpuStorage` (defined here) owns raw tensor data as a typed `HostBuffer` and
//! implements `DynBackendStorage` directly. `CpuBackendDevice` is the stateless
//! device handle. `CpuBackendStorage` is a backward-compat alias for `CpuStorage`.

use fuel_core_types::conv::{
    ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D,
};
use fuel_core_types::cpu::erf;
use fuel_core_types::dyn_backend::{DynBackendDevice, DynBackendStorage};
use fuel_core_types::op::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use fuel_core_types::{CpuStorage as HostBuffer, DType, DeviceLocation, Error, Layout, Result,
                         Scalar, Shape};
use float8::F8E4M3;
use half::{bf16, f16};
use num_traits::Float as _;
use std::any::Any;
use std::sync::Arc;

use crate::utils::{unary_map, binary_map, Map1, Map1Any, Map2,
                   Map2U8, Map2InPlace};

// ---------------------------------------------------------------------------
// CpuStorage — newtype wrapping HostBuffer
// ---------------------------------------------------------------------------

/// CPU backend storage: owns raw tensor data as a typed `HostBuffer`.
///
/// Defined in `fuel-cpu-backend` so the orphan rule allows implementing
/// `DynBackendStorage` here with full access to CPU kernels.
#[derive(Debug, Clone)]
pub struct CpuStorage(pub HostBuffer);

/// Backward-compat alias.
pub type CpuBackendStorage = CpuStorage;

impl CpuStorage {
    /// Unwrap the inner `HostBuffer`.
    pub fn into_inner(self) -> HostBuffer {
        self.0
    }

    /// Borrow the inner `HostBuffer`.
    pub fn inner(&self) -> &HostBuffer {
        &self.0
    }
}

impl From<HostBuffer> for CpuStorage {
    fn from(s: HostBuffer) -> Self {
        Self(s)
    }
}

impl From<CpuStorage> for HostBuffer {
    fn from(s: CpuStorage) -> Self {
        s.0
    }
}

impl fuel_core_types::backend::HostStorage for CpuStorage {
    fn as_host_buffer_ref(
        &self,
    ) -> fuel_core_types::Result<fuel_core_types::HostBufferRef<'_>> {
        Ok(self.0.as_ref())
    }

    // Override the default to hand out the existing `Vec<T>` without
    // a copy — we own the buffer outright.
    fn into_host_buffer(self) -> fuel_core_types::Result<fuel_core_types::HostBuffer> {
        Ok(self.0)
    }
}

// ---------------------------------------------------------------------------
// CpuBackendDevice — newtype wrapper
// ---------------------------------------------------------------------------

/// CPU device handle (stateless) implementing [`DynBackendDevice`].
#[derive(Debug, Clone, Copy)]
pub struct CpuBackendDevice;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Downcast a `&dyn DynBackendStorage` to `&CpuStorage`.
fn downcast(s: &dyn DynBackendStorage) -> Result<&CpuStorage> {
    s.as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::DeviceMismatchBinaryOp {
            lhs: DeviceLocation::Cpu,
            rhs: s.device_dyn().location_dyn(),
            op: "dyn_backend",
        }.bt())
}

/// Downcast a `&mut dyn DynBackendStorage` to `&mut CpuStorage`.
fn downcast_mut(s: &mut dyn DynBackendStorage) -> Result<&mut CpuStorage> {
    let loc = s.device_dyn().location_dyn();
    s.as_any_mut()
        .downcast_mut::<CpuStorage>()
        .ok_or_else(|| Error::DeviceMismatchBinaryOp {
            lhs: DeviceLocation::Cpu,
            rhs: loc,
            op: "dyn_backend",
        }.bt())
}

// ---------------------------------------------------------------------------
// Unary / binary dispatch helpers
// ---------------------------------------------------------------------------

/// Helper: apply a per-element unary operation selected by [`UnaryOp`].
fn cpu_unary_op(s: &HostBuffer, layout: &Layout, op: UnaryOp) -> Result<HostBuffer> {
    use UnaryOp::*;

    // Most unary ops only make sense on floating types; for integer types they
    // either return the identity or should error.  We first handle the full
    // float-path ops generically, then special-case per variant when needed.

    match op {
        Exp => float_unary(s, layout, |v: f32| v.exp(), |v: f64| v.exp()),
        Log => float_unary(s, layout, |v: f32| v.ln(), |v: f64| v.ln()),
        Sin => float_unary(s, layout, |v: f32| v.sin(), |v: f64| v.sin()),
        Cos => float_unary(s, layout, |v: f32| v.cos(), |v: f64| v.cos()),
        Tanh => float_unary(s, layout, |v: f32| v.tanh(), |v: f64| v.tanh()),
        Neg => all_unary_neg(s, layout),
        Recip => float_unary(s, layout, |v: f32| v.recip(), |v: f64| v.recip()),
        Sqr => all_unary_sqr(s, layout),
        Sqrt => float_unary(s, layout, |v: f32| v.sqrt(), |v: f64| v.sqrt()),
        Abs => all_unary_abs(s, layout),
        Relu => all_unary_relu(s, layout),
        Floor => float_unary_identity_int(s, layout, |v: f32| v.floor(), |v: f64| v.floor()),
        Ceil => float_unary_identity_int(s, layout, |v: f32| v.ceil(), |v: f64| v.ceil()),
        Round => float_unary_identity_int(s, layout, |v: f32| v.round(), |v: f64| v.round()),
        Sign => all_unary_sign(s, layout),
        Gelu => float_unary(s, layout, gelu_f32, gelu_f64),
        GeluErf => float_unary(s, layout, gelu_erf_f32, gelu_erf_f64),
        Erf => float_unary(s, layout, erf::erf_f32, erf::erf_f64),
        Silu => float_unary(s, layout, silu_f32, silu_f64),
    }
}

#[allow(clippy::excessive_precision)]
fn gelu_f32(v: f32) -> f32 {
    0.5 * v * (1.0 + f32::tanh(0.79788456080286535587989211986876373 * v * (1.0 + 0.044715 * v * v)))
}

#[allow(clippy::excessive_precision)]
fn gelu_f64(v: f64) -> f64 {
    0.5 * v * (1.0 + f64::tanh(0.79788456080286535587989211986876373 * v * (1.0 + 0.044715 * v * v)))
}

fn gelu_erf_f32(v: f32) -> f32 {
    (erf::erf_f32(v * std::f32::consts::FRAC_1_SQRT_2) + 1.) * 0.5 * v
}

fn gelu_erf_f64(v: f64) -> f64 {
    (erf::erf_f64(v * std::f64::consts::FRAC_1_SQRT_2) + 1.) * 0.5 * v
}

fn silu_f32(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn silu_f64(v: f64) -> f64 {
    v / (1.0 + (-v).exp())
}

/// Apply a float-only unary op.  For half/f8 types, promote to f32/f64 then demote.
/// Integer types return an error.
fn float_unary(
    s: &HostBuffer,
    layout: &Layout,
    f32_fn: impl Fn(f32) -> f32 + Copy,
    f64_fn: impl Fn(f64) -> f64 + Copy,
) -> Result<HostBuffer> {
    match s {
        HostBuffer::BF16(data) => {
            let out = unary_map(data, layout, |v: bf16| bf16::from_f32(f32_fn(v.to_f32())));
            Ok(HostBuffer::BF16(out))
        }
        HostBuffer::F16(data) => {
            let out = unary_map(data, layout, |v: f16| f16::from_f32(f32_fn(v.to_f32())));
            Ok(HostBuffer::F16(out))
        }
        HostBuffer::F32(data) => {
            let out = unary_map(data, layout, f32_fn);
            Ok(HostBuffer::F32(out))
        }
        HostBuffer::F64(data) => {
            let out = unary_map(data, layout, f64_fn);
            Ok(HostBuffer::F64(out))
        }
        HostBuffer::F8E4M3(data) => {
            let out = unary_map(data, layout, |v: F8E4M3| F8E4M3::from_f32(f32_fn(v.to_f32())));
            Ok(HostBuffer::F8E4M3(out))
        }
        other => Err(Error::UnsupportedDTypeForOp(other.dtype(), "unary_op").bt()),
    }
}

/// Apply a float unary op; integer types pass through as identity.
fn float_unary_identity_int(
    s: &HostBuffer,
    layout: &Layout,
    f32_fn: impl Fn(f32) -> f32 + Copy,
    f64_fn: impl Fn(f64) -> f64 + Copy,
) -> Result<HostBuffer> {
    match s {
        HostBuffer::U8(d) => Ok(HostBuffer::U8(unary_map(d, layout, |v| v))),
        HostBuffer::U32(d) => Ok(HostBuffer::U32(unary_map(d, layout, |v| v))),
        HostBuffer::I16(d) => Ok(HostBuffer::I16(unary_map(d, layout, |v| v))),
        HostBuffer::I32(d) => Ok(HostBuffer::I32(unary_map(d, layout, |v| v))),
        HostBuffer::I64(d) => Ok(HostBuffer::I64(unary_map(d, layout, |v| v))),
        _ => float_unary(s, layout, f32_fn, f64_fn),
    }
}

fn all_unary_neg(s: &HostBuffer, layout: &Layout) -> Result<HostBuffer> {
    match s {
        HostBuffer::BF16(d) => Ok(HostBuffer::BF16(unary_map(d, layout, |v: bf16| -v))),
        HostBuffer::F16(d) => Ok(HostBuffer::F16(unary_map(d, layout, |v: f16| -v))),
        HostBuffer::F32(d) => Ok(HostBuffer::F32(unary_map(d, layout, |v: f32| -v))),
        HostBuffer::F64(d) => Ok(HostBuffer::F64(unary_map(d, layout, |v: f64| -v))),
        HostBuffer::I16(d) => Ok(HostBuffer::I16(unary_map(d, layout, |v: i16| -v))),
        HostBuffer::I32(d) => Ok(HostBuffer::I32(unary_map(d, layout, |v: i32| -v))),
        HostBuffer::I64(d) => Ok(HostBuffer::I64(unary_map(d, layout, |v: i64| -v))),
        HostBuffer::F8E4M3(d) => Ok(HostBuffer::F8E4M3(unary_map(d, layout, |v: F8E4M3| -v))),
        other => Err(Error::UnsupportedDTypeForOp(other.dtype(), "neg").bt()),
    }
}

fn all_unary_sqr(s: &HostBuffer, layout: &Layout) -> Result<HostBuffer> {
    match s {
        HostBuffer::BF16(d) => Ok(HostBuffer::BF16(unary_map(d, layout, |v: bf16| v * v))),
        HostBuffer::F16(d) => Ok(HostBuffer::F16(unary_map(d, layout, |v: f16| v * v))),
        HostBuffer::F32(d) => Ok(HostBuffer::F32(unary_map(d, layout, |v: f32| v * v))),
        HostBuffer::F64(d) => Ok(HostBuffer::F64(unary_map(d, layout, |v: f64| v * v))),
        HostBuffer::U8(d) => Ok(HostBuffer::U8(unary_map(d, layout, |v: u8| v * v))),
        HostBuffer::U32(d) => Ok(HostBuffer::U32(unary_map(d, layout, |v: u32| v * v))),
        HostBuffer::I16(d) => Ok(HostBuffer::I16(unary_map(d, layout, |v: i16| v * v))),
        HostBuffer::I32(d) => Ok(HostBuffer::I32(unary_map(d, layout, |v: i32| v * v))),
        HostBuffer::I64(d) => Ok(HostBuffer::I64(unary_map(d, layout, |v: i64| v * v))),
        HostBuffer::F8E4M3(d) => Ok(HostBuffer::F8E4M3(unary_map(d, layout, |v: F8E4M3| v * v))),
        other => Err(Error::UnsupportedDTypeForOp(other.dtype(), "sqr").bt()),
    }
}

fn all_unary_abs(s: &HostBuffer, layout: &Layout) -> Result<HostBuffer> {
    match s {
        HostBuffer::BF16(d) => Ok(HostBuffer::BF16(unary_map(d, layout, |v: bf16| v.abs()))),
        HostBuffer::F16(d) => Ok(HostBuffer::F16(unary_map(d, layout, |v: f16| v.abs()))),
        HostBuffer::F32(d) => Ok(HostBuffer::F32(unary_map(d, layout, |v: f32| v.abs()))),
        HostBuffer::F64(d) => Ok(HostBuffer::F64(unary_map(d, layout, |v: f64| v.abs()))),
        HostBuffer::U8(d) => Ok(HostBuffer::U8(unary_map(d, layout, |v: u8| v))),
        HostBuffer::U32(d) => Ok(HostBuffer::U32(unary_map(d, layout, |v: u32| v))),
        HostBuffer::I16(d) => Ok(HostBuffer::I16(unary_map(d, layout, |v: i16| v.abs()))),
        HostBuffer::I32(d) => Ok(HostBuffer::I32(unary_map(d, layout, |v: i32| v.abs()))),
        HostBuffer::I64(d) => Ok(HostBuffer::I64(unary_map(d, layout, |v: i64| v.abs()))),
        HostBuffer::F8E4M3(d) => Ok(HostBuffer::F8E4M3(unary_map(d, layout, |v: F8E4M3| v.abs()))),
        other => Err(Error::UnsupportedDTypeForOp(other.dtype(), "abs").bt()),
    }
}

fn all_unary_relu(s: &HostBuffer, layout: &Layout) -> Result<HostBuffer> {
    match s {
        HostBuffer::BF16(d) => Ok(HostBuffer::BF16(unary_map(d, layout, |v: bf16| v.max(bf16::ZERO)))),
        HostBuffer::F16(d) => Ok(HostBuffer::F16(unary_map(d, layout, |v: f16| v.max(f16::ZERO)))),
        HostBuffer::F32(d) => Ok(HostBuffer::F32(unary_map(d, layout, |v: f32| v.max(0.0)))),
        HostBuffer::F64(d) => Ok(HostBuffer::F64(unary_map(d, layout, |v: f64| v.max(0.0)))),
        HostBuffer::U8(d) => Ok(HostBuffer::U8(unary_map(d, layout, |v: u8| v))),
        HostBuffer::U32(d) => Ok(HostBuffer::U32(unary_map(d, layout, |v: u32| v))),
        HostBuffer::I16(d) => Ok(HostBuffer::I16(unary_map(d, layout, |v: i16| v.max(0)))),
        HostBuffer::I32(d) => Ok(HostBuffer::I32(unary_map(d, layout, |v: i32| v.max(0)))),
        HostBuffer::I64(d) => Ok(HostBuffer::I64(unary_map(d, layout, |v: i64| v.max(0)))),
        HostBuffer::F8E4M3(d) => Ok(HostBuffer::F8E4M3(unary_map(d, layout, |v: F8E4M3| v.max(F8E4M3::ZERO)))),
        other => Err(Error::UnsupportedDTypeForOp(other.dtype(), "relu").bt()),
    }
}

fn all_unary_sign(s: &HostBuffer, layout: &Layout) -> Result<HostBuffer> {
    match s {
        HostBuffer::BF16(d) => Ok(HostBuffer::BF16(unary_map(d, layout, |v: bf16| {
            bf16::from((v > bf16::ZERO) as i8) - bf16::from((v < bf16::ZERO) as i8)
        }))),
        HostBuffer::F16(d) => Ok(HostBuffer::F16(unary_map(d, layout, |v: f16| {
            f16::from((v > f16::ZERO) as i8) - f16::from((v < f16::ZERO) as i8)
        }))),
        HostBuffer::F32(d) => Ok(HostBuffer::F32(unary_map(d, layout, |v: f32| {
            f32::from(v > 0.) - f32::from(v < 0.)
        }))),
        HostBuffer::F64(d) => Ok(HostBuffer::F64(unary_map(d, layout, |v: f64| {
            f64::from(v > 0.) - f64::from(v < 0.)
        }))),
        HostBuffer::I16(d) => Ok(HostBuffer::I16(unary_map(d, layout, |v: i16| v.signum()))),
        HostBuffer::I32(d) => Ok(HostBuffer::I32(unary_map(d, layout, |v: i32| v.signum()))),
        HostBuffer::I64(d) => Ok(HostBuffer::I64(unary_map(d, layout, |v: i64| v.signum()))),
        HostBuffer::F8E4M3(d) => Ok(HostBuffer::F8E4M3(unary_map(d, layout, |v: F8E4M3| {
            let f = v.to_f32();
            F8E4M3::from_f32(f32::from(f > 0.) - f32::from(f < 0.))
        }))),
        other => Err(Error::UnsupportedDTypeForOp(other.dtype(), "sign").bt()),
    }
}

/// Apply a binary op selected by [`BinaryOp`].
fn cpu_binary_op(
    lhs: &HostBuffer,
    rhs: &HostBuffer,
    lhs_l: &Layout,
    rhs_l: &Layout,
    op: BinaryOp,
) -> Result<HostBuffer> {
    use BinaryOp::*;
    match op {
        Add => all_binary(lhs, rhs, lhs_l, rhs_l, |a, b| a + b, "add"),
        Sub => all_binary(lhs, rhs, lhs_l, rhs_l, |a, b| a - b, "sub"),
        Mul => all_binary(lhs, rhs, lhs_l, rhs_l, |a, b| a * b, "mul"),
        Div => all_binary(lhs, rhs, lhs_l, rhs_l, |a, b| a / b, "div"),
        Maximum => all_binary(lhs, rhs, lhs_l, rhs_l, |a, b| if a < b { b } else { a }, "maximum"),
        Minimum => all_binary(lhs, rhs, lhs_l, rhs_l, |a, b| if a > b { b } else { a }, "minimum"),
    }
}

fn all_binary<F>(
    lhs: &HostBuffer,
    rhs: &HostBuffer,
    lhs_l: &Layout,
    rhs_l: &Layout,
    f: F,
    op_name: &'static str,
) -> Result<HostBuffer>
where
    F: Fn(f64, f64) -> f64 + Copy,
{
    // Dispatch on matching dtype pairs.  Uses binary_map from utils.
    macro_rules! dispatch_pair {
        ($lhs_v:ident, $rhs_v:ident, $variant:ident, $conv_fn:expr, $result_fn:expr) => {{
            let out = binary_map(lhs_l, rhs_l, $lhs_v, $rhs_v, |a, b| {
                let result = f($conv_fn(a), $conv_fn(b));
                $result_fn(result)
            });
            Ok(HostBuffer::$variant(out))
        }};
    }

    match (lhs, rhs) {
        (HostBuffer::BF16(a), HostBuffer::BF16(b)) => dispatch_pair!(a, b, BF16, |v: bf16| v.to_f64(), |v: f64| bf16::from_f64(v)),
        (HostBuffer::F16(a), HostBuffer::F16(b)) => dispatch_pair!(a, b, F16, |v: f16| v.to_f64(), |v: f64| f16::from_f64(v)),
        (HostBuffer::F32(a), HostBuffer::F32(b)) => {
            let out = binary_map(lhs_l, rhs_l, a, b, |a, b| f(a as f64, b as f64) as f32);
            Ok(HostBuffer::F32(out))
        }
        (HostBuffer::F64(a), HostBuffer::F64(b)) => {
            let out = binary_map(lhs_l, rhs_l, a, b, |a, b| f(a, b));
            Ok(HostBuffer::F64(out))
        }
        (HostBuffer::U8(a), HostBuffer::U8(b)) => {
            let out = binary_map(lhs_l, rhs_l, a, b, |a, b| f(a as f64, b as f64) as u8);
            Ok(HostBuffer::U8(out))
        }
        (HostBuffer::U32(a), HostBuffer::U32(b)) => {
            let out = binary_map(lhs_l, rhs_l, a, b, |a, b| f(a as f64, b as f64) as u32);
            Ok(HostBuffer::U32(out))
        }
        (HostBuffer::I16(a), HostBuffer::I16(b)) => {
            let out = binary_map(lhs_l, rhs_l, a, b, |a, b| f(a as f64, b as f64) as i16);
            Ok(HostBuffer::I16(out))
        }
        (HostBuffer::I32(a), HostBuffer::I32(b)) => {
            let out = binary_map(lhs_l, rhs_l, a, b, |a, b| f(a as f64, b as f64) as i32);
            Ok(HostBuffer::I32(out))
        }
        (HostBuffer::I64(a), HostBuffer::I64(b)) => {
            let out = binary_map(lhs_l, rhs_l, a, b, |a, b| f(a as f64, b as f64) as i64);
            Ok(HostBuffer::I64(out))
        }
        (HostBuffer::F8E4M3(a), HostBuffer::F8E4M3(b)) => dispatch_pair!(a, b, F8E4M3, |v: F8E4M3| v.to_f64(), |v: f64| F8E4M3::from_f64(v)),
        _ => Err(Error::DTypeMismatchBinaryOp {
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
            op: op_name,
        }.bt()),
    }
}

// ---------------------------------------------------------------------------
// Copy helpers (same as in fuel-core/src/cpu_backend/mod.rs)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn copy2d_<T: Copy>(
    src: &[T],
    dst: &mut [T],
    d1: usize,
    d2: usize,
    src_stride1: usize,
    dst_stride1: usize,
    src_offset: usize,
    dst_offset: usize,
) {
    for i1 in 0..d1 {
        let dst_idx = i1 * dst_stride1 + dst_offset;
        let src_idx = i1 * src_stride1 + src_offset;
        dst[dst_idx..dst_idx + d2].copy_from_slice(&src[src_idx..src_idx + d2]);
    }
}

fn copy_strided_src_<T: Copy>(src: &[T], dst: &mut [T], dst_offset: usize, src_l: &Layout) {
    match src_l.strided_blocks() {
        fuel_core_types::StridedBlocks::SingleBlock { start_offset, len } => {
            let to_copy = (dst.len() - dst_offset).min(len);
            dst[dst_offset..dst_offset + to_copy]
                .copy_from_slice(&src[start_offset..start_offset + to_copy])
        }
        fuel_core_types::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len: 1,
        } => {
            for (dst_index, src_index) in block_start_index.enumerate() {
                let dst_index = dst_index + dst_offset;
                if dst_index >= dst.len() {
                    break;
                }
                dst[dst_index] = src[src_index];
            }
        }
        fuel_core_types::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            let mut dst_index = dst_offset;
            for src_index in block_start_index {
                let next_dst_index = dst_index + block_len;
                if dst_index >= dst.len() {
                    break;
                }
                let to_copy = usize::min(block_len, dst.len() - dst_index);
                dst[dst_index..dst_index + to_copy]
                    .copy_from_slice(&src[src_index..src_index + to_copy]);
                dst_index = next_dst_index;
            }
        }
    }
}

fn elu<T: num_traits::Float>(v: T, alpha: T) -> T {
    if v.is_sign_positive() {
        v
    } else {
        (v.exp() - T::one()) * alpha
    }
}

// ---------------------------------------------------------------------------
// impl DynBackendStorage for CpuStorage
// ---------------------------------------------------------------------------

impl DynBackendStorage for CpuStorage {
    fn try_clone_dyn(&self, _layout: &Layout) -> Result<Box<dyn DynBackendStorage>> {
        Ok(Box::new(self.clone()))
    }

    fn dtype_dyn(&self) -> DType {
        self.0.dtype()
    }

    fn device_dyn(&self) -> &dyn DynBackendDevice {
        // We can return a static reference because CpuBackendDevice is stateless.
        &CpuBackendDevice
    }

    fn device_arc_dyn(&self) -> Arc<dyn DynBackendDevice> {
        Arc::new(CpuBackendDevice)
    }

    fn to_host_buffer_dyn(&self) -> Result<HostBuffer> {
        Ok(self.0.clone())
    }

    fn affine_dyn(
        &self,
        layout: &Layout,
        mul: f64,
        add: f64,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let result = Map1::map(&crate::ops::Affine(mul, add), &self.0, layout)?;
        Ok(Box::new(CpuStorage(result)))
    }

    fn powf_dyn(&self, layout: &Layout, e: f64) -> Result<Box<dyn DynBackendStorage>> {
        use num_traits::Float;
        let result = match &self.0 {
            HostBuffer::BF16(d) => HostBuffer::BF16(unary_map(d, layout, |v: bf16| v.powf(bf16::from_f64(e)))),
            HostBuffer::F16(d) => HostBuffer::F16(unary_map(d, layout, |v: f16| v.powf(f16::from_f64(e)))),
            HostBuffer::F32(d) => HostBuffer::F32(unary_map(d, layout, |v: f32| v.powf(e as f32))),
            HostBuffer::F64(d) => HostBuffer::F64(unary_map(d, layout, |v: f64| v.powf(e))),
            HostBuffer::F8E4M3(d) => HostBuffer::F8E4M3(unary_map(d, layout, |v: F8E4M3| v.powf(F8E4M3::from_f64(e)))),
            other => return Err(Error::UnsupportedDTypeForOp(other.dtype(), "powf").bt()),
        };
        Ok(Box::new(CpuStorage(result)))
    }

    fn elu_dyn(&self, layout: &Layout, alpha: f64) -> Result<Box<dyn DynBackendStorage>> {
        let result = match &self.0 {
            HostBuffer::BF16(d) => HostBuffer::BF16(unary_map(d, layout, |v| elu(v, bf16::from_f64(alpha)))),
            HostBuffer::F16(d) => HostBuffer::F16(unary_map(d, layout, |v| elu(v, f16::from_f64(alpha)))),
            HostBuffer::F32(d) => HostBuffer::F32(unary_map(d, layout, |v| elu(v, alpha as f32))),
            HostBuffer::F64(d) => HostBuffer::F64(unary_map(d, layout, |v| elu(v, alpha))),
            HostBuffer::F8E4M3(d) => HostBuffer::F8E4M3(unary_map(d, layout, |v| elu(v, F8E4M3::from_f64(alpha)))),
            other => return Err(Error::UnsupportedDTypeForOp(other.dtype(), "elu").bt()),
        };
        Ok(Box::new(CpuStorage(result)))
    }

    fn reduce_op_dyn(
        &self,
        op: ReduceOp,
        layout: &Layout,
        axes: &[usize],
    ) -> Result<Box<dyn DynBackendStorage>> {
        let result = match op {
            ReduceOp::Sum => {
                let src_dims = layout.dims();
                let mut dst_dims = src_dims.to_vec();
                for &dim in axes.iter() {
                    dst_dims[dim] = 1;
                }
                let dst_shape = Shape::from(dst_dims);
                let mut reduce_dims = axes.to_vec();
                reduce_dims.sort();
                let reduce_dims_and_stride: Vec<_> = reduce_dims
                    .iter()
                    .map(|&d| (src_dims[d], src_dims[d + 1..].iter().product::<usize>()))
                    .collect();
                Map1::map(
                    &crate::ops::ReduceSum {
                        dst_shape: &dst_shape,
                        reduce_dims: &reduce_dims,
                        reduce_dims_and_stride,
                    },
                    &self.0,
                    layout,
                )?
            }
            ReduceOp::Min | ReduceOp::ArgMin | ReduceOp::Max | ReduceOp::ArgMax => {
                let reduce_dim_index = match axes {
                    [dim] => *dim,
                    _ => {
                        let op_name = match op {
                            ReduceOp::Min => "min",
                            ReduceOp::ArgMin => "argmin",
                            ReduceOp::Max => "max",
                            ReduceOp::ArgMax => "argmax",
                            _ => unreachable!(),
                        };
                        return Err(Error::OnlySingleDimension {
                            op: op_name,
                            dims: axes.to_vec(),
                        }.bt());
                    }
                };
                let (use_min, return_index) = match op {
                    ReduceOp::Min => (true, false),
                    ReduceOp::ArgMin => (true, true),
                    ReduceOp::Max => (false, false),
                    ReduceOp::ArgMax => (false, true),
                    _ => unreachable!(),
                };
                Map1Any::map(
                    &crate::ops::ReduceIndex {
                        reduce_dim_index,
                        use_min,
                        return_index,
                    },
                    &self.0,
                    layout,
                )?
            }
        };
        Ok(Box::new(CpuStorage(result)))
    }

    fn cmp_dyn(
        &self,
        op: CmpOp,
        rhs: &dyn DynBackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let rhs = downcast(rhs)?;
        let result = Map2U8::map(&crate::ops::Cmp(op), &self.0, lhs_layout, &rhs.0, rhs_layout)?;
        Ok(Box::new(CpuStorage(result)))
    }

    fn to_dtype_dyn(&self, layout: &Layout, dtype: DType) -> Result<Box<dyn DynBackendStorage>> {
        // Delegate to the existing to_dtype machinery.
        // This is one of the complex methods with O(dtypes^2) match arms.
        // For now, go through CPU storage → convert via unary_map.
        //
        // We replicate the approach from fuel-core's BackendStorage::to_dtype:
        // match (self.dtype(), target) and use unary_map for each conversion.
        let result = cpu_to_dtype(&self.0, layout, dtype)?;
        Ok(Box::new(CpuStorage(result)))
    }

    fn unary_op_dyn(
        &self,
        layout: &Layout,
        op: UnaryOp,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let result = cpu_unary_op(&self.0, layout, op)?;
        Ok(Box::new(CpuStorage(result)))
    }

    fn binary_op_dyn(
        &self,
        rhs: &dyn DynBackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: BinaryOp,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let rhs = downcast(rhs)?;
        let result = cpu_binary_op(&self.0, &rhs.0, lhs_layout, rhs_layout, op)?;
        Ok(Box::new(CpuStorage(result)))
    }

    fn where_cond_dyn(
        &self,
        cond_layout: &Layout,
        on_true: &dyn DynBackendStorage,
        on_true_layout: &Layout,
        on_false: &dyn DynBackendStorage,
        on_false_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let t = downcast(on_true)?;
        let f = downcast(on_false)?;
        let result = match &self.0 {
            HostBuffer::U8(pred) => Map2::map(&crate::ops::WCond(pred, cond_layout), &t.0, on_true_layout, &f.0, on_false_layout)?,
            HostBuffer::U32(pred) => Map2::map(&crate::ops::WCond(pred, cond_layout), &t.0, on_true_layout, &f.0, on_false_layout)?,
            HostBuffer::I16(pred) => Map2::map(&crate::ops::WCond(pred, cond_layout), &t.0, on_true_layout, &f.0, on_false_layout)?,
            HostBuffer::I32(pred) => Map2::map(&crate::ops::WCond(pred, cond_layout), &t.0, on_true_layout, &f.0, on_false_layout)?,
            HostBuffer::I64(pred) => Map2::map(&crate::ops::WCond(pred, cond_layout), &t.0, on_true_layout, &f.0, on_false_layout)?,
            _ => return Err(Error::UnsupportedDTypeForOp(self.0.dtype(), "where-cond").bt()),
        };
        Ok(Box::new(CpuStorage(result)))
    }

    fn conv1d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let kernel = downcast(kernel)?;
        // Use the im2col approach for conv1d (same as BackendStorage impl).
        let l_k = params.k_size;
        let col = Map1::map(
            &crate::ops::Im2Col1D {
                l_k,
                padding: params.padding,
                stride: params.stride,
                dilation: params.dilation,
            },
            &self.0,
            l,
        )?;
        let b = params.b_size;
        let n = params.c_out;
        let l_out = params.l_out();
        let k = l_k * params.c_in;
        let m = l_out;
        let col_l = Layout::contiguous((b, m, k));
        let kernel_l_c = if kernel_l.is_contiguous() {
            Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?
        } else {
            // Make kernel contiguous
            let mut kernel_c = unsafe { cpu_alloc_uninit(kernel_l.shape(), kernel.0.dtype())? };
            cpu_copy_strided_src(&kernel.0, &mut kernel_c, 0, kernel_l)?;
            let new_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            // Also update the kernel ref
            return {
                let res = Map2::map(&crate::ops::MatMul((b, m, n, k)), &col, &col_l, &kernel_c, &new_l)?;
                let res_l = Layout::contiguous((b, l_out, params.c_out)).transpose(1, 2)?;
                let mut res_t = unsafe { cpu_alloc_uninit(res_l.shape(), res.dtype())? };
                cpu_copy_strided_src(&res, &mut res_t, 0, &res_l)?;
                Ok(Box::new(CpuStorage(res_t)))
            };
        };
        let res = Map2::map(&crate::ops::MatMul((b, m, n, k)), &col, &col_l, &kernel.0, &kernel_l_c)?;
        let res_l = Layout::contiguous((b, l_out, params.c_out)).transpose(1, 2)?;
        let mut res_t = unsafe { cpu_alloc_uninit(res_l.shape(), res.dtype())? };
        cpu_copy_strided_src(&res, &mut res_t, 0, &res_l)?;
        Ok(Box::new(CpuStorage(res_t)))
    }

    fn conv_transpose1d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let kernel = downcast(kernel)?;
        let result = Map2::map(
            &crate::ops::ConvTranspose1D(params),
            &self.0,
            l,
            &kernel.0,
            kernel_l,
        )?;
        Ok(Box::new(CpuStorage(result)))
    }

    fn conv2d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let kernel = downcast(kernel)?;
        let result = Map2::map(
            &crate::conv2d::Conv2D(params),
            &self.0,
            l,
            &kernel.0,
            kernel_l,
        )?;
        Ok(Box::new(CpuStorage(result)))
    }

    fn conv_transpose2d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let kernel = downcast(kernel)?;
        let result = Map2::map(
            &crate::ops::ConvTranspose2D(params),
            &self.0,
            l,
            &kernel.0,
            kernel_l,
        )?;
        Ok(Box::new(CpuStorage(result)))
    }

    fn avg_pool2d_dyn(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn DynBackendStorage>> {
        let result = Map1::map(&crate::ops::AvgPool2D(kernel, stride), &self.0, layout)?;
        Ok(Box::new(CpuStorage(result)))
    }

    fn max_pool2d_dyn(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn DynBackendStorage>> {
        let result = Map1::map(&crate::ops::MaxPool2D(kernel, stride), &self.0, layout)?;
        Ok(Box::new(CpuStorage(result)))
    }

    fn upsample_nearest1d_dyn(
        &self,
        layout: &Layout,
        target_size: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let result = Map1::map(&crate::ops::UpsampleNearest1D(target_size), &self.0, layout)?;
        Ok(Box::new(CpuStorage(result)))
    }

    fn upsample_nearest2d_dyn(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let result = Map1::map(&crate::ops::UpsampleNearest2D(target_h, target_w), &self.0, layout)?;
        Ok(Box::new(CpuStorage(result)))
    }

    fn upsample_bilinear2d_dyn(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let result = Map1::map(
            &crate::ops::UpsampleBilinear2D {
                target_h,
                target_w,
                align_corners,
                scale_h_factor: scale_h,
                scale_w_factor: scale_w,
            },
            &self.0,
            layout,
        )?;
        Ok(Box::new(CpuStorage(result)))
    }

    fn gather_dyn(
        &self,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let ids_s = downcast(ids)?;
        let result = match &ids_s.0 {
            HostBuffer::U8(ids) => Map1::map(&crate::ops::Gather { ids, ids_l: ids_layout, dim }, &self.0, src_layout)?,
            HostBuffer::U32(ids) => Map1::map(&crate::ops::Gather { ids, ids_l: ids_layout, dim }, &self.0, src_layout)?,
            HostBuffer::I64(ids) => Map1::map(&crate::ops::Gather { ids, ids_l: ids_layout, dim }, &self.0, src_layout)?,
            _ => return Err(Error::UnsupportedDTypeForOp(ids_s.0.dtype(), "gather").bt()),
        };
        Ok(Box::new(CpuStorage(result)))
    }

    fn scatter_set_dyn(
        &mut self,
        self_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        let ids_s = downcast(ids)?;
        let src_s = downcast(src)?;
        match &ids_s.0 {
            HostBuffer::U8(ids) => Map2InPlace::map(&crate::ops::Scatter::<_, crate::ops::Set>::new(ids, ids_layout, dim), &mut self.0, self_layout, &src_s.0, src_layout)?,
            HostBuffer::U32(ids) => Map2InPlace::map(&crate::ops::Scatter::<_, crate::ops::Set>::new(ids, ids_layout, dim), &mut self.0, self_layout, &src_s.0, src_layout)?,
            HostBuffer::I64(ids) => Map2InPlace::map(&crate::ops::Scatter::<_, crate::ops::Set>::new(ids, ids_layout, dim), &mut self.0, self_layout, &src_s.0, src_layout)?,
            _ => return Err(Error::UnsupportedDTypeForOp(ids_s.0.dtype(), "scatter").bt()),
        };
        Ok(())
    }

    fn scatter_add_set_dyn(
        &mut self,
        self_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        let ids_s = downcast(ids)?;
        let src_s = downcast(src)?;
        match &ids_s.0 {
            HostBuffer::U8(ids) => Map2InPlace::map(&crate::ops::Scatter::<_, crate::ops::Add>::new(ids, ids_layout, dim), &mut self.0, self_layout, &src_s.0, src_layout)?,
            HostBuffer::U32(ids) => Map2InPlace::map(&crate::ops::Scatter::<_, crate::ops::Add>::new(ids, ids_layout, dim), &mut self.0, self_layout, &src_s.0, src_layout)?,
            HostBuffer::I16(ids) => Map2InPlace::map(&crate::ops::Scatter::<_, crate::ops::Add>::new(ids, ids_layout, dim), &mut self.0, self_layout, &src_s.0, src_layout)?,
            HostBuffer::I32(ids) => Map2InPlace::map(&crate::ops::Scatter::<_, crate::ops::Add>::new(ids, ids_layout, dim), &mut self.0, self_layout, &src_s.0, src_layout)?,
            HostBuffer::I64(ids) => Map2InPlace::map(&crate::ops::Scatter::<_, crate::ops::Add>::new(ids, ids_layout, dim), &mut self.0, self_layout, &src_s.0, src_layout)?,
            _ => return Err(Error::UnsupportedDTypeForOp(ids_s.0.dtype(), "scatter-add").bt()),
        };
        Ok(())
    }

    fn index_select_dyn(
        &self,
        ids: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let ids_s = downcast(ids)?;
        let result = match &ids_s.0 {
            HostBuffer::U8(ids) => Map1::map(&crate::ops::IndexSelect { ids, ids_l: ids_layout, dim }, &self.0, src_layout)?,
            HostBuffer::U32(ids) => Map1::map(&crate::ops::IndexSelect { ids, ids_l: ids_layout, dim }, &self.0, src_layout)?,
            HostBuffer::I64(ids) => Map1::map(&crate::ops::IndexSelect { ids, ids_l: ids_layout, dim }, &self.0, src_layout)?,
            _ => return Err(Error::UnsupportedDTypeForOp(ids_s.0.dtype(), "index-select").bt()),
        };
        Ok(Box::new(CpuStorage(result)))
    }

    fn index_add_dyn(
        &self,
        self_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let ids_s = downcast(ids)?;
        let src_s = downcast(src)?;
        let result = match &ids_s.0 {
            HostBuffer::U8(ids) => {
                let ids = match ids_layout.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => return Err(Error::RequiresContiguous { op: "index-add" }.bt()),
                };
                Map2::map(&crate::ops::IndexAdd { ids, dim }, &self.0, self_layout, &src_s.0, src_layout)?
            }
            HostBuffer::U32(ids) => {
                let ids = match ids_layout.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => return Err(Error::RequiresContiguous { op: "index-add" }.bt()),
                };
                Map2::map(&crate::ops::IndexAdd { ids, dim }, &self.0, self_layout, &src_s.0, src_layout)?
            }
            HostBuffer::I16(ids) => {
                let ids = match ids_layout.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => return Err(Error::RequiresContiguous { op: "index-add" }.bt()),
                };
                Map2::map(&crate::ops::IndexAdd { ids, dim }, &self.0, self_layout, &src_s.0, src_layout)?
            }
            HostBuffer::I32(ids) => {
                let ids = match ids_layout.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => return Err(Error::RequiresContiguous { op: "index-add" }.bt()),
                };
                Map2::map(&crate::ops::IndexAdd { ids, dim }, &self.0, self_layout, &src_s.0, src_layout)?
            }
            HostBuffer::I64(ids) => {
                let ids = match ids_layout.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => return Err(Error::RequiresContiguous { op: "index-add" }.bt()),
                };
                Map2::map(&crate::ops::IndexAdd { ids, dim }, &self.0, self_layout, &src_s.0, src_layout)?
            }
            _ => return Err(Error::UnsupportedDTypeForOp(ids_s.0.dtype(), "index-add").bt()),
        };
        Ok(Box::new(CpuStorage(result)))
    }

    fn matmul_dyn(
        &self,
        rhs: &dyn DynBackendStorage,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let rhs = downcast(rhs)?;
        let result = Map2::map(&crate::ops::MatMul(bmnk), &self.0, lhs_layout, &rhs.0, rhs_layout)?;
        Ok(Box::new(CpuStorage(result)))
    }

    fn copy_strided_src_dyn(
        &self,
        dst: &mut dyn DynBackendStorage,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> Result<()> {
        let dst = downcast_mut(dst)?;
        cpu_copy_strided_src(&self.0, &mut dst.0, dst_offset, src_layout)
    }

    fn copy2d_dyn(
        &self,
        dst: &mut dyn DynBackendStorage,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        let dst = downcast_mut(dst)?;
        cpu_copy2d(&self.0, &mut dst.0, d1, d2, src_stride1, dst_stride1, src_offset, dst_offset)
    }

    fn const_set_dyn(&mut self, value: Scalar, layout: &Layout) -> Result<()> {
        cpu_const_set(&mut self.0, value, layout)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// impl DynBackendDevice for CpuBackendDevice
// ---------------------------------------------------------------------------

impl DynBackendDevice for CpuBackendDevice {
    fn location_dyn(&self) -> DeviceLocation {
        DeviceLocation::Cpu
    }

    fn same_device_dyn(&self, other: &dyn DynBackendDevice) -> bool {
        other.as_any().downcast_ref::<CpuBackendDevice>().is_some()
    }

    fn zeros_impl_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let storage = cpu_zeros(shape, dtype)?;
        Ok(Box::new(CpuStorage(storage)))
    }

    unsafe fn alloc_uninit_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let storage = unsafe { cpu_alloc_uninit(shape, dtype)? };
        Ok(Box::new(CpuStorage(storage)))
    }

    fn storage_from_host_buffer_dyn(
        &self,
        buf: &HostBuffer,
    ) -> Result<Box<dyn DynBackendStorage>> {
        Ok(Box::new(CpuStorage(buf.clone())))
    }

    fn storage_from_host_buffer_owned_dyn(
        &self,
        buf: HostBuffer,
    ) -> Result<Box<dyn DynBackendStorage>> {
        Ok(Box::new(CpuStorage(buf)))
    }

    fn rand_uniform_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: f64,
        hi: f64,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let storage = cpu_rand_uniform(shape, dtype, lo, hi)?;
        Ok(Box::new(CpuStorage(storage)))
    }

    fn rand_normal_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let storage = cpu_rand_normal(shape, dtype, mean, std)?;
        Ok(Box::new(CpuStorage(storage)))
    }

    fn set_seed_dyn(&self, _seed: u64) -> Result<()> {
        Err(Error::Msg("cannot seed the CPU rng with set_seed".into()).bt())
    }

    fn get_current_seed_dyn(&self) -> Result<u64> {
        Err(Error::Msg("cannot get the CPU rng seed with get_current_seed".into()).bt())
    }

    fn synchronize_dyn(&self) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_quantized_kernels(
        &self,
    ) -> Option<&dyn fuel_core_types::quantized::QuantizedDeviceKernels> {
        Some(self)
    }
}

// ---------------------------------------------------------------------------
// Device helper implementations
// ---------------------------------------------------------------------------

fn cpu_zeros(shape: &Shape, dtype: DType) -> Result<HostBuffer> {
    let elem_count = shape.elem_count();
    let storage = match dtype {
        DType::U8 => HostBuffer::U8(vec![0u8; elem_count]),
        DType::I8 => HostBuffer::I8(vec![0i8; elem_count]),
        DType::U32 => HostBuffer::U32(vec![0u32; elem_count]),
        DType::I16 => HostBuffer::I16(vec![0i16; elem_count]),
        DType::I32 => HostBuffer::I32(vec![0i32; elem_count]),
        DType::I64 => HostBuffer::I64(vec![0i64; elem_count]),
        DType::BF16 => HostBuffer::BF16(vec![bf16::ZERO; elem_count]),
        DType::F16 => HostBuffer::F16(vec![f16::ZERO; elem_count]),
        DType::F32 => HostBuffer::F32(vec![0f32; elem_count]),
        DType::F64 => HostBuffer::F64(vec![0f64; elem_count]),
        DType::F8E4M3 => HostBuffer::F8E4M3(vec![F8E4M3::ZERO; elem_count]),
        DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
            return Err(Error::UnsupportedDTypeForOp(dtype, "zeros").bt())
        }
    };
    Ok(storage)
}

#[allow(clippy::uninit_vec)]
unsafe fn cpu_alloc_uninit(shape: &Shape, dtype: DType) -> Result<HostBuffer> {
    let elem_count = shape.elem_count();
    let storage = match dtype {
        DType::U8 => { let mut v = Vec::with_capacity(elem_count); unsafe { v.set_len(elem_count) }; HostBuffer::U8(v) }
        DType::I8 => { let mut v = Vec::with_capacity(elem_count); unsafe { v.set_len(elem_count) }; HostBuffer::I8(v) }
        DType::U32 => { let mut v = Vec::with_capacity(elem_count); unsafe { v.set_len(elem_count) }; HostBuffer::U32(v) }
        DType::I16 => { let mut v = Vec::with_capacity(elem_count); unsafe { v.set_len(elem_count) }; HostBuffer::I16(v) }
        DType::I32 => { let mut v = Vec::with_capacity(elem_count); unsafe { v.set_len(elem_count) }; HostBuffer::I32(v) }
        DType::I64 => { let mut v = Vec::with_capacity(elem_count); unsafe { v.set_len(elem_count) }; HostBuffer::I64(v) }
        DType::BF16 => { let mut v = Vec::with_capacity(elem_count); unsafe { v.set_len(elem_count) }; HostBuffer::BF16(v) }
        DType::F16 => { let mut v = Vec::with_capacity(elem_count); unsafe { v.set_len(elem_count) }; HostBuffer::F16(v) }
        DType::F32 => { let mut v = Vec::with_capacity(elem_count); unsafe { v.set_len(elem_count) }; HostBuffer::F32(v) }
        DType::F64 => { let mut v = Vec::with_capacity(elem_count); unsafe { v.set_len(elem_count) }; HostBuffer::F64(v) }
        DType::F8E4M3 => { let mut v = Vec::with_capacity(elem_count); unsafe { v.set_len(elem_count) }; HostBuffer::F8E4M3(v) }
        DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
            return Err(Error::UnsupportedDTypeForOp(dtype, "alloc_uninit").bt())
        }
    };
    Ok(storage)
}

fn cpu_rand_uniform(shape: &Shape, dtype: DType, min: f64, max: f64) -> Result<HostBuffer> {
    use rand::prelude::*;
    let elem_count = shape.elem_count();
    let mut rng = rand::rng();
    match dtype {
        DType::BF16 => {
            let uniform = rand::distr::Uniform::new(bf16::from_f64(min), bf16::from_f64(max))
                .map_err(fuel_core_types::Error::wrap)?;
            let data: Vec<_> = (0..elem_count).map(|_| rng.sample::<bf16, _>(uniform)).collect();
            Ok(HostBuffer::BF16(data))
        }
        DType::F16 => {
            let uniform = rand::distr::Uniform::new(f16::from_f64(min), f16::from_f64(max))
                .map_err(fuel_core_types::Error::wrap)?;
            let data: Vec<_> = (0..elem_count).map(|_| rng.sample::<f16, _>(uniform)).collect();
            Ok(HostBuffer::F16(data))
        }
        DType::F8E4M3 => {
            let uniform = rand::distr::Uniform::new(F8E4M3::from_f64(min), F8E4M3::from_f64(max))
                .map_err(fuel_core_types::Error::wrap)?;
            let data: Vec<_> = (0..elem_count).map(|_| rng.sample::<F8E4M3, _>(uniform)).collect();
            Ok(HostBuffer::F8E4M3(data))
        }
        DType::F32 => {
            let uniform = rand::distr::Uniform::new(min as f32, max as f32)
                .map_err(fuel_core_types::Error::wrap)?;
            let data: Vec<_> = (0..elem_count).map(|_| rng.sample::<f32, _>(uniform)).collect();
            Ok(HostBuffer::F32(data))
        }
        DType::F64 => {
            let uniform = rand::distr::Uniform::new(min, max)
                .map_err(fuel_core_types::Error::wrap)?;
            let data: Vec<_> = (0..elem_count).map(|_| rng.sample::<f64, _>(uniform)).collect();
            Ok(HostBuffer::F64(data))
        }
        _ => Err(Error::UnsupportedDTypeForOp(dtype, "rand_uniform").bt()),
    }
}

fn cpu_rand_normal(shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<HostBuffer> {
    use rand::prelude::*;
    let elem_count = shape.elem_count();
    let mut rng = rand::rng();
    match dtype {
        DType::BF16 => {
            let normal = rand_distr::Normal::new(bf16::from_f64(mean), bf16::from_f64(std))
                .map_err(fuel_core_types::Error::wrap)?;
            let data: Vec<_> = (0..elem_count).map(|_| normal.sample(&mut rng)).collect();
            Ok(HostBuffer::BF16(data))
        }
        DType::F16 => {
            let normal = rand_distr::Normal::new(f16::from_f64(mean), f16::from_f64(std))
                .map_err(fuel_core_types::Error::wrap)?;
            let data: Vec<_> = (0..elem_count).map(|_| normal.sample(&mut rng)).collect();
            Ok(HostBuffer::F16(data))
        }
        DType::F8E4M3 => {
            let normal = rand_distr::Normal::new(F8E4M3::from_f64(mean), F8E4M3::from_f64(std))
                .map_err(fuel_core_types::Error::wrap)?;
            let data: Vec<_> = (0..elem_count).map(|_| normal.sample(&mut rng)).collect();
            Ok(HostBuffer::F8E4M3(data))
        }
        DType::F32 => {
            let normal = rand_distr::Normal::new(mean as f32, std as f32)
                .map_err(fuel_core_types::Error::wrap)?;
            let data: Vec<_> = (0..elem_count).map(|_| normal.sample(&mut rng)).collect();
            Ok(HostBuffer::F32(data))
        }
        DType::F64 => {
            let normal = rand_distr::Normal::new(mean, std)
                .map_err(fuel_core_types::Error::wrap)?;
            let data: Vec<_> = (0..elem_count).map(|_| normal.sample(&mut rng)).collect();
            Ok(HostBuffer::F64(data))
        }
        _ => Err(Error::UnsupportedDTypeForOp(dtype, "rand_normal").bt()),
    }
}

// ---------------------------------------------------------------------------
// Storage helper functions (copy, const_set, to_dtype)
// ---------------------------------------------------------------------------

fn cpu_copy_strided_src(
    src: &HostBuffer,
    dst: &mut HostBuffer,
    dst_offset: usize,
    src_l: &Layout,
) -> Result<()> {
    match (src, dst) {
        (HostBuffer::U8(s), HostBuffer::U8(d)) => copy_strided_src_(s, d, dst_offset, src_l),
        (HostBuffer::U32(s), HostBuffer::U32(d)) => copy_strided_src_(s, d, dst_offset, src_l),
        (HostBuffer::I16(s), HostBuffer::I16(d)) => copy_strided_src_(s, d, dst_offset, src_l),
        (HostBuffer::I32(s), HostBuffer::I32(d)) => copy_strided_src_(s, d, dst_offset, src_l),
        (HostBuffer::I64(s), HostBuffer::I64(d)) => copy_strided_src_(s, d, dst_offset, src_l),
        (HostBuffer::BF16(s), HostBuffer::BF16(d)) => copy_strided_src_(s, d, dst_offset, src_l),
        (HostBuffer::F16(s), HostBuffer::F16(d)) => copy_strided_src_(s, d, dst_offset, src_l),
        (HostBuffer::F32(s), HostBuffer::F32(d)) => copy_strided_src_(s, d, dst_offset, src_l),
        (HostBuffer::F64(s), HostBuffer::F64(d)) => copy_strided_src_(s, d, dst_offset, src_l),
        (HostBuffer::F8E4M3(s), HostBuffer::F8E4M3(d)) => copy_strided_src_(s, d, dst_offset, src_l),
        (_, d) => {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: src.dtype(),
                rhs: d.dtype(),
                op: "copy_strided",
            }.bt());
        }
    }
    Ok(())
}

fn cpu_copy2d(
    src: &HostBuffer,
    dst: &mut HostBuffer,
    d1: usize,
    d2: usize,
    src_s: usize,
    dst_s: usize,
    src_o: usize,
    dst_o: usize,
) -> Result<()> {
    match (src, dst) {
        (HostBuffer::U8(s), HostBuffer::U8(d)) => copy2d_(s, d, d1, d2, src_s, dst_s, src_o, dst_o),
        (HostBuffer::U32(s), HostBuffer::U32(d)) => copy2d_(s, d, d1, d2, src_s, dst_s, src_o, dst_o),
        (HostBuffer::I16(s), HostBuffer::I16(d)) => copy2d_(s, d, d1, d2, src_s, dst_s, src_o, dst_o),
        (HostBuffer::I32(s), HostBuffer::I32(d)) => copy2d_(s, d, d1, d2, src_s, dst_s, src_o, dst_o),
        (HostBuffer::I64(s), HostBuffer::I64(d)) => copy2d_(s, d, d1, d2, src_s, dst_s, src_o, dst_o),
        (HostBuffer::BF16(s), HostBuffer::BF16(d)) => copy2d_(s, d, d1, d2, src_s, dst_s, src_o, dst_o),
        (HostBuffer::F16(s), HostBuffer::F16(d)) => copy2d_(s, d, d1, d2, src_s, dst_s, src_o, dst_o),
        (HostBuffer::F32(s), HostBuffer::F32(d)) => copy2d_(s, d, d1, d2, src_s, dst_s, src_o, dst_o),
        (HostBuffer::F64(s), HostBuffer::F64(d)) => copy2d_(s, d, d1, d2, src_s, dst_s, src_o, dst_o),
        (HostBuffer::F8E4M3(s), HostBuffer::F8E4M3(d)) => copy2d_(s, d, d1, d2, src_s, dst_s, src_o, dst_o),
        (_, d) => {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: src.dtype(),
                rhs: d.dtype(),
                op: "copy2d",
            }.bt());
        }
    }
    Ok(())
}

fn cpu_const_set(storage: &mut HostBuffer, s: Scalar, l: &Layout) -> Result<()> {
    use fuel_core_types::Scalar as S;
    fn set<T: fuel_core_types::WithDType + Copy>(src: &mut [T], l: &Layout, s: T) {
        match l.strided_blocks() {
            fuel_core_types::StridedBlocks::SingleBlock { start_offset, len } => {
                src[start_offset..start_offset + len].fill(s)
            }
            fuel_core_types::StridedBlocks::MultipleBlocks { block_start_index, block_len: 1 } => {
                for src_index in block_start_index {
                    src[src_index] = s;
                }
            }
            fuel_core_types::StridedBlocks::MultipleBlocks { block_start_index, block_len } => {
                for src_index in block_start_index {
                    src[src_index..src_index + block_len].fill(s);
                }
            }
        }
    }
    match (storage, s) {
        (HostBuffer::BF16(d), S::BF16(v)) => set(d, l, v),
        (HostBuffer::F16(d), S::F16(v)) => set(d, l, v),
        (HostBuffer::F32(d), S::F32(v)) => set(d, l, v),
        (HostBuffer::F64(d), S::F64(v)) => set(d, l, v),
        (HostBuffer::U8(d), S::U8(v)) => set(d, l, v),
        (HostBuffer::U32(d), S::U32(v)) => set(d, l, v),
        (HostBuffer::I16(d), S::I16(v)) => set(d, l, v),
        (HostBuffer::I32(d), S::I32(v)) => set(d, l, v),
        (HostBuffer::I64(d), S::I64(v)) => set(d, l, v),
        (HostBuffer::F8E4M3(d), S::F8E4M3(v)) => set(d, l, v),
        (st, s) => return Err(Error::Msg(format!(
            "const_set dtype mismatch, expected {:?} but got {:?}",
            st.dtype(), s
        )).bt()),
    }
    Ok(())
}

/// Dtype conversion.  This has O(dtypes²) match arms — mirrors
/// `BackendStorage::to_dtype` from fuel-core's cpu_backend.
fn cpu_to_dtype(src: &HostBuffer, layout: &Layout, dtype: DType) -> Result<HostBuffer> {
    // Short-circuit: if source dtype matches target, just copy via layout.
    if src.dtype() == dtype {
        // Clone elements described by layout.
        return match src {
            HostBuffer::U8(d) => Ok(HostBuffer::U8(unary_map(d, layout, |v: u8| v))),
            HostBuffer::U32(d) => Ok(HostBuffer::U32(unary_map(d, layout, |v: u32| v))),
            HostBuffer::I16(d) => Ok(HostBuffer::I16(unary_map(d, layout, |v: i16| v))),
            HostBuffer::I32(d) => Ok(HostBuffer::I32(unary_map(d, layout, |v: i32| v))),
            HostBuffer::I64(d) => Ok(HostBuffer::I64(unary_map(d, layout, |v: i64| v))),
            HostBuffer::BF16(d) => Ok(HostBuffer::BF16(unary_map(d, layout, |v: bf16| v))),
            HostBuffer::F16(d) => Ok(HostBuffer::F16(unary_map(d, layout, |v: f16| v))),
            HostBuffer::F32(d) => Ok(HostBuffer::F32(unary_map(d, layout, |v: f32| v))),
            HostBuffer::F64(d) => Ok(HostBuffer::F64(unary_map(d, layout, |v: f64| v))),
            HostBuffer::F8E4M3(d) => Ok(HostBuffer::F8E4M3(unary_map(d, layout, |v: F8E4M3| v))),
            _ => Err(Error::UnsupportedDTypeForOp(src.dtype(), "to_dtype").bt()),
        };
    }

    // Generic conversion: go through f64 as intermediate.
    // This handles all non-dummy dtype pairs.
    let as_f64: Vec<f64> = match src {
        HostBuffer::U8(d) => unary_map(d, layout, |v: u8| v as f64),
        HostBuffer::U32(d) => unary_map(d, layout, |v: u32| v as f64),
        HostBuffer::I16(d) => unary_map(d, layout, |v: i16| v as f64),
        HostBuffer::I32(d) => unary_map(d, layout, |v: i32| v as f64),
        HostBuffer::I64(d) => unary_map(d, layout, |v: i64| v as f64),
        HostBuffer::BF16(d) => unary_map(d, layout, |v: bf16| v.to_f64()),
        HostBuffer::F16(d) => unary_map(d, layout, |v: f16| v.to_f64()),
        HostBuffer::F32(d) => unary_map(d, layout, |v: f32| v as f64),
        HostBuffer::F64(d) => unary_map(d, layout, |v: f64| v),
        HostBuffer::F8E4M3(d) => unary_map(d, layout, |v: F8E4M3| v.to_f64()),
        _ => return Err(Error::UnsupportedDTypeForOp(src.dtype(), "to_dtype").bt()),
    };

    match dtype {
        DType::U8 => Ok(HostBuffer::U8(as_f64.into_iter().map(|v| v as u8).collect())),
        DType::U32 => Ok(HostBuffer::U32(as_f64.into_iter().map(|v| v as u32).collect())),
        DType::I16 => Ok(HostBuffer::I16(as_f64.into_iter().map(|v| v as i16).collect())),
        DType::I32 => Ok(HostBuffer::I32(as_f64.into_iter().map(|v| v as i32).collect())),
        DType::I64 => Ok(HostBuffer::I64(as_f64.into_iter().map(|v| v as i64).collect())),
        DType::BF16 => Ok(HostBuffer::BF16(as_f64.into_iter().map(bf16::from_f64).collect())),
        DType::F16 => Ok(HostBuffer::F16(as_f64.into_iter().map(f16::from_f64).collect())),
        DType::F32 => Ok(HostBuffer::F32(as_f64.into_iter().map(|v| v as f32).collect())),
        DType::F64 => Ok(HostBuffer::F64(as_f64)),
        DType::F8E4M3 => Ok(HostBuffer::F8E4M3(as_f64.into_iter().map(F8E4M3::from_f64).collect())),
        _ => Err(Error::UnsupportedDTypeForOp(dtype, "to_dtype").bt()),
    }
}
