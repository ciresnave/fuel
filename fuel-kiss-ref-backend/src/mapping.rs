//! Fuel ↔ kiss-ref vocabulary mapping for the scalar/elementwise floor.
//!
//! Maps Fuel's [`OpTag`] and [`DType`] onto kiss-ref's `Op` / `Dtype`. Only the
//! floor subset kiss-ref covers is mapped; everything else declines (`None`), so
//! [`supports`] gates every adapter call. NaN-propagating `Maximum`/`Minimum`
//! (Fuel's pinned convention) map to kiss `MaxProp`/`MinProp`; Fuel's `Gelu`
//! (tanh-approx) maps to kiss `GeluTanh` and `GeluErf` (exact) to kiss `Gelu`.

use fuel_ir::DType;
use fuel_kernel_seam_types::OpTag;
use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;

/// Map a Fuel op tag to its kiss-ref `Op`, or `None` if off the mapped floor.
pub fn op_to_kiss(op: OpTag) -> Option<Op> {
    use OpTag as T;
    Some(match op {
        // binary arithmetic / extremum
        T::Add => Op::Add,
        T::Sub => Op::Sub,
        T::Mul => Op::Mul,
        T::Div => Op::Div,
        T::Maximum => Op::MaxProp, // NaN-propagating (Fuel convention)
        T::Minimum => Op::MinProp,
        T::Pow => Op::Pow,
        // unary math
        T::Neg => Op::Neg,
        T::Abs => Op::Abs,
        T::Sqr => Op::Sqr,
        T::Sqrt => Op::Sqrt,
        T::Rsqrt => Op::Rsqrt,
        T::Recip => Op::Recip,
        T::Sign => Op::Sign,
        T::Exp => Op::Exp,
        T::Log => Op::Log,
        T::Sin => Op::Sin,
        T::Cos => Op::Cos,
        T::Tanh => Op::Tanh,
        T::Erf => Op::Erf,
        // activations
        T::Relu => Op::Relu,
        T::Sigmoid => Op::Sigmoid,
        T::Silu => Op::Silu,
        T::Gelu => Op::GeluTanh, // Fuel Gelu = tanh-approx
        T::GeluErf => Op::Gelu,  // Fuel GeluErf = exact erf
        T::Step => Op::Step,
        // rounding
        T::Floor => Op::Floor,
        T::Ceil => Op::Ceil,
        T::Round => Op::RoundEven,
        // everything else (Rem, MatMul, reductions, shape/index ops, …) declines
        _ => return None,
    })
}

/// Map a Fuel dtype to its kiss-ref `Dtype`, or `None` if kiss-ref has no
/// equivalent (Fuel's MX formats `F8E8M0`/`F6E2M3`/`F6E3M2`/`F4`).
pub fn dtype_to_kiss(d: DType) -> Option<Dtype> {
    use DType as D;
    Some(match d {
        D::F16 => Dtype::F16,
        D::BF16 => Dtype::Bf16,
        D::F32 => Dtype::F32,
        D::F64 => Dtype::F64,
        D::U8 => Dtype::U8,
        D::I8 => Dtype::S8,
        D::U32 => Dtype::U32,
        D::I16 => Dtype::S16,
        D::I32 => Dtype::I32,
        D::I64 => Dtype::I64,
        D::F8E4M3 => Dtype::E4m3,
        _ => return None,
    })
}

/// Whether `(op, dtype)` is a live kiss-ref diff target — mapped both ways AND
/// `Support::Done` (an unmapped op/dtype, or a `Pending`/`NotApplicable` cell,
/// declines). Uses `matches!` so the growing `Support` enum needs no arm here.
pub fn supports(op: OpTag, dtype: DType) -> bool {
    let (Some(o), Some(d)) = (op_to_kiss(op), dtype_to_kiss(dtype)) else {
        return false;
    };
    matches!(kiss_ref_core::support(o, d), kiss_ref_core::Support::Done)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_the_floor() {
        assert!(matches!(op_to_kiss(OpTag::Add), Some(Op::Add)));
        assert!(matches!(op_to_kiss(OpTag::Maximum), Some(Op::MaxProp)));
        assert!(matches!(op_to_kiss(OpTag::Gelu), Some(Op::GeluTanh)));
        assert!(matches!(op_to_kiss(OpTag::GeluErf), Some(Op::Gelu)));
        assert!(matches!(dtype_to_kiss(DType::F32), Some(Dtype::F32)));
        assert!(matches!(dtype_to_kiss(DType::I8), Some(Dtype::S8)));
        assert!(matches!(dtype_to_kiss(DType::BF16), Some(Dtype::Bf16)));
    }

    #[test]
    fn declines_off_floor_op() {
        assert!(op_to_kiss(OpTag::MatMul).is_none());
    }

    #[test]
    fn declines_dtype_without_kiss_equivalent() {
        // An MX format Fuel has but kiss-classify lacks.
        assert!(dtype_to_kiss(DType::F6E2M3).is_none());
        assert!(!supports(OpTag::Add, DType::F6E2M3));
    }

    #[test]
    fn supports_floor_cell_and_declines_off_floor() {
        assert!(supports(OpTag::Add, DType::F32));
        assert!(!supports(OpTag::MatMul, DType::F32)); // op declines
    }
}
