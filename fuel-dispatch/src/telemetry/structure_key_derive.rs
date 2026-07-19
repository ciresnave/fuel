//! Fuel's INDEPENDENT `structure_key` derivation — the second implementation
//! for the two-implementation freeze-gate (KISS-CLASSIFY §6.6/§6.7).
//!
//! This is deliberately **Baracuda-free**: it recomputes the same `sk2` token
//! from Fuel's own [`FdxOperandDesc`] projection, with **no** `baracuda_kernels_*`
//! import, so a byte-match against Baracuda's emitted token is a genuine
//! two-implementation agreement. (K1 opacity — "Fuel never derives the key" —
//! governs the DISPATCH seam in [`super::structure_key`]; the freeze-gate is the
//! deliberate exception: Fuel derives the key independently *to check* it, never
//! to route.)
//!
//! Scope today: the elementwise families (`bin`), the `relu_add` f32 grid-stride
//! freeze-gate cell. `red` / `gem` / mixed-rank stay unbuilt until a consumer
//! needs them (mirrors the provider's own v1 staging).

use super::structure_key::{Contiguity, FdxOperandDesc};
use fuel_ir::DType;

/// The op-family category a `structure_key` keys on (KISS-CLASSIFY §6.5-0006).
/// Only the elementwise-binary case is derived today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuelOpCategory {
    /// Two-input, one-output elementwise (`bin`).
    BinaryElementwise,
}

impl FuelOpCategory {
    fn code(self) -> &'static str {
        match self {
            FuelOpCategory::BinaryElementwise => "bin",
        }
    }
}

/// Derive the KISS `sk2` `structure_key` token for a cell, independently of
/// Baracuda. `operands` are in canonical order — inputs then output
/// (§6.6-0014). Returns `None` (a typed decline, never a wrong token) on an
/// unmappable dtype, an empty operand list, or a rank over 8.
pub fn derive_structure_key_token(
    op: FuelOpCategory,
    operands: &[FdxOperandDesc],
    target: &str,
) -> Option<String> {
    let first = operands.first()?;
    if operands.iter().any(|o| o.shape.len() > 8) {
        return None; // MAX_OPERANDS rank cap (§6.6-0013)
    }
    let dtype = dtype_token(first.dtype)?;
    if !target.contains(':') {
        return None; // namespaced target required (§6.8-0001)
    }

    // Field 4 — index width: max touched offset Σ|stride|·(ext−1) across operands.
    let max_touched: i128 = operands
        .iter()
        .map(|o| {
            o.strides
                .iter()
                .zip(o.shape.iter())
                .map(|(&s, &e)| i128::from(s.unsigned_abs()) * i128::from(e.max(1) - 1))
                .sum::<i128>()
        })
        .max()
        .unwrap_or(0);
    let index_width = if max_touched >= (1i128 << 31) { "ix64" } else { "ix32" };

    // Field 5 — work class: total elements of operand 0 (§6.5-0007/0010).
    let work_elems: i128 = first.shape.iter().map(|&e| i128::from(e.max(1))).product();
    let work_class = if work_elems <= 32 {
        "warp"
    } else if work_elems <= 1024 {
        "block"
    } else {
        "grid"
    };

    // Field 6 — rank: widest operand rank (§6.6-0006).
    let rank = operands.iter().map(|o| o.shape.len()).max().unwrap_or(0);

    // Field 7 — per-operand sub-keys, canonical order (inputs then output, §6.6-0014).
    let operand_keys: Vec<String> = operands.iter().map(operand_sub_key).collect();

    Some(format!(
        "sk2|{op}|{dtype}|{target}|{idx}|{work}|r{rank}|{ops}|-",
        op = op.code(),
        idx = index_width,
        work = work_class,
        ops = operand_keys.join(";"),
    ))
}

/// KISS-CLASSIFY §6.1 dtype token for the keyed operand-0 dtype. Returns `None`
/// for a dtype not yet mapped to a canonical KISS token — a typed decline; the
/// deriver never emits a guessed token. `f32` is the freeze-gate case; the other
/// common floats are mapped, and the integer / MX tokens are deferred until a
/// cell needs them (pending the §6.1 token confirmation).
fn dtype_token(dt: DType) -> Option<&'static str> {
    Some(match dt {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F64 => "f64",
        _ => return None,
    })
}

/// One operand's `<contig>/<bcasthex>/<vec>/<div>/<flip>` sub-key (§6.6-0007).
fn operand_sub_key(o: &FdxOperandDesc) -> String {
    format!(
        "{}/{:02x}/{}/{}/{}",
        contig_code(o),
        bcast_mask(o),
        vec_code(o),
        div_code(o),
        if o.flipped { "r" } else { "f" },
    )
}

/// Contiguity class (§6.5-0001/0002): broadcast → `br`; fully C-contiguous →
/// `co`; inner-unit-stride but not fully contiguous → `ic`; else `st`.
fn contig_code(o: &FdxOperandDesc) -> &'static str {
    if o.broadcast {
        "br"
    } else if o.contiguity == Contiguity::Contiguous {
        "co"
    } else if inner_stride(o) == Some(1) {
        "ic"
    } else {
        "st"
    }
}

/// Broadcast-axis bitmask: bit `i` set iff axis `i` is a stride-0, extent>1 axis
/// (§6.6-0008 / §6.7-0010; rendered as lowercase 2-digit hex by the caller).
fn bcast_mask(o: &FdxOperandDesc) -> u8 {
    let mut m = 0u8;
    for (i, (&s, &e)) in o.strides.iter().zip(o.shape.iter()).enumerate().take(8) {
        if s == 0 && e > 1 {
            m |= 1 << i;
        }
    }
    m
}

/// Vector-width class (§6.5-0003/0009): the largest v ∈ {8,4,2} whose byte width
/// ≤ 16, divides the base alignment, and divides the inner extent, over a
/// forward unit-stride non-broadcast inner axis (§6.5-0013); else `v1`.
fn vec_code(o: &FdxOperandDesc) -> &'static str {
    let dsz = o.dtype.size_in_bytes();
    if dsz == 0 || o.broadcast || inner_stride(o) != Some(1) {
        return "v1";
    }
    let ext = inner_extent(o);
    for &v in &[8u32, 4, 2] {
        let vbytes = v * dsz as u32;
        if vbytes <= 16 && o.align_bytes % vbytes == 0 && ext % i64::from(v) == 0 {
            return match v {
                8 => "v8",
                4 => "v4",
                _ => "v2",
            };
        }
    }
    "v1"
}

/// Inner-extent divisibility ladder (§6.5-0004/0012).
fn div_code(o: &FdxOperandDesc) -> &'static str {
    let ext = inner_extent(o);
    if ext % 16 == 0 {
        "d16"
    } else if ext % 8 == 0 {
        "d8"
    } else if ext % 4 == 0 {
        "d4"
    } else if ext % 2 == 0 {
        "d2"
    } else {
        "da"
    }
}

/// The inner (vectorized-walk) axis = the highest-index axis with extent > 1,
/// falling back to the last axis; `None` for a rank-0 / all-ones operand.
fn inner_axis(o: &FdxOperandDesc) -> Option<usize> {
    o.shape
        .iter()
        .rposition(|&e| e > 1)
        .or_else(|| o.shape.len().checked_sub(1))
}
fn inner_extent(o: &FdxOperandDesc) -> i64 {
    inner_axis(o).map(|i| o.shape[i]).unwrap_or(1)
}
fn inner_stride(o: &FdxOperandDesc) -> Option<i64> {
    inner_axis(o).map(|i| o.strides[i])
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::{DType, Layout, Shape};

    #[test]
    fn fuel_derives_relu_add_sk2_token_byte_for_byte() {
        // 3 rank-1 f32 operands [4096], contiguous, offset 0 (align 256):
        // in0, in1, out — the committed `relu_add` f32 grid-stride cell.
        let f32_4096 =
            FdxOperandDesc::from_layout(&Layout::contiguous(Shape::from_dims(&[4096])), DType::F32);
        let token = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[f32_4096.clone(), f32_4096.clone(), f32_4096],
            "cuda:sm89",
        )
        .expect("relu_add f32 must derive a token");
        assert_eq!(
            token,
            "sk2|bin|f32|cuda:sm89|ix32|grid|r1|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-"
        );
    }

    #[test]
    fn declines_rather_than_guessing() {
        let bad_dtype =
            FdxOperandDesc::from_layout(&Layout::contiguous(Shape::from_dims(&[4096])), DType::I32);
        // Unmapped dtype → typed decline, never a guessed token.
        assert_eq!(
            derive_structure_key_token(
                FuelOpCategory::BinaryElementwise,
                &[bad_dtype],
                "cuda:sm89"
            ),
            None
        );
        // A non-namespaced target is rejected (§6.8-0001 requires `<ns>:<cap>`).
        let f32 =
            FdxOperandDesc::from_layout(&Layout::contiguous(Shape::from_dims(&[4096])), DType::F32);
        assert_eq!(
            derive_structure_key_token(FuelOpCategory::BinaryElementwise, &[f32], "sm89"),
            None
        );
        // No operands → decline.
        assert_eq!(
            derive_structure_key_token(FuelOpCategory::BinaryElementwise, &[], "cuda:sm89"),
            None
        );
    }

    #[test]
    fn general_fields_derive_correctly() {
        // A small odd extent exercises the v1 / da / warp fallbacks (not the
        // v4/d16/grid path of the golden cell): [7] f32 contiguous, 1 operand.
        let f32_7 =
            FdxOperandDesc::from_layout(&Layout::contiguous(Shape::from_dims(&[7])), DType::F32);
        let token =
            derive_structure_key_token(FuelOpCategory::BinaryElementwise, &[f32_7], "cuda:sm89")
                .expect("f32 must derive");
        assert_eq!(token, "sk2|bin|f32|cuda:sm89|ix32|warp|r1|co/00/v1/da/f|-");
    }
}
