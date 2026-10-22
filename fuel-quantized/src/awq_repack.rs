// SPDX-License-Identifier: MIT OR Apache-2.0
//! Host repack of a HuggingFace `*-AWQ` (GEMM-version) checkpoint into the
//! buffers baracuda's W4A16 AWQ kernel reads (ROADMAP row 31).
//!
//! Pure host code: no CUDA, no baracuda dependency. The kernel layout is the one
//! documented on baracuda's `Int4AwqGemmArgs` (baracuda#219, alpha.147), which its
//! `tests/awq_reference.rs` verifies on a device against a dequantise-then-GEMM
//! reference. Everything is OC-major, with `groups = IC / group_size` and
//! `mult = 128 / group_size`:
//!
//! | buffer    | kernel layout                                                        |
//! |-----------|----------------------------------------------------------------------|
//! | `weight`  | `[OC, IC/8]` `u32`; nibble `ic % 8` of word `ic / 8` in row `oc`     |
//! | `zeros`   | `[OC, zeros_w]` `u32`; nibble `g % 8` of word `g / 8` in row `oc`    |
//! | `scales`  | `[OC, 8 * zeros_w]` `f16`; entry `g` of row `oc`                     |
//!
//! with `zeros_w = ceil(ceil(groups / 8) / mult) * mult`. The tail of each
//! `zeros`/`scales` row past `groups` is padding the kernel never reads as data;
//! it is written as zero.
//!
//! The HF checkpoint is the transpose and uses AutoAWQ's interleaved nibble order
//! along `OC` ([`HF_AWQ_PACK_ORDER`]):
//!
//! | tensor    | HF layout                                                            |
//! |-----------|----------------------------------------------------------------------|
//! | `qweight` | `[IC, OC/8]` `u32`                                                   |
//! | `qzeros`  | `[IC/group_size, OC/8]` `u32`                                        |
//! | `scales`  | `[IC/group_size, OC]` `f16`                                          |
//!
//! Each HF word `col` holds logical columns `col*8 + HF_AWQ_PACK_ORDER[i]` at
//! nibble `i`.

use half::f16;

/// AutoAWQ's `order_map` (`awq/modules/linear/gemm.py`, `WQLinear_GEMM.from_linear`):
/// nibble position `i` of an HF-packed word holds the value of logical column
/// `HF_AWQ_PACK_ORDER[i]` within its group of 8. The same order applies to `qweight`
/// and `qzeros`. This is the HuggingFace checkpoint's order only: the op
/// (`fuel_graph::registry::awq_matmul`) takes the sequential kernel layout this
/// module produces.
pub const HF_AWQ_PACK_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// The kernel-layout buffers for one linear layer.
#[derive(Clone, Debug, PartialEq)]
pub struct AwqKernelBuffers {
    /// `[OC, IC/8]` packed int4 weights.
    pub weight: Vec<u32>,
    /// `[OC, zeros_w]` packed int4 zero-points.
    pub zeros: Vec<u32>,
    /// `[OC, 8 * zeros_w]` scales.
    pub scales: Vec<f16>,
    /// Words per `zeros` row (`scales` rows hold `8 * zeros_w`).
    pub zeros_w: usize,
}

/// Why a repack was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AwqRepackError {
    /// `group_size` is not 64 or 128.
    BadGroupSize(usize),
    /// A dimension is not a multiple of what the packing requires.
    BadDim {
        what: &'static str,
        value: usize,
        multiple_of: usize,
    },
    /// An input slice has the wrong length.
    BadLen {
        what: &'static str,
        got: usize,
        want: usize,
    },
}

impl std::fmt::Display for AwqRepackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadGroupSize(g) => write!(f, "AWQ group size must be 64 or 128, got {g}"),
            Self::BadDim {
                what,
                value,
                multiple_of,
            } => {
                write!(
                    f,
                    "AWQ {what} = {value} must be a multiple of {multiple_of}"
                )
            }
            Self::BadLen { what, got, want } => {
                write!(f, "AWQ {what} has {got} elements, expected {want}")
            }
        }
    }
}

impl std::error::Error for AwqRepackError {}

/// Words per `zeros` row for `ic` input features at `group_size` (the kernel pads
/// each row): `ceil(ceil(groups / 8) / mult) * mult`.
pub fn awq_zeros_w(ic: usize, group_size: usize) -> Result<usize, AwqRepackError> {
    if group_size != 64 && group_size != 128 {
        return Err(AwqRepackError::BadGroupSize(group_size));
    }
    if !ic.is_multiple_of(group_size) {
        return Err(AwqRepackError::BadDim {
            what: "IC",
            value: ic,
            multiple_of: group_size,
        });
    }
    let groups = ic / group_size;
    let mult = 128 / group_size;
    Ok(groups.div_ceil(8).div_ceil(mult) * mult)
}

fn check_len(what: &'static str, got: usize, want: usize) -> Result<(), AwqRepackError> {
    if got == want {
        Ok(())
    } else {
        Err(AwqRepackError::BadLen { what, got, want })
    }
}

fn check_multiple(
    what: &'static str,
    value: usize,
    multiple_of: usize,
) -> Result<(), AwqRepackError> {
    if value.is_multiple_of(multiple_of) {
        Ok(())
    } else {
        Err(AwqRepackError::BadDim {
            what,
            value,
            multiple_of,
        })
    }
}

/// Unpack one HF-packed row-major `[rows, oc/8]` tensor into logical `[rows, oc]`
/// 4-bit codes.
fn unpack_hf(words: &[u32], rows: usize, oc: usize) -> Vec<u8> {
    let wpr = oc / 8;
    let mut out = vec![0u8; rows * oc];
    for r in 0..rows {
        for col in 0..wpr {
            let w = words[r * wpr + col];
            for (i, &logical) in HF_AWQ_PACK_ORDER.iter().enumerate() {
                out[r * oc + col * 8 + logical] = ((w >> (4 * i)) & 0xF) as u8;
            }
        }
    }
    out
}

/// Repack an HF `*-AWQ` layer (`qweight [IC, OC/8]`, `qzeros [IC/g, OC/8]`,
/// `scales [IC/g, OC]`) into the kernel layout described in the module docs.
///
/// Checks only what the packing itself needs (`IC % 8`, `OC % 8`, `IC % group_size`,
/// slice lengths). The kernel's own constraints (`OC % 64`, `IC % (32 * split_k)`,
/// sm_80+) are baracuda's `can_implement`'s to enforce.
pub fn repack_hf_awq_for_kernel(
    qweight: &[u32],
    qzeros: &[u32],
    scales: &[f16],
    ic: usize,
    oc: usize,
    group_size: usize,
) -> Result<AwqKernelBuffers, AwqRepackError> {
    let zeros_w = awq_zeros_w(ic, group_size)?;
    check_multiple("IC", ic, 8)?;
    check_multiple("OC", oc, 8)?;
    let groups = ic / group_size;
    check_len("qweight", qweight.len(), ic * (oc / 8))?;
    check_len("qzeros", qzeros.len(), groups * (oc / 8))?;
    check_len("scales", scales.len(), groups * oc)?;

    let q = unpack_hf(qweight, ic, oc); // [IC, OC]
    let z = unpack_hf(qzeros, groups, oc); // [groups, OC]
    Ok(AwqKernelBuffers {
        weight: pack_oc_major(&q, ic, oc, ic / 8),
        zeros: pack_oc_major(&z, groups, oc, zeros_w),
        scales: transpose_scales(scales, groups, oc, zeros_w * 8),
        zeros_w,
    })
}

/// Transpose logical `[rows, oc]` 4-bit codes to OC-major and pack 8 per `u32` along
/// the row axis: output is `[oc, words_per_oc]`, word `r / 8` nibble `r % 8`.
fn pack_oc_major(codes: &[u8], rows: usize, oc: usize, words_per_oc: usize) -> Vec<u32> {
    let mut out = vec![0u32; oc * words_per_oc];
    for o in 0..oc {
        for r in 0..rows {
            out[o * words_per_oc + r / 8] |= u32::from(codes[r * oc + o]) << (4 * (r % 8));
        }
    }
    out
}

/// Transpose `[groups, oc]` scales to OC-major `[oc, sf_w]`, zero-padded past `groups`.
fn transpose_scales(scales: &[f16], groups: usize, oc: usize, sf_w: usize) -> Vec<f16> {
    let mut out = vec![f16::ZERO; oc * sf_w];
    for o in 0..oc {
        for g in 0..groups {
            out[o * sf_w + g] = scales[g * oc + o];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Logical codes for one layer: `q [IC, OC]`, `z [groups, OC]`, `s [groups, OC]`.
    struct Logical {
        q: Vec<u8>,
        z: Vec<u8>,
        s: Vec<f16>,
    }

    fn logical(ic: usize, oc: usize, group: usize) -> Logical {
        let mut r = Rng(0xD1B5_4A32_D192_ED03 ^ ((ic * 31 + oc) * 31 + group) as u64);
        let groups = ic / group;
        Logical {
            q: (0..ic * oc).map(|_| (r.next() & 0xF) as u8).collect(),
            z: (0..groups * oc).map(|_| (r.next() & 0xF) as u8).collect(),
            // Distinct, non-zero values so a misplaced scale is visible.
            s: (0..groups * oc)
                .map(|i| f16::from_f32(0.01 + i as f32 * 0.001))
                .collect(),
        }
    }

    /// Pack `[rows, oc]` codes the way AutoAWQ's `from_linear` does (the writer
    /// side, written independently of `unpack_hf`): word `col` of a row gets the
    /// value of logical column `col*8 + order_map[i]` at nibble `i`.
    fn pack_hf(codes: &[u8], rows: usize, oc: usize, order: [usize; 8]) -> Vec<u32> {
        let mut out = vec![0u32; rows * oc / 8];
        for r in 0..rows {
            for col in 0..oc / 8 {
                for i in 0..8 {
                    out[r * (oc / 8) + col] |=
                        u32::from(codes[r * oc + col * 8 + order[i]]) << (i * 4);
                }
            }
        }
        out
    }

    /// Read the kernel buffers back with baracuda's documented read formula and
    /// return the logical codes it would see.
    fn read_kernel(b: &AwqKernelBuffers, ic: usize, oc: usize, group: usize) -> Logical {
        let groups = ic / group;
        let sf_w = b.zeros_w * 8;
        let mut q = vec![0u8; ic * oc];
        let mut z = vec![0u8; groups * oc];
        let mut s = vec![f16::ZERO; groups * oc];
        for o in 0..oc {
            for i in 0..ic {
                q[i * oc + o] = ((b.weight[o * (ic / 8) + i / 8] >> (4 * (i % 8))) & 0xF) as u8;
            }
            for g in 0..groups {
                z[g * oc + o] = ((b.zeros[o * b.zeros_w + g / 8] >> (4 * (g % 8))) & 0xF) as u8;
                s[g * oc + o] = b.scales[o * sf_w + g];
            }
        }
        Logical { q, z, s }
    }

    /// `(ic, oc, group)`: group 128 with `groups % 8 != 0` (padded rows), group 64
    /// with an odd `ceil(groups/8)` (padded to `mult`), and a tidy case.
    const SHAPES: [(usize, usize, usize); 4] = [
        (128 * 9, 64, 128),
        (64 * 8, 64, 64),
        (128 * 16, 64, 128),
        (64 * 24, 128, 64),
    ];

    #[test]
    fn repack_round_trips_through_the_documented_kernel_read_formula() {
        for (ic, oc, group) in SHAPES {
            let l = logical(ic, oc, group);
            let groups = ic / group;
            let qw = pack_hf(&l.q, ic, oc, HF_AWQ_PACK_ORDER);
            let qz = pack_hf(&l.z, groups, oc, HF_AWQ_PACK_ORDER);
            let b = repack_hf_awq_for_kernel(&qw, &qz, &l.s, ic, oc, group).unwrap();
            assert_eq!(b.weight.len(), oc * ic / 8, "{ic}x{oc} g{group}");
            assert_eq!(b.zeros.len(), oc * b.zeros_w);
            assert_eq!(b.scales.len(), oc * b.zeros_w * 8);
            let back = read_kernel(&b, ic, oc, group);
            assert!(back.q == l.q, "weights {ic}x{oc} g{group}");
            assert!(back.z == l.z, "zeros {ic}x{oc} g{group}");
            assert!(back.s == l.s, "scales {ic}x{oc} g{group}");
        }
    }

    #[test]
    fn zeros_w_matches_baracuda_padding_formula_and_rows_are_padded_with_zero() {
        // Pinned values from baracuda's `tests/awq_reference.rs` formula.
        assert_eq!(awq_zeros_w(128 * 9, 128).unwrap(), 2); // groups 9 -> ceil(9/8)=2, mult 1
        assert_eq!(awq_zeros_w(64 * 8, 64).unwrap(), 2); // groups 8 -> 1, mult 2 -> 2
        assert_eq!(awq_zeros_w(128 * 16, 128).unwrap(), 2); // groups 16 -> 2
        assert_eq!(awq_zeros_w(64 * 24, 64).unwrap(), 4); // groups 24 -> 3, mult 2 -> 4
        let (ic, oc, group) = SHAPES[0];
        let l = logical(ic, oc, group);
        let qw = pack_hf(&l.q, ic, oc, HF_AWQ_PACK_ORDER);
        let qz = pack_hf(&l.z, ic / group, oc, HF_AWQ_PACK_ORDER);
        let b = repack_hf_awq_for_kernel(&qw, &qz, &l.s, ic, oc, group).unwrap();
        let groups = ic / group;
        let sf_w = b.zeros_w * 8;
        for o in 0..oc {
            assert!(
                b.scales[o * sf_w + groups..(o + 1) * sf_w]
                    .iter()
                    .all(|x| *x == f16::ZERO)
            );
            // Nibbles of the last zeros word at or past `groups % 8` are padding.
            let last = b.zeros[o * b.zeros_w + groups / 8];
            assert_eq!(last >> (4 * (groups % 8)), 0, "row {o} padding nibbles");
            assert!(
                b.zeros[o * b.zeros_w + groups / 8 + 1..(o + 1) * b.zeros_w]
                    .iter()
                    .all(|w| *w == 0)
            );
        }
    }

    /// Sabotage sibling: the interleaved order is load-bearing. A checkpoint packed
    /// in plain sequential nibble order must NOT repack to the right codes.
    #[test]
    fn a_sequentially_packed_checkpoint_does_not_round_trip() {
        let (ic, oc, group) = SHAPES[0];
        let l = logical(ic, oc, group);
        let seq = [0, 1, 2, 3, 4, 5, 6, 7];
        let qw = pack_hf(&l.q, ic, oc, seq);
        let qz = pack_hf(&l.z, ic / group, oc, seq);
        let b = repack_hf_awq_for_kernel(&qw, &qz, &l.s, ic, oc, group).unwrap();
        let back = read_kernel(&b, ic, oc, group);
        assert!(back.q != l.q, "the AWQ order must matter for qweight");
        assert!(back.z != l.z, "the AWQ order must matter for qzeros");
    }

    #[test]
    fn bad_inputs_are_refused() {
        let ok = |ic, oc, g, nw, nz, ns| {
            repack_hf_awq_for_kernel(&vec![0; nw], &vec![0; nz], &vec![f16::ZERO; ns], ic, oc, g)
        };
        assert_eq!(
            ok(256, 64, 32, 0, 0, 0),
            Err(AwqRepackError::BadGroupSize(32))
        );
        assert!(matches!(
            ok(200, 64, 128, 0, 0, 0),
            Err(AwqRepackError::BadDim { what: "IC", .. })
        ));
        assert!(matches!(
            ok(256, 60, 128, 0, 0, 0),
            Err(AwqRepackError::BadDim { what: "OC", .. })
        ));
        assert!(matches!(
            ok(256, 64, 128, 1, 0, 0),
            Err(AwqRepackError::BadLen {
                what: "qweight",
                ..
            })
        ));
        // Control: the same call with right lengths succeeds.
        assert!(ok(256, 64, 128, 256 * 8, 2 * 8, 2 * 64).is_ok());
    }
}
