//! # fuel-conv
//!
//! Reference 2D convolution primitives, deliberately backend-agnostic.
//!
//! Two concrete forms live here:
//!
//! 1. [`conv2d_direct`] — the textbook nested-loop forward pass.
//!    What `fuel-reference-backend::ops::conv2d` has always done.
//!    Slow but correct; the parity oracle every other backend
//!    measures itself against.
//!
//! 2. [`im2col`] — the input-rearrangement step that lets a vendor
//!    BLAS gemm (AOCL, oneMKL, GPU matmul shader) carry conv2d's
//!    arithmetic. After `im2col` writes `[batch * groups, c_in/groups
//!    * k_h * k_w, h_out * w_out]`, the conv reduces to a per-group
//!    matmul: `weight[g] @ patches[g] -> out[g]`. The matmul step is
//!    NOT in this crate — callers plug their own.
//!
//! No dependency on `fuel-core-types`. `fuel-conv` operates on
//! `&[T]` slices and a small [`ConvShape`] descriptor it owns. That
//! keeps it cheap to depend on from anywhere — the reference
//! backend, AOCL/MKL/OpenBLAS CPU backends, and the Vulkan/Metal
//! integration tests that need a parity oracle for their shader-
//! based im2col implementations.
//!
//! # NCHW layout assumption
//!
//! Input: `[batch, c_in, h, w]`, row-major contiguous.
//! Weight: `[c_out, c_in/groups, k_h, k_w]`, row-major contiguous.
//! Output: `[batch, c_out, h_out, w_out]`, row-major contiguous.
//!
//! Output spatial dimensions:
//!
//! ```text
//! h_out = (h + 2 * pad_h - k_h) / stride_h + 1
//! w_out = (w + 2 * pad_w - k_w) / stride_w + 1
//! ```
//!
//! Asymmetric stride/padding is supported (pass `(usize, usize)`).
//! Dilation is not — add when a model needs it.

use num_traits::Float;

/// Conv2d shape descriptor. Owned by this crate so consumers don't
/// have to drag in `fuel-core-types::conv::ParamsConv2D` just to
/// call [`conv2d_direct`] / [`im2col`]. Cheap to construct and pass
/// by reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvShape {
    /// Batch size.
    pub batch: usize,
    /// Total input channels.
    pub c_in: usize,
    /// Total output channels (must be a multiple of `groups`).
    pub c_out: usize,
    /// Input height.
    pub h: usize,
    /// Input width.
    pub w: usize,
    /// Kernel height.
    pub k_h: usize,
    /// Kernel width.
    pub k_w: usize,
    /// Stride `(h, w)`. Asymmetric supported.
    pub stride: (usize, usize),
    /// Zero-padding `(h, w)` added to all four input sides. Asymmetric
    /// supported (pad-h applied to top + bottom; pad-w to left + right).
    pub padding: (usize, usize),
    /// Number of groups. `groups == 1` is dense conv;
    /// `groups == c_in == c_out` is depthwise. Must divide both.
    pub groups: usize,
}

impl ConvShape {
    pub fn h_out(&self) -> usize {
        (self.h + 2 * self.padding.0 - self.k_h) / self.stride.0 + 1
    }
    pub fn w_out(&self) -> usize {
        (self.w + 2 * self.padding.1 - self.k_w) / self.stride.1 + 1
    }
    pub fn c_in_per_group(&self) -> usize { self.c_in / self.groups }
    pub fn c_out_per_group(&self) -> usize { self.c_out / self.groups }

    /// Validate that the descriptor is well-formed. Returns an error
    /// message describing the first issue, or `Ok(())`. Cheap; cheap
    /// enough to call at the top of every kernel.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.groups == 0 { return Err("groups must be ≥ 1"); }
        if self.c_in % self.groups != 0 {
            return Err("c_in must be divisible by groups");
        }
        if self.c_out % self.groups != 0 {
            return Err("c_out must be divisible by groups");
        }
        if self.stride.0 == 0 || self.stride.1 == 0 {
            return Err("stride components must be ≥ 1");
        }
        if self.k_h == 0 || self.k_w == 0 {
            return Err("kernel dimensions must be ≥ 1");
        }
        if self.h + 2 * self.padding.0 < self.k_h {
            return Err("kernel taller than padded input");
        }
        if self.w + 2 * self.padding.1 < self.k_w {
            return Err("kernel wider than padded input");
        }
        Ok(())
    }

    /// Element count of `[batch, c_out, h_out, w_out]`.
    pub fn output_len(&self) -> usize {
        self.batch * self.c_out * self.h_out() * self.w_out()
    }

    /// Element count of the im2col patches buffer:
    /// `[batch * groups, c_in_per_group * k_h * k_w, h_out * w_out]`.
    pub fn im2col_len(&self) -> usize {
        self.batch
            * self.groups
            * self.c_in_per_group() * self.k_h * self.k_w
            * self.h_out() * self.w_out()
    }
}

// =====================================================================
// Direct conv2d (the parity oracle)
// =====================================================================

/// Direct 2D convolution via the textbook nested-loop reference. No
/// im2col, no matmul fusion — slow but unambiguously correct. Every
/// other backend's conv kernel is verified against this output.
///
/// `bias` is optional; when present it's added per-output-channel.
///
/// `out` must be sized `s.output_len()`. The function writes every
/// output element exactly once; pre-zeroing isn't required.
pub fn conv2d_direct<T: Float>(
    x: &[T],
    weight: &[T],
    bias: Option<&[T]>,
    s: &ConvShape,
    out: &mut [T],
) {
    s.validate().expect("ConvShape::validate failed");
    let h_out = s.h_out();
    let w_out = s.w_out();
    let cin_per_g = s.c_in_per_group();
    let cout_per_g = s.c_out_per_group();
    let (stride_h, stride_w) = s.stride;
    let (pad_h, pad_w) = s.padding;

    debug_assert_eq!(x.len(), s.batch * s.c_in * s.h * s.w, "x size mismatch");
    debug_assert_eq!(weight.len(), s.c_out * cin_per_g * s.k_h * s.k_w, "weight size");
    debug_assert_eq!(out.len(), s.output_len(), "out size mismatch");
    if let Some(b) = bias { debug_assert_eq!(b.len(), s.c_out, "bias len"); }

    for ni in 0..s.batch {
        for g in 0..s.groups {
            for co_in_g in 0..cout_per_g {
                let co = g * cout_per_g + co_in_g;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = T::zero();
                        for ci_in_g in 0..cin_per_g {
                            let ci = g * cin_per_g + ci_in_g;
                            for ky in 0..s.k_h {
                                let ih_padded = oh * stride_h + ky;
                                if ih_padded < pad_h || ih_padded >= s.h + pad_h { continue; }
                                let ih = ih_padded - pad_h;
                                for kx in 0..s.k_w {
                                    let iw_padded = ow * stride_w + kx;
                                    if iw_padded < pad_w || iw_padded >= s.w + pad_w { continue; }
                                    let iw = iw_padded - pad_w;
                                    let x_off = ((ni * s.c_in + ci) * s.h + ih) * s.w + iw;
                                    let w_off = ((co * cin_per_g + ci_in_g) * s.k_h + ky) * s.k_w + kx;
                                    acc = acc + x[x_off] * weight[w_off];
                                }
                            }
                        }
                        if let Some(b) = bias { acc = acc + b[co]; }
                        let out_off = ((ni * s.c_out + co) * h_out + oh) * w_out + ow;
                        out[out_off] = acc;
                    }
                }
            }
        }
    }
}

// =====================================================================
// im2col — the rearrangement step for conv2d-via-matmul
// =====================================================================

/// Rearrange the NCHW input into the im2col patches matrix. After
/// this, conv2d reduces to a per-group matmul:
///
/// ```text
/// for g in 0..groups:
///     out[batch, g * cout_per_g .. (g+1) * cout_per_g, :, :]
///       = weight[g * cout_per_g .. (g+1) * cout_per_g, :, :, :]      // [cout_per_g, cin_per_g * k_h * k_w]
///         @
///         patches[batch * groups + g, :, :]                          // [cin_per_g * k_h * k_w, h_out * w_out]
/// ```
///
/// Output layout (row-major, C-style indexing):
///
/// ```text
/// out[batch_idx, group_idx, channel_idx_in_group, ky, kx, oh, ow]
///   = (zero if (oh*stride_h + ky - pad_h, ow*stride_w + kx - pad_w)
///         is outside [0,h) × [0,w))
///   else x[batch_idx, group_idx*cin_per_g + channel_idx_in_group,
///          oh*stride_h + ky - pad_h, ow*stride_w + kx - pad_w]
/// ```
///
/// Flattened: `out` has length `s.im2col_len()`.
///
/// Index ordering inside the patches axes is `(channel, ky, kx)` —
/// the same ordering the weight tensor uses for its inner three
/// axes — so the per-group matmul's K-dimension lines up directly
/// with the weight's "input channels × kernel area" reshape.
pub fn im2col<T: Float>(
    x: &[T],
    s: &ConvShape,
    out: &mut [T],
) {
    s.validate().expect("ConvShape::validate failed");
    let h_out = s.h_out();
    let w_out = s.w_out();
    let cin_per_g = s.c_in_per_group();
    let (stride_h, stride_w) = s.stride;
    let (pad_h, pad_w) = s.padding;

    debug_assert_eq!(x.len(), s.batch * s.c_in * s.h * s.w, "x size mismatch");
    debug_assert_eq!(out.len(), s.im2col_len(), "out size mismatch");

    // Patches axis = cin_per_g * k_h * k_w
    // Spatial axis = h_out * w_out
    let spatial = h_out * w_out;
    let patch_dim = cin_per_g * s.k_h * s.k_w;

    for ni in 0..s.batch {
        for g in 0..s.groups {
            // Output sub-buffer for this (batch, group) — shape
            // [patch_dim, spatial].
            let bg_idx = ni * s.groups + g;
            let group_offset = bg_idx * patch_dim * spatial;
            for ci_in_g in 0..cin_per_g {
                let ci = g * cin_per_g + ci_in_g;
                let x_channel_offset = (ni * s.c_in + ci) * s.h * s.w;
                for ky in 0..s.k_h {
                    for kx in 0..s.k_w {
                        let patch_row = (ci_in_g * s.k_h + ky) * s.k_w + kx;
                        let patch_offset = group_offset + patch_row * spatial;
                        for oh in 0..h_out {
                            let ih_padded = oh * stride_h + ky;
                            for ow in 0..w_out {
                                let iw_padded = ow * stride_w + kx;
                                let in_bounds = ih_padded >= pad_h
                                    && ih_padded < s.h + pad_h
                                    && iw_padded >= pad_w
                                    && iw_padded < s.w + pad_w;
                                let val = if in_bounds {
                                    let ih = ih_padded - pad_h;
                                    let iw = iw_padded - pad_w;
                                    x[x_channel_offset + ih * s.w + iw]
                                } else {
                                    T::zero()
                                };
                                out[patch_offset + oh * w_out + ow] = val;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Compose [`im2col`] + a caller-provided gemm into a full conv2d.
/// `gemm` is invoked once per `(batch, group)` pair with:
///
/// ```text
/// gemm(
///     m = cout_per_group,
///     n = h_out * w_out,
///     k = cin_per_group * k_h * k_w,
///     a = weight slice for this group  ([m, k] row-major),
///     b = patches slice for this (batch, group)  ([k, n] row-major),
///     c = output slice for this (batch, group)  ([m, n] row-major),
/// )
/// ```
///
/// The caller's gemm is responsible for `c = a @ b` (no accumulate;
/// caller pre-zeroes the output buffer if needed). Bias add is done
/// here after gemm if `bias` is provided.
///
/// This lets AOCL plug `aocl_blas::gemm`, MKL plug `onemkl::gemm`,
/// and the reference backend use [`conv2d_direct`] without any of
/// them re-implementing the im2col loop.
pub fn conv2d_via_gemm<T, F>(
    x: &[T],
    weight: &[T],
    bias: Option<&[T]>,
    s: &ConvShape,
    out: &mut [T],
    patches_scratch: &mut [T],
    mut gemm: F,
) where
    T: Float,
    F: FnMut(usize, usize, usize, &[T], &[T], &mut [T]),
{
    s.validate().expect("ConvShape::validate failed");
    let h_out = s.h_out();
    let w_out = s.w_out();
    let cin_per_g = s.c_in_per_group();
    let cout_per_g = s.c_out_per_group();
    let m = cout_per_g;
    let n = h_out * w_out;
    let k = cin_per_g * s.k_h * s.k_w;
    let weight_per_g = m * k;
    let out_per_bg = m * n;
    let patches_per_bg = k * n;

    // Step 1: rearrange the input into patches.
    im2col(x, s, patches_scratch);

    // Step 2: per-group matmul.
    for ni in 0..s.batch {
        for g in 0..s.groups {
            let bg_idx = ni * s.groups + g;
            let weight_off = g * weight_per_g;
            let patches_off = bg_idx * patches_per_bg;
            let out_off = (ni * s.c_out + g * cout_per_g) * n; // [m, n] block per group
            gemm(
                m, n, k,
                &weight[weight_off..weight_off + weight_per_g],
                &patches_scratch[patches_off..patches_off + patches_per_bg],
                &mut out[out_off..out_off + out_per_bg],
            );
        }
    }

    // Step 3: bias add (if any).
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), s.c_out);
        for ni in 0..s.batch {
            for co in 0..s.c_out {
                let row_off = (ni * s.c_out + co) * n;
                let bv = b[co];
                for j in 0..n {
                    out[row_off + j] = out[row_off + j] + bv;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small dense conv: groups=1, no padding, stride 1.
    #[test]
    fn direct_dense_2x2_kernel_3x3_input() {
        // 1 batch, 1 in-channel, 1 out-channel, 3x3 input, 2x2 kernel.
        let x: Vec<f32> = (1..=9).map(|i| i as f32).collect();
        let w: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0]; // identity-on-diagonal
        let s = ConvShape {
            batch: 1, c_in: 1, c_out: 1,
            h: 3, w: 3, k_h: 2, k_w: 2,
            stride: (1, 1), padding: (0, 0), groups: 1,
        };
        let mut out = vec![0.0_f32; s.output_len()];
        conv2d_direct(&x, &w, None, &s, &mut out);
        // Output should be 2x2: [1+5, 2+6, 4+8, 5+9] = [6, 8, 12, 14]
        assert_eq!(out, vec![6.0, 8.0, 12.0, 14.0]);
    }

    /// Depthwise conv: groups == c_in == c_out.
    #[test]
    fn direct_depthwise_3x3_with_padding() {
        let s = ConvShape {
            batch: 1, c_in: 2, c_out: 2,
            h: 3, w: 3, k_h: 3, k_w: 3,
            stride: (1, 1), padding: (1, 1), groups: 2,
        };
        // Channel 0: identity 3x3 (only center weight = 1)
        // Channel 1: zero
        let mut w = vec![0.0_f32; s.c_out * s.c_in_per_group() * s.k_h * s.k_w];
        // Channel 0's kernel center
        w[(0 * 1 + 0) * 9 + 4] = 1.0;
        // Channel 1's kernel: stays zero

        let x: Vec<f32> = (1..=18).map(|i| i as f32).collect();
        let mut out = vec![0.0_f32; s.output_len()];
        conv2d_direct(&x, &w, None, &s, &mut out);
        // Channel 0 output: identity → matches input channel 0 = [1..=9]
        // Channel 1 output: all zeros
        let expected_c0: Vec<f32> = (1..=9).map(|i| i as f32).collect();
        assert_eq!(&out[0..9], expected_c0.as_slice());
        assert_eq!(&out[9..18], vec![0.0; 9].as_slice());
    }

    /// im2col output layout — eyeball-test on a 1×1×3×3 input with
    /// 1×1 kernel + groups=1. Each patch should be a single input
    /// pixel, so im2col output is just the input flattened.
    #[test]
    fn im2col_1x1_kernel_is_input_flatten() {
        let x: Vec<f32> = (1..=9).map(|i| i as f32).collect();
        let s = ConvShape {
            batch: 1, c_in: 1, c_out: 1,
            h: 3, w: 3, k_h: 1, k_w: 1,
            stride: (1, 1), padding: (0, 0), groups: 1,
        };
        let mut col = vec![0.0_f32; s.im2col_len()];
        im2col(&x, &s, &mut col);
        assert_eq!(col, x);
    }

    /// conv2d_via_gemm with a hand-rolled gemm should match
    /// conv2d_direct exactly (deterministic accumulation order).
    #[test]
    fn via_gemm_matches_direct() {
        let s = ConvShape {
            batch: 2, c_in: 4, c_out: 6,
            h: 5, w: 5, k_h: 3, k_w: 3,
            stride: (1, 1), padding: (1, 1), groups: 2,
        };
        let x: Vec<f32> = (0..s.batch * s.c_in * s.h * s.w)
            .map(|i| (i as f32) * 0.013 - 0.4).collect();
        let w: Vec<f32> = (0..s.c_out * s.c_in_per_group() * s.k_h * s.k_w)
            .map(|i| ((i as f32) * 0.07).sin()).collect();
        let bias: Vec<f32> = (0..s.c_out).map(|i| (i as f32) * 0.01).collect();

        let mut direct_out = vec![0.0_f32; s.output_len()];
        conv2d_direct(&x, &w, Some(&bias), &s, &mut direct_out);

        let mut gemm_out = vec![0.0_f32; s.output_len()];
        let mut patches = vec![0.0_f32; s.im2col_len()];
        conv2d_via_gemm(
            &x, &w, Some(&bias), &s, &mut gemm_out, &mut patches,
            |m, n, k, a, b, c| {
                // Naive row-major gemm: c = a @ b. Caller pre-zeroes
                // not required here — overwrite each cell.
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0.0_f32;
                        for kk in 0..k {
                            acc += a[i * k + kk] * b[kk * n + j];
                        }
                        c[i * n + j] = acc;
                    }
                }
            },
        );

        for (i, (&a, &b)) in direct_out.iter().zip(gemm_out.iter()).enumerate() {
            let denom = a.abs().max(b.abs()).max(f32::MIN_POSITIVE);
            let rel = (a - b).abs() / denom;
            assert!(rel < 1e-5, "mismatch at {i}: direct={a}, via_gemm={b}");
        }
    }

    #[test]
    fn validate_rejects_bad_shapes() {
        assert!(ConvShape {
            batch: 1, c_in: 5, c_out: 4,
            h: 3, w: 3, k_h: 1, k_w: 1,
            stride: (1, 1), padding: (0, 0), groups: 2,  // 5 not div by 2
        }.validate().is_err());

        assert!(ConvShape {
            batch: 1, c_in: 1, c_out: 1,
            h: 3, w: 3, k_h: 4, k_w: 4,  // larger than padded input
            stride: (1, 1), padding: (0, 0), groups: 1,
        }.validate().is_err());
    }
}
