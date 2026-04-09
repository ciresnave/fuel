//! CPU backend operation helper structs and functions.
//!
//! These are the core computation kernels used by the CPU backend's
//! BackendStorage and BackendDevice implementations.

use crate::utils::{
    Map1, Map1Any, Map2, Map2InPlace, Map2U8, binary_map, binary_map_vec, unary_map, unary_map_vec,
};
use candle_core_types::op::{BinaryOpT, CmpOp, UnaryOpT};
use candle_core_types::{CpuStorage, DType, Error, IntDType, Layout, Result, Shape, WithDType};
use rayon::prelude::*;

/// Get the number of threads to use for parallelism.
pub(crate) fn get_num_threads() -> usize {
    match std::env::var("RAYON_NUM_THREADS") {
        Ok(s) => s.parse::<usize>().unwrap_or_else(|_| num_cpus::get()),
        Err(_) => num_cpus::get(),
    }
}
pub struct Cmp(pub CmpOp);
impl Map2U8 for Cmp {
    const OP: &'static str = "cmp";
    #[inline(always)]
    fn f<T: WithDType>(
        &self,
        lhs: &[T],
        lhs_l: &Layout,
        rhs: &[T],
        rhs_l: &Layout,
    ) -> Result<Vec<u8>> {
        let dst = match self.0 {
            CmpOp::Eq => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x == y)),
            CmpOp::Ne => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x != y)),
            CmpOp::Lt => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x < y)),
            CmpOp::Le => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x <= y)),
            CmpOp::Gt => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x > y)),
            CmpOp::Ge => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x >= y)),
        };
        Ok(dst)
    }
}

pub struct WCond<'a, T: IntDType>(pub &'a [T], pub &'a Layout);

impl<I: IntDType> Map2 for WCond<'_, I> {
    const OP: &'static str = "where";
    #[inline(always)]
    fn f<T: WithDType>(&self, t: &[T], t_l: &Layout, f: &[T], f_l: &Layout) -> Result<Vec<T>> {
        let vs = match (
            self.1.contiguous_offsets(),
            t_l.contiguous_offsets(),
            f_l.contiguous_offsets(),
        ) {
            (Some((o1, o2)), Some((o_t1, o_t2)), Some((o_f1, o_f2))) => {
                let pred = &self.0[o1..o2];
                let t = &t[o_t1..o_t2];
                let f = &f[o_f1..o_f2];
                pred.iter()
                    .zip(t.iter().zip(f.iter()))
                    .map(|(p, (&t, &f))| if p.is_true() { t } else { f })
                    .collect::<Vec<_>>()
            }
            _ => self
                .1
                .strided_index()
                .zip(t_l.strided_index().zip(f_l.strided_index()))
                .map(|(i_p, (i_t, i_f))| {
                    if self.0[i_p].is_true() {
                        t[i_t]
                    } else {
                        f[i_f]
                    }
                })
                .collect::<Vec<_>>(),
        };
        Ok(vs)
    }
}

pub struct ReduceIndex {
    pub reduce_dim_index: usize,
    pub use_min: bool,
    pub return_index: bool,
}

impl ReduceIndex {
    // The value gets replaced if f(s[current_acc], s[i]) returns true.
    #[inline(always)]
    fn fold_impl<T, U, F, G>(&self, src: &[T], src_l: &Layout, f: F, g: G) -> Result<Vec<U>>
    where
        T: Clone + Copy,
        U: Clone + Copy,
        F: Fn(T, T) -> bool,
        G: Fn(T, usize) -> U,
    {
        let reduce_dim_size = src_l.dims()[self.reduce_dim_index];
        let reduce_dim_stride = src_l.stride()[self.reduce_dim_index];
        let dst_len = src_l.shape().elem_count() / reduce_dim_size;
        let mut dst: Vec<U> = Vec::with_capacity(dst_len);
        let dst_to_set = dst.spare_capacity_mut();
        let dst_to_set =
            unsafe { std::mem::transmute::<&mut [std::mem::MaybeUninit<U>], &mut [U]>(dst_to_set) };
        match src_l.contiguous_offsets() {
            Some((o1, o2)) => {
                let src = &src[o1..o2];
                if reduce_dim_stride == 1 {
                    for (start_src_i, dst_v) in dst_to_set.iter_mut().enumerate() {
                        let start_src_i = start_src_i * reduce_dim_size;
                        let src = &src[start_src_i..start_src_i + reduce_dim_size];
                        let mut acc = 0;
                        let mut val = src[0];
                        for (src_i, &s) in src.iter().enumerate() {
                            if f(val, s) {
                                acc = src_i;
                                val = s
                            }
                        }
                        *dst_v = g(val, acc)
                    }
                } else {
                    for (start_src_i, dst_v) in dst_to_set.iter_mut().enumerate() {
                        let (p, q) = (
                            start_src_i / reduce_dim_stride,
                            start_src_i % reduce_dim_stride,
                        );
                        // start_src_i = p * reduce_dim_stride + q
                        let start_src_i = p * reduce_dim_stride * reduce_dim_size + q;
                        let src = &src[start_src_i..];
                        let mut acc = 0;
                        let mut val = src[0];
                        for src_i in 0..reduce_dim_size {
                            let s = src[src_i * reduce_dim_stride];
                            if f(val, s) {
                                acc = src_i;
                                val = s
                            }
                        }
                        *dst_v = g(val, acc)
                    }
                }
            }
            None => {
                let l = src_l.narrow(self.reduce_dim_index, 0, 1)?;
                for (unstr_index, src_index) in l.strided_index().enumerate() {
                    let src = &src[src_index..];
                    let mut acc = 0;
                    let mut val = src[0];
                    for src_i in 0..reduce_dim_size {
                        let s = src[src_i * reduce_dim_stride];
                        if f(val, s) {
                            acc = src_i;
                            val = s
                        }
                    }
                    dst_to_set[unstr_index] = g(val, acc)
                }
            }
        }
        unsafe { dst.set_len(dst_len) };
        Ok(dst)
    }
}

impl Map1Any for ReduceIndex {
    #[inline(always)]
    fn f<T: WithDType, W: Fn(Vec<T>) -> CpuStorage>(
        &self,
        src: &[T],
        src_l: &Layout,
        wrap: W,
    ) -> Result<CpuStorage> {
        if src_l.shape().elem_count() == 0 {
            Err(Error::EmptyTensor { op: "reduce" }.bt())?
        }
        let dst = match (self.return_index, self.use_min) {
            (false, true) => wrap(self.fold_impl(src, src_l, |x, y| x > y, |v, _i| v)?),
            (false, false) => wrap(self.fold_impl(src, src_l, |x, y| x < y, |v, _i| v)?),
            (true, true) => {
                CpuStorage::U32(self.fold_impl(src, src_l, |x, y| x > y, |_v, i| i as u32)?)
            }
            (true, false) => {
                CpuStorage::U32(self.fold_impl(src, src_l, |x, y| x < y, |_v, i| i as u32)?)
            }
        };
        Ok(dst)
    }
}

pub struct ReduceSum<'a> {
    pub dst_shape: &'a Shape,
    pub reduce_dims: &'a [usize],
    pub reduce_dims_and_stride: Vec<(usize, usize)>,
}

impl ReduceSum<'_> {
    #[inline(always)]
    fn fold_impl<T>(&self, src: &[T], src_l: &Layout, start_elt: T) -> Result<Vec<T>>
    where
        T: WithDType,
    {
        let mut dst = vec![start_elt; self.dst_shape.elem_count()];
        match src_l.contiguous_offsets() {
            Some((o1, o2)) => {
                let src = &src[o1..o2];
                // Handle the case where we reduce over the last dimensions separately as it is
                // fairly common and easy to optimize. This rely on the layout being contiguous!
                // reduce_dims is sorted, check if it is ranging from a to n-1.
                let reduce_over_last_dims = self
                    .reduce_dims
                    .iter()
                    .rev()
                    .enumerate()
                    .all(|(i, &v)| v == src_l.shape().rank() - 1 - i);
                if reduce_over_last_dims {
                    let reduce_sz = self
                        .reduce_dims_and_stride
                        .iter()
                        .map(|(u, _)| u)
                        .product::<usize>();
                    for (dst_i, dst_v) in dst.iter_mut().enumerate() {
                        let src_i = dst_i * reduce_sz;
                        unsafe {
                            T::vec_reduce_sum(
                                src[src_i..src_i + reduce_sz].as_ptr(),
                                dst_v,
                                reduce_sz,
                            )
                        };
                    }
                    return Ok(dst);
                };
                for (unstr_index, &src) in src.iter().enumerate() {
                    let mut dst_index = unstr_index;
                    // Set the reduce_dims indexes to 0.
                    for &(dim, stride) in self.reduce_dims_and_stride.iter() {
                        // The compiler is able to optimize the following in a single divmod op.
                        let (pre, post) = (dst_index / stride, dst_index % stride);
                        dst_index = (pre / dim) * stride + post;
                    }
                    dst[dst_index] += src;
                }
            }
            None => {
                for (unstr_index, src_index) in src_l.strided_index().enumerate() {
                    let mut dst_index = unstr_index;
                    // Set the reduce_dims indexes to 0.
                    for &(dim, stride) in self.reduce_dims_and_stride.iter() {
                        // The compiler is able to optimize the following in a single divmod op.
                        let (pre, post) = (dst_index / stride, dst_index % stride);
                        dst_index = (pre / dim) * stride + post;
                    }
                    dst[dst_index] += src[src_index];
                }
            }
        }
        Ok(dst)
    }
}

impl Map1 for ReduceSum<'_> {
    #[inline(always)]
    fn f<T: WithDType>(&self, src: &[T], src_l: &Layout) -> Result<Vec<T>> {
        self.fold_impl(src, src_l, T::zero())
    }
}

pub struct Affine(pub f64, pub f64);

impl Map1 for Affine {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>> {
        let mul = T::from_f64(self.0);
        let add = T::from_f64(self.1);
        Ok(unary_map(vs, layout, |v| v * mul + add))
    }
}

pub struct AvgPool2D(pub (usize, usize), pub (usize, usize));

impl Map1 for AvgPool2D {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        // https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
        let (k_h, k_w) = self.0;
        let (s_h, s_w) = self.1;
        let (b_sz, c, h, w) = layout.shape().dims4()?;
        let stride = layout.stride();
        let (stride_h, stride_w) = (stride[2], stride[3]);
        let h_out = (h - k_h) / s_h + 1;
        let w_out = (w - k_w) / s_w + 1;
        let src_offset = layout.start_offset();
        let dst = vec![T::zero(); b_sz * c * h_out * w_out];
        let scale = 1f64 / (k_h * k_w) as f64;
        let scale = T::from_f64(scale);
        let stride_batch = stride[0];
        let stride_c = stride[1];
        (0..b_sz * c).into_par_iter().for_each(|bc_idx| {
            let b_idx = bc_idx / c;
            let c_idx = bc_idx % c;
            let dst_idx_base = b_idx * c * h_out * w_out + c_idx * h_out * w_out;
            let src_index = src_offset + b_idx * stride_batch + c_idx * stride_c;
            for h_idx in 0..h_out {
                for w_idx in 0..w_out {
                    let mut sum = T::zero();
                    for m in 0..k_h {
                        for n in 0..k_w {
                            let m = s_h * h_idx + m;
                            let n = s_w * w_idx + n;
                            sum += src[src_index + m * stride_h + n * stride_w]
                        }
                    }
                    let dst_p = dst.as_ptr();
                    // Safety: each bc_idx writes to a unique contiguous region of dst,
                    // so no two threads can write to the same location.
                    unsafe {
                        let ptr = dst_p.add(dst_idx_base + h_idx * w_out + w_idx) as *mut T;
                        *ptr = sum * scale;
                    }
                }
            }
        });
        Ok(dst)
    }
}

pub struct MaxPool2D(pub (usize, usize), pub (usize, usize));

impl Map1 for MaxPool2D {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        let (k_h, k_w) = self.0;
        let (s_h, s_w) = self.1;
        let (b_sz, c, h, w) = layout.shape().dims4()?;
        let stride = layout.stride();
        let (stride_h, stride_w) = (stride[2], stride[3]);
        let h_out = (h - k_h) / s_h + 1;
        let w_out = (w - k_w) / s_w + 1;
        let src_offset = layout.start_offset();
        let dst = vec![T::zero(); b_sz * c * h_out * w_out];
        let stride_batch = stride[0];
        let stride_c = stride[1];
        (0..b_sz * c).into_par_iter().for_each(|bc_idx| {
            let b_idx = bc_idx / c;
            let c_idx = bc_idx % c;
            let dst_idx_base = b_idx * c * h_out * w_out + c_idx * h_out * w_out;
            let src_index = src_offset + b_idx * stride_batch + c_idx * stride_c;
            for h_idx in 0..h_out {
                for w_idx in 0..w_out {
                    let mut largest =
                        src[src_index + s_h * h_idx * stride_h + s_w * w_idx * stride_w];
                    for m in 0..k_h {
                        for n in 0..k_w {
                            let m = s_h * h_idx + m;
                            let n = s_w * w_idx + n;
                            if largest < src[src_index + m * stride_h + n * stride_w] {
                                largest = src[src_index + m * stride_h + n * stride_w]
                            }
                        }
                    }
                    let dst_p = dst.as_ptr();
                    // Safety: each bc_idx writes to a unique contiguous region of dst,
                    // so no two threads can write to the same location.
                    unsafe {
                        let ptr = dst_p.add(dst_idx_base + h_idx * w_out + w_idx) as *mut T;
                        *ptr = largest;
                    }
                }
            }
        });
        Ok(dst)
    }
}

pub struct UpsampleNearest1D(pub usize);

impl Map1 for UpsampleNearest1D {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        // TODO: Specialized implementation for the case 2*sz?
        let dst_sz = self.0;
        let (b_sz, c, src_sz) = layout.shape().dims3()?;
        let stride = layout.stride();
        let stride_sz = stride[2];
        let src_index = layout.start_offset();
        let scale_sz = src_sz as f64 / dst_sz as f64;
        let mut dst = vec![T::zero(); b_sz * c * dst_sz];
        let src_idxs = (0..dst_sz)
            .map(|idx| usize::min(src_sz - 1, (idx as f64 * scale_sz) as usize))
            .collect::<Vec<_>>();
        for b_idx in 0..b_sz {
            let dst = &mut dst[b_idx * c * dst_sz..];
            let src_index = src_index + b_idx * stride[0];
            for c_idx in 0..c {
                let dst = &mut dst[c_idx * dst_sz..];
                let src_index = src_index + c_idx * stride[1];
                for (idx, src_idx) in src_idxs.iter().enumerate() {
                    dst[idx] = src[src_index + src_idx * stride_sz]
                }
            }
        }
        Ok(dst)
    }
}

pub struct UpsampleNearest2D(pub usize, pub usize);

impl Map1 for UpsampleNearest2D {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        // TODO: Specialized implementation for the case 2*h, 2*w?
        let (dst_h, dst_w) = (self.0, self.1);
        let (b_sz, c, src_h, src_w) = layout.shape().dims4()?;
        let stride = layout.stride();
        let (stride_h, stride_w) = (stride[2], stride[3]);
        let src_index = layout.start_offset();
        let scale_h = src_h as f64 / dst_h as f64;
        let scale_w = src_w as f64 / dst_w as f64;
        let mut dst = vec![T::zero(); b_sz * c * dst_h * dst_w];
        let src_h_idxs = (0..dst_h)
            .map(|h_idx| usize::min(src_h - 1, (h_idx as f64 * scale_h) as usize))
            .collect::<Vec<_>>();
        let src_w_idxs = (0..dst_w)
            .map(|w_idx| usize::min(src_w - 1, (w_idx as f64 * scale_w) as usize))
            .collect::<Vec<_>>();
        for b_idx in 0..b_sz {
            let dst = &mut dst[b_idx * c * dst_h * dst_w..];
            let src_index = src_index + b_idx * stride[0];
            for c_idx in 0..c {
                let dst = &mut dst[c_idx * dst_h * dst_w..];
                let src_index = src_index + c_idx * stride[1];
                for (h_idx, src_h_idx) in src_h_idxs.iter().enumerate() {
                    for (w_idx, src_w_idx) in src_w_idxs.iter().enumerate() {
                        let src_index = src_index + src_h_idx * stride_h + src_w_idx * stride_w;
                        dst[h_idx * dst_w + w_idx] = src[src_index]
                    }
                }
            }
        }
        Ok(dst)
    }
}

pub struct UpsampleBilinear2D {
    pub target_h: usize,
    pub target_w: usize,
    pub align_corners: bool,
    pub scale_h_factor: Option<f64>,
    pub scale_w_factor: Option<f64>,
}

impl Map1 for UpsampleBilinear2D {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        let (batch, channels, height_in, width_in) = layout.shape().dims4()?;
        let height_out = self.target_h;
        let width_out = self.target_w;

        // Early return for identity case
        if height_in == height_out && width_in == width_out {
            return Ok(src.to_vec());
        }

        let stride = layout.stride();
        let src_offset = layout.start_offset();

        // Calculate scale factors following PyTorch's area_pixel_compute_scale logic
        let scale_h = if self.align_corners {
            if height_out > 1 {
                (height_in - 1) as f64 / (height_out - 1) as f64
            } else {
                0.0
            }
        } else {
            // PyTorch's compute_scales_value logic:
            // If scale_factor was provided, use 1.0 / scale_factor
            // Otherwise, use input_size / output_size
            if let Some(scale_factor) = self.scale_h_factor {
                1.0 / scale_factor
            } else {
                height_in as f64 / height_out as f64
            }
        };

        let scale_w = if self.align_corners {
            if width_out > 1 {
                (width_in - 1) as f64 / (width_out - 1) as f64
            } else {
                0.0
            }
        } else if let Some(scale_factor) = self.scale_w_factor {
            1.0 / scale_factor
        } else {
            width_in as f64 / width_out as f64
        };

        // Precompute indices and weights for height
        let mut h_indices = Vec::with_capacity(height_out);
        for h_out in 0..height_out {
            let src_h = if self.align_corners {
                scale_h * h_out as f64
            } else {
                scale_h * (h_out as f64 + 0.5) - 0.5
            };
            let src_h_clamped = src_h.max(0.0);
            let h0 = src_h_clamped.floor() as usize;
            let h1 = (h0 + 1).min(height_in - 1);
            let weight_h = (src_h_clamped - h0 as f64).clamp(0.0, 1.0);
            h_indices.push((h0, h1, weight_h));
        }

        // Precompute indices and weights for width
        let mut w_indices = Vec::with_capacity(width_out);
        for w_out in 0..width_out {
            let src_w = if self.align_corners {
                scale_w * w_out as f64
            } else {
                scale_w * (w_out as f64 + 0.5) - 0.5
            };
            let src_w_clamped = src_w.max(0.0);
            let w0 = src_w_clamped.floor() as usize;
            let w1 = (w0 + 1).min(width_in - 1);
            let weight_w = (src_w_clamped - w0 as f64).clamp(0.0, 1.0);
            w_indices.push((w0, w1, weight_w));
        }

        // Allocate output
        let mut dst = vec![T::zero(); batch * channels * height_out * width_out];

        // Perform bilinear interpolation
        for b in 0..batch {
            for c in 0..channels {
                let base_idx = src_offset + b * stride[0] + c * stride[1];
                let dst_base = (b * channels + c) * height_out * width_out;

                for (h_out, &(h0, h1, weight_h)) in h_indices.iter().enumerate() {
                    for (w_out, &(w0, w1, weight_w)) in w_indices.iter().enumerate() {
                        // Get four neighboring pixels
                        let idx_00 = base_idx + h0 * stride[2] + w0 * stride[3];
                        let idx_10 = base_idx + h0 * stride[2] + w1 * stride[3];
                        let idx_01 = base_idx + h1 * stride[2] + w0 * stride[3];
                        let idx_11 = base_idx + h1 * stride[2] + w1 * stride[3];

                        let v00 = src[idx_00].to_f64();
                        let v10 = src[idx_10].to_f64();
                        let v01 = src[idx_01].to_f64();
                        let v11 = src[idx_11].to_f64();

                        // Bilinear interpolation
                        let v_top = v00 * (1.0 - weight_w) + v10 * weight_w;
                        let v_bottom = v01 * (1.0 - weight_w) + v11 * weight_w;
                        let value = v_top * (1.0 - weight_h) + v_bottom * weight_h;

                        dst[dst_base + h_out * width_out + w_out] = T::from_f64(value);
                    }
                }
            }
        }

        Ok(dst)
    }
}

pub struct Gather<'a, I: IntDType> {
    pub ids: &'a [I],
    pub ids_l: &'a Layout,
    pub dim: usize,
}

impl<I: IntDType> Map1 for Gather<'_, I> {
    fn f<T: WithDType>(&self, src: &[T], src_l: &Layout) -> Result<Vec<T>> {
        let ids = match self.ids_l.contiguous_offsets() {
            Some((a, b)) => &self.ids[a..b],
            None => Err(Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let src = match src_l.contiguous_offsets() {
            Some((a, b)) => &src[a..b],
            None => Err(Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let dim = self.dim;
        let ids_dims = self.ids_l.dims();
        let src_dims = src_l.dims();
        let dst_len: usize = ids_dims.iter().product();
        let dst_left_len: usize = ids_dims[..dim].iter().product();
        let dst_dim_len = ids_dims[dim];
        let dst_right_len: usize = ids_dims[dim + 1..].iter().product();

        let src_dim_len = src_dims[dim];
        let src_right_len: usize = src_dims[dim + 1..].iter().product();

        let mut dst = vec![T::zero(); dst_len];
        for left_i in 0..dst_left_len {
            let start_src_idx = left_i * src_right_len * src_dim_len;
            let start_dst_idx = left_i * dst_right_len * dst_dim_len;
            for i in 0..dst_dim_len {
                let start_dst_idx = start_dst_idx + i * dst_right_len;
                for right_i in 0..dst_right_len {
                    let dst_idx = start_dst_idx + right_i;
                    let index = ids[dst_idx];
                    if index == I::max_value() {
                        dst[dst_idx] = T::zero();
                    } else {
                        let index = index.as_usize();
                        if index >= src_dim_len {
                            Err(Error::InvalidIndex {
                                index,
                                size: src_dim_len,
                                op: "gather",
                            }
                            .bt())?
                        }
                        let src_idx = start_src_idx + index * src_right_len + right_i;
                        dst[dst_idx] = src[src_idx]
                    }
                }
            }
        }
        Ok(dst)
    }
}

pub struct IndexSelect<'a, T: IntDType> {
    pub ids: &'a [T],
    pub ids_l: &'a Layout,
    pub dim: usize,
}

impl<I: IntDType> Map1 for IndexSelect<'_, I> {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        let src = match layout.contiguous_offsets() {
            Some((a, b)) => &src[a..b],
            None => Err(Error::RequiresContiguous { op: "index-select" }.bt())?,
        };
        let dim = self.dim;
        let n_ids = match self.ids_l.dims() {
            [n_ids] => *n_ids,
            d => Err(Error::UnexpectedNumberOfDims {
                expected: 1,
                got: d.len(),
                shape: self.ids_l.shape().clone(),
            }
            .bt())?,
        };
        let stride_ids = self.ids_l.stride()[0];
        let mut dst_dims = layout.dims().to_vec();
        let src_dim = dst_dims[dim];
        dst_dims[dim] = n_ids;
        let dst_len: usize = dst_dims.iter().product();
        let left_len: usize = dst_dims[..dim].iter().product();
        let right_len: usize = dst_dims[dim + 1..].iter().product();
        let mut dst = vec![T::zero(); dst_len];
        for left_i in 0..left_len {
            let start_src_idx = left_i * right_len * src_dim;
            let start_dst_idx = left_i * right_len * n_ids;
            for i in 0..n_ids {
                let start_dst_idx = start_dst_idx + i * right_len;
                let index = self.ids[self.ids_l.start_offset() + stride_ids * i];
                if index == I::max_value() {
                    dst[start_dst_idx..start_dst_idx + right_len].fill(T::zero());
                } else {
                    let index = index.as_usize();
                    if index >= src_dim {
                        Err(Error::InvalidIndex {
                            index,
                            size: src_dim,
                            op: "index-select",
                        }
                        .bt())?
                    }
                    let start_src_idx = start_src_idx + index * right_len;
                    dst[start_dst_idx..start_dst_idx + right_len]
                        .copy_from_slice(&src[start_src_idx..start_src_idx + right_len])
                }
            }
        }
        Ok(dst)
    }
}

pub trait ElemUpdate {
    fn f<T: WithDType>(dst: &mut T, src: T);
}

pub struct Set;
pub struct Add;

impl ElemUpdate for Set {
    fn f<T: WithDType>(dst: &mut T, src: T) {
        *dst = src
    }
}

impl ElemUpdate for Add {
    fn f<T: WithDType>(dst: &mut T, src: T) {
        *dst += src
    }
}

pub struct Scatter<'a, I: IntDType, M: ElemUpdate> {
    pub ids: &'a [I],
    pub ids_l: &'a Layout,
    pub dim: usize,
    pub _phantom: std::marker::PhantomData<M>,
}

impl<'a, I: IntDType, M: ElemUpdate> Scatter<'a, I, M> {
    pub fn new(ids: &'a [I], ids_l: &'a Layout, dim: usize) -> Self {
        Self {
            ids,
            ids_l,
            dim,
            _phantom: Default::default(),
        }
    }
}

impl<I: IntDType, M: ElemUpdate> Map2InPlace for Scatter<'_, I, M> {
    const OP: &'static str = "scatter";
    fn f<T: WithDType>(
        &self,
        dst: &mut [T],
        dst_l: &Layout,
        src: &[T],
        src_l: &Layout,
    ) -> Result<()> {
        let dst = match dst_l.contiguous_offsets() {
            None => Err(Error::RequiresContiguous { op: "scatter" }.bt())?,
            Some((o1, o2)) => &mut dst[o1..o2],
        };

        let src = match src_l.contiguous_offsets() {
            None => Err(Error::RequiresContiguous { op: "scatter" }.bt())?,
            Some((o1, o2)) => &src[o1..o2],
        };

        let dim = self.dim;
        let ids_dims = self.ids_l.dims();
        let dst_dims = dst_l.dims();
        let dst_dim_len = dst_dims[dim];
        let dst_right_len: usize = dst_dims[dim + 1..].iter().product();

        let ids_left_len: usize = ids_dims[..dim].iter().product();
        let ids_dim_len = ids_dims[dim];
        let ids_right_len: usize = ids_dims[dim + 1..].iter().product();

        let ids = match self.ids_l.contiguous_offsets() {
            Some((a, b)) => &self.ids[a..b],
            None => Err(Error::RequiresContiguous { op: "scatter" }.bt())?,
        };
        for left_i in 0..ids_left_len {
            let start_ids_idx = left_i * ids_right_len * ids_dim_len;
            let start_dst_idx = left_i * dst_right_len * dst_dim_len;
            for i in 0..ids_dim_len {
                let start_ids_idx = start_ids_idx + i * ids_right_len;
                for right_i in 0..dst_right_len {
                    let ids_idx = start_ids_idx + right_i;
                    let index = ids[ids_idx];
                    if index == I::max_value() {
                        continue;
                    }
                    let index = index.as_usize();
                    if index >= dst_dim_len {
                        Err(Error::InvalidIndex {
                            index,
                            size: dst_dim_len,
                            op: "gather",
                        }
                        .bt())?
                    }
                    let dst_idx = start_dst_idx + index * dst_right_len + right_i;
                    M::f(&mut dst[dst_idx], src[ids_idx])
                }
            }
        }

        Ok(())
    }
}

pub struct IndexAdd<'a, I: IntDType> {
    pub ids: &'a [I],
    pub dim: usize,
}

impl<I: IntDType> Map2 for IndexAdd<'_, I> {
    const OP: &'static str = "index-add";
    // https://pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html#torch.Tensor.index_add_
    // v1, l1 -> self
    fn f<T: WithDType>(&self, v1: &[T], l1: &Layout, src: &[T], src_l: &Layout) -> Result<Vec<T>> {
        let dst_len = l1.shape().elem_count();
        let mut dst = vec![T::zero(); dst_len];
        copy_strided_src_(v1, &mut dst, 0, l1);
        let src = match src_l.contiguous_offsets() {
            None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
            Some((o1, o2)) => &src[o1..o2],
        };
        let dim = self.dim;
        let max_idx = l1.dims()[dim];
        let pre_dim = src_l.dims()[..dim].iter().product::<usize>();
        let src_dim_sz = src_l.dims()[dim];
        let post_dim = src_l.dims()[dim + 1..].iter().product::<usize>();
        if dim == 0 {
            for (src_idx, dst_idx) in self.ids.iter().enumerate() {
                if *dst_idx == I::max_value() {
                    continue;
                }
                let dst_idx = dst_idx.as_usize();
                if dst_idx >= max_idx {
                    Err(Error::InvalidIndex {
                        index: dst_idx,
                        op: "index-add",
                        size: max_idx,
                    })?
                }
                let src_idx = src_idx * post_dim;
                let dst_idx = dst_idx * post_dim;
                let src = &src[src_idx..src_idx + post_dim];
                let dst = &mut dst[dst_idx..dst_idx + post_dim];
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d += s
                }
            }
        } else {
            for (src_idx, dst_idx) in self.ids.iter().enumerate() {
                if *dst_idx == I::max_value() {
                    continue;
                }
                let dst_idx = dst_idx.as_usize();
                if dst_idx >= max_idx {
                    Err(Error::InvalidIndex {
                        index: dst_idx,
                        op: "index-add",
                        size: max_idx,
                    })?
                }
                for pre_i in 0..pre_dim {
                    let pre_src_i = (pre_i * src_dim_sz + src_idx) * post_dim;
                    let pre_dst_i = (pre_i * max_idx + dst_idx) * post_dim;
                    let src = &src[pre_src_i..pre_src_i + post_dim];
                    let dst = &mut dst[pre_dst_i..pre_dst_i + post_dim];
                    for (d, &s) in dst.iter_mut().zip(src.iter()) {
                        *d += s
                    }
                }
            }
        }
        Ok(dst)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn copy2d_<T: Copy>(
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
        let dst = &mut dst[dst_idx..dst_idx + d2];
        let src = &src[src_idx..src_idx + d2];
        dst.copy_from_slice(src)
    }
}

/// Minimum element count to switch from sequential to parallel strided copy.
pub const PAR_STRIDED_COPY_THRESHOLD: usize = 100_000;

/// Compute the source offset for a given linear destination index by decomposing
/// it into multi-dimensional coordinates (row-major order) and applying strides.
#[inline]
pub fn compute_strided_offset(
    linear_idx: usize,
    dims: &[usize],
    strides: &[usize],
    start_offset: usize,
) -> usize {
    let mut offset = start_offset;
    let mut remaining = linear_idx;
    for (&d, &s) in dims.iter().zip(strides.iter()).rev() {
        offset += (remaining % d) * s;
        remaining /= d;
    }
    offset
}

pub fn copy_strided_src_<T: Copy + Send + Sync>(
    src: &[T],
    dst: &mut [T],
    dst_offset: usize,
    src_l: &Layout,
) {
    match src_l.strided_blocks() {
        candle_core_types::StridedBlocks::SingleBlock { start_offset, len } => {
            let to_copy = (dst.len() - dst_offset).min(len);
            dst[dst_offset..dst_offset + to_copy]
                .copy_from_slice(&src[start_offset..start_offset + to_copy])
        }
        candle_core_types::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len: 1,
        } => {
            let dims = src_l.dims();
            let strides = src_l.stride();
            let start = src_l.start_offset();
            let elem_count: usize = dims.iter().product();
            let elem_count = elem_count.min(dst.len() - dst_offset);

            if elem_count >= PAR_STRIDED_COPY_THRESHOLD {
                drop(block_start_index);
                dst[dst_offset..dst_offset + elem_count]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, dst_val)| {
                        let src_offset = compute_strided_offset(i, dims, strides, start);
                        *dst_val = src[src_offset];
                    });
            } else {
                for (dst_index, src_index) in block_start_index.enumerate() {
                    let dst_index = dst_index + dst_offset;
                    if dst_index >= dst.len() {
                        break;
                    }
                    dst[dst_index] = src[src_index]
                }
            }
        }
        candle_core_types::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            let dims = src_l.dims();
            let strides = src_l.stride();
            let start = src_l.start_offset();
            // Recompute the number of non-contiguous leading dimensions (same logic
            // as Layout::strided_blocks) so we can index blocks directly.
            let mut bl = 1;
            let mut contiguous_dims = 0;
            for (&stride, &dim) in strides.iter().zip(dims.iter()).rev() {
                if stride != bl {
                    break;
                }
                bl *= dim;
                contiguous_dims += 1;
            }
            let index_dims = dims.len() - contiguous_dims;
            let index_dims_slice = &dims[..index_dims];
            let index_strides_slice = &strides[..index_dims];
            let num_blocks: usize = index_dims_slice.iter().product();
            let total_elems = (num_blocks * block_len).min(dst.len() - dst_offset);

            if total_elems >= PAR_STRIDED_COPY_THRESHOLD {
                drop(block_start_index);
                let full_blocks = total_elems / block_len;
                dst[dst_offset..dst_offset + full_blocks * block_len]
                    .par_chunks_mut(block_len)
                    .enumerate()
                    .for_each(|(block_idx, dst_chunk)| {
                        let src_start = compute_strided_offset(
                            block_idx,
                            index_dims_slice,
                            index_strides_slice,
                            start,
                        );
                        dst_chunk.copy_from_slice(&src[src_start..src_start + block_len]);
                    });
                // Handle a possible partial trailing block.
                let remaining = total_elems % block_len;
                if remaining > 0 {
                    let last_dst_start = dst_offset + full_blocks * block_len;
                    let src_start = compute_strided_offset(
                        full_blocks,
                        index_dims_slice,
                        index_strides_slice,
                        start,
                    );
                    dst[last_dst_start..last_dst_start + remaining]
                        .copy_from_slice(&src[src_start..src_start + remaining]);
                }
            } else {
                let mut dst_index = dst_offset;
                for src_index in block_start_index {
                    let next_dst_index = dst_index + block_len;
                    if dst_index >= dst.len() {
                        break;
                    }
                    let to_copy = usize::min(block_len, dst.len() - dst_index);
                    dst[dst_index..dst_index + to_copy]
                        .copy_from_slice(&src[src_index..src_index + to_copy]);
                    dst_index = next_dst_index
                }
            }
        }
    }
}

pub struct Conv1D<'a>(pub &'a candle_core_types::conv::ParamsConv1D);

impl Map2 for Conv1D<'_> {
    const OP: &'static str = "conv1d";
    fn f<T: WithDType>(&self, inp: &[T], inp_l: &Layout, k: &[T], k_l: &Layout) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let k = &k[k_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2) = candle_core_types::shape::dims3(inp_l.stride())?;
        let (k_s0, k_s1, k_s2) = candle_core_types::shape::dims3(k_l.stride())?;
        let l_out = p.l_out();
        let dst_elems = p.c_out * l_out * p.b_size;
        // The output shape is [b_size, c_out, l_out]
        let dst = vec![T::zero(); dst_elems];

        // TODO: Avoid making this copy if `inp` already has the appropriate layout.
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.l_in];
        for b_idx in 0..p.b_size {
            for src_l in 0..p.l_in {
                for src_c_idx in 0..p.c_in {
                    let inp_idx = b_idx * inp_s0 + src_c_idx * inp_s1 + src_l * inp_s2;
                    inp_cont[b_idx * p.l_in * p.c_in + src_l * p.c_in + src_c_idx] = inp[inp_idx]
                }
            }
        }

        for offset in 0..p.k_size {
            (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                let dst_idx = dst_c_idx * l_out;
                let k_cont = (0..p.c_in)
                    .map(|c_in_idx| k[dst_c_idx * k_s0 + c_in_idx * k_s1 + offset * k_s2])
                    .collect::<Vec<_>>();
                for b_idx in 0..p.b_size {
                    let dst_idx = dst_idx + b_idx * p.c_out * l_out;
                    for dst_l in 0..l_out {
                        let dst_idx = dst_idx + dst_l;
                        let src_l = p.stride * dst_l + offset * p.dilation;
                        if src_l < p.padding || src_l >= p.padding + p.l_in {
                            continue;
                        }
                        let src_l = src_l - p.padding;
                        let inp_cont = &inp_cont[b_idx * p.l_in * p.c_in + src_l * p.c_in..];
                        assert!(inp_cont.len() >= p.c_in);
                        assert!(k_cont.len() >= p.c_in);
                        let mut d = T::zero();
                        unsafe { T::vec_dot(inp_cont.as_ptr(), k_cont.as_ptr(), &mut d, p.c_in) }
                        let dst_p = dst.as_ptr();
                        // Safety: dst_idx are uniques per dst_c_idx which is used to parallelise
                        // the different tasks so no two threads can try to write at the same
                        // location.
                        unsafe {
                            let ptr = dst_p.add(dst_idx) as *mut T;
                            *ptr += d
                        }
                    }
                }
            })
        }
        Ok(dst)
    }
}

pub struct Im2Col1D {
    pub l_k: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
}

impl Im2Col1D {
    fn l_out(&self, l: usize) -> usize {
        (l + 2 * self.padding - self.dilation * (self.l_k - 1) - 1) / self.stride + 1
    }
}

impl Map1 for Im2Col1D {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>> {
        let &Self {
            l_k,
            stride,
            dilation,
            padding,
        } = self;
        let (b, c, l) = layout.shape().dims3()?;
        let l_out = self.l_out(l);
        let src = &vs[layout.start_offset()..];
        let mut dst = vec![T::zero(); b * l_out * c * l_k];
        let (src_s0, src_s1, src_s2) = {
            let s = layout.stride();
            (s[0], s[1], s[2])
        };
        // TODO: provide specialized kernels for the common use cases.
        // - l_k = 1
        // - padding = 0
        // - stride = 1
        // - dilation = 1
        for b_idx in 0..b {
            let src_idx = b_idx * src_s0;
            let dst_idx = b_idx * l_out * c * l_k;
            for l_idx in 0..l_out {
                let dst_idx = dst_idx + l_idx * c * l_k;
                for c_idx in 0..c {
                    let dst_idx = dst_idx + c_idx * l_k;
                    let src_idx = c_idx * src_s1 + src_idx;
                    for l_k_idx in 0..l_k {
                        let src_l = l_idx * stride + l_k_idx * dilation;
                        if padding != 0 && (src_l < padding || src_l >= l + padding) {
                            continue;
                        }
                        let src_l = src_l - padding;
                        let src_idx = src_idx + src_l * src_s2;
                        let dst_idx = dst_idx + l_k_idx;
                        dst[dst_idx] = src[src_idx]
                    }
                }
            }
        }
        Ok(dst)
    }
}

pub struct Im2Col {
    pub h_k: usize,
    pub w_k: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
}

impl Im2Col {
    fn hw_out(&self, h: usize, w: usize) -> (usize, usize) {
        let h_out = (h + 2 * self.padding - self.dilation * (self.h_k - 1) - 1) / self.stride + 1;
        let w_out = (w + 2 * self.padding - self.dilation * (self.w_k - 1) - 1) / self.stride + 1;
        (h_out, w_out)
    }
}

impl Map1 for Im2Col {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>> {
        let &Self {
            h_k,
            w_k,
            stride,
            dilation,
            padding,
        } = self;
        let (b, c, h, w) = layout.shape().dims4()?;
        let (h_out, w_out) = self.hw_out(h, w);
        let src = &vs[layout.start_offset()..];
        let mut dst = vec![T::zero(); b * h_out * w_out * c * h_k * w_k];
        let (src_s0, src_s1, src_s2, src_s3) = {
            let s = layout.stride();
            (s[0], s[1], s[2], s[3])
        };
        // TODO: provide specialized kernels for the common use cases.
        // - h_k = w_k = 1
        // - padding = 0
        // - stride = 1
        // - dilation = 1
        for b_idx in 0..b {
            let src_idx = b_idx * src_s0;
            let dst_idx = b_idx * h_out * w_out * c * h_k * w_k;
            for h_idx in 0..h_out {
                let dst_idx = dst_idx + h_idx * w_out * c * h_k * w_k;
                for w_idx in 0..w_out {
                    let dst_idx = dst_idx + w_idx * c * h_k * w_k;
                    for c_idx in 0..c {
                        let dst_idx = dst_idx + c_idx * h_k * w_k;
                        let src_idx = c_idx * src_s1 + src_idx;
                        for h_k_idx in 0..h_k {
                            let src_h = h_idx * stride + h_k_idx * dilation;
                            if padding != 0 && (src_h < padding || src_h >= h + padding) {
                                continue;
                            }
                            let src_h = src_h - padding;
                            let src_idx = src_idx + src_h * src_s2;
                            let dst_idx = dst_idx + h_k_idx * w_k;
                            for w_k_idx in 0..w_k {
                                let src_w = w_idx * stride + w_k_idx * dilation;
                                if padding != 0 && (src_w < padding || src_w >= w + padding) {
                                    continue;
                                }
                                let src_w = src_w - padding;
                                let src_idx = src_idx + src_w * src_s3;
                                let dst_idx = dst_idx + w_k_idx;
                                dst[dst_idx] = src[src_idx]
                            }
                        }
                    }
                }
            }
        }
        Ok(dst)
    }
}

pub struct Col2Im1D {
    pub stride: usize,
}

impl Map1 for Col2Im1D {
    fn f<T: WithDType>(&self, col: &[T], l: &Layout) -> Result<Vec<T>> {
        let (b_size, l_in, c_out, k_size) = l.shape().dims4()?;
        let stride = self.stride;
        let l_out = (l_in - 1) * stride + k_size;
        let mut im = vec![T::zero(); b_size * c_out * l_out];
        let (dst_s0, dst_s1) = (c_out * l_out, l_out);
        let (src_s0, src_s1, src_s2) = (c_out * k_size * l_in, c_out * k_size, k_size);
        for l_in_i in 0..l_in {
            for k_i in 0..k_size {
                let l_out_i = l_in_i * stride + k_i;
                for b_i in 0..b_size {
                    for c_i in 0..c_out {
                        let dst_idx = b_i * dst_s0 + c_i * dst_s1 + l_out_i;
                        let src_idx = b_i * src_s0 + l_in_i * src_s1 + c_i * src_s2 + k_i;
                        im[dst_idx] += col[src_idx]
                    }
                }
            }
        }
        Ok(im)
    }
}

pub struct ConvTranspose1D<'a>(pub &'a candle_core_types::conv::ParamsConvTranspose1D);

impl Map2 for ConvTranspose1D<'_> {
    const OP: &'static str = "conv_transpose1d";
    fn f<T: WithDType>(&self, inp: &[T], inp_l: &Layout, k: &[T], k_l: &Layout) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let k = &k[k_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2) = candle_core_types::shape::dims3(inp_l.stride())?;
        let (k_s0, k_s1, k_s2) = candle_core_types::shape::dims3(k_l.stride())?;
        let l_out = p.l_out();

        // Output shape: [b_size, c_out, l_out].
        let dst_elems = p.c_out * l_out * p.b_size;
        let dst = vec![T::zero(); dst_elems];
        let dst_s0 = p.c_out * l_out;
        let dst_s1 = l_out;
        let dst_s2 = 1;

        // TODO: Avoid making this copy if `inp` already has the appropriate layout.
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.l_in];
        let cont_s0 = p.l_in * p.c_in;
        let cont_s1 = p.c_in;
        for b_idx in 0..p.b_size {
            for l_idx in 0..p.l_in {
                for c_idx in 0..p.c_in {
                    let src_idx = b_idx * inp_s0 + c_idx * inp_s1 + l_idx * inp_s2;
                    let dst_idx = b_idx * cont_s0 + l_idx * cont_s1 + c_idx;
                    inp_cont[dst_idx] = inp[src_idx]
                }
            }
        }

        for k_idx in 0..p.k_size {
            (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                let k_cont = (0..p.c_in)
                    .map(|c_in_idx| k[c_in_idx * k_s0 + dst_c_idx * k_s1 + k_idx * k_s2])
                    .collect::<Vec<_>>();
                for b_idx in 0..p.b_size {
                    for l_idx in 0..p.l_in {
                        let out_idx = l_idx * p.stride + k_idx * p.dilation;
                        if out_idx < p.padding {
                            continue;
                        }
                        let out_idx = out_idx - p.padding;
                        if out_idx < l_out {
                            let inp_cont = &inp_cont[b_idx * cont_s0 + l_idx * cont_s1..];
                            let dst_idx = b_idx * dst_s0 + out_idx * dst_s2 + dst_c_idx * dst_s1;
                            let mut d = T::zero();
                            unsafe {
                                T::vec_dot(inp_cont.as_ptr(), k_cont.as_ptr(), &mut d, p.c_in)
                            }
                            let dst_p = dst.as_ptr();
                            // Safety: dst_idx are uniques per dst_c_idx which is used to
                            // parallelise the different tasks so no two threads can try to
                            // write at the same location.
                            unsafe {
                                let ptr = dst_p.add(dst_idx) as *mut T;
                                *ptr += d
                            }
                        }
                    }
                }
            })
        }
        Ok(dst)
    }
}

pub struct ConvTranspose2D<'a>(pub &'a candle_core_types::conv::ParamsConvTranspose2D);

impl Map2 for ConvTranspose2D<'_> {
    const OP: &'static str = "conv_transpose2d";
    fn f<T: WithDType>(&self, inp: &[T], inp_l: &Layout, k: &[T], k_l: &Layout) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2, inp_s3) = candle_core_types::shape::dims4(inp_l.stride())?;
        let k = &k[k_l.start_offset()..];
        let (k_s0, k_s1, k_s2, k_s3) = candle_core_types::shape::dims4(k_l.stride())?;
        let (out_h, out_w) = (p.out_h(), p.out_w());

        // Output shape: [b_size, c_out, out_h, out_w].
        let dst = vec![T::zero(); p.b_size * p.c_out * out_h * out_w];
        let dst_s0 = p.c_out * out_h * out_w;
        let dst_s1 = out_h * out_w;
        let dst_s2 = out_w;
        let dst_s3 = 1;

        // TODO: Avoid making this copy if `inp` already has the appropriate layout.
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.i_h * p.i_w];
        let cont_s0 = p.i_h * p.i_w * p.c_in;
        let cont_s1 = p.i_w * p.c_in;
        let cont_s2 = p.c_in;
        for b_idx in 0..p.b_size {
            for h_idx in 0..p.i_h {
                for w_idx in 0..p.i_w {
                    for c_idx in 0..p.c_in {
                        let src_idx =
                            b_idx * inp_s0 + c_idx * inp_s1 + h_idx * inp_s2 + w_idx * inp_s3;
                        let dst_idx = b_idx * cont_s0 + h_idx * cont_s1 + w_idx * cont_s2 + c_idx;
                        inp_cont[dst_idx] = inp[src_idx]
                    }
                }
            }
        }

        for k_y in 0..p.k_h {
            for k_x in 0..p.k_w {
                (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                    let k_cont = (0..p.c_in)
                        .map(|c_in_idx| {
                            k[c_in_idx * k_s0 + dst_c_idx * k_s1 + k_y * k_s2 + k_x * k_s3]
                        })
                        .collect::<Vec<_>>();
                    for b_idx in 0..p.b_size {
                        for inp_y in 0..p.i_h {
                            for inp_x in 0..p.i_w {
                                let out_x = inp_x * p.stride + k_x * p.dilation;
                                let out_y = inp_y * p.stride + k_y * p.dilation;
                                if out_x < p.padding || out_y < p.padding {
                                    continue;
                                }
                                let out_x = out_x - p.padding;
                                let out_y = out_y - p.padding;
                                if out_x < out_w && out_y < out_h {
                                    let inp_cont = &inp_cont
                                        [b_idx * cont_s0 + inp_y * cont_s1 + inp_x * cont_s2..];
                                    let dst_idx = b_idx * dst_s0
                                        + out_y * dst_s2
                                        + out_x * dst_s3
                                        + dst_c_idx * dst_s1;
                                    let mut d = T::zero();
                                    unsafe {
                                        T::vec_dot(
                                            inp_cont.as_ptr(),
                                            k_cont.as_ptr(),
                                            &mut d,
                                            p.c_in,
                                        )
                                    }
                                    let dst_p = dst.as_ptr();
                                    // Safety: dst_idx are uniques per dst_c_idx which is used to
                                    // parallelise the different tasks so no two threads can try to
                                    // write at the same location.
                                    unsafe {
                                        let ptr = dst_p.add(dst_idx) as *mut T;
                                        *ptr += d
                                    }
                                }
                            }
                        }
                    }
                })
            }
        }
        Ok(dst)
    }
}

pub struct MatMul(pub (usize, usize, usize, usize));

impl MatMul {
    fn striding_error(&self, lhs_l: &Layout, rhs_l: &Layout, msg: &'static str) -> Error {
        Error::MatMulUnexpectedStriding(Box::new(
            candle_core_types::error::MatMulUnexpectedStriding {
                lhs_l: lhs_l.clone(),
                rhs_l: rhs_l.clone(),
                bmnk: self.0,
                msg,
            },
        ))
        .bt()
    }

    fn ab_skip(&self, lhs_l: &Layout, rhs_l: &Layout) -> Result<(usize, usize)> {
        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();
        let rank = lhs_stride.len();
        let (_b, m, n, k) = self.0;
        let a_skip: usize = match lhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
            [_, stride] if lhs_l.dims()[0] == 1 => stride,
            [stride, _] if lhs_l.dims()[1] == 1 => stride,
            [stride] => stride,
            [] => m * k,
            _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?,
        };
        let b_skip: usize = match rhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
            [_, stride] if rhs_l.dims()[0] == 1 => stride,
            [stride, _] if rhs_l.dims()[1] == 1 => stride,
            [stride] => stride,
            [] => n * k,
            _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?,
        };
        Ok((a_skip, b_skip))
    }
}

impl Map2 for MatMul {
    const OP: &'static str = "mat_mul";

    #[cfg(all(not(feature = "mkl"), not(feature = "accelerate")))]
    fn f<T: 'static + WithDType + num_traits::Num + Copy>(
        &self,
        lhs: &[T],
        lhs_l: &Layout,
        rhs: &[T],
        rhs_l: &Layout,
    ) -> Result<Vec<T>> {
        use gemm::{Parallelism, gemm};

        match T::DTYPE {
            DType::F16 | DType::F32 | DType::F64 => {}
            _ => Err(Error::UnsupportedDTypeForOp(T::DTYPE, "matmul").bt())?,
        }

        let (b, m, n, k) = self.0;
        let lhs = &lhs[lhs_l.start_offset()..];
        let rhs = &rhs[rhs_l.start_offset()..];

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();
        let rank = lhs_stride.len();
        let lhs_cs = lhs_stride[rank - 1];
        let lhs_rs = lhs_stride[rank - 2];

        let rhs_cs = rhs_stride[rank - 1];
        let rhs_rs = rhs_stride[rank - 2];

        let (a_skip, b_skip) = self.ab_skip(lhs_l, rhs_l)?;
        let c_skip: usize = m * n;

        let dst_shape: Shape = (m, n).into();
        let dst_strides = dst_shape.stride_contiguous();
        let dst_rs = dst_strides[0];
        let dst_cs = dst_strides[1];

        let mut dst = vec![T::zero(); b * m * n];
        let num_threads = crate::ops::get_num_threads();
        let parallelism = if num_threads > 1 {
            Parallelism::Rayon(num_threads)
        } else {
            Parallelism::None
        };
        let (b, m, n, k) = if b_skip == 0 && a_skip == m * k {
            // a_skip and c_skip should be updated but step is always 0 so
            // it wouldn't matter.
            (1, b * m, n, k)
        } else if a_skip == 0 && b_skip == n * k {
            (1, m, b * n, k)
        } else {
            (b, m, n, k)
        };
        if b > 1 && num_threads > 1 && m * n * k < 100_000 && c_skip > 0 {
            // Small matrices with multiple batch elements: parallelize across
            // the batch dimension instead of within each gemm call. This is
            // beneficial for workloads like transformer attention where batch
            // sizes are large but individual matrix dimensions are small.
            // Each per-batch gemm uses Parallelism::None to avoid
            // over-subscribing threads.
            //
            // Safety of par_chunks_mut: each chunk is a disjoint
            // &mut [T] slice of length c_skip, so no data races occur.
            // lhs and rhs are immutable shared references (&[T]) which
            // are Send+Sync since T: WithDType implies T: Send+Sync.
            dst.par_chunks_mut(c_skip)
                .enumerate()
                .for_each(|(step, dst_chunk)| {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    unsafe {
                        gemm(
                            /* m: usize = */ m,
                            /* n: usize = */ n,
                            /* k: usize = */ k,
                            /* dst: *mut T = */ dst_chunk.as_mut_ptr(),
                            /* dst_cs: isize = */ dst_cs as isize,
                            /* dst_rs: isize = */ dst_rs as isize,
                            /* read_dst: bool = */ false,
                            /* lhs: *const T = */ lhs_p.as_ptr(),
                            /* lhs_cs: isize = */ lhs_cs as isize,
                            /* lhs_rs: isize = */ lhs_rs as isize,
                            /* rhs: *const T = */ rhs_p.as_ptr(),
                            /* rhs_cs: isize = */ rhs_cs as isize,
                            /* rhs_rs: isize = */ rhs_rs as isize,
                            /* alpha: T = */ T::zero(),
                            /* beta: T = */ T::one(),
                            /* conj_dst: bool = */ false,
                            /* conj_lhs: bool = */ false,
                            /* conj_rhs: bool = */ false,
                            Parallelism::None,
                        )
                    }
                });
        } else {
            // Large matrices or single batch: sequential loop. For large
            // matrices, gemm's internal parallelism is more effective than
            // batch-level parallelism.
            for step in 0..b {
                let lhs_p = &lhs[step * a_skip..];
                let rhs_p = &rhs[step * b_skip..];
                let dst_p = &mut dst[step * c_skip..];
                unsafe {
                    gemm(
                        /* m: usize = */ m,
                        /* n: usize = */ n,
                        /* k: usize = */ k,
                        /* dst: *mut T = */ dst_p.as_mut_ptr(),
                        /* dst_cs: isize = */ dst_cs as isize,
                        /* dst_rs: isize = */ dst_rs as isize,
                        /* read_dst: bool = */ false,
                        /* lhs: *const T = */ lhs_p.as_ptr(),
                        /* lhs_cs: isize = */ lhs_cs as isize,
                        /* lhs_rs: isize = */ lhs_rs as isize,
                        /* rhs: *const T = */ rhs_p.as_ptr(),
                        /* rhs_cs: isize = */ rhs_cs as isize,
                        /* rhs_rs: isize = */ rhs_rs as isize,
                        /* alpha: T = */ T::zero(),
                        /* beta: T = */ T::one(),
                        /* conj_dst: bool = */ false,
                        /* conj_lhs: bool = */ false,
                        /* conj_rhs: bool = */ false,
                        parallelism,
                    )
                }
            }
        }
        Ok(dst)
    }

    #[cfg(feature = "accelerate")]
    fn f<T: 'static + WithDType + num_traits::Num + Copy>(
        &self,
        lhs: &[T],
        lhs_l: &Layout,
        rhs: &[T],
        rhs_l: &Layout,
    ) -> Result<Vec<T>> {
        let (b, m, n, k) = self.0;
        let lhs = &lhs[lhs_l.start_offset()..];
        let rhs = &rhs[rhs_l.start_offset()..];

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();

        let (a_skip, b_skip) = self.ab_skip(lhs_l, rhs_l)?;
        let c_skip: usize = m * n;

        let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
        let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
        let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
        let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

        let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
            (n as i32, b'N')
        } else if rhs_m1 == k && rhs_m2 == 1 {
            (k as i32, b'T')
        } else {
            Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?
        };
        // The b tensor has dims batching, m, k (lhs)
        let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
            (k as i32, b'N')
        } else if lhs_m1 == m && lhs_m2 == 1 {
            (m as i32, b'T')
        } else {
            Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?
        };

        let mut dst = vec![T::zero(); b * m * n];
        match T::DTYPE {
            DType::F16 => {
                candle_core_types::bail!("the accelerate backend does not support f16 matmul")
            }
            DType::F32 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f32;
                        let b = lhs_p.as_ptr() as *const f32;
                        let c = dst_p.as_mut_ptr() as *mut f32;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::accelerate::sgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            DType::F64 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f64;
                        let b = lhs_p.as_ptr() as *const f64;
                        let c = dst_p.as_mut_ptr() as *mut f64;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::accelerate::dgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            dtype => Err(Error::UnsupportedDTypeForOp(dtype, "matmul").bt())?,
        }
        Ok(dst)
    }

    #[cfg(feature = "mkl")]
    fn f<T: 'static + WithDType + num_traits::Num + Copy>(
        &self,
        lhs: &[T],
        lhs_l: &Layout,
        rhs: &[T],
        rhs_l: &Layout,
    ) -> Result<Vec<T>> {
        let (b, m, n, k) = self.0;
        let lhs = &lhs[lhs_l.start_offset()..];
        let rhs = &rhs[rhs_l.start_offset()..];

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();

        let (a_skip, b_skip) = self.ab_skip(lhs_l, rhs_l)?;
        let c_skip: usize = m * n;

        let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
        let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
        let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
        let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

        let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
            (n as i32, b'N')
        } else if rhs_m1 == k && rhs_m2 == 1 {
            (k as i32, b'T')
        } else {
            Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?
        };
        // The b tensor has dims batching, m, k (lhs)
        let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
            (k as i32, b'N')
        } else if lhs_m1 == m && lhs_m2 == 1 {
            (m as i32, b'T')
        } else {
            Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?
        };

        let mut dst = vec![T::zero(); b * m * n];
        match T::DTYPE {
            DType::F16 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f16;
                        let b = lhs_p.as_ptr() as *const f16;
                        let c = dst_p.as_mut_ptr() as *mut f16;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::mkl::hgemm(
                            transa,
                            transb,
                            /* m= */ n as i32,
                            /* n= */ m as i32,
                            /* k= */ k as i32,
                            /* alpha= */ f16::ONE,
                            /* a= */ a,
                            /* lda= */ lda,
                            /* b= */ b,
                            /* ldb= */ ldb,
                            /* beta= */ f16::ZERO,
                            /* c= */ c,
                            /* ldc= */ n as i32,
                        )
                    }
                }
            }
            DType::F32 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f32;
                        let b = lhs_p.as_ptr() as *const f32;
                        let c = dst_p.as_mut_ptr() as *mut f32;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::mkl::sgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            DType::F64 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f64;
                        let b = lhs_p.as_ptr() as *const f64;
                        let c = dst_p.as_mut_ptr() as *mut f64;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::mkl::dgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            dtype => Err(Error::UnsupportedDTypeForOp(dtype, "matmul").bt())?,
        }
        Ok(dst)
    }
}

pub fn elu<T: num_traits::Float>(v: T, alpha: T) -> T {
    if v.is_sign_positive() {
        v
    } else {
        (v.exp() - T::one()) * alpha
    }
}

// ---------------------------------------------------------------------------
// Generic unary/binary dispatch over CpuStorage
// ---------------------------------------------------------------------------
// These functions centralise the per-dtype method selection from UnaryOpT /
// BinaryOpT so that callers only need a single call instead of a 14-arm match.

/// Apply a [`UnaryOpT`] element-wise to a [`CpuStorage`] buffer, respecting
/// the given [`Layout`].  Vectorised code-paths are used when the operator
/// advertises them (`B::F32_VEC`, etc.).
pub fn unary_dispatch<B: UnaryOpT>(storage: &CpuStorage, layout: &Layout) -> Result<CpuStorage> {
    match storage {
        CpuStorage::BF16(s) => Ok(CpuStorage::BF16(if B::BF16_VEC {
            unary_map_vec(s, layout, B::bf16, B::bf16_vec)
        } else {
            unary_map(s, layout, B::bf16)
        })),
        CpuStorage::F16(s) => Ok(CpuStorage::F16(if B::F16_VEC {
            unary_map_vec(s, layout, B::f16, B::f16_vec)
        } else {
            unary_map(s, layout, B::f16)
        })),
        CpuStorage::F32(s) => Ok(CpuStorage::F32(if B::F32_VEC {
            unary_map_vec(s, layout, B::f32, B::f32_vec)
        } else {
            unary_map(s, layout, B::f32)
        })),
        CpuStorage::F64(s) => Ok(CpuStorage::F64(if B::F64_VEC {
            unary_map_vec(s, layout, B::f64, B::f64_vec)
        } else {
            unary_map(s, layout, B::f64)
        })),
        CpuStorage::U8(s) => Ok(CpuStorage::U8(unary_map(s, layout, B::u8))),
        CpuStorage::U32(s) => Ok(CpuStorage::U32(unary_map(s, layout, B::u32))),
        CpuStorage::I16(s) => Ok(CpuStorage::I16(unary_map(s, layout, B::i16))),
        CpuStorage::I32(s) => Ok(CpuStorage::I32(unary_map(s, layout, B::i32))),
        CpuStorage::I64(s) => Ok(CpuStorage::I64(unary_map(s, layout, B::i64))),
        CpuStorage::F8E4M3(s) => Ok(CpuStorage::F8E4M3(unary_map(s, layout, B::f8e4m3))),
        CpuStorage::F6E2M3(_) => Err(Error::UnsupportedDTypeForOp(DType::F6E2M3, "unary").bt()),
        CpuStorage::F6E3M2(_) => Err(Error::UnsupportedDTypeForOp(DType::F6E3M2, "unary").bt()),
        CpuStorage::F4(_) => Err(Error::UnsupportedDTypeForOp(DType::F4, "unary").bt()),
        CpuStorage::F8E8M0(_) => Err(Error::UnsupportedDTypeForOp(DType::F8E8M0, "unary").bt()),
    }
}

/// Apply a [`BinaryOpT`] element-wise to two [`CpuStorage`] buffers.
/// Both buffers must have the same dtype; returns [`Error::DTypeMismatchBinaryOp`]
/// otherwise.  Vectorised code-paths are used when advertised.
pub fn binary_dispatch<B: BinaryOpT>(
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<CpuStorage> {
    match (lhs, rhs) {
        (CpuStorage::BF16(l), CpuStorage::BF16(r)) => Ok(CpuStorage::BF16(if B::BF16_VEC {
            binary_map_vec(lhs_l, rhs_l, l, r, B::bf16, B::bf16_vec)
        } else {
            binary_map(lhs_l, rhs_l, l, r, B::bf16)
        })),
        (CpuStorage::F16(l), CpuStorage::F16(r)) => Ok(CpuStorage::F16(if B::F16_VEC {
            binary_map_vec(lhs_l, rhs_l, l, r, B::f16, B::f16_vec)
        } else {
            binary_map(lhs_l, rhs_l, l, r, B::f16)
        })),
        (CpuStorage::F32(l), CpuStorage::F32(r)) => Ok(CpuStorage::F32(if B::F32_VEC {
            binary_map_vec(lhs_l, rhs_l, l, r, B::f32, B::f32_vec)
        } else {
            binary_map(lhs_l, rhs_l, l, r, B::f32)
        })),
        (CpuStorage::F64(l), CpuStorage::F64(r)) => Ok(CpuStorage::F64(if B::F64_VEC {
            binary_map_vec(lhs_l, rhs_l, l, r, B::f64, B::f64_vec)
        } else {
            binary_map(lhs_l, rhs_l, l, r, B::f64)
        })),
        (CpuStorage::U32(l), CpuStorage::U32(r)) => Ok(CpuStorage::U32(if B::U32_VEC {
            binary_map_vec(lhs_l, rhs_l, l, r, B::u32, B::u32_vec)
        } else {
            binary_map(lhs_l, rhs_l, l, r, B::u32)
        })),
        (CpuStorage::I16(l), CpuStorage::I16(r)) => {
            Ok(CpuStorage::I16(binary_map(lhs_l, rhs_l, l, r, B::i16)))
        }
        (CpuStorage::I32(l), CpuStorage::I32(r)) => {
            Ok(CpuStorage::I32(binary_map(lhs_l, rhs_l, l, r, B::i32)))
        }
        (CpuStorage::I64(l), CpuStorage::I64(r)) => Ok(CpuStorage::I64(if B::I64_VEC {
            binary_map_vec(lhs_l, rhs_l, l, r, B::i64, B::i64_vec)
        } else {
            binary_map(lhs_l, rhs_l, l, r, B::i64)
        })),
        (CpuStorage::U8(l), CpuStorage::U8(r)) => Ok(CpuStorage::U8(if B::U8_VEC {
            binary_map_vec(lhs_l, rhs_l, l, r, B::u8, B::u8_vec)
        } else {
            binary_map(lhs_l, rhs_l, l, r, B::u8)
        })),
        (CpuStorage::F8E4M3(l), CpuStorage::F8E4M3(r)) => Ok(CpuStorage::F8E4M3(binary_map(
            lhs_l,
            rhs_l,
            l,
            r,
            B::f8e4m3,
        ))),
        _ => Err(Error::DTypeMismatchBinaryOp {
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
            op: B::NAME,
        }
        .bt()),
    }
}
