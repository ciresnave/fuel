//! 1D and 2D convolution (and transposed convolution) operations.
//!
//! This module re-exports the parameter structs from `candle-core-types` and
//! provides the [`Tensor`] methods that execute convolutions.

use crate::{op::BackpropOp, op::Op, Error, Result, Tensor};

pub use candle_core_types::conv::{
    CudnnFwdAlgo, ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D,
};

impl Tensor {
    fn conv1d_single_group(&self, kernel: &Self, params: &ParamsConv1D) -> Result<Self> {
        let storage =
            self.storage()
                .conv1d(self.layout(), &kernel.storage(), kernel.layout(), params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv1D {
            arg,
            kernel,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 1D convolution over the input tensor.
    ///
    /// Input shape: `(batch, in_channels, length)`  
    /// Kernel shape: `(out_channels, in_channels/groups, k_size)`  
    /// Output shape: `(batch, out_channels, l_out)`
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device, DType};
    /// let inp = Tensor::zeros((1, 1, 5), DType::F32, &Device::Cpu)?;
    /// let kernel = Tensor::zeros((1, 1, 3), DType::F32, &Device::Cpu)?;
    /// let out = inp.conv1d(&kernel, 0, 1, 1, 1)?;
    /// assert_eq!(out.dims(), &[1, 1, 3]); // l_out = 5 - 3 + 1 = 3
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        self.conv1d_with_algo(kernel, padding, stride, dilation, groups, None)
    }

    /// Applies a 1D convolution over the input tensor.
    pub fn conv1d_with_algo(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        cudnn_fwd_algo: Option<CudnnFwdAlgo>,
    ) -> Result<Self> {
        let (c_out, c_in_k, k_size) = kernel.dims3()?;
        let (b_size, c_in, l_in) = self.dims3()?;
        if c_in != c_in_k * groups {
            Err(Error::Conv1dInvalidArgs {
                inp_shape: self.shape().clone(),
                k_shape: kernel.shape().clone(),
                padding,
                stride,
                msg: "the number of in-channels on the input doesn't match the kernel size",
            }
            .bt())?
        }

        let params = ParamsConv1D {
            b_size,
            l_in,
            c_out: c_out / groups,
            c_in: c_in / groups,
            k_size,
            padding,
            stride,
            dilation,
            cudnn_fwd_algo,
        };
        if groups == 1 {
            self.conv1d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv1d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    fn conv_transpose1d_single_group(
        &self,
        kernel: &Self,
        params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        let storage = self.storage().conv_transpose1d(
            self.layout(),
            &kernel.storage(),
            kernel.layout(),
            params,
        )?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose1D {
            arg,
            kernel,
            padding: params.padding,
            output_padding: params.output_padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 1D transposed convolution over the input tensor.
    ///
    /// Input shape: `(batch, in_channels, length)`  
    /// Kernel shape: `(in_channels, out_channels/groups, k_size)`
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device, DType};
    /// let inp = Tensor::zeros((1, 1, 3), DType::F32, &Device::Cpu)?;
    /// let kernel = Tensor::zeros((1, 1, 3), DType::F32, &Device::Cpu)?;
    /// let out = inp.conv_transpose1d(&kernel, 0, 0, 1, 1, 1)?;
    /// assert_eq!(out.dims(), &[1, 1, 5]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn conv_transpose1d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (c_in_k, c_out, k_size) = kernel.dims3()?;
        let (b_size, c_in, l_in) = self.dims3()?;
        if c_in != c_in_k {
            crate::bail!("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
        }
        if c_in % groups != 0 {
            crate::bail!("in_channel {c_in} is not divisible by the number of groups")
        }
        let params = ParamsConvTranspose1D {
            b_size,
            l_in,
            k_size,
            c_out,
            c_in: c_in / groups,
            padding,
            output_padding,
            stride,
            dilation,
        };
        if groups == 1 {
            self.conv_transpose1d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv_transpose1d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    fn conv2d_single_group(&self, kernel: &Self, params: &ParamsConv2D) -> Result<Self> {
        let storage =
            self.storage()
                .conv2d(self.layout(), &kernel.storage(), kernel.layout(), params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv2D {
            arg,
            kernel,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 2D convolution over the input tensor.
    ///
    /// Input shape: `(batch, in_channels, height, width)`  
    /// Kernel shape: `(out_channels, in_channels/groups, k_h, k_w)`  
    /// Output shape: `(batch, out_channels, out_h, out_w)`
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device, DType};
    /// let inp = Tensor::zeros((1, 1, 4, 4), DType::F32, &Device::Cpu)?;
    /// let kernel = Tensor::zeros((1, 1, 3, 3), DType::F32, &Device::Cpu)?;
    /// let out = inp.conv2d(&kernel, 0, 1, 1, 1)?;
    /// assert_eq!(out.dims(), &[1, 1, 2, 2]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        self.conv2d_with_algo(kernel, padding, stride, dilation, groups, None)
    }

    /// Like [`conv2d`](Self::conv2d) but allows specifying an optional cuDNN algorithm hint.
    ///
    /// When `cudnn_fwd_algo` is `None`, cuDNN (if available) will auto-select the algorithm.
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device, DType};
    /// let inp = Tensor::zeros((1, 1, 4, 4), DType::F32, &Device::Cpu)?;
    /// let kernel = Tensor::zeros((1, 1, 3, 3), DType::F32, &Device::Cpu)?;
    /// let out = inp.conv2d_with_algo(&kernel, 0, 1, 1, 1, None)?;
    /// assert_eq!(out.dims(), &[1, 1, 2, 2]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn conv2d_with_algo(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        cudnn_fwd_algo: Option<CudnnFwdAlgo>,
    ) -> Result<Self> {
        let (b_size, c_in, i_h, i_w) = self.dims4()?;
        let (c_out, c_in_k, k_h, k_w) = kernel.dims4()?;
        if c_in != c_in_k * groups {
            crate::bail!(
                "in_channel mismatch between input ({c_in}, groups {groups}) and kernel ({c_in_k})"
            )
        }
        let params = ParamsConv2D {
            b_size,
            i_h,
            i_w,
            k_h,
            k_w,
            c_out: c_out / groups,
            c_in: c_in / groups,
            padding,
            stride,
            dilation,
            cudnn_fwd_algo,
        };
        if groups == 1 {
            // Fast path: a 1x1 convolution with stride=1, no padding, and no dilation
            // is equivalent to a matrix multiply, avoiding the im2col transform.
            if k_h == 1 && k_w == 1 && stride == 1 && padding == 0 && dilation == 1 {
                // kernel: (c_out, c_in, 1, 1) -> (c_out, c_in)
                let w = kernel.reshape((c_out, c_in))?;
                // input: (b_size, c_in, i_h, i_w) -> (b_size, c_in, i_h * i_w)
                let x = self.reshape((b_size, c_in, i_h * i_w))?;
                // matmul: (c_out, c_in) @ (b_size, c_in, i_h * i_w) -> (b_size, c_out, i_h * i_w)
                let out = w.broadcast_matmul(&x)?;
                // reshape back to (b_size, c_out, i_h, i_w)
                return out.reshape((b_size, c_out, i_h, i_w));
            }
            self.conv2d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv2d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    /// Applies a 2D transposed convolution over the input tensor.
    ///
    /// Input shape: `(batch, in_channels, height, width)`  
    /// Kernel shape: `(in_channels, out_channels, k_h, k_w)`
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device, DType};
    /// let inp = Tensor::zeros((1, 1, 2, 2), DType::F32, &Device::Cpu)?;
    /// let kernel = Tensor::zeros((1, 1, 3, 3), DType::F32, &Device::Cpu)?;
    /// let out = inp.conv_transpose2d(&kernel, 0, 0, 1, 1)?;
    /// assert_eq!(out.dims(), &[1, 1, 4, 4]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn conv_transpose2d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<Self> {
        let (b_size, c_in, i_h, i_w) = self.dims4()?;
        let (c_in_k, c_out, k_h, k_w) = kernel.dims4()?;
        if c_in != c_in_k {
            crate::bail!("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
        }
        let params = ParamsConvTranspose2D {
            b_size,
            i_h,
            i_w,
            k_h,
            k_w,
            c_out,
            c_in,
            padding,
            output_padding,
            stride,
            dilation,
        };
        let storage = self.storage().conv_transpose2d(
            self.layout(),
            &kernel.storage(),
            kernel.layout(),
            &params,
        )?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose2D {
            arg,
            kernel,
            padding: params.padding,
            output_padding: params.output_padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }
}
