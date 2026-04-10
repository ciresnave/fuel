//! 1D and 2D convolution parameter structs.
//!
//! This module defines the parameter structs that describe a convolution's
//! geometry ([`ParamsConv1D`], [`ParamsConv2D`], [`ParamsConvTranspose1D`],
//! [`ParamsConvTranspose2D`]).

/// Parameters describing a 1-D convolution operation.
///
/// These are derived from the input and kernel shapes together with the
/// user-supplied `padding`, `stride`, and `dilation` values.  The output
/// length is computed as:
///
/// ```text
/// l_out = (l_in + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv1D {
    /// Batch size (first dimension of the input tensor).
    pub b_size: usize,
    /// Input sequence length.
    pub l_in: usize,
    /// Number of output channels (filters).
    pub c_out: usize,
    /// Number of input channels (per group).
    pub c_in: usize,
    /// Kernel (filter) size along the length dimension.
    pub k_size: usize,
    /// Zero-padding added to both sides of the input.
    pub padding: usize,
    /// Stride of the convolution.
    pub stride: usize,
    /// Spacing between kernel elements (atrous / dilated convolution).
    pub dilation: usize,
    /// Optional cuDNN algorithm selection for the forward pass.
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl ParamsConv1D {
    pub fn l_out(&self) -> usize {
        (self.l_in + 2 * self.padding - self.dilation * (self.k_size - 1) - 1) / self.stride + 1
    }

    pub fn out_dims(&self) -> Vec<usize> {
        let l_out = self.l_out();
        vec![self.b_size, self.c_out, l_out]
    }
}

/// Parameters describing a 1-D transposed (deconvolution) operation.
///
/// The output length is:
///
/// ```text
/// l_out = (l_in - 1) * stride - 2 * padding + dilation * (k_size - 1) + output_padding + 1
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose1D {
    /// Batch size.
    pub b_size: usize,
    /// Input sequence length.
    pub l_in: usize,
    /// Number of output channels.
    pub c_out: usize,
    /// Number of input channels (per group).
    pub c_in: usize,
    /// Kernel size along the length dimension.
    pub k_size: usize,
    /// Zero-padding subtracted from both sides of the output.
    pub padding: usize,
    /// Additional size added to one side of the output to resolve the
    /// ambiguity inherent in transposed convolutions.
    pub output_padding: usize,
    /// Stride of the transposed convolution.
    pub stride: usize,
    /// Dilation factor applied to the kernel.
    pub dilation: usize,
}

impl ParamsConvTranspose1D {
    pub fn l_out(&self) -> usize {
        (self.l_in - 1) * self.stride - 2 * self.padding
            + self.dilation * (self.k_size - 1)
            + self.output_padding
            + 1
    }

    pub fn out_dims(&self) -> Vec<usize> {
        let l_out = self.l_out();
        vec![self.b_size, self.c_out, l_out]
    }
}

/// Selection of cuDNN forward convolution algorithm.
///
/// Maps to `cudnnConvolutionFwdAlgo_t`. When set to `None` in a params
/// struct, cuDNN will use its own heuristic to pick the fastest algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudnnFwdAlgo {
    ImplicitGemm,
    ImplicitPrecompGemm,
    Gemm,
    Direct,
    Fft,
    FftTiling,
    Winograd,
    WinogradNonFused,
    Count,
}

/// Parameters describing a 2-D convolution operation.
///
/// Output spatial dimensions are computed as:
///
/// ```text
/// out_h = (i_h + 2 * padding - dilation * (k_h - 1) - 1) / stride + 1
/// out_w = (i_w + 2 * padding - dilation * (k_w - 1) - 1) / stride + 1
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv2D {
    /// Batch size.
    pub b_size: usize,
    /// Input height.
    pub i_h: usize,
    /// Input width.
    pub i_w: usize,
    /// Kernel height.
    pub k_h: usize,
    /// Kernel width.
    pub k_w: usize,
    /// Number of output channels (filters, per group).
    pub c_out: usize,
    /// Number of input channels (per group).
    pub c_in: usize,
    /// Zero-padding added to all four sides of the input.
    pub padding: usize,
    /// Stride of the convolution in both spatial dimensions.
    pub stride: usize,
    /// Dilation factor applied to the kernel.
    pub dilation: usize,
    /// Optional cuDNN algorithm selection for the forward pass.
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl ParamsConv2D {
    pub fn out_h(&self) -> usize {
        (self.i_h + 2 * self.padding - self.dilation * (self.k_h - 1) - 1) / self.stride + 1
    }

    pub fn out_w(&self) -> usize {
        (self.i_w + 2 * self.padding - self.dilation * (self.k_w - 1) - 1) / self.stride + 1
    }

    pub fn out_dims(&self) -> Vec<usize> {
        vec![self.b_size, self.c_out, self.out_h(), self.out_w()]
    }
}

/// Parameters describing a 2-D transposed (deconvolution) operation.
///
/// Output spatial dimensions are computed as:
///
/// ```text
/// out_h = (i_h - 1) * stride + dilation * (k_h - 1) + output_padding + 1 - 2 * padding
/// out_w = (i_w - 1) * stride + dilation * (k_w - 1) + output_padding + 1 - 2 * padding
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose2D {
    /// Batch size.
    pub b_size: usize,
    /// Input height.
    pub i_h: usize,
    /// Input width.
    pub i_w: usize,
    /// Kernel height.
    pub k_h: usize,
    /// Kernel width.
    pub k_w: usize,
    /// Number of output channels.
    pub c_out: usize,
    /// Number of input channels.
    pub c_in: usize,
    /// Zero-padding subtracted from the output spatial dimensions.
    pub padding: usize,
    /// Additional size added to one side of each output spatial dimension.
    pub output_padding: usize,
    /// Stride of the transposed convolution.
    pub stride: usize,
    /// Dilation factor applied to the kernel.
    pub dilation: usize,
}

impl ParamsConvTranspose2D {
    pub fn out_h(&self) -> usize {
        (self.i_h - 1) * self.stride + self.dilation * (self.k_h - 1) + self.output_padding + 1
            - 2 * self.padding
    }

    pub fn out_w(&self) -> usize {
        (self.i_w - 1) * self.stride + self.dilation * (self.k_w - 1) + self.output_padding + 1
            - 2 * self.padding
    }

    pub fn out_dims(&self) -> Vec<usize> {
        vec![self.b_size, self.c_out, self.out_h(), self.out_w()]
    }
}
