// SPDX-License-Identifier: MIT OR Apache-2.0
//! Validated entry for model inputs (GAP-326).
//!
//! Models check rank with the existing `fuel_ir` helpers:
//! `Shape::dims3()` and `Shape::dims4()` return a typed
//! `Error::UnexpectedNumberOfDims`, and `Error::context` names the model:
//!
//! ```text
//! let (n, c, h, w) = image.shape().dims4().map_err(|e| e.context("ResNet::forward: image"))?;
//! ```
//!
//! This module adds only what `fuel_ir` lacks: a fixed channel count on an
//! `[N, C, H, W]` image.
//!
//! Deliberately NOT here, because other rows own them: batch == 1 guards,
//! fixed spatial sizes, square or divisible-by-N inputs, codebook counts,
//! tile counts (GAP-328), and config-vs-config agreement (GAP-314). If any of
//! them is ruled in, it gets its own helper, not an option on this one.

use fuel_core::lazy::Tensor;
use fuel_core::{Error, Result, Shape};

/// The `(n, c, h, w)` of an `[N, C, H, W]` image whose channel count is
/// `channels`.
///
/// Never panics. A rank other than 4 is `Error::UnexpectedNumberOfDims`,
/// wrapped with `what`. A channel count other than `channels` is
/// `Error::UnexpectedShape`, whose message names `what`.
pub(crate) fn image_nchw(
    x: &Tensor,
    channels: usize,
    what: &'static str,
) -> Result<(usize, usize, usize, usize)> {
    let shape = x.shape();
    let (n, c, h, w) = shape.dims4().map_err(|e| e.context(what))?;
    if c != channels {
        return Err(Error::UnexpectedShape {
            msg: format!("{what}: expected {channels} input channels, got {c}"),
            expected: Box::new(Shape::from_dims(&[n, channels, h, w])),
            got: Box::new(shape),
        }
        .bt());
    }
    Ok((n, c, h, w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core::Device;

    fn tensor(dims: &[usize]) -> Tensor {
        let count = dims.iter().product();
        Tensor::from_f32(vec![0.0; count], Shape::from_dims(dims), &Device::cpu()).unwrap()
    }

    /// The error under any `Context` / `WithBacktrace` / `WithPath` wrapping.
    fn root(e: &Error) -> &Error {
        match e {
            Error::Context { inner, .. }
            | Error::WithBacktrace { inner, .. }
            | Error::WithPath { inner, .. } => root(inner),
            other => other,
        }
    }

    #[test]
    fn accepts_the_requested_channel_count() {
        assert_eq!(
            image_nchw(&tensor(&[2, 3, 4, 5]), 3, "m").unwrap(),
            (2, 3, 4, 5)
        );
        // The count is a parameter, not a hard-coded 3.
        assert_eq!(
            image_nchw(&tensor(&[1, 1, 2, 2]), 1, "m").unwrap(),
            (1, 1, 2, 2)
        );
    }

    #[test]
    fn a_wrong_rank_is_a_typed_rank_error_naming_the_model() {
        let e = image_nchw(&tensor(&[3, 4, 5]), 3, "ResNet::forward: image").unwrap_err();
        assert!(
            matches!(
                root(&e),
                Error::UnexpectedNumberOfDims {
                    expected: 4,
                    got: 3,
                    ..
                }
            ),
            "{e}"
        );
        assert!(e.to_string().contains("ResNet::forward: image"), "{e}");
    }

    #[test]
    fn a_wrong_channel_count_is_a_typed_shape_error_naming_the_model() {
        let e = image_nchw(&tensor(&[1, 4, 2, 2]), 3, "ResNet::forward: image").unwrap_err();
        match root(&e) {
            Error::UnexpectedShape { msg, expected, got } => {
                assert!(msg.contains("ResNet::forward: image"), "{msg}");
                assert_eq!(expected.dims(), &[1, 3, 2, 2]);
                assert_eq!(got.dims(), &[1, 4, 2, 2]);
            }
            other => panic!("expected UnexpectedShape, got {other}"),
        }
    }
}
