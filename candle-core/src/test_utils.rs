use crate::{Result, Tensor};

#[macro_export]
macro_rules! test_device {
    // TODO: Switch to generating the two last arguments automatically once concat_idents is
    // stable. https://github.com/rust-lang/rust/issues/29599
    ($fn_name: ident, $test_cpu: ident, $test_cuda: ident, $test_metal: ident) => {
        #[test]
        fn $test_cpu() -> Result<()> {
            $fn_name(&Device::Cpu)
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn $test_cuda() -> Result<()> {
            $fn_name(&Device::new_cuda(0)?)
        }

        #[cfg(feature = "metal")]
        #[test]
        fn $test_metal() -> Result<()> {
            $fn_name(&Device::new_metal(0)?)
        }
    };
}

/// Asserts that two tensors have the same shape and identical element values.
///
/// # Example
///
/// ```rust
/// use candle_core::{Tensor, Device, DType};
/// use candle_core::test_utils::assert_tensor_eq;
/// let a = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
/// let b = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
/// assert_tensor_eq(&a, &b)?;
/// # Ok::<(), candle_core::Error>(())
/// ```
pub fn assert_tensor_eq(t1: &Tensor, t2: &Tensor) -> Result<()> {
    assert_eq!(t1.shape(), t2.shape());
    // Default U8 may not be large enough to hold the sum (`t.sum_all` defaults to the dtype of `t`)
    let eq_tensor = t1.eq(t2)?.to_dtype(crate::DType::U32)?;
    let all_equal = eq_tensor.sum_all()?;
    assert_eq!(all_equal.to_scalar::<u32>()?, eq_tensor.elem_count() as u32);
    Ok(())
}

/// Extracts a scalar f32 value from a rank-0 tensor, rounded to `digits` decimal places.
///
/// # Example
///
/// ```rust
/// use candle_core::{Tensor, Device};
/// use candle_core::test_utils::to_vec0_round;
/// let t = Tensor::new(3.14159f32, &Device::Cpu)?;
/// assert_eq!(to_vec0_round(&t, 2)?, 3.14);
/// # Ok::<(), candle_core::Error>(())
/// ```
pub fn to_vec0_round(t: &Tensor, digits: i32) -> Result<f32> {
    let b = 10f32.powi(digits);
    let t = t.to_vec0::<f32>()?;
    Ok(f32::round(t * b) / b)
}

/// Extracts a 1-D tensor to a `Vec<f32>`, rounding each element to `digits` decimal places.
///
/// # Example
///
/// ```rust
/// use candle_core::{Tensor, Device};
/// use candle_core::test_utils::to_vec1_round;
/// let t = Tensor::new(&[1.11111f32, 2.22222], &Device::Cpu)?;
/// assert_eq!(to_vec1_round(&t, 2)?, vec![1.11, 2.22]);
/// # Ok::<(), candle_core::Error>(())
/// ```
pub fn to_vec1_round(t: &Tensor, digits: i32) -> Result<Vec<f32>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec1::<f32>()?;
    let t = t.iter().map(|t| f32::round(t * b) / b).collect();
    Ok(t)
}

/// Extracts a 2-D tensor to a `Vec<Vec<f32>>`, rounding each element to `digits` decimal places.
///
/// # Example
///
/// ```rust
/// use candle_core::{Tensor, Device};
/// use candle_core::test_utils::to_vec2_round;
/// let t = Tensor::new(&[[1.005f32, 2.005], [3.005, 4.005]], &Device::Cpu)?;
/// let r = to_vec2_round(&t, 2)?;
/// assert_eq!(r[0][0], 1.01);
/// # Ok::<(), candle_core::Error>(())
/// ```
pub fn to_vec2_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec2::<f32>()?;
    let t = t
        .iter()
        .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
        .collect();
    Ok(t)
}

/// Extracts a 3-D tensor to a `Vec<Vec<Vec<f32>>>`, rounding each element to `digits` decimal places.
///
/// # Example
///
/// ```rust
/// use candle_core::{Tensor, Device};
/// use candle_core::test_utils::to_vec3_round;
/// let t = Tensor::zeros((2, 2, 2), candle_core::DType::F32, &Device::Cpu)?;
/// let r = to_vec3_round(&t, 2)?;
/// assert_eq!(r.len(), 2);
/// # Ok::<(), candle_core::Error>(())
/// ```
pub fn to_vec3_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}
