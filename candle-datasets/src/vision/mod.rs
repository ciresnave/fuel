use candle::Tensor;

/// A vision dataset split into train/test images and labels.
///
/// # Example
///
/// ```no_run
/// use candle_datasets::vision::Dataset;
/// // Dataset is typically constructed by loader functions such as
/// // `candle_datasets::vision::mnist::load()`.
/// # let _ds: Dataset = unimplemented!();
/// ```
pub struct Dataset {
    pub train_images: Tensor,
    pub train_labels: Tensor,
    pub test_images: Tensor,
    pub test_labels: Tensor,
    pub labels: usize,
}

pub mod cifar;
pub mod fashion_mnist;
pub mod mnist;
