//! Zalando Fashion MNIST dataset.
//! A slightly more difficult dataset that is drop-in compatible with MNIST.
//!
//! Taken from here: https://huggingface.co/datasets/zalando-datasets/fashion_mnist
use candle::Result;

/// Load the Fashion-MNIST dataset.
///
/// # Example
///
/// ```no_run
/// use candle_datasets::vision::fashion_mnist;
/// let dataset = fashion_mnist::load()?;
/// println!("train images: {:?}", dataset.train_images.dims());
/// # Ok::<(), candle::Error>(())
/// ```
pub fn load() -> Result<crate::vision::Dataset> {
    crate::vision::mnist::load_mnist_like(
        "zalando-datasets/fashion_mnist",
        "refs/convert/parquet",
        "fashion_mnist/test/0000.parquet",
        "fashion_mnist/train/0000.parquet",
    )
}
