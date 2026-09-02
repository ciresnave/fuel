// SPDX-License-Identifier: MIT OR Apache-2.0
/// A vision dataset split into train/test images and labels.
///
/// # Host buffers, not tensors
///
/// The four data fields are plain host buffers, **not** `Tensor`. Dataset
/// loading is file I/O plus a normalization pass; the consumer turns a batch
/// into a tensor at the point it enters a graph, and it must do so on *its own*
/// graph — `Tensor::from_*` mints a NEW graph per call, so a tensor handed
/// out by a loader could never be combined with the caller's activations
/// anyway. Keeping the decode side host-typed is therefore not a downgrade: it
/// is the only shape that composes.
///
/// Images are flattened row-major; use [`Dataset::image_dims`] and the sample
/// counts to reshape. The eager version leaked shape through `Tensor::dims()`,
/// so the dimensions are now carried explicitly instead of inferred.
///
/// # Example
///
/// ```no_run
/// use fuel_datasets::vision::Dataset;
/// // Dataset is typically constructed by loader functions such as
/// // `fuel_datasets::vision::mnist::load()`.
/// # let _ds: Dataset = unimplemented!();
/// ```
pub struct Dataset {
    /// Flattened, row-major, already scaled to `[0.0, 1.0]`.
    /// Length = `train_samples * image_dims.iter().product::<usize>()`.
    pub train_images: Vec<f32>,
    /// One label per training sample. Length = `train_samples`.
    pub train_labels: Vec<u32>,
    /// Flattened, row-major, already scaled to `[0.0, 1.0]`.
    /// Length = `test_samples * image_dims.iter().product::<usize>()`.
    pub test_images: Vec<f32>,
    /// One label per test sample. Length = `test_samples`.
    pub test_labels: Vec<u32>,
    /// Number of distinct classes.
    pub labels: usize,
    /// Per-image dimensions, e.g. `[28, 28]` for MNIST or `[3, 32, 32]` for
    /// CIFAR-10. Carried explicitly because host buffers are flat.
    pub image_dims: Vec<usize>,
    /// Number of training samples.
    pub train_samples: usize,
    /// Number of test samples.
    pub test_samples: usize,
}

impl Dataset {
    /// Elements per image (`image_dims` product).
    pub fn image_elems(&self) -> usize {
        self.image_dims.iter().product()
    }
}

pub mod cifar;
pub mod fashion_mnist;
pub mod mnist;
