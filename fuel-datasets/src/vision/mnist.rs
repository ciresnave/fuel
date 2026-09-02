// SPDX-License-Identifier: MIT OR Apache-2.0
//! The MNIST hand-written digit dataset.
//!
//! The files can be obtained from the following link:
//! <http://yann.lecun.com/exdb/mnist/>
use fuel_core::{Error, Result};
use hf_hub::{Repo, RepoType, api::sync::Api};
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fs::File;
use std::io::{self, BufReader, Read};

fn read_u32<T: Read>(reader: &mut T) -> std::io::Result<u32> {
    use byteorder::ReadBytesExt;
    reader.read_u32::<byteorder::BigEndian>()
}

fn check_magic_number<T: Read>(reader: &mut T, expected: u32) -> Result<()> {
    let magic_number = read_u32(reader)?;
    if magic_number != expected {
        Err(io::Error::other(format!(
            "incorrect magic number {magic_number} != {expected}"
        )))?;
    }
    Ok(())
}

fn read_labels(filename: &std::path::Path) -> Result<Vec<u32>> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    check_magic_number(&mut buf_reader, 2049)?;
    let samples = read_u32(&mut buf_reader)?;
    let mut data = vec![0u8; samples as usize];
    buf_reader.read_exact(&mut data)?;
    Ok(data.into_iter().map(u32::from).collect())
}

/// Returns `(pixels, samples, rows, cols)`. Pixels are flattened row-major and
/// scaled to `[0.0, 1.0]` on the host — the eager version built a `[samples,
/// rows*cols]` u8 tensor only to `to_dtype(F32)` and divide it, which is one
/// host pass expressed as three graph nodes. `Tensor` could not express it
/// at all: it has no `from_u8`/`const_u8_like` constructor.
fn read_images(filename: &std::path::Path) -> Result<(Vec<f32>, usize, usize, usize)> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    check_magic_number(&mut buf_reader, 2051)?;
    let samples = read_u32(&mut buf_reader)? as usize;
    let rows = read_u32(&mut buf_reader)? as usize;
    let cols = read_u32(&mut buf_reader)? as usize;
    let data_len = samples * rows * cols;
    let mut data = vec![0u8; data_len];
    buf_reader.read_exact(&mut data)?;
    let pixels = data.into_iter().map(|b| f32::from(b) / 255.0).collect();
    Ok((pixels, samples, rows, cols))
}

/// Load the MNIST dataset from a directory containing the original IDX binary files.
///
/// # Example
///
/// ```no_run
/// use fuel_datasets::vision::mnist;
/// let dataset = mnist::load_dir("data/mnist")?;
/// # Ok::<(), fuel_core::Error>(())
/// ```
pub fn load_dir<T: AsRef<std::path::Path>>(dir: T) -> Result<crate::vision::Dataset> {
    let dir = dir.as_ref();
    let (train_images, train_samples, rows, cols) =
        read_images(&dir.join("train-images-idx3-ubyte"))?;
    let train_labels = read_labels(&dir.join("train-labels-idx1-ubyte"))?;
    let (test_images, test_samples, _, _) = read_images(&dir.join("t10k-images-idx3-ubyte"))?;
    let test_labels = read_labels(&dir.join("t10k-labels-idx1-ubyte"))?;
    Ok(crate::vision::Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: 10,
        image_dims: vec![rows, cols],
        train_samples,
        test_samples,
    })
}

fn load_parquet(
    parquet: SerializedFileReader<std::fs::File>,
) -> Result<(Vec<f32>, Vec<u32>, usize)> {
    let samples = parquet.metadata().file_metadata().num_rows() as usize;
    let mut buffer_images: Vec<u8> = Vec::with_capacity(samples * 784);
    let mut buffer_labels: Vec<u8> = Vec::with_capacity(samples);
    for row in parquet.into_iter().flatten() {
        for (_name, field) in row.get_column_iter() {
            if let parquet::record::Field::Group(subrow) = field {
                for (_name, field) in subrow.get_column_iter() {
                    if let parquet::record::Field::Bytes(value) = field {
                        let image = image::load_from_memory(value.data()).unwrap();
                        buffer_images.extend(image.to_luma8().as_raw());
                    }
                }
            } else if let parquet::record::Field::Long(label) = field {
                buffer_labels.push(*label as u8);
            }
        }
    }
    let images: Vec<f32> = buffer_images
        .into_iter()
        .map(|b| f32::from(b) / 255.0)
        .collect();
    let labels: Vec<u32> = buffer_labels.into_iter().map(u32::from).collect();
    Ok((images, labels, samples))
}

pub(crate) fn load_mnist_like(
    dataset_id: &str,
    revision: &str,
    test_filename: &str,
    train_filename: &str,
) -> Result<crate::vision::Dataset> {
    let api = Api::new().map_err(|e| Error::Msg(format!("Api error: {e}")))?;
    let repo = Repo::with_revision(
        dataset_id.to_string(),
        RepoType::Dataset,
        revision.to_string(),
    );
    let repo = api.repo(repo);
    let test_parquet_filename = repo
        .get(test_filename)
        .map_err(|e| Error::Msg(format!("Api error: {e}")))?;
    let train_parquet_filename = repo
        .get(train_filename)
        .map_err(|e| Error::Msg(format!("Api error: {e}")))?;
    let test_parquet = SerializedFileReader::new(std::fs::File::open(test_parquet_filename)?)
        .map_err(|e| Error::Msg(format!("Parquet error: {e}")))?;
    let train_parquet = SerializedFileReader::new(std::fs::File::open(train_parquet_filename)?)
        .map_err(|e| Error::Msg(format!("Parquet error: {e}")))?;
    let (test_images, test_labels, test_samples) = load_parquet(test_parquet)?;
    let (train_images, train_labels, train_samples) = load_parquet(train_parquet)?;
    Ok(crate::vision::Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: 10,
        image_dims: vec![28, 28],
        train_samples,
        test_samples,
    })
}

/// Download and load the MNIST dataset via the Hugging Face hub.
///
/// # Example
///
/// ```no_run
/// use fuel_datasets::vision::mnist;
/// let dataset = mnist::load()?;
/// println!("train samples: {}, dims: {:?}", dataset.train_samples, dataset.image_dims);
/// # Ok::<(), fuel_core::Error>(())
/// ```
pub fn load() -> Result<crate::vision::Dataset> {
    load_mnist_like(
        "ylecun/mnist",
        "refs/convert/parquet",
        "mnist/test/0000.parquet",
        "mnist/train/0000.parquet",
    )
}
