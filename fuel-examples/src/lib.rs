// SPDX-License-Identifier: MIT OR Apache-2.0
pub mod audio;
pub mod bs1770;
pub mod chat_template;
pub mod coco_classes;
pub mod imagenet;
pub mod mnist_train;
pub mod token_output_stream;
pub mod wav;
use fuel::utils::{cuda_is_available, metal_is_available};
use fuel::{Device, Result};

/// A decoded image held on the host in **CHW** order (channel-major),
/// 3 channels, one `u8` per channel per pixel.
///
/// B6 retired the eager `Tensor`, and image decode/encode never needed one:
/// these helpers only ever used it as a shaped byte buffer plus a `permute`.
/// Callers that want a tensor build one themselves — for the lazy stack that
/// is `Tensor::from_vec(img.data, (3, img.height, img.width), dev)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostImage {
    /// `3 * height * width` bytes, laid out channel-major.
    pub data: Vec<u8>,
    pub height: usize,
    pub width: usize,
}

impl HostImage {
    /// Build from an interleaved RGB (HWC) buffer, transposing to CHW.
    ///
    /// Errors if `hwc.len() != 3 * height * width`.
    pub fn from_hwc(hwc: &[u8], height: usize, width: usize) -> Result<Self> {
        if hwc.len() != 3 * height * width {
            fuel::bail!(
                "HostImage::from_hwc: expected {} bytes for {height}x{width}x3, got {}",
                3 * height * width,
                hwc.len()
            )
        }
        let mut data = vec![0u8; hwc.len()];
        for h in 0..height {
            for w in 0..width {
                for c in 0..3 {
                    data[(c * height + h) * width + w] = hwc[(h * width + w) * 3 + c];
                }
            }
        }
        Ok(Self {
            data,
            height,
            width,
        })
    }

    /// Transpose back to interleaved RGB (HWC) — the layout the `image`
    /// crate's `ImageBuffer` expects.
    pub fn to_hwc(&self) -> Vec<u8> {
        let (h, w) = (self.height, self.width);
        let mut out = vec![0u8; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    out[(y * w + x) * 3 + c] = self.data[(c * h + y) * w + x];
                }
            }
        }
        out
    }
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::cpu())
    } else if cuda_is_available() {
        fuel::cuda_backend::new_device(0)
    } else if metal_is_available() {
        fuel::metal_backend::new_device(0)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::cpu())
    }
}

/// Decode an image, optionally downscaling so its longest side is
/// `resize_longest`. Returns the CHW image plus the **original** height and
/// width (before any resize).
pub fn load_image<P: AsRef<std::path::Path>>(
    p: P,
    resize_longest: Option<usize>,
) -> Result<(HostImage, usize, usize)> {
    let img = image::ImageReader::open(p)?
        .decode()
        .map_err(fuel::Error::wrap)?;
    let (initial_h, initial_w) = (img.height() as usize, img.width() as usize);
    let img = match resize_longest {
        None => img,
        Some(resize_longest) => {
            let (height, width) = (img.height(), img.width());
            let resize_longest = resize_longest as u32;
            let (height, width) = if height < width {
                let h = (resize_longest * height) / width;
                (h, resize_longest)
            } else {
                let w = (resize_longest * width) / height;
                (resize_longest, w)
            };
            img.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        }
    };
    let (height, width) = (img.height() as usize, img.width() as usize);
    let img = img.to_rgb8();
    let data = HostImage::from_hwc(&img.into_raw(), height, width)?;
    Ok((data, initial_h, initial_w))
}

/// Decode an image and resize it to exactly `width` x `height`, cropping to
/// fill. Returns a CHW image.
pub fn load_image_and_resize<P: AsRef<std::path::Path>>(
    p: P,
    width: usize,
    height: usize,
) -> Result<HostImage> {
    let img = image::ImageReader::open(p)?
        .decode()
        .map_err(fuel::Error::wrap)?
        .resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        );
    let img = img.to_rgb8();
    // NOTE: `resize_to_fill` above was given (width, height) in that order, so
    // the decoded buffer is `height` rows of `width` pixels.
    HostImage::from_hwc(&img.into_raw(), height, width)
}

/// Saves a CHW image to disk using the image crate.
pub fn save_image<P: AsRef<std::path::Path>>(img: &HostImage, p: P) -> Result<()> {
    let p = p.as_ref();
    let pixels = img.to_hwc();
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(img.width as u32, img.height as u32, pixels) {
            Some(image) => image,
            None => fuel::bail!("error saving image {p:?}"),
        };
    image.save(p).map_err(fuel::Error::wrap)?;
    Ok(())
}

/// Saves a CHW image to disk, rescaling it to `h` x `w` on the way out.
pub fn save_image_resize<P: AsRef<std::path::Path>>(
    img: &HostImage,
    p: P,
    h: usize,
    w: usize,
) -> Result<()> {
    let p = p.as_ref();
    let pixels = img.to_hwc();
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(img.width as u32, img.height as u32, pixels) {
            Some(image) => image,
            None => fuel::bail!("error saving image {p:?}"),
        };
    let image = image::DynamicImage::from(image);
    let image = image.resize_to_fill(w as u32, h as u32, image::imageops::FilterType::CatmullRom);
    image.save(p).map_err(fuel::Error::wrap)?;
    Ok(())
}

/// Loads the safetensors files for a model from the hub based on a json index file.
pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(fuel::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value = serde_json::from_reader(&json_file).map_err(fuel::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => fuel::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => fuel::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(fuel::Error::wrap))
        .collect::<Result<Vec<_>>>()?;
    Ok(safetensors_files)
}

pub fn hub_load_local_safetensors<P: AsRef<std::path::Path>>(
    path: P,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let path = path.as_ref();
    let jsfile = std::fs::File::open(path.join(json_file))?;
    let json: serde_json::Value = serde_json::from_reader(&jsfile).map_err(fuel::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => fuel::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => fuel::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file);
        }
    }
    let safetensors_files: Vec<_> = safetensors_files
        .into_iter()
        .map(|v| path.join(v))
        .collect();
    Ok(safetensors_files)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HWC → CHW → HWC must be the identity. This is the whole of what the
    /// eager `Tensor::permute` was doing for image I/O, so it is the one thing
    /// worth pinning after the port.
    #[test]
    fn hwc_chw_roundtrip_is_identity() -> Result<()> {
        // 2 rows x 3 cols, RGB interleaved, every byte distinct.
        let hwc: Vec<u8> = (0..18).collect();
        let img = HostImage::from_hwc(&hwc, 2, 3)?;
        assert_eq!(img.height, 2);
        assert_eq!(img.width, 3);
        assert_eq!(img.to_hwc(), hwc);
        Ok(())
    }

    /// CHW really is channel-major: the red plane must come first, contiguously.
    #[test]
    fn from_hwc_lays_out_channel_major() -> Result<()> {
        // 1 row x 2 cols: pixel0 = (1,2,3), pixel1 = (4,5,6).
        let img = HostImage::from_hwc(&[1, 2, 3, 4, 5, 6], 1, 2)?;
        assert_eq!(img.data, vec![1, 4, 2, 5, 3, 6]);
        Ok(())
    }

    /// Non-square images must not transpose. The eager `load_image_and_resize`
    /// it replaces built its tensor as `(width, height, 3)` after decoding a
    /// buffer that is really `height` rows of `width` pixels — the element count
    /// matched, so `from_vec` accepted it and the CHW result came out scrambled
    /// for every non-square target. Square targets hid the bug.
    #[test]
    fn from_hwc_does_not_transpose_a_non_square_image() -> Result<()> {
        // 2 rows x 3 cols. Red plane should be the 0th byte of each pixel, in
        // row-major order: pixels are 10,13,16 / 19,22,25.
        let hwc: Vec<u8> = (10..28).collect();
        let img = HostImage::from_hwc(&hwc, 2, 3)?;
        assert_eq!(&img.data[0..6], &[10, 13, 16, 19, 22, 25], "red plane");
        assert_eq!(&img.data[6..12], &[11, 14, 17, 20, 23, 26], "green plane");
        assert_eq!(&img.data[12..18], &[12, 15, 18, 21, 24, 27], "blue plane");
        // And the inverse must restore the original exactly.
        assert_eq!(img.to_hwc(), hwc);
        Ok(())
    }

    /// A wrong-length buffer is an error, not a panic or a silent crop.
    #[test]
    fn from_hwc_rejects_a_mismatched_length() {
        let err = HostImage::from_hwc(&[0; 17], 2, 3).unwrap_err();
        assert!(
            err.to_string().contains("expected 18 bytes"),
            "unexpected error: {err}"
        );
    }
}
