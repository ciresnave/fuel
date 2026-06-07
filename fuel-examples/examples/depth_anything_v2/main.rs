//! Depth Anything V2
//! https://huggingface.co/spaces/depth-anything/Depth-Anything-V2

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Error as E;
use clap::Parser;
use std::{ffi::OsString, path::PathBuf, sync::Arc};

use fuel::lazy::LazyTensor;
use fuel::lazy_depth_anything_v2::{
    DepthAnythingV2Config, DepthAnythingV2Model, DepthAnythingV2Weights,
};
use fuel::lazy_dinov2::Dinov2Config;
use fuel::safetensors::MmapedSafetensors;
use fuel::DType::{F32, U8};
use fuel::{Device, Result, Shape, Tensor};
use fuel_examples::{load_image, save_image};

use crate::color_map::SpectralRColormap;

mod color_map;

// taken these from: https://huggingface.co/spaces/depth-anything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py#L207
const MAGIC_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const MAGIC_STD: [f32; 3] = [0.229, 0.224, 0.225];

const DINO_IMG_SIZE: usize = 518;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    dinov2_model: Option<PathBuf>,

    #[arg(long)]
    depth_anything_v2_model: Option<PathBuf>,

    #[arg(long)]
    image: PathBuf,

    #[arg(long)]
    output_dir: Option<PathBuf>,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    color_map: bool,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Lazy path realizes through CPU/router; `cpu` flag preserved for CLI
    // parity.
    let _ = args.cpu;
    let device = Device::cpu();

    let dinov2_model_file = match args.dinov2_model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/fuel-dino-v2".into());
            api.get("dinov2_vits14.safetensors")?
        }
        Some(dinov2_model) => dinov2_model,
    };
    println!("Using file {:?}", dinov2_model_file);

    let depth_anything_model_file = match args.depth_anything_v2_model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("jeroenvlek/depth-anything-v2-safetensors".into());
            api.get("depth_anything_v2_vits.safetensors")?
        }
        Some(depth_anything_model) => depth_anything_model,
    };
    println!("Using file {:?}", depth_anything_model_file);

    // Composition binary: depth-anything wraps a DINOv2 backbone. The lazy
    // wrapper's loader composes the two safetensors files into a single
    // DepthAnythingV2Weights via the eager-port layout (`pretrained.*`
    // backbone prefix + `depth_head.*` head prefix). Today the loader is a
    // stub that will surface as a runtime error; the migration ships the
    // binary against the lazy API so it compiles and is ready when the
    // loader lands.
    let config = DepthAnythingV2Config::vit_small();
    let dinov2_config = Dinov2Config::vit_small();

    let st = unsafe {
        MmapedSafetensors::multi(&[&dinov2_model_file, &depth_anything_model_file])
    }
    .map_err(|e| E::msg(format!("mmap: {e}")))?;
    let weights = DepthAnythingV2Weights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("weights: {e}")))?;
    let depth_anything = DepthAnythingV2Model {
        config: config.clone(),
        dinov2_config,
        weights,
    };
    println!("DepthAnythingV2 model built");

    let (original_height, original_width, image) =
        load_and_prep_image(&args.image, &device)?;
    println!("Loaded image {:?}", image.shape());

    let depth = depth_anything
        .forward(&image)
        .map_err(|e| E::msg(format!("forward: {e}")))?;
    println!("Got predictions {:?}", depth.shape());

    // Realize the depth map to host f32 and finish post-processing eagerly
    // (interpolate2d → min/max-normalize → optional color map → uint8) so
    // we can reuse `fuel_examples::save_image` and `SpectralRColormap`
    // which both expect an eager `Tensor`.
    let depth_dims = depth.shape().dims().to_vec();
    let depth_data = depth.realize_f32();
    let depth_eager = Tensor::from_vec(depth_data, depth_dims, &device)?;

    let output_image = post_process_image(
        &depth_eager,
        original_height,
        original_width,
        args.color_map,
    )?;

    let output_path = full_output_path(&args.image, &args.output_dir);
    println!("Saving image to {}", output_path.to_string_lossy());
    save_image(&output_image, output_path)?;

    Ok(())
}

fn full_output_path(image_path: &PathBuf, output_dir: &Option<PathBuf>) -> PathBuf {
    let input_file_name = image_path.file_name().unwrap();
    let mut output_file_name = OsString::from("depth_");
    output_file_name.push(input_file_name);
    let mut output_path = match output_dir {
        None => image_path.parent().unwrap().to_path_buf(),
        Some(output_path) => output_path.clone(),
    };
    output_path.push(output_file_name);

    output_path
}

/// Load + resize + normalize the input image. Normalization is computed
/// on the host as plain f32 vector arithmetic and the result is wrapped
/// as a [`LazyTensor`] of shape `(1, 3, DINO_IMG_SIZE, DINO_IMG_SIZE)`.
fn load_and_prep_image(
    image_path: &PathBuf,
    device: &Device,
) -> anyhow::Result<(usize, usize, LazyTensor)> {
    let (_original_image, original_height, original_width) = load_image(image_path, None)?;

    // Resize + CHW eager (uint8 → f32 with eager helpers, then drop to a
    // plain Vec for host-side normalize).
    let resized = fuel_examples::load_image_and_resize(
        image_path, DINO_IMG_SIZE, DINO_IMG_SIZE,
    )?
    .to_dtype(F32)?;
    let mut chw: Vec<f32> = resized.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(chw.len(), 3 * DINO_IMG_SIZE * DINO_IMG_SIZE);

    // Normalize: pixel/255, then channel-wise (- mean) / std.
    let plane = DINO_IMG_SIZE * DINO_IMG_SIZE;
    for c in 0..3 {
        let mean = MAGIC_MEAN[c];
        let std = MAGIC_STD[c];
        for px in &mut chw[c * plane..(c + 1) * plane] {
            *px = (*px / 255.0 - mean) / std;
        }
    }

    let image = LazyTensor::from_f32(
        Arc::<[f32]>::from(chw),
        Shape::from_dims(&[1, 3, DINO_IMG_SIZE, DINO_IMG_SIZE]),
        device,
    );

    Ok((original_height, original_width, image))
}

fn post_process_image(
    image: &Tensor,
    original_height: usize,
    original_width: usize,
    color_map: bool,
) -> Result<Tensor> {
    let out = image.interpolate2d(original_height, original_width)?;
    let out = scale_image(&out)?;

    let out = if color_map {
        let spectral_r = SpectralRColormap::new();
        spectral_r.gray2color(&out)?
    } else {
        let rgb_slice = [&out, &out, &out];
        Tensor::cat(&rgb_slice, 0)?.squeeze(1)?
    };

    let max_pixel_val = Tensor::try_from(255.0f32)?
        .to_device(out.device())?
        .broadcast_as(out.shape())?;
    let out = (out * max_pixel_val)?;

    out.to_dtype(U8)
}

fn scale_image(depth: &Tensor) -> Result<Tensor> {
    let flat_values: Vec<f32> = depth.flatten_all()?.to_vec1()?;

    let min_val = flat_values.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let max_val = flat_values.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    let min_val_tensor = Tensor::try_from(*min_val)?
        .to_device(depth.device())?
        .broadcast_as(depth.shape())?;
    let depth = (depth - min_val_tensor)?;

    let range = max_val - min_val;
    let range_tensor = Tensor::try_from(range)?
        .to_device(depth.device())?
        .broadcast_as(depth.shape())?;

    depth / range_tensor
}
