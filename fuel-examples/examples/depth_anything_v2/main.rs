// SPDX-License-Identifier: MIT OR Apache-2.0
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
use fuel::{Device, Shape};
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

    let st = unsafe { MmapedSafetensors::multi(&[&dinov2_model_file, &depth_anything_model_file]) }
        .map_err(|e| E::msg(format!("mmap: {e}")))?;
    let weights = DepthAnythingV2Weights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("weights: {e}")))?;
    let depth_anything = DepthAnythingV2Model {
        config: config.clone(),
        dinov2_config,
        weights,
    };
    println!("DepthAnythingV2 model built");

    let (original_height, original_width, image) = load_and_prep_image(&args.image, &device)?;
    println!("Loaded image {:?}", image.shape());

    let depth = depth_anything
        .forward(&image)
        .map_err(|e| E::msg(format!("forward: {e}")))?;
    println!("Got predictions {:?}", depth.shape());

    // Resize on the graph, then finish post-processing on the host
    // (min/max-normalize → optional color map → uint8). B6 deleted the eager
    // `Tensor` this used to route through; none of the remaining steps needed
    // one, and `interpolate2d` exists on `LazyTensor`.
    let output_image = post_process_image(&depth, original_height, original_width, args.color_map)?;

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

    // Resize to CHW u8, then widen to f32 for the host-side normalize.
    let resized = fuel_examples::load_image_and_resize(image_path, DINO_IMG_SIZE, DINO_IMG_SIZE)?;
    let mut chw: Vec<f32> = resized.data.iter().map(|&b| b as f32).collect();
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
    depth: &LazyTensor,
    original_height: usize,
    original_width: usize,
    color_map: bool,
) -> anyhow::Result<fuel_examples::HostImage> {
    // Resize on the graph — this is the only step that wants a tensor.
    let resized = depth
        .interpolate2d(original_height, original_width)
        .map_err(|e| E::msg(format!("interpolate2d: {e}")))?;
    let gray = scale_to_unit_range(&resized.realize_f32());

    let plane = original_height * original_width;
    if gray.len() != plane {
        anyhow::bail!(
            "expected a {original_height}x{original_width} depth plane ({plane} values), got {}",
            gray.len()
        );
    }

    // Grayscale → RGB, channel-major. The colormap path maps each depth value
    // through the spectral gradient; the plain path replicates the single
    // channel three times (what `Tensor::cat(&[g, g, g], 0)` used to do).
    let rgb: Vec<f32> = if color_map {
        SpectralRColormap::new().gray2color(&gray, original_height, original_width)
    } else {
        let mut v = Vec::with_capacity(3 * plane);
        for _ in 0..3 {
            v.extend_from_slice(&gray);
        }
        v
    };

    Ok(fuel_examples::HostImage {
        data: rgb
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect(),
        height: original_height,
        width: original_width,
    })
}

/// Min/max-normalize a depth plane into `[0, 1]`. A flat plane (max == min)
/// maps to all-zero rather than dividing by zero.
fn scale_to_unit_range(depth: &[f32]) -> Vec<f32> {
    let min_val = depth.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = depth.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;
    if range == 0.0 || !range.is_finite() {
        return vec![0.0; depth.len()];
    }
    depth.iter().map(|v| (v - min_val) / range).collect()
}
