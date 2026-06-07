//! SAM: Segment Anything Model
//! https://github.com/facebookresearch/segment-anything

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use fuel::lazy_sam::{
    SamImageEncoderWeights, SamMaskDecoderWeights, SamModel, SamModelConfig,
    SamPromptEncoderWeights,
};

/// SAM input side — image is normalized, zero-padded to this size before the
/// image encoder runs. Mirrors the eager `sam::IMAGE_SIZE` constant.
const IMAGE_SIZE: usize = 1024;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    generate_masks: bool,

    /// List of x,y coordinates, between 0 and 1 (0.5 is at the middle of the image). These points
    /// should be part of the generated mask.
    #[arg(long)]
    point: Vec<String>,

    /// List of x,y coordinates, between 0 and 1 (0.5 is at the middle of the image). These points
    /// should not be part of the generated mask and should be part of the background instead.
    #[arg(long)]
    neg_point: Vec<String>,

    /// The detection threshold for the mask, 0 is the default value, negative values mean a larger
    /// mask, positive makes the mask more selective.
    #[arg(long, allow_hyphen_values = true, default_value_t = 0.)]
    threshold: f32,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Use the TinyViT based models from MobileSAM.
    ///
    /// Not supported by the lazy port yet — the lazy crate exposes the
    /// `TinyViT` image encoder (`fuel::lazy_tiny_vit`) but has no composed
    /// MobileSAM prompt-encoder + mask-decoder loader, so we error out here.
    #[arg(long)]
    use_tiny: bool,
}

/// Load the input image as raw `(3, h, w)` f32 pixels in `0..=255` and return
/// the original (pre-resize) `(height, width)` for downstream mask cropping.
fn load_image_as_f32(
    path: &str,
    resize_longest: Option<usize>,
) -> Result<(Vec<f32>, usize, usize, usize, usize)> {
    let img = image::ImageReader::open(path)?
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
    let height = img.height() as usize;
    let width = img.width() as usize;
    let img = img.to_rgb8();
    let raw = img.into_raw(); // (h, w, 3) row-major u8
    // Re-order into (3, h, w) row-major f32 — matches the eager `Tensor::permute((2,0,1))`.
    let mut chw = vec![0.0_f32; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                chw[c * height * width + y * width + x] =
                    raw[(y * width + x) * 3 + c] as f32;
            }
        }
    }
    Ok((chw, height, width, initial_h, initial_w))
}

pub fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // `--cpu` is accepted for backwards-compat with the eager binary's CLI but
    // the lazy executor selects its backend itself; no device handle is
    // threaded through to the lazy API.
    let _ = args.cpu;

    if args.use_tiny {
        anyhow::bail!(
            "--use-tiny (MobileSAM) is not supported by the lazy port yet — \
             the TinyViT image encoder lives in `fuel::lazy_tiny_vit`, but \
             there is no composed MobileSAM prompt-encoder + mask-decoder \
             loader. Drop --use-tiny to run the standard SAM ViT-B model."
        );
    }

    let (image_chw, h, w, initial_h, initial_w) =
        load_image_as_f32(&args.image, Some(IMAGE_SIZE))?;
    println!("loaded image (3, {h}, {w})");

    let model_path = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/fuel-sam".to_string());
            let filename = "sam_vit_b_01ec64.safetensors";
            api.get(filename)?
        }
    };

    let cfg = SamModelConfig::vit_b();
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[model_path]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let image_encoder_weights =
        SamImageEncoderWeights::load_from_mmapped(&st, &cfg.image_encoder)
            .map_err(|e| E::msg(format!("image-encoder weights: {e}")))?;
    let prompt_encoder_weights =
        SamPromptEncoderWeights::load_from_mmapped(&st, &cfg.prompt_encoder)
            .map_err(|e| E::msg(format!("prompt-encoder weights: {e}")))?;
    let mask_decoder_weights =
        SamMaskDecoderWeights::load_from_mmapped(&st, &cfg.mask_decoder)
            .map_err(|e| E::msg(format!("mask-decoder weights: {e}")))?;
    let sam = SamModel::new(
        cfg,
        image_encoder_weights,
        prompt_encoder_weights,
        mask_decoder_weights,
    );

    if args.generate_masks {
        anyhow::bail!(
            "--generate-masks is not implemented in the lazy port — the eager \
             `Sam::generate_masks` uses a multi-crop, multi-prompt sweep + \
             non-maximum suppression that hasn't been ported to LazyTensor yet."
        );
    }

    let iter_points = args.point.iter().map(|p| (p, true));
    let iter_neg_points = args.neg_point.iter().map(|p| (p, false));
    let points = iter_points
        .chain(iter_neg_points)
        .map(|(point, b)| {
            use std::str::FromStr;
            let xy = point.split(',').collect::<Vec<_>>();
            if xy.len() != 2 {
                anyhow::bail!("expected format for points is 0.4,0.2")
            }
            Ok((f64::from_str(xy[0])?, f64::from_str(xy[1])?, b))
        })
        .collect::<Result<Vec<_>>>()?;

    // The lazy `SamModel::forward` expects the prompts in original-image
    // pixel coordinates and labels as a `&[f32]` parallel array.
    let points_xy: Vec<f32> = points
        .iter()
        .flat_map(|(x, y, _b)| {
            let x = (*x as f32) * (w as f32);
            let y = (*y as f32) * (h as f32);
            [x, y]
        })
        .collect();
    let point_labels: Vec<f32> = points
        .iter()
        .map(|(_x, _y, b)| if *b { 1.0_f32 } else { 0.0_f32 })
        .collect();

    let start_time = std::time::Instant::now();
    let (mask_lazy, iou_lazy) = sam
        .forward(&image_chw, h, w, &points_xy, &point_labels, false)
        .map_err(|e| E::msg(format!("sam forward: {e}")))?;
    let mask_dims = mask_lazy.shape();
    let iou_dims = iou_lazy.shape();
    let mask_data = mask_lazy.realize_f32();
    let iou_data = iou_lazy.realize_f32();
    println!(
        "mask generated in {:.2}s",
        start_time.elapsed().as_secs_f32()
    );
    println!("mask: shape={:?}", mask_dims.dims());
    println!("iou_predictions: shape={:?} values={:?}", iou_dims.dims(), iou_data);

    // `mask_data` has shape `(num_returned, h, w)` (batch dim was squeezed by
    // `SamModel::forward`). With `multimask_output=false`, `num_returned == 1`.
    let dims = mask_dims.dims();
    if dims.len() != 3 {
        anyhow::bail!("expected mask rank 3, got shape {dims:?}");
    }
    let one = dims[0];
    if one != 1 {
        anyhow::bail!("expected single-mask output (num_returned=1), got {one}");
    }
    let mh = dims[1];
    let mw = dims[2];

    let threshold = args.threshold;
    let mask_pixels: Vec<u8> = (0..mh)
        .flat_map(|y| {
            let mask_data = &mask_data;
            (0..mw).map(move |x| {
                let v = mask_data[y * mw + x];
                if v >= threshold { 255_u8 } else { 0_u8 }
            })
        })
        .collect();
    // Build a (h, w, 3) buffer for image::ImageBuffer — broadcast single-channel
    // mask across RGB.
    let mut rgb = vec![0_u8; mh * mw * 3];
    for y in 0..mh {
        for x in 0..mw {
            let v = mask_pixels[y * mw + x];
            rgb[(y * mw + x) * 3] = v;
            rgb[(y * mw + x) * 3 + 1] = v;
            rgb[(y * mw + x) * 3 + 2] = v;
        }
    }

    let _ = initial_h;
    let _ = initial_w;

    let mut img = image::ImageReader::open(&args.image)?
        .decode()
        .map_err(fuel::Error::wrap)?;
    let mask_img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(mw as u32, mh as u32, rgb) {
            Some(image) => image,
            None => anyhow::bail!("error saving merged image"),
        };
    let mask_img = image::DynamicImage::from(mask_img).resize_to_fill(
        img.width(),
        img.height(),
        image::imageops::FilterType::CatmullRom,
    );
    for x in 0..img.width() {
        for y in 0..img.height() {
            let mask_p = imageproc::drawing::Canvas::get_pixel(&mask_img, x, y);
            if mask_p.0[0] > 100 {
                let mut img_p = imageproc::drawing::Canvas::get_pixel(&img, x, y);
                img_p.0[2] = 255 - (255 - img_p.0[2]) / 2;
                img_p.0[1] /= 2;
                img_p.0[0] /= 2;
                imageproc::drawing::Canvas::draw_pixel(&mut img, x, y, img_p)
            }
        }
    }
    for (x, y, b) in points {
        let x = (x * img.width() as f64) as i32;
        let y = (y * img.height() as f64) as i32;
        let color = if b {
            image::Rgba([255, 0, 0, 200])
        } else {
            image::Rgba([0, 255, 0, 200])
        };
        imageproc::drawing::draw_filled_circle_mut(&mut img, (x, y), 3, color);
    }
    img.save("sam_merged.jpg")?;
    Ok(())
}
