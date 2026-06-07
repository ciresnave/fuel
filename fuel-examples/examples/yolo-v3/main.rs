#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use image::{DynamicImage, ImageBuffer};

use fuel::lazy_yolov3::{
    decode_and_nms, YoloV3Config, YoloV3Detection, YoloV3Model, YoloV3NmsConfig, YoloV3Weights,
};
use fuel::safetensors::MmapedSafetensors;

// Assumes x1 <= x2 and y1 <= y2
pub fn draw_rect(
    img: &mut ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    x1: u32,
    x2: u32,
    y1: u32,
    y2: u32,
) {
    for x in x1..=x2 {
        let pixel = img.get_pixel_mut(x, y1);
        *pixel = image::Rgb([255, 0, 0]);
        let pixel = img.get_pixel_mut(x, y2);
        *pixel = image::Rgb([255, 0, 0]);
    }
    for y in y1..=y2 {
        let pixel = img.get_pixel_mut(x1, y);
        *pixel = image::Rgb([255, 0, 0]);
        let pixel = img.get_pixel_mut(x2, y);
        *pixel = image::Rgb([255, 0, 0]);
    }
}

/// Render detections onto the original image. `dets` is in **network
/// input pixel space** (size `w × h`); we scale to the original image
/// dimensions on the fly.
pub fn report(
    dets: &[YoloV3Detection],
    img: DynamicImage,
    w: usize,
    h: usize,
) -> Result<DynamicImage> {
    let (initial_h, initial_w) = (img.height(), img.width());
    let w_ratio = initial_w as f32 / w as f32;
    let h_ratio = initial_h as f32 / h as f32;
    let mut img = img.to_rgb8();
    for det in dets {
        println!(
            "{}: score={:.3} bbox={:?}",
            fuel_examples::coco_classes::NAMES[det.class_id],
            det.score,
            det.bbox,
        );
        let xmin = ((det.bbox[0] * w_ratio) as u32).clamp(0, initial_w - 1);
        let ymin = ((det.bbox[1] * h_ratio) as u32).clamp(0, initial_h - 1);
        let xmax = ((det.bbox[2] * w_ratio) as u32).clamp(0, initial_w - 1);
        let ymax = ((det.bbox[3] * h_ratio) as u32).clamp(0, initial_h - 1);
        // Guard against degenerate boxes where xmin == xmax or ymin == ymax.
        let xmax = xmax.max(xmin);
        let ymax = ymax.max(ymin);
        draw_rect(&mut img, xmin, xmax, ymin, ymax);
    }
    Ok(DynamicImage::ImageRgb8(img))
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model weights, in safetensors format.
    #[arg(long)]
    model: Option<String>,

    /// Darknet `.cfg` path. Optional and unused by the lazy port (the
    /// canonical `YoloV3Config` matches the official 608×608 / 80-class
    /// configuration). Kept for CLI compatibility with the eager
    /// binary.
    #[arg(long)]
    config: Option<String>,

    /// Override the network input size (must be divisible by 32).
    /// Defaults to the canonical 608.
    #[arg(long)]
    image_size: Option<usize>,

    images: Vec<String>,

    /// Threshold for the model confidence level.
    #[arg(long, default_value_t = 0.5)]
    confidence_threshold: f32,

    /// Threshold for non-maximum suppression.
    #[arg(long, default_value_t = 0.4)]
    nms_threshold: f32,
}

impl Args {
    fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        let path = match &self.model {
            Some(model) => std::path::PathBuf::from(model),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("lmz/fuel-yolo-v3".to_string());
                api.get("yolo-v3.safetensors")?
            }
        };
        Ok(path)
    }
}

pub fn main() -> Result<()> {
    let args = Args::parse();

    // Build the config — canonical defaults with optional size override.
    let mut cfg = YoloV3Config::yolo_v3();
    if let Some(sz) = args.image_size {
        if sz % 32 != 0 {
            anyhow::bail!("image_size {sz} must be divisible by 32 (5 stride-2 downsamples)");
        }
        cfg.image_size = sz;
    }
    let _ = &args.config; // kept for CLI parity; not consumed by the lazy port.

    // Load weights from the safetensors checkpoint.
    let model_file = args.model()?;
    let st = unsafe { MmapedSafetensors::multi(&[model_file]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = YoloV3Weights::load_from_mmapped(&st, &cfg)
        .map_err(|e| E::msg(format!("load yolov3 weights: {e}")))?;
    let model = YoloV3Model {
        config: cfg.clone(),
        weights,
    };

    let nms_cfg = YoloV3NmsConfig {
        score_threshold: args.confidence_threshold,
        iou_threshold: args.nms_threshold,
        top_k: 300,
    };

    for image_name in args.images.iter() {
        println!("processing {image_name}");
        let mut image_name = std::path::PathBuf::from(image_name);
        let net_width = cfg.image_size;
        let net_height = cfg.image_size;

        // Load image, resize to (net_width, net_height), and produce a
        // CHW f32 vector in `[0, 1]` pixel values.
        let original_image = image::ImageReader::open(&image_name)?
            .decode()
            .map_err(|e| E::msg(format!("decode image: {e}")))?;
        let resized = original_image.resize_exact(
            net_width as u32,
            net_height as u32,
            image::imageops::FilterType::Triangle,
        );
        let rgb = resized.to_rgb8();
        let mut chw = vec![0.0_f32; 3 * net_height * net_width];
        for y in 0..net_height {
            for x in 0..net_width {
                let p = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    chw[(c * net_height + y) * net_width + x] = p.0[c] as f32 / 255.0;
                }
            }
        }

        let raw = model
            .forward(&chw)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let dets = decode_and_nms(&raw, cfg.num_classes, &nms_cfg);
        println!("generated {} detections", dets.len());

        let image = report(&dets, original_image, net_width, net_height)?;
        image_name.set_extension("pp.jpg");
        println!("writing {image_name:?}");
        image.save(image_name)?
    }
    Ok(())
}
