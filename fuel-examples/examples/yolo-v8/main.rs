//! YOLOv8 object detection — lazy-graph port.
//!
//! The eager `fuel_transformers`/`fuel_nn` YOLOv8 + Pose implementation
//! has been replaced by `fuel::lazy_yolov8`. The lazy module currently
//! only ships the **nano (`v8n`) detection** variant: pose, alternate
//! width/depth multiples (s/m/l/x), and the safetensors loader are
//! stubs upstream. The CLI surface is preserved so existing scripts
//! keep working; unsupported tasks/sizes error out gracefully and the
//! HF safetensors load will surface a "pending" runtime error from the
//! lazy loader stub.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};
use fuel::lazy_yolov8::{
    decode_and_nms, NmsConfig, YoloV8Config, YoloV8Model, YoloV8Weights,
};
use fuel::safetensors::MmapedSafetensors;
use image::DynamicImage;

// Keypoints (kept as reference for future Pose port; pose head not
// yet wired through the lazy graph).
// Nose, Left/Right Eye, Left/Right Ear, Left/Right Shoulder, Left/Right
// Elbow, Left/Right Wrist, Left/Right Hip, Left/Right Knee,
// Left/Right Ankle.
#[allow(dead_code)]
const KP_CONNECTIONS: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
];

/// Annotate `img` with detections decoded from the lazy YOLOv8 graph.
///
/// `w` and `h` are the network input dimensions (square, equal to
/// `cfg.image_size`); the original image is rescaled accordingly so
/// boxes drawn in pixel space line up with the network's
/// `(image_size, image_size)` predictions.
pub fn report_detect(
    detections: &[fuel::lazy_yolov8::Detection],
    img: DynamicImage,
    w: usize,
    h: usize,
    legend_size: u32,
) -> anyhow::Result<DynamicImage> {
    let (initial_h, initial_w) = (img.height(), img.width());
    let w_ratio = initial_w as f32 / w as f32;
    let h_ratio = initial_h as f32 / h as f32;
    let mut img = img.to_rgb8();
    let font = Vec::from(include_bytes!("roboto-mono-stripped.ttf") as &[u8]);
    let font = ab_glyph::FontRef::try_from_slice(&font).map_err(|e| anyhow::anyhow!("{e}"))?;
    for det in detections.iter() {
        let class_index = det.class_id;
        let [x1, y1, x2, y2] = det.bbox;
        println!(
            "{}: {:?}",
            fuel_examples::coco_classes::NAMES[class_index],
            det
        );
        let xmin = (x1 * w_ratio) as i32;
        let ymin = (y1 * h_ratio) as i32;
        let dx = (x2 - x1) * w_ratio;
        let dy = (y2 - y1) * h_ratio;
        if dx >= 0. && dy >= 0. {
            imageproc::drawing::draw_hollow_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, dy as u32),
                image::Rgb([255, 0, 0]),
            );
        }
        if legend_size > 0 {
            imageproc::drawing::draw_filled_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, legend_size),
                image::Rgb([170, 0, 0]),
            );
            let legend = format!(
                "{}   {:.0}%",
                fuel_examples::coco_classes::NAMES[class_index],
                100. * det.score
            );
            imageproc::drawing::draw_text_mut(
                &mut img,
                image::Rgb([255, 255, 255]),
                xmin,
                ymin,
                ab_glyph::PxScale {
                    x: legend_size as f32 - 1.,
                    y: legend_size as f32 - 1.,
                },
                &font,
                &legend,
            )
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum Which {
    N,
    S,
    M,
    L,
    X,
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum YoloTask {
    Detect,
    Pose,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Model weights, in safetensors format.
    #[arg(long)]
    model: Option<String>,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value_t = Which::S)]
    which: Which,

    images: Vec<String>,

    /// Threshold for the model confidence level.
    #[arg(long, default_value_t = 0.25)]
    confidence_threshold: f32,

    /// Threshold for non-maximum suppression.
    #[arg(long, default_value_t = 0.45)]
    nms_threshold: f32,

    /// The task to be run.
    #[arg(long, default_value = "detect")]
    task: YoloTask,

    /// The size for the legend, 0 means no legend.
    #[arg(long, default_value_t = 14)]
    legend_size: u32,
}

impl Args {
    fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        let path = match &self.model {
            Some(model) => std::path::PathBuf::from(model),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("lmz/fuel-yolo-v8".to_string());
                let size = match self.which {
                    Which::N => "n",
                    Which::S => "s",
                    Which::M => "m",
                    Which::L => "l",
                    Which::X => "x",
                };
                let task = match self.task {
                    YoloTask::Pose => "-pose",
                    YoloTask::Detect => "",
                };
                api.get(&format!("yolov8{size}{task}.safetensors"))?
            }
        };
        Ok(path)
    }
}

pub fn run(args: Args) -> anyhow::Result<()> {
    // Lazy YOLOv8 only ships the `v8n` (nano) detection variant.
    // Non-nano sizes and Pose use the same CLI surface but the lazy
    // module does not yet implement them.
    match args.task {
        YoloTask::Detect => {}
        YoloTask::Pose => {
            anyhow::bail!(
                "yolo-v8 lazy port currently supports `--task detect` only; \
                 pose head not ported"
            );
        }
    }
    if !matches!(args.which, Which::N) {
        eprintln!(
            "warning: lazy yolo-v8 only implements the nano (v8n) variant; \
             ignoring --which {:?}",
            args.which
        );
    }
    // Lazy path realizes via CPU/router; `cpu` flag is preserved
    // for CLI parity with the eager binary.
    let _ = args.cpu;

    // Build the lazy model.
    let cfg = YoloV8Config::v8n();
    let model_file = args.model()?;
    let st = unsafe { MmapedSafetensors::new(&model_file) }?;
    let weights = YoloV8Weights::load_from_mmapped(&st, &cfg)?;
    let model = YoloV8Model {
        config: cfg.clone(),
        weights,
    };
    println!("model loaded");

    let nms = NmsConfig {
        score_threshold: args.confidence_threshold,
        iou_threshold:   args.nms_threshold,
        top_k:           300,
    };

    for image_name in args.images.iter() {
        println!("processing {image_name}");
        let mut image_name = std::path::PathBuf::from(image_name);
        let original_image = image::ImageReader::open(&image_name)?
            .decode()
            .map_err(fuel::Error::wrap)?;

        // The lazy graph expects a square `(1, 3, image_size, image_size)`
        // input. Resize to the network's canonical square.
        let (width, height) = (cfg.image_size, cfg.image_size);
        let resized = original_image.resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let rgb = resized.to_rgb8();
        // Convert HWC u8 → CHW f32 normalized to [0,1].
        let (h_u, w_u) = (height, width);
        let mut chw = vec![0.0_f32; 3 * h_u * w_u];
        for y in 0..h_u {
            for x in 0..w_u {
                let p = rgb.get_pixel(x as u32, y as u32);
                let idx = y * w_u + x;
                chw[0 * h_u * w_u + idx] = p[0] as f32 / 255.0;
                chw[1 * h_u * w_u + idx] = p[1] as f32 / 255.0;
                chw[2 * h_u * w_u + idx] = p[2] as f32 / 255.0;
            }
        }

        // The lazy `forward` takes the raw flat buffer and constructs
        // its own LazyTensor inside.
        let raw = model.forward(&chw)?;
        let detections = decode_and_nms(&raw, cfg.num_classes, &nms);
        println!("generated {} detection(s)", detections.len());

        let image_out = report_detect(
            &detections,
            original_image,
            width,
            height,
            args.legend_size,
        )?;
        image_name.set_extension("pp.jpg");
        println!("writing {image_name:?}");
        image_out.save(image_name)?
    }

    Ok(())
}

pub fn main() -> anyhow::Result<()> {
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

    run(args)?;
    Ok(())
}
