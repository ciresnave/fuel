use fuel::lazy_sam::SamModel;
use fuel_wasm_example_sam as sam;
use wasm_bindgen::prelude::*;

struct Embeddings {
    original_width: u32,
    original_height: u32,
    width: u32,
    height: u32,
    // Lazy graph output of `SamModel::embeddings()` — shape
    // `(1, out_chans, patches_per_side, patches_per_side)`.
    data: fuel::lazy::LazyTensor,
}

#[wasm_bindgen]
pub struct Model {
    sam: SamModel,
    embeddings: Option<Embeddings>,
}

#[wasm_bindgen]
impl Model {
    /// Construct a SAM model from a packed safetensors buffer.
    ///
    /// `use_tiny = true` selects the MobileSAM TinyViT image encoder
    /// (see `fuel_core::lazy_tiny_vit`); otherwise the Meta SAM ViT-B
    /// preset is used. The remaining sub-modules (prompt encoder + mask
    /// decoder) are SAM-standard in both cases.
    ///
    /// **WASM blocker (2026-06-07).** Today this constructor returns a
    /// JS error because the lazy-SAM weight loaders are still
    /// path-mapped (`fuel_core::lazy_sam::*Weights::load_from_mmapped`
    /// only accept `&MmapedSafetensors`, and a wasm32 target has no
    /// filesystem to mmap from). The structural rewire to the lazy
    /// substrate is in place; the runtime path comes online once a
    /// `load_from_buffered(&BufferedSafetensors, …)` seam ships on the
    /// three `Sam*Weights` types (and the HF-naming TODOs noted in the
    /// existing `load_from_mmapped` stubs are filled in).
    #[wasm_bindgen(constructor)]
    pub fn new(weights: Vec<u8>, use_tiny: bool) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        let _ = weights;
        let _ = use_tiny;
        Err(JsError::new(
            "fuel-wasm SAM constructor: lazy_sam Weights::load_from_mmapped \
             does not yet accept a buffered safetensors source. The wasm \
             entry point is structurally migrated to fuel_core::lazy_sam \
             but cannot construct weights from the JS-supplied Vec<u8> \
             until a buffered loader (or temp-file mmap shim) ships.",
        ))
    }

    pub fn set_image_embeddings(&mut self, image_data: Vec<u8>) -> Result<(), JsError> {
        sam::console_log!("image data: {}", image_data.len());
        let image_data = std::io::Cursor::new(image_data);
        let image = image::ImageReader::new(image_data)
            .with_guessed_format()?
            .decode()
            .map_err(|e| JsError::new(&e.to_string()))?;
        let (original_height, original_width) = (image.height(), image.width());
        let resize_longest = sam::IMAGE_SIZE as u32;
        let (height, width) = if original_height < original_width {
            let h = (resize_longest * original_height) / original_width;
            (h, resize_longest)
        } else {
            let w = (resize_longest * original_width) / original_height;
            (resize_longest, w)
        };
        let img = image.resize_exact(width, height, image::imageops::FilterType::CatmullRom);
        let raw_rgb = img.to_rgb8().into_raw();
        // HWC u8 → CHW f32 (the format `SamModel::embeddings` expects).
        let h = img.height() as usize;
        let w = img.width() as usize;
        let mut chw = vec![0.0_f32; 3 * h * w];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let src_idx = (y * w + x) * 3 + c;
                    let dst_idx = c * h * w + y * w + x;
                    chw[dst_idx] = raw_rgb[src_idx] as f32;
                }
            }
        }
        let data = self
            .sam
            .embeddings(&chw, h, w)
            .map_err(|e| JsError::new(&e.to_string()))?;
        self.embeddings = Some(Embeddings {
            original_width,
            original_height,
            width,
            height,
            data,
        });
        Ok(())
    }

    pub fn mask_for_point(&self, input: JsValue) -> Result<JsValue, JsError> {
        let input: PointsInput =
            serde_wasm_bindgen::from_value(input).map_err(|m| JsError::new(&m.to_string()))?;
        let transformed_points = input.points;

        for &(x, y, _bool) in &transformed_points {
            if !(0.0..=1.0).contains(&x) {
                return Err(JsError::new(&format!(
                    "x has to be between 0 and 1, got {x}"
                )));
            }
            if !(0.0..=1.0).contains(&y) {
                return Err(JsError::new(&format!(
                    "y has to be between 0 and 1, got {y}"
                )));
            }
        }
        let embeddings = match &self.embeddings {
            None => Err(JsError::new("image embeddings have not been set"))?,
            Some(embeddings) => embeddings,
        };
        // The lazy SamModel expects `points_xy` as a flat `(N, 2)`
        // row-major f32 buffer of pixel coordinates relative to the
        // original (pre-padding) image, and `point_labels` as a flat
        // `(N,)` buffer of 0/1 flags (background/foreground). The
        // not-a-point padding marker is appended internally.
        let mut points_xy = Vec::with_capacity(transformed_points.len() * 2);
        let mut point_labels = Vec::with_capacity(transformed_points.len());
        for &(x, y, is_fg) in &transformed_points {
            points_xy.push(x as f32 * embeddings.width as f32);
            points_xy.push(y as f32 * embeddings.height as f32);
            point_labels.push(if is_fg { 1.0_f32 } else { 0.0_f32 });
        }
        let (mask, iou_predictions) = self
            .sam
            .forward_for_embeddings(
                &embeddings.data,
                embeddings.height as usize,
                embeddings.width as usize,
                &points_xy,
                &point_labels,
                false,
            )
            .map_err(|e| JsError::new(&e.to_string()))?;
        let iou_vec = iou_predictions.realize_f32();
        let iou = iou_vec[0];
        let mask_shape = mask.shape().dims().to_vec();
        // Threshold the mask at 0.0 — same semantics as the eager
        // `mask.ge(0f32)?`. The `ge` family returns a U8 mask; we cast
        // to f32 → realize → convert to u8 because there is no
        // `realize_u8` in the lazy substrate yet.
        let zero = mask
            .zeros_like()
            .map_err(|e| JsError::new(&e.to_string()))?;
        let mask_u8: Vec<u8> = mask
            .ge(&zero)
            .map_err(|e| JsError::new(&e.to_string()))?
            .to_dtype(fuel::DType::F32)
            .map_err(|e| JsError::new(&e.to_string()))?
            .realize_f32()
            .into_iter()
            .map(|v| if v != 0.0 { 1u8 } else { 0u8 })
            .collect();
        let mask = Mask {
            iou,
            mask_shape,
            mask_data: mask_u8,
        };
        let image = Image {
            original_width: embeddings.original_width,
            original_height: embeddings.original_height,
            width: embeddings.width,
            height: embeddings.height,
        };
        Ok(serde_wasm_bindgen::to_value(&MaskImage { mask, image })?)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Mask {
    iou: f32,
    mask_shape: Vec<usize>,
    mask_data: Vec<u8>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct Image {
    original_width: u32,
    original_height: u32,
    width: u32,
    height: u32,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct MaskImage {
    mask: Mask,
    image: Image,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PointsInput {
    points: Vec<(f64, f64, bool)>,
}

fn main() {
    console_error_panic_hook::set_once();
}
