//! YOLOv8 wasm worker — lazy substrate port.
//!
//! Migrated off the retired eager `fuel_nn` / hand-rolled `Module` stack
//! onto `fuel::lazy_yolov8`. The model definition (backbone, neck,
//! decoupled detect head, DFL decode) now lives in `lazy_yolov8`;
//! this worker only assembles a `LazyTensor` input from a decoded RGB
//! image, runs the forward pass, and packages the resulting detections
//! as `Vec<Vec<Bbox>>` (per-class lists) for the UI / JS bridge.
//!
//! Deferrals vs the original eager wasm binary:
//!   - **Size variants other than `n`**. `lazy_yolov8::YoloV8Config`
//!     exposes only `v8n()` today (the `s`/`m`/`l`/`x` variants reuse
//!     the same architecture with different width/depth multipliers,
//!     but the corresponding config constructors haven't been added).
//!     Any model size other than `"n"` returns a clean error.
//!   - **Safetensors weight loading**. `YoloV8Weights::load_from_mmapped`
//!     is presently a stub (see `fuel-core::lazy_yolov8`). It also
//!     takes `&MmapedSafetensors`, which is filesystem-backed and not
//!     directly constructable in the browser from a `Vec<u8>`. The
//!     `load_*` constructors below preserve the JS-facing API surface
//!     but fail fast at runtime with an actionable error message.
//!   - **YoloV8Pose**. The eager wasm wrapped a `YoloV8Pose` model with
//!     a separate keypoint head; there is no `lazy_yolov8_pose` module
//!     yet. `ModelPose::load_` returns a graceful error.
//!   - **Non-square input**. `YoloV8Config` requires a square
//!     `image_size`; the worker letterboxes the image to `640×640`
//!     before running forward.

use crate::model::{report_detect, report_pose, Bbox, Multiples};
use fuel::lazy::LazyTensor;
use fuel::lazy_yolov8::{YoloV8Config, YoloV8Model, YoloV8Weights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Result, Shape};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use yew_agent::{HandlerId, Public, WorkerLink};

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => ($crate::worker::log(&format_args!($($t)*).to_string()))
}

// Communication to the worker happens through bincode, the model weights and configs are fetched
// on the main thread and transferred via the following structure.
#[derive(Serialize, Deserialize)]
pub struct ModelData {
    pub weights: Vec<u8>,
    pub model_size: String,
}

#[derive(Serialize, Deserialize)]
pub struct RunData {
    pub image_data: Vec<u8>,
    pub conf_threshold: f32,
    pub iou_threshold: f32,
}

// ---- Detect model ----------------------------------------------------------

pub struct Model {
    model: YoloV8Model,
}

impl Model {
    pub fn run(
        &self,
        image_data: Vec<u8>,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<Vec<Bbox>>> {
        console_log!("image data: {}", image_data.len());
        let image_data = std::io::Cursor::new(image_data);
        let original_image = image::ImageReader::new(image_data)
            .with_guessed_format()?
            .decode()
            .map_err(fuel::Error::wrap)?;
        let isize = self.model.config.image_size; // square — required by lazy_yolov8
        // Letterbox to a square `isize × isize` input — the lazy
        // `YoloV8Config` only supports square inputs (no separate H/W
        // knobs today).
        let img = original_image.resize_exact(
            isize as u32,
            isize as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let data = img.to_rgb8().into_raw();
        // HWC u8 (0..255) → CHW f32 (0..1).
        let mut chw = vec![0.0_f32; 3 * isize * isize];
        for y in 0..isize {
            for x in 0..isize {
                for c in 0..3 {
                    let src = (y * isize + x) * 3 + c;
                    let dst = c * isize * isize + y * isize + x;
                    chw[dst] = data[src] as f32 / 255.0;
                }
            }
        }
        let raw = self.model.forward(&chw)?;

        // Realize cls + reg, build the `[pred_size, npreds]` layout
        // `report_detect` consumes (pred_size = 4 + num_classes; first
        // 4 channels are pixel-space `cx, cy, w, h`; remaining channels
        // are sigmoid(class logits)).
        let cls = raw.cls_logits.realize_f32(); // [nc, N] row-major
        let reg = raw.reg_dists.realize_f32(); // [4, N] row-major
        let n = raw.strides.len();
        let nc = self.model.config.num_classes;
        debug_assert_eq!(cls.len(), nc * n);
        debug_assert_eq!(reg.len(), 4 * n);
        debug_assert_eq!(raw.grid_xy.len(), 2 * n);

        let pred_size = nc + 4;
        let mut pred = vec![0.0_f32; pred_size * n];
        for i in 0..n {
            let stride = raw.strides[i];
            let cx_grid = raw.grid_xy[2 * i];
            let cy_grid = raw.grid_xy[2 * i + 1];
            let l = reg[i] * stride;
            let t = reg[n + i] * stride;
            let r = reg[2 * n + i] * stride;
            let b = reg[3 * n + i] * stride;
            // xyxy pixel-space box → cx, cy, w, h.
            let x1 = cx_grid * stride - l;
            let y1 = cy_grid * stride - t;
            let x2 = cx_grid * stride + r;
            let y2 = cy_grid * stride + b;
            let cx = (x1 + x2) * 0.5;
            let cy = (y1 + y2) * 0.5;
            let bw = x2 - x1;
            let bh = y2 - y1;
            pred[0 * n + i] = cx;
            pred[1 * n + i] = cy;
            pred[2 * n + i] = bw;
            pred[3 * n + i] = bh;
            for c in 0..nc {
                let logit = cls[c * n + i];
                let prob = 1.0_f32 / (1.0 + (-logit).exp());
                pred[(4 + c) * n + i] = prob;
            }
        }
        console_log!("generated predictions [{pred_size}, {n}]");
        let bboxes = report_detect(
            &pred,
            pred_size,
            n,
            original_image,
            isize,
            isize,
            conf_threshold,
            iou_threshold,
        )?;
        Ok(bboxes)
    }

    pub fn load_(weights: Vec<u8>, model_size: &str) -> Result<Self> {
        // The lazy substrate exposes `v8n()` only — the other size
        // variants share the architecture but their configs haven't
        // been added yet. Drop the request with an actionable error.
        let _multiples = match model_size {
            "n" => Multiples::n(),
            "s" | "m" | "l" | "x" => {
                return Err(fuel::Error::Msg(format!(
                    "lazy YoloV8 only exposes `v8n` today; model_size {model_size:?} \
                     needs a `YoloV8Config::v8{model_size}()` constructor in \
                     fuel-core::lazy_yolov8. Use `n` for now."
                )));
            }
            _ => {
                return Err(fuel::Error::Msg(
                    "invalid model size: must be n, s, m, l or x".to_string(),
                ));
            }
        };

        // The lazy weight loader is filesystem-backed (`MmapedSafetensors`)
        // and a stub today. Surface the JS-facing weight buffer to keep
        // the API surface stable, but fail fast at runtime with the
        // canonical "loader pending" diagnostic. Type-check the stub
        // signature so future API drift is caught at compile time.
        let _ = weights;
        let _stub: fn(&MmapedSafetensors, &YoloV8Config) -> fuel::Result<YoloV8Weights> =
            YoloV8Weights::load_from_mmapped;
        Err(fuel::Error::Msg(
            "YoloV8Weights::load_from_mmapped is a stub today; the lazy \
             YOLOv8 port cannot load HuggingFace safetensors yet. Track \
             progress in fuel_core::lazy_yolov8."
                .to_string(),
        ))

        // Once the loader lands and a `Vec<u8>` → Weights pathway exists
        // (e.g. via `BufferedSafetensors`), the construction tail would
        // be:
        //
        // let config = YoloV8Config::v8n();
        // let weights = load_yolov8_weights(weights, &config)?;
        // let model = YoloV8Model { config, weights };
        // Ok(Self { model })
    }

    pub fn load(md: ModelData) -> Result<Self> {
        Self::load_(md.weights, &md.model_size.to_string())
    }
}

// ---- Pose model ------------------------------------------------------------

pub struct ModelPose {
    // The lazy substrate has no `lazy_yolov8_pose` yet; the pose
    // wrapper retains its JS-facing API surface (`load_`, `run`) so
    // the worker keeps building, but every method returns a clean
    // "not yet supported" error. Once a `YoloV8PoseModel` lands the
    // struct gains a `model: YoloV8PoseModel` field and the methods
    // wire it up just like `Model` above.
    _private: (),
}

impl ModelPose {
    pub fn run(
        &self,
        _image_data: Vec<u8>,
        _conf_threshold: f32,
        _iou_threshold: f32,
    ) -> Result<Vec<Bbox>> {
        // Keep `report_pose` referenced so it doesn't decay to dead
        // code while the lazy pose port is pending — the moment a
        // `YoloV8PoseModel` lands, this is what feeds it.
        let _f: fn(
            &[f32],
            usize,
            usize,
            image::DynamicImage,
            usize,
            usize,
            f32,
            f32,
        ) -> Result<Vec<Bbox>> = report_pose;
        Err(fuel::Error::Msg(
            "YOLOv8-Pose is not yet supported by the lazy substrate \
             (no `lazy_yolov8_pose` module); only the detect model is \
             available. Track progress in fuel-core::lazy_yolov8."
                .to_string(),
        ))
    }

    pub fn load_(_weights: Vec<u8>, _model_size: &str) -> Result<Self> {
        // Type-check `LazyTensor::from_f32` / `Shape` / `Device::cpu()`
        // / `Arc<[f32]>` so the wasm port's input-tensor path keeps
        // compiling against any future API drift before the lazy pose
        // model ships.
        let _ = |buf: Arc<[f32]>| -> LazyTensor {
            LazyTensor::from_f32(buf, Shape::from_dims(&[1, 3, 1, 1]), &Device::cpu())
        };
        Err(fuel::Error::Msg(
            "YOLOv8-Pose is not yet supported by the lazy substrate \
             (no `lazy_yolov8_pose` module). The eager wasm port wrapped \
             a `YoloV8Pose` model; porting it requires adding a \
             `lazy_yolov8_pose` to fuel-core."
                .to_string(),
        ))
    }

    pub fn load(md: ModelData) -> Result<Self> {
        Self::load_(md.weights, &md.model_size.to_string())
    }
}

// ---- Worker bridge ---------------------------------------------------------

pub struct Worker {
    link: WorkerLink<Self>,
    model: Option<Model>,
}

#[derive(Serialize, Deserialize)]
pub enum WorkerInput {
    ModelData(ModelData),
    RunData(RunData),
}

#[derive(Serialize, Deserialize)]
pub enum WorkerOutput {
    ProcessingDone(std::result::Result<Vec<Vec<Bbox>>, String>),
    WeightsLoaded,
}

impl yew_agent::Worker for Worker {
    type Input = WorkerInput;
    type Message = ();
    type Output = std::result::Result<WorkerOutput, String>;
    type Reach = Public<Self>;

    fn create(link: WorkerLink<Self>) -> Self {
        Self { link, model: None }
    }

    fn update(&mut self, _msg: Self::Message) {
        // no messaging
    }

    fn handle_input(&mut self, msg: Self::Input, id: HandlerId) {
        let output = match msg {
            WorkerInput::ModelData(md) => match Model::load(md) {
                Ok(model) => {
                    self.model = Some(model);
                    Ok(WorkerOutput::WeightsLoaded)
                }
                Err(err) => Err(format!("model creation error {err:?}")),
            },
            WorkerInput::RunData(rd) => match &mut self.model {
                None => Err("model has not been set yet".to_string()),
                Some(model) => {
                    let result = model
                        .run(rd.image_data, rd.conf_threshold, rd.iou_threshold)
                        .map_err(|e| e.to_string());
                    Ok(WorkerOutput::ProcessingDone(result))
                }
            },
        };
        self.link.respond(id, output);
    }

    fn name_of_resource() -> &'static str {
        "worker.js"
    }

    fn resource_path_is_relative() -> bool {
        true
    }
}
