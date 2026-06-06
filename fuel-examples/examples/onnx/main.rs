#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use fuel::lazy::LazyTensor;
use fuel::Device;
use fuel_examples::save_image;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    SqueezeNet,
    EfficientNet,
    EsrGan,
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    image: String,

    #[arg(long)]
    model: Option<String>,

    /// The model to be used.
    #[arg(value_enum, long, default_value_t = Which::SqueezeNet)]
    which: Which,
}

// Mirrors `fuel_examples::imagenet::load_image_with_std_mean` but
// returns a host-resident (data, shape) tuple ready for
// LazyTensor::from_f32 instead of an eager Tensor.
fn load_image_chw_f32(
    p: &str,
    res: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Result<(Vec<f32>, (usize, usize, usize))> {
    let img = image::ImageReader::open(p)?
        .decode()
        .map_err(|e| anyhow!("image decode: {e}"))?
        .resize_to_fill(
            res as u32,
            res as u32,
            image::imageops::FilterType::Triangle,
        );
    let img = img.to_rgb8();
    let raw = img.into_raw(); // HxWx3, u8
    // Convert HWC u8 -> CHW f32, normalized.
    let mut chw = vec![0.0_f32; 3 * res * res];
    for h in 0..res {
        for w in 0..res {
            for c in 0..3 {
                let v = raw[(h * res + w) * 3 + c] as f32 / 255.0;
                chw[c * res * res + h * res + w] = (v - mean[c]) / std[c];
            }
        }
    }
    Ok((chw, (3, res, res)))
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::cpu();

    let (image_chw, image_shape) = match args.which {
        Which::SqueezeNet | Which::EfficientNet => load_image_chw_f32(
            &args.image,
            224,
            &[0.485_f32, 0.456, 0.406],
            &[0.229_f32, 0.224, 0.225],
        )?,
        Which::EsrGan => load_image_chw_f32(
            &args.image,
            128,
            &[0.0_f32, 0.0, 0.0],
            &[1.0_f32, 1.0, 1.0],
        )?,
    };

    let image_lazy = LazyTensor::from_f32(image_chw, image_shape, &device);
    // EfficientNet wants HWC, others want CHW.
    let image_lazy = match args.which {
        Which::SqueezeNet => image_lazy,
        Which::EfficientNet => image_lazy.permute((1, 2, 0))?,
        Which::EsrGan => image_lazy,
    };

    println!("loaded image (lazy) shape {:?}", image_lazy.shape());

    let model = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => match args.which {
            Which::SqueezeNet => hf_hub::api::sync::Api::new()?
                .model("lmz/fuel-onnx".into())
                .get("squeezenet1.1-7.onnx")?,
            Which::EfficientNet => hf_hub::api::sync::Api::new()?
                .model("onnx/EfficientNet-Lite4".into())
                .get("efficientnet-lite4-11.onnx")?,
            Which::EsrGan => hf_hub::api::sync::Api::new()?
                .model("qualcomm/Real-ESRGAN-x4plus".into())
                .get("Real-ESRGAN-x4plus.onnx")?,
        },
    };

    let evaluator = fuel_onnx::LazyOnnxEval::from_path(&model)?;
    let graph = evaluator
        .model()
        .graph
        .as_ref()
        .ok_or_else(|| anyhow!("ONNX model has no graph"))?;
    let mut inputs = std::collections::HashMap::new();
    inputs.insert(graph.input[0].name.to_string(), image_lazy.unsqueeze(0)?);
    let mut outputs = evaluator.run(&inputs)?;
    let output_name = graph.output[0].name.clone();
    let output = outputs.remove(&output_name).unwrap();

    match args.which {
        Which::EfficientNet | Which::SqueezeNet => {
            let prs = match args.which {
                Which::SqueezeNet => output.softmax_last_dim()?,
                _ => output,
            };
            // Realize logits/probabilities to host.
            let prs_vec = prs.realize_f32();
            // Drop the leading batch dim: take the first `n` rows where
            // n == elem count / batch. Output is always [batch=1, classes].
            let batch = prs.shape().dims()[0];
            let classes = prs_vec.len() / batch;
            let row: &[f32] = &prs_vec[..classes];

            // Sort the predictions and take the top 5.
            let mut top: Vec<_> = row.iter().enumerate().collect();
            top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            let top = top.into_iter().take(5).collect::<Vec<_>>();

            for &(i, p) in &top {
                println!(
                    "{:50}: {:.2}%",
                    fuel_examples::imagenet::CLASSES[i],
                    p * 100.0
                );
            }
        }
        Which::EsrGan => {
            // ESR-GAN: realize output, scale by 255 in host f32, then
            // emit as u8 image. The output shape is (1, 3, H, W).
            let dims = output.shape();
            let dims = dims.dims();
            let out_vec = output.realize_f32();
            // Drop batch dim.
            let (c, h, w) = (dims[1], dims[2], dims[3]);
            let mut chw_u8 = vec![0_u8; c * h * w];
            for i in 0..(c * h * w) {
                let v = (out_vec[i].clamp(0.0, 1.0) * 255.0).round() as u8;
                chw_u8[i] = v;
            }
            // Rebuild a lazy u8 tensor for save_image (which expects an
            // eager Tensor). save_image is part of fuel-examples; bridge
            // by going through a fresh eager Tensor via fuel::Tensor.
            let pb = std::path::PathBuf::from(args.image);
            let input_file_name = pb.file_name().unwrap();
            let mut output_file_name = std::ffi::OsString::from("super_");
            output_file_name.push(input_file_name);

            let eager_u8 =
                fuel::Tensor::from_vec(chw_u8, (c, h, w), &Device::cpu())?;
            save_image(&eager_u8, output_file_name)?;
        }
    }

    Ok(())
}
