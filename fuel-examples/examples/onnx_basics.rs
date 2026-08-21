use anyhow::Result;
use fuel::lazy::LazyTensor;
use fuel::{Device, Shape};
use std::sync::Arc;

use clap::{Parser, Subcommand};

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Print {
        #[arg(long)]
        file: String,
    },
    SimpleEval {
        #[arg(long)]
        file: String,
    },
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Print { file } => {
            let model = fuel_onnx::read_file(file)?;
            println!("{model:?}");
            let graph = model.graph.unwrap();
            for node in graph.node.iter() {
                println!("{node:?}");
            }
        }
        Command::SimpleEval { file } => {
            let model = fuel_onnx::read_file(file)?;
            let graph = model.graph.as_ref().unwrap();
            let constants: std::collections::HashSet<_> =
                graph.initializer.iter().map(|i| i.name.as_str()).collect();
            let mut inputs = std::collections::HashMap::new();
            // Every `LazyTensor::from_*` mints a NEW graph and tensors only
            // combine within one graph, so the first input becomes the anchor
            // and the rest are built on its graph.
            let mut anchor: Option<LazyTensor> = None;
            let dev = Device::cpu();
            for input in graph.input.iter() {
                if constants.contains(input.name.as_str()) {
                    continue;
                }

                let type_ = input.r#type.as_ref().expect("no type for input");
                let type_ = type_.value.as_ref().expect("no type.value for input");
                let value = match type_ {
                    fuel_onnx::onnx::type_proto::Value::TensorType(tt) => {
                        let dt = fuel_onnx::dtype(tt.elem_type).map_err(|e| {
                            anyhow::anyhow!("unsupported data-type for {}: {e}", input.name)
                        })?;
                        let shape = tt.shape.as_ref().expect("no tensortype.shape for input");
                        let dims = shape
                            .dim
                            .iter()
                            .map(|dim| match dim.value.as_ref().expect("no dim value") {
                                fuel_onnx::onnx::tensor_shape_proto::dimension::Value::DimValue(
                                    v,
                                ) => Ok(*v as usize),
                                fuel_onnx::onnx::tensor_shape_proto::dimension::Value::DimParam(
                                    _,
                                ) => Ok(42),
                            })
                            .collect::<Result<Vec<usize>>>()?;
                        let n: usize = dims.iter().product();
                        let zeros: Arc<[f32]> = Arc::from(vec![0f32; n]);
                        let t = match &anchor {
                            None => LazyTensor::from_f32(zeros, Shape::from_dims(&dims), &dev),
                            Some(a) => LazyTensor::from_f32_on(
                                a.graph(),
                                zeros,
                                Shape::from_dims(&dims),
                                &dev,
                            ),
                        };
                        t.to_dtype(dt)?
                    }
                    type_ => anyhow::bail!("unsupported input type {type_:?}"),
                };
                if anchor.is_none() {
                    anchor = Some(value.clone());
                }
                println!(
                    "input {}: shape {:?} dtype {:?}",
                    input.name,
                    value.shape().dims(),
                    value.dtype()
                );
                inputs.insert(input.name.clone(), value);
            }
            let outputs = fuel_onnx::LazyOnnxEval::from_model(model).run(&inputs)?;
            for (name, value) in outputs.iter() {
                // Lazy: realize to inspect.
                println!(
                    "output {name}: shape {:?} = {:?}",
                    value.shape().dims(),
                    value.to_dtype(fuel::DType::F32)?.realize_f32()
                );
            }
        }
    }
    Ok(())
}
