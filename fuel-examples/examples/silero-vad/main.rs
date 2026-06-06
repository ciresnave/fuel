#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::Parser;

use fuel::lazy::LazyTensor;
use fuel::Device;
use std::sync::Arc;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "silero")]
    Silero,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum SampleRate {
    #[value(name = "8000")]
    Sr8k,
    #[value(name = "16000")]
    Sr16k,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    input: Option<String>,

    #[arg(long)]
    sample_rate: SampleRate,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    /// The model to use.
    #[arg(long, default_value = "silero")]
    which: Which,
}

/// an iterator which reads consecutive frames of le i16 values from a reader
struct I16Frames<R> {
    rdr: R,
    buf: Box<[u8]>,
    len: usize,
    eof: bool,
}
impl<R> I16Frames<R> {
    fn new(rdr: R, frame_size: usize) -> Self {
        I16Frames {
            rdr,
            buf: vec![0; frame_size * std::mem::size_of::<i16>()].into_boxed_slice(),
            len: 0,
            eof: false,
        }
    }
}
impl<R: std::io::Read> Iterator for I16Frames<R> {
    type Item = std::io::Result<Vec<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.eof {
            return None;
        }
        self.len += match self.rdr.read(&mut self.buf[self.len..]) {
            Ok(0) => {
                self.eof = true;
                0
            }
            Ok(n) => n,
            Err(e) => return Some(Err(e)),
        };
        if self.eof || self.len == self.buf.len() {
            let buf = self.buf[..self.len]
                .chunks(2)
                .map(|bs| match bs {
                    [a, b] => i16::from_le_bytes([*a, *b]),
                    _ => unreachable!(),
                })
                .map(|i| i as f32 / i16::MAX as f32)
                .collect();
            self.len = 0;
            Some(Ok(buf))
        } else {
            self.next()
        }
    }
}

fn main() -> Result<()> {
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
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        fuel::utils::with_avx(),
        fuel::utils::with_neon(),
        fuel::utils::with_simd128(),
        fuel::utils::with_f16c()
    );

    let start = std::time::Instant::now();
    let model_id = match &args.model_id {
        Some(model_id) => std::path::PathBuf::from(model_id),
        None => match args.which {
            Which::Silero => hf_hub::api::sync::Api::new()?
                .model("onnx-community/silero-vad".into())
                .get("onnx/model.onnx")?,
        },
    };
    let (sample_rate, frame_size, context_size): (i64, usize, usize) = match args.sample_rate {
        SampleRate::Sr8k => (8000, 256, 32),
        SampleRate::Sr16k => (16000, 512, 64),
    };
    println!("retrieved the files in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let _device = if args.cpu { Device::cpu() } else { Device::cpu() };
    let evaluator = fuel_onnx::LazyOnnxEval::from_path(&model_id)?;

    println!("loaded the model in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    // Persistent state across frames: f32 vector of shape (2, 1, 128).
    let mut state_host: Vec<f32> = vec![0.0_f32; 2 * 1 * 128];
    // Persistent context (carried across frames): f32 vector of shape
    // (1, context_size).
    let mut context_host: Vec<f32> = vec![0.0_f32; context_size];

    let out_names = evaluator
        .model()
        .graph
        .as_ref()
        .map(|g| g.output.clone())
        .unwrap_or_default();

    let mut res = vec![];
    for chunk in I16Frames::new(std::io::stdin().lock(), frame_size) {
        let chunk = chunk.unwrap();
        if chunk.len() < frame_size {
            continue;
        }
        let next_context: Vec<f32> = chunk[frame_size - context_size..].to_vec();

        // Build a fresh graph this frame: anchor on the audio chunk (f32),
        // then build sample_rate (i64 scalar) and state (f32) as siblings.
        let device = Device::cpu();
        let chunk_only =
            LazyTensor::from_f32(chunk.clone(), (1, frame_size), &device);
        let context_t = chunk_only.const_f32_like(
            Arc::<[f32]>::from(context_host.clone().into_boxed_slice()),
            (1, context_size),
        );
        let input_full = context_t.concat(&chunk_only, 1)?;
        let sr_t = chunk_only.const_i64_like(vec![sample_rate], ());
        let state_t = chunk_only.const_f32_like(
            Arc::<[f32]>::from(state_host.clone().into_boxed_slice()),
            (2, 1, 128),
        );

        let inputs = std::collections::HashMap::from_iter([
            ("input".to_string(), input_full),
            ("sr".to_string(), sr_t),
            ("state".to_string(), state_t),
        ]);
        let outputs = evaluator.run(&inputs)?;
        let output = outputs.get(&out_names[0].name).unwrap().clone();
        let new_state = outputs.get(&out_names[1].name).unwrap().clone();
        assert_eq!(new_state.shape().dims(), &[2, 1, 128]);

        // Persist state for next frame.
        state_host = new_state.realize_f32();
        context_host = next_context;

        let output_vec = output.flatten_all()?.realize_f32();
        assert_eq!(output_vec.len(), 1);
        let output = output_vec[0];
        println!("vad chunk prediction: {output}");
        res.push(output);
    }
    println!("calculated prediction in {:?}", start.elapsed());

    let res_len = res.len() as f32;
    let prediction = res.iter().sum::<f32>() / res_len;
    println!("vad average prediction: {prediction}");
    Ok(())
}
