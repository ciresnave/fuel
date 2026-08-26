// SPDX-License-Identifier: MIT OR Apache-2.0
//! `fuel-tensor-tools` — inspect and requantize model weight files.
//!
//! # Why this is host code, not tensor code
//!
//! B6 retired the eager `Tensor` and this CLI was archived with it, because
//! every path went through `QTensor::quantize` / `dequantize`. Restoring it did
//! **not** require porting it to `Tensor`: this is a *file-format
//! converter*. It reads bytes, rearranges them, and writes bytes. The eager
//! tensor was never doing anything a `Vec<f32>` could not, and routing a
//! whole-file requantization through a lazy graph would build a DAG whose only
//! purpose is to be realized immediately.
//!
//! So it now speaks `fuel-formats` readers, host `Vec<f32>`, and
//! `fuel_quantized`'s `QuantizedType` trait directly.
//!
//! # Formats
//!
//! | format       | ls | print | quantize |
//! |--------------|----|-------|----------|
//! | gguf         | y  | y     | y        |
//! | ggml         | y  | y     |          |
//! | safetensors  | y  | y     |          |
//! | pth          | y  |       |          |
//!
//! **npz support was dropped, not ported.** `fuel-core`'s `npy.rs` was deleted
//! in B6 and no npy reader survives anywhere in the tree — `fuel-formats` has
//! ggml, gguf, imatrix, pickle and safetensors, and no npy. Re-adding it means
//! writing a reader, which is its own change rather than part of this restore.
//!
//! `pth` lists tensor metadata only: `fuel_formats::pickle` exposes
//! `read_pth_tensor_info`, which yields shapes and dtypes but not the tensor
//! payloads, so `print` for pth would need a pickle *payload* reader.

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use fuel_formats::gguf;
use fuel_quantized::{GgmlDType, cpu_from_data, cpu_zeros};
use std::io::{Read, Seek, SeekFrom, Write};

// ------------------------------------------------------------------ format --

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Safetensors,
    Pth,
    Ggml,
    Gguf,
}

impl Format {
    fn infer<P: AsRef<std::path::Path>>(p: P) -> Option<Self> {
        match p.as_ref().extension().and_then(|e| e.to_str())? {
            "safetensors" | "safetensor" => Some(Self::Safetensors),
            "pth" | "pt" | "bin" => Some(Self::Pth),
            "ggml" => Some(Self::Ggml),
            "gguf" => Some(Self::Gguf),
            _ => None,
        }
    }

    fn resolve(explicit: Option<Self>, path: &std::path::Path) -> Result<Self> {
        match explicit.or_else(|| Self::infer(path)) {
            Some(f) => Ok(f),
            None => bail!("{path:?}: cannot infer format from the file extension — pass --format"),
        }
    }
}

// ------------------------------------------------------------- quantization --

#[derive(ValueEnum, Debug, Clone, Copy)]
enum Quantization {
    #[value(name = "q4_0")]
    Q4_0,
    #[value(name = "q4_1")]
    Q4_1,
    #[value(name = "q5_0")]
    Q5_0,
    #[value(name = "q5_1")]
    Q5_1,
    #[value(name = "q8_0")]
    Q8_0,
    #[value(name = "q8_1")]
    Q8_1,
    Q2k,
    Q3k,
    Q4k,
    Q5k,
    Q6k,
    Q8k,
    F16,
    F32,
}

impl Quantization {
    fn dtype(self) -> GgmlDType {
        match self {
            Self::Q4_0 => GgmlDType::Q4_0,
            Self::Q4_1 => GgmlDType::Q4_1,
            Self::Q5_0 => GgmlDType::Q5_0,
            Self::Q5_1 => GgmlDType::Q5_1,
            Self::Q8_0 => GgmlDType::Q8_0,
            Self::Q8_1 => GgmlDType::Q8_1,
            Self::Q2k => GgmlDType::Q2K,
            Self::Q3k => GgmlDType::Q3K,
            Self::Q4k => GgmlDType::Q4K,
            Self::Q5k => GgmlDType::Q5K,
            Self::Q6k => GgmlDType::Q6K,
            Self::Q8k => GgmlDType::Q8K,
            Self::F16 => GgmlDType::F16,
            Self::F32 => GgmlDType::F32,
        }
    }
}

/// Which tensors to requantize. Mirrors llama.cpp: 2-D `.weight` tensors get
/// the requested dtype, `output.weight` is pinned to Q6_K, everything else is
/// passed through untouched.
#[derive(ValueEnum, Debug, Clone, Copy)]
enum QuantizationMode {
    Llama,
}

impl QuantizationMode {
    fn target(self, name: &str, rank: usize, requested: GgmlDType) -> Option<GgmlDType> {
        match self {
            Self::Llama => {
                if name.ends_with(".weight") && rank == 2 {
                    if name == "output.weight" {
                        Some(GgmlDType::Q6K)
                    } else {
                        Some(requested)
                    }
                } else {
                    None
                }
            }
        }
    }
}

// ------------------------------------------------------------------- CLI ----

#[derive(Subcommand, Debug, Clone)]
enum Command {
    /// List the tensors in a file with their shapes and dtypes.
    Ls {
        files: Vec<std::path::PathBuf>,
        #[arg(long)]
        format: Option<Format>,
        /// Also print each tensor's element count / byte size.
        #[arg(long)]
        verbose: bool,
    },
    /// Print summary statistics for named tensors (or all of them).
    Print {
        file: std::path::PathBuf,
        names: Vec<String>,
        #[arg(long)]
        format: Option<Format>,
        /// Also print this many leading values.
        #[arg(long, default_value_t = 0)]
        head: usize,
    },
    /// Requantize a gguf file.
    Quantize {
        /// Input gguf file.
        in_file: std::path::PathBuf,
        #[arg(long)]
        out_file: std::path::PathBuf,
        #[arg(long, value_enum, default_value_t = Quantization::Q4k)]
        quantization: Quantization,
        #[arg(long, value_enum, default_value_t = QuantizationMode::Llama)]
        mode: QuantizationMode,
    },
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

// ------------------------------------------------------------- gguf helpers -

/// A gguf tensor's raw bytes, read straight out of the file.
struct RawGguf {
    dtype: GgmlDType,
    dims: Vec<usize>,
    bytes: Vec<u8>,
}

fn gguf_read_raw<R: Read + Seek>(
    reader: &mut R,
    content: &gguf::Content,
    name: &str,
) -> Result<RawGguf> {
    let info = content
        .tensor_infos
        .get(name)
        .with_context(|| format!("gguf: no tensor named '{name}'"))?;
    let elems = info.shape.elem_count();
    let block = info.ggml_dtype.block_size();
    if !elems.is_multiple_of(block) {
        bail!("gguf: '{name}' has {elems} elements, not a multiple of block size {block}");
    }
    let size = elems / block * info.ggml_dtype.type_size();
    let mut bytes = vec![0u8; size];
    reader.seek(SeekFrom::Start(content.tensor_data_offset + info.offset))?;
    reader.read_exact(&mut bytes)?;
    Ok(RawGguf {
        dtype: info.ggml_dtype,
        dims: info.shape.dims().to_vec(),
        bytes,
    })
}

/// Dequantize raw block bytes to host f32.
fn to_f32(dtype: GgmlDType, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
    let q = cpu_from_data(dtype, std::borrow::Cow::Borrowed(bytes));
    let buf = q.dequantize(elem_count)?;
    Ok(buf.as_slice::<f32>()?.to_vec())
}

/// Quantize host f32 into raw block bytes for `dtype`.
fn from_f32(dtype: GgmlDType, xs: &[f32]) -> Result<Vec<u8>> {
    let mut q = cpu_zeros(dtype, xs.len());
    q.from_float(xs);
    // SAFETY: `as_ptr` and `storage_size_in_bytes` describe the same live
    // buffer owned by `q`, and the bytes are copied out before `q` is dropped.
    let bytes = unsafe { std::slice::from_raw_parts(q.as_ptr(), q.storage_size_in_bytes()) };
    Ok(bytes.to_vec())
}

fn summarize(name: &str, dims: &[usize], dtype: GgmlDType, xs: &[f32], head: usize) {
    let n = xs.len();
    let (mut lo, mut hi, mut sum, mut nonfinite) = (f32::INFINITY, f32::NEG_INFINITY, 0f64, 0usize);
    for &v in xs {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
            sum += v as f64;
        } else {
            nonfinite += 1;
        }
    }
    let finite = n - nonfinite;
    let mean = if finite > 0 {
        sum / finite as f64
    } else {
        f64::NAN
    };
    println!("==== {name} ====");
    println!("  dtype={dtype:?}  shape={dims:?}  elems={n}");
    if finite > 0 {
        println!("  min={lo:+.6}  mean={mean:+.6}  max={hi:+.6}");
    }
    if nonfinite > 0 {
        println!("  NON-FINITE VALUES: {nonfinite}");
    }
    if head > 0 {
        let k = head.min(n);
        println!("  first {k}: {:?}", &xs[..k]);
    }
}

// ------------------------------------------------------------------- ls -----

fn run_ls(file: &std::path::PathBuf, format: Option<Format>, verbose: bool) -> Result<()> {
    let format = Format::resolve(format, file)?;
    println!("--- {file:?} [{format:?}] ---");
    match format {
        Format::Gguf => {
            let mut f = std::fs::File::open(file)?;
            let content = gguf::Content::read(&mut f)?;
            let mut names: Vec<_> = content.tensor_infos.keys().cloned().collect();
            names.sort();
            for name in names {
                let info = &content.tensor_infos[&name];
                if verbose {
                    println!(
                        "{name}: {:?} {:?} ({} elems)",
                        info.ggml_dtype,
                        info.shape.dims(),
                        info.shape.elem_count()
                    );
                } else {
                    println!("{name}: {:?} {:?}", info.ggml_dtype, info.shape.dims());
                }
            }
        }
        Format::Ggml => {
            let mut f = std::fs::File::open(file)?;
            let header = fuel_formats::ggml::Header::read(&mut f)?;
            println!("hparams: {:?}", header.hparams);
            while let Ok(raw) = fuel_formats::ggml::read_one_raw_tensor(&mut f, header.magic) {
                println!("{}: {:?} {:?}", raw.name, raw.dtype, raw.dims);
            }
        }
        Format::Safetensors => {
            // SAFETY: the file must not be modified while the mapping is alive.
            let st = unsafe { fuel::safetensors::MmapedSafetensors::new(file)? };
            let mut tensors = st.tensors();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, view) in tensors {
                if verbose {
                    println!(
                        "{name}: {:?} {:?} ({} bytes)",
                        view.dtype(),
                        view.shape(),
                        view.data().len()
                    );
                } else {
                    println!("{name}: {:?} {:?}", view.dtype(), view.shape());
                }
            }
        }
        Format::Pth => {
            let infos = fuel_formats::pickle::read_pth_tensor_info(file, false, None)?;
            for info in infos {
                println!(
                    "{}: {:?} {:?}",
                    info.name,
                    info.dtype,
                    info.layout.shape().dims()
                );
            }
        }
    }
    Ok(())
}

// ----------------------------------------------------------------- print ----

fn run_print(
    file: &std::path::PathBuf,
    names: Vec<String>,
    format: Option<Format>,
    head: usize,
) -> Result<()> {
    let format = Format::resolve(format, file)?;
    match format {
        Format::Gguf => {
            let mut f = std::fs::File::open(file)?;
            let content = gguf::Content::read(&mut f)?;
            let names = if names.is_empty() {
                let mut n: Vec<_> = content.tensor_infos.keys().cloned().collect();
                n.sort();
                n
            } else {
                names
            };
            for name in names {
                match gguf_read_raw(&mut f, &content, &name) {
                    Ok(raw) => {
                        let elems: usize = raw.dims.iter().product();
                        let xs = to_f32(raw.dtype, &raw.bytes, elems)?;
                        summarize(&name, &raw.dims, raw.dtype, &xs, head);
                    }
                    Err(e) => println!("==== {name} ====\n  {e}"),
                }
            }
        }
        Format::Ggml => {
            let mut f = std::fs::File::open(file)?;
            let header = fuel_formats::ggml::Header::read(&mut f)?;
            let want: std::collections::HashSet<_> = names.iter().cloned().collect();
            while let Ok(raw) = fuel_formats::ggml::read_one_raw_tensor(&mut f, header.magic) {
                if !want.is_empty() && !want.contains(&raw.name) {
                    continue;
                }
                let elems: usize = raw.dims.iter().product();
                let xs = to_f32(raw.dtype, &raw.data, elems)?;
                summarize(&raw.name, &raw.dims, raw.dtype, &xs, head);
            }
        }
        Format::Safetensors => {
            // SAFETY: the file must not be modified while the mapping is alive.
            let st = unsafe { fuel::safetensors::MmapedSafetensors::new(file)? };
            let all = st.tensors();
            let want: std::collections::HashSet<_> = names.iter().cloned().collect();
            for (name, view) in all {
                if !want.is_empty() && !want.contains(&name) {
                    continue;
                }
                // safetensors carries a logical dtype, not a ggml block dtype;
                // only plain f32 is decodable without a widening table here.
                let xs: Vec<f32> = match view.dtype() {
                    safetensors::Dtype::F32 => view
                        .data()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                    other => {
                        println!("==== {name} ====\n  dtype {other:?} not decodable here");
                        continue;
                    }
                };
                let dims: Vec<usize> = view.shape().to_vec();
                summarize(&name, &dims, GgmlDType::F32, &xs, head);
            }
        }
        Format::Pth => bail!(
            "print is not supported for pth: fuel_formats::pickle exposes tensor \
             METADATA (read_pth_tensor_info) but not payloads. Use `ls`."
        ),
    }
    Ok(())
}

// -------------------------------------------------------------- quantize ----

fn run_quantize(
    in_file: &std::path::PathBuf,
    out_file: &std::path::PathBuf,
    quantization: Quantization,
    mode: QuantizationMode,
) -> Result<()> {
    if Format::resolve(None, in_file).ok() != Some(Format::Gguf) {
        bail!("quantize currently supports gguf input only, got {in_file:?}");
    }
    let requested = quantization.dtype();
    let mut f = std::fs::File::open(in_file)?;
    let content = gguf::Content::read(&mut f)?;

    let mut names: Vec<_> = content.tensor_infos.keys().cloned().collect();
    names.sort();

    // Read + requantize into memory first. A streaming writer would need every
    // tensor's FINAL byte size to lay out the offset table, and requantizing is
    // what determines those sizes.
    let mut out: Vec<(String, GgmlDType, Vec<usize>, Vec<u8>)> = Vec::with_capacity(names.len());
    for name in &names {
        let raw = gguf_read_raw(&mut f, &content, name)?;
        let elems: usize = raw.dims.iter().product();
        match mode.target(name, raw.dims.len(), requested) {
            Some(target) if target != raw.dtype => {
                let xs = to_f32(raw.dtype, &raw.bytes, elems)?;
                let bytes = from_f32(target, &xs)?;
                println!("{name}: {:?} -> {target:?}", raw.dtype);
                out.push((name.clone(), target, raw.dims, bytes));
            }
            _ => {
                println!("{name}: {:?} (unchanged)", raw.dtype);
                out.push((name.clone(), raw.dtype, raw.dims, raw.bytes));
            }
        }
    }

    let metadata: Vec<(&str, &gguf::Value)> = content
        .metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    write_gguf(out_file, &metadata, &out)?;
    println!("wrote {out_file:?}");
    Ok(())
}

/// Serialize a gguf v2 file from raw, already-quantized tensor bytes.
///
/// Replaces the deleted `fuel_core::quantized::gguf_file::write`, which took
/// `&QTensor`. Same wire format, raw bytes instead of tensors.
fn write_gguf(
    path: &std::path::Path,
    metadata: &[(&str, &gguf::Value)],
    tensors: &[(String, GgmlDType, Vec<usize>, Vec<u8>)],
) -> Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);

    w.write_u32::<LittleEndian>(0x46554747)?; // "GGUF"
    w.write_u32::<LittleEndian>(2)?; // version
    w.write_u64::<LittleEndian>(tensors.len() as u64)?;
    w.write_u64::<LittleEndian>(metadata.len() as u64)?;
    for (name, value) in metadata {
        gguf::write_string(&mut w, name)?;
        w.write_u32::<LittleEndian>(value.value_type().to_u32())?;
        value.write(&mut w)?;
    }

    // Tensor-info table. Each entry carries the offset of its data RELATIVE to
    // the start of the data section, so the padded layout is computed up front.
    let mut offset = 0usize;
    let mut offsets = Vec::with_capacity(tensors.len());
    for (name, dtype, dims, bytes) in tensors {
        gguf::write_string(&mut w, name)?;
        w.write_u32::<LittleEndian>(dims.len() as u32)?;
        // gguf stores dims in reverse order on disk
        for &d in dims.iter().rev() {
            w.write_u64::<LittleEndian>(d as u64)?;
        }
        w.write_u32::<LittleEndian>(dtype.to_u32())?;
        w.write_u64::<LittleEndian>(offset as u64)?;
        offsets.push(offset);
        let padding = 31 - (31 + bytes.len()) % 32;
        offset += bytes.len() + padding;
    }

    // Align the start of the data section to 32 bytes.
    let pos = w.stream_position()? as usize;
    let padding = 31 - (31 + pos) % 32;
    w.write_all(&vec![0u8; padding])?;
    let data_start = w.stream_position()? as usize;

    for (expected, (name, _, _, bytes)) in offsets.iter().zip(tensors) {
        let pos = w.stream_position()? as usize;
        if data_start + expected != pos {
            bail!(
                "gguf write: '{name}' should start at {} but the writer is at {pos} \
                 (data_start={data_start}, offset={expected})",
                data_start + expected
            );
        }
        w.write_all(bytes)?;
        let padding = 31 - (31 + bytes.len()) % 32;
        w.write_all(&vec![0u8; padding])?;
    }
    w.flush()?;
    Ok(())
}

// ------------------------------------------------------------------ main ----

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Ls {
            files,
            format,
            verbose,
        } => {
            for file in files.iter() {
                if let Err(e) = run_ls(file, format, verbose) {
                    println!("{file:?}: {e}");
                }
            }
        }
        Command::Print {
            file,
            names,
            format,
            head,
        } => run_print(&file, names, format, head)?,
        Command::Quantize {
            in_file,
            out_file,
            quantization,
            mode,
        } => run_quantize(&in_file, &out_file, quantization, mode)?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip through the quantizer: f32 -> Q4_0 blocks -> f32. Q4_0 is
    /// lossy, so this pins that the plumbing is wired (right element count,
    /// values in the right neighbourhood), not bit-exactness.
    #[test]
    fn quantize_dequantize_round_trips_through_host_bytes() -> Result<()> {
        // 64 elements = 2 full Q4_0 blocks (block size 32).
        let xs: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 8.0).collect();
        let bytes = from_f32(GgmlDType::Q4_0, &xs)?;
        assert!(!bytes.is_empty(), "quantizer produced no bytes");

        let back = to_f32(GgmlDType::Q4_0, &bytes, xs.len())?;
        assert_eq!(
            back.len(),
            xs.len(),
            "element count must survive the round trip"
        );
        for (i, (a, b)) in xs.iter().zip(&back).enumerate() {
            assert!(
                (a - b).abs() < 0.5,
                "index {i}: {a} -> {b} is further off than Q4_0 should be"
            );
        }
        Ok(())
    }

    /// F32 "quantization" must be exactly lossless — if this drifts, the byte
    /// plumbing (as_ptr / storage_size_in_bytes) is wrong.
    #[test]
    fn f32_round_trip_is_bit_exact() -> Result<()> {
        let xs: Vec<f32> = vec![-1.5, 0.0, 1.0, 3.25, -7.125];
        let bytes = from_f32(GgmlDType::F32, &xs)?;
        assert_eq!(bytes.len(), xs.len() * 4, "F32 must be 4 bytes per element");
        let back = to_f32(GgmlDType::F32, &bytes, xs.len())?;
        assert_eq!(back, xs, "F32 round trip must be bit-exact");
        Ok(())
    }

    /// The llama mode's selection rule, which decides what actually gets
    /// requantized. Getting this wrong silently produces a file that is the
    /// right size and the wrong precision.
    #[test]
    fn llama_mode_selects_2d_weights_and_pins_the_output_head() {
        let m = QuantizationMode::Llama;
        let q = GgmlDType::Q4K;
        assert_eq!(m.target("blk.0.attn_q.weight", 2, q), Some(GgmlDType::Q4K));
        // the output head is pinned to Q6_K regardless of the request
        assert_eq!(m.target("output.weight", 2, q), Some(GgmlDType::Q6K));
        // 1-D weights (norms/biases) are left alone
        assert_eq!(m.target("blk.0.attn_norm.weight", 1, q), None);
        // non-".weight" tensors are left alone
        assert_eq!(m.target("blk.0.attn_q.bias", 2, q), None);
    }

    #[test]
    fn format_inference_from_extension() {
        assert_eq!(Format::infer("m.gguf"), Some(Format::Gguf));
        assert_eq!(Format::infer("m.safetensors"), Some(Format::Safetensors));
        assert_eq!(Format::infer("m.pth"), Some(Format::Pth));
        assert_eq!(Format::infer("m.ggml"), Some(Format::Ggml));
        assert_eq!(Format::infer("m.unknown"), None);
        // an unknown extension must be an error, not a silent default
        assert!(Format::resolve(None, std::path::Path::new("m.unknown")).is_err());
    }
}
