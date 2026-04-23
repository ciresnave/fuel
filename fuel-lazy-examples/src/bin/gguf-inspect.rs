// Print GGUF metadata keys and tensor names. Helps debug loader issues.
//
// Usage: cargo run --release --bin gguf-inspect -- path/to/model.gguf

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fuel::quantized::gguf_mmap::MmapedContent;
    let path = std::env::args().nth(1).expect("usage: gguf-inspect <path>");
    let mc = MmapedContent::from_path(&path)?;
    eprintln!("=== Metadata ({}) ===", mc.metadata().len());
    let mut keys: Vec<_> = mc.metadata().keys().collect();
    keys.sort();
    for k in &keys {
        let v = &mc.metadata()[*k];
        let repr = match v {
            fuel::quantized::gguf_file::Value::U8(x)  => format!("U8({x})"),
            fuel::quantized::gguf_file::Value::I8(x)  => format!("I8({x})"),
            fuel::quantized::gguf_file::Value::U16(x) => format!("U16({x})"),
            fuel::quantized::gguf_file::Value::I16(x) => format!("I16({x})"),
            fuel::quantized::gguf_file::Value::U32(x) => format!("U32({x})"),
            fuel::quantized::gguf_file::Value::I32(x) => format!("I32({x})"),
            fuel::quantized::gguf_file::Value::U64(x) => format!("U64({x})"),
            fuel::quantized::gguf_file::Value::I64(x) => format!("I64({x})"),
            fuel::quantized::gguf_file::Value::F32(x) => format!("F32({x})"),
            fuel::quantized::gguf_file::Value::F64(x) => format!("F64({x})"),
            fuel::quantized::gguf_file::Value::Bool(x) => format!("Bool({x})"),
            fuel::quantized::gguf_file::Value::String(s) => {
                if s.len() > 80 { format!("String({:?}...)", &s[..80]) } else { format!("String({s:?})") }
            }
            fuel::quantized::gguf_file::Value::Array(a) => format!("Array(len={})", a.len()),
        };
        eprintln!("  {k:60}  {repr}");
    }
    eprintln!();
    let mut names: Vec<_> = mc.tensor_names().cloned().collect();
    names.sort();
    eprintln!("=== Tensors ({}) ===", names.len());
    // Show all tensors that don't match "blk.<digit>" (non-layer tensors
    // like embeddings, output head, global norms), then a sample of
    // layer tensors.
    eprintln!("  --- non-layer tensors ---");
    for n in names.iter().filter(|n| !n.starts_with("blk.")) {
        let info = mc.content().tensor_infos.get(n).unwrap();
        eprintln!("  {n:65}  dtype={:?}  shape={:?}", info.ggml_dtype, info.shape.dims());
    }
    eprintln!("  --- first layer tensors ---");
    for n in names.iter().filter(|n| n.starts_with("blk.0.")) {
        let info = mc.content().tensor_infos.get(n).unwrap();
        eprintln!("  {n:65}  dtype={:?}  shape={:?}", info.ggml_dtype, info.shape.dims());
    }
    Ok(())
}
