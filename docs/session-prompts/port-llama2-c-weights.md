# Port: Llama2.c binary weight loader

## Eager source

- `fuel-transformers/src/models/llm/llama2_c_weights.rs` (239 LOC)
  — Karpathy's `llama2.c` binary checkpoint format reader. Provides
    `TransformerWeights::from_reader(r, cfg, &device)` returning a
    struct of eager `Tensor`s in the layout the eager `llama2c`
    model expects.

## Lazy module name

Extend `fuel-core/src/lazy_llama2c.rs` with a new public
`load_llama2c_weights<R: Read>` function that returns a
`Llama2cWeights` struct of `LazyTensor`s in the same layout the
shipped `lazy_llama2c::Llama2c` model already consumes.

## Architecture summary

Pure host-side I/O. The file format is a packed sequence of f32
buffers in a fixed order:

1. Token embedding table `[vocab_size, dim]`
2. Per-layer RMSNorm weights `[n_layers, dim]` (att-norm then
   ffn-norm)
3. Per-layer wq/wk/wv/wo `[n_layers, dim, n_heads * head_size]`
   (transposed)
4. Per-layer w1/w2/w3 `[n_layers, hidden_dim, dim]` / `[dim, hidden_dim]`
5. Final RMSNorm `[dim]`
6. Optional rope tables (older versions) or shared classifier
7. Classifier head `[vocab_size, dim]` (optional — defaults to
   tied embeddings)

No graph ops. Each tensor is read into a `Vec<f32>` and wrapped via
`LazyTensor::from_vec(data, shape, &device)?`.

## Primitives needed

- None — pure I/O. Uses only `LazyTensor::from_vec` which already
  exists.

## Reusable modules

- `lazy_llama2c::Llama2c` — the model already exists; this port
  just adds the weight loader.
- `byteorder::ReadBytesExt` (already used by the eager file) for
  reading the little-endian header.

## Open questions

- `from_path(p)` vs `from_reader(r)` — keep both? Eager exposes
  `from_reader` only and users wrap with `BufReader::new(File::open(p)?)`.
  Match that surface.
- Karpathy's format has gone through two versions (v1 with header
  magic `0x616b3432`, v0 without). Eager supports both via the
  `Llama2cVersion` enum. Mirror exactly.

## Splits

Single session, ~200 lines mechanical translation. No splits.

## Test strategy

- Synthesize a tiny binary blob in-memory matching the v1 format
  (vocab=4, dim=8, n_layers=1, n_heads=2, hidden_dim=16, seq_len=4),
  call `load_llama2c_weights(&mut Cursor::new(&bytes), &cfg, &device)`,
  assert returned tensor shapes match.
- Round-trip: `lazy_llama2c::Llama2c::new(weights, cfg)?.forward(&tokens, 0)?`
  produces a finite logits tensor of the expected shape.

## References

- Eager source: `fuel-transformers/src/models/llm/llama2_c_weights.rs`
- Karpathy's spec: <https://github.com/karpathy/llama2.c>
  (`export.py` `model_export_v1` is authoritative for the byte
  layout).
- Already-shipped: `lazy_llama2c` (consumer — this port feeds it).
