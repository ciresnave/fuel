# fuel-formats

Transport-independent parsers for tensor wire formats:

- [`safetensors`](src/safetensors.rs) — re-exports from upstream `safetensors` crate + `MmapedFile` convenience
- [`pickle`](src/pickle.rs) — minimal PyTorch `.pth` reader (subset of Python pickle protocol 2)
- [`gguf`](src/gguf.rs) — llama.cpp's GGUF v1/v2/v3 container
- [`imatrix`](src/imatrix.rs) — llama.cpp activation-importance matrix

Every parser operates on `impl Read` / `impl Seek` / `&[u8]` / `Cow<'_, [u8]>` and returns
format-typed structs. **No item in this crate references `Tensor`, `Device`, `Storage`, or any
other backend-frontend type.** The only borrowed types are foundational primitives from
`fuel-core-types`: `DType`, `Shape`, `GgmlDType`.

## Why a separate crate

A parser tied to a specific transport (file path, mmap) cannot easily serve other transports.
Splitting the parser surface from the transport adapter unlocks several concrete use cases:

- **Streaming weight load** — read safetensors / GGUF off an `impl Read` (HTTP body,
  S3 stream, decompressor) without ever materializing the full file on disk.
- **Inter-process tensor exchange** — two Fuel processes (or Fuel ↔ Lightbulb ↔ mlmf) can
  exchange tensors as safetensors-on-the-wire over a Unix socket / shared-mem region. The
  format is already a public, language-agnostic schema — using it as IPC means there is no
  Fuel-internal protocol to maintain, and any HF-ecosystem tool can read what Fuel emits.
- **KV-cache / activation hand-off** — between an inference and a draft model in
  speculative decoding, or between an embedding model and a retrieval model in a pipeline.
- **`RemoteHostStorage`** (Phase 7c) — the transport returns typed buffers; the parser
  surface sits naturally on top.
- **Hot weight reload** during serving — write current weights to a wire format, parse
  back into a new process. No special Fuel-native serialization.

## Where Tensor-construction lives

Phase 7.5's work item E has landed: the wrappers that turn parsed metadata into a `Tensor`
no longer live in `fuel-core`. `fuel_core::safetensors`/`fuel_core::quantized` were pure
compat shims to `fuel_loaders` (fuel-core dissolution, GAP-347 PR 5/6) and have since been
retired outright — current call sites are `fuel_loaders::safetensors::load` and
`fuel_loaders::quantized::gguf_file::Content::read` (the `ggml_file` module they once
discussed was deleted entirely; see `fuel-loaders/src/quantized/mod.rs`). `PthTensors::get`
took a different path: pickle's Tensor-construction layer was absorbed directly into
`fuel_formats::pickle`, with no separate `fuel-loaders` wrapper needed.

## Pattern for new transports

A new transport adapter (HTTP, S3, IPC, etc.) follows the same shape as `fuel-loaders`:

1. Acquire bytes from your transport — `Vec<u8>`, `Cursor<&[u8]>`, an `impl Read`.
2. Hand them to the appropriate `fuel-formats` parser:
   - `fuel_formats::safetensors::SafeTensors::deserialize(&bytes)`
   - `fuel_formats::gguf::Content::read(&mut cursor)`
   - `fuel_formats::pickle::read_pth_tensor_info(path, false, key)` (currently file-bound;
     a stream variant is straightforward to add)
   - `fuel_formats::imatrix::parse(&mut reader)`
3. Hand the parsed metadata + raw bytes to whoever builds `Tensor`s on a `Device`.

The smoke test in [`tests/transport_independence.rs`](tests/transport_independence.rs) shows
each parser running against an in-memory `Cursor<&[u8]>` without touching the filesystem.

## Status

This crate landed as part of ROADMAP Phase 7.5 work item A (`fuel-formats` extraction).
Subsequent work items (B/C/D/E) will reduce `fuel-core` further; A2 will move the
file-transport adapters into a dedicated `fuel-loaders` crate once `Tensor` lives in
`fuel-tensor`.
