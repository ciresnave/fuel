# Moving model loading and saving to MLMF — step 1 survey

**Status:** survey only, no code. Taken at `origin/main@5aaad145` (0.24.0), 2026-10-24.
**Ruling:** CireSnave, 2026-10-24, relayed by the portfolio PM, verbatim: *"Agreed on decision 265. Move any loading/saving portions to MLMF."* Earlier, on the `fuel-core` dissolution (quoted in `fuel-loaders/src/quantized/gguf_file.rs`, lines 17-18): *"fuel should use MLMF for loading and saving"*.
**Rule for this work (PM):** a slice's equivalence test (same bytes from the old and the new path on the real `Qwen3-0.6B-AWQ` checkpoint) passes BEFORE fuel's copy is deleted. MLMF crates come from crates.io only.

## 1. The line I drew

- **FORMAT/IO (moves):** turning bytes on disk (or in a buffer) into named, typed byte ranges and typed metadata, and the reverse; the layout of a quantization *as a file format* (how AWQ/GPTQ/GGUF pack bits); mmap/IO of the file; sidecar JSON that only describes the file (shard index, quantization config).
- **STAYS in fuel:** anything that names a `Tensor`, `WeightStorage`, `Device`, a graph node or an `OpKind`; which tensor names a given *architecture* uses and what shapes its config implies (MLMF's own `mlmf-hf-layout` states that per-family `config.json` meaning "is architecture knowledge ... with Fuel", board #111); the layouts that exist for *our kernels* rather than for the file format (baracuda's padded AWQ layout).

Where one function does both, the survey says so and names the seam.

## 2. What MLMF has today (crates.io, `0.5.11`, read from the published crate sources)

| crate | what it provides | note |
|---|---|---|
| `mlmf-safetensors` | `parse_header`, `parse_tensors` (directory with rebased offsets), `parse_metadata`, `dtype_of`; "no I/O, no interpretation" | header parsing only; no mmap |
| `mlmf-source-file` | finding and reading the file (the I/O half) | fuel-loaders already depends on it |
| `mlmf-hf-layout` | `ShardIndex` (`model.safetensors.index.json`), `GenerationConfig`, `SpecialTokensMap`, `ConfigJson` (keys exactly as declared) | architecture meaning is explicitly Fuel's |
| `mlmf-awq` | `AwqConfig::{parse_standalone, parse_from_model_config}`, `locate_layers` -> `PackedLinearLayer` | geometry and config only; **no repack** |
| `mlmf-gguf`, `mlmf-ggml`, `mlmf-core` | GGUF header/metadata parse, ggml block specs, the `TensorContainer`/`MetadataSource` seam | `fuel-loaders` pins `0.5.10` (`fuel-loaders/Cargo.toml` lines 23-25): behind the latest, a dependency-rule item |
| `mlmf-gptq`, `mlmf-exl2` | GPTQ/EXL2 geometry | not used by fuel |

Not found in the published list: a pickle (`.pth`) reader, an imatrix reader, a tokenizer-from-GGUF crate, and a writer for safetensors/GGUF. (There is a `GGUF_EXPORT_SUMMARY.md` in the `mlmf` checkout, so a GGUF writer may exist unpublished; the mlmf lane should say.)

## 3. Inventory (file:line at `origin/main@5aaad145`)

### 3a. FORMAT/IO — candidates to move

| # | Where | What it does | Lines | MLMF counterpart | Verdict |
|---|---|---|---|---|---|
| F1 | `fuel-loaders/src/safetensors.rs:52` `MmapedSafetensors` (`:123 tensors`, `:135 get`), `:167` `BufferedSafetensors`; `fuel-formats/src/safetensors.rs:52` `MmapedFile` | mmap a `.safetensors` file / buffer and hand out `TensorView`s via the upstream `safetensors` crate | 247 + 100 | `mlmf-safetensors` (header/directory) + `mlmf-source-file` (mmap) | MOVE. **Blast radius: 226 files reference `MmapedSafetensors`.** Keep the type name and its `get`/`tensors` API as a thin facade so the model loaders do not change. |
| F2 | `fuel-formats/src/gguf.rs` (`:44 VersionedMagic`, `:68 TensorInfo`, `:74 read_string`, `:108 write_string`, `:116 ValueType`, `:187 Value`), `fuel-loaders/src/quantized/gguf_file.rs:52 Content` (`:80 read`, `:173 open` is already MLMF-backed), `gguf_mmap.rs:33 MmapedContent` | GGUF container parse + mmap; the file's own header says `Content::read` is the pre-MLMF path kept additively | 750 + 496 + 126 | `mlmf-gguf` (+ `mlmf-core`) | MOVE, partly done. `fuel-formats/src/gguf.rs:91-99` and `:423-426` record GAP-205 as "DEFERRED to `mlmf-gguf`" (a `HashMap` drops duplicate keys; no `Bytes` value). |
| F3 | `fuel-formats/src/imatrix.rs:33 parse`, `:89 parse_bytes`, `:95 load_path`; `fuel-loaders/src/quantized/imatrix_file.rs:20 load_imatrix` | llama.cpp importance-matrix parse | 136 + 22 | none published | MOVE if MLMF wants it; else ask. |
| F4 | `fuel-formats/src/pickle.rs:31 OpCode`, `:121 Object` | minimal Python pickle parse to read PyTorch `.pth` | 754 | none published | MOVE if MLMF wants it; else ask. |
| F5 | `fuel-loaders/src/quantized/tokenizer.rs:21 TokenizerFromGguf`, `:206 impl for Tokenizer` | reads vocab, merges, special ids out of GGUF metadata and builds a `tokenizers::Tokenizer` | 333 | none | SPLIT: extracting the vocab/merges/special-token data from GGUF metadata is FORMAT (moves as plain data); constructing the external `tokenizers::Tokenizer` stays. |
| F6 | `fuel-tensor/src/awq.rs:83 awq_from_hf_buffers`, `:166 load_awq_layer`, `:230 AwqHostWeight`, `:250 from_hf_buffers`, `:275 load` | find `{name}.qweight/.qzeros/.scales`, check shapes, hand the three buffers to the repack | 794 (mixed) | `mlmf-awq::locate_layers` | SPLIT. Locating the three tensors and the group size is FORMAT and moves; `AwqWeight`/`AwqHostWeight::matmul` build graph nodes and stay. |
| F7 | `fuel-quantized/src/awq_repack.rs:100 awq_zeros_w`, `:162 repack_hf_awq_for_kernel`, `:45 AwqKernelBuffers` | HF AWQ bit packing -> baracuda's padded OC-major kernel layout (host bytes in, host bytes out; no `Tensor`) | 381 | `mlmf-awq` has no repack | **DECISION NEEDED.** It is two things fused: (1) unpacking the HF AWQ packing (the format: the "packing axis is transposed" fact `mlmf-awq`'s own docs name), which belongs in MLMF; (2) re-packing into the layout baracuda's kernel wants (kernel knowledge, which is not a file format). Proposal: MLMF exposes the unpacked form (qweight/qzeros as plain integers + scales); the kernel-layout step stays in `fuel-quantized` (or goes to baracuda). Needs the PM/CireSnave to say where kernel layouts live. |
| F8 | `fuel-tensor/src/nf4.rs:484 load_nf4_layer` and the bitsandbytes NF4 helpers | NF4 block + double-quant tensors named in the checkpoint | 1093 (mixed) | none | SPLIT like F6: locating/decoding the stored blocks is FORMAT; the lazy tensors stay. Larger than AWQ; later slice. |
| F9 | `fuel-tensor/src/lazy.rs:9220 load_tensor_as_f32`, `:9271 load_transposed_matrix`, `:9308 load_transposed_matrix_preserve_dtype` | read a named tensor's bytes, convert bf16/f16 to f32 or keep dtype, transpose | ~120 | dtype decode: `mlmf-safetensors::dtype_of` / `mlmf-core` | SPLIT. Byte read + dtype decode is FORMAT; the transpose and the `WeightStorage` result stay. **These three functions are the choke point: 124 files and 1326 lines call `load_tensor_as_f32`**, so re-pointing F1 here changes the whole model zoo in one place. |
| F10 | `fuel-loaders/src/hf_config.rs` (`:71 head_dim`, `:136 num_key_value_heads`), per-model `*Config::from_hf_json_str` | read `config.json` and apply defaults | 274 | `mlmf-hf-layout::ConfigJson` returns the keys as declared | STAYS (architecture knowledge, per `mlmf-hf-layout`'s own ruling), but the JSON read itself can sit on `ConfigJson`. Low value. |
| F11 | `fuel-loaders/src/quantized/arch.rs:109 detect_from_gguf`, `config_from_gguf.rs:244 derive_config` | map GGUF metadata keys to an `Architecture` and hyperparameters | 331 + 523 | `mlmf-core` `MetadataSource` | STAYS: architecture knowledge. (`arch.rs:97` already says a fix was ported from `mlmf-gguf-arch`.) |
| F12 | `fuel-loaders/src/model_progress.rs:32 ProgressEvent`, `:64 ProgressReporter` | load-progress callbacks | 138 | MLMF's `ProgressFn` (cherry-picked; header lines 21-22) | MOVE only if MLMF publishes it; otherwise a duplicate by design. Low value. |

### 3b. SAVING — what exists

| # | Where | What | Verdict |
|---|---|---|---|
| S1 | `fuel-nn/src/varmap.rs:101` `safetensors::tensor::serialize_to_file(views, None, path)` (module `varmap`, `fuel-nn/src/lib.rs:42`) | `VarMap::save` writes trained variables as safetensors | MOVE the byte writing to an MLMF safetensors writer once one exists; `VarMap` itself (a `Tensor` container) stays. |
| S2 | `fuel-tensor-tools/src/main.rs:450` `write_gguf` | the quantize/convert tool's GGUF writer | MOVE to an MLMF GGUF writer (see the `GGUF_EXPORT_SUMMARY.md` note above). |
| S3 | `fuel-formats/src/gguf.rs:108` `write_string` | GGUF string encoding for the writer | MOVE with S2. |

No other checkpoint-saving path was found (searched for `serialize_to_file`, `save_`, `export_`, `write_gguf`, `write_safetensors` over `*.rs` at the ref). Image saving in `fuel-examples` is not a model file.

### 3c. STAYS — lazy-type / graph-build / architecture knowledge

- The per-architecture weight loaders: **178 `load_*from_mmapped` functions across 130 files in `fuel-transformers/src/models`**, plus `fuel-model-llama`, `fuel-model-phi`, `fuel-nn` layers. They decide which tensor names an architecture has, validate shapes against the config, fold norm conventions (`Qwen3_5Weights` stores `1 + w`), tie the LM head (`fuel-transformers/src/models/lazy_llama_full.rs:592 tied_lm_head_from_embeddings`) and build `WeightStorage`. That is architecture knowledge (MLMF's `mlmf-hf-layout` says so itself). They keep working untouched as long as the F1/F9 facade keeps its API.
- `fuel-transformers/src/models/lazy_qwen3_5.rs` (the new Qwen3.5 loader): same category. Its tensor reading goes through F9.
- `WeightStorage` and everything in `fuel-tensor/src/lazy.rs` that is not F9; `fuel-onnx` (an ONNX *graph evaluator*; its protobuf parse is generated by `prost` from `fuel-onnx/src/onnx.proto3`, so there is no hand-written format parser to move).

## 4. Proposed slices (each gated by its equivalence test; fuel's copy is deleted only after it passes)

Reference checkpoint: `C:\Models\Qwen3-0.6B-AWQ` (`FUEL_QWEN3_AWQ_DIR`); for non-AWQ slices the same safetensors file is the corpus, and `Qwen3.5-0.8B` (`FUEL_QWEN35_DIR`, 320 language-model tensors) a second one.

1. **Slice A — AWQ discovery and config (F6, locating half).** Use `mlmf-awq::{AwqConfig, locate_layers}` for the group size and the layer set. *Equivalence:* for every projection of the checkpoint, the located `qweight/qzeros/scales` names and the `group_size` equal what `load_awq_from_mmapped` uses today (the oracle currently re-derives `group_size` by hand in `tests/qwen3_awq_oracle.rs`). Small, no blast radius.
2. **Slice B — safetensors reading (F1 + the byte-read half of F9).** Re-implement `MmapedSafetensors` over `mlmf-safetensors` + `mlmf-source-file`, same public API. *Equivalence:* on the AWQ checkpoint and the Qwen3.5 checkpoint, the old and new paths return identical tensor-name sets, identical dtype/shape per tensor and byte-identical data for every tensor (hash of each tensor's bytes, old vs new). Biggest blast radius (226 files) but behind an unchanged API.
3. **Slice C — AWQ unpack/repack split (F7).** Blocked on the DECISION in F7. *Equivalence:* `repack_hf_awq_for_kernel` output byte-identical before and after the unpack moves, for every projection of the checkpoint.
4. **Slice D — GGUF remainder (F2).** Finish moving `Content::read` and `gguf_mmap` to `mlmf-gguf`, and close GAP-205 in the process. *Equivalence:* metadata map + tensor directory equal on the GGUF corpus (`C:\Models\gguf-corpus`, `TinyLlama-1.1B-Chat-v1.0-GGUF`).
5. **Slice E — imatrix, pickle, tokenizer data, NF4 (F3, F4, F5, F8).** Each only when MLMF has the counterpart; the mlmf lane says which it will take.
6. **Slice F — saving (S1-S3).** Needs MLMF writers. *Equivalence:* write with the MLMF writer, read back with the old reader and the new reader, byte-identical tensors; for GGUF compare against the file `write_gguf` produces today.

Also, independent of the slices: raise `mlmf-core`/`mlmf-gguf`/`mlmf-source-file` in `fuel-loaders/Cargo.toml` from `0.5.10` to `0.5.11` (CireSnave's standing dependency rule), with its own check.

## 5. Open questions for the mlmf lane and the PM

1. Which of imatrix, pickle, tokenizer-data and NF4 does MLMF want, and are there unpublished writers (safetensors, GGUF)?
2. F7: where do kernel layouts live? (My proposal: MLMF owns "unpack HF AWQ"; the baracuda-specific padded layout stays outside MLMF.)
3. `MmapedSafetensors` returns the upstream `safetensors::tensor::TensorView`. If MLMF's reader returns a different view type, the facade must adapt it for the 226 callers or change them; which does MLMF prefer?
4. Version policy: each slice changes a public path in a published crate (`fuel-loaders`, `fuel-formats`), so the PM allocates the bump per slice.

## 6. What I did not do

I did not run any of the moved code, change any dependency or open an MLMF-side change. The MLMF column comes from the published `0.5.11` sources (module docs and public items), not from running them. The 226/124/178/130 counts are `git grep` counts at the ref over `*.rs` (files that reference the name); they are populations of references, not of call sites that would need editing.
