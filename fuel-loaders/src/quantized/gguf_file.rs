// SPDX-License-Identifier: MIT OR Apache-2.0
//! GGUF file format — fuel's metadata view.
//!
//! The wire-format TYPES (`VersionedMagic`, `ValueType`, `Value`, `TensorInfo`) and the writer live
//! in [`fuel_formats::gguf`]; this file re-exports them and wraps the parsed header in a
//! [`Content`]. READING a GGUF is `mlmf-gguf`'s job (MLMF move, slice D, 2026-10-24: CireSnave,
//! "Move any loading/saving portions to MLMF"): [`Content::open`] maps a file and
//! [`Content::from_bytes`] parses bytes the caller already holds (what `MmapedContent` does). The
//! byte-stream parser `Content::read` that used to sit beside them is deleted, and with it
//! GAP-205 (a panic on `general.alignment = 0`, unbounded allocations, lossy names, last-wins
//! duplicate keys).
//!
//! GGUF **v1** (32-bit counts, written by llama.cpp before mid-2023) is not readable: `mlmf-gguf`
//! refuses it on purpose and fuel keeps no second parser, so it is a typed error that says to
//! re-export the model.
//!
//! B6 removed the eager half: `Content::tensor`, `Content::tensor_from_mmap`,
//! and the free `write` function all built or consumed the eager `QTensor`.
//! Reading tensor *bytes* needs none of that — take the `TensorInfo` from
//! [`Content::tensor_infos`] plus [`Content::tensor_data_offset`] and slice the
//! backing mmap yourself. That is what the lazy loaders already do.
//!
//! MLMF's parsers are byte-SLICE-based, and `parse_tensors` range-checks every tensor's declared
//! byte range against `bytes.len()`, so the slice must be the real file's full length: the mmap
//! of [`mlmf_source_file::FileSource`] (or `MmapedContent`'s own) gives that for free, and the OS
//! pages in only what the parsers touch (header, KV metadata, tensor-info table), never the
//! multi-GB tensor-data region.

use std::collections::HashMap;
use std::path::Path;

use fuel_ir::error::{Error, Result};

pub use fuel_formats::gguf::{
    DEFAULT_ALIGNMENT, TensorInfo, Value, ValueType, VersionedMagic, write_string,
};

/// GGUF v1 (32-bit counts, early-2023 llama.cpp) is refused by `mlmf-gguf` on purpose, and fuel
/// keeps no second parser (MLMF slice D, PM ruling 2026-10-24). Say so, and say what to do: the
/// stock "version 1 is not supported" message does not.
fn refuse_v1(bytes: &[u8], path: &Path) -> Result<()> {
    let is_gguf = bytes.get(..4) == Some(b"GGUF".as_slice());
    let version = bytes
        .get(4..8)
        .and_then(|b| <[u8; 4]>::try_from(b).ok())
        .map(u32::from_le_bytes);
    if is_gguf && version == Some(1) {
        return Err(Error::Msg(
            "gguf: GGUF v1 files (32-bit counts, written by llama.cpp before mid-2023) are not \
             supported; re-export the model with a current llama.cpp"
                .to_string(),
        )
        .with_path(path));
    }
    Ok(())
}

/// Parsed GGUF header: metadata, the tensor-info table, and the byte
/// offset at which tensor data begins.
///
/// The parsed header: what the deleted `fuel_formats::gguf::Content` carried, field for field.
#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    /// Tensors whose ggml type code fuel cannot represent, by name with the
    /// raw code (GAP-211). The file opens,
    /// the tensor is absent from `tensor_infos`, and loading it is a typed
    /// error from [`Content::missing_tensor_error`].
    pub unsupported_tensors: HashMap<String, u32>,
    pub tensor_data_offset: u64,
}

impl Content {
    /// The error for a tensor lookup that missed [`Self::tensor_infos`]; names
    /// the ggml type code when the file declares the tensor but its type is
    /// unsupported, plain "missing tensor" otherwise.
    pub fn missing_tensor_error(&self, name: &str) -> Error {
        match self.unsupported_tensors.get(name) {
            Some(code) => Error::Msg(format!(
                "gguf: tensor {name:?} is stored as ggml type code {code}, which this build                  cannot load (no GgmlDType variant; GAP-211)"
            )),
            None => Error::Msg(format!("gguf: missing tensor {name:?}")),
        }
    }

    /// Parse `path` via MLMF's mmap-backed GGUF reader
    /// (`mlmf_source_file::FileSource` + `mlmf_gguf`'s staged parsers).
    ///
    /// Behavior-equal to the deleted `Content::read` on a well-formed file (recorded in
    /// `tests/fixtures/gguf_corpus_digests.json`, checked by `tests/gguf_corpus_equivalence.rs`
    /// on the 30 readable files of the local corpus), with three differences this crate's two
    /// typed-Err disciplines force, not optional choices:
    ///
    /// - A metadata key this build's index could not fully walk
    ///   (`!meta.index_complete()`) is a typed [`Error`], never a partial
    ///   [`Content::metadata`] map.
    /// - A tensor whose ggml type code MLMF itself cannot resolve at all is
    ///   *omitted* by MLMF's own container (that omission is seam-level,
    ///   not a defect) — this constructor turns that omission back into a
    ///   typed `Err` naming the tensor, matching `Content::read`'s
    ///   "the whole read fails" discipline rather than silently returning
    ///   fewer tensors than the file declares.
    /// - A tensor whose ggml type code MLMF resolves but [`fuel_ir::GgmlDType`]
    ///   (fuel's narrower set) does not — `F64`/`I8`/`I16`/`I32`/`I64` and
    ///   every `IQ*`/`TQ*`/`MXFP4`/`NVFP4`/`Q1_0`/`Q2_0` code — is a typed
    ///   `Err` from the same `GgmlDType::from_u32` call `Content::read`
    ///   already uses, reached uniformly for every tensor via
    ///   `GgmlType::code()`'s raw ggml wire code (verified byte-identical
    ///   to `GgmlDType::to_u32`'s numbering) rather than branching on
    ///   MLMF's `Encoding::Dense`/`Blocked` split.
    ///
    /// # GAP (non-UTF-8 metadata strings)
    ///
    /// MLMF's [`MetaValue::Bytes`] (a declared string whose bytes are not
    /// valid UTF-8, preserved verbatim per GGUF spec §9 clause 2.1) has no
    /// [`Value`] equivalent — `Value::String` is the only string variant,
    /// and `fuel_formats::gguf::read_string` already does a **lossy**
    /// `String::from_utf8_lossy` conversion for the *same* case via
    /// `Content::read`. This constructor replicates that exact lossy
    /// behavior (an explicit match arm, not a silent default) so the two
    /// constructors read a malformed-but-spec-legal file identically; it
    /// does not fix the underlying lossiness. Filed as a GAP for the
    /// breaking wave that also fixes `read_string` and adds a real
    /// `Value::Bytes` variant (adding a variant to the non-`#[non_exhaustive]`
    /// `Value` enum breaks exhaustive matches elsewhere — a second-number
    /// change, not a patch, so it is deliberately not bundled here).
    /// # Known limitations (adversarial review, triaged, not all fixed here)
    ///
    /// - **Quadratic metadata walk (finding B) -- FIXED, 2026-10-03.**
    ///   Originally O(N) separate `get(key)` calls, each re-walking MLMF's
    ///   own index, for a file with N metadata keys. `decode_metadata` now
    ///   uses `MetadataSource::entries()` (mlmf-core/mlmf-gguf 0.5.10+), a
    ///   genuine single pass -- see that function's own doc for the exact
    ///   verification. Left here so a reader following the finding list
    ///   sees it closed rather than silently vanished.
    /// - **Array decode doubles peak momentarily (finding 6b).** A huge
    ///   `MetaValue::Array` already costs the same to decode via
    ///   `Content::read`; `meta_value_to_fuel_value`'s `.collect()` on the
    ///   `?`-propagating iterator holds the source items and the fuel
    ///   `Vec<Value>` live at once, a transient ~2x on that one key's array,
    ///   same shape as `Content::read`'s existing risk, not a new one this
    ///   constructor introduces.
    /// - **Concurrent truncation of the mapped file (finding 6c).** mmap's
    ///   inherent risk: if another process truncates `path` after `open`
    ///   maps it, a later read through the mapping can raise `SIGBUS`
    ///   (POSIX) or an access violation (Windows), inherent to any
    ///   mmap-backed reader, not specific to this adapter.
    ///   `mlmf_source_file::FileSource::open_read` (copy-based, TOCTOU-immune)
    ///   exists as an untaken mitigation if this ever needs hardening.
    /// - **Metadata parity, not correctness, differences vs. `Content::read`
    ///   -- MUST be resolved or explicitly accepted before any consumer is
    ///   repointed from `read` to `open` (tracked in `docs/gaps.md`; this is
    ///   a precondition of that future sweep, not a closed known-limitation):**
    ///   duplicate keys keep MLMF's (first-wins) resolution rather than
    ///   `read`'s (last-wins) when a malformed file declares the same key
    ///   twice; `Value::String` from `open` does not strip a trailing NUL
    ///   the way `fuel_formats::gguf::read_string` does for `read` (`"abc\0"`
    ///   stays `"abc\0"` via `open`, becomes `"abc"` via `read`); a GGUF
    ///   `Bool` byte value in `2..=255` is accepted by MLMF (any nonzero
    ///   byte is `true`) where `read`'s own decoder may reject it; and a
    ///   non-UTF-8 metadata *key* or tensor *name* (as opposed to a
    ///   *string value*, which is the `MetaValue::Bytes` case this
    ///   constructor already handles) is rejected by MLMF where `read`
    ///   tolerates it via lossy decoding. All four are pre-existing,
    ///   malformed-input-only edge cases, not observed on any real model
    ///   file, and none is fixed here -- but each one is a BEHAVIOR CHANGE a
    ///   consumer repoint would introduce silently if not revisited first.
    pub fn open(path: &Path) -> Result<Self> {
        use mlmf_core::ByteSource as _;

        let source = mlmf_source_file::FileSource::open(path)
            .map_err(|e| Error::Msg(format!("gguf: {e}")).with_path(path))?;
        Self::from_bytes(source.as_bytes(), path)
    }

    /// Parse the whole GGUF file `bytes` with MLMF's staged parsers: what [`Content::open`] does
    /// after mapping `path`, for a caller that already holds the mapping (`MmapedContent`).
    /// `path` is used only to name the file in errors. `bytes` must be the real file's full
    /// length: `parse_tensors` range-checks every tensor against it.
    pub fn from_bytes(bytes: &[u8], path: &Path) -> Result<Self> {
        refuse_v1(bytes, path)?;
        let origin = path.display().to_string();
        let gguf_err = |e: mlmf_gguf::GgufError| Error::Msg(format!("gguf: {e}")).with_path(path);

        let mut cursor = mlmf_gguf::cursor::Cursor::new(bytes);
        let header = mlmf_gguf::parse_header(&mut cursor).map_err(gguf_err)?;
        let (meta, _meta_report) =
            mlmf_gguf::GgufMetadata::parse(bytes, &origin).map_err(gguf_err)?;
        let (tensors, tensors_report) =
            mlmf_gguf::parse_tensors(bytes, &meta, &origin).map_err(gguf_err)?;
        Self::from_parsed(&header, &meta, &tensors, &tensors_report, path)
    }

    /// Turn MLMF's parsed header, metadata and tensor directory into fuel's [`Content`],
    /// applying fuel's two typed-error disciplines (see [`Content::open`]).
    fn from_parsed(
        header: &mlmf_gguf::Header,
        meta: &mlmf_gguf::GgufMetadata<'_>,
        tensors: &mlmf_gguf::GgufTensors<'_>,
        tensors_report: &mlmf_core::Report,
        path: &Path,
    ) -> Result<Self> {
        let metadata = decode_metadata(meta, path)?;
        check_alignment_agrees(&metadata, meta.alignment(), path)?;
        let mut unsupported_tensors =
            check_all_tensors_resolved(tensors, tensors_report, header, path)?;
        let (tensor_infos, more_unsupported) = build_tensor_infos(tensors, path)?;
        unsupported_tensors.extend(more_unsupported);
        Ok(Self {
            magic: resolve_magic(header.version, path)?,
            metadata,
            tensor_infos,
            unsupported_tensors,
            tensor_data_offset: tensors.data_start(),
        })
    }
}

/// Decodes every metadata key up front into an eager `HashMap<String,
/// Value>`: fuel's `Content::metadata` is eager today (`Content::read`
/// decodes every key up front), so `open` preserves that shape rather than
/// MLMF's own lazy get()-by-key model.
///
/// Also enforces `index_complete()`: a metadata key this build's index
/// could not fully walk is a typed [`Error`], never a partial map.
///
/// Uses `MetadataSource::entries()` (mlmf-core/mlmf-gguf 0.5.10+), a true
/// single pass over MLMF's own stored entry list -- NOT the `keys()` +
/// per-key `get()` walk this previously used, which was O(N) separate
/// internal re-scans for N keys (adversarial review finding B, the
/// quadratic-metadata-walk deferral). Verified by reading
/// `GgufMetadata::entries()`'s override directly (mlmf-gguf 0.5.10
/// metadata.rs:510): it iterates its own `entries` field once, not
/// `self.keys().iter().map(|k| self.get(k))`. Closes that deferral.
///
/// `entries()` OMITS an undecodable entry rather than erroring (matching
/// `get()`'s own contract) -- not reachable today because
/// `index_complete() == false` already refuses such a file above, but the
/// length check below is a typed Err, not a `.expect`, so a future MLMF
/// release that weakens that invariant fails loudly here instead of
/// silently returning fewer metadata keys than the file declares
/// (adversarial review finding #7's same discipline, applied to this walk).
fn decode_metadata(
    meta: &mlmf_gguf::GgufMetadata<'_>,
    path: &Path,
) -> Result<HashMap<String, Value>> {
    use mlmf_core::MetadataSource as _;

    if !meta.index_complete() {
        return Err(Error::Msg(
            "gguf: metadata index incomplete -- an unrecognized value \
             type blocked the walk before the end of the key-value block"
                .to_string(),
        )
        .with_path(path));
    }

    let entries = meta.entries();
    let key_count = meta.keys().len();
    if entries.len() != key_count {
        return Err(Error::Msg(format!(
            "gguf: metadata entries() returned {} pairs but keys() lists {key_count} keys -- \
             an index/value inconsistency in the underlying parser",
            entries.len()
        ))
        .with_path(path));
    }

    let mut metadata = HashMap::new();
    for (key, mv) in entries {
        metadata.insert(
            key.to_string(),
            meta_value_to_fuel_value(mv).map_err(|e| {
                Error::Msg(format!("gguf: metadata key {key:?}: {e}")).with_path(path)
            })?,
        );
    }
    Ok(metadata)
}

/// Finding A (adversarial review): `fuel_formats::gguf::Content::read`
/// resolves `general.alignment` from SIX value-type arms (U8/U16/U32/
/// I8/I16/I32, non-negative); MLMF's `GgufMetadata::alignment()` accepts
/// ONLY U32 and silently falls back to its own default (32) for every
/// other type or an invalid U32 -- a file declaring the alignment as, say,
/// `I32(64)` would make `read` and an unguarded `open` compute DIFFERENT
/// `tensor_data_offset`s with no error at all. Resolve it fuel's way from
/// the metadata already decoded, and REFUSE rather than silently trust
/// `tensors.data_start()` (which was already computed with MLMF's narrower
/// rule, passed in as `mlmf_alignment`) if the two disagree -- a typed Err
/// is always safe; a silently wrong byte offset into the tensor-data
/// region is not.
fn check_alignment_agrees(
    metadata: &HashMap<String, Value>,
    mlmf_alignment: u64,
    path: &Path,
) -> Result<()> {
    let fuel_alignment = match metadata.get("general.alignment") {
        Some(Value::U8(v)) => *v as u64,
        Some(Value::U16(v)) => *v as u64,
        Some(Value::U32(v)) => *v as u64,
        Some(Value::I8(v)) if *v >= 0 => *v as u64,
        Some(Value::I16(v)) if *v >= 0 => *v as u64,
        Some(Value::I32(v)) if *v >= 0 => *v as u64,
        _ => DEFAULT_ALIGNMENT,
    };
    if !fuel_alignment.is_power_of_two() {
        return Err(Error::Msg(format!(
            "gguf: general.alignment must be a non-zero power of two, got {fuel_alignment}"
        ))
        .with_path(path));
    }
    if fuel_alignment != mlmf_alignment {
        return Err(Error::Msg(format!(
            "gguf: general.alignment resolves to {fuel_alignment} under fuel's rules \
             (U8/U16/U32/I8/I16/I32) but to {mlmf_alignment} under MLMF's (U32 only) -- \
             refusing rather than risk a silently wrong tensor_data_offset"
        ))
        .with_path(path));
    }
    Ok(())
}

/// A tensor MLMF itself could not resolve at all is OMITTED from
/// `tensors.tensors()` (seam-level, not a defect in MLMF) -- turn that back
/// into a typed Err so `open` fails the whole read the same way
/// `Content::read` does, rather than silently returning fewer tensors than
/// the file declares.
fn check_all_tensors_resolved(
    tensors: &mlmf_gguf::GgufTensors<'_>,
    tensors_report: &mlmf_core::Report,
    header: &mlmf_gguf::Header,
    path: &Path,
) -> Result<HashMap<String, u32>> {
    use mlmf_core::TensorContainer as _;

    let missing =
        header.tensor_count as usize - tensors.tensors().len().min(header.tensor_count as usize);
    if missing == 0 {
        return Ok(HashMap::new());
    }
    // GAP-211: a tensor whose type code MLMF cannot resolve is omitted by MLMF
    // and reported with its raw code. Fuel cannot represent that type either, so
    // it joins `Content::unsupported_tensors` (the file still opens; loading the
    // tensor fails by name). Anything MLMF dropped WITHOUT a code we can record
    // stays a typed error, never a silently smaller tensor table.
    let unresolved: HashMap<String, u32> = tensors_report
        .entries()
        .iter()
        .filter_map(|u| match &u.kind {
            mlmf_core::UnrecognizedKind::TensorEncoding {
                name,
                declared: mlmf_core::DeclaredType::Code(code),
                ..
            } => Some((name.clone(), *code)),
            _ => None,
        })
        .collect();
    if unresolved.len() == missing {
        return Ok(unresolved);
    }
    Err(Error::Msg(format!(
        "gguf: {missing} of {} declared tensors were dropped by MLMF and only {} carry a recordable type code: {unresolved:?}",
        header.tensor_count,
        unresolved.len(),
    ))
    .with_path(path))
}

/// Maps every MLMF `TensorDescriptor` to fuel's `TensorInfo` shape.
fn build_tensor_infos(
    tensors: &mlmf_gguf::GgufTensors<'_>,
    path: &Path,
) -> Result<(HashMap<String, TensorInfo>, HashMap<String, u32>)> {
    use mlmf_core::TensorContainer as _;

    let mut tensor_infos = HashMap::new();
    let mut unsupported = HashMap::new();
    for d in tensors.tensors() {
        let code = encoding_ggml_code(&d.encoding).ok_or_else(|| {
            Error::Msg(format!(
                "gguf: tensor {:?}: dense dtype has no ggml code mapping",
                d.name
            ))
            .with_path(path)
        })?;
        // GAP-211: a type fuel has no variant for is recorded, not fatal.
        let Ok(ggml_dtype) = fuel_ir::GgmlDType::from_u32(code) else {
            unsupported.insert(d.name.clone(), code);
            continue;
        };
        // GGUF's on-disk dimension order needs reversing to match fuel's
        // row-major convention -- the same reversal
        // `fuel_formats::gguf::Content::read` performs
        // (`dimensions.reverse()`); MLMF's `TensorDescriptor::shape`
        // preserves the file's declared (non-reversed) order.
        let mut dims: Vec<usize> = d.shape.dims().to_vec();
        dims.reverse();
        tensor_infos.insert(
            d.name.clone(),
            TensorInfo {
                shape: fuel_ir::Shape::from(dims),
                // Relative to tensor_data_offset, matching
                // `TensorInfo::offset`'s existing contract (`Content::read`
                // stores the on-disk relative offset, not an absolute file
                // position).
                offset: d.bytes.start - tensors.data_start(),
                ggml_dtype,
            },
        );
    }
    Ok((tensor_infos, unsupported))
}

/// mlmf-gguf's own SUPPORTED version set is exactly {2, 3} -- `parse_header`
/// already refuses anything else with `GgufError::UnsupportedVersion`, so
/// the wildcard arm here is unreachable in practice, not a silent default.
fn resolve_magic(version: u32, path: &Path) -> Result<VersionedMagic> {
    match version {
        2 => Ok(VersionedMagic::GgufV2),
        3 => Ok(VersionedMagic::GgufV3),
        v => Err(Error::Msg(format!("gguf: unexpected parsed version {v}")).with_path(path)),
    }
}

/// `MetaValue`'s 14 variants (13 GGUF value kinds + `Bytes`) to fuel's
/// `Value`'s 13 -- every GGUF-declarable kind maps directly except `Bytes`,
/// which has no fuel equivalent. See [`Content::open`]'s doc for why that
/// one arm is lossy on purpose and pinned by a test.
///
/// Returns `Err` for anything outside the 14 variants known today.
/// `MetaValue` is `#[non_exhaustive]` and this crate's own `mlmf-core`
/// dependency is a caret requirement (`"0.5.8"`), so a semver-compatible
/// 0.5.x release adding a variant would otherwise hit this function with
/// zero compile-time warning (adversarial review finding #5: an earlier
/// version of this match silently returned `Value::String(String::new())`
/// for that case -- a plausible real value, e.g. an empty `chat_template`,
/// indistinguishable from the corruption). A typed Err that names the
/// unhandled variant is the only choice that can't be confused with real
/// data.
fn meta_value_to_fuel_value(mv: &mlmf_core::MetaValue) -> std::result::Result<Value, String> {
    use mlmf_core::MetaValue as M;
    Ok(match mv {
        M::U8(v) => Value::U8(*v),
        M::I8(v) => Value::I8(*v),
        M::U16(v) => Value::U16(*v),
        M::I16(v) => Value::I16(*v),
        M::U32(v) => Value::U32(*v),
        M::I32(v) => Value::I32(*v),
        M::U64(v) => Value::U64(*v),
        M::I64(v) => Value::I64(*v),
        M::F32(v) => Value::F32(*v),
        M::F64(v) => Value::F64(*v),
        M::Bool(v) => Value::Bool(*v),
        M::String(s) => Value::String(s.clone()),
        M::Array(items) => Value::Array(
            items
                .iter()
                .map(meta_value_to_fuel_value)
                .collect::<std::result::Result<Vec<_>, _>>()?,
        ),
        // GAP-344: non-UTF-8 string bytes, preserved verbatim by MLMF (GGUF
        // spec Β§9 clause 2.1). fuel has no Value::Bytes; replicate
        // Content::read's own lossy fuel_formats::gguf::read_string
        // behavior exactly, rather than defaulting silently, so both
        // constructors agree on a malformed-but-spec-legal file. Pinned by
        // `gguf_open_non_utf8_string_is_lossy_like_read` below; fixing the
        // lossiness (adding Value::Bytes) is its own breaking-wave PR.
        M::Bytes(raw) => Value::String(String::from_utf8_lossy(raw).into_owned()),
        // MetaValue is #[non_exhaustive] -- every variant that exists in
        // mlmf-core 0.5.8 is matched above; this arm is reachable only by a
        // FUTURE mlmf release adding one, and must stay an Err, never a
        // default (see this function's own doc).
        other => return Err(format!("unrecognized MetaValue variant: {other:?}")),
    })
}

/// The raw ggml wire-format type code for a resolved `TensorDescriptor`'s
/// encoding -- uniform across MLMF's `Dense`/`Blocked` split, since fuel's
/// own `GgmlDType::from_u32` is keyed on that same wire code space
/// regardless of which MLMF branch produced it.
///
/// `None` only for a `Dense(dt)` whose `dt` is outside the 8 dtypes
/// `GgmlType::encoding()` can ever produce for the Dense case
/// (F32/F16/BF16/F64/I8/I16/I32/I64) -- unreachable in practice, named
/// rather than panicked on, per this crate's never-panic-on-production-paths
/// rule.
fn encoding_ggml_code(encoding: &mlmf_core::Encoding) -> Option<u32> {
    use mlmf_core::{DType, Encoding};
    match encoding {
        Encoding::Blocked(spec) => Some(spec.code),
        // Dense's `code` isn't carried on the descriptor (only the decoded
        // DType is) -- map back to the ggml wire code by hand. Verified
        // against mlmf-ggml's own row() table: F32=0, F16=1, BF16=30,
        // I8=24, I16=25, I32=26, I64=27, F64=28 -- the only 8 DTypes
        // GgmlType::encoding() ever returns as Dense.
        Encoding::Dense(DType::F32) => Some(0),
        Encoding::Dense(DType::F16) => Some(1),
        Encoding::Dense(DType::BF16) => Some(30),
        Encoding::Dense(DType::I8) => Some(24),
        Encoding::Dense(DType::I16) => Some(25),
        Encoding::Dense(DType::I32) => Some(26),
        Encoding::Dense(DType::I64) => Some(27),
        Encoding::Dense(DType::F64) => Some(28),
        Encoding::Dense(_) => None,
    }
}
