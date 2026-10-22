// SPDX-License-Identifier: MIT OR Apache-2.0
//! Deriving a model's hyperparameters from a GGUF file's own metadata, generically over
//! [`Architecture`] — the same "same job, different container" role this crate already plays
//! for HF `config.json` (see [`crate::hf_config`]).
//!
//! # This is not neutral by accident of naming — it is neutral because it is parameterized
//!
//! GGUF metadata keys are architecture-prefixed: `llama.attention.head_count`,
//! `qwen2.attention.head_count`, `gemma.attention.head_count`, and so on. A function that
//! hardcodes `"llama."` and is merely NAMED generically would silently be a Llama-only reader
//! that happens to compile against and pass tests for any architecture — the exact failure
//! shape that would look done and be wrong for every other checkpoint. [`derive_config`]
//! takes the [`Architecture`] as an explicit input and builds every key from
//! `arch.as_str()`, never a literal prefix.
//!
//! # What has actually been verified, and what has not
//!
//! The key SUFFIXES this reads (`attention.head_count`, `attention.head_count_kv`,
//! `block_count`, `embedding_length`, `rope.dimension_count`, `attention.layer_norm_rms_epsilon`,
//! `context_length`, `rope.freq_base`, `feed_forward_length`) are llama.cpp's own
//! GGUF-metadata convention, used across most of its supported architectures, not something
//! specific to Llama. That said: **only Llama checkpoints have been read through this function
//! so far.** If another architecture's GGUF export renames or drops one of these keys, this
//! declines with a typed error naming the missing key and the architecture — it does not
//! silently produce a wrong number. Widen the verified set by exercising a real checkpoint per
//! architecture before trusting it un-reviewed for that architecture.
//!
//! # Design constraints (per ruling, not oversight)
//!
//! - **Nine fields, no more.** `bos_token_id`, `eos_token_id`, `rope_scaling`, and
//!   `tie_word_embeddings` are Llama-specific config concerns, not GGUF-metadata extraction —
//!   they belong in each model crate's own `from_gguf` conversion, not here. The moment this
//!   struct grows a field only one architecture uses, it has stopped being a neutral
//!   extraction and become a universal config, a much larger commitment.
//! - **`vocab_size` is a different KIND of field from the other eight**: it is not declared
//!   metadata, it is inferred from `token_embd.weight`'s tensor shape (llama.cpp does not
//!   write a `{arch}.vocab_size` metadata key). A missing or malformed `token_embd.weight`
//!   fails differently from a missing metadata key — see [`GgufConfigError::MissingTensor`]
//!   and [`GgufConfigError::UnexpectedTensorShape`], kept distinct from
//!   [`GgufConfigError::MissingMetadata`] for exactly this reason.
//! - **Decline, never default.** Every one of these nine fields has a plausible "helpful"
//!   fallback, and this project's own registry records what that costs elsewhere on this same
//!   config-parse path (a silently truncated GQA ratio, a `hidden_size / 0` panic). A missing
//!   key is a typed [`GgufConfigError`] naming the key, never a guess and never an `unwrap`.
//!   The one exception is `attention.head_count_kv`: llama.cpp itself reads an absent key as
//!   `attention.head_count`, so that default is the format's own convention, not a guess.
//!   The other is `rope.freq_base`: llama.cpp initialises it to 10000.0 and reads the key as
//!   optional (the GGUF spec lists it without a default, so the reference implementation is the
//!   authority).

use super::arch::Architecture;
use super::gguf_file::{Content, TensorInfo};
use fuel_ir::error::{Error, Result};

/// A model's core hyperparameters, extracted from a GGUF file's declared metadata plus one
/// tensor-shape inference (`vocab_size` — see the module doc). Deliberately holds only fields
/// that are genuinely architecture-agnostic in shape (every llama.cpp-family architecture
/// declares all nine under its own key prefix); anything architecture-specific belongs in the
/// consuming model crate's own config type instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GgufDerivedConfig {
    /// `{arch}.embedding_length` — the model's hidden dimension.
    pub hidden_size: usize,
    /// `{arch}.feed_forward_length` — the MLP's intermediate dimension.
    pub intermediate_size: usize,
    /// Inferred from `token_embd.weight`'s tensor shape, NOT a declared metadata key — see the
    /// module doc's note on why this field is a different kind from the other eight.
    pub vocab_size: usize,
    /// `{arch}.block_count` — number of transformer layers.
    pub n_layers: usize,
    /// `{arch}.attention.head_count`.
    pub n_heads: usize,
    /// `{arch}.attention.head_count_kv` — equal to `n_heads` for plain multi-head attention,
    /// smaller under GQA. An absent key defaults to `n_heads` (llama.cpp convention).
    pub n_kv_heads: usize,
    /// `{arch}.attention.layer_norm_rms_epsilon`.
    pub rms_norm_eps: f64,
    /// `{arch}.rope.freq_base` — the rotary position embedding frequency base. An absent key
    /// defaults to 10000.0 (llama.cpp convention).
    pub rope_theta: f64,
    /// `{arch}.context_length` — the model's trained maximum sequence length.
    pub max_position_embeddings: usize,
}

/// Why [`derive_config`] declined to produce a [`GgufDerivedConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufConfigError {
    /// `{arch}.{key}` is absent from the GGUF file's metadata.
    MissingMetadata {
        architecture: &'static str,
        key: String,
    },
    /// The metadata value for `{arch}.{key}` is present but not the expected value type
    /// (e.g. a string where a `u32` was declared).
    WrongMetadataType {
        architecture: &'static str,
        key: String,
        reason: String,
    },
    /// `token_embd.weight` — the tensor `vocab_size` is inferred from — is absent from the
    /// file's tensor table.
    MissingTensor { name: &'static str },
    /// `token_embd.weight` exists but its shape doesn't have the expected rank, or neither
    /// dimension matches the already-extracted `embedding_length`.
    UnexpectedTensorShape {
        name: &'static str,
        dims: Vec<usize>,
        expected_hidden: usize,
    },
}

impl std::fmt::Display for GgufConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufConfigError::MissingMetadata { architecture, key } => write!(
                f,
                "gguf: missing metadata key \"{architecture}.{key}\" -- required to derive a \
                 {architecture} config, not defaulted"
            ),
            GgufConfigError::WrongMetadataType {
                architecture,
                key,
                reason,
            } => write!(
                f,
                "gguf: metadata key \"{architecture}.{key}\" has the wrong type: {reason}"
            ),
            GgufConfigError::MissingTensor { name } => write!(
                f,
                "gguf: missing tensor \"{name}\" -- vocab_size is inferred from its shape"
            ),
            GgufConfigError::UnexpectedTensorShape {
                name,
                dims,
                expected_hidden,
            } => write!(
                f,
                "gguf: tensor \"{name}\" has shape {dims:?}, expected rank 2 with one dim equal \
                 to hidden_size={expected_hidden}"
            ),
        }
    }
}

impl std::error::Error for GgufConfigError {}

/// A metadata reader bound to one `content` and one architecture `prefix`, constructed ONCE
/// per [`derive_config`] call. The prefix is captured at construction, not passed at each call
/// site — this is the property that matters: with an explicit `prefix: &str` parameter on
/// every read, each of the nine call sites would be an independent chance to pass the wrong
/// architecture's prefix, producing a config that mixes two architectures' fields, is
/// internally inconsistent, and is still green on every test that only checks an error
/// MESSAGE names the right prefix rather than checking that every FIELD used it. Binding the
/// prefix once removes that class of mistake at the type level: there is no `prefix` argument
/// to get wrong at a call site because there is no call-site prefix argument at all.
struct GgufMeta<'a> {
    content: &'a Content,
    // 'static, not 'a: every real caller gets this from `Architecture::as_str()`, which
    // returns a literal, and `GgufConfigError::{MissingMetadata,WrongMetadataType}` need a
    // 'static `architecture` field to stay simple `Debug`/`Eq`-derivable error data rather
    // than an owned `String` for a value that is always one of a small fixed set of literals.
    prefix: &'static str,
}

impl<'a> GgufMeta<'a> {
    fn u32(&self, suffix: &str) -> Result<u32> {
        let key = format!("{}.{suffix}", self.prefix);
        let value = self.content.metadata.get(&key).ok_or_else(|| {
            Error::msg(GgufConfigError::MissingMetadata {
                architecture: self.prefix,
                key: suffix.to_string(),
            })
        })?;
        value.to_u32().map_err(|e| {
            Error::msg(GgufConfigError::WrongMetadataType {
                architecture: self.prefix,
                key: suffix.to_string(),
                reason: e.to_string(),
            })
        })
    }

    /// [`Self::u32`], except an ABSENT key yields `default`. A present key of the wrong type
    /// still returns [`GgufConfigError::WrongMetadataType`].
    fn u32_or(&self, suffix: &str, default: u32) -> Result<u32> {
        if self
            .content
            .metadata
            .contains_key(&format!("{}.{suffix}", self.prefix))
        {
            self.u32(suffix)
        } else {
            Ok(default)
        }
    }

    /// [`Self::f32`], except an ABSENT key yields `default`. A present key of the wrong type
    /// still returns [`GgufConfigError::WrongMetadataType`].
    fn f32_or(&self, suffix: &str, default: f32) -> Result<f32> {
        if self
            .content
            .metadata
            .contains_key(&format!("{}.{suffix}", self.prefix))
        {
            self.f32(suffix)
        } else {
            Ok(default)
        }
    }

    fn f32(&self, suffix: &str) -> Result<f32> {
        let key = format!("{}.{suffix}", self.prefix);
        let value = self.content.metadata.get(&key).ok_or_else(|| {
            Error::msg(GgufConfigError::MissingMetadata {
                architecture: self.prefix,
                key: suffix.to_string(),
            })
        })?;
        value.to_f32().map_err(|e| {
            Error::msg(GgufConfigError::WrongMetadataType {
                architecture: self.prefix,
                key: suffix.to_string(),
                reason: e.to_string(),
            })
        })
    }
}

/// llama.cpp's default `rope.freq_base` when the key is absent (`llama-model.cpp`:
/// `hparams.rope_freq_base_train = 10000.0f`, then an optional `get_key`).
const DEFAULT_ROPE_FREQ_BASE: f32 = 10000.0;

/// Extract [`GgufDerivedConfig`] from `content`'s metadata and tensor table, reading every key
/// under `arch.as_str()`'s prefix. Never defaults a missing field (except `head_count_kv`,
/// which follows llama.cpp's convention of falling back to `head_count`, and `rope.freq_base`,
/// which falls back to llama.cpp's 10000.0); every failure names the
/// exact key or tensor that was absent or malformed.
pub fn derive_config(content: &Content, arch: Architecture) -> Result<GgufDerivedConfig> {
    let m = GgufMeta {
        content,
        prefix: arch.as_str(),
    };

    let hidden_size = m.u32("embedding_length")? as usize;
    let intermediate_size = m.u32("feed_forward_length")? as usize;
    let n_layers = m.u32("block_count")? as usize;
    let n_heads = m.u32("attention.head_count")? as usize;
    // llama.cpp reads an ABSENT `head_count_kv` as `head_count` (plain multi-head attention);
    // a present-but-malformed value still declines.
    let n_kv_heads = m.u32_or("attention.head_count_kv", n_heads as u32)? as usize;
    let rms_norm_eps = m.f32("attention.layer_norm_rms_epsilon")? as f64;
    // llama.cpp initialises the base to 10000.0 and reads the key as optional; a
    // present-but-malformed value still declines.
    let rope_theta = m.f32_or("rope.freq_base", DEFAULT_ROPE_FREQ_BASE)? as f64;
    let max_position_embeddings = m.u32("context_length")? as usize;

    let vocab_size = infer_vocab_size(&content.tensor_infos, hidden_size)?;

    Ok(GgufDerivedConfig {
        hidden_size,
        intermediate_size,
        vocab_size,
        n_layers,
        n_heads,
        n_kv_heads,
        rms_norm_eps,
        rope_theta,
        max_position_embeddings,
    })
}

/// `token_embd.weight` is stored `[hidden, vocab]` or `[vocab, hidden]` depending on the
/// export tool's dim-order convention (see the wider dim-order finding logged against the
/// GGUF readers in this project) — pick whichever dimension does NOT match the already-known
/// `hidden_size` as `vocab_size`.
fn infer_vocab_size(
    tensor_infos: &std::collections::HashMap<String, TensorInfo>,
    hidden_size: usize,
) -> Result<usize> {
    const TENSOR: &str = "token_embd.weight";
    let info = tensor_infos
        .get(TENSOR)
        .ok_or_else(|| Error::msg(GgufConfigError::MissingTensor { name: TENSOR }))?;
    let dims = info.shape.dims();
    if dims.len() != 2 {
        return Err(Error::msg(GgufConfigError::UnexpectedTensorShape {
            name: TENSOR,
            dims: dims.to_vec(),
            expected_hidden: hidden_size,
        }));
    }
    if dims[0] == hidden_size {
        Ok(dims[1])
    } else if dims[1] == hidden_size {
        Ok(dims[0])
    } else {
        Err(Error::msg(GgufConfigError::UnexpectedTensorShape {
            name: TENSOR,
            dims: dims.to_vec(),
            expected_hidden: hidden_size,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::super::gguf_file::{Value, VersionedMagic};
    use super::*;

    fn content_with(metadata: std::collections::HashMap<String, Value>) -> Content {
        Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos: std::collections::HashMap::new(),
            unsupported_tensors: Default::default(),
            tensor_data_offset: 0,
        }
    }

    #[test]
    fn missing_key_names_the_exact_key_and_architecture() {
        let content = content_with(std::collections::HashMap::new());
        let err = derive_config(&content, Architecture::Llama).unwrap_err();
        assert!(err.to_string().contains("llama.embedding_length"));
    }

    #[test]
    fn different_architectures_read_different_prefixes() {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("llama.embedding_length".to_string(), Value::U32(2048));
        let content = content_with(metadata);

        // The same file, read as Qwen2, must NOT find Llama's key -- proves the prefix is
        // parameterized, not a hardcoded "llama." literal that happens to also work here.
        let err = derive_config(&content, Architecture::Qwen2).unwrap_err();
        assert!(err.to_string().contains("qwen2.embedding_length"));
        assert!(!err.to_string().contains("llama."));
    }

    /// The hazard `GgufMeta` binding the prefix once (rather than passing it at each call
    /// site) exists to close: a complete `qwen2.*` metadata set, PLUS a conflicting
    /// `llama.block_count` under the wrong prefix. If any field's read leaked the wrong
    /// prefix, `n_layers` would come back as the Llama value and this would still return
    /// `Ok` — every other test in this file would still pass, because none of them checks
    /// that ALL NINE fields came from the SAME prefix, only that error messages name the
    /// right one.
    #[test]
    fn every_field_comes_from_the_same_prefix_not_just_the_error_messages() {
        let mut metadata = std::collections::HashMap::new();
        for (k, v) in [
            ("embedding_length", Value::U32(64)),
            ("feed_forward_length", Value::U32(128)),
            ("block_count", Value::U32(7)), // the value this test checks for
            ("attention.head_count", Value::U32(4)),
            ("attention.head_count_kv", Value::U32(4)),
            ("context_length", Value::U32(2048)),
        ] {
            metadata.insert(format!("qwen2.{k}"), v);
        }
        metadata.insert(
            "qwen2.attention.layer_norm_rms_epsilon".to_string(),
            Value::F32(1e-5),
        );
        metadata.insert("qwen2.rope.freq_base".to_string(), Value::F32(10000.0));
        // Conflicting value under the WRONG prefix -- if this leaks in, n_layers reads 999.
        metadata.insert("llama.block_count".to_string(), Value::U32(999));

        let mut tensor_infos = std::collections::HashMap::new();
        tensor_infos.insert(
            "token_embd.weight".to_string(),
            TensorInfo {
                ggml_dtype: fuel_ir::quantized::GgmlDType::F32,
                shape: fuel_ir::Shape::from_dims(&[64, 100]),
                offset: 0,
            },
        );
        let content = Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos,
            unsupported_tensors: Default::default(),
            tensor_data_offset: 0,
        };

        let cfg = derive_config(&content, Architecture::Qwen2).unwrap();
        assert_eq!(
            cfg.n_layers, 7,
            "n_layers must come from qwen2.block_count (7), not the conflicting \
             llama.block_count (999) -- a leaked prefix would return 999 here"
        );
    }

    /// A complete llama metadata set (8 heads) with `attention.head_count_kv` set to `kv`
    /// (`None` = key absent), plus the `token_embd.weight` vocab inference needs.
    fn llama_content_with_kv(kv: Option<Value>) -> Content {
        let mut metadata = std::collections::HashMap::new();
        for (k, v) in [
            ("embedding_length", Value::U32(64)),
            ("feed_forward_length", Value::U32(128)),
            ("block_count", Value::U32(2)),
            ("attention.head_count", Value::U32(8)),
            ("context_length", Value::U32(2048)),
        ] {
            metadata.insert(format!("llama.{k}"), v);
        }
        if let Some(kv) = kv {
            metadata.insert("llama.attention.head_count_kv".to_string(), kv);
        }
        metadata.insert(
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            Value::F32(1e-5),
        );
        metadata.insert("llama.rope.freq_base".to_string(), Value::F32(10000.0));
        let mut content = content_with(metadata);
        content.tensor_infos.insert(
            "token_embd.weight".to_string(),
            TensorInfo {
                ggml_dtype: fuel_ir::quantized::GgmlDType::F32,
                shape: fuel_ir::Shape::from_dims(&[64, 100]),
                offset: 0,
            },
        );
        content
    }

    /// llama.cpp reads an absent `head_count_kv` as `head_count` (plain multi-head attention).
    #[test]
    fn absent_head_count_kv_defaults_to_head_count() {
        let cfg = derive_config(&llama_content_with_kv(None), Architecture::Llama).unwrap();
        assert_eq!(cfg.n_heads, 8);
        assert_eq!(cfg.n_kv_heads, 8);
    }

    /// Known-bad control for the default above: a PRESENT key that differs from `head_count`
    /// must keep its value, so the default cannot be applied unconditionally.
    #[test]
    fn present_head_count_kv_keeps_its_own_value() {
        let cfg = derive_config(
            &llama_content_with_kv(Some(Value::U32(2))),
            Architecture::Llama,
        )
        .unwrap();
        assert_eq!(cfg.n_heads, 8);
        assert_eq!(cfg.n_kv_heads, 2);
    }

    /// llama.cpp initialises `rope_freq_base_train` to 10000.0 and reads the key as optional
    /// (`llama-model.cpp`, `ml.get_key(LLM_KV_ROPE_FREQ_BASE, ..., false)`), so an ABSENT
    /// `rope.freq_base` means 10000.0.
    #[test]
    fn absent_rope_freq_base_defaults_to_10000() {
        let mut content = llama_content_with_kv(None);
        assert!(content.metadata.remove("llama.rope.freq_base").is_some());
        let cfg = derive_config(&content, Architecture::Llama).unwrap();
        assert_eq!(cfg.rope_theta, 10000.0);
    }

    /// Known-bad control for the default above: a PRESENT value must keep its own value.
    #[test]
    fn present_rope_freq_base_keeps_its_own_value() {
        let mut content = llama_content_with_kv(None);
        content
            .metadata
            .insert("llama.rope.freq_base".to_string(), Value::F32(500000.0));
        let cfg = derive_config(&content, Architecture::Llama).unwrap();
        assert_eq!(cfg.rope_theta, 500000.0);
    }

    /// The default covers an ABSENT key only; a present key of the wrong type still declines.
    #[test]
    fn malformed_rope_freq_base_still_declines() {
        let mut content = llama_content_with_kv(None);
        content.metadata.insert(
            "llama.rope.freq_base".to_string(),
            Value::String("10000".to_string()),
        );
        let err = derive_config(&content, Architecture::Llama).unwrap_err();
        assert!(err.to_string().contains("llama.rope.freq_base"));
    }

    /// The default covers an ABSENT key only; a present key of the wrong type still declines.
    #[test]
    fn malformed_head_count_kv_still_declines() {
        let err = derive_config(
            &llama_content_with_kv(Some(Value::String("4".to_string()))),
            Architecture::Llama,
        )
        .unwrap_err();
        assert!(err.to_string().contains("llama.attention.head_count_kv"));
    }

    #[test]
    fn vocab_size_is_inferred_not_read_from_metadata() {
        // Even with every metadata key present, a missing token_embd.weight must still
        // decline -- proving vocab_size's error path is independent of the metadata path.
        let mut metadata = std::collections::HashMap::new();
        for (k, v) in [
            ("embedding_length", Value::U32(64)),
            ("feed_forward_length", Value::U32(128)),
            ("block_count", Value::U32(2)),
            ("attention.head_count", Value::U32(4)),
            ("attention.head_count_kv", Value::U32(4)),
            ("context_length", Value::U32(2048)),
        ] {
            metadata.insert(format!("llama.{k}"), v);
        }
        metadata.insert(
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            Value::F32(1e-5),
        );
        metadata.insert("llama.rope.freq_base".to_string(), Value::F32(10000.0));
        let content = content_with(metadata);

        let err = derive_config(&content, Architecture::Llama).unwrap_err();
        assert!(err.to_string().contains("token_embd.weight"));
    }
}
