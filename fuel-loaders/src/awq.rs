// SPDX-License-Identifier: MIT OR Apache-2.0
//! Finding a HuggingFace `*-AWQ` checkpoint's quantization parameters and its packed linear
//! layers. MLMF move, slice A (`docs/design-notes/2026-10-24-mlmf-move-survey.md`): the parsing
//! is `mlmf-awq` (`AwqConfig`, `locate_layers`) over `mlmf-safetensors`' header and directory;
//! this module only adds fuel's policy on top, namely which AWQ variants the kernels can read.
//!
//! Like the rest of this crate it knows nothing about `Tensor`, `Storage` or `Device`.
//!
//! # Scope
//!
//! One safetensors file. A sharded checkpoint would have to merge per-shard reports (a layer's
//! three tensors can sit in different shards, which a per-shard scan reports as incomplete), and
//! nothing here does that yet.

use fuel_ir::error::{Error, Result};
use mlmf_awq::{AwqConfig, LocateReport, PackedLinearLayer, locate_layers};
use mlmf_safetensors::{parse_header, parse_tensors};

/// What fuel's AWQ path can read: 4-bit, zero-point, GEMM-version packing.
const BITS: u64 = 4;

/// The checkpoint's quantization group size and its located AWQ-packed linear layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwqLayout {
    /// `quantization_config.group_size`.
    pub group_size: usize,
    /// Every fully located, geometrically consistent packed layer, sorted by prefix.
    pub layers: Vec<PackedLinearLayer>,
}

fn refuse(text: String) -> Error {
    Error::Msg(format!("AWQ checkpoint: {text}")).bt()
}

/// `quantization_config.group_size` from a model's `config.json`, after refusing every
/// quantization the loaders here do not read: a different method, bit width, packing or
/// zero-point convention would load as plausible garbage.
pub fn awq_group_size(config_json: &[u8]) -> Result<usize> {
    let cfg = AwqConfig::parse_from_model_config(config_json)
        .map_err(|e| refuse(format!("config.json: {e}")))?
        .ok_or_else(|| refuse("config.json has no AWQ `quantization_config`".to_string()))?;
    if !cfg.malformed.is_empty() {
        return Err(refuse(format!(
            "unreadable quantization_config fields: {:?}",
            cfg.malformed
        )));
    }
    if cfg.quant_method.as_deref() != Some("awq") {
        return Err(refuse(format!(
            "quant_method is {:?}, not \"awq\" (a missing method is not accepted)",
            cfg.quant_method
        )));
    }
    if cfg.bits != Some(BITS) {
        return Err(refuse(format!(
            "only {BITS}-bit AWQ is read, got bits {:?}",
            cfg.bits
        )));
    }
    if !cfg
        .version
        .as_deref()
        .is_some_and(|v| v.eq_ignore_ascii_case("gemm"))
    {
        return Err(refuse(format!(
            "only the GEMM packing is read, got version {:?}",
            cfg.version
        )));
    }
    if cfg.zero_point != Some(true) {
        return Err(refuse(format!(
            "only zero-point AWQ is read, got zero_point {:?}",
            cfg.zero_point
        )));
    }
    let group_size = cfg
        .group_size
        .filter(|g| *g > 0)
        .ok_or_else(|| refuse("quantization_config.group_size is missing or zero".to_string()))?;
    usize::try_from(group_size).map_err(|_| refuse(format!("group_size {group_size} overflows")))
}

/// Locate every AWQ-packed linear layer in one safetensors file's bytes. A layer MLMF reports
/// as incomplete or geometrically inconsistent is an error here (a half-loaded model is worse
/// than a refused one), with every offender named.
pub fn locate_awq_layers(config_json: &[u8], safetensors: &[u8]) -> Result<AwqLayout> {
    let group_size = awq_group_size(config_json)?;
    let header =
        parse_header(safetensors).map_err(|e| refuse(format!("safetensors header: {e}")))?;
    let (tensors, _report) = parse_tensors(safetensors, &header, "awq checkpoint")
        .map_err(|e| refuse(format!("safetensors directory: {e}")))?;
    let report = locate_layers(&tensors, BITS, group_size as u64)
        .map_err(|e| refuse(format!("locating layers: {e}")))?;
    reject_unusable(&report)?;
    Ok(AwqLayout {
        group_size,
        layers: report.layers,
    })
}

fn reject_unusable(report: &LocateReport) -> Result<()> {
    if report.incomplete.is_empty() && report.malformed.is_empty() {
        return Ok(());
    }
    Err(refuse(format!(
        "{} layer(s) missing qzeros or scales: {:?}; {} layer(s) with inconsistent geometry: {:?}",
        report.incomplete.len(),
        report.incomplete,
        report.malformed.len(),
        report.malformed
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(extra: &str) -> Vec<u8> {
        format!(
            r#"{{"quantization_config": {{"quant_method": "awq", "bits": 4, "version": "gemm",
                "zero_point": true, "group_size": 128 {extra}}}}}"#
        )
        .into_bytes()
    }

    #[test]
    fn reads_the_group_size_of_a_gemm_zero_point_4_bit_config() {
        assert_eq!(awq_group_size(&config("")).unwrap(), 128);
        // The two conventions differ in casing; GEMM and gemm are the same packing.
        let upper = String::from_utf8(config(""))
            .unwrap()
            .replace("\"gemm\"", "\"GEMM\"");
        assert_eq!(awq_group_size(upper.as_bytes()).unwrap(), 128);
    }

    /// Each refusal is its own arm: a config that differs from the accepted one in exactly one
    /// way must be refused, so no arm is decoration.
    #[test]
    fn refuses_every_variant_the_kernels_do_not_read() {
        let ok = String::from_utf8(config("")).unwrap();
        for (label, bad) in [
            ("8-bit", ok.replace("\"bits\": 4", "\"bits\": 8")),
            ("gemv packing", ok.replace("\"gemm\"", "\"gemv\"")),
            (
                "no zero point",
                ok.replace("\"zero_point\": true", "\"zero_point\": false"),
            ),
            ("gptq", ok.replace("\"awq\"", "\"gptq\"")),
            ("no method", ok.replace("\"quant_method\": \"awq\",", "")),
            (
                "zero group",
                ok.replace("\"group_size\": 128", "\"group_size\": 0"),
            ),
            ("no group", ok.replace(", \"group_size\": 128", "")),
            (
                "group size as a string",
                ok.replace("\"group_size\": 128", "\"group_size\": \"128\""),
            ),
        ] {
            assert!(
                awq_group_size(bad.as_bytes()).is_err(),
                "{label} must be refused"
            );
        }
        assert!(awq_group_size(b"{}").is_err(), "no quantization_config");
        assert!(awq_group_size(b"not json").is_err());
    }
}
