//! Build-time validators (the `V-FKC-*` battery, FKC §10).
//!
//! These run at import time, `Result`-returning, never panicking, never a
//! silent fix-up (FKC G6, constitution: validate-at-build-time, never-panic).
//! They operate on the **parsed/structured** form ([`FkcFile`] / [`FkcKernel`])
//! — structural + coherence checks — NOT on a registered binding table. The
//! register slice (`register.rs`) calls [`validate_file`] after parse+lower so a
//! bad contract fails import; the CI lint (`tests`) calls it standalone over the
//! whole checked-in corpus.
//!
//! ## Rule → check → error map (FKC §10, rules 1–16)
//!
//! | rule | check | `FkcError` |
//! |------|-------|------------|
//! | 1  | `fkc_version <= FKC_VERSION_MAX` | `UnsupportedVersion` |
//! | 2  | required fields (kernel; EXACTLY ONE of op_kind/fused_op; blurb; entry_point; ≥1 input; ≥1 output/bundle; cost; precision; determinism). Describe-only (§3.10) EXEMPTS exactly-one-of-op_kind/fused_op AND the ≥1-input rule (a zero-operand documentation op is legitimate). | `MissingRequiredField` / `MissingBlurb` / `OpTargetAmbiguous` |
//! | 3  | dtype validity; sub-byte ⇒ fdx.quant; ggml_dtype a real `GgmlDType` | `BadScalarType` / `QuantIncoherent` |
//! | 4  | layout coherence (≥1 of contiguous/strided OK; broadcast/reverse ⇒ strided) | `LayoutIncoherent` |
//! | 5  | awkward-layout coherence PER OPERAND | `AwkwardStrategyIncoherent` |
//! | 6  | quant coherence + scale single-place + MX/AFFINE_BLOCK not-yet-registrable | `QuantIncoherent` / `ScaleDoubleDeclared` / `MxNotYetRegistrable` |
//! | 7  | op-param variant in the correct namespace | `BadOpParamsVariant` |
//! | 8  | cost expressions parse, in-scope | `CostExprParse` |
//! | 8a | cost provenance present + non-placeholder | `CostProvenanceMissing` / `PlaceholderCost` |
//! | 9  | precision coverage (UNAUDITED flagged; nondeterministic ⇒ audited+no-bitstable) | (note / `QuantIncoherent`-shaped) |
//! | 10 | duplicate KernelRef — at register time (`finalize`), not here |
//! | 11 | prose/structured blurb agreement — see §10.11 note below |
//! | 13 | bundle slot rank ≤ 6 | `BundleSlotRankExceeded` |
//! | 14 | gather (paged) coherence | `GatherIncoherent` / `GatherNotYetSupported` |
//! | 15 | affine / symbolic extent coherence | `UnknownAdmissibilityEnum` |
//! | 16 | FDX-subset drift-guard (every dtype/quant/granularity/ggml token ∈ FDX table) | `FdxTokenNotInTable` |
//!
//! ### §10.11 (prose blurb) — not feasible on the structured form
//!
//! The prose blurb is the first non-empty line of the markdown section, which
//! the parser drops (it keeps only the structured [`FkcKernel`], FKC §3.1). So
//! the structured-vs-prose comparison is not available to a validator that
//! operates on the parsed form. We implement the *structured* half of the rule
//! (the `blurb:` field must be a non-empty one-line string → `MissingBlurb`) and
//! leave the prose-vs-structured re-render diff to the `fkc fmt --check` lint
//! that has the raw markdown ([`FkcError::BlurbMismatch`] stays reserved for it).

use crate::fkc::error::FkcError;
use crate::fkc::lower;
use crate::fkc::schema::{FkcFile, FkcKernel, QuantSpec, TensorDesc};

/// The maximum `fkc_version` this importer understands (FKC §10.1 / §11).
pub const FKC_VERSION_MAX: u32 = 1;

// ===========================================================================
// FDX normative token tables (§10.16 drift-guard)
// ===========================================================================
//
// FKC re-numbers nothing; FDX owns the codes (§0). The dtype token resolves
// through `lower::lower_dtype` (the same `DType` set FDX §6.1 owns); the
// quant `family` / `granularity` / `ggml_dtype` tokens resolve through the
// tables below, which mirror the FDX §6.2 symbol sets. A token absent from
// these is `FdxTokenNotInTable` (rule 16) — so FKC's set stays a subset of
// FDX's and the two specs cannot drift.

/// FDX `FDXQuant.family` symbols (FDX §6.2): `NONE | GGML_BLOCK | MX |
/// AFFINE_INT | AFFINE_FLOAT | AFFINE_BLOCK`. `"none"` is accepted in either
/// case (the corpus writes `none`).
fn is_fdx_quant_family(tok: &str) -> bool {
    matches!(
        tok,
        "none" | "NONE" | "GGML_BLOCK" | "MX" | "AFFINE_INT" | "AFFINE_FLOAT" | "AFFINE_BLOCK"
    )
}

/// FDX `FDXScaleGranularity` symbols (FDX §6.2): `PerTensor | PerToken |
/// PerChannel | PerBlock`.
fn is_fdx_granularity(tok: &str) -> bool {
    matches!(tok, "PerTensor" | "PerToken" | "PerChannel" | "PerBlock")
}

/// The as-built `GgmlDType` variant set (`fuel-core-types/src/quantized.rs`),
/// matched **by code** per FKC §3.4 — i.e. these are exactly the variant
/// names, and `Q4_K_M` (a GGUF file-format name, NOT a `GgmlDType` variant) is
/// deliberately absent (writing it fails §10.6 / rule 16).
fn ggml_dtype_code(tok: &str) -> Option<u32> {
    let code = match tok {
        "F32" => 0,
        "F16" => 1,
        "Q4_0" => 2,
        "Q4_1" => 3,
        "Q5_0" => 6,
        "Q5_1" => 7,
        "Q8_0" => 8,
        "Q8_1" => 9,
        "Q2K" => 10,
        "Q3K" => 11,
        "Q4K" => 12,
        "Q5K" => 13,
        "Q6K" => 14,
        "Q8K" => 15,
        "BF16" => 30,
        _ => return None,
    };
    Some(code)
}

/// The `ScaleGranularity` set that has an as-built `fuel-core-types`
/// counterpart (`quant_scale.rs`): `{ PerTensor, PerToken, PerChannel }`.
/// `PerBlock` is FDX/FKC-only with no target type yet (FKC §6 / §10.6).
fn is_registrable_granularity(tok: &str) -> bool {
    matches!(tok, "PerTensor" | "PerToken" | "PerChannel")
}

// ===========================================================================
// Public entry points
// ===========================================================================

/// Run the full `V-FKC-*` battery over a parsed file. The caller (the lint /
/// `import_bundle_str`) runs this after parse so a structurally-bad contract
/// fails import. Stops at the first violation (a typed `FkcError`).
pub fn validate_file(file: &FkcFile) -> Result<(), FkcError> {
    // Rule 1: format version supported.
    if file.front_matter.fkc_version > FKC_VERSION_MAX {
        return Err(FkcError::UnsupportedVersion {
            found: file.front_matter.fkc_version,
            max: FKC_VERSION_MAX,
        });
    }
    for kernel in &file.kernels {
        match validate_kernel(kernel) {
            Ok(()) => {}
            // §3.10 + §14/§6: a describe-only (`registrable: false`) section is
            // documentation — it carries no dispatch target and is excluded from
            // lowering/registration (`lower_file` filters it). It may legitimately
            // trip a CONSUMER-AHEAD gate — an `fdx.gather` operand
            // (`GatherNotYetSupported`, rule 14) or an MX/AFFINE_BLOCK quant
            // (`MxNotYetRegistrable`, rule 6) — which is a CORRECT
            // "describable-but-not-yet-registrable" outcome, NOT a defect. Such a
            // describe-only gate MUST NOT block a bundle's importable sections, so
            // it is swallowed HERE (the same "deferred" posture the corpus CI lint
            // takes). Every OTHER error still fails import — a describe-only
            // section's DESCRIPTIVE checks (bad dtype, incoherent layout, …) are
            // still enforced, and a REGISTRABLE section's consumer-ahead gate still
            // fails (it WOULD try to register).
            Err(FkcError::GatherNotYetSupported { .. } | FkcError::MxNotYetRegistrable { .. })
                if !kernel.registrable => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Validate one parsed kernel section against the structural + coherence
/// rules (every rule that operates on the parsed form). The duplicate-
/// `KernelRef` rule (10) is a *register-time* check (`finalize`), not here.
pub fn validate_kernel(kernel: &FkcKernel) -> Result<(), FkcError> {
    let section = kernel.kernel.as_str();

    // §3.10 (rule 17): a describe-only section is documentation. SKIP the
    // dispatch-resolution checks (exactly-one-of op_kind/fused_op, op resolves,
    // op-param namespace) but STILL run every descriptive check below.
    let describe_only = !kernel.registrable;

    // Rule 2: required fields (the describe-only exceptions are applied inside).
    required_fields(kernel, section, describe_only)?;

    // Determine the op target namespace (primitive vs fused) for rule 7. For a
    // describe-only section there is no resolved dispatch target, so the
    // namespace check is skipped (and the op token may be `~` / descriptive).
    let is_fused = if describe_only {
        // No dispatch resolution; op_kind/fused_op need not name a real target.
        // (A descriptive token like `binary`, or `~`, is legal here — §3.10.)
        false
    } else {
        match (kernel.op_kind.as_deref(), kernel.fused_op.as_deref()) {
            (Some(op), None) => {
                // Rule 2/3: op_kind token resolves (reuse the lower table).
                lower::lower_op_kind(op, section)?;
                false
            }
            (None, Some(fused)) => {
                // fused_op token resolves through the SCREAMING_SNAKE table.
                lower::lower_fused_op(fused, section)?;
                true
            }
            (op, fused) => {
                return Err(FkcError::OpTargetAmbiguous {
                    section: section.to_string(),
                    op_kind: op.map(String::from),
                    fused_op: fused.map(String::from),
                });
            }
        }
    };

    // Per-operand checks (rules 3, 4, 5, 6, 14, 15, 16) — descriptive, run for
    // describe-only sections too.
    if let Some(accept) = &kernel.accept {
        for d in &accept.inputs {
            let operand = d.name.as_deref().unwrap_or("<input>").to_string();
            validate_operand(section, &operand, d, accept)?;
        }
        // Rule 7: op-param variant in the correct namespace — SKIPPED for a
        // describe-only section (no resolved dispatch target to bind against).
        if !describe_only {
            if let Some(op_params) = &accept.op_params {
                if let Some(variant) = op_params.variant.as_deref() {
                    validate_op_params_namespace(section, variant, is_fused)?;
                }
            }
        }
    }

    // Rule 16 also covers the OUTPUT dtype tokens (`fixed(DType)` rules) — and
    // rule 13 covers bundle slot ranks. Returns are validated here.
    if let Some(ret) = &kernel.return_ {
        for out in &ret.outputs {
            // `fixed(DType)` output dtype rules name a dtype token; resolve it
            // through the lower table (also covers rule 16 dtype membership).
            if let Some(rule) = out.dtype_rule.as_deref() {
                check_output_dtype_rule(section, out.name.as_deref().unwrap_or("<output>"), rule)?;
            }
        }
        // Rule 13: bundle slot rank ≤ 6 (when a static rank is declarable).
        if let Some(bundle) = &ret.bundle {
            check_bundle_ranks(section, bundle)?;
        }
    }

    // Rule 8/8a: cost expressions parse + provenance present + non-placeholder.
    validate_cost(kernel, section)?;

    // Rule 9 (precision coverage): nondeterministic ⇒ audited + no bit-stable.
    validate_precision_coherence(kernel, section)?;

    Ok(())
}

// ===========================================================================
// Rule 2 — required fields
// ===========================================================================

fn required_fields(kernel: &FkcKernel, section: &str, describe_only: bool) -> Result<(), FkcError> {
    // kernel name is always present (it is the struct's required field).
    // exactly-one-of op_kind/fused_op is checked in validate_kernel's match.

    // blurb: a non-empty one-line string.
    match kernel.blurb.as_deref() {
        Some(b) if !b.trim().is_empty() => {}
        _ => return Err(FkcError::MissingBlurb { section: section.to_string() }),
    }

    // entry_point.
    require(kernel.entry_point.as_deref(), section, "entry_point")?;

    // ≥1 accept.inputs — EXCEPT a describe-only section (§3.10): a
    // documentation-only zero-operand op is legitimate (a zero-input random
    // fill like `rand_uniform`/`rand_normal` whose only "input" is backend-
    // private RNG state, not a graph tensor). The ≥1-input requirement is
    // still enforced for every REGISTRABLE section (a real dispatch target
    // must consume at least one graph operand).
    if !describe_only {
        let has_inputs = kernel
            .accept
            .as_ref()
            .map(|a| !a.inputs.is_empty())
            .unwrap_or(false);
        if !has_inputs {
            return Err(FkcError::MissingRequiredField {
                section: section.to_string(),
                field: "accept.inputs (≥1)".to_string(),
            });
        }
    }

    // ≥1 return.outputs OR a bundle.
    let has_outputs = kernel
        .return_
        .as_ref()
        .map(|r| !r.outputs.is_empty() || r.bundle.is_some())
        .unwrap_or(false);
    if !has_outputs {
        return Err(FkcError::MissingRequiredField {
            section: section.to_string(),
            field: "return.outputs (≥1) or return.bundle".to_string(),
        });
    }

    // a cost block.
    if kernel.cost.is_none() {
        return Err(FkcError::MissingRequiredField {
            section: section.to_string(),
            field: "cost".to_string(),
        });
    }
    // a precision block.
    if kernel.precision.is_none() {
        return Err(FkcError::MissingRequiredField {
            section: section.to_string(),
            field: "precision".to_string(),
        });
    }
    // determinism.
    require(kernel.determinism.as_deref(), section, "determinism")?;

    Ok(())
}

fn require(v: Option<&str>, section: &str, field: &str) -> Result<(), FkcError> {
    match v {
        Some(s) if !s.trim().is_empty() => Ok(()),
        _ => Err(FkcError::MissingRequiredField {
            section: section.to_string(),
            field: field.to_string(),
        }),
    }
}

// ===========================================================================
// Per-operand: rules 3, 4, 5, 6, 14, 15, 16
// ===========================================================================

fn validate_operand(
    section: &str,
    operand: &str,
    d: &TensorDesc,
    accept: &crate::fkc::schema::AcceptBlock,
) -> Result<(), FkcError> {
    // Rule 3 / Rule 16: every dtype token is a real DType (and ∈ FDX table).
    for tok in &d.dtypes {
        lower::lower_dtype(tok, section, operand).map_err(|_| FkcError::FdxTokenNotInTable {
            section: section.to_string(),
            field: format!("{operand}.dtypes"),
            token: tok.clone(),
        })?;
    }

    // Rule 4: layout coherence.
    if let Some(layout) = &d.layout {
        layout_coherence(section, operand, layout)?;
    }

    // Rule 5: awkward-layout coherence PER OPERAND (effective strategy).
    awkward_strategy_coherence(section, operand, d, accept)?;

    // Rule 3/6/16: quant coherence (incl. sub-byte ⇒ fdx.quant, scale single-
    // place, ggml/family/granularity membership, MX/AFFINE_BLOCK gate).
    if let Some(fdx) = &d.fdx {
        // Rule 3: a sub-byte base dtype MUST carry an fdx.quant block.
        let is_sub_byte = d.dtypes.iter().any(|t| matches!(t.as_str(), "F4" | "F6E2M3" | "F6E3M2"));
        if is_sub_byte && fdx.quant.is_none() && fdx.sub_byte.is_none() {
            return Err(FkcError::QuantIncoherent {
                section: section.to_string(),
                operand: operand.to_string(),
                reason: "sub-byte dtype requires an fdx.sub_byte + fdx.quant block (no reliance \
                         on size_in_bytes()==0)"
                    .to_string(),
            });
        }

        if let Some(quant) = &fdx.quant {
            quant_coherence(section, operand, quant, accept)?;
        }

        // Rule 14: gather (paged) coherence.
        if let Some(gather) = &fdx.gather {
            gather_coherence(section, operand, gather, fdx, accept)?;
        }

        // Rule 15: affine / symbolic extent coherence.
        extent_coherence(section, operand, fdx)?;
    }

    Ok(())
}

// --- Rule 4: layout coherence ---

fn layout_coherence(
    section: &str,
    operand: &str,
    layout: &crate::fkc::schema::LayoutSpec,
) -> Result<(), FkcError> {
    use crate::fkc::caps_map::{resolve_layout, Tri};
    // Parse the five flags (also validates each value token; bad value →
    // BadLayoutFlag from resolve_layout).
    let r = resolve_layout(Some(layout), section, operand)?;

    let contiguous_ok = matches!(r.contiguous, Tri::Required | Tri::Accepted);
    let strided_ok = matches!(r.strided, Tri::Accepted);

    // At least one of contiguous / strided must be acceptable.
    if !contiguous_ok && !strided_ok {
        return Err(FkcError::LayoutIncoherent {
            section: section.to_string(),
            operand: operand.to_string(),
            reason: "neither `contiguous` (required|accepted) nor `strided: accepted` is set — \
                     the kernel accepts no layout"
                .to_string(),
        });
    }

    // broadcast_stride0: accepted ⇒ strided: accepted.
    if matches!(r.broadcast_stride0, Tri::Accepted) && !strided_ok {
        return Err(FkcError::LayoutIncoherent {
            section: section.to_string(),
            operand: operand.to_string(),
            reason: "`broadcast_stride0: accepted` requires `strided: accepted` (broadcast is a \
                     stride-0 special case)"
                .to_string(),
        });
    }
    // reverse_strides: accepted ⇒ strided: accepted.
    if matches!(r.reverse_strides, Tri::Accepted) && !strided_ok {
        return Err(FkcError::LayoutIncoherent {
            section: section.to_string(),
            operand: operand.to_string(),
            reason: "`reverse_strides: accepted` requires `strided: accepted` (a negative stride \
                     is still a strided walk)"
                .to_string(),
        });
    }

    // broadcast_axes ⟺ broadcast_stride0: required (§6-additive mask). The
    // mask names the iteration axes the operand must be stride-0 on; it is
    // meaningful ONLY for a REQUIRED (baked) broadcast, and REQUIRED needs it
    // — the shape-blind (op, dtypes, backend) binder can't tell a correctly-
    // broadcast operand from a wrongly-shaped one without the axis set. So:
    // present iff required, and non-empty when required.
    match (&layout.broadcast_axes, matches!(r.broadcast_stride0, Tri::Required)) {
        (Some(axes), true) if axes.is_empty() => {
            return Err(FkcError::LayoutIncoherent {
                section: section.to_string(),
                operand: operand.to_string(),
                reason: "`broadcast_stride0: required` with an EMPTY `broadcast_axes` — name the \
                         iteration axes the operand must be stride-0 on"
                    .to_string(),
            });
        }
        (Some(_), false) => {
            return Err(FkcError::LayoutIncoherent {
                section: section.to_string(),
                operand: operand.to_string(),
                reason: "`broadcast_axes` set without `broadcast_stride0: required` — the mask is \
                         meaningful only for a required (baked) broadcast"
                    .to_string(),
            });
        }
        (None, true) => {
            return Err(FkcError::LayoutIncoherent {
                section: section.to_string(),
                operand: operand.to_string(),
                reason: "`broadcast_stride0: required` needs `broadcast_axes` — the shape-blind \
                         binder can't verify the broadcast without the axis mask"
                    .to_string(),
            });
        }
        _ => {}
    }
    Ok(())
}

// --- Rule 5: awkward-layout strategy coherence (per operand) ---

fn awkward_strategy_coherence(
    section: &str,
    operand: &str,
    d: &TensorDesc,
    _accept: &crate::fkc::schema::AcceptBlock,
) -> Result<(), FkcError> {
    use crate::fkc::caps_map::{resolve_layout, Tri};

    // Effective strategy: per-operand override, else (handled by caller for the
    // kernel-wide default — but the per-operand check only needs the per-operand
    // value when present, since the kernel-wide default is validated against
    // each operand by the caps default below).
    let per_operand = d
        .layout
        .as_ref()
        .and_then(|l| l.awkward_layout_strategy.as_deref());

    let Some(strategy) = per_operand else {
        // No per-operand override: inheritance from the kernel-wide default is
        // checked at the kernel level via the operands' flags; nothing to do.
        return Ok(());
    };

    let r = resolve_layout(d.layout.as_ref(), section, operand)?;
    match strategy {
        "handles_strided" => {
            if !matches!(r.strided, Tri::Accepted) {
                return Err(FkcError::AwkwardStrategyIncoherent {
                    section: section.to_string(),
                    operand: operand.to_string(),
                    strategy: strategy.to_string(),
                    reason: "`handles_strided` requires `strided: accepted` on this operand"
                        .to_string(),
                });
            }
        }
        "requires_contiguous" => {
            if !matches!(r.contiguous, Tri::Required) {
                return Err(FkcError::AwkwardStrategyIncoherent {
                    section: section.to_string(),
                    operand: operand.to_string(),
                    strategy: strategy.to_string(),
                    reason: "`requires_contiguous` requires `contiguous: required` on this operand"
                        .to_string(),
                });
            }
        }
        "contiguize_internally" => {
            // Folds this operand's copy into the kernel's bytes_moved; it must
            // at least *accept* strided input (otherwise there is nothing to
            // contiguize internally).
            if !matches!(r.strided, Tri::Accepted) {
                return Err(FkcError::AwkwardStrategyIncoherent {
                    section: section.to_string(),
                    operand: operand.to_string(),
                    strategy: strategy.to_string(),
                    reason: "`contiguize_internally` requires `strided: accepted` on this operand \
                             (it accepts strided then copies internally)"
                        .to_string(),
                });
            }
        }
        other => {
            return Err(FkcError::AwkwardStrategyIncoherent {
                section: section.to_string(),
                operand: operand.to_string(),
                strategy: other.to_string(),
                reason: "unknown awkward_layout_strategy (meaning-bearing — §11.1)".to_string(),
            });
        }
    }
    Ok(())
}

// --- Rule 6 + 16: quant coherence ---

fn quant_coherence(
    section: &str,
    operand: &str,
    quant: &QuantSpec,
    accept: &crate::fkc::schema::AcceptBlock,
) -> Result<(), FkcError> {
    let family = quant.family.as_deref().unwrap_or("none");

    // Rule 16: family token ∈ FDX table.
    if !is_fdx_quant_family(family) {
        return Err(FkcError::FdxTokenNotInTable {
            section: section.to_string(),
            field: format!("{operand}.fdx.quant.family"),
            token: family.to_string(),
        });
    }
    // Rule 16: granularity token ∈ FDX table (when present, non-null).
    if let Some(g) = quant.granularity.as_deref() {
        if !is_fdx_granularity(g) {
            return Err(FkcError::FdxTokenNotInTable {
                section: section.to_string(),
                field: format!("{operand}.fdx.quant.granularity"),
                token: g.to_string(),
            });
        }
    }
    // Rule 16 + 3: ggml_dtype is a real GgmlDType variant (by code).
    if let Some(g) = quant.ggml_dtype.as_deref() {
        if ggml_dtype_code(g).is_none() {
            // `Q4_K_M` (GGUF name, NOT a variant) lands here.
            return Err(FkcError::QuantIncoherent {
                section: section.to_string(),
                operand: operand.to_string(),
                reason: format!(
                    "ggml_dtype `{g}` is not a real GgmlDType variant (matched by code; \
                     `Q4_K_M` is a GGUF file-format name → use `Q4K`)"
                ),
            });
        }
    }

    // Scale single-place rule: a separate scale_operand XOR a sidecar scale,
    // never both for the same scale. We model the "sidecar scale" as a
    // sidecar `scale_buffer` (not in the schema today) — so we check that
    // `scale_operand`, when present, names a REAL accept.inputs role and that
    // it is not also a GGML INLINE family (GGML scales are baked, never a
    // separate operand) — that is the double-declaration we can detect here.
    if let Some(scale_role) = quant.scale_operand.as_deref() {
        // The role must be a real input operand (single-place: it IS the
        // authority, so it must exist).
        let names_real = accept
            .inputs
            .iter()
            .any(|i| i.name.as_deref() == Some(scale_role));
        if !names_real {
            return Err(FkcError::ScaleDoubleDeclared {
                section: section.to_string(),
                operand: operand.to_string(),
                reason: format!(
                    "scale_operand `{scale_role}` does not name a real accept.inputs role"
                ),
            });
        }
        // GGML_BLOCK scales are INLINE-baked; a separate scale_operand for a
        // GGML family is the double-declaration §10.6 forbids.
        if family == "GGML_BLOCK" {
            return Err(FkcError::ScaleDoubleDeclared {
                section: section.to_string(),
                operand: operand.to_string(),
                reason: "GGML_BLOCK bakes its scale INLINE (per ggml_dtype) — it must NOT also \
                         declare a separate scale_operand"
                    .to_string(),
            });
        }
    }

    // Per-family coherence (§10.6).
    match family {
        "none" | "NONE" => {
            // Dense — no ggml_dtype, no granularity, no scale.
        }
        "GGML_BLOCK" => {
            // Requires a real ggml_dtype; carries NO granularity / NO PerBlock.
            if quant.ggml_dtype.is_none() {
                return Err(FkcError::QuantIncoherent {
                    section: section.to_string(),
                    operand: operand.to_string(),
                    reason: "GGML_BLOCK requires a `ggml_dtype` (the block format)".to_string(),
                });
            }
            if quant.granularity.is_some() {
                return Err(FkcError::QuantIncoherent {
                    section: section.to_string(),
                    operand: operand.to_string(),
                    reason: "GGML_BLOCK carries NO granularity (the scale is baked INLINE per \
                             ggml_dtype; no ScalePair, no PerBlock)"
                        .to_string(),
                });
            }
        }
        "AFFINE_INT" | "AFFINE_FLOAT" => {
            // granularity ∈ {PerTensor, PerToken, PerChannel} (NOT PerBlock).
            match quant.granularity.as_deref() {
                Some(g) if is_registrable_granularity(g) => {}
                Some("PerBlock") => {
                    return Err(FkcError::QuantIncoherent {
                        section: section.to_string(),
                        operand: operand.to_string(),
                        reason: format!(
                            "{family} granularity must be PerTensor|PerToken|PerChannel \
                             (PerBlock is MX-only)"
                        ),
                    });
                }
                _ => {
                    return Err(FkcError::QuantIncoherent {
                        section: section.to_string(),
                        operand: operand.to_string(),
                        reason: format!(
                            "{family} requires granularity ∈ PerTensor|PerToken|PerChannel"
                        ),
                    });
                }
            }
        }
        "MX" => {
            // MX ⇒ granularity: PerBlock; parse-validates but NOT registrable.
            if quant.granularity.as_deref() != Some("PerBlock") {
                return Err(FkcError::QuantIncoherent {
                    section: section.to_string(),
                    operand: operand.to_string(),
                    reason: "MX requires granularity: PerBlock (F8E8M0 per-block scale)"
                        .to_string(),
                });
            }
            return Err(FkcError::MxNotYetRegistrable {
                section: section.to_string(),
                family: "MX".to_string(),
                reason: "no ScaleGranularity::PerBlock target type yet (§6)".to_string(),
            });
        }
        "AFFINE_BLOCK" => {
            // Block geometry present — EITHER `block_shape` OR a block-shaped
            // SEPARATE scale operand (§10.6: "block geometry present
            // (`block_shape` / a block-shaped separate scale operand)"). Both
            // forms are legal; a contract that carries the per-block absmax as
            // a separate `scale_operand` (the single-place SEPARATE_BUFFER
            // form) does not also need an inline `block_shape`. Granularity is
            // NOT PerBlock (its grain rides block_shape / the absmax operand).
            // Parse-validates but is NOT registrable yet (no block-quant
            // descriptor target).
            let has_block_shape = quant.block_shape.is_some();
            let has_separate_scale = quant.scale_operand.is_some();
            if !has_block_shape && !has_separate_scale {
                return Err(FkcError::QuantIncoherent {
                    section: section.to_string(),
                    operand: operand.to_string(),
                    reason: "AFFINE_BLOCK requires block geometry — either `block_shape` or a \
                             block-shaped separate `scale_operand` (§10.6)"
                        .to_string(),
                });
            }
            if quant.granularity.as_deref() == Some("PerBlock") {
                return Err(FkcError::QuantIncoherent {
                    section: section.to_string(),
                    operand: operand.to_string(),
                    reason: "AFFINE_BLOCK must NOT use granularity: PerBlock (PerBlock stays \
                             MX-only; its grain rides block_shape)"
                        .to_string(),
                });
            }
            return Err(FkcError::MxNotYetRegistrable {
                section: section.to_string(),
                family: "AFFINE_BLOCK".to_string(),
                reason: "no block-quant descriptor target type yet (§6)".to_string(),
            });
        }
        _ => unreachable!("family membership already checked against the FDX table"),
    }

    Ok(())
}

// --- Rule 14: gather (paged) coherence ---

fn gather_coherence(
    section: &str,
    operand: &str,
    gather: &crate::fkc::schema::GatherSpec,
    fdx: &crate::fkc::schema::FdxSpec,
    accept: &crate::fkc::schema::AcceptBlock,
) -> Result<(), FkcError> {
    let Some(kind) = gather.kind.as_deref() else {
        // `kind: ~` ⇒ no gather declared.
        return Ok(());
    };
    match kind {
        "paged_blocks" => {
            // (a) requires_ext: true.
            if fdx.requires_ext != Some(true) {
                return Err(FkcError::GatherIncoherent {
                    section: section.to_string(),
                    operand: operand.to_string(),
                    reason: "paged_blocks requires `fdx.requires_ext: true`".to_string(),
                });
            }
            // (b) symbolic_extent: required.
            if fdx.symbolic_extent.as_deref() != Some("required") {
                return Err(FkcError::GatherIncoherent {
                    section: section.to_string(),
                    operand: operand.to_string(),
                    reason: "paged_blocks requires `fdx.symbolic_extent: required`".to_string(),
                });
            }
            // (c) block_table / context_lens, when non-~, name real input roles.
            for (label, role) in [
                ("block_table", gather.block_table.as_deref()),
                ("context_lens", gather.context_lens.as_deref()),
            ] {
                if let Some(role) = role {
                    let exists = accept.inputs.iter().any(|i| i.name.as_deref() == Some(role));
                    if !exists {
                        return Err(FkcError::GatherIncoherent {
                            section: section.to_string(),
                            operand: operand.to_string(),
                            reason: format!(
                                "gather.{label} `{role}` does not name a real accept.inputs role"
                            ),
                        });
                    }
                }
            }
            // IMPORT-SIDE LIFT (FDX gather-sidecar arc, slice A). The `gather`
            // block is boundary METADATA that DESCRIBES what an FDX view of this
            // operand will someday carry (`FDXIndexedResidency`, FDX §6.9 —
            // "Description only: no cost, no decision"). Registration/dispatch of
            // the consuming op (`OpKind::PagedAttn`) does NOT depend on it: the
            // as-built ABI passes `block_table` / `context_lens` as ORDINARY U32
            // graph inputs (named by the coherence check above) + the geometry in
            // `OpParams::PagedAttn`; the kernel reads them directly, never through
            // an FDX gather view. So a COHERENT `paged_blocks` operand is
            // registrable NOW — we validated its internal coherence (kind /
            // requires_ext / symbolic_extent / real block_table+context_lens
            // roles); nothing further is a build-time gate. What remains
            // [consumer-ahead] is the FDX VIEW layer that would MATERIALIZE this
            // descriptor at the kernel boundary (`view_with_gather` +
            // `Capability::DlpackExtGather` direct-admission, FKC §3.9.1) — a
            // separate seam with no consumer yet, NOT the import gate. An
            // INCOHERENT gather still errors above (`GatherIncoherent`), never a
            // silent pass. (`GatherNotYetSupported` stays a reserved variant for
            // a future ragged/CSR `kind >= 2` that FKC cannot yet key.)
            Ok(())
        }
        other => Err(FkcError::UnknownAdmissibilityEnum {
            section: section.to_string(),
            field: format!("{operand}.fdx.gather.kind"),
            value: other.to_string(),
        }),
    }
}

// --- Rule 15: affine / symbolic extent coherence ---

fn extent_coherence(
    section: &str,
    operand: &str,
    fdx: &crate::fkc::schema::FdxSpec,
) -> Result<(), FkcError> {
    // symbolic_extent ∈ {rejected, tolerated, required} (when present).
    if let Some(se) = fdx.symbolic_extent.as_deref() {
        if !matches!(se, "rejected" | "tolerated" | "required") {
            return Err(FkcError::UnknownAdmissibilityEnum {
                section: section.to_string(),
                field: format!("{operand}.fdx.symbolic_extent"),
                value: se.to_string(),
            });
        }
    }
    // extent_kind ∈ {rejected, scalar, range, affine} (when present).
    if let Some(ek) = fdx.extent_kind.as_deref() {
        if !matches!(ek, "rejected" | "scalar" | "range" | "affine") {
            return Err(FkcError::UnknownAdmissibilityEnum {
                section: section.to_string(),
                field: format!("{operand}.fdx.extent_kind"),
                value: ek.to_string(),
            });
        }
        // extent_kind: range|affine ⇒ symbolic_extent: required.
        if matches!(ek, "range" | "affine") && fdx.symbolic_extent.as_deref() != Some("required") {
            return Err(FkcError::UnknownAdmissibilityEnum {
                section: section.to_string(),
                field: format!("{operand}.fdx.extent_kind"),
                value: format!(
                    "{ek} requires symbolic_extent: required (got {:?})",
                    fdx.symbolic_extent
                ),
            });
        }
    }
    Ok(())
}

// ===========================================================================
// Rule 7 — op-param namespace
// ===========================================================================

/// An `op_kind` contract names an `OpParams` variant; a `fused_op` contract
/// names a `FusedOpParams` variant (§3.7, §10.7). The two namespaces are
/// distinct; checking the wrong one is `BadOpParamsVariant`.
fn validate_op_params_namespace(
    section: &str,
    variant: &str,
    is_fused: bool,
) -> Result<(), FkcError> {
    let ok = if is_fused {
        is_fused_op_params_variant(variant)
    } else {
        is_op_params_variant(variant)
    };
    if ok {
        Ok(())
    } else {
        Err(FkcError::BadOpParamsVariant {
            section: section.to_string(),
            variant: variant.to_string(),
            namespace: if is_fused { "FusedOpParams" } else { "OpParams" }.to_string(),
        })
    }
}

/// The as-built `OpParams` variant names (`fuel-dispatch/src/kernel.rs`).
fn is_op_params_variant(v: &str) -> bool {
    matches!(
        v,
        "None"
            | "Reduce"
            | "Matmul"
            | "Conv1D"
            | "Conv2D"
            | "ConvTranspose1D"
            | "ConvTranspose2D"
            | "ReduceSumTo"
            | "ReduceMaxTo"
            | "ReduceMaxToBackward"
            | "Cast"
            | "Affine"
            | "Clamp"
            | "PowI"
            | "Concat"
            | "Slice"
            | "Pad"
            | "PadBackward"
            | "Flip"
            | "Roll"
            | "CumSum"
            | "Triangular"
            | "MaskedFill"
            | "IndexSelect"
            | "Gather"
            | "IndexAdd"
            | "ScatterAdd"
            | "Rope"
            | "SoftmaxLastDim"
            | "LogSoftmaxLastDim"
            | "NormLastDim"
            | "FlashAttn"
            | "PagedAttn"
            | "QMatMul"
            | "Nf4Matmul"
            | "WriteSlice"
            | "WriteSliceRotating"
            | "WriteSliceDoff"
            | "SelectiveScan"
            | "SsdChunkScan"
            | "CausalConv1d"
            | "FusedSoftmaxCrossEntropy"
    )
}

/// The as-built `FusedOpParams` variant names (`fuel-graph/src/registry.rs`).
fn is_fused_op_params_variant(v: &str) -> bool {
    matches!(
        v,
        "SoftmaxLastDim"
            | "FusedLinear"
            | "RmsNormLastDim"
            | "LayerNormLastDim"
            | "Rope"
            | "Conv2D"
            | "SoftmaxLastDimBackward"
            | "LayerNormLastDimBackward"
            | "RmsNormLastDimBackward"
            | "ReduceMaxToBackward"
            | "PowIBackward"
            | "ConvTranspose2D"
            | "FlashAttn"
            | "PagedAttn"
            | "QMatMul"
            | "InplaceAffine"
            | "SsdChunkScan"
            | "Nf4Matmul"
            | "FlashAttnBackward"
            | "SelectiveScan"
            | "CausalConv1d"
            | "FusedSoftmaxCrossEntropy"
    )
}

// ===========================================================================
// Returns: rule 16 (output dtype) + rule 13 (bundle rank)
// ===========================================================================

/// A `fixed(DType)` output dtype rule names a dtype token that must be a real
/// DType (rule 16). `passthrough(role)` / other rules carry no dtype literal.
fn check_output_dtype_rule(section: &str, operand: &str, rule: &str) -> Result<(), FkcError> {
    let rule = rule.trim();
    if let Some(inner) = rule.strip_prefix("fixed(").and_then(|s| s.strip_suffix(")")) {
        let tok = inner.trim();
        lower::lower_dtype(tok, section, operand).map_err(|_| FkcError::FdxTokenNotInTable {
            section: section.to_string(),
            field: format!("{operand}.dtype_rule fixed(...)"),
            token: tok.to_string(),
        })?;
    }
    Ok(())
}

/// Rule 13: every bundle slot's declared shape must be rank ≤ 6 when a static
/// shape is given (an inline `[d0, d1, ...]` literal). A `shape_rule:` string
/// has no statically-knowable rank without evaluating the rule, so it is not
/// rank-checked here (the register slice's FusedOpEntry cross-check covers it).
fn check_bundle_ranks(
    section: &str,
    bundle: &serde_yml::Value,
) -> Result<(), FkcError> {
    let serde_yml::Value::Sequence(slots) = bundle else {
        return Ok(());
    };
    for (i, slot) in slots.iter().enumerate() {
        let serde_yml::Value::Mapping(map) = slot else {
            continue;
        };
        let slot_name = map
            .get(serde_yml::Value::String("name".into()))
            .and_then(|v| v.as_str())
            .unwrap_or(&format!("slot{i}"))
            .to_string();
        // A static `shape:` literal list is rank-checkable.
        if let Some(serde_yml::Value::Sequence(dims)) =
            map.get(serde_yml::Value::String("shape".into()))
        {
            if dims.len() > 6 {
                return Err(FkcError::BundleSlotRankExceeded {
                    section: section.to_string(),
                    slot: slot_name,
                    rank: dims.len(),
                });
            }
        }
    }
    Ok(())
}

// ===========================================================================
// Rule 8 / 8a — cost
// ===========================================================================

fn validate_cost(kernel: &FkcKernel, section: &str) -> Result<(), FkcError> {
    let Some(cost) = &kernel.cost else {
        // required_fields already errored on an absent cost block.
        return Ok(());
    };

    // Rule 8a: provenance present + ∈ {declared, judge_measured}.
    match cost.provenance.as_deref() {
        Some("declared") | Some("judge_measured") => {}
        other => {
            return Err(FkcError::CostProvenanceMissing {
                section: section.to_string(),
                found: other.map(String::from),
            });
        }
    }

    // Rule 8: every present cost expression parses (reuse the cost_expr
    // parser). A field that is `~`/absent is fine (Unknown).
    use crate::fkc::cost_expr;
    let parse = |field: &str, src: Option<&str>| -> Result<(), FkcError> {
        cost_expr::compile_field(src)
            .map(|_| ())
            .map_err(|e| FkcError::CostExprParse {
                section: section.to_string(),
                field: field.to_string(),
                expr: src.unwrap_or("").to_string(),
                reason: e.to_string(),
            })
    };
    parse("flops", cost.flops.as_deref())?;
    parse("bytes_moved", cost.bytes_moved.as_deref())?;
    if let Some(serde_yml::Value::String(s)) = &cost.overhead_ns {
        parse("overhead_ns", Some(s))?;
    }
    if let Some(mem) = &cost.memory {
        for (field, v) in [
            ("memory.device_bytes", &mem.device_bytes),
            ("memory.host_bytes", &mem.host_bytes),
            ("memory.disk_bytes", &mem.disk_bytes),
        ] {
            if let Some(serde_yml::Value::String(s)) = v {
                parse(field, Some(s))?;
            }
        }
    }

    // Rule 8a (placeholder): a `class: free` op with all-zero coefficients is
    // the honest metadata-only declaration. Otherwise a cost that ships NO
    // coefficient (all `~`) AND no `class` is a placeholder. A `judge_measured`
    // cost legitimately ships shape-hint expressions OR all-`~` (the Judge
    // populates it) — the corpus convention is `judge_measured` + `~`
    // coefficients, which is NOT a placeholder (it is explicitly measured),
    // so we DO NOT flag all-`~` under `judge_measured`.
    let class = cost.class.as_deref().unwrap_or("");
    let has_any_expr = cost.flops.as_deref().is_some_and(|s| !s.trim().is_empty())
        || cost.bytes_moved.as_deref().is_some_and(|s| !s.trim().is_empty());
    if cost.provenance.as_deref() == Some("declared")
        && !has_any_expr
        && class != "free"
        && class.is_empty()
    {
        return Err(FkcError::PlaceholderCost {
            section: section.to_string(),
            reason: "provenance: declared with no coefficient expressions and no class — a bare \
                     placeholder (use class: free for an honest metadata-only op)"
                .to_string(),
        });
    }

    Ok(())
}

// ===========================================================================
// Rule 9 — precision coherence (determinism cross-check)
// ===========================================================================

fn validate_precision_coherence(kernel: &FkcKernel, section: &str) -> Result<(), FkcError> {
    let det = kernel.determinism.as_deref().unwrap_or("");
    if det == "nondeterministic" {
        // nondeterministic ⇒ bit_stable=false + audited:true (no silent
        // unaudited nondeterminism; §4.9 / §10.9).
        if let Some(p) = &kernel.precision {
            if p.bit_stable_on_same_hardware == Some(true) {
                return Err(FkcError::QuantIncoherent {
                    section: section.to_string(),
                    operand: "<precision>".to_string(),
                    reason: "determinism: nondeterministic requires bit_stable_on_same_hardware = \
                             false"
                        .to_string(),
                });
            }
            if p.audited != Some(true) {
                return Err(FkcError::QuantIncoherent {
                    section: section.to_string(),
                    operand: "<precision>".to_string(),
                    reason: "determinism: nondeterministic requires audited: true with a \
                             none(reason) precision (no silent unaudited nondeterminism)"
                        .to_string(),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fkc::parse_file;

    // --- A minimal VALID contract builder for unit tests. ---

    /// A full, valid single-kernel bundle with the required fields, mutated by
    /// the negative tests via string replacement of the `__SLOT__` markers.
    fn valid_bundle() -> String {
        r#"---
fkc_version: 1
provider:
  name: test-provider
  backend: Cpu
  kernel_source: "test-cpu"
---

# test bundle

## demo

A blurb.

```fkc
kernel: demo
op_kind: AddElementwise
blurb: "demo blurb"
entry_point: "x::y"
accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
  op_params: { variant: None }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
caps:
  awkward_layout_strategy: requires_contiguous
cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * 4"
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```
"#
        .to_string()
    }

    fn validate_str(src: &str) -> Result<(), FkcError> {
        let file = parse_file(src).expect("fixture parses");
        validate_file(&file)
    }

    #[test]
    fn valid_contract_passes() {
        validate_str(&valid_bundle()).expect("the valid fixture validates");
    }

    // ===== Rule 1 =====
    #[test]
    fn version_too_new_is_unsupported() {
        let src = valid_bundle().replace("fkc_version: 1", "fkc_version: 99");
        let err = validate_str(&src).expect_err("version 99 unsupported");
        assert!(matches!(err, FkcError::UnsupportedVersion { found: 99, max: 1 }), "got {err:?}");
    }

    // ===== Rule 2 =====
    #[test]
    fn missing_blurb_errors() {
        let src = valid_bundle().replace("blurb: \"demo blurb\"\n", "");
        let err = validate_str(&src).expect_err("no blurb");
        assert!(matches!(err, FkcError::MissingBlurb { .. }), "got {err:?}");
    }

    #[test]
    fn missing_cost_block_errors() {
        // Remove the whole cost block.
        let src = valid_bundle();
        let cut = src.replace(
            "cost:\n  provenance: declared\n  class: cheap_elementwise\n  flops: \"n\"\n  bytes_moved: \"3 * n * 4\"\n",
            "",
        );
        let err = validate_str(&cut).expect_err("no cost block");
        assert!(
            matches!(err, FkcError::MissingRequiredField { ref field, .. } if field == "cost"),
            "got {err:?}"
        );
    }

    #[test]
    fn missing_determinism_errors() {
        let src = valid_bundle().replace("determinism: same_hardware_bitwise\n", "");
        let err = validate_str(&src).expect_err("no determinism");
        assert!(
            matches!(err, FkcError::MissingRequiredField { ref field, .. } if field == "determinism"),
            "got {err:?}"
        );
    }

    #[test]
    fn both_op_kind_and_fused_op_is_ambiguous() {
        let src = valid_bundle().replace(
            "op_kind: AddElementwise",
            "op_kind: AddElementwise\nfused_op: SOFTMAX_LAST_DIM",
        );
        let err = validate_str(&src).expect_err("both targets");
        assert!(matches!(err, FkcError::OpTargetAmbiguous { .. }), "got {err:?}");
    }

    // ===== Rule 3 / 16 — dtype =====
    #[test]
    fn bad_dtype_token_is_fdx_token_not_in_table() {
        let src = valid_bundle().replace("dtypes: [F32]", "dtypes: [F99]");
        let err = validate_str(&src).expect_err("F99 not a dtype");
        assert!(matches!(err, FkcError::FdxTokenNotInTable { .. }), "got {err:?}");
    }

    // ===== Rule 4 — layout coherence =====
    #[test]
    fn layout_no_acceptable_form_is_incoherent() {
        // contiguous: n/a + strided: rejected ⇒ accepts no layout.
        let src = valid_bundle().replace(
            "layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }",
            "layout: { contiguous: \"n/a\", strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }",
        );
        let err = validate_str(&src).expect_err("no acceptable layout");
        assert!(matches!(err, FkcError::LayoutIncoherent { .. }), "got {err:?}");
    }

    #[test]
    fn broadcast_without_strided_is_incoherent() {
        let src = valid_bundle().replace(
            "layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }",
            "layout: { contiguous: required, strided: rejected, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }",
        );
        let err = validate_str(&src).expect_err("broadcast w/o strided");
        assert!(matches!(err, FkcError::LayoutIncoherent { .. }), "got {err:?}");
    }

    #[test]
    fn reverse_without_strided_is_incoherent() {
        let src = valid_bundle().replace(
            "layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }",
            "layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: accepted }",
        );
        let err = validate_str(&src).expect_err("reverse w/o strided");
        assert!(matches!(err, FkcError::LayoutIncoherent { .. }), "got {err:?}");
    }

    // ===== Rule 4 — broadcast_axes mask (§6-additive, path 1a) =====
    #[test]
    fn required_broadcast_with_axes_is_valid() {
        // Baracuda's §6-additive spelling for a baked bias-add cell.
        let src = valid_bundle()
            .replace(
                "layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }",
                "layout: { contiguous: accepted, strided: accepted, broadcast_stride0: required, broadcast_axes: [0], start_offset: rejected, reverse_strides: rejected }",
            )
            .replace(
                "awkward_layout_strategy: requires_contiguous",
                "awkward_layout_strategy: handles_strided",
            );
        validate_str(&src)
            .expect("required broadcast with a non-empty broadcast_axes mask validates");
    }

    #[test]
    fn required_broadcast_without_axes_is_incoherent() {
        let src = valid_bundle().replace(
            "layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }",
            "layout: { contiguous: accepted, strided: accepted, broadcast_stride0: required, start_offset: rejected, reverse_strides: rejected }",
        );
        let err = validate_str(&src).expect_err("required broadcast needs a broadcast_axes mask");
        assert!(matches!(err, FkcError::LayoutIncoherent { .. }), "got {err:?}");
    }

    #[test]
    fn broadcast_axes_without_required_is_incoherent() {
        let src = valid_bundle().replace(
            "layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }",
            "layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, broadcast_axes: [0], start_offset: rejected, reverse_strides: rejected }",
        );
        let err = validate_str(&src).expect_err("broadcast_axes is meaningful only with required");
        assert!(matches!(err, FkcError::LayoutIncoherent { .. }), "got {err:?}");
    }

    #[test]
    fn required_broadcast_with_empty_axes_is_incoherent() {
        let src = valid_bundle().replace(
            "layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }",
            "layout: { contiguous: accepted, strided: accepted, broadcast_stride0: required, broadcast_axes: [], start_offset: rejected, reverse_strides: rejected }",
        );
        let err = validate_str(&src).expect_err("empty broadcast_axes on a required broadcast");
        assert!(matches!(err, FkcError::LayoutIncoherent { .. }), "got {err:?}");
    }

    // ===== Rule 5 — awkward strategy =====
    #[test]
    fn handles_strided_without_strided_accepted_is_incoherent() {
        let src = valid_bundle().replace(
            "layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }",
            "layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected, awkward_layout_strategy: handles_strided }",
        );
        let err = validate_str(&src).expect_err("handles_strided but strided rejected");
        assert!(matches!(err, FkcError::AwkwardStrategyIncoherent { .. }), "got {err:?}");
    }

    #[test]
    fn unknown_awkward_strategy_is_typed_error() {
        let src = valid_bundle().replace(
            "layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }",
            "layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected, awkward_layout_strategy: teleports }",
        );
        let err = validate_str(&src).expect_err("unknown strategy");
        assert!(matches!(err, FkcError::AwkwardStrategyIncoherent { .. }), "got {err:?}");
    }

    // ===== Rule 6 — quant =====
    fn quant_bundle(family: &str, extra: &str) -> String {
        // A QMatMul-shaped contract whose weight carries an fdx.quant block.
        format!(
            r#"---
fkc_version: 1
provider:
  name: test-provider
  backend: Cpu
  kernel_source: "test-cpu"
---

# q

## q

blurb.

```fkc
kernel: q
op_kind: QMatMul
blurb: "q"
entry_point: "x::q"
accept:
  inputs:
    - name: act
      dtypes: [F32]
      layout: {{ contiguous: required, strided: rejected }}
    - name: weight
      dtypes: [U8]
      layout: {{ contiguous: required, strided: rejected }}
      fdx:
        requires_ext: true
        quant:
          family: {family}
{extra}
  op_params: {{ variant: QMatMul }}
return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
cost:
  provenance: judge_measured
  class: gemm_like
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```
"#
        )
    }

    #[test]
    fn ggml_block_with_real_dtype_ok() {
        let src = quant_bundle("GGML_BLOCK", "          ggml_dtype: Q4_0\n          role: weight");
        validate_str(&src).expect("GGML_BLOCK Q4_0 weight validates");
    }

    #[test]
    fn ggml_dtype_q4_k_m_is_quant_incoherent() {
        // The GGUF file-format name `Q4_K_M` is NOT a GgmlDType variant.
        let src = quant_bundle("GGML_BLOCK", "          ggml_dtype: Q4_K_M\n          role: weight");
        let err = validate_str(&src).expect_err("Q4_K_M bad");
        assert!(matches!(err, FkcError::QuantIncoherent { .. }), "got {err:?}");
    }

    #[test]
    fn ggml_block_with_granularity_is_incoherent() {
        let src = quant_bundle(
            "GGML_BLOCK",
            "          ggml_dtype: Q4_0\n          granularity: PerChannel\n          role: weight",
        );
        let err = validate_str(&src).expect_err("GGML carries no granularity");
        assert!(matches!(err, FkcError::QuantIncoherent { .. }), "got {err:?}");
    }

    #[test]
    fn mx_family_is_not_yet_registrable() {
        let src = quant_bundle("MX", "          granularity: PerBlock\n          role: weight");
        let err = validate_str(&src).expect_err("MX not registrable");
        assert!(matches!(err, FkcError::MxNotYetRegistrable { ref family, .. } if family == "MX"), "got {err:?}");
    }

    #[test]
    fn affine_block_without_block_shape_is_incoherent() {
        let src = quant_bundle("AFFINE_BLOCK", "          role: weight\n          scale_operand: scl");
        // scale_operand `scl` won't exist either, but block_shape missing OR the
        // scale-operand check fires; both are coherence errors.
        let err = validate_str(&src).expect_err("AFFINE_BLOCK needs block_shape / real scale");
        assert!(
            matches!(err, FkcError::QuantIncoherent { .. } | FkcError::ScaleDoubleDeclared { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn affine_int_bad_granularity_is_incoherent() {
        let src = quant_bundle("AFFINE_INT", "          granularity: PerBlock\n          role: weight");
        let err = validate_str(&src).expect_err("AFFINE_INT PerBlock bad");
        assert!(matches!(err, FkcError::QuantIncoherent { .. }), "got {err:?}");
    }

    #[test]
    fn unknown_family_is_fdx_token_not_in_table() {
        let src = quant_bundle("MADE_UP", "          role: weight");
        let err = validate_str(&src).expect_err("MADE_UP not an FDX family");
        assert!(matches!(err, FkcError::FdxTokenNotInTable { .. }), "got {err:?}");
    }

    #[test]
    fn ggml_with_separate_scale_operand_is_double_declared() {
        // GGML scales are INLINE; declaring a separate scale_operand is the
        // double-declaration §10.6 forbids. (act is a real input role.)
        let src = quant_bundle(
            "GGML_BLOCK",
            "          ggml_dtype: Q4_0\n          role: weight\n          scale_operand: act",
        );
        let err = validate_str(&src).expect_err("GGML + separate scale");
        assert!(matches!(err, FkcError::ScaleDoubleDeclared { .. }), "got {err:?}");
    }

    // ===== Rule 7 — op-param namespace =====
    #[test]
    fn op_kind_with_fused_only_variant_is_bad_namespace() {
        // `FusedLinear` is a FusedOpParams variant, NOT an OpParams variant; an
        // op_kind contract naming it must fail the namespace check.
        let src = valid_bundle().replace("op_params: { variant: None }", "op_params: { variant: FusedLinear }");
        let err = validate_str(&src).expect_err("FusedLinear not in OpParams");
        assert!(
            matches!(err, FkcError::BadOpParamsVariant { ref namespace, .. } if namespace == "OpParams"),
            "got {err:?}"
        );
    }

    // ===== Rule 8 / 8a — cost =====
    #[test]
    fn missing_provenance_errors() {
        let src = valid_bundle().replace("  provenance: declared\n", "");
        let err = validate_str(&src).expect_err("no provenance");
        assert!(matches!(err, FkcError::CostProvenanceMissing { .. }), "got {err:?}");
    }

    #[test]
    fn bad_cost_expr_is_cost_expr_parse() {
        let src = valid_bundle().replace("flops: \"n\"", "flops: \"2 * * n\"");
        let err = validate_str(&src).expect_err("malformed flops");
        assert!(matches!(err, FkcError::CostExprParse { .. }), "got {err:?}");
    }

    #[test]
    fn declared_with_no_coeffs_and_no_class_is_placeholder() {
        let src = valid_bundle()
            .replace("  class: cheap_elementwise\n", "")
            .replace("  flops: \"n\"\n", "")
            .replace("  bytes_moved: \"3 * n * 4\"\n", "");
        let err = validate_str(&src).expect_err("bare declared cost");
        assert!(matches!(err, FkcError::PlaceholderCost { .. }), "got {err:?}");
    }

    #[test]
    fn judge_measured_all_tilde_is_not_placeholder() {
        // The corpus convention: judge_measured + all-`~` coefficients is an
        // explicit measured cost, NOT a placeholder.
        let src = valid_bundle()
            .replace("provenance: declared", "provenance: judge_measured")
            .replace("  flops: \"n\"\n", "  flops: ~\n")
            .replace("  bytes_moved: \"3 * n * 4\"\n", "  bytes_moved: ~\n");
        validate_str(&src).expect("judge_measured + ~ is fine");
    }

    // ===== Rule 9 — determinism/precision =====
    #[test]
    fn nondeterministic_with_bitstable_true_is_incoherent() {
        let src = valid_bundle().replace("determinism: same_hardware_bitwise", "determinism: nondeterministic");
        let err = validate_str(&src).expect_err("nondeterministic + bit_stable true");
        assert!(matches!(err, FkcError::QuantIncoherent { .. }), "got {err:?}");
    }

    // ===== Rule 15 — extent =====
    #[test]
    fn unknown_extent_kind_is_typed_error() {
        let src = quant_bundle("none", "")
            .replace(
                "      fdx:\n        requires_ext: true\n        quant:\n          family: none\n",
                "      fdx:\n        extent_kind: wobbly\n        symbolic_extent: required\n",
            );
        let err = validate_str(&src).expect_err("bad extent_kind");
        assert!(matches!(err, FkcError::UnknownAdmissibilityEnum { .. }), "got {err:?}");
    }

    // ===== Rule 17 / §3.10 — describe-only (non-registrable) sections =====

    /// A describe-only chassis-umbrella bundle whose `op_kind:` is the
    /// DESCRIPTIVE token `binary` (NOT a real `OpKind`) — legal because
    /// `registrable: false` skips the dispatch-resolution checks (§3.10).
    fn describe_only_bundle() -> String {
        r#"---
fkc_version: 1
provider:
  name: test-provider
  backend: Cpu
  kernel_source: "test-cpu"
---

# describe-only bundle

## binary

A chassis umbrella.

```fkc
kernel: binary
registrable: false
op_kind: binary
blurb: "shared binary chassis (documentation umbrella)"
entry_point: "x::chassis"
accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
cost:
  provenance: judge_measured
  class: cheap_elementwise
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```
"#
        .to_string()
    }

    #[test]
    fn describe_only_with_descriptive_op_kind_validates() {
        // `op_kind: binary` is NOT a real OpKind, but registrable: false skips
        // the resolution check — the section validates as documentation.
        validate_str(&describe_only_bundle()).expect("describe-only section validates");
    }

    #[test]
    fn describe_only_with_tilde_op_kind_validates() {
        // `op_kind: ~` (absent) is legal for a describe-only section — neither
        // op_kind nor fused_op need resolve.
        let src = describe_only_bundle().replace("op_kind: binary\n", "op_kind: ~\n");
        validate_str(&src).expect("describe-only with op_kind: ~ validates");
    }

    #[test]
    fn non_describe_only_still_requires_real_op_kind_regression() {
        // REGRESSION: the SAME section WITHOUT registrable: false must still
        // fail — a descriptive `binary` token is not a real OpKind, so a
        // registrable section is rejected (no relaxation of the validator).
        let src = describe_only_bundle().replace("registrable: false\n", "");
        let err = validate_str(&src).expect_err("registrable section needs a real op_kind");
        assert!(matches!(err, FkcError::UnknownOpKind { .. }), "got {err:?}");
    }

    #[test]
    fn describe_only_still_validates_descriptive_dtypes() {
        // A describe-only section's descriptive checks STILL run: a bogus dtype
        // token is rejected by the FDX-subset drift-guard even though dispatch
        // resolution is skipped.
        let src = describe_only_bundle().replace("dtypes: [F32, F64, BF16, F16]", "dtypes: [F99]");
        let err = validate_str(&src).expect_err("bad dtype in describe-only docs");
        assert!(matches!(err, FkcError::FdxTokenNotInTable { .. }), "got {err:?}");
    }

    #[test]
    fn describe_only_still_validates_layout_coherence() {
        // Layout coherence (rule 4) still runs for describe-only docs.
        let src = describe_only_bundle().replace(
            "layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }",
            "layout: { contiguous: \"n/a\", strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }",
        );
        let err = validate_str(&src).expect_err("incoherent layout in describe-only docs");
        assert!(matches!(err, FkcError::LayoutIncoherent { .. }), "got {err:?}");
    }

    /// A describe-only zero-operand op (e.g. a `rand_uniform`/`rand_normal`
    /// random fill whose only "input" is backend-private RNG state, not a graph
    /// tensor): `registrable: false` + `accept.inputs: []`. Mirrors the
    /// `metal/sort-random.fkc.md` random-fill sections.
    fn describe_only_zero_input_bundle() -> String {
        r#"---
fkc_version: 1
provider:
  name: test-provider
  backend: Cpu
  kernel_source: "test-cpu"
---

# describe-only zero-input bundle

## rand_fill

A zero-operand random fill.

```fkc
kernel: rand_fill
registrable: false
op_kind: RandUniform
blurb: "fill a dense buffer with U(min,max); no input tensor (RNG state is backend-private)"
entry_point: "x::rand_fill"
accept:
  inputs: []
return:
  outputs:
    - name: out
      dtype_rule: cast(out)
cost:
  provenance: judge_measured
  class: cheap_elementwise
precision:
  bit_stable_on_same_hardware: false
  audited: true
determinism: nondeterministic
```
"#
        .to_string()
    }

    #[test]
    fn describe_only_zero_input_fill_validates() {
        // §3.10 carve-out: a describe-only section with `accept.inputs: []` is a
        // legitimate zero-operand documentation op (random fill) — the ≥1-input
        // required-field check is EXEMPTED for describe-only sections.
        validate_str(&describe_only_zero_input_bundle())
            .expect("describe-only zero-input fill validates");
    }

    #[test]
    fn registrable_zero_input_still_fails_missing_required_field() {
        // REGRESSION: the SAME zero-input section WITHOUT registrable: false must
        // still fail the ≥1-input rule — the carve-out applies ONLY to
        // describe-only sections (a real dispatch target must consume ≥1 graph
        // operand). No relaxation for registrable sections.
        let src = describe_only_zero_input_bundle()
            .replace("registrable: false\n", "")
            // Use a real OpKind so the missing-input rule (not the op-resolution
            // rule) is the failure under test. The required-field battery runs
            // before op resolution, so `accept.inputs (≥1)` fires first regardless.
            .replace("op_kind: RandUniform\n", "op_kind: ReluElementwise\n");
        let err = validate_str(&src).expect_err("registrable zero-input section is rejected");
        assert!(
            matches!(err, FkcError::MissingRequiredField { ref field, .. } if field == "accept.inputs (≥1)"),
            "got {err:?}"
        );
    }

    // =====================================================================
    // Rule 14 — gather (paged_blocks) coherence + the IMPORT-SIDE LIFT
    // (FDX gather-sidecar arc, slice A).
    // =====================================================================

    /// A minimal but COHERENT `registrable: true` paged_blocks bundle: the
    /// PagedAttn-shaped operand carries `fdx.gather: paged_blocks` naming the
    /// separate `block_table` / `context_lens` U32 input roles (the as-built
    /// ABI). `__GATHER__` marks the pool operand's `fdx` block so the negative
    /// tests can mutate it into an incoherent form.
    fn paged_blocks_bundle() -> String {
        r#"---
fkc_version: 1
provider:
  name: test-provider
  backend: Cpu
  kernel_source: "test-cpu"
---

# paged bundle

## paged_demo

A paged attention blurb.

```fkc
kernel: paged_demo
op_kind: PagedAttn
blurb: "naive paged attention over a blocked KV cache"
entry_point: "x::paged_demo"
accept:
  inputs:
    - name: q
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
    - name: k_cache
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      fdx:
        requires_ext: true
        symbolic_extent: required
        gather: { kind: paged_blocks, block_table: block_table, context_lens: context_lens }
    - name: block_table
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
    - name: context_lens
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
  op_params: { variant: PagedAttn }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
cost:
  provenance: declared
  class: attention
  flops: "4 * b * hq * sq * ctx * d"
  bytes_moved: "2 * b * hq * sq * d * 4"
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```
"#
        .to_string()
    }

    #[test]
    fn coherent_registrable_paged_blocks_validates() {
        // BORN-RED → GREEN (slice A). Before the import-side lift, a REGISTRABLE
        // section whose operand carries `fdx.gather: paged_blocks` failed import
        // with `GatherNotYetSupported` (validate_file re-raises it for a
        // registrable section — only describe-only ones are swallowed). After the
        // lift, the gather block is validated for COHERENCE (kind / requires_ext /
        // symbolic_extent / real block_table+context_lens roles) and then ACCEPTED
        // — registration of `OpKind::PagedAttn` does not depend on the FDX gather
        // descriptor (the block_table/context_lens ride as ordinary U32 operands).
        validate_str(&paged_blocks_bundle())
            .expect("a coherent registrable paged_blocks section validates + is registrable");
    }

    #[test]
    fn incoherent_gather_missing_requires_ext_still_errors() {
        // GUARD: the lift must NOT be a blanket pass. An INCOHERENT gather (here:
        // no `requires_ext: true`) still fails with the typed `GatherIncoherent`,
        // never a silent pass (never-panic, no-silent-fixup).
        let src = paged_blocks_bundle().replace("        requires_ext: true\n", "");
        let err = validate_str(&src).expect_err("gather without requires_ext must error");
        assert!(
            matches!(err, FkcError::GatherIncoherent { ref reason, .. } if reason.contains("requires_ext")),
            "got {err:?}",
        );
    }

    #[test]
    fn incoherent_gather_dangling_block_table_role_still_errors() {
        // GUARD: `gather.block_table` naming a role that is not a real
        // `accept.inputs` name is `GatherIncoherent` — the coherence cross-check
        // (a table is described in exactly one place + a consistency check, FDX
        // §6.9.3) is preserved by the lift.
        let src = paged_blocks_bundle().replace("block_table: block_table,", "block_table: nope,");
        let err = validate_str(&src).expect_err("dangling block_table role must error");
        assert!(
            matches!(err, FkcError::GatherIncoherent { ref reason, .. } if reason.contains("block_table")),
            "got {err:?}",
        );
    }

    #[test]
    fn unknown_gather_kind_is_unknown_admissibility_enum() {
        // GUARD: an unknown `gather.kind` (a future ragged/CSR kind FKC cannot yet
        // key) is a typed `UnknownAdmissibilityEnum`, never a silent default.
        let src = paged_blocks_bundle().replace("kind: paged_blocks,", "kind: ragged_csr,");
        let err = validate_str(&src).expect_err("unknown gather kind must error");
        assert!(
            matches!(err, FkcError::UnknownAdmissibilityEnum { ref value, .. } if value == "ragged_csr"),
            "got {err:?}",
        );
    }

    // =====================================================================
    // TASK 3 — the CI lint over the whole checked-in contract corpus.
    // =====================================================================

    /// A permissive stub `LinkRegistry` that resolves any entry_point to a
    /// distinct dummy `KernelRef` (so lowering succeeds). The lint checks
    /// parse + lower + validate, NOT real registration.
    struct StubLink {
        seen: std::sync::Mutex<std::collections::HashMap<String, crate::kernel::KernelRef>>,
    }
    impl StubLink {
        fn new() -> Self {
            Self { seen: std::sync::Mutex::new(std::collections::HashMap::new()) }
        }
        fn resolve(&self, symbol: &str) -> Option<crate::kernel::KernelRef> {
            use std::sync::Arc;
            use std::sync::RwLock;
            // A family of distinct fn items so distinct symbols get distinct
            // pointers (avoids spurious duplicate-pointer collapse on lower —
            // lowering does not finalize, but keep them distinct anyway).
            fn k0(_i: &[Arc<RwLock<fuel_memory::Storage>>], _o: &mut [Arc<RwLock<fuel_memory::Storage>>], _l: &[fuel_ir::Layout], _p: &crate::kernel::OpParams) -> fuel_ir::Result<()> { Ok(()) }
            fn k1(_i: &[Arc<RwLock<fuel_memory::Storage>>], _o: &mut [Arc<RwLock<fuel_memory::Storage>>], _l: &[fuel_ir::Layout], _p: &crate::kernel::OpParams) -> fuel_ir::Result<()> { Ok(()) }
            fn k2(_i: &[Arc<RwLock<fuel_memory::Storage>>], _o: &mut [Arc<RwLock<fuel_memory::Storage>>], _l: &[fuel_ir::Layout], _p: &crate::kernel::OpParams) -> fuel_ir::Result<()> { Ok(()) }
            fn k3(_i: &[Arc<RwLock<fuel_memory::Storage>>], _o: &mut [Arc<RwLock<fuel_memory::Storage>>], _l: &[fuel_ir::Layout], _p: &crate::kernel::OpParams) -> fuel_ir::Result<()> { Ok(()) }
            let table: [crate::kernel::KernelRef; 4] = [k0, k1, k2, k3];
            let mut g = self.seen.lock().unwrap();
            if let Some(k) = g.get(symbol) {
                return Some(*k);
            }
            let k = table[g.len() % table.len()];
            g.insert(symbol.to_string(), k);
            Some(k)
        }
    }
    impl crate::fkc::lower::LinkRegistry for StubLink {
        fn resolve_primitive(&self, s: &str) -> Option<crate::kernel::KernelRef> { self.resolve(s) }
        fn resolve_fused(&self, s: &str) -> Option<crate::kernel::KernelRef> { self.resolve(s) }
        fn resolve_cost_fn(&self, _name: &str) -> Option<crate::kernel::CostFn> {
            // Permissive (§4.4 cost-fn trampoline): the corpus lint checks that a
            // NAMED `cost.cost_fn` parses + lowers, not that it resolves to a
            // specific production fn (that is each provider's real link registry's
            // job). Any name resolves to a dummy CostFn so a cost-fn-pinning
            // contract lowers cleanly.
            fn c(
                _s: &[fuel_ir::Shape],
                _d: &[fuel_ir::DType],
                _p: &crate::kernel::OpParams,
                _c: &fuel_ir::backend::BackendCapabilities,
            ) -> crate::fused::CostEstimate {
                crate::fused::CostEstimate::default()
            }
            Some(c)
        }
    }

    /// Recursively collect every `*.fkc.md` under `dir`, excluding any path
    /// containing an `_inventory` component.
    fn collect_fkc_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().and_then(|n| n.to_str()) == Some("_inventory") {
                    continue;
                }
                collect_fkc_files(&path, out);
            } else if path.to_string_lossy().ends_with(".fkc.md") {
                out.push(path);
            }
        }
    }

    /// THE CI LINT (Task 3). Walk `docs/kernel-contracts/**/*.fkc.md`
    /// (excluding `_inventory/`), parse + lower + validate EVERY file with a
    /// permissive stub LinkRegistry, and report the file/section counts. A
    /// failing contract is CAPTURED and REPORTED (file + kernel + the rule it
    /// tripped), never silently relaxed.
    ///
    /// `MxNotYetRegistrable` / `GatherNotYetSupported` are
    /// describable-but-not-yet-registrable per spec §6/§3.9.1 — they are a
    /// CORRECT validate outcome (the contract is legal, the consumer is
    /// behind), so the lint records them separately as "deferred", not as
    /// hard failures. `FanoutDtypeMismatch` (§3.4) is the same posture at
    /// lower time: a MULTI-AXIS dtype contract (varying operands with
    /// DIFFERENT dtype lists) is legal but not-yet-fannable by the uniform
    /// fan-out importer — surfaced (never silently picked), recorded as
    /// deferred, and awaiting a cartesian / per-axis fan-out follow-up.
    ///
    /// This is a **real CI gate**: it runs as a normal unit test (no
    /// `#[ignore]`) and FAILS the build if any checked-in contract has a hard
    /// parse/lower/validate failure. The whole corpus is currently clean of hard
    /// failures; only the `MxNotYetRegistrable` / `GatherNotYetSupported`
    /// deferred (consumer-ahead) cases remain, which are a CORRECT outcome and
    /// are NOT counted as failures. Run it verbosely with
    /// `cargo test -p fuel-dispatch --lib -- \
    ///  ci_lint_corpus_parse_lower_validate --nocapture` to see the full
    /// file/section/deferred counts. It does NOT relax a validator or edit the
    /// corpus to hide defects — a new hard failure must be fixed (or the section
    /// marked describe-only per §3.10) to make this gate green again.
    #[test]
    fn ci_lint_corpus_parse_lower_validate() {
        // Locate the corpus relative to this crate (CARGO_MANIFEST_DIR =
        // .../fuel-dispatch). The corpus lives at ../docs/kernel-contracts.
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let corpus = manifest.join("..").join("docs").join("kernel-contracts");
        assert!(corpus.is_dir(), "corpus dir not found at {}", corpus.display());

        let mut files = Vec::new();
        collect_fkc_files(&corpus, &mut files);
        files.sort();
        assert!(!files.is_empty(), "no .fkc.md files found");

        let link = StubLink::new();
        let mut file_count = 0usize;
        let mut section_count = 0usize;
        let mut hard_failures: Vec<String> = Vec::new();
        let mut deferred: Vec<String> = Vec::new();

        for path in &files {
            file_count += 1;
            let rel = path.strip_prefix(manifest).unwrap_or(path).display().to_string();
            let src = std::fs::read_to_string(path).expect("read corpus file");

            // 1) parse.
            let file = match parse_file(&src) {
                Ok(f) => f,
                Err(e) => {
                    hard_failures.push(format!("{rel}: PARSE: {e}"));
                    continue;
                }
            };
            section_count += file.kernels.len();

            // 2) validate (whole file) — but to attribute a failure to a
            // specific kernel + rule, validate each kernel individually too.
            // First the file-level (version) check.
            if file.front_matter.fkc_version > FKC_VERSION_MAX {
                hard_failures.push(format!("{rel}: version {} > {}", file.front_matter.fkc_version, FKC_VERSION_MAX));
                continue;
            }
            for kernel in &file.kernels {
                let kname = kernel.kernel.as_str();
                match validate_kernel(kernel) {
                    Ok(()) => {}
                    Err(e @ (FkcError::MxNotYetRegistrable { .. }
                    | FkcError::GatherNotYetSupported { .. })) => {
                        deferred.push(format!("{rel} :: {kname}: {e}"));
                    }
                    Err(e) => {
                        hard_failures.push(format!("{rel} :: {kname}: {e}"));
                        continue;
                    }
                }
                // 3) lower (with the stub link) — only attempt when validate
                // passed (or deferred). Lowering catches link/dtype/cost issues
                // validate doesn't; surface those too.
                if let Err(e) = lower_one_kernel(&file, kernel, &link) {
                    // MxNotYetRegistrable is a register-time gate, not lower —
                    // lowering an MX/AFFINE_BLOCK contract succeeds; any lower
                    // error here is a real finding.
                    match e {
                        FkcError::MxNotYetRegistrable { .. } | FkcError::GatherNotYetSupported { .. } => {}
                        // Describable but NOT-YET-FANNABLE (§3.4 multi-dtype
                        // fan-out): a MULTI-AXIS dtype contract whose varying
                        // operands enumerate DIFFERENT dtype lists (e.g. a
                        // mixed-precision matmul, or an indexing op with an
                        // independent data-dtype axis and index-dtype axis).
                        // The uniform fan-out importer surfaces this as a typed
                        // error rather than silently picking one operand's list
                        // (never a silent pick); registering it needs a
                        // cartesian / per-axis fan-out follow-up. Recorded as
                        // deferred (consumer-behind), not a hard failure.
                        FkcError::FanoutDtypeMismatch { .. } => {
                            deferred.push(format!("{rel} :: {kname}: LOWER: {e}"));
                        }
                        // Describable but NOT-YET-FANNABLE (§3.4 optional-operand
                        // fan-out): a section with MULTIPLE / chained trailing
                        // optional operands (e.g. Metal layernorm's `alpha` AND
                        // `beta`, so `alpha` is optional but not last). The
                        // key-builder supports a SINGLE optional LAST input
                        // (conv's `bias`); a chained-optional ABI needs a nested
                        // fan follow-up. The importer surfaces it as a typed error
                        // rather than silently mis-keying (before optional support
                        // it was silently coerced to an all-present required key).
                        // Recorded as deferred (consumer-behind), not a hard fail.
                        FkcError::OptionalOperandNotLast { .. } => {
                            deferred.push(format!("{rel} :: {kname}: LOWER: {e}"));
                        }
                        _ => hard_failures.push(format!("{rel} :: {kname}: LOWER: {e}")),
                    }
                }
            }
        }

        eprintln!(
            "FKC corpus lint: {file_count} files, {section_count} sections; {} deferred (MX/gather not-yet-registrable + multi-axis dtype / chained-optional not-yet-fannable), {} hard failures",
            deferred.len(),
            hard_failures.len()
        );
        if !deferred.is_empty() {
            eprintln!("--- deferred (describable, not-yet-registrable / not-yet-fannable; spec §6/§3.9.1/§3.4) ---");
            for d in &deferred {
                eprintln!("  {d}");
            }
        }
        if !hard_failures.is_empty() {
            eprintln!("--- HARD FAILURES (file :: kernel: rule) ---");
            for f in &hard_failures {
                eprintln!("  {f}");
            }
            panic!(
                "{} corpus contract(s) failed parse/lower/validate — see the list above \
                 (REPORT these, do not relax the validators)",
                hard_failures.len()
            );
        }
    }

    /// Lower a single kernel within a file using the stub link, mirroring what
    /// `lower_file` does per-kernel (front-matter defaults applied).
    fn lower_one_kernel(
        file: &FkcFile,
        kernel: &FkcKernel,
        link: &dyn crate::fkc::lower::LinkRegistry,
    ) -> Result<(), FkcError> {
        // Reuse the public lower_file by lowering the whole file would re-lower
        // everything; instead lower just this kernel via a one-kernel clone.
        let mut one = file.clone();
        one.kernels = vec![kernel.clone()];
        crate::fkc::lower::lower_file(&one, link, &mut Vec::new()).map(|_| ())
    }
}
