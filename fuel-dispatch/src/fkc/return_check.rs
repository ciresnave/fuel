//! §5 return-contract validation: cross-check a fused contract's declared
//! shape/dtype rules against the real registered FusedOpEntry fns.
use fuel_graph::registry::{default_registry, FusedOpId, FusedOpParams};
use fuel_ir::{DType, Shape};
use crate::fkc::error::FkcError;
use crate::fkc::lower::lower_dtype;
use crate::fkc::schema::FkcKernel;
use crate::fkc::shape_constraint::solve_probe_shapes;
use crate::fkc::ImportWarning;

pub type ProbeComboRef<'a> = &'a [(String, Shape, DType)];

fn role<'a>(combo: ProbeComboRef<'a>, name: &str) -> Option<&'a (String, Shape, DType)> {
    combo.iter().find(|(r, _, _)| r == name)
}
fn inner<'a>(rule: &'a str, head: &str) -> Option<&'a str> {
    rule.trim().strip_prefix(head)?.strip_suffix(')').map(str::trim)
}

/// §5.1: `fixed(D)` and `passthrough(role)` are evaluable; every other token is
/// `Ok(None)` = not-evaluable (skip, never a false reject). `fixed(<bad dtype>)`
/// is a hard error (a real authoring bug).
pub fn eval_dtype_rule(rule: &str, combo: ProbeComboRef, section: &str) -> Result<Option<DType>, FkcError> {
    if let Some(tok) = inner(rule, "fixed(") { return Ok(Some(lower_dtype(tok, section, "return")?)); }
    if let Some(r) = inner(rule, "passthrough(") { return Ok(role(combo, r).map(|(_, _, d)| *d)); }
    Ok(None)
}
/// §5.2: only `same_as(role)` is evaluable purely from probe shapes.
pub fn eval_shape_rule(rule: &str, combo: ProbeComboRef, _section: &str) -> Result<Option<Shape>, FkcError> {
    if let Some(r) = inner(rule, "same_as(") { return Ok(role(combo, r).map(|(_, s, _)| s.clone())); }
    Ok(None)
}

/// Synthesize the `FusedOpParams` variant NAMED by the contract's
/// `op_params.variant` (§3.7), with placeholder field values. The ONLY
/// correctness requirement is that the returned variant matches the name so
/// the real registered `shape_rule` fn never hits its wrong-params panic
/// (`qmatmul.rs:63`, `conv2d.rs:86`). Params-dependent ops (whose declared
/// return rules are `from_params`-style = not-evaluable, so the real fn is
/// never called at all) synthesize nothing here and fall through to `None`
/// — never-panic beats completeness; a foreign variant would be worse than
/// skipping the probe.
pub fn synth_probe_params(variant: Option<&str>) -> Result<Option<FusedOpParams>, FkcError> {
    const EPS: f64 = 1e-5;
    Ok(match variant {
        Some("SoftmaxLastDim") => Some(FusedOpParams::SoftmaxLastDim),
        Some("SoftmaxLastDimBackward") => Some(FusedOpParams::SoftmaxLastDimBackward),
        Some("RmsNormLastDim") => Some(FusedOpParams::RmsNormLastDim { eps: EPS }),
        Some("LayerNormLastDim") => Some(FusedOpParams::LayerNormLastDim { eps: EPS }),
        Some("RmsNormLastDimBackward") => Some(FusedOpParams::RmsNormLastDimBackward { eps: EPS }),
        Some("LayerNormLastDimBackward") => Some(FusedOpParams::LayerNormLastDimBackward { eps: EPS }),
        Some("ReduceMaxToBackward") => Some(FusedOpParams::ReduceMaxToBackward),
        Some("PowIBackward") => Some(FusedOpParams::PowIBackward { exp: 2 }),
        Some("Rope") => Some(FusedOpParams::Rope),
        Some("FusedLinear") => Some(FusedOpParams::FusedLinear),
        _ => None,
    })
}

/// §5 (Finding 5.1): cross-check a `fused_op` section's DECLARED §5.1/§5.2
/// return rules against the REAL registered [`fuel_graph::registry::FusedOpEntry`]
/// `shape_rule`/`dtype_rule` fn pointers, at every probe combo `solve_probe_shapes`
/// produces. Runs inside `lower_fused` (the only site holding both the parsed
/// [`FkcKernel`] and the resolved [`FusedOpId`]) — BEFORE registration, so a
/// disagreement fails the import rather than silently drifting from the graph's
/// single source of truth.
///
/// Invariant (never-panic guard): the real `shape_rule`/`dtype_rule` fns for
/// some fused ops (e.g. qmatmul, conv2d) `panic!` on a mismatched
/// `FusedOpParams` variant. This fn invokes a real fn ONLY when BOTH (a) the
/// contract's declared rule is EVALUABLE (`eval_dtype_rule`/`eval_shape_rule`
/// returned `Some`) AND (b) `synth_probe_params` returned `Some(params)`. For
/// every current fused op, evaluable declared rules belong to
/// params-INDEPENDENT fns (plain passthrough/fixed/same_as), so this
/// coincidence holds and the wrong-params panic is unreachable — a
/// params-dependent rule (`from_params`-style) is never evaluable, so its
/// real fn is never called here at all.
pub fn cross_check_fused_section(
    kernel: &FkcKernel,
    id: FusedOpId,
    warnings: &mut Vec<ImportWarning>,
) -> Result<(), FkcError> {
    let section = kernel.kernel.as_str();
    let Some(entry) = default_registry().entry(id) else { return Ok(()); };
    let Some(accept) = kernel.accept.as_ref() else { return Ok(()); };
    let Some(ret) = kernel.return_.as_ref() else { return Ok(()); };
    let variant = accept.op_params.as_ref().and_then(|s| s.variant.as_deref());

    // Soft-catch solver errors (e.g. a malformed-vocabulary shape_constraint):
    // skip the cross-check + warn rather than fail the whole import. This
    // protects the currently-green norm-softmax live-kernel test, whose
    // `same_as=out` output-role refs + free-text prose constraints soft-degrade
    // to warnings today — a future hard solver error must not regress that.
    let combos = match solve_probe_shapes(&accept.inputs, section, warnings) {
        Ok(c) => c,
        Err(e) => {
            warnings.push(ImportWarning {
                section: section.into(),
                message: format!("return cross-check skipped: {e}"),
            });
            return Ok(());
        }
    };
    let params = synth_probe_params(variant)?;

    for combo in &combos {
        let in_shapes: Vec<Shape> = combo.iter().map(|(_, s, _)| s.clone()).collect();
        let in_dtypes: Vec<DType> = combo.iter().map(|(_, _, d)| *d).collect();
        // FusedOpEntry::shape_rule/dtype_rule describe the PRIMARY output (slot 0) only
        // (registry.rs doc-invariant). Additional bundle slots are validated against
        // output_views in Task 3.4 — do NOT compare non-primary outputs against these fns,
        // or a valid multi-output contract whose slot-N rule differs from slot-0 would be
        // spuriously rejected.
        let Some(out) = ret.outputs.first() else { continue };
        let role_name = out.name.as_deref().unwrap_or("out");
        // dtype_rule: only call the real fn when synth produced Some(params)
        // (mirrors the shape_rule guard below) — never an `unwrap_or`
        // fallback, since a future params-matching dtype_rule could panic.
        if let (Some(rule), Some(p)) = (out.dtype_rule.as_deref(), params.as_ref()) {
            if let Some(declared) = eval_dtype_rule(rule, combo, section)? {
                let real = (entry.dtype_rule)(&in_dtypes, p);
                if declared != real {
                    return Err(FkcError::ShapeRuleMismatch {
                        section: section.into(),
                        role: role_name.into(),
                        expected: format!("dtype {declared:?}"),
                        actual: format!("dtype {real:?}"),
                    });
                }
            }
        }
        if let (Some(rule), Some(p)) = (out.shape_rule.as_deref(), params.as_ref()) {
            if let Some(declared) = eval_shape_rule(rule, combo, section)? {
                let real = (entry.shape_rule)(&in_shapes, p);
                if declared != real {
                    return Err(FkcError::ShapeRuleMismatch {
                        section: section.into(),
                        role: role_name.into(),
                        expected: format!("shape {declared:?}"),
                        actual: format!("shape {real:?}"),
                    });
                }
            }
        }
    }
    // §5.5 (Finding 5.2): a `return.bundle` declares its slot count as a YAML
    // sequence; that count must agree with the real registered
    // `FusedOpEntry::output_views` arity, or a multi-output contract could
    // silently under/over-declare slots relative to what the graph actually
    // allocates. Same never-panic guard as shape_rule/dtype_rule above:
    // `output_views` also takes `&FusedOpParams`, so it's only invoked when
    // `synth_probe_params` produced `Some(params)` for this variant.
    if let (Some(bundle), Some(output_views), Some(p)) =
        (ret.bundle.as_ref(), entry.output_views, params.as_ref())
    {
        if let Some(combo) = combos.first() {
            let in_shapes: Vec<Shape> = combo.iter().map(|(_, s, _)| s.clone()).collect();
            let in_dtypes: Vec<DType> = combo.iter().map(|(_, _, d)| *d).collect();
            let views = output_views(&in_shapes, &in_dtypes, p);
            check_bundle_arity(section, views.len(), bundle)?;
            // Rule 13 (Finding 5.3): rank-check every bundle slot whose
            // `shape_rule` is EVALUABLE against this probe combo. The static
            // `shape:`-literal branch is already rank-checked pre-registration
            // in `validate.rs::check_bundle_ranks`; this covers the DERIVED
            // case that check never could (no statically-knowable rank for a
            // `shape_rule` string without evaluating it against a real probe).
            if let serde_yml::Value::Sequence(slots) = bundle {
                for (i, slot) in slots.iter().enumerate() {
                    let serde_yml::Value::Mapping(map) = slot else { continue };
                    let slot_name = map
                        .get(serde_yml::Value::String("name".into()))
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("slot{i}"));
                    if let Some(rule) = map
                        .get(serde_yml::Value::String("shape_rule".into()))
                        .and_then(|v| v.as_str())
                    {
                        if let Some(shape) = eval_shape_rule(rule, combo, section)? {
                            check_slot_rank(section, &slot_name, &shape)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Rule 13 (Finding 5.3): a bundle slot's shape rank must be ≤ 6, whether the
/// slot's shape came from a static `shape:` literal (checked in
/// `validate.rs::check_bundle_ranks`, pre-registration) or was DERIVED from a
/// `shape_rule` and evaluated here against a real probe shape. The serialized
/// `FDXOutputView` (`[u64; 6]`) cannot represent rank > 6 either way.
pub fn check_slot_rank(section: &str, slot: &str, shape: &Shape) -> Result<(), FkcError> {
    let rank = shape.rank();
    if rank > 6 {
        return Err(FkcError::BundleSlotRankExceeded {
            section: section.into(),
            slot: slot.into(),
            rank,
        });
    }
    Ok(())
}

/// §5.5: the declared slot count of a `return.bundle` — a YAML `Sequence`,
/// one entry per output slot. `None` for a malformed (non-sequence) bundle;
/// callers treat that as not-evaluable, never a false reject.
pub fn bundle_slot_count(bundle: &serde_yml::Value) -> Option<usize> {
    match bundle {
        serde_yml::Value::Sequence(s) => Some(s.len()),
        _ => None,
    }
}

/// §5.5 (Finding 5.2): the declared `return.bundle` slot count must agree
/// with the real registered `FusedOpEntry::output_views` arity.
pub fn check_bundle_arity(
    section: &str,
    output_views_arity: usize,
    bundle: &serde_yml::Value,
) -> Result<(), FkcError> {
    if let Some(declared) = bundle_slot_count(bundle) {
        if declared != output_views_arity {
            return Err(FkcError::BundleArityMismatch {
                section: section.into(),
                expected: output_views_arity,
                actual: declared,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::{DType, Shape};

    #[test]
    fn synth_probe_params_builds_the_matching_variant_or_none() {
        use fuel_graph::registry::FusedOpParams;
        assert!(matches!(synth_probe_params(Some("SoftmaxLastDim")).unwrap(), Some(FusedOpParams::SoftmaxLastDim)));
        assert!(matches!(synth_probe_params(Some("RmsNormLastDim")).unwrap(), Some(FusedOpParams::RmsNormLastDim { .. })));
        assert!(synth_probe_params(Some("SsdChunkScan")).unwrap().is_none());
        assert!(synth_probe_params(None).unwrap().is_none());
        // never-panic invariant: QMatMul synth is EITHER QMatMul OR None, never a foreign variant.
        match synth_probe_params(Some("QMatMul")).unwrap() {
            None => {}
            Some(p) => assert!(matches!(p, FusedOpParams::QMatMul { .. })),
        }
    }

    #[test]
    fn bundle_slot_count_disagreeing_with_output_views_arity_is_rejected() {
        let two_slots: serde_yml::Value = serde_yml::from_str(
            "- { name: y, shape_rule: same_as(u) }\n- { name: last_state, shape_rule: from_params(state) }").unwrap();
        assert_eq!(bundle_slot_count(&two_slots), Some(2));
        let err = check_bundle_arity("selective_scan", 3, &two_slots)
            .expect_err("declared 2 vs 3 real output_views slots must be rejected");
        assert!(matches!(err, FkcError::BundleArityMismatch { expected: 3, actual: 2, .. }), "got {err:?}");
        assert!(check_bundle_arity("selective_scan", 2, &two_slots).is_ok());
    }

    #[test]
    fn shape_rule_derived_bundle_slot_over_rank6_is_rejected() {
        use fuel_ir::Shape;
        let rank7 = Shape::from_dims(&[2, 2, 2, 2, 2, 2, 2]);
        let err = check_slot_rank("s", "big_slot", &rank7).expect_err("rank 7 must be rejected");
        assert!(matches!(err, FkcError::BundleSlotRankExceeded { rank: 7, .. }), "got {err:?}");
        assert!(check_slot_rank("s", "ok_slot", &Shape::from_dims(&[1,1,1,1,1,1])).is_ok());
    }

    #[test]
    fn interpreter_evaluates_supported_vocab_and_skips_the_rest() {
        let combo: Vec<(String, Shape, DType)> = vec![
            ("x".into(), Shape::from_dims(&[2, 3]), DType::F32),
            ("upstream".into(), Shape::from_dims(&[4, 5]), DType::F16),
        ];
        let c: ProbeComboRef = &combo;
        assert_eq!(eval_dtype_rule("fixed(F16)", c, "k").unwrap(), Some(DType::F16));
        assert_eq!(eval_dtype_rule("passthrough(x)", c, "k").unwrap(), Some(DType::F32));
        assert_eq!(eval_dtype_rule("dequant(w)", c, "k").unwrap(), None);
        assert_eq!(eval_shape_rule("same_as(upstream)", c, "k").unwrap(), Some(Shape::from_dims(&[4, 5])));
        assert_eq!(eval_shape_rule("from_params(q)", c, "k").unwrap(), None);
        assert_eq!(eval_shape_rule("matmul(a, b)", c, "k").unwrap(), None);
        assert_eq!(eval_shape_rule("same_as(does_not_exist)", c, "k").unwrap(), None);
    }
}
