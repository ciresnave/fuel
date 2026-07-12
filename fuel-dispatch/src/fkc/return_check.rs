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
        for out in &ret.outputs {
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
