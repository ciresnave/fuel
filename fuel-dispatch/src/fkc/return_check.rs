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
/// §5.2: `same_as(role)` yields the operand's whole shape; a `DimExpr` string
/// (`extent(role,axis)` / `const(N)` / `param(N)` / `add|sub|mul|div(a,b)`) evaluates to a
/// single-dim shape via the shape-expr oracle (§6.20). Every other token, a surfaced gap, or
/// a decline on a recognized DimExpr → `Ok(None)` (not-evaluable; never a false reject).
pub fn eval_shape_rule(rule: &str, combo: ProbeComboRef, _section: &str) -> Result<Option<Shape>, FkcError> {
    let rule = rule.trim();
    if let Some(r) = inner(rule, "same_as(") {
        return Ok(role(combo, r).map(|(_, s, _)| s.clone()));
    }
    // A DimExpr form: parse (role names → positional AST), then evaluate over the combo.
    if is_dimexpr_head(rule) {
        let Some(dim) = crate::fkc::shape_expr_parse::parse_dim(rule, combo) else { return Ok(None) };
        let operands: Vec<Vec<i64>> = combo.iter().map(|(_, s, _)| shape_to_i64(s)).collect();
        return match crate::fkc::shape_expr::eval_dim(&dim, &operands, &[]) {
            // A DimExpr denotes a single dimension → a rank-1 output shape (d ≥ 0).
            Ok(crate::fkc::shape_expr::DimValue::Concrete(d)) if d >= 0 => {
                Ok(Some(Shape::from_dims(&[d as usize])))
            }
            // Negative dim (not a valid shape), a surfaced Gap, or a decline → skip; never a false reject.
            Ok(_) | Err(_) => Ok(None),
        };
    }
    Ok(None)
}

/// Invoke a registry rule fn (`shape_rule` / `dtype_rule` / `output_views`),
/// catching a `debug_assert`/panic so a corpus contract can never CRASH the
/// importer (never-panic on the import path). Some registry fns `debug_assert`
/// a specific input arity (e.g. `flash_attn`'s "4 or 5 inputs"); a fused
/// contract elsewhere in the corpus that declares a different operand count
/// (e.g. `metal/matmul-attn.fkc.md`'s 3-input `FLASH_ATTN`) would otherwise
/// panic when the cross-check hands the real fn a probe combo of that arity.
/// A caught panic → `None` = skip this op's differential for this combo (a
/// coverage gap, never a false reject; in release the `debug_assert` is a
/// no-op and the fn just reads operand 0, which the differential would pass).
/// Mirrors the `catch_unwind` guard in `verify::harness`.
fn guard_rule<T>(f: impl FnOnce() -> T) -> Option<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).ok()
}

/// True iff `rule` starts with a recognized `DimExpr` constructor head.
fn is_dimexpr_head(rule: &str) -> bool {
    const HEADS: &[&str] = &["extent(", "const(", "param(", "add(", "sub(", "mul(", "div("];
    HEADS.iter().any(|h| rule.starts_with(h))
}

/// Widen a `fuel_ir::Shape`'s concrete `usize` extents to `i64` for the evaluator. Probe
/// shapes are concrete (no symbolic sentinel arises on this path — see the C-1 Task-4 note).
fn shape_to_i64(s: &Shape) -> Vec<i64> {
    s.dims().iter().map(|&d| d as i64).collect()
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
        // Convergence-C C-3: widen synth so the shape-oracle cross-check fires
        // for the expressible attention / affine / scan fused ops. Every variant
        // added here has a params-INDEPENDENT shape rule (`same_as`/`matmul`
        // returning `input_shapes[0/1/2]`, or a bundle whose slot-0 shape is a
        // plain passthrough), so the placeholder field values below never reach a
        // params-dependent branch and the wrong-params panic stays unreachable
        // (the never-panic invariant). Field values are arbitrary valid
        // placeholders — the shape rules ignore them.
        Some("InplaceAffine") => Some(FusedOpParams::InplaceAffine { mul: 1.0, add: 0.0 }),
        Some("FlashAttn") => Some(FusedOpParams::FlashAttn {
            softmax_scale: 1.0,
            causal: false,
            window_size_left: None,
            window_size_right: None,
            softcap: None,
            k_len: None,
        }),
        Some("PagedAttn") => Some(FusedOpParams::PagedAttn {
            softmax_scale: 1.0,
            block_size: 16,
            softcap: None,
        }),
        // FLASH_ATTN_BACKWARD_{Q,K,V} share ONE `FlashAttnBackward` variant
        // (the FusedOpId distinguishes dQ/dK/dV); a single arm covers all three.
        Some("FlashAttnBackward") => Some(FusedOpParams::FlashAttnBackward {
            softmax_scale: 1.0,
            causal: false,
            window_size_left: None,
            window_size_right: None,
            softcap: None,
        }),
        Some("SelectiveScan") => Some(FusedOpParams::SelectiveScan { delta_softplus: false }),
        Some("SsdChunkScan") => Some(FusedOpParams::SsdChunkScan { chunk_size: 1 }),
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
                // Never-panic guard: a registry fn that `debug_assert`s an arity
                // the probe combo doesn't satisfy → skip this combo (see `guard_rule`).
                let Some(real) = guard_rule(|| (entry.dtype_rule)(&in_dtypes, p)) else { continue };
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
                let Some(real) = guard_rule(|| (entry.shape_rule)(&in_shapes, p)) else { continue };
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
    //
    // Convergence-C C-3: the only bundle-bearing fused ops (SELECTIVE_SCAN,
    // SSD_CHUNK_SCAN) are now synth-supported (`synth_probe_params` returns
    // `Some`), so this bundle check is corpus-LIVE — it validates the declared
    // `return.bundle` slot count against the real `output_views` arity and
    // rank-checks each evaluable slot. `output_views` is guarded by `guard_rule`
    // (never-panic) exactly like `shape_rule`/`dtype_rule`.
    if let (Some(bundle), Some(output_views), Some(p)) =
        (ret.bundle.as_ref(), entry.output_views, params.as_ref())
    {
        if let Some(combo) = combos.first() {
            let in_shapes: Vec<Shape> = combo.iter().map(|(_, s, _)| s.clone()).collect();
            let in_dtypes: Vec<DType> = combo.iter().map(|(_, _, d)| *d).collect();
            // Never-panic guard (see `guard_rule`): an `output_views` fn that
            // asserts an arity the probe doesn't satisfy → skip the bundle checks.
            if let Some(views) = guard_rule(|| output_views(&in_shapes, &in_dtypes, p)) {
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

/// Finding 5.4 (FKC side, Task 3.6): extract a `return.bundle`'s per-slot
/// NAMES, in declared order. `Vec::new()` when `ret` is `None`, or `ret`'s
/// `bundle` is `None`/not a `Sequence`, or a slot is not a `Mapping` — never
/// a false reject; the FDX bundle-arity check (`check_bundle_arity`) is the
/// authority on WHETHER the slot count is right, this fn only reads names.
/// A slot with no `name:` key falls back to `slot{i}` (mirrors the same
/// fallback in `cross_check_fused_section`'s rank-check loop, §5.5).
pub fn bundle_slot_names(ret: &Option<crate::fkc::schema::ReturnBlock>) -> Vec<String> {
    let Some(ret) = ret.as_ref() else { return Vec::new() };
    let Some(bundle) = ret.bundle.as_ref() else { return Vec::new() };
    let serde_yml::Value::Sequence(slots) = bundle else { return Vec::new() };
    slots
        .iter()
        .enumerate()
        .map(|(i, slot)| {
            let serde_yml::Value::Mapping(map) = slot else {
                return format!("slot{i}");
            };
            map.get(serde_yml::Value::String("name".into()))
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| format!("slot{i}"))
        })
        .collect()
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
        // Convergence-C C-3: the newly-widened arms return their matching variant.
        assert!(matches!(synth_probe_params(Some("InplaceAffine")).unwrap(), Some(FusedOpParams::InplaceAffine { .. })));
        assert!(matches!(synth_probe_params(Some("FlashAttn")).unwrap(), Some(FusedOpParams::FlashAttn { .. })));
        assert!(matches!(synth_probe_params(Some("PagedAttn")).unwrap(), Some(FusedOpParams::PagedAttn { .. })));
        assert!(matches!(synth_probe_params(Some("FlashAttnBackward")).unwrap(), Some(FusedOpParams::FlashAttnBackward { .. })));
        assert!(matches!(synth_probe_params(Some("SelectiveScan")).unwrap(), Some(FusedOpParams::SelectiveScan { .. })));
        assert!(matches!(synth_probe_params(Some("SsdChunkScan")).unwrap(), Some(FusedOpParams::SsdChunkScan { .. })));
        // A still-unsupported variant + the absent case both stay None.
        assert!(synth_probe_params(Some("Conv2D")).unwrap().is_none());
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

        // Convergence-C: DimExpr rules evaluate to a single-dim shape via the §6.20 oracle.
        // combo "x" = [2, 3].
        assert_eq!(eval_shape_rule("extent(x, 0)", c, "k").unwrap(), Some(Shape::from_dims(&[2])));
        assert_eq!(eval_shape_rule("const(9)", c, "k").unwrap(), Some(Shape::from_dims(&[9])));
        // div(extent(x, last), const(2)) on x=[2,3]: floor(3/2) = 1.
        assert_eq!(eval_shape_rule("div(extent(x, last), const(2))", c, "k").unwrap(), Some(Shape::from_dims(&[1])));
        // mul(extent(x, 0)=2, const(3)=3) = 6.
        assert_eq!(eval_shape_rule("mul(extent(x, 0), const(3))", c, "k").unwrap(), Some(Shape::from_dims(&[6])));
        // Unknown role, a param rule (no params threaded), a ÷0 decline, and a negative
        // result all surface as not-evaluable → None (never a false reject).
        assert_eq!(eval_shape_rule("extent(nope, 0)", c, "k").unwrap(), None);
        assert_eq!(eval_shape_rule("param(0)", c, "k").unwrap(), None);
        assert_eq!(eval_shape_rule("div(extent(x, last), const(0))", c, "k").unwrap(), None);
        assert_eq!(eval_shape_rule("sub(const(2), extent(x, 1))", c, "k").unwrap(), None); // 2 − 3 = −1
    }

    /// Finding 5.4 deliverable (Task 3.6 coverage gap): the extractor must
    /// actually PULL the right names out of a real `ReturnBlock`, not merely
    /// avoid panicking. Mirrors the real corpus shape
    /// (`docs/kernel-contracts/fused/conv-rope.fkc.md`'s `SELECTIVE_SCAN`
    /// bundle: `y` / `last_state`). Bare `y` is a YAML-1.1 "Norway problem"
    /// candidate (`y`/`yes`/`on` historically coerce to bool `true`), but the
    /// slot names live inside a `serde_yml::Value` (untyped) tree here, and
    /// `serde_yml` 0.0.12 keeps bare `y` as the plain string `"y"` — pinned by
    /// this assertion, not just asserted in prose.
    #[test]
    fn bundle_slot_names_extracts_names_from_a_real_return_block() {
        let ret: crate::fkc::schema::ReturnBlock = serde_yml::from_str(
            "outputs: []\nbundle:\n  - { name: y, shape_rule: same_as(u) }\n  - { name: last_state, shape_rule: from_params(state) }",
        )
        .expect("parses");
        assert_eq!(
            bundle_slot_names(&Some(ret)),
            vec!["y".to_string(), "last_state".to_string()]
        );
        assert_eq!(bundle_slot_names(&None), Vec::<String>::new());
    }
}
