//! # FKC kernel-contract importer (`fkc`)
//!
//! The Fuel Kernel Contract (FKC) importer: parse a provider's markdown
//! contract file(s) and (eventually) auto-register every described kernel onto
//! Fuel's dispatch surface (`KernelBindingTable` / `FusedKernelRegistry`) with
//! zero hand-written registration glue.
//!
//! This module is gated behind the default-off `fkc` cargo feature
//! (`fuel-dispatch/Cargo.toml`), so plain builds are untouched while the
//! importer is WIP (kernel-contract adoption plan §1, §11).
//!
//! ## Status — first slice
//!
//! Implemented here: the module skeleton, the [`FkcError`] type, the serde
//! schema structs mirroring FKC §3.1 / §3.2 / §3.3, and the markdown +
//! restricted-YAML parser ([`parse_file`] / [`parse_path`]) that:
//! - splits the file-level `---` front-matter (§3.1),
//! - extracts each `## ` section's single ` ```fkc ` block (§3.1),
//! - enforces the §3.8 restricted YAML subset (tabs / anchors / aliases /
//!   merge keys / the Norway problem) BEFORE deserializing, then
//! - deserializes each block into the schema and assembles an [`FkcFile`].
//!
//! NOT yet implemented (later slices of the plan): lowering to dispatch types
//! (`lower.rs`), caps/cost/precision projection, the `LinkRegistry`, the
//! `register_into` path, the revision hash, and the full `V-FKC-*` validators.
//!
//! ## Authoritative references
//!
//! - Format spec: `docs/specs/kernel-contract-format.md`.
//! - Adoption plan: `docs/session-prompts/kernel-contract-adoption-plan.md`.
//! - Authored corpus (valid instances): `docs/kernel-contracts/**/*.fkc.md`.

mod caps_map;
mod cost_expr;
mod error;
mod lower;
mod parse;
mod precision;
mod revhash;
mod schema;

pub use caps_map::{ResolvedLayout, Tri};
pub use cost_expr::{eval as eval_cost, CompiledCostExpr, CostEvalError, CostNode};
pub use error::FkcError;
pub use lower::{
    lower_file, LinkRegistry, Resolved, ResolvedFused, ResolvedPrimitive,
};
pub use parse::{parse_file, parse_path};
pub use revhash::compute_revision;
pub use schema::{
    AcceptBlock, CapsBlock, CostBlock, CostMemory, FdxSpec, FkcFile, FkcFrontMatter, FkcKernel,
    FkcProvider, GatherSpec, LayoutSpec, OpParamFieldSpec, OpParamsSchema, OutputDesc,
    PrecisionBlock, QuantSpec, ReturnBlock, TensorDesc,
};

#[cfg(test)]
mod tests {
    use super::*;

    // ----- Real-contract corpus paths (load-bearing schema-match test) -----

    /// `elementwise-binary.fkc.md` — the simplest authored CPU contract.
    const ELEMENTWISE_BINARY: &str = include_str!(
        "../../../docs/kernel-contracts/cpu/elementwise-binary.fkc.md"
    );
    /// `quant-matmul.fkc.md` — a more complex authored CPU contract (GGML quant
    /// weight matmuls + NF4 with a separate scale operand, gemm cost class,
    /// fdx.quant blocks).
    const QUANT_MATMUL: &str =
        include_str!("../../../docs/kernel-contracts/cpu/quant-matmul.fkc.md");

    // =====================================================================
    // PARSE A REAL CONTRACT — the key correctness test (§3.1/§3.3 schema match)
    // =====================================================================

    #[test]
    fn parses_real_elementwise_binary_contract() {
        let file = parse_file(ELEMENTWISE_BINARY)
            .expect("authored elementwise-binary.fkc.md must parse");

        // Front-matter.
        assert_eq!(file.front_matter.fkc_version, 1);
        assert_eq!(file.front_matter.provider.name, "fuel-cpu-backend");
        assert_eq!(file.front_matter.provider.backend, "Cpu");
        assert_eq!(file.front_matter.provider.kernel_source, "portable-cpu");
        assert_eq!(
            file.front_matter.provider.link_registry.as_deref(),
            Some("fuel_cpu_backend::fkc::ENTRY_POINTS")
        );

        // Non-empty kernels.
        assert!(
            !file.kernels.is_empty(),
            "elementwise-binary must yield kernels"
        );

        // The umbrella `binary` chassis section + per-(op, dtype) thunks.
        let names: Vec<&str> = file.kernels.iter().map(|k| k.kernel.as_str()).collect();
        assert!(names.contains(&"binary"), "expected the `binary` chassis section; got {names:?}");
        assert!(names.contains(&"add_f32"), "expected `add_f32`; got {names:?}");

        // op_kind round-trips as a string; fused_op absent for these primitives.
        let add_f32 = file
            .kernels
            .iter()
            .find(|k| k.kernel == "add_f32")
            .expect("add_f32 present");
        assert_eq!(add_f32.op_kind.as_deref(), Some("AddElementwise"));
        assert!(add_f32.fused_op.is_none());

        // Accept block: two inputs (lhs, rhs), dtypes parsed as strings.
        let accept = add_f32.accept.as_ref().expect("add_f32 has accept");
        assert_eq!(accept.inputs.len(), 2);
        assert_eq!(accept.inputs[0].name.as_deref(), Some("lhs"));
        assert_eq!(accept.inputs[0].dtypes, vec!["F32".to_string()]);
        // Layout five-flag set parsed.
        let layout = accept.inputs[0].layout.as_ref().expect("lhs has layout");
        assert_eq!(layout.contiguous.as_deref(), Some("required"));
        assert_eq!(layout.strided.as_deref(), Some("rejected"));

        // Return block: one output with rules carried as strings.
        let ret = add_f32.return_.as_ref().expect("add_f32 has return");
        assert_eq!(ret.outputs.len(), 1);
        assert_eq!(ret.outputs[0].dtype_rule.as_deref(), Some("passthrough(lhs)"));

        // Cost: provenance + expression strings carried verbatim.
        let cost = add_f32.cost.as_ref().expect("add_f32 has cost");
        assert_eq!(cost.provenance.as_deref(), Some("judge_measured"));
        assert_eq!(cost.flops.as_deref(), Some("n"));
        assert_eq!(cost.bytes_moved.as_deref(), Some("3 * n * 4"));

        // Determinism token.
        assert_eq!(add_f32.determinism.as_deref(), Some("same_hardware_bitwise"));
    }

    #[test]
    fn parses_real_quant_matmul_contract() {
        let file =
            parse_file(QUANT_MATMUL).expect("authored quant-matmul.fkc.md must parse");

        assert_eq!(file.front_matter.provider.name, "fuel-cpu-backend");
        assert!(!file.kernels.is_empty());

        let names: Vec<&str> = file.kernels.iter().map(|k| k.kernel.as_str()).collect();
        assert!(
            names.contains(&"qmatmul_q4_0_f32"),
            "expected qmatmul_q4_0_f32; got {names:?}"
        );
        assert!(
            names.contains(&"nf4_matmul_f32"),
            "expected nf4_matmul_f32; got {names:?}"
        );

        // A GGML quant weight operand: family/ggml_dtype parsed as strings,
        // scale_operand stays None (INLINE single-place rule).
        let q40 = file
            .kernels
            .iter()
            .find(|k| k.kernel == "qmatmul_q4_0_f32")
            .unwrap();
        assert_eq!(q40.op_kind.as_deref(), Some("QMatMul"));
        let weight = q40
            .accept
            .as_ref()
            .unwrap()
            .inputs
            .iter()
            .find(|d| d.name.as_deref() == Some("weight"))
            .expect("q4_0 has a weight operand");
        let quant = weight
            .fdx
            .as_ref()
            .unwrap()
            .quant
            .as_ref()
            .expect("weight has fdx.quant");
        assert_eq!(quant.family.as_deref(), Some("GGML_BLOCK"));
        assert_eq!(quant.ggml_dtype.as_deref(), Some("Q4_0"));
        assert_eq!(quant.role.as_deref(), Some("weight"));
        assert!(quant.scale_operand.is_none(), "INLINE scale: no separate operand");

        // op_params variant + a field with a constraint string.
        let op_params = q40.accept.as_ref().unwrap().op_params.as_ref().unwrap();
        assert_eq!(op_params.variant.as_deref(), Some("QMatMul"));
        assert!(op_params.fields.contains_key("k"));

        // NF4: the absmax is a SEPARATE scale operand (single-place rule).
        let nf4 = file
            .kernels
            .iter()
            .find(|k| k.kernel == "nf4_matmul_f32")
            .unwrap();
        let nf4_inputs = &nf4.accept.as_ref().unwrap().inputs;
        assert!(
            nf4_inputs.iter().any(|d| d.name.as_deref() == Some("absmax")),
            "NF4 has a separate absmax operand"
        );
        let w_packed = nf4_inputs
            .iter()
            .find(|d| d.name.as_deref() == Some("w_packed"))
            .unwrap();
        let nf4_quant = w_packed.fdx.as_ref().unwrap().quant.as_ref().unwrap();
        assert_eq!(nf4_quant.family.as_deref(), Some("AFFINE_BLOCK"));
        assert_eq!(nf4_quant.scale_operand.as_deref(), Some("absmax"));
    }

    // =====================================================================
    // §3.8 NEGATIVES
    // =====================================================================

    const VALID_MINIMAL: &str = "\
---
fkc_version: 1
provider:
  name: test-provider
  backend: Cpu
  kernel_source: \"test-cpu\"
---

# test bundle

## demo

A blurb.

```fkc
kernel: demo
op_kind: AddElementwise
blurb: \"demo\"
entry_point: \"x::y\"
```
";

    #[test]
    fn valid_minimal_parses() {
        let file = parse_file(VALID_MINIMAL).expect("minimal valid file parses");
        assert_eq!(file.kernels.len(), 1);
        assert_eq!(file.kernels[0].kernel, "demo");
    }

    #[test]
    fn tab_indentation_in_block_is_rejected() {
        // A tab-indented line inside the fkc block.
        let src = "\
---
fkc_version: 1
provider:
  name: p
  backend: Cpu
  kernel_source: \"c\"
---

## demo

b

```fkc
kernel: demo
accept:
\tinputs: []
```
";
        let err = parse_file(src).expect_err("tab must be rejected");
        assert!(
            matches!(err, FkcError::TabIndentation { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn anchor_is_rejected() {
        let src = "\
---
fkc_version: 1
provider:
  name: p
  backend: Cpu
  kernel_source: \"c\"
---

## demo

b

```fkc
kernel: &base demo
op_kind: AddElementwise
```
";
        let err = parse_file(src).expect_err("anchor must be rejected");
        assert!(matches!(err, FkcError::AnchorDisallowed { .. }), "got {err:?}");
    }

    #[test]
    fn alias_is_rejected() {
        let src = "\
---
fkc_version: 1
provider:
  name: p
  backend: Cpu
  kernel_source: \"c\"
---

## demo

b

```fkc
kernel: demo
blurb: *base
```
";
        let err = parse_file(src).expect_err("alias must be rejected");
        assert!(matches!(err, FkcError::AliasDisallowed { .. }), "got {err:?}");
    }

    #[test]
    fn merge_key_is_rejected() {
        let src = "\
---
fkc_version: 1
provider:
  name: p
  backend: Cpu
  kernel_source: \"c\"
---

## demo

b

```fkc
kernel: demo
caps:
  <<: defaults
```
";
        let err = parse_file(src).expect_err("merge key must be rejected");
        assert!(
            matches!(err, FkcError::MergeKeyDisallowed { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn unquoted_norway_token_is_rejected() {
        // `family: no` unquoted, in a value position — must be flagged, never
        // silently coerced to a bool.
        let src = "\
---
fkc_version: 1
provider:
  name: p
  backend: Cpu
  kernel_source: \"c\"
---

## demo

b

```fkc
kernel: demo
family: no
```
";
        let err = parse_file(src).expect_err("unquoted Norway token must be rejected");
        assert!(
            matches!(err, FkcError::NorwayToken { ref token, .. } if token == "no"),
            "got {err:?}"
        );
    }

    #[test]
    fn quoted_norway_token_is_accepted_as_string() {
        // `family: "none"` (quoted) must NOT trip the Norway check and must
        // deserialize to the string "none".
        let src = "\
---
fkc_version: 1
provider:
  name: p
  backend: Cpu
  kernel_source: \"c\"
---

## demo

b

```fkc
kernel: demo
accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      fdx:
        quant:
          family: \"none\"
```
";
        let file = parse_file(src).expect("quoted token parses");
        let fam = file.kernels[0].accept.as_ref().unwrap().inputs[0]
            .fdx
            .as_ref()
            .unwrap()
            .quant
            .as_ref()
            .unwrap()
            .family
            .as_deref();
        assert_eq!(fam, Some("none"));
    }

    // =====================================================================
    // §3.1 fenced-block anatomy
    // =====================================================================

    #[test]
    fn missing_fkc_block_is_rejected() {
        let src = "\
---
fkc_version: 1
provider:
  name: p
  backend: Cpu
  kernel_source: \"c\"
---

## demo

Just prose, no fkc block.
";
        let err = parse_file(src).expect_err("section with no fkc block must error");
        assert!(
            matches!(err, FkcError::MissingFkcBlock { ref section } if section == "demo"),
            "got {err:?}"
        );
    }

    #[test]
    fn duplicate_fkc_block_is_rejected() {
        let src = "\
---
fkc_version: 1
provider:
  name: p
  backend: Cpu
  kernel_source: \"c\"
---

## demo

b

```fkc
kernel: demo
```

more prose

```fkc
kernel: demo2
```
";
        let err = parse_file(src).expect_err("section with two fkc blocks must error");
        assert!(
            matches!(err, FkcError::MultipleFkcBlocks { ref section, count } if section == "demo" && count == 2),
            "got {err:?}"
        );
    }

    #[test]
    fn missing_front_matter_is_rejected() {
        let src = "## demo\n\n```fkc\nkernel: demo\n```\n";
        let err = parse_file(src).expect_err("file without front-matter must error");
        assert!(
            matches!(err, FkcError::MalformedFrontMatter(_)),
            "got {err:?}"
        );
    }
}
