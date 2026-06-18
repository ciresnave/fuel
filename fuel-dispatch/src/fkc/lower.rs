//! Lowering: parsed FKC schema (strings) → typed dispatch records
//! (adoption plan §2, §2.1 / §2.2 mapping tables, §2.3 cost AST, §6 caps).
//!
//! This slice converts a parsed [`FkcKernel`] into a [`ResolvedPrimitive`]
//! or [`ResolvedFused`] — typed records ready for the NEXT slice (the
//! trampoline / global cost-table / `register_into`). It stops at typed
//! records + a **parsed** cost AST; it does NOT build a `CostFn`
//! fn-pointer.
//!
//! Every string → typed conversion is an **explicit, exhaustive `match`**
//! (NOT `FromStr`-by-discriminant) so that adding a new source variant
//! (a new `OpKind`, a new `DType`) forces a compile error here to extend
//! the table — the table cannot silently drift behind the type. The one
//! exception is `fused_op`, whose name table is **generated** from
//! `fuel-graph`'s `default_registry()` so it cannot drift from the graph's
//! single source of truth.

use fuel_core_types::dispatch::OpKind;
use fuel_core_types::probe::BackendId;
use fuel_core_types::DType;
use fuel_graph::registry::{FusedOpId, FusedOps};
use smallvec::SmallVec;

use crate::fkc::caps_map::{self, ResolvedLayout};
use crate::fkc::cost_expr::{self, CompiledCostExpr};
use crate::fkc::error::FkcError;
use crate::fkc::precision;
use crate::fkc::revhash;
use crate::fkc::schema::{CostBlock, FkcKernel};
use crate::fused::{KernelRevisionHash, PrecisionGuarantee};
use crate::kernel::{KernelCaps, KernelDTypes, KernelRef};

// ===========================================================================
// LinkRegistry — entry_point symbol → KernelRef (P9, §12.6)
// ===========================================================================

/// Resolve a contract's `entry_point` symbol string into a concrete
/// [`KernelRef`] function pointer. Each provider crate implements this
/// over its exported `&'static [(&str, KernelRef)]` table (FKC §12.6); a
/// test stub maps every symbol to a dummy kernel.
///
/// The importer never fabricates a function pointer — it looks the symbol
/// up and errors ([`FkcError::UnknownEntryPoint`]) if absent. This keeps
/// P9 (serializable contracts, no pointers in the file) honest.
pub trait LinkRegistry {
    /// Resolve a primitive (`op_kind`) kernel's entry point.
    fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef>;
    /// Resolve a fused (`fused_op`) kernel's entry point.
    fn resolve_fused(&self, symbol: &str) -> Option<KernelRef>;
}

// ===========================================================================
// Resolved records
// ===========================================================================

/// A fully-lowered primitive (`op_kind`) contract — typed dispatch record
/// ready for the register slice. The cost is held as a parsed AST
/// ([`CompiledCostExpr`]), NOT a `CostFn` fn-pointer (the trampoline is the
/// next slice).
#[derive(Debug, Clone)]
pub struct ResolvedPrimitive {
    /// The dispatch op (key.0).
    pub op: OpKind,
    /// Per-operand dtypes, inputs in order then outputs (key.1).
    pub dtypes: KernelDTypes,
    /// The backend (key.2).
    pub backend: BackendId,
    /// Projected layout capabilities (§6) — one bool today, every parsed
    /// flag retained on `layouts`.
    pub caps: KernelCaps,
    /// Per-operand parsed five-flag layout sets (retained for forward use;
    /// §6 [consumer-ahead]).
    pub layouts: Vec<ResolvedLayout>,
    /// Lowered precision guarantee.
    pub precision: PrecisionGuarantee,
    /// Parsed cost AST (capacity-eval next slice; `Unknown` ⇒ `unknown_cost`).
    pub cost: CompiledCostExpr,
    /// The resolved kernel function pointer.
    pub kernel: KernelRef,
    /// The `kernel_source` tag (`BindingEntry.kernel_source`).
    pub kernel_source: String,
    /// The kernel revision hash (§4.7).
    pub revision: KernelRevisionHash,
}

/// A fully-lowered fused (`fused_op`) contract — analogous to
/// [`ResolvedPrimitive`] but targeting the `FusedKernelRegistry`
/// (`FusedOpId` instead of `OpKind`; the fused cost target).
#[derive(Debug, Clone)]
pub struct ResolvedFused {
    /// The fused op id (registry key, from `default_registry()`).
    pub id: FusedOpId,
    /// Per-operand dtypes, inputs in order then outputs.
    pub dtypes: KernelDTypes,
    /// The backend (registry key).
    pub backend: BackendId,
    /// Projected layout capabilities (§6).
    pub caps: KernelCaps,
    /// Per-operand parsed five-flag layout sets (retained; §6).
    pub layouts: Vec<ResolvedLayout>,
    /// Lowered precision guarantee.
    pub precision: PrecisionGuarantee,
    /// Parsed cost AST (the fused cost target; `Unknown` ⇒ `unknown_cost`).
    pub cost: CompiledCostExpr,
    /// The resolved kernel function pointer.
    pub kernel: KernelRef,
    /// The `kernel_source` tag.
    pub kernel_source: String,
    /// The kernel revision hash (§4.7; `BackendImpl.revision`).
    pub revision: KernelRevisionHash,
}

/// The result of lowering one kernel section: a primitive xor a fused
/// record (exactly one of `op_kind` / `fused_op`).
#[derive(Debug, Clone)]
pub enum Resolved {
    /// An `op_kind` contract → the binding table.
    Primitive(ResolvedPrimitive),
    /// A `fused_op` contract → the fused registry.
    Fused(ResolvedFused),
}

// ===========================================================================
// op_kind String → OpKind (explicit exhaustive match; §2.1)
// ===========================================================================

/// Map an `op_kind` string to an [`OpKind`]. The `match` is exhaustive
/// over `OpKind` so adding a new variant forces this table to grow
/// (a compile error, not a silent miss). `UnknownOpKind` on a bad string.
pub(crate) fn lower_op_kind(s: &str, section: &str) -> Result<OpKind, FkcError> {
    // Helper so the match arms read as `name => Variant`. We round-trip
    // through a known-exhaustive coverage check below.
    let mapped = match s {
        "MatMul" => Some(OpKind::MatMul),
        "AddElementwise" => Some(OpKind::AddElementwise),
        "SubElementwise" => Some(OpKind::SubElementwise),
        "MulElementwise" => Some(OpKind::MulElementwise),
        "DivElementwise" => Some(OpKind::DivElementwise),
        "ReluElementwise" => Some(OpKind::ReluElementwise),
        "NegElementwise" => Some(OpKind::NegElementwise),
        "SqrElementwise" => Some(OpKind::SqrElementwise),
        "SqrtElementwise" => Some(OpKind::SqrtElementwise),
        "RecipElementwise" => Some(OpKind::RecipElementwise),
        "AbsElementwise" => Some(OpKind::AbsElementwise),
        "TanhElementwise" => Some(OpKind::TanhElementwise),
        "ExpElementwise" => Some(OpKind::ExpElementwise),
        "LogElementwise" => Some(OpKind::LogElementwise),
        "SinElementwise" => Some(OpKind::SinElementwise),
        "CosElementwise" => Some(OpKind::CosElementwise),
        "SigmoidElementwise" => Some(OpKind::SigmoidElementwise),
        "SiluElementwise" => Some(OpKind::SiluElementwise),
        "GeluElementwise" => Some(OpKind::GeluElementwise),
        "StepElementwise" => Some(OpKind::StepElementwise),
        "SumReduce" => Some(OpKind::SumReduce),
        "MaxReduce" => Some(OpKind::MaxReduce),
        "MinReduce" => Some(OpKind::MinReduce),
        "MeanReduce" => Some(OpKind::MeanReduce),
        "Cast" => Some(OpKind::Cast),
        "Conv2D" => Some(OpKind::Conv2D),
        "ConvTranspose2D" => Some(OpKind::ConvTranspose2D),
        "ReduceSumTo" => Some(OpKind::ReduceSumTo),
        "ReduceMaxTo" => Some(OpKind::ReduceMaxTo),
        "FusedLinear" => Some(OpKind::FusedLinear),
        "FlashAttn" => Some(OpKind::FlashAttn),
        "FlashAttnBackwardQ" => Some(OpKind::FlashAttnBackwardQ),
        "FlashAttnBackwardK" => Some(OpKind::FlashAttnBackwardK),
        "FlashAttnBackwardV" => Some(OpKind::FlashAttnBackwardV),
        "PagedAttn" => Some(OpKind::PagedAttn),
        "Affine" => Some(OpKind::Affine),
        "ClampElementwise" => Some(OpKind::ClampElementwise),
        "PowIElementwise" => Some(OpKind::PowIElementwise),
        "PowIElementwiseBackward" => Some(OpKind::PowIElementwiseBackward),
        "MaximumElementwise" => Some(OpKind::MaximumElementwise),
        "MinimumElementwise" => Some(OpKind::MinimumElementwise),
        "EqualElementwise" => Some(OpKind::EqualElementwise),
        "NotEqualElementwise" => Some(OpKind::NotEqualElementwise),
        "LessElementwise" => Some(OpKind::LessElementwise),
        "LessEqualElementwise" => Some(OpKind::LessEqualElementwise),
        "GreaterElementwise" => Some(OpKind::GreaterElementwise),
        "GreaterEqualElementwise" => Some(OpKind::GreaterEqualElementwise),
        "Where" => Some(OpKind::Where),
        "FloorElementwise" => Some(OpKind::FloorElementwise),
        "CeilElementwise" => Some(OpKind::CeilElementwise),
        "RoundElementwise" => Some(OpKind::RoundElementwise),
        "SignElementwise" => Some(OpKind::SignElementwise),
        "ErfElementwise" => Some(OpKind::ErfElementwise),
        "GeluErfElementwise" => Some(OpKind::GeluErfElementwise),
        "PowElementwise" => Some(OpKind::PowElementwise),
        "RsqrtElementwise" => Some(OpKind::RsqrtElementwise),
        "RemElementwise" => Some(OpKind::RemElementwise),
        "Flip" => Some(OpKind::Flip),
        "Roll" => Some(OpKind::Roll),
        "CumSum" => Some(OpKind::CumSum),
        "Pad" => Some(OpKind::Pad),
        "Triu" => Some(OpKind::Triu),
        "Tril" => Some(OpKind::Tril),
        "LogSoftmaxLastDim" => Some(OpKind::LogSoftmaxLastDim),
        "LogSoftmaxLastDimBackward" => Some(OpKind::LogSoftmaxLastDimBackward),
        "MaskedFill" => Some(OpKind::MaskedFill),
        "PadBackward" => Some(OpKind::PadBackward),
        "Concat" => Some(OpKind::Concat),
        "SoftmaxLastDim" => Some(OpKind::SoftmaxLastDim),
        "SoftmaxLastDimBackward" => Some(OpKind::SoftmaxLastDimBackward),
        "RmsNormLastDim" => Some(OpKind::RmsNormLastDim),
        "RmsNormLastDimBackward" => Some(OpKind::RmsNormLastDimBackward),
        "LayerNormLastDim" => Some(OpKind::LayerNormLastDim),
        "LayerNormLastDimBackward" => Some(OpKind::LayerNormLastDimBackward),
        "ReduceMaxToBackward" => Some(OpKind::ReduceMaxToBackward),
        "IndexSelect" => Some(OpKind::IndexSelect),
        "Gather" => Some(OpKind::Gather),
        "Rope" => Some(OpKind::Rope),
        "IndexAdd" => Some(OpKind::IndexAdd),
        "ScatterAdd" => Some(OpKind::ScatterAdd),
        "ArgMaxDim" => Some(OpKind::ArgMaxDim),
        "ArgMinDim" => Some(OpKind::ArgMinDim),
        "QMatMul" => Some(OpKind::QMatMul),
        "WriteSlice" => Some(OpKind::WriteSlice),
        "WriteSliceRotating" => Some(OpKind::WriteSliceRotating),
        "Copy" => Some(OpKind::Copy),
        "ReluInplace" => Some(OpKind::ReluInplace),
        "SiluInplace" => Some(OpKind::SiluInplace),
        "GeluInplace" => Some(OpKind::GeluInplace),
        "TanhInplace" => Some(OpKind::TanhInplace),
        "SigmoidInplace" => Some(OpKind::SigmoidInplace),
        "NegInplace" => Some(OpKind::NegInplace),
        "AbsInplace" => Some(OpKind::AbsInplace),
        "SqrInplace" => Some(OpKind::SqrInplace),
        "SqrtInplace" => Some(OpKind::SqrtInplace),
        "RsqrtInplace" => Some(OpKind::RsqrtInplace),
        "RecipInplace" => Some(OpKind::RecipInplace),
        "ExpInplace" => Some(OpKind::ExpInplace),
        "LogInplace" => Some(OpKind::LogInplace),
        "SinInplace" => Some(OpKind::SinInplace),
        "CosInplace" => Some(OpKind::CosInplace),
        "SignInplace" => Some(OpKind::SignInplace),
        "FloorInplace" => Some(OpKind::FloorInplace),
        "CeilInplace" => Some(OpKind::CeilInplace),
        "RoundInplace" => Some(OpKind::RoundInplace),
        "ErfInplace" => Some(OpKind::ErfInplace),
        "GeluErfInplace" => Some(OpKind::GeluErfInplace),
        "ClampInplace" => Some(OpKind::ClampInplace),
        "PowIInplace" => Some(OpKind::PowIInplace),
        "InplaceAffine" => Some(OpKind::InplaceAffine),
        "FusedSoftmaxCrossEntropy" => Some(OpKind::FusedSoftmaxCrossEntropy),
        "CausalConv1d" => Some(OpKind::CausalConv1d),
        "SelectiveScan" => Some(OpKind::SelectiveScan),
        "SsdChunkScan" => Some(OpKind::SsdChunkScan),
        "Nf4Matmul" => Some(OpKind::Nf4Matmul),
        _ => None,
    };
    // NOTE: `OpKind` is `#[non_exhaustive]` in `fuel-core-types`, so a
    // wildcard-free exhaustiveness anchor is not possible across the crate
    // boundary. The string table above is still explicit + audited (each
    // `OpKind` variant has its own arm); a new upstream variant simply
    // won't be reachable until a token is added here (an `UnknownOpKind`
    // at runtime, not a compile error — the non_exhaustive contract).
    match mapped {
        Some(op) => Ok(op),
        None => Err(FkcError::UnknownOpKind {
            section: section.to_string(),
            op_kind: s.to_string(),
        }),
    }
}

// ===========================================================================
// fused_op String → FusedOpId (SCREAMING_SNAKE FusedOps constant table; §2.2)
// ===========================================================================

/// Map a `fused_op` token to its [`FusedOpId`].
///
/// **The spec/contracts use the `FusedOps` CONSTANT NAME** — SCREAMING_SNAKE,
/// e.g. `SOFTMAX_LAST_DIM` / `FLASH_ATTN` / `QMATMUL` (FKC §3.1 token format,
/// §3.7). This is NOT the registry entry's `name` field, which is PascalCase
/// (`"SoftmaxLastDim"`). The earlier `default_registry().id_for_name(...)`
/// resolver matched the PascalCase `name` and therefore always *missed* on a
/// real contract token.
///
/// The mapping is an explicit `match` from each SCREAMING_SNAKE constant name
/// to its `FusedOps::<NAME>` `FusedOpId` (`UnknownFusedOp` on a miss). It is
/// kept honest by the [`tests::every_registry_id_is_reachable_through_table`]
/// drift-guard: every `FusedOpId` present in `default_registry()` must be
/// reachable through this table, so adding a new `FusedOps` const without a
/// table entry fails that test.
pub(crate) fn lower_fused_op(s: &str, section: &str) -> Result<FusedOpId, FkcError> {
    fused_op_id_for_const_name(s).ok_or_else(|| FkcError::UnknownFusedOp {
        section: section.to_string(),
        fused_op: s.to_string(),
    })
}

/// The SCREAMING_SNAKE `FusedOps::*` constant-name → [`FusedOpId`] table.
/// Returns `None` for an unknown token. One arm per `FusedOps` associated
/// const (`fuel-graph/src/registry.rs`); the drift-guard test asserts this
/// covers every registered id.
pub(crate) fn fused_op_id_for_const_name(s: &str) -> Option<FusedOpId> {
    let id = match s {
        "SOFTMAX_LAST_DIM" => FusedOps::SOFTMAX_LAST_DIM,
        "FUSED_LINEAR" => FusedOps::FUSED_LINEAR,
        "RMS_NORM_LAST_DIM" => FusedOps::RMS_NORM_LAST_DIM,
        "LAYER_NORM_LAST_DIM" => FusedOps::LAYER_NORM_LAST_DIM,
        "ROPE" => FusedOps::ROPE,
        "CONV2D" => FusedOps::CONV2D,
        "SOFTMAX_LAST_DIM_BACKWARD" => FusedOps::SOFTMAX_LAST_DIM_BACKWARD,
        "LAYER_NORM_LAST_DIM_BACKWARD" => FusedOps::LAYER_NORM_LAST_DIM_BACKWARD,
        "RMS_NORM_LAST_DIM_BACKWARD" => FusedOps::RMS_NORM_LAST_DIM_BACKWARD,
        "REDUCE_MAX_TO_BACKWARD" => FusedOps::REDUCE_MAX_TO_BACKWARD,
        "CONV_TRANSPOSE2D" => FusedOps::CONV_TRANSPOSE2D,
        "FLASH_ATTN" => FusedOps::FLASH_ATTN,
        "PAGED_ATTN" => FusedOps::PAGED_ATTN,
        "QMATMUL" => FusedOps::QMATMUL,
        "POWI_BACKWARD" => FusedOps::POWI_BACKWARD,
        "INPLACE_AFFINE" => FusedOps::INPLACE_AFFINE,
        "FUSED_SOFTMAX_CROSS_ENTROPY" => FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY,
        "CAUSAL_CONV1D" => FusedOps::CAUSAL_CONV1D,
        "SELECTIVE_SCAN" => FusedOps::SELECTIVE_SCAN,
        "SSD_CHUNK_SCAN" => FusedOps::SSD_CHUNK_SCAN,
        "NF4_MATMUL" => FusedOps::NF4_MATMUL,
        "FLASH_ATTN_BACKWARD_Q" => FusedOps::FLASH_ATTN_BACKWARD_Q,
        "FLASH_ATTN_BACKWARD_K" => FusedOps::FLASH_ATTN_BACKWARD_K,
        "FLASH_ATTN_BACKWARD_V" => FusedOps::FLASH_ATTN_BACKWARD_V,
        _ => return None,
    };
    Some(id)
}

// ===========================================================================
// dtype token / dtype_class String → DType (explicit match; §3.4)
// ===========================================================================

/// Map a single dtype token to a [`DType`]. Explicit exhaustive `match`
/// (FDX codes are a different axis; this is NOT `FromStr`-by-discriminant).
/// `BadScalarType` on a bad token.
pub(crate) fn lower_dtype(token: &str, section: &str, operand: &str) -> Result<DType, FkcError> {
    let mapped = match token {
        "U8" => Some(DType::U8),
        "I8" => Some(DType::I8),
        "U32" => Some(DType::U32),
        "I16" => Some(DType::I16),
        "I32" => Some(DType::I32),
        "I64" => Some(DType::I64),
        "BF16" => Some(DType::BF16),
        "F16" => Some(DType::F16),
        "F32" => Some(DType::F32),
        "F64" => Some(DType::F64),
        "F8E4M3" => Some(DType::F8E4M3),
        "F6E2M3" => Some(DType::F6E2M3),
        "F6E3M2" => Some(DType::F6E3M2),
        "F4" => Some(DType::F4),
        "F8E8M0" => Some(DType::F8E8M0),
        _ => None,
    };
    if let Some(dt) = mapped {
        // Exhaustiveness anchor (no wildcard): a new DType variant breaks
        // this match, forcing the token table above to grow.
        let _assert_exhaustive = match dt {
            DType::U8 | DType::I8 | DType::U32 | DType::I16 | DType::I32 | DType::I64
            | DType::BF16 | DType::F16 | DType::F32 | DType::F64 | DType::F8E4M3
            | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => (),
        };
        Ok(dt)
    } else {
        Err(FkcError::BadScalarType {
            section: section.to_string(),
            operand: operand.to_string(),
            token: token.to_string(),
        })
    }
}

/// Expand a `dtype_class` shorthand (§3.4) into its dtype list. `any`
/// resolves to the operand's enumerated `dtypes` (passed in `enumerated`).
fn expand_dtype_class(
    class: &str,
    enumerated: &[DType],
    section: &str,
    operand: &str,
) -> Result<Vec<DType>, FkcError> {
    match class {
        "float" => Ok(vec![DType::BF16, DType::F16, DType::F32, DType::F64]),
        "int" => Ok(vec![DType::I8, DType::I16, DType::I32, DType::I64]),
        "uint" => Ok(vec![DType::U8, DType::U32]),
        "any" => Ok(enumerated.to_vec()),
        other => Err(FkcError::BadScalarType {
            section: section.to_string(),
            operand: operand.to_string(),
            token: format!("dtype_class={other}"),
        }),
    }
}

// ===========================================================================
// backend String → BackendId (explicit match; §2.1)
// ===========================================================================

/// Map a `backend` string to a [`BackendId`]. Explicit exhaustive match.
fn lower_backend(s: &str, section: &str) -> Result<BackendId, FkcError> {
    let mapped = match s {
        "Cpu" => Some(BackendId::Cpu),
        "Cuda" => Some(BackendId::Cuda),
        "Vulkan" => Some(BackendId::Vulkan),
        "Metal" => Some(BackendId::Metal),
        _ => None,
    };
    // `BackendId` is `#[non_exhaustive]`; same note as `lower_op_kind` —
    // the explicit string table is the audit surface, not a compile gate.
    match mapped {
        Some(b) => Ok(b),
        None => Err(FkcError::UnknownBackend {
            section: section.to_string(),
            backend: s.to_string(),
        }),
    }
}

// ===========================================================================
// Per-operand dtype assembly → KernelDTypes
// ===========================================================================

/// Build the per-operand dtype list for the binding key: each INPUT
/// operand contributes one dtype (its first enumerated dtype, or the
/// single value of its `dtype_class` expansion when a class is used),
/// inputs in order, then each OUTPUT.
///
/// Per FKC §3.4 / the binding-table key shape (`[in1, in2, ..., out]`):
/// when an operand enumerates multiple dtypes (a family chassis like
/// `binary`'s `[F32, F64, BF16, F16]`) the FIRST dtype is taken as the
/// key's representative (a chassis section is documentation; the
/// per-(op, dtype) thunks carry the precise single-dtype keys). When a
/// `dtype_class` shorthand is present, the operand must resolve to a
/// single representative too.
fn assemble_dtypes(
    kernel: &FkcKernel,
    section: &str,
) -> Result<(KernelDTypes, Vec<ResolvedLayout>), FkcError> {
    let mut dtypes: KernelDTypes = SmallVec::new();
    let mut layouts: Vec<ResolvedLayout> = Vec::new();

    if let Some(accept) = &kernel.accept {
        for d in &accept.inputs {
            let operand = d.name.as_deref().unwrap_or("<input>");
            // Resolve this operand's enumerated dtypes first.
            let mut enumerated: Vec<DType> = Vec::new();
            for tok in &d.dtypes {
                enumerated.push(lower_dtype(tok, section, operand)?);
            }
            // dtype_class shorthand expands here (§3.4). If both a class
            // and explicit dtypes are present, the explicit list wins as
            // the enumeration; the class only fills an empty list.
            let representative = if let Some(class) = &d.dtype_class {
                let expanded = expand_dtype_class(class, &enumerated, section, operand)?;
                *expanded.first().ok_or_else(|| FkcError::BadScalarType {
                    section: section.to_string(),
                    operand: operand.to_string(),
                    token: format!("dtype_class={class} expanded to empty"),
                })?
            } else {
                *enumerated.first().ok_or_else(|| FkcError::BadScalarType {
                    section: section.to_string(),
                    operand: operand.to_string(),
                    token: "<no dtypes and no dtype_class>".to_string(),
                })?
            };
            dtypes.push(representative);
            layouts.push(caps_map::resolve_layout(d.layout.as_ref(), section, operand)?);
        }
    }

    // Outputs (return.outputs) contribute their dtype to the key tail
    // when it is a literal `fixed(DType)` rule; `passthrough(role)`
    // outputs default to the first input dtype (the key convention for
    // same-dtype ops). Best-effort: an output we cannot type is omitted
    // from the key tail rather than failing (the full return validation
    // is a later slice).
    if let Some(ret) = &kernel.return_ {
        for out in &ret.outputs {
            let operand = out.name.as_deref().unwrap_or("<output>");
            if let Some(rule) = out.dtype_rule.as_deref() {
                if let Some(dt) = parse_fixed_dtype_rule(rule, section, operand)? {
                    dtypes.push(dt);
                } else {
                    // passthrough(...) — mirror the first input dtype.
                    if let Some(first) = dtypes.first().copied() {
                        dtypes.push(first);
                    }
                }
            }
        }
    }

    Ok((dtypes, layouts))
}

/// Parse a `dtype_rule` string. Returns `Some(DType)` for `fixed(DType)`,
/// `None` for `passthrough(role)` / anything that is not a literal fixed
/// dtype. `BadScalarType` only if `fixed(...)` names a bad dtype.
fn parse_fixed_dtype_rule(
    rule: &str,
    section: &str,
    operand: &str,
) -> Result<Option<DType>, FkcError> {
    let rule = rule.trim();
    if let Some(inner) = rule.strip_prefix("fixed(").and_then(|s| s.strip_suffix(")")) {
        Ok(Some(lower_dtype(inner.trim(), section, operand)?))
    } else {
        Ok(None)
    }
}

// ===========================================================================
// Cost block → CompiledCostExpr (primary: flops; §2.3 AST half)
// ===========================================================================

/// Compile a contract's cost block into a [`CompiledCostExpr`].
///
/// Strategy: this slice carries the AST. A cost block that is absent,
/// `class`-only, or `judge_measured` with no coefficient expressions
/// compiles to [`CompiledCostExpr::Unknown`] (the register slice maps
/// that to `unknown_cost`). When expressions ARE present, the **`flops`**
/// expression is compiled as the primary cost AST (the per-tier `memory`
/// beyond device_bytes and the other coefficient fields are parsed for
/// validation but the primary held expression is `flops`; a full
/// multi-field cost vector is a register-slice concern).
///
/// Every present cost expression is parse-validated (V-FKC-8); a malformed
/// one surfaces as [`FkcError::CostExprParse`] with the field name.
fn compile_cost(block: Option<&CostBlock>, section: &str) -> Result<CompiledCostExpr, FkcError> {
    let Some(cost) = block else {
        return Ok(CompiledCostExpr::Unknown);
    };

    // Parse-validate every present coefficient expression (so a malformed
    // bytes_moved is caught even though flops is the primary held AST).
    let parse = |field: &str, src: Option<&str>| -> Result<CompiledCostExpr, FkcError> {
        cost_expr::compile_field(src).map_err(|e| FkcError::CostExprParse {
            section: section.to_string(),
            field: field.to_string(),
            expr: src.unwrap_or("").to_string(),
            reason: e.to_string(),
        })
    };

    let flops = parse("flops", cost.flops.as_deref())?;
    let _bytes_moved = parse("bytes_moved", cost.bytes_moved.as_deref())?;
    // overhead_ns / memory.device_bytes are raw scalars (number or `~`);
    // when they are an expression STRING they are parse-validated too.
    if let Some(mem) = &cost.memory {
        if let Some(serde_yml::Value::String(s)) = &mem.device_bytes {
            let _ = parse("memory.device_bytes", Some(s))?;
        }
    }
    if let Some(serde_yml::Value::String(s)) = &cost.overhead_ns {
        let _ = parse("overhead_ns", Some(s))?;
    }

    // The primary held AST is flops. If flops is Unknown but bytes_moved
    // carries an expression, hold bytes_moved instead (so a cost block with
    // only a bytes_moved formula is not collapsed to Unknown).
    match flops {
        CompiledCostExpr::Unknown => Ok(_bytes_moved),
        expr => Ok(expr),
    }
}

// ===========================================================================
// The lowering entry points
// ===========================================================================

/// Resolve the effective `backend` / `kernel_source` / `entry_point` for a
/// kernel, applying the front-matter fallbacks the caller passes in.
struct Defaults<'a> {
    backend: &'a str,
    kernel_source: &'a str,
    revision_base: &'a str,
}

/// Lower one parsed [`FkcKernel`] into a [`Resolved`] record, resolving its
/// `entry_point` through `link`. Exactly one of `op_kind` / `fused_op`
/// must be present.
///
/// `defaults` carries the front-matter `backend` / `kernel_source` /
/// `revision_base` so a kernel that omits them inherits the provider's.
fn lower_kernel(
    kernel: &FkcKernel,
    defaults: &Defaults<'_>,
    link: &dyn LinkRegistry,
) -> Result<Resolved, FkcError> {
    let section = kernel.kernel.as_str();

    // Exactly one of op_kind / fused_op.
    match (kernel.op_kind.as_deref(), kernel.fused_op.as_deref()) {
        (Some(op_str), None) => {
            let op = lower_op_kind(op_str, section)?;
            lower_primitive(kernel, op, defaults, link).map(Resolved::Primitive)
        }
        (None, Some(fused_str)) => {
            let id = lower_fused_op(fused_str, section)?;
            lower_fused(kernel, id, defaults, link).map(Resolved::Fused)
        }
        (op, fused) => Err(FkcError::OpTargetAmbiguous {
            section: section.to_string(),
            op_kind: op.map(String::from),
            fused_op: fused.map(String::from),
        }),
    }
}

fn lower_primitive(
    kernel: &FkcKernel,
    op: OpKind,
    defaults: &Defaults<'_>,
    link: &dyn LinkRegistry,
) -> Result<ResolvedPrimitive, FkcError> {
    let section = kernel.kernel.as_str();
    let backend_str = kernel.backend.as_deref().unwrap_or(defaults.backend);
    let backend = lower_backend(backend_str, section)?;
    let (dtypes, layouts) = assemble_dtypes(kernel, section)?;
    let caps = caps_map::project_kernel_caps(&layouts);
    let precision = precision::lower_precision(kernel.precision.as_ref(), section)?;
    let cost = compile_cost(kernel.cost.as_ref(), section)?;

    let entry_point = kernel.entry_point.as_deref().ok_or_else(|| {
        FkcError::MissingRequiredField {
            section: section.to_string(),
            field: "entry_point".to_string(),
        }
    })?;
    let kernel_ref = link.resolve_primitive(entry_point).ok_or_else(|| {
        FkcError::UnknownEntryPoint {
            section: section.to_string(),
            entry_point: entry_point.to_string(),
        }
    })?;

    let kernel_source = kernel
        .kernel_source
        .as_deref()
        .unwrap_or(defaults.kernel_source)
        .to_string();
    let revision = revhash::compute_revision(kernel, entry_point, defaults.revision_base)?;

    Ok(ResolvedPrimitive {
        op,
        dtypes,
        backend,
        caps,
        layouts,
        precision,
        cost,
        kernel: kernel_ref,
        kernel_source,
        revision,
    })
}

fn lower_fused(
    kernel: &FkcKernel,
    id: FusedOpId,
    defaults: &Defaults<'_>,
    link: &dyn LinkRegistry,
) -> Result<ResolvedFused, FkcError> {
    let section = kernel.kernel.as_str();
    let backend_str = kernel.backend.as_deref().unwrap_or(defaults.backend);
    let backend = lower_backend(backend_str, section)?;
    let (dtypes, layouts) = assemble_dtypes(kernel, section)?;
    let caps = caps_map::project_kernel_caps(&layouts);
    let precision = precision::lower_precision(kernel.precision.as_ref(), section)?;
    let cost = compile_cost(kernel.cost.as_ref(), section)?;

    let entry_point = kernel.entry_point.as_deref().ok_or_else(|| {
        FkcError::MissingRequiredField {
            section: section.to_string(),
            field: "entry_point".to_string(),
        }
    })?;
    let kernel_ref = link.resolve_fused(entry_point).ok_or_else(|| {
        FkcError::UnknownEntryPoint {
            section: section.to_string(),
            entry_point: entry_point.to_string(),
        }
    })?;

    let kernel_source = kernel
        .kernel_source
        .as_deref()
        .unwrap_or(defaults.kernel_source)
        .to_string();
    let revision = revhash::compute_revision(kernel, entry_point, defaults.revision_base)?;

    Ok(ResolvedFused {
        id,
        dtypes,
        backend,
        caps,
        layouts,
        precision,
        cost,
        kernel: kernel_ref,
        kernel_source,
        revision,
    })
}

/// Lower every kernel section of a parsed file into [`Resolved`] records,
/// using the file's front-matter for the backend / kernel_source /
/// revision_base defaults and resolving entry points through `link`.
pub fn lower_file(
    file: &crate::fkc::schema::FkcFile,
    link: &dyn LinkRegistry,
) -> Result<Vec<Resolved>, FkcError> {
    let provider = &file.front_matter.provider;
    let defaults = Defaults {
        backend: provider.backend.as_str(),
        kernel_source: provider.kernel_source.as_str(),
        revision_base: provider.revision_base.as_deref().unwrap_or(""),
    };
    file.kernels
        .iter()
        .map(|k| lower_kernel(k, &defaults, link))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fkc::parse_file;
    use std::sync::{Arc, RwLock};

    const ELEMENTWISE_BINARY: &str =
        include_str!("../../../docs/kernel-contracts/cpu/elementwise-binary.fkc.md");
    const QUANT_MATMUL: &str =
        include_str!("../../../docs/kernel-contracts/cpu/quant-matmul.fkc.md");
    /// A real FUSED contract bundle whose `fused_op: SOFTMAX_LAST_DIM` etc.
    /// must now resolve through the SCREAMING_SNAKE constant table (Task 1).
    const FUSED_NORM_SOFTMAX: &str =
        include_str!("../../../docs/kernel-contracts/fused/norm-softmax.fkc.md");

    // ---- A test LinkRegistry stub mapping every entry_point to a dummy ----

    fn dummy_kernel(
        _inputs: &[Arc<RwLock<fuel_memory::Storage>>],
        _outputs: &mut [Arc<RwLock<fuel_memory::Storage>>],
        _layouts: &[fuel_core_types::Layout],
        _params: &crate::kernel::OpParams,
    ) -> fuel_core_types::Result<()> {
        Ok(())
    }

    struct StubLink;
    impl LinkRegistry for StubLink {
        fn resolve_primitive(&self, _symbol: &str) -> Option<KernelRef> {
            Some(dummy_kernel)
        }
        fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
            Some(dummy_kernel)
        }
    }

    /// A stub that resolves nothing (for the UnknownEntryPoint negative).
    struct EmptyLink;
    impl LinkRegistry for EmptyLink {
        fn resolve_primitive(&self, _symbol: &str) -> Option<KernelRef> {
            None
        }
        fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
            None
        }
    }

    fn lower_one(src: &str, kernel_name: &str) -> Result<Resolved, FkcError> {
        let file = parse_file(src).expect("parses");
        let provider = &file.front_matter.provider;
        let defaults = Defaults {
            backend: provider.backend.as_str(),
            kernel_source: provider.kernel_source.as_str(),
            revision_base: provider.revision_base.as_deref().unwrap_or(""),
        };
        let kernel = file
            .kernels
            .iter()
            .find(|k| k.kernel == kernel_name)
            .unwrap_or_else(|| panic!("kernel {kernel_name} present"));
        lower_kernel(kernel, &defaults, &StubLink)
    }

    // =====================================================================
    // POSITIVE: real contracts lower
    // =====================================================================

    #[test]
    fn lowers_real_add_f32() {
        let r = lower_one(ELEMENTWISE_BINARY, "add_f32").expect("add_f32 lowers");
        let Resolved::Primitive(p) = r else {
            panic!("add_f32 is a primitive");
        };
        assert_eq!(p.op, OpKind::AddElementwise);
        // inputs lhs, rhs (F32, F32) then output (passthrough → F32).
        assert_eq!(p.dtypes.as_slice(), &[DType::F32, DType::F32, DType::F32]);
        assert_eq!(p.backend, BackendId::Cpu);
        // contiguous-only contract ⇒ strided_input false.
        assert!(!p.caps.strided_input);
        // precision mapped (bit-stable, ulp 0).
        assert!(p.precision.bit_stable_on_same_hardware);
        assert_eq!(p.precision.max_ulp, Some(0));
        // a non-null KernelRef.
        assert_eq!(p.kernel as *const () as usize, dummy_kernel as *const () as usize);
        // cost: flops = "n" parsed (not Unknown).
        assert!(matches!(p.cost, CompiledCostExpr::Expr(_)));
        assert_eq!(p.kernel_source, "portable-cpu");
    }

    #[test]
    fn lowers_binary_chassis_section() {
        // The umbrella `binary` section uses dtype lists [F32,F64,BF16,F16]
        // ⇒ representative F32 (first).
        let r = lower_one(ELEMENTWISE_BINARY, "binary").expect("binary lowers");
        let Resolved::Primitive(p) = r else { panic!() };
        assert_eq!(p.op, OpKind::AddElementwise);
        assert_eq!(p.dtypes[0], DType::F32);
    }

    #[test]
    fn lowers_real_qmatmul_q4_0() {
        let r = lower_one(QUANT_MATMUL, "qmatmul_q4_0_f32").expect("q4_0 lowers");
        let Resolved::Primitive(p) = r else { panic!("QMatMul is a primitive") };
        assert_eq!(p.op, OpKind::QMatMul);
        // activations F32, weight U8, output fixed(F32).
        assert_eq!(p.dtypes.as_slice(), &[DType::F32, DType::U8, DType::F32]);
        assert_eq!(p.backend, BackendId::Cpu);
        // judge_measured with all-`~` coefficients ⇒ Unknown cost sentinel.
        assert_eq!(p.cost, CompiledCostExpr::Unknown);
        assert!(p.precision.bit_stable_on_same_hardware);
    }

    #[test]
    fn lowers_real_nf4_matmul_f32() {
        // NF4 reaches the dedicated OpKind::Nf4Matmul path → it LOWERS in
        // this slice (the AFFINE_BLOCK / MxNotYetRegistrable gate is a
        // register-slice concern; lowering produces a typed record).
        let r = lower_one(QUANT_MATMUL, "nf4_matmul_f32").expect("nf4 lowers");
        let Resolved::Primitive(p) = r else { panic!() };
        assert_eq!(p.op, OpKind::Nf4Matmul);
        // activations F32, w_packed U8, absmax F32, output passthrough → F32.
        assert_eq!(
            p.dtypes.as_slice(),
            &[DType::F32, DType::U8, DType::F32, DType::F32]
        );
    }

    #[test]
    fn lower_file_lowers_every_kernel() {
        let file = parse_file(ELEMENTWISE_BINARY).expect("parses");
        let resolved = lower_file(&file, &StubLink).expect("all lower");
        assert_eq!(resolved.len(), file.kernels.len());
    }

    /// Task 1 end-to-end verification: a REAL fused contract from the corpus
    /// (`fused/norm-softmax.fkc.md`, `fused_op: SOFTMAX_LAST_DIM`) lowers via
    /// the stub link — proving the SCREAMING_SNAKE token now resolves.
    #[test]
    fn lowers_real_fused_softmax_last_dim() {
        let r = lower_one(FUSED_NORM_SOFTMAX, "softmax_last_dim")
            .expect("softmax_last_dim (fused) lowers");
        let Resolved::Fused(f) = r else {
            panic!("softmax_last_dim is a fused op");
        };
        assert_eq!(f.id, FusedOps::SOFTMAX_LAST_DIM);
        assert_eq!(f.backend, BackendId::Cpu);
        assert_eq!(f.kernel_source, "portable-cpu");

        // The whole bundle lowers (every fused_op token resolves).
        let file = parse_file(FUSED_NORM_SOFTMAX).expect("parses");
        let resolved = lower_file(&file, &StubLink).expect("all fused kernels lower");
        assert_eq!(resolved.len(), file.kernels.len());
        assert!(resolved.iter().all(|r| matches!(r, Resolved::Fused(_))));
    }

    // =====================================================================
    // NEGATIVES
    // =====================================================================

    #[test]
    fn bogus_op_kind_is_unknown_op_kind() {
        let err = lower_op_kind("NotARealOp", "k").expect_err("bogus op");
        assert!(matches!(err, FkcError::UnknownOpKind { .. }), "got {err:?}");
    }

    #[test]
    fn bogus_dtype_is_bad_scalar_type() {
        let err = lower_dtype("F99", "k", "lhs").expect_err("bogus dtype");
        assert!(matches!(err, FkcError::BadScalarType { .. }), "got {err:?}");
    }

    #[test]
    fn bogus_fused_op_is_unknown_fused_op() {
        let err = lower_fused_op("NotAFusedOp", "k").expect_err("bogus fused");
        assert!(matches!(err, FkcError::UnknownFusedOp { .. }), "got {err:?}");
    }

    #[test]
    fn screaming_snake_const_name_resolves_not_pascalcase() {
        // The contract token is the SCREAMING_SNAKE FusedOps constant name.
        assert_eq!(
            lower_fused_op("SOFTMAX_LAST_DIM", "k").unwrap(),
            FusedOps::SOFTMAX_LAST_DIM
        );
        assert_eq!(lower_fused_op("FLASH_ATTN", "k").unwrap(), FusedOps::FLASH_ATTN);
        assert_eq!(lower_fused_op("QMATMUL", "k").unwrap(), FusedOps::QMATMUL);
        // The PascalCase registry `name` (what the OLD id_for_name resolver
        // matched) must NOT resolve — that was the bug.
        assert!(lower_fused_op("SoftmaxLastDim", "k").is_err());
        assert!(lower_fused_op("FlashAttn", "k").is_err());
    }

    /// DRIFT GUARD (Task 1): every `FusedOpId` present in `default_registry()`
    /// must be reachable through the SCREAMING_SNAKE constant-name table — so
    /// adding a new `FusedOps` const + registry entry without a table arm
    /// fails this test (it would otherwise be silently unimportable).
    #[test]
    fn every_registry_id_is_reachable_through_table() {
        use std::collections::HashSet;
        // Every id the constant-name table can produce.
        let reachable: HashSet<FusedOpId> = [
            "SOFTMAX_LAST_DIM",
            "FUSED_LINEAR",
            "RMS_NORM_LAST_DIM",
            "LAYER_NORM_LAST_DIM",
            "ROPE",
            "CONV2D",
            "SOFTMAX_LAST_DIM_BACKWARD",
            "LAYER_NORM_LAST_DIM_BACKWARD",
            "RMS_NORM_LAST_DIM_BACKWARD",
            "REDUCE_MAX_TO_BACKWARD",
            "CONV_TRANSPOSE2D",
            "FLASH_ATTN",
            "PAGED_ATTN",
            "QMATMUL",
            "POWI_BACKWARD",
            "INPLACE_AFFINE",
            "FUSED_SOFTMAX_CROSS_ENTROPY",
            "CAUSAL_CONV1D",
            "SELECTIVE_SCAN",
            "SSD_CHUNK_SCAN",
            "NF4_MATMUL",
            "FLASH_ATTN_BACKWARD_Q",
            "FLASH_ATTN_BACKWARD_K",
            "FLASH_ATTN_BACKWARD_V",
        ]
        .iter()
        .map(|n| super::fused_op_id_for_const_name(n).expect("table arm exists"))
        .collect();

        // Every registered id MUST be in `reachable`.
        for entry in fuel_graph::registry::default_registry().entries_iter() {
            assert!(
                reachable.contains(&entry.id),
                "FusedOpId {:?} (registry name {:?}) is registered but NOT reachable through the \
                 SCREAMING_SNAKE constant-name table — add a `fused_op_id_for_const_name` arm",
                entry.id,
                entry.name,
            );
        }
    }

    #[test]
    fn bogus_backend_is_unknown_backend() {
        let err = lower_backend("Tpu", "k").expect_err("bogus backend");
        assert!(matches!(err, FkcError::UnknownBackend { .. }), "got {err:?}");
    }

    #[test]
    fn unknown_entry_point_errors() {
        let file = parse_file(ELEMENTWISE_BINARY).expect("parses");
        let provider = &file.front_matter.provider;
        let defaults = Defaults {
            backend: provider.backend.as_str(),
            kernel_source: provider.kernel_source.as_str(),
            revision_base: provider.revision_base.as_deref().unwrap_or(""),
        };
        let kernel = file.kernels.iter().find(|k| k.kernel == "add_f32").unwrap();
        let err = lower_kernel(kernel, &defaults, &EmptyLink).expect_err("no entry point");
        assert!(matches!(err, FkcError::UnknownEntryPoint { .. }), "got {err:?}");
    }

    #[test]
    fn malformed_cost_expr_is_cost_expr_parse() {
        let mut block = CostBlock {
            provenance: Some("declared".into()),
            class: None,
            flops: Some("2 * * n".into()), // malformed
            bytes_moved: None,
            overhead_ns: None,
            memory: None,
        };
        let err = compile_cost(Some(&block), "k").expect_err("malformed flops");
        assert!(matches!(err, FkcError::CostExprParse { .. }), "got {err:?}");

        // Also catch a malformed bytes_moved even when flops is fine.
        block.flops = Some("n".into());
        block.bytes_moved = Some("3 * n *".into());
        let err = compile_cost(Some(&block), "k").expect_err("malformed bytes_moved");
        assert!(matches!(err, FkcError::CostExprParse { field, .. } if field == "bytes_moved"));
    }

    #[test]
    fn both_op_kind_and_fused_op_is_ambiguous() {
        let file = parse_file(ELEMENTWISE_BINARY).expect("parses");
        let mut kernel = file.kernels[0].clone();
        kernel.fused_op = Some("SoftmaxLastDim".into()); // now both present
        let defaults = Defaults {
            backend: "Cpu",
            kernel_source: "x",
            revision_base: "",
        };
        let err = lower_kernel(&kernel, &defaults, &StubLink).expect_err("ambiguous");
        assert!(matches!(err, FkcError::OpTargetAmbiguous { .. }), "got {err:?}");
    }
}
