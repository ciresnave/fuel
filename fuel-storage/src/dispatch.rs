//! Capability-driven dispatch tables. Phase 7.5 A5.
//!
//! `CapabilityRegistry` collects [`BackendCapabilities`] from each
//! registered backend; `TransferMatrix` encodes the cheapest path
//! between every pair of registered devices.
//!
//! Together they let DAG construction (Phase B) answer two
//! questions:
//!
//! 1. **Which backend should handle `(op, dtype)`?** — query
//!    [`CapabilityRegistry::find_backends`] / [`find_backend_for`]
//!    to get the set of registered backends that support the pair.
//! 2. **How does data move between two devices?** — query
//!    [`TransferMatrix::path`] for the chosen path; falls back to
//!    `HostStaging` if no direct path exists.
//!
//! The registry is process-wide (typically initialized once at
//! application startup via `OnceLock`) but exposed here as a value
//! so tests and alternative dispatch policies can construct their
//! own. fuel-graph-router will host the canonical process-wide
//! instance in Phase B.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use fuel_core_types::backend::{BackendCapabilities, TransferPath};
use fuel_core_types::dispatch::OpKind;
use fuel_core_types::probe::BackendId;
use fuel_core_types::{DType, DeviceLocation, Error, Result};

use crate::kernel::{KernelBindingTable, OpParams};
use crate::{BackendStorage, Storage};

/// Collection of backend capabilities, queried during DAG
/// construction to pick which backend handles each op.
#[derive(Debug, Default)]
pub struct CapabilityRegistry {
    backends: Vec<BackendCapabilities>,
}

impl CapabilityRegistry {
    /// Construct an empty registry. Add backends with `register`.
    pub fn new() -> Self {
        Self { backends: Vec::new() }
    }

    /// Register a backend's capabilities. Order of registration is
    /// preserved; lookup methods return registrations in their
    /// original order so callers can encode preference (CPU
    /// fallback last; GPU first).
    pub fn register(&mut self, caps: BackendCapabilities) {
        self.backends.push(caps);
    }

    /// All registered backends.
    pub fn backends(&self) -> &[BackendCapabilities] {
        &self.backends
    }

    /// Backends supporting `(op, dtype)`, in registration order.
    pub fn find_backends(&self, op: OpKind, dtype: DType) -> Vec<&BackendCapabilities> {
        self.backends
            .iter()
            .filter(|caps| caps.op_dtype_support.contains(&(op, dtype)))
            .collect()
    }

    /// Pick the first registered backend that supports
    /// `(op, dtype)`. Returns
    /// [`Error::NoBackendForOp`](fuel_core_types::Error::NoBackendForOp)
    /// with diagnostic data if none does. Production-correct: never
    /// panics, always surfaces the gap as a typed error.
    ///
    /// "First" follows the order of `register` calls. Convention:
    /// register GPU before CPU so GPU wins ties; the universal CPU
    /// fallback gets picked iff no GPU registered for `(op, dtype)`.
    pub fn find_backend_for(&self, op: OpKind, dtype: DType) -> Result<&BackendCapabilities> {
        for caps in &self.backends {
            if caps.op_dtype_support.contains(&(op, dtype)) {
                return Ok(caps);
            }
        }
        Err(Error::NoBackendForOp {
            op,
            dtype,
            available_backends: self.backends.iter().map(|c| c.backend_id).collect(),
            supported_combinations: self
                .backends
                .iter()
                .flat_map(|c| {
                    c.op_dtype_support
                        .iter()
                        .map(|&(o, d)| (c.backend_id, o, d))
                })
                .collect(),
        }
        .bt())
    }

    /// Build a [`TransferMatrix`] from the registered backends'
    /// advertised transfer paths. Each backend contributes its
    /// outbound paths; the matrix consolidates them into a
    /// `(src, dst) -> TransferPath` lookup.
    pub fn build_transfer_matrix(&self) -> TransferMatrix {
        let mut entries = HashMap::new();
        for caps in &self.backends {
            for (dst, path) in &caps.transfer_paths {
                entries.insert((caps.device_location, *dst), *path);
            }
        }
        TransferMatrix { entries }
    }
}

/// Lookup table mapping `(source_device, dest_device)` pairs to the
/// cheapest available [`TransferPath`]. Built once at registration
/// time from each backend's advertised outbound paths; consulted
/// every time the DAG inserts an `Op::Move` / `Op::Copy`.
#[derive(Debug, Default)]
pub struct TransferMatrix {
    entries: HashMap<(DeviceLocation, DeviceLocation), TransferPath>,
}

impl TransferMatrix {
    /// Look up the registered path between two devices. Returns
    /// `None` if no direct path was advertised; the caller can
    /// fall back to host-staging via the universal `HostStaging`
    /// path.
    pub fn path(&self, src: DeviceLocation, dst: DeviceLocation) -> Option<TransferPath> {
        if src == dst {
            return Some(TransferPath::SameDevice);
        }
        self.entries.get(&(src, dst)).copied()
    }

    /// Same as [`path`] but always returns a path: falls back to
    /// `TransferPath::HostStaging` (the universal fallback) when no
    /// direct advertised path exists. CPU is reachable from every
    /// backend through host-staging, so this never returns an error
    /// for practical use cases — though see
    /// [`Error::UnsupportedTransfer`](fuel_core_types::Error::UnsupportedTransfer)
    /// for the case when a backend can't even host-stage.
    pub fn path_or_staging(&self, src: DeviceLocation, dst: DeviceLocation) -> TransferPath {
        self.path(src, dst).unwrap_or(TransferPath::HostStaging)
    }

    /// All entries in the matrix.
    pub fn entries(&self) -> impl Iterator<Item = (&(DeviceLocation, DeviceLocation), &TransferPath)> {
        self.entries.iter()
    }
}

/// Resolve which backend should handle `(op, dtype)` given the
/// registry of available backends. Returns the chosen `BackendId`
/// or [`Error::NoBackendForOp`] with diagnostic data on miss.
///
/// The chosen backend is the first registered backend that
/// supports `(op, dtype)`. Convention: register GPU before CPU so
/// GPU wins ties; the universal CPU fallback is picked iff no GPU
/// registered.
///
/// Phase 7.5 B3 — used by op-builder methods to populate
/// `Graph::target_backends` at DAG construction time. After the
/// full migration, every Node has a target_backend set this way.
pub fn resolve_target_backend(
    registry: &CapabilityRegistry,
    op: OpKind,
    dtype: DType,
) -> Result<BackendId> {
    registry.find_backend_for(op, dtype).map(|caps| caps.backend_id)
}

/// Residency-aware variant of [`resolve_target_backend`]. Prefers
/// backends that have at least one input already resident on their
/// device, breaking ties only on the residency axis (within the
/// "supports (op, dtype)" set). Falls back to registration order
/// when no candidate has local input residency.
///
/// `input_locations` is the list of input tensors' current
/// `DeviceLocation`s; pass an empty slice for ops that take no
/// inputs (constants / factories).
///
/// Phase 7.5 B3 — used when the dispatcher should avoid transfers
/// when possible. A simpler dispatch policy is to call
/// [`resolve_target_backend`] directly and let Router auto-insert
/// transfers; the residency-aware version saves transfers when
/// the choice is otherwise free.
pub fn resolve_target_backend_residency_aware(
    registry: &CapabilityRegistry,
    op: OpKind,
    dtype: DType,
    input_locations: &[DeviceLocation],
) -> Result<BackendId> {
    let candidates = registry.find_backends(op, dtype);
    if candidates.is_empty() {
        // Reuse find_backend_for to construct the canonical error.
        return registry.find_backend_for(op, dtype).map(|c| c.backend_id);
    }
    // Pick a candidate whose device matches at least one input
    // location.
    for caps in &candidates {
        if input_locations.contains(&caps.device_location) {
            return Ok(caps.backend_id);
        }
    }
    // No residency match — fall back to first candidate
    // (registration order).
    Ok(candidates[0].backend_id)
}

// =============================================================================
// Phase 7.5 B5 — CPU dispatch wrappers + registration
// =============================================================================

/// Helper: extract `&CpuStorageBytes` from `&Storage`. Returns
/// Err if the variant isn't `BackendStorage::Cpu`.
fn cpu_input(s: &Storage) -> Result<&fuel_cpu_backend::CpuStorageBytes> {
    match &s.inner {
        BackendStorage::Cpu(c) => Ok(c),
        #[allow(unreachable_patterns)]
        _ => Err(Error::Msg(
            "cpu kernel wrapper called with non-CPU input".to_string(),
        )
        .bt()),
    }
}

/// Helper: extract `&mut CpuStorageBytes` from `&mut Storage`.
fn cpu_output(s: &mut Storage) -> Result<&mut fuel_cpu_backend::CpuStorageBytes> {
    match &mut s.inner {
        BackendStorage::Cpu(c) => Ok(c),
        #[allow(unreachable_patterns)]
        _ => Err(Error::Msg(
            "cpu kernel wrapper called with non-CPU output".to_string(),
        )
        .bt()),
    }
}

/// Acquire a poisoned-lock-aware read guard. Lock poisoning is a
/// programming bug (a previous writer panicked while holding the
/// lock); production code surfaces it as a typed error rather than
/// re-panicking through `unwrap`.
fn read_storage(arc: &Arc<RwLock<Storage>>) -> Result<std::sync::RwLockReadGuard<'_, Storage>> {
    arc.read()
        .map_err(|_| Error::Msg("kernel wrapper: storage RwLock poisoned (read)".to_string()).bt())
}

fn write_storage(
    arc: &Arc<RwLock<Storage>>,
) -> Result<std::sync::RwLockWriteGuard<'_, Storage>> {
    arc.write()
        .map_err(|_| Error::Msg("kernel wrapper: storage RwLock poisoned (write)".to_string()).bt())
}

/// Build a `(2 inputs, 1 output)` CPU dispatch wrapper that calls a
/// typed binary kernel. The expanded function matches the
/// [`KernelRef`] signature and is suitable for direct registration
/// in the binding table.
macro_rules! cpu_binary_wrapper {
    ($wrapper:ident, $kernel:path, $op_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 2 inputs, got {}",
                    $op_name,
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 1 output, got {}",
                    $op_name,
                    outputs.len(),
                ))
                .bt());
            }
            let lhs_guard = read_storage(&inputs[0])?;
            let rhs_guard = read_storage(&inputs[1])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let lhs_cpu = cpu_input(&lhs_guard)?;
            let rhs_cpu = cpu_input(&rhs_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(lhs_cpu, rhs_cpu, out_cpu)
        }
    };
}

/// Build a `(1 input, 1 output)` CPU dispatch wrapper that calls a
/// typed unary kernel.
macro_rules! cpu_unary_wrapper {
    ($wrapper:ident, $kernel:path, $op_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 1 input, got {}",
                    $op_name,
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 1 output, got {}",
                    $op_name,
                    outputs.len(),
                ))
                .bt());
            }
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(in_cpu, out_cpu)
        }
    };
}

cpu_binary_wrapper!(add_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::add_f32, "add_elementwise");
cpu_binary_wrapper!(sub_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::sub_f32, "sub_elementwise");
cpu_binary_wrapper!(mul_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::mul_f32, "mul_elementwise");
cpu_binary_wrapper!(div_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::div_f32, "div_elementwise");

cpu_unary_wrapper!(relu_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::relu_f32, "relu_elementwise");
cpu_unary_wrapper!(neg_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::neg_f32, "neg_elementwise");
cpu_unary_wrapper!(sqr_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::sqr_f32, "sqr_elementwise");
cpu_unary_wrapper!(sqrt_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::sqrt_f32, "sqrt_elementwise");
cpu_unary_wrapper!(recip_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::recip_f32, "recip_elementwise");
cpu_unary_wrapper!(abs_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::abs_f32, "abs_elementwise");
cpu_unary_wrapper!(tanh_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::tanh_f32, "tanh_elementwise");
cpu_unary_wrapper!(exp_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::exp_f32, "exp_elementwise");
cpu_unary_wrapper!(log_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::log_f32, "log_elementwise");
cpu_unary_wrapper!(sin_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::sin_f32, "sin_elementwise");
cpu_unary_wrapper!(cos_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::cos_f32, "cos_elementwise");
cpu_unary_wrapper!(sigmoid_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::sigmoid_f32, "sigmoid_elementwise");
cpu_unary_wrapper!(silu_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::silu_f32, "silu_elementwise");
cpu_unary_wrapper!(gelu_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::gelu_f32, "gelu_elementwise");
cpu_unary_wrapper!(step_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::step_f32, "step_elementwise");

// f64 elementwise wrappers — same wrapper macros, different
// underlying kernels. The dispatch wrappers themselves are
// dtype-agnostic (they just call the typed kernel); the
// (op, dtype, backend) registration in `register_cpu_kernels`
// is what selects the right one at lookup time.
cpu_binary_wrapper!(add_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::add_f64, "add_elementwise");
cpu_binary_wrapper!(sub_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::sub_f64, "sub_elementwise");
cpu_binary_wrapper!(mul_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::mul_f64, "mul_elementwise");
cpu_binary_wrapper!(div_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::div_f64, "div_elementwise");

cpu_unary_wrapper!(relu_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::relu_f64, "relu_elementwise");
cpu_unary_wrapper!(neg_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::neg_f64, "neg_elementwise");
cpu_unary_wrapper!(sqr_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::sqr_f64, "sqr_elementwise");
cpu_unary_wrapper!(sqrt_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::sqrt_f64, "sqrt_elementwise");
cpu_unary_wrapper!(recip_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::recip_f64, "recip_elementwise");
cpu_unary_wrapper!(abs_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::abs_f64, "abs_elementwise");
cpu_unary_wrapper!(tanh_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::tanh_f64, "tanh_elementwise");
cpu_unary_wrapper!(exp_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::exp_f64, "exp_elementwise");
cpu_unary_wrapper!(log_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::log_f64, "log_elementwise");
cpu_unary_wrapper!(sin_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::sin_f64, "sin_elementwise");
cpu_unary_wrapper!(cos_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::cos_f64, "cos_elementwise");
cpu_unary_wrapper!(sigmoid_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::sigmoid_f64, "sigmoid_elementwise");
cpu_unary_wrapper!(silu_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::silu_f64, "silu_elementwise");
cpu_unary_wrapper!(gelu_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::gelu_f64, "gelu_elementwise");
cpu_unary_wrapper!(step_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::step_f64, "step_elementwise");

cpu_binary_wrapper!(maximum_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::maximum_f32, "maximum_elementwise");
cpu_binary_wrapper!(minimum_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::minimum_f32, "minimum_elementwise");
cpu_binary_wrapper!(maximum_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::maximum_f64, "maximum_elementwise");
cpu_binary_wrapper!(minimum_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::minimum_f64, "minimum_elementwise");

// bf16 elementwise wrappers (via-f32 round-trip kernels).
cpu_binary_wrapper!(add_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::add_bf16, "add_elementwise");
cpu_binary_wrapper!(sub_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::sub_bf16, "sub_elementwise");
cpu_binary_wrapper!(mul_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::mul_bf16, "mul_elementwise");
cpu_binary_wrapper!(div_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::div_bf16, "div_elementwise");
cpu_binary_wrapper!(maximum_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::maximum_bf16, "maximum_elementwise");
cpu_binary_wrapper!(minimum_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::minimum_bf16, "minimum_elementwise");

cpu_unary_wrapper!(relu_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::relu_bf16, "relu_elementwise");
cpu_unary_wrapper!(neg_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::neg_bf16, "neg_elementwise");
cpu_unary_wrapper!(sqr_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::sqr_bf16, "sqr_elementwise");
cpu_unary_wrapper!(sqrt_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::sqrt_bf16, "sqrt_elementwise");
cpu_unary_wrapper!(recip_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::recip_bf16, "recip_elementwise");
cpu_unary_wrapper!(abs_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::abs_bf16, "abs_elementwise");
cpu_unary_wrapper!(tanh_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::tanh_bf16, "tanh_elementwise");
cpu_unary_wrapper!(exp_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::exp_bf16, "exp_elementwise");
cpu_unary_wrapper!(log_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::log_bf16, "log_elementwise");
cpu_unary_wrapper!(sin_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::sin_bf16, "sin_elementwise");
cpu_unary_wrapper!(cos_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::cos_bf16, "cos_elementwise");
cpu_unary_wrapper!(sigmoid_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::sigmoid_bf16, "sigmoid_elementwise");
cpu_unary_wrapper!(silu_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::silu_bf16, "silu_elementwise");
cpu_unary_wrapper!(gelu_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::gelu_bf16, "gelu_elementwise");
cpu_unary_wrapper!(step_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::step_bf16, "step_elementwise");

// f16 elementwise wrappers — direct mirrors of bf16.
cpu_binary_wrapper!(add_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::add_f16, "add_elementwise");
cpu_binary_wrapper!(sub_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::sub_f16, "sub_elementwise");
cpu_binary_wrapper!(mul_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::mul_f16, "mul_elementwise");
cpu_binary_wrapper!(div_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::div_f16, "div_elementwise");
cpu_binary_wrapper!(maximum_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::maximum_f16, "maximum_elementwise");
cpu_binary_wrapper!(minimum_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::minimum_f16, "minimum_elementwise");

cpu_unary_wrapper!(relu_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::relu_f16, "relu_elementwise");
cpu_unary_wrapper!(neg_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::neg_f16, "neg_elementwise");
cpu_unary_wrapper!(sqr_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::sqr_f16, "sqr_elementwise");
cpu_unary_wrapper!(sqrt_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::sqrt_f16, "sqrt_elementwise");
cpu_unary_wrapper!(recip_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::recip_f16, "recip_elementwise");
cpu_unary_wrapper!(abs_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::abs_f16, "abs_elementwise");
cpu_unary_wrapper!(tanh_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::tanh_f16, "tanh_elementwise");
cpu_unary_wrapper!(exp_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::exp_f16, "exp_elementwise");
cpu_unary_wrapper!(log_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::log_f16, "log_elementwise");
cpu_unary_wrapper!(sin_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::sin_f16, "sin_elementwise");
cpu_unary_wrapper!(cos_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::cos_f16, "cos_elementwise");
cpu_unary_wrapper!(sigmoid_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::sigmoid_f16, "sigmoid_elementwise");
cpu_unary_wrapper!(silu_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::silu_f16, "silu_elementwise");
cpu_unary_wrapper!(gelu_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::gelu_f16, "gelu_elementwise");
cpu_unary_wrapper!(step_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::step_f16, "step_elementwise");

/// Generate a CPU argextremum wrapper. Output dtype is U32; the
/// binding-table key is keyed on the OUTPUT dtype = U32. The
/// wrapper validates the input is F32 (only F32 is wired today).
macro_rules! cpu_arg_dim_wrapper {
    ($wrapper:ident, $kernel:path, $op_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 1 input + 1 output, got {} + {}",
                    $op_name, inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let (input_layout, dims) = match params {
                OpParams::Reduce { input_layout, dims, .. } => (input_layout, dims),
                other => {
                    return Err(Error::Msg(format!(
                        "{} wrapper expects OpParams::Reduce, got {other:?}",
                        $op_name,
                    ))
                    .bt())
                }
            };
            if dims.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects exactly 1 reduce dim, got {dims:?}",
                    $op_name,
                ))
                .bt());
            }
            let dim = dims[0];
            let in_guard = read_storage(&inputs[0])?;
            if in_guard.dtype != DType::F32 {
                return Err(Error::Msg(format!(
                    "{}: only F32 input is wired today, got {:?}",
                    $op_name, in_guard.dtype,
                ))
                .bt());
            }
            let mut out_guard = write_storage(&outputs[0])?;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(in_cpu, out_cpu, input_layout.shape().dims(), dim)
        }
    };
}

cpu_arg_dim_wrapper!(
    argmax_dim_u32_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::argmax_dim_f32,
    "argmax_dim"
);
cpu_arg_dim_wrapper!(
    argmin_dim_u32_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::argmin_dim_f32,
    "argmin_dim"
);

/// Dispatch wrapper for `(IndexAdd, F32, Cpu)`. Three inputs:
/// `(base, indices, src)` (rank-1 U32 indices). Output shape == base.
fn index_add_f32_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 3 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "index_add wrapper expects 3 inputs (base, indices, src) + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let (outer_count, base_dim_size, n_indices, inner_count) = match params {
        OpParams::IndexAdd {
            outer_count, base_dim_size, n_indices, inner_count,
        } => (*outer_count, *base_dim_size, *n_indices, *inner_count),
        other => {
            return Err(Error::Msg(format!(
                "index_add wrapper expects OpParams::IndexAdd, got {other:?}",
            ))
            .bt())
        }
    };
    let base_guard = read_storage(&inputs[0])?;
    let idx_guard = read_storage(&inputs[1])?;
    if idx_guard.dtype != DType::U32 {
        return Err(Error::Msg(format!(
            "index_add: indices must be U32, got {:?}",
            idx_guard.dtype,
        ))
        .bt());
    }
    let src_guard = read_storage(&inputs[2])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let base_cpu = cpu_input(&base_guard)?;
    let idx_cpu = cpu_input(&idx_guard)?;
    let src_cpu = cpu_input(&src_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::index_add_f32(
        base_cpu, idx_cpu, src_cpu, out_cpu,
        outer_count, base_dim_size, n_indices, inner_count,
    )
}

/// Dispatch wrapper for `(ScatterAdd, F32, Cpu)`. Three inputs:
/// `(base, indices, src)` (same-rank U32 indices). Output shape == base.
fn scatter_add_f32_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 3 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "scatter_add wrapper expects 3 inputs (base, indices, src) + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let (base_shape, src_shape, dim) = match params {
        OpParams::ScatterAdd { base_shape, src_shape, dim } => (base_shape, src_shape, *dim),
        other => {
            return Err(Error::Msg(format!(
                "scatter_add wrapper expects OpParams::ScatterAdd, got {other:?}",
            ))
            .bt())
        }
    };
    let base_guard = read_storage(&inputs[0])?;
    let idx_guard = read_storage(&inputs[1])?;
    if idx_guard.dtype != DType::U32 {
        return Err(Error::Msg(format!(
            "scatter_add: indices must be U32, got {:?}",
            idx_guard.dtype,
        ))
        .bt());
    }
    let src_guard = read_storage(&inputs[2])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let base_cpu = cpu_input(&base_guard)?;
    let idx_cpu = cpu_input(&idx_guard)?;
    let src_cpu = cpu_input(&src_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::scatter_add_f32(
        base_cpu, idx_cpu, src_cpu, out_cpu,
        base_shape, src_shape, dim,
    )
}

/// Dispatch wrapper for `(Rope, F32, Cpu)`. Three inputs:
/// `(x, cos, sin)`. `OpParams::Rope` carries the geometry.
fn rope_f32_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 3 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "rope wrapper expects 3 inputs (x, cos, sin) + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let (outer_count, seq, head_dim) = match params {
        OpParams::Rope { outer_count, seq, head_dim } => {
            (*outer_count, *seq, *head_dim)
        }
        other => {
            return Err(Error::Msg(format!(
                "rope wrapper expects OpParams::Rope, got {other:?}",
            ))
            .bt())
        }
    };
    let x_guard = read_storage(&inputs[0])?;
    let cos_guard = read_storage(&inputs[1])?;
    let sin_guard = read_storage(&inputs[2])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let x_cpu = cpu_input(&x_guard)?;
    let cos_cpu = cpu_input(&cos_guard)?;
    let sin_cpu = cpu_input(&sin_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::rope_f32(
        x_cpu, cos_cpu, sin_cpu, out_cpu,
        outer_count, seq, head_dim,
    )
}

/// Dispatch wrapper for `(Gather, *, Cpu)`. Dtype-agnostic at
/// the byte level — `dtype_size` flows from the output Storage.
fn gather_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "gather wrapper expects 2 inputs + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let (source_shape, output_shape, dim) = match params {
        OpParams::Gather { source_shape, output_shape, dim } => {
            (source_shape, output_shape, *dim)
        }
        other => {
            return Err(Error::Msg(format!(
                "gather wrapper expects OpParams::Gather, got {other:?}",
            ))
            .bt())
        }
    };
    let src_guard = read_storage(&inputs[0])?;
    let idx_guard = read_storage(&inputs[1])?;
    if idx_guard.dtype != DType::U32 {
        return Err(Error::Msg(format!(
            "gather: indices must be U32, got {:?}",
            idx_guard.dtype,
        ))
        .bt());
    }
    let mut out_guard = write_storage(&outputs[0])?;
    let dtype_size = out_guard.dtype.size_in_bytes();
    let src_cpu = cpu_input(&src_guard)?;
    let idx_cpu = cpu_input(&idx_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::gather_cpu(
        src_cpu, idx_cpu, out_cpu,
        source_shape, output_shape, dim, dtype_size,
    )
}

/// Dispatch wrapper for `(IndexSelect, *, Cpu)`. Dtype-agnostic
/// at the byte level — `dtype_size` flows from the output Storage.
/// Indices are always U32 (validated at runtime).
fn index_select_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "index_select wrapper expects 2 inputs (source, indices), got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "index_select wrapper expects 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let (outer_count, source_dim_size, n_indices, inner_count) = match params {
        OpParams::IndexSelect {
            outer_count, source_dim_size, n_indices, inner_count,
        } => (*outer_count, *source_dim_size, *n_indices, *inner_count),
        other => {
            return Err(Error::Msg(format!(
                "index_select wrapper expects OpParams::IndexSelect, got {other:?}",
            ))
            .bt())
        }
    };
    let src_guard = read_storage(&inputs[0])?;
    let idx_guard = read_storage(&inputs[1])?;
    if idx_guard.dtype != DType::U32 {
        return Err(Error::Msg(format!(
            "index_select: indices must be U32, got {:?}",
            idx_guard.dtype,
        ))
        .bt());
    }
    let mut out_guard = write_storage(&outputs[0])?;
    let dtype_size = out_guard.dtype.size_in_bytes();
    let src_cpu = cpu_input(&src_guard)?;
    let idx_cpu = cpu_input(&idx_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::index_select_cpu(
        src_cpu, idx_cpu, out_cpu,
        outer_count, source_dim_size, n_indices, inner_count, dtype_size,
    )
}

/// Generate a CPU last-dim norm wrapper that pulls
/// `(outer_count, last_dim, eps)` from `OpParams::NormLastDim`
/// and forwards to a typed kernel.
macro_rules! cpu_norm_last_dim_wrapper {
    ($wrapper:ident, $kernel:path, $op_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 1 input + 1 output, got {} + {}",
                    $op_name, inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let (outer_count, last_dim, eps) = match params {
                OpParams::NormLastDim { outer_count, last_dim, eps } => {
                    (*outer_count, *last_dim, *eps)
                }
                other => {
                    return Err(Error::Msg(format!(
                        "{} wrapper expects OpParams::NormLastDim, got {other:?}",
                        $op_name,
                    ))
                    .bt())
                }
            };
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(in_cpu, out_cpu, outer_count, last_dim, eps)
        }
    };
}

cpu_norm_last_dim_wrapper!(
    rms_norm_last_dim_f32_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::rms_norm_last_dim_f32,
    "rms_norm_last_dim"
);
cpu_norm_last_dim_wrapper!(
    layer_norm_last_dim_f32_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::layer_norm_last_dim_f32,
    "layer_norm_last_dim"
);
cpu_norm_last_dim_wrapper!(
    rms_norm_last_dim_bf16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::rms_norm_last_dim_bf16,
    "rms_norm_last_dim"
);
cpu_norm_last_dim_wrapper!(
    rms_norm_last_dim_f16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::rms_norm_last_dim_f16,
    "rms_norm_last_dim"
);
cpu_norm_last_dim_wrapper!(
    layer_norm_last_dim_bf16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::layer_norm_last_dim_bf16,
    "layer_norm_last_dim"
);
cpu_norm_last_dim_wrapper!(
    layer_norm_last_dim_f16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::layer_norm_last_dim_f16,
    "layer_norm_last_dim"
);
cpu_norm_last_dim_wrapper!(
    rms_norm_last_dim_f64_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::rms_norm_last_dim_f64,
    "rms_norm_last_dim"
);
cpu_norm_last_dim_wrapper!(
    layer_norm_last_dim_f64_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::layer_norm_last_dim_f64,
    "layer_norm_last_dim"
);

/// Dispatch wrapper for `(SoftmaxLastDim, F32, Cpu)`. Single
/// input + single output; (outer_count, last_dim) flow through
/// `OpParams::SoftmaxLastDim`.
/// Generate a CPU SoftmaxLastDim wrapper for any element type.
/// All entries share the (1 input, 1 output, OpParams::SoftmaxLastDim)
/// shape; only the underlying typed kernel differs.
macro_rules! cpu_softmax_last_dim_wrapper {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "softmax_last_dim wrapper expects 1 input + 1 output, got {} + {}",
                    inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let (outer_count, last_dim) = match params {
                OpParams::SoftmaxLastDim { outer_count, last_dim } => (*outer_count, *last_dim),
                other => {
                    return Err(Error::Msg(format!(
                        "softmax_last_dim wrapper expects OpParams::SoftmaxLastDim, got {other:?}",
                    ))
                    .bt())
                }
            };
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(in_cpu, out_cpu, outer_count, last_dim)
        }
    };
}

cpu_softmax_last_dim_wrapper!(softmax_last_dim_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::softmax_last_dim_f32);
cpu_softmax_last_dim_wrapper!(softmax_last_dim_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::softmax_last_dim_f64);
cpu_softmax_last_dim_wrapper!(softmax_last_dim_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::softmax_last_dim_bf16);
cpu_softmax_last_dim_wrapper!(softmax_last_dim_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::softmax_last_dim_f16);

/// Generate a CPU Rope wrapper for any element type. The wrapper
/// shape is identical across dtypes — three inputs (x, cos, sin)
/// and `OpParams::Rope` carries the geometry.
macro_rules! cpu_rope_wrapper {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 3 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "rope wrapper expects 3 inputs + 1 output, got {} + {}",
                    inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let (outer_count, seq, head_dim) = match params {
                OpParams::Rope { outer_count, seq, head_dim } => {
                    (*outer_count, *seq, *head_dim)
                }
                other => {
                    return Err(Error::Msg(format!(
                        "rope wrapper expects OpParams::Rope, got {other:?}",
                    ))
                    .bt())
                }
            };
            let x_guard = read_storage(&inputs[0])?;
            let cos_guard = read_storage(&inputs[1])?;
            let sin_guard = read_storage(&inputs[2])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let x_cpu = cpu_input(&x_guard)?;
            let cos_cpu = cpu_input(&cos_guard)?;
            let sin_cpu = cpu_input(&sin_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(x_cpu, cos_cpu, sin_cpu, out_cpu, outer_count, seq, head_dim)
        }
    };
}

cpu_rope_wrapper!(rope_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::rope_bf16);
cpu_rope_wrapper!(rope_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::rope_f16);
cpu_rope_wrapper!(rope_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::rope_f64);

/// Dispatch wrapper for `(QMatMul, F32, Cpu)`. Two inputs:
/// activations (F32) and quantized weight bytes (U32-typed).
/// `OpParams::QMatMul` carries the quant_type + (batch, m, n, k);
/// the wrapper picks the right typed kernel based on quant_type.
fn qmatmul_f32_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "qmatmul wrapper expects 2 inputs (activations, weight_bytes), got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "qmatmul wrapper expects 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let (quant_type, batch_count, m, n, k) = match params {
        OpParams::QMatMul { quant_type, batch_count, m, n, k } => {
            (*quant_type, *batch_count, *m, *n, *k)
        }
        other => {
            return Err(Error::Msg(format!(
                "qmatmul wrapper expects OpParams::QMatMul, got {other:?}",
            ))
            .bt())
        }
    };
    let act_guard = read_storage(&inputs[0])?;
    let w_guard = read_storage(&inputs[1])?;
    if w_guard.dtype != DType::U32 {
        return Err(Error::Msg(format!(
            "qmatmul: weight bytes must be U32-typed (raw block stream), got {:?}",
            w_guard.dtype,
        ))
        .bt());
    }
    let mut out_guard = write_storage(&outputs[0])?;
    let act_cpu = cpu_input(&act_guard)?;
    let w_cpu = cpu_input(&w_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    use fuel_graph::QuantType;
    match quant_type {
        QuantType::Q4_0 => fuel_cpu_backend::byte_kernels::qmatmul_q4_0_f32(
            act_cpu, w_cpu, out_cpu, batch_count, m, n, k,
        ),
        QuantType::Q4_1 => fuel_cpu_backend::byte_kernels::qmatmul_q4_1_f32(
            act_cpu, w_cpu, out_cpu, batch_count, m, n, k,
        ),
        QuantType::Q5_0 => fuel_cpu_backend::byte_kernels::qmatmul_q5_0_f32(
            act_cpu, w_cpu, out_cpu, batch_count, m, n, k,
        ),
        QuantType::Q5_1 => fuel_cpu_backend::byte_kernels::qmatmul_q5_1_f32(
            act_cpu, w_cpu, out_cpu, batch_count, m, n, k,
        ),
        QuantType::Q8_0 => fuel_cpu_backend::byte_kernels::qmatmul_q8_0_f32(
            act_cpu, w_cpu, out_cpu, batch_count, m, n, k,
        ),
        QuantType::Q8_1 => fuel_cpu_backend::byte_kernels::qmatmul_q8_1_f32(
            act_cpu, w_cpu, out_cpu, batch_count, m, n, k,
        ),
        QuantType::Q2K => fuel_cpu_backend::byte_kernels::qmatmul_q2k_f32(
            act_cpu, w_cpu, out_cpu, batch_count, m, n, k,
        ),
        QuantType::Q3K => fuel_cpu_backend::byte_kernels::qmatmul_q3k_f32(
            act_cpu, w_cpu, out_cpu, batch_count, m, n, k,
        ),
        QuantType::Q4_K_M => fuel_cpu_backend::byte_kernels::qmatmul_q4_k_m_f32(
            act_cpu, w_cpu, out_cpu, batch_count, m, n, k,
        ),
        QuantType::Q5K => fuel_cpu_backend::byte_kernels::qmatmul_q5k_f32(
            act_cpu, w_cpu, out_cpu, batch_count, m, n, k,
        ),
        QuantType::Q6K => fuel_cpu_backend::byte_kernels::qmatmul_q6k_f32(
            act_cpu, w_cpu, out_cpu, batch_count, m, n, k,
        ),
    }
}

/// Dispatch wrapper for `(Concat, *, Cpu)`. Dtype-agnostic — the
/// underlying kernel is `concat_cpu(... dtype_size)`. The wrapper
/// reads dtype_size from the output Storage's dtype tag.
fn concat_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()> {
    if inputs.is_empty() {
        return Err(Error::Msg("concat wrapper expects ≥ 1 input, got 0".to_string()).bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "concat wrapper expects 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let (outer_count, input_dim_sizes, inner_count) = match params {
        OpParams::Concat { outer_count, input_dim_sizes, inner_count } => {
            (*outer_count, input_dim_sizes, *inner_count)
        }
        other => {
            return Err(Error::Msg(format!(
                "concat wrapper expects OpParams::Concat, got {other:?}",
            ))
            .bt())
        }
    };
    if input_dim_sizes.len() != inputs.len() {
        return Err(Error::Msg(format!(
            "concat wrapper: OpParams declares {} inputs but the work item \
             carries {}",
            input_dim_sizes.len(),
            inputs.len(),
        ))
        .bt());
    }
    let in_guards: Vec<_> = inputs
        .iter()
        .map(read_storage)
        .collect::<Result<Vec<_>>>()?;
    let mut in_cpus: Vec<&fuel_cpu_backend::CpuStorageBytes> = Vec::with_capacity(in_guards.len());
    for g in &in_guards {
        in_cpus.push(cpu_input(g)?);
    }
    let mut out_guard = write_storage(&outputs[0])?;
    let dtype_size = out_guard.dtype.size_in_bytes();
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::concat_cpu(
        &in_cpus,
        out_cpu,
        outer_count,
        input_dim_sizes,
        inner_count,
        dtype_size,
    )
}

/// Generate a CPU IndexAdd wrapper. Same shape across all dtypes;
/// only the underlying typed kernel differs.
macro_rules! cpu_index_add_wrapper {
    ($wrapper:ident, $kernel:path, $idx_ck:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 3 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{}: expects 3 inputs + 1 output, got {} + {}",
                    $idx_ck, inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let (outer_count, base_dim_size, n_indices, inner_count) = match params {
                OpParams::IndexAdd {
                    outer_count, base_dim_size, n_indices, inner_count,
                } => (*outer_count, *base_dim_size, *n_indices, *inner_count),
                other => {
                    return Err(Error::Msg(format!(
                        "{}: expects OpParams::IndexAdd, got {other:?}", $idx_ck,
                    ))
                    .bt())
                }
            };
            let base_guard = read_storage(&inputs[0])?;
            let idx_guard = read_storage(&inputs[1])?;
            if idx_guard.dtype != DType::U32 {
                return Err(Error::Msg(format!(
                    "{}: indices must be U32", $idx_ck,
                ))
                .bt());
            }
            let src_guard = read_storage(&inputs[2])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let base_cpu = cpu_input(&base_guard)?;
            let idx_cpu = cpu_input(&idx_guard)?;
            let src_cpu = cpu_input(&src_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(
                base_cpu, idx_cpu, src_cpu, out_cpu,
                outer_count, base_dim_size, n_indices, inner_count,
            )
        }
    };
}

cpu_index_add_wrapper!(index_add_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::index_add_f64,  "index_add_f64");
cpu_index_add_wrapper!(index_add_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::index_add_bf16, "index_add_bf16");
cpu_index_add_wrapper!(index_add_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::index_add_f16,  "index_add_f16");

/// Generate a CPU ScatterAdd wrapper.
macro_rules! cpu_scatter_add_wrapper {
    ($wrapper:ident, $kernel:path, $name_str:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 3 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{}: expects 3 inputs + 1 output, got {} + {}",
                    $name_str, inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let (base_shape, src_shape, dim) = match params {
                OpParams::ScatterAdd { base_shape, src_shape, dim } => (base_shape, src_shape, *dim),
                other => {
                    return Err(Error::Msg(format!(
                        "{}: expects OpParams::ScatterAdd, got {other:?}", $name_str,
                    ))
                    .bt())
                }
            };
            let base_guard = read_storage(&inputs[0])?;
            let idx_guard = read_storage(&inputs[1])?;
            if idx_guard.dtype != DType::U32 {
                return Err(Error::Msg(format!(
                    "{}: indices must be U32", $name_str,
                ))
                .bt());
            }
            let src_guard = read_storage(&inputs[2])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let base_cpu = cpu_input(&base_guard)?;
            let idx_cpu = cpu_input(&idx_guard)?;
            let src_cpu = cpu_input(&src_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(
                base_cpu, idx_cpu, src_cpu, out_cpu,
                base_shape, src_shape, dim,
            )
        }
    };
}

cpu_scatter_add_wrapper!(scatter_add_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::scatter_add_f64,  "scatter_add_f64");
cpu_scatter_add_wrapper!(scatter_add_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::scatter_add_bf16, "scatter_add_bf16");
cpu_scatter_add_wrapper!(scatter_add_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::scatter_add_f16,  "scatter_add_f16");

/// Build a CPU Affine wrapper for any element type. Cast from f64
/// scalars in OpParams::Affine to the target arithmetic type
/// happens inside the macro (different cast for f64 vs half).
macro_rules! cpu_affine_wrapper_native {
    ($wrapper:ident, $kernel:path, $T:ty) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg("affine: 1 input + 1 output".to_string()).bt());
            }
            let (mul, add) = match params {
                OpParams::Affine { mul, add } => (*mul as $T, *add as $T),
                other => return Err(Error::Msg(format!("affine: bad params {other:?}")).bt()),
            };
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(in_cpu, out_cpu, mul, add)
        }
    };
}

cpu_affine_wrapper_native!(affine_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::affine_f64, f64);

macro_rules! cpu_affine_wrapper_half {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg("affine: 1 input + 1 output".to_string()).bt());
            }
            let (mul, add) = match params {
                OpParams::Affine { mul, add } => (*mul as f32, *add as f32),
                other => return Err(Error::Msg(format!("affine: bad params {other:?}")).bt()),
            };
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(in_cpu, out_cpu, mul, add)
        }
    };
}

cpu_affine_wrapper_half!(affine_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::affine_bf16);
cpu_affine_wrapper_half!(affine_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::affine_f16);

macro_rules! cpu_clamp_wrapper {
    ($wrapper:ident, $kernel:path, $T:ty) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg("clamp: 1 input + 1 output".to_string()).bt());
            }
            let (min, max) = match params {
                OpParams::Clamp { min, max } => (*min as $T, *max as $T),
                other => return Err(Error::Msg(format!("clamp: bad params {other:?}")).bt()),
            };
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(in_cpu, out_cpu, min, max)
        }
    };
}

cpu_clamp_wrapper!(clamp_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::clamp_f64,  f64);
cpu_clamp_wrapper!(clamp_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::clamp_bf16, f32);
cpu_clamp_wrapper!(clamp_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::clamp_f16,  f32);

macro_rules! cpu_powi_wrapper {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg("powi: 1 input + 1 output".to_string()).bt());
            }
            let exp = match params {
                OpParams::PowI { exp } => *exp,
                other => return Err(Error::Msg(format!("powi: bad params {other:?}")).bt()),
            };
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(in_cpu, out_cpu, exp)
        }
    };
}

cpu_powi_wrapper!(powi_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::powi_f64);
cpu_powi_wrapper!(powi_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::powi_bf16);
cpu_powi_wrapper!(powi_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::powi_f16);

// ArgMax/ArgMin per-input-dtype wrappers. The existing
// `cpu_arg_dim_wrapper!` macro hardcodes F32 input — generalize it
// for non-F32 input dtypes.
macro_rules! cpu_arg_dim_wrapper_typed {
    ($wrapper:ident, $kernel:path, $expected_in_dtype:expr, $op_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{}: 1 input + 1 output", $op_name,
                ))
                .bt());
            }
            let (input_layout, dims) = match params {
                OpParams::Reduce { input_layout, dims, .. } => (input_layout, dims),
                other => return Err(Error::Msg(format!(
                    "{}: expects OpParams::Reduce, got {other:?}", $op_name,
                ))
                .bt()),
            };
            if dims.len() != 1 {
                return Err(Error::Msg(format!(
                    "{}: expects exactly 1 reduce dim, got {dims:?}", $op_name,
                ))
                .bt());
            }
            let dim = dims[0];
            let in_guard = read_storage(&inputs[0])?;
            if in_guard.dtype != $expected_in_dtype {
                return Err(Error::Msg(format!(
                    "{}: expects input dtype {:?}, got {:?}",
                    $op_name, $expected_in_dtype, in_guard.dtype,
                ))
                .bt());
            }
            let mut out_guard = write_storage(&outputs[0])?;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(in_cpu, out_cpu, input_layout.shape().dims(), dim)
        }
    };
}

// New ArgMax/ArgMin entries are keyed on output U32 like the
// existing ones, but we need separate wrappers per input dtype.
// Use a dispatch macro that combines them.
fn argmax_dim_u32_cpu_dispatch(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()> {
    let in_dtype = read_storage(&inputs[0])?.dtype;
    match in_dtype {
        DType::F32 => argmax_dim_u32_cpu_wrapper(inputs, outputs, params),
        DType::F64 => argmax_dim_f64_only_wrapper(inputs, outputs, params),
        DType::BF16 => argmax_dim_bf16_only_wrapper(inputs, outputs, params),
        DType::F16 => argmax_dim_f16_only_wrapper(inputs, outputs, params),
        other => Err(Error::Msg(format!(
            "argmax_dim: unsupported input dtype {other:?}",
        ))
        .bt()),
    }
}

fn argmin_dim_u32_cpu_dispatch(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()> {
    let in_dtype = read_storage(&inputs[0])?.dtype;
    match in_dtype {
        DType::F32 => argmin_dim_u32_cpu_wrapper(inputs, outputs, params),
        DType::F64 => argmin_dim_f64_only_wrapper(inputs, outputs, params),
        DType::BF16 => argmin_dim_bf16_only_wrapper(inputs, outputs, params),
        DType::F16 => argmin_dim_f16_only_wrapper(inputs, outputs, params),
        other => Err(Error::Msg(format!(
            "argmin_dim: unsupported input dtype {other:?}",
        ))
        .bt()),
    }
}

cpu_arg_dim_wrapper_typed!(argmax_dim_f64_only_wrapper,  fuel_cpu_backend::byte_kernels::argmax_dim_f64,  DType::F64,  "argmax_dim_f64");
cpu_arg_dim_wrapper_typed!(argmin_dim_f64_only_wrapper,  fuel_cpu_backend::byte_kernels::argmin_dim_f64,  DType::F64,  "argmin_dim_f64");
cpu_arg_dim_wrapper_typed!(argmax_dim_bf16_only_wrapper, fuel_cpu_backend::byte_kernels::argmax_dim_bf16, DType::BF16, "argmax_dim_bf16");
cpu_arg_dim_wrapper_typed!(argmin_dim_bf16_only_wrapper, fuel_cpu_backend::byte_kernels::argmin_dim_bf16, DType::BF16, "argmin_dim_bf16");
cpu_arg_dim_wrapper_typed!(argmax_dim_f16_only_wrapper,  fuel_cpu_backend::byte_kernels::argmax_dim_f16,  DType::F16,  "argmax_dim_f16");
cpu_arg_dim_wrapper_typed!(argmin_dim_f16_only_wrapper,  fuel_cpu_backend::byte_kernels::argmin_dim_f16,  DType::F16,  "argmin_dim_f16");

/// Dispatch wrapper for `(Affine, F32, Cpu)`. Extracts scalar
/// coefficients from `OpParams::Affine`.
fn affine_f32_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "affine wrapper expects 1 input + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let (mul, add) = match params {
        OpParams::Affine { mul, add } => (*mul as f32, *add as f32),
        other => {
            return Err(Error::Msg(format!(
                "affine wrapper expects OpParams::Affine, got {other:?}",
            ))
            .bt())
        }
    };
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let in_cpu = cpu_input(&in_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::affine_f32(in_cpu, out_cpu, mul, add)
}

/// Dispatch wrapper for `(ClampElementwise, F32, Cpu)`.
fn clamp_elementwise_f32_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "clamp wrapper expects 1 input + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let (min, max) = match params {
        OpParams::Clamp { min, max } => (*min as f32, *max as f32),
        other => {
            return Err(Error::Msg(format!(
                "clamp wrapper expects OpParams::Clamp, got {other:?}",
            ))
            .bt())
        }
    };
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let in_cpu = cpu_input(&in_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::clamp_f32(in_cpu, out_cpu, min, max)
}

/// Dispatch wrapper for `(PowIElementwise, F32, Cpu)`.
fn powi_elementwise_f32_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "powi wrapper expects 1 input + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let exp = match params {
        OpParams::PowI { exp } => *exp,
        other => {
            return Err(Error::Msg(format!(
                "powi wrapper expects OpParams::PowI, got {other:?}",
            ))
            .bt())
        }
    };
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let in_cpu = cpu_input(&in_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::powi_f32(in_cpu, out_cpu, exp)
}

/// Build a CPU reduction wrapper that calls a typed `(input,
/// output, input_shape, dims)` reduce kernel. Verifies the
/// `OpParams::Reduce` variant and forwards the shape + dims to the
/// kernel.
macro_rules! cpu_reduce_wrapper {
    ($wrapper:ident, $kernel:path, $op_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 1 input, got {}",
                    $op_name,
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 1 output, got {}",
                    $op_name,
                    outputs.len(),
                ))
                .bt());
            }
            let (input_layout, dims) = match params {
                OpParams::Reduce { input_layout, dims, .. } => (input_layout, dims),
                other => {
                    return Err(Error::Msg(format!(
                        "{} wrapper expects OpParams::Reduce, got {:?}",
                        $op_name, other,
                    ))
                    .bt())
                }
            };
            // The pipelined executor's auto-Contiguize pass
            // (stage 4 of Layout-on-Node) materializes contiguous
            // bytes for every kernel input before this wrapper is
            // called — so by the time we reach here, the input
            // bytes match `input_layout.shape()` in row-major
            // order. We pass only the shape to the typed kernel.
            let input_shape: &[usize] = input_layout.shape().dims();
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(in_cpu, out_cpu, input_shape, dims)
        }
    };
}

cpu_reduce_wrapper!(sum_reduce_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::sum_reduce_f32, "sum_reduce");
cpu_reduce_wrapper!(max_reduce_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::max_reduce_f32, "max_reduce");
cpu_reduce_wrapper!(min_reduce_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::min_reduce_f32, "min_reduce");
cpu_reduce_wrapper!(mean_reduce_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::mean_reduce_f32, "mean_reduce");
cpu_reduce_wrapper!(sum_reduce_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::sum_reduce_f64, "sum_reduce");
cpu_reduce_wrapper!(max_reduce_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::max_reduce_f64, "max_reduce");
cpu_reduce_wrapper!(min_reduce_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::min_reduce_f64, "min_reduce");
cpu_reduce_wrapper!(mean_reduce_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::mean_reduce_f64, "mean_reduce");

cpu_reduce_wrapper!(sum_reduce_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::sum_reduce_bf16, "sum_reduce");
cpu_reduce_wrapper!(max_reduce_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::max_reduce_bf16, "max_reduce");
cpu_reduce_wrapper!(min_reduce_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::min_reduce_bf16, "min_reduce");
cpu_reduce_wrapper!(mean_reduce_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::mean_reduce_bf16, "mean_reduce");

cpu_reduce_wrapper!(sum_reduce_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::sum_reduce_f16, "sum_reduce");
cpu_reduce_wrapper!(max_reduce_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::max_reduce_f16, "max_reduce");
cpu_reduce_wrapper!(min_reduce_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::min_reduce_f16, "min_reduce");
cpu_reduce_wrapper!(mean_reduce_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::mean_reduce_f16, "mean_reduce");

/// Generate a dispatch wrapper for `(Cast, <target>, Cpu)`. The
/// binding-table key is keyed on the *target* dtype (= the
/// Node's dtype = the output Storage's dtype); the wrapper reads
/// the input Storage's dtype at runtime and dispatches to the
/// right typed conversion kernel.
macro_rules! cpu_cast_wrapper {
    (
        $wrapper:ident,
        $target_dtype:expr,
        $target_name:literal,
        { $( $source:pat => $kernel:path ),+ $(,)? } $(,)?
    ) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "cast→{} wrapper expects 1 input, got {}",
                    $target_name,
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "cast→{} wrapper expects 1 output, got {}",
                    $target_name,
                    outputs.len(),
                ))
                .bt());
            }
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let source_dtype = in_guard.dtype;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            match source_dtype {
                $( $source => $kernel(in_cpu, out_cpu), )+
                other => Err(Error::Msg(format!(
                    "cast→{}: source dtype {:?} not yet wired (Phase C \
                     extends the cast matrix as needed)",
                    $target_name, other,
                ))
                .bt()),
            }
        }
    };
}

/// Dispatch wrapper for `(Conv2D, *, Cpu)`. Two or three inputs
/// (x, weight, optional bias). Shapes + geometry flow through
/// `OpParams::Conv2D`.
macro_rules! cpu_conv2d_wrapper {
    ($name:ident, $kernel:path) => {
        fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 && inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "conv2d wrapper expects 2 or 3 inputs (x, weight, [bias]), got {}",
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "conv2d wrapper expects 1 output, got {}",
                    outputs.len(),
                ))
                .bt());
            }
            let (x_shape, w_shape, out_shape, stride, padding, dilation, groups) = match params {
                OpParams::Conv2D {
                    x_shape, w_shape, out_shape, stride, padding, dilation, groups,
                } => (*x_shape, *w_shape, *out_shape, *stride, *padding, *dilation, *groups),
                other => {
                    return Err(Error::Msg(format!(
                        "conv2d wrapper expects OpParams::Conv2D, got {other:?}",
                    ))
                    .bt())
                }
            };
            let x_guard = read_storage(&inputs[0])?;
            let w_guard = read_storage(&inputs[1])?;
            let bias_guard = match inputs.get(2) {
                Some(arc) => Some(read_storage(arc)?),
                None => None,
            };
            let mut out_guard = write_storage(&outputs[0])?;
            let x_cpu = cpu_input(&x_guard)?;
            let w_cpu = cpu_input(&w_guard)?;
            let bias_cpu = match &bias_guard {
                Some(g) => Some(cpu_input(g)?),
                None => None,
            };
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(
                x_cpu, w_cpu, bias_cpu, out_cpu,
                x_shape, w_shape, out_shape,
                stride, padding, dilation, groups,
            )
        }
    };
}

cpu_conv2d_wrapper!(conv2d_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::conv2d_f32);
cpu_conv2d_wrapper!(conv2d_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::conv2d_f64);
cpu_conv2d_wrapper!(conv2d_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::conv2d_bf16);
cpu_conv2d_wrapper!(conv2d_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::conv2d_f16);

/// Dispatch wrapper for `(ConvTranspose2D, *, Cpu)`. Two or three
/// inputs (x, weight, [bias]). Geometry flows through
/// `OpParams::ConvTranspose2D`.
macro_rules! cpu_conv_transpose2d_wrapper {
    ($name:ident, $kernel:path) => {
        fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 && inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "conv_transpose2d wrapper expects 2 or 3 inputs (x, weight, [bias]), got {}",
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "conv_transpose2d wrapper expects 1 output, got {}",
                    outputs.len(),
                ))
                .bt());
            }
            let (x_shape, w_shape, out_shape, stride, padding, dilation, groups) = match params {
                OpParams::ConvTranspose2D {
                    x_shape, w_shape, out_shape, stride, padding,
                    output_padding: _, dilation, groups,
                } => (*x_shape, *w_shape, *out_shape, *stride, *padding, *dilation, *groups),
                other => {
                    return Err(Error::Msg(format!(
                        "conv_transpose2d wrapper expects OpParams::ConvTranspose2D, got {other:?}",
                    ))
                    .bt())
                }
            };
            let x_guard = read_storage(&inputs[0])?;
            let w_guard = read_storage(&inputs[1])?;
            let bias_guard = match inputs.get(2) {
                Some(arc) => Some(read_storage(arc)?),
                None => None,
            };
            let mut out_guard = write_storage(&outputs[0])?;
            let x_cpu = cpu_input(&x_guard)?;
            let w_cpu = cpu_input(&w_guard)?;
            let bias_cpu = match &bias_guard {
                Some(g) => Some(cpu_input(g)?),
                None => None,
            };
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(
                x_cpu, w_cpu, bias_cpu, out_cpu,
                x_shape, w_shape, out_shape,
                stride, padding, dilation, groups,
            )
        }
    };
}

cpu_conv_transpose2d_wrapper!(conv_transpose2d_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::conv_transpose2d_f32);
cpu_conv_transpose2d_wrapper!(conv_transpose2d_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::conv_transpose2d_f64);
cpu_conv_transpose2d_wrapper!(conv_transpose2d_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::conv_transpose2d_bf16);
cpu_conv_transpose2d_wrapper!(conv_transpose2d_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::conv_transpose2d_f16);

/// Dispatch wrapper for `(ReduceSumTo, *, Cpu)`. Single input → single
/// output; shapes flow through `OpParams::ReduceSumTo`.
macro_rules! cpu_reduce_sum_to_wrapper {
    ($name:ident, $kernel:path) => {
        fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "reduce_sum_to wrapper expects 1 input, got {}",
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "reduce_sum_to wrapper expects 1 output, got {}",
                    outputs.len(),
                ))
                .bt());
            }
            let (input_shape, output_shape) = match params {
                OpParams::ReduceSumTo { input_shape, output_shape } => {
                    (input_shape.clone(), output_shape.clone())
                }
                other => {
                    return Err(Error::Msg(format!(
                        "reduce_sum_to wrapper expects OpParams::ReduceSumTo, got {other:?}",
                    ))
                    .bt())
                }
            };
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(in_cpu, out_cpu, &input_shape, &output_shape)
        }
    };
}

cpu_reduce_sum_to_wrapper!(reduce_sum_to_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::reduce_sum_to_f32);
cpu_reduce_sum_to_wrapper!(reduce_sum_to_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::reduce_sum_to_f64);
cpu_reduce_sum_to_wrapper!(reduce_sum_to_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::reduce_sum_to_bf16);
cpu_reduce_sum_to_wrapper!(reduce_sum_to_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::reduce_sum_to_f16);

/// Dispatch wrapper for `(FusedLinear, *, Cpu)`. Three inputs
/// (lhs, rhs, bias). Reuses `OpParams::Matmul` for shape.
macro_rules! cpu_fused_linear_wrapper {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "fused_linear wrapper expects 3 inputs (lhs, rhs, bias), got {}",
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "fused_linear wrapper expects 1 output, got {}",
                    outputs.len(),
                ))
                .bt());
            }
            let (lhs_batch_dims, rhs_batch_dims, m, n, k) = match params {
                OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k } => {
                    (lhs_batch_dims, rhs_batch_dims, *m, *n, *k)
                }
                other => {
                    return Err(Error::Msg(format!(
                        "fused_linear wrapper expects OpParams::Matmul, got {other:?}",
                    ))
                    .bt())
                }
            };
            let lhs_guard = read_storage(&inputs[0])?;
            let rhs_guard = read_storage(&inputs[1])?;
            let bias_guard = read_storage(&inputs[2])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let lhs_cpu = cpu_input(&lhs_guard)?;
            let rhs_cpu = cpu_input(&rhs_guard)?;
            let bias_cpu = cpu_input(&bias_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(
                lhs_cpu, rhs_cpu, bias_cpu, out_cpu,
                lhs_batch_dims, rhs_batch_dims, m, n, k,
            )
        }
    };
}

cpu_fused_linear_wrapper!(fused_linear_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::fused_linear_f32);
cpu_fused_linear_wrapper!(fused_linear_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::fused_linear_f64);
cpu_fused_linear_wrapper!(fused_linear_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::fused_linear_bf16);
cpu_fused_linear_wrapper!(fused_linear_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::fused_linear_f16);

/// Dispatch wrapper for `(FlashAttn, *, Cpu)`. Three or four inputs
/// (q, k, v, optional alibi_slopes). Geometry + math params flow
/// through `OpParams::FlashAttn`.
macro_rules! cpu_flash_attn_wrapper {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 3 && inputs.len() != 4 {
                return Err(Error::Msg(format!(
                    "flash_attn wrapper expects 3 or 4 inputs (q, k, v, [alibi]), got {}",
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "flash_attn wrapper expects 1 output, got {}",
                    outputs.len(),
                ))
                .bt());
            }
            let (b, hq, hkv, sq, sk, d, scale, causal, wl, wr, softcap) = match params {
                OpParams::FlashAttn {
                    b, hq, hkv, sq, sk, d,
                    softmax_scale, causal,
                    window_size_left, window_size_right, softcap,
                } => (
                    *b, *hq, *hkv, *sq, *sk, *d,
                    *softmax_scale, *causal,
                    *window_size_left, *window_size_right, *softcap,
                ),
                other => {
                    return Err(Error::Msg(format!(
                        "flash_attn wrapper expects OpParams::FlashAttn, got {other:?}",
                    ))
                    .bt())
                }
            };
            let q_guard = read_storage(&inputs[0])?;
            let k_guard = read_storage(&inputs[1])?;
            let v_guard = read_storage(&inputs[2])?;
            let alibi_guard = match inputs.get(3) {
                Some(arc) => Some(read_storage(arc)?),
                None => None,
            };
            let mut out_guard = write_storage(&outputs[0])?;
            let q_cpu = cpu_input(&q_guard)?;
            let k_cpu = cpu_input(&k_guard)?;
            let v_cpu = cpu_input(&v_guard)?;
            let alibi_cpu = match &alibi_guard {
                Some(g) => Some(cpu_input(g)?),
                None => None,
            };
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(
                q_cpu, k_cpu, v_cpu, alibi_cpu, out_cpu,
                b, hq, hkv, sq, sk, d,
                scale, causal, wl, wr, softcap,
            )
        }
    };
}

cpu_flash_attn_wrapper!(flash_attn_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_f32);
cpu_flash_attn_wrapper!(flash_attn_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_f64);
cpu_flash_attn_wrapper!(flash_attn_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::flash_attn_bf16);
cpu_flash_attn_wrapper!(flash_attn_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_f16);

/// Dispatch wrapper for `(PagedAttn, *, Cpu)`. 5 or 6 inputs (q,
/// k_cache, v_cache, block_table, context_lens, optional alibi_slopes).
macro_rules! cpu_paged_attn_wrapper {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 5 && inputs.len() != 6 {
                return Err(Error::Msg(format!(
                    "paged_attn wrapper expects 5 or 6 inputs, got {}",
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "paged_attn wrapper expects 1 output, got {}",
                    outputs.len(),
                ))
                .bt());
            }
            let (b, hq, hkv, sq, d, block_size, max_blocks_per_seq, num_blocks, scale, softcap) =
                match params {
                    OpParams::PagedAttn {
                        b, hq, hkv, sq, d,
                        block_size, max_blocks_per_seq, num_blocks,
                        softmax_scale, softcap,
                    } => (
                        *b, *hq, *hkv, *sq, *d,
                        *block_size, *max_blocks_per_seq, *num_blocks,
                        *softmax_scale, *softcap,
                    ),
                    other => {
                        return Err(Error::Msg(format!(
                            "paged_attn wrapper expects OpParams::PagedAttn, got {other:?}",
                        ))
                        .bt())
                    }
                };
            let q_g = read_storage(&inputs[0])?;
            let kc_g = read_storage(&inputs[1])?;
            let vc_g = read_storage(&inputs[2])?;
            let bt_g = read_storage(&inputs[3])?;
            let cl_g = read_storage(&inputs[4])?;
            let alibi_g = match inputs.get(5) {
                Some(arc) => Some(read_storage(arc)?),
                None => None,
            };
            if bt_g.dtype != DType::U32 || cl_g.dtype != DType::U32 {
                return Err(Error::Msg(format!(
                    "paged_attn: block_table and context_lens must be U32, got {:?} / {:?}",
                    bt_g.dtype, cl_g.dtype,
                ))
                .bt());
            }
            let mut out_guard = write_storage(&outputs[0])?;
            let q_cpu = cpu_input(&q_g)?;
            let kc_cpu = cpu_input(&kc_g)?;
            let vc_cpu = cpu_input(&vc_g)?;
            let bt_cpu = cpu_input(&bt_g)?;
            let cl_cpu = cpu_input(&cl_g)?;
            let alibi_cpu = match &alibi_g {
                Some(g) => Some(cpu_input(g)?),
                None => None,
            };
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(
                q_cpu, kc_cpu, vc_cpu, bt_cpu, cl_cpu, alibi_cpu, out_cpu,
                b, hq, hkv, sq, d,
                block_size, max_blocks_per_seq, num_blocks,
                scale, softcap,
            )
        }
    };
}

cpu_paged_attn_wrapper!(paged_attn_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::paged_attn_f32);
cpu_paged_attn_wrapper!(paged_attn_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::paged_attn_f64);
cpu_paged_attn_wrapper!(paged_attn_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::paged_attn_bf16);
cpu_paged_attn_wrapper!(paged_attn_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::paged_attn_f16);

cpu_cast_wrapper!(
    cast_to_f32_cpu_wrapper,
    DType::F32,
    "f32",
    {
        DType::F64  => fuel_cpu_backend::byte_kernels::cast_f64_to_f32,
        DType::BF16 => fuel_cpu_backend::byte_kernels::cast_bf16_to_f32,
        DType::F16  => fuel_cpu_backend::byte_kernels::cast_f16_to_f32,
    },
);
cpu_cast_wrapper!(
    cast_to_f64_cpu_wrapper,
    DType::F64,
    "f64",
    {
        DType::F32 => fuel_cpu_backend::byte_kernels::cast_f32_to_f64,
    },
);
cpu_cast_wrapper!(
    cast_to_bf16_cpu_wrapper,
    DType::BF16,
    "bf16",
    {
        DType::F32 => fuel_cpu_backend::byte_kernels::cast_f32_to_bf16,
    },
);
cpu_cast_wrapper!(
    cast_to_f16_cpu_wrapper,
    DType::F16,
    "f16",
    {
        DType::F32 => fuel_cpu_backend::byte_kernels::cast_f32_to_f16,
    },
);

/// Dispatch wrapper for `(MatMul, F32, Cpu)`. Extracts the
/// `OpParams::Matmul { m, n, k }` and forwards to the typed
/// kernel. Both inputs are guaranteed contiguous f32 by the
/// executor's auto-Contiguize pass.
fn matmul_f32_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "matmul wrapper expects 2 inputs, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "matmul wrapper expects 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let (lhs_batch_dims, rhs_batch_dims, m, n, k) = match params {
        OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k } => {
            (lhs_batch_dims, rhs_batch_dims, *m, *n, *k)
        }
        other => {
            return Err(Error::Msg(format!(
                "matmul wrapper expects OpParams::Matmul, got {other:?}",
            ))
            .bt())
        }
    };
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cpu = cpu_input(&lhs_guard)?;
    let rhs_cpu = cpu_input(&rhs_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::matmul_f32(
        lhs_cpu,
        rhs_cpu,
        out_cpu,
        lhs_batch_dims,
        rhs_batch_dims,
        m,
        n,
        k,
    )
}

/// Build a CPU matmul wrapper for any element type. The wrapper
/// shape is identical across dtypes — it just forwards to a typed
/// kernel that has the matching `(lhs, rhs, out, lhs_batch_dims,
/// rhs_batch_dims, m, n, k) -> Result<()>` signature.
macro_rules! cpu_matmul_wrapper {
    ($wrapper:ident, $kernel:path, $type_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 {
                return Err(Error::Msg(format!(
                    "matmul wrapper expects 2 inputs, got {}", inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "matmul wrapper expects 1 output, got {}", outputs.len(),
                ))
                .bt());
            }
            let (lhs_batch_dims, rhs_batch_dims, m, n, k) = match params {
                OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k } => {
                    (lhs_batch_dims, rhs_batch_dims, *m, *n, *k)
                }
                other => {
                    return Err(Error::Msg(format!(
                        "matmul wrapper expects OpParams::Matmul, got {other:?}",
                    ))
                    .bt())
                }
            };
            let lhs_guard = read_storage(&inputs[0])?;
            let rhs_guard = read_storage(&inputs[1])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let lhs_cpu = cpu_input(&lhs_guard)?;
            let rhs_cpu = cpu_input(&rhs_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            let _ = $type_name;
            $kernel(
                lhs_cpu, rhs_cpu, out_cpu,
                lhs_batch_dims, rhs_batch_dims, m, n, k,
            )
        }
    };
}

cpu_matmul_wrapper!(matmul_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::matmul_bf16, "matmul_bf16");
cpu_matmul_wrapper!(matmul_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::matmul_f16,  "matmul_f16");

/// f64 mirror of [`matmul_f32_cpu_wrapper`]. Same OpKind
/// (MatMul); the binding-table key picks this entry when the
/// node's dtype is F64.
fn matmul_f64_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "matmul wrapper expects 2 inputs, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "matmul wrapper expects 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let (lhs_batch_dims, rhs_batch_dims, m, n, k) = match params {
        OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k } => {
            (lhs_batch_dims, rhs_batch_dims, *m, *n, *k)
        }
        other => {
            return Err(Error::Msg(format!(
                "matmul wrapper expects OpParams::Matmul, got {other:?}",
            ))
            .bt())
        }
    };
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cpu = cpu_input(&lhs_guard)?;
    let rhs_cpu = cpu_input(&rhs_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::matmul_f64(
        lhs_cpu, rhs_cpu, out_cpu,
        lhs_batch_dims, rhs_batch_dims, m, n, k,
    )
}

/// Register CPU dispatch wrappers in the binding table. Call once
/// at process startup or on first table creation. The CPU backend
/// is the universal fallback; its bindings cover every standard
/// (op, dtype) combination after Phase C migration. Today's set
/// covers the elementwise binary + unary `f32` families.
pub fn register_cpu_kernels(table: &mut KernelBindingTable) {
    use OpKind::*;
    let cpu = BackendId::Cpu;
    let f32_dt = DType::F32;

    table.register(AddElementwise,   f32_dt, cpu, add_elementwise_f32_cpu_wrapper);
    table.register(SubElementwise,   f32_dt, cpu, sub_elementwise_f32_cpu_wrapper);
    table.register(MulElementwise,   f32_dt, cpu, mul_elementwise_f32_cpu_wrapper);
    table.register(DivElementwise,   f32_dt, cpu, div_elementwise_f32_cpu_wrapper);

    table.register(ReluElementwise,    f32_dt, cpu, relu_elementwise_f32_cpu_wrapper);
    table.register(NegElementwise,     f32_dt, cpu, neg_elementwise_f32_cpu_wrapper);
    table.register(SqrElementwise,     f32_dt, cpu, sqr_elementwise_f32_cpu_wrapper);
    table.register(SqrtElementwise,    f32_dt, cpu, sqrt_elementwise_f32_cpu_wrapper);
    table.register(RecipElementwise,   f32_dt, cpu, recip_elementwise_f32_cpu_wrapper);
    table.register(AbsElementwise,     f32_dt, cpu, abs_elementwise_f32_cpu_wrapper);
    table.register(TanhElementwise,    f32_dt, cpu, tanh_elementwise_f32_cpu_wrapper);
    table.register(ExpElementwise,     f32_dt, cpu, exp_elementwise_f32_cpu_wrapper);
    table.register(LogElementwise,     f32_dt, cpu, log_elementwise_f32_cpu_wrapper);
    table.register(SinElementwise,     f32_dt, cpu, sin_elementwise_f32_cpu_wrapper);
    table.register(CosElementwise,     f32_dt, cpu, cos_elementwise_f32_cpu_wrapper);
    table.register(SigmoidElementwise, f32_dt, cpu, sigmoid_elementwise_f32_cpu_wrapper);
    table.register(SiluElementwise,    f32_dt, cpu, silu_elementwise_f32_cpu_wrapper);
    table.register(GeluElementwise,    f32_dt, cpu, gelu_elementwise_f32_cpu_wrapper);
    table.register(StepElementwise,    f32_dt, cpu, step_elementwise_f32_cpu_wrapper);

    // f64/bf16/f16 dtype shorthands — used across all the
    // multi-dtype registration blocks below.
    let f64_dt = DType::F64;
    let bf16_dt = DType::BF16;
    let f16_dt  = DType::F16;
    table.register(AddElementwise,     f64_dt, cpu, add_elementwise_f64_cpu_wrapper);
    table.register(SubElementwise,     f64_dt, cpu, sub_elementwise_f64_cpu_wrapper);
    table.register(MulElementwise,     f64_dt, cpu, mul_elementwise_f64_cpu_wrapper);
    table.register(DivElementwise,     f64_dt, cpu, div_elementwise_f64_cpu_wrapper);
    table.register(ReluElementwise,    f64_dt, cpu, relu_elementwise_f64_cpu_wrapper);
    table.register(NegElementwise,     f64_dt, cpu, neg_elementwise_f64_cpu_wrapper);
    table.register(SqrElementwise,     f64_dt, cpu, sqr_elementwise_f64_cpu_wrapper);
    table.register(SqrtElementwise,    f64_dt, cpu, sqrt_elementwise_f64_cpu_wrapper);
    table.register(RecipElementwise,   f64_dt, cpu, recip_elementwise_f64_cpu_wrapper);
    table.register(AbsElementwise,     f64_dt, cpu, abs_elementwise_f64_cpu_wrapper);
    table.register(TanhElementwise,    f64_dt, cpu, tanh_elementwise_f64_cpu_wrapper);
    table.register(ExpElementwise,     f64_dt, cpu, exp_elementwise_f64_cpu_wrapper);
    table.register(LogElementwise,     f64_dt, cpu, log_elementwise_f64_cpu_wrapper);
    table.register(SinElementwise,     f64_dt, cpu, sin_elementwise_f64_cpu_wrapper);
    table.register(CosElementwise,     f64_dt, cpu, cos_elementwise_f64_cpu_wrapper);
    table.register(SigmoidElementwise, f64_dt, cpu, sigmoid_elementwise_f64_cpu_wrapper);
    table.register(SiluElementwise,    f64_dt, cpu, silu_elementwise_f64_cpu_wrapper);
    table.register(GeluElementwise,    f64_dt, cpu, gelu_elementwise_f64_cpu_wrapper);
    table.register(StepElementwise,    f64_dt, cpu, step_elementwise_f64_cpu_wrapper);

    table.register(SumReduce,          f32_dt, cpu, sum_reduce_f32_cpu_wrapper);
    table.register(MaxReduce,          f32_dt, cpu, max_reduce_f32_cpu_wrapper);
    table.register(MinReduce,          f32_dt, cpu, min_reduce_f32_cpu_wrapper);
    table.register(MeanReduce,         f32_dt, cpu, mean_reduce_f32_cpu_wrapper);
    table.register(SumReduce,          f64_dt, cpu, sum_reduce_f64_cpu_wrapper);
    table.register(MaxReduce,          f64_dt, cpu, max_reduce_f64_cpu_wrapper);
    table.register(MinReduce,          f64_dt, cpu, min_reduce_f64_cpu_wrapper);
    table.register(MeanReduce,         f64_dt, cpu, mean_reduce_f64_cpu_wrapper);

    table.register(MatMul,             f32_dt, cpu, matmul_f32_cpu_wrapper);
    table.register(MatMul,             f64_dt, cpu, matmul_f64_cpu_wrapper);
    table.register(MatMul,             bf16_dt, cpu, matmul_bf16_cpu_wrapper);
    table.register(MatMul,             f16_dt, cpu, matmul_f16_cpu_wrapper);

    // bf16 + f16 reductions — accumulate in f32 for stability.
    table.register(SumReduce,          bf16_dt, cpu, sum_reduce_bf16_cpu_wrapper);
    table.register(MaxReduce,          bf16_dt, cpu, max_reduce_bf16_cpu_wrapper);
    table.register(MinReduce,          bf16_dt, cpu, min_reduce_bf16_cpu_wrapper);
    table.register(MeanReduce,         bf16_dt, cpu, mean_reduce_bf16_cpu_wrapper);
    table.register(SumReduce,          f16_dt, cpu, sum_reduce_f16_cpu_wrapper);
    table.register(MaxReduce,          f16_dt, cpu, max_reduce_f16_cpu_wrapper);
    table.register(MinReduce,          f16_dt, cpu, min_reduce_f16_cpu_wrapper);
    table.register(MeanReduce,         f16_dt, cpu, mean_reduce_f16_cpu_wrapper);

    // Cast keys on the *target* dtype; each wrapper handles its
    // supported source dtypes internally. Add new (target, source)
    // pairs by extending the wrapper's match arms.
    table.register(Cast, DType::F32,  cpu, cast_to_f32_cpu_wrapper);
    table.register(Cast, DType::F64,  cpu, cast_to_f64_cpu_wrapper);
    table.register(Cast, DType::BF16, cpu, cast_to_bf16_cpu_wrapper);
    table.register(Cast, DType::F16,  cpu, cast_to_f16_cpu_wrapper);

    table.register(Conv2D, f32_dt,  cpu, conv2d_f32_cpu_wrapper);
    table.register(Conv2D, f64_dt,  cpu, conv2d_f64_cpu_wrapper);
    table.register(Conv2D, bf16_dt, cpu, conv2d_bf16_cpu_wrapper);
    table.register(Conv2D, f16_dt,  cpu, conv2d_f16_cpu_wrapper);

    table.register(ConvTranspose2D, f32_dt,  cpu, conv_transpose2d_f32_cpu_wrapper);
    table.register(ConvTranspose2D, f64_dt,  cpu, conv_transpose2d_f64_cpu_wrapper);
    table.register(ConvTranspose2D, bf16_dt, cpu, conv_transpose2d_bf16_cpu_wrapper);
    table.register(ConvTranspose2D, f16_dt,  cpu, conv_transpose2d_f16_cpu_wrapper);

    table.register(ReduceSumTo, f32_dt,  cpu, reduce_sum_to_f32_cpu_wrapper);
    table.register(ReduceSumTo, f64_dt,  cpu, reduce_sum_to_f64_cpu_wrapper);
    table.register(ReduceSumTo, bf16_dt, cpu, reduce_sum_to_bf16_cpu_wrapper);
    table.register(ReduceSumTo, f16_dt,  cpu, reduce_sum_to_f16_cpu_wrapper);

    table.register(FusedLinear, f32_dt,  cpu, fused_linear_f32_cpu_wrapper);
    table.register(FusedLinear, f64_dt,  cpu, fused_linear_f64_cpu_wrapper);
    table.register(FusedLinear, bf16_dt, cpu, fused_linear_bf16_cpu_wrapper);
    table.register(FusedLinear, f16_dt,  cpu, fused_linear_f16_cpu_wrapper);

    table.register(FlashAttn, f32_dt,  cpu, flash_attn_f32_cpu_wrapper);
    table.register(FlashAttn, f64_dt,  cpu, flash_attn_f64_cpu_wrapper);
    table.register(FlashAttn, bf16_dt, cpu, flash_attn_bf16_cpu_wrapper);
    table.register(FlashAttn, f16_dt,  cpu, flash_attn_f16_cpu_wrapper);

    table.register(PagedAttn, f32_dt,  cpu, paged_attn_f32_cpu_wrapper);
    table.register(PagedAttn, f64_dt,  cpu, paged_attn_f64_cpu_wrapper);
    table.register(PagedAttn, bf16_dt, cpu, paged_attn_bf16_cpu_wrapper);
    table.register(PagedAttn, f16_dt,  cpu, paged_attn_f16_cpu_wrapper);

    table.register(Affine,             f32_dt, cpu, affine_f32_cpu_wrapper);
    table.register(ClampElementwise,   f32_dt, cpu, clamp_elementwise_f32_cpu_wrapper);
    table.register(PowIElementwise,    f32_dt, cpu, powi_elementwise_f32_cpu_wrapper);
    table.register(MaximumElementwise, f32_dt, cpu, maximum_elementwise_f32_cpu_wrapper);
    table.register(MinimumElementwise, f32_dt, cpu, minimum_elementwise_f32_cpu_wrapper);
    table.register(MaximumElementwise, f64_dt, cpu, maximum_elementwise_f64_cpu_wrapper);
    table.register(MinimumElementwise, f64_dt, cpu, minimum_elementwise_f64_cpu_wrapper);

    // bf16 + f16 elementwise — via-f32 round-trip kernels.
    table.register(AddElementwise,     bf16_dt, cpu, add_elementwise_bf16_cpu_wrapper);
    table.register(SubElementwise,     bf16_dt, cpu, sub_elementwise_bf16_cpu_wrapper);
    table.register(MulElementwise,     bf16_dt, cpu, mul_elementwise_bf16_cpu_wrapper);
    table.register(DivElementwise,     bf16_dt, cpu, div_elementwise_bf16_cpu_wrapper);
    table.register(MaximumElementwise, bf16_dt, cpu, maximum_elementwise_bf16_cpu_wrapper);
    table.register(MinimumElementwise, bf16_dt, cpu, minimum_elementwise_bf16_cpu_wrapper);
    table.register(ReluElementwise,    bf16_dt, cpu, relu_elementwise_bf16_cpu_wrapper);
    table.register(NegElementwise,     bf16_dt, cpu, neg_elementwise_bf16_cpu_wrapper);
    table.register(SqrElementwise,     bf16_dt, cpu, sqr_elementwise_bf16_cpu_wrapper);
    table.register(SqrtElementwise,    bf16_dt, cpu, sqrt_elementwise_bf16_cpu_wrapper);
    table.register(RecipElementwise,   bf16_dt, cpu, recip_elementwise_bf16_cpu_wrapper);
    table.register(AbsElementwise,     bf16_dt, cpu, abs_elementwise_bf16_cpu_wrapper);
    table.register(TanhElementwise,    bf16_dt, cpu, tanh_elementwise_bf16_cpu_wrapper);
    table.register(ExpElementwise,     bf16_dt, cpu, exp_elementwise_bf16_cpu_wrapper);
    table.register(LogElementwise,     bf16_dt, cpu, log_elementwise_bf16_cpu_wrapper);
    table.register(SinElementwise,     bf16_dt, cpu, sin_elementwise_bf16_cpu_wrapper);
    table.register(CosElementwise,     bf16_dt, cpu, cos_elementwise_bf16_cpu_wrapper);
    table.register(SigmoidElementwise, bf16_dt, cpu, sigmoid_elementwise_bf16_cpu_wrapper);
    table.register(SiluElementwise,    bf16_dt, cpu, silu_elementwise_bf16_cpu_wrapper);
    table.register(GeluElementwise,    bf16_dt, cpu, gelu_elementwise_bf16_cpu_wrapper);
    table.register(StepElementwise,    bf16_dt, cpu, step_elementwise_bf16_cpu_wrapper);

    table.register(AddElementwise,     f16_dt, cpu, add_elementwise_f16_cpu_wrapper);
    table.register(SubElementwise,     f16_dt, cpu, sub_elementwise_f16_cpu_wrapper);
    table.register(MulElementwise,     f16_dt, cpu, mul_elementwise_f16_cpu_wrapper);
    table.register(DivElementwise,     f16_dt, cpu, div_elementwise_f16_cpu_wrapper);
    table.register(MaximumElementwise, f16_dt, cpu, maximum_elementwise_f16_cpu_wrapper);
    table.register(MinimumElementwise, f16_dt, cpu, minimum_elementwise_f16_cpu_wrapper);
    table.register(ReluElementwise,    f16_dt, cpu, relu_elementwise_f16_cpu_wrapper);
    table.register(NegElementwise,     f16_dt, cpu, neg_elementwise_f16_cpu_wrapper);
    table.register(SqrElementwise,     f16_dt, cpu, sqr_elementwise_f16_cpu_wrapper);
    table.register(SqrtElementwise,    f16_dt, cpu, sqrt_elementwise_f16_cpu_wrapper);
    table.register(RecipElementwise,   f16_dt, cpu, recip_elementwise_f16_cpu_wrapper);
    table.register(AbsElementwise,     f16_dt, cpu, abs_elementwise_f16_cpu_wrapper);
    table.register(TanhElementwise,    f16_dt, cpu, tanh_elementwise_f16_cpu_wrapper);
    table.register(ExpElementwise,     f16_dt, cpu, exp_elementwise_f16_cpu_wrapper);
    table.register(LogElementwise,     f16_dt, cpu, log_elementwise_f16_cpu_wrapper);
    table.register(SinElementwise,     f16_dt, cpu, sin_elementwise_f16_cpu_wrapper);
    table.register(CosElementwise,     f16_dt, cpu, cos_elementwise_f16_cpu_wrapper);
    table.register(SigmoidElementwise, f16_dt, cpu, sigmoid_elementwise_f16_cpu_wrapper);
    table.register(SiluElementwise,    f16_dt, cpu, silu_elementwise_f16_cpu_wrapper);
    table.register(GeluElementwise,    f16_dt, cpu, gelu_elementwise_f16_cpu_wrapper);
    table.register(StepElementwise,    f16_dt, cpu, step_elementwise_f16_cpu_wrapper);

    // Concat is dtype-agnostic at the byte level — register the
    // same wrapper for every dtype the executor might allocate.
    for dt in [DType::F32, DType::F64, DType::BF16, DType::F16, DType::U32, DType::U8, DType::I16, DType::I32, DType::I64] {
        table.register(Concat, dt, cpu, concat_cpu_wrapper);
    }
    table.register(SoftmaxLastDim,     f32_dt, cpu, softmax_last_dim_f32_cpu_wrapper);
    table.register(SoftmaxLastDim,     bf16_dt, cpu, softmax_last_dim_bf16_cpu_wrapper);
    table.register(SoftmaxLastDim,     f16_dt,  cpu, softmax_last_dim_f16_cpu_wrapper);
    table.register(RmsNormLastDim,     f32_dt, cpu, rms_norm_last_dim_f32_cpu_wrapper);
    table.register(RmsNormLastDim,     bf16_dt, cpu, rms_norm_last_dim_bf16_cpu_wrapper);
    table.register(RmsNormLastDim,     f16_dt,  cpu, rms_norm_last_dim_f16_cpu_wrapper);
    table.register(LayerNormLastDim,   f32_dt, cpu, layer_norm_last_dim_f32_cpu_wrapper);
    table.register(LayerNormLastDim,   bf16_dt, cpu, layer_norm_last_dim_bf16_cpu_wrapper);
    table.register(LayerNormLastDim,   f16_dt,  cpu, layer_norm_last_dim_f16_cpu_wrapper);
    // IndexSelect and Gather are dtype-agnostic at the byte level —
    // register the same wrappers across every supported dtype.
    for dt in [DType::F32, DType::F64, DType::BF16, DType::F16, DType::U32, DType::U8, DType::I16, DType::I32, DType::I64] {
        table.register(IndexSelect, dt, cpu, index_select_cpu_wrapper);
        table.register(Gather,      dt, cpu, gather_cpu_wrapper);
    }
    table.register(Rope,               f32_dt, cpu, rope_f32_cpu_wrapper);
    table.register(Rope,               bf16_dt, cpu, rope_bf16_cpu_wrapper);
    table.register(Rope,               f16_dt,  cpu, rope_f16_cpu_wrapper);
    table.register(Rope,               f64_dt,  cpu, rope_f64_cpu_wrapper);
    table.register(SoftmaxLastDim,     f64_dt, cpu, softmax_last_dim_f64_cpu_wrapper);
    table.register(RmsNormLastDim,     f64_dt, cpu, rms_norm_last_dim_f64_cpu_wrapper);
    table.register(LayerNormLastDim,   f64_dt, cpu, layer_norm_last_dim_f64_cpu_wrapper);
    table.register(QMatMul,            f32_dt, cpu, qmatmul_f32_cpu_wrapper);
    table.register(IndexAdd,           f32_dt, cpu, index_add_f32_cpu_wrapper);
    table.register(ScatterAdd,         f32_dt, cpu, scatter_add_f32_cpu_wrapper);

    // ArgMax/ArgMin output U32 indices regardless of input dtype.
    // Binding-table key is keyed on the OUTPUT dtype; the wrapper
    // dispatches to the right per-input-dtype kernel internally.
    table.register(ArgMaxDim,          DType::U32, cpu, argmax_dim_u32_cpu_dispatch);
    table.register(ArgMinDim,          DType::U32, cpu, argmin_dim_u32_cpu_dispatch);

    // IndexAdd, ScatterAdd, Affine, Clamp, PowI for f64/bf16/f16.
    table.register(IndexAdd,           f64_dt,  cpu, index_add_f64_cpu_wrapper);
    table.register(IndexAdd,           bf16_dt, cpu, index_add_bf16_cpu_wrapper);
    table.register(IndexAdd,           f16_dt,  cpu, index_add_f16_cpu_wrapper);
    table.register(ScatterAdd,         f64_dt,  cpu, scatter_add_f64_cpu_wrapper);
    table.register(ScatterAdd,         bf16_dt, cpu, scatter_add_bf16_cpu_wrapper);
    table.register(ScatterAdd,         f16_dt,  cpu, scatter_add_f16_cpu_wrapper);
    table.register(Affine,             f64_dt,  cpu, affine_f64_cpu_wrapper);
    table.register(Affine,             bf16_dt, cpu, affine_bf16_cpu_wrapper);
    table.register(Affine,             f16_dt,  cpu, affine_f16_cpu_wrapper);
    table.register(ClampElementwise,   f64_dt,  cpu, clamp_f64_cpu_wrapper);
    table.register(ClampElementwise,   bf16_dt, cpu, clamp_bf16_cpu_wrapper);
    table.register(ClampElementwise,   f16_dt,  cpu, clamp_f16_cpu_wrapper);
    table.register(PowIElementwise,    f64_dt,  cpu, powi_f64_cpu_wrapper);
    table.register(PowIElementwise,    bf16_dt, cpu, powi_bf16_cpu_wrapper);
    table.register(PowIElementwise,    f16_dt,  cpu, powi_f16_cpu_wrapper);
}

// =============================================================================
// Phase 7.5 — CUDA dispatch wrappers + registration
// =============================================================================
//
// First CUDA op through the unified binding table. Mirrors the CPU
// extractor + wrapper pattern but operates on `CudaStorageBytes`.
// Only `(AddElementwise, F32, Cuda)` is registered today; this is the
// "smallest possible first commit" that proves the CUDA migration
// pattern. Subsequent commits fan out to the rest of the op surface
// using the same wrapper-macro shape.

/// Helper: extract `&CudaStorageBytes` from `&Storage`. Returns
/// Err if the variant isn't `BackendStorage::Cuda`.
#[cfg(feature = "cuda")]
fn cuda_input(s: &Storage) -> Result<&fuel_cuda_backend::CudaStorageBytes> {
    match &s.inner {
        BackendStorage::Cuda(c) => Ok(c),
        #[allow(unreachable_patterns)]
        _ => Err(Error::Msg(
            "cuda kernel wrapper called with non-CUDA input".to_string(),
        )
        .bt()),
    }
}

/// Helper: extract `&mut CudaStorageBytes` from `&mut Storage`.
#[cfg(feature = "cuda")]
fn cuda_output(s: &mut Storage) -> Result<&mut fuel_cuda_backend::CudaStorageBytes> {
    match &mut s.inner {
        BackendStorage::Cuda(c) => Ok(c),
        #[allow(unreachable_patterns)]
        _ => Err(Error::Msg(
            "cuda kernel wrapper called with non-CUDA output".to_string(),
        )
        .bt()),
    }
}

/// Dispatch wrapper for `(AddElementwise, F32, Cuda)`. Two F32
/// CUDA inputs of equal byte length, one F32 CUDA output. The
/// kernel call lives in `fuel-cuda-backend::byte_kernels`; this
/// wrapper does the variant extraction + replaces the
/// pre-allocated output's storage with the freshly-computed one.
///
/// Replacement-of-output (vs. write-into-output) is the pragmatic
/// shape for this first commit: the legacy CUDA kernel-launch
/// infrastructure allocates fresh `DeviceBuffer<u8>` for output,
/// and getting unique `&mut Arc<DeviceBuffer<u8>>` from the
/// pre-allocated `CudaStorageBytes` would require `Arc::get_mut`
/// (only succeeds if no other holder). Replacement avoids the
/// Arc-uniqueness dance at the cost of one drop on the executor's
/// pre-allocated placeholder. Future-revisit if profiling shows
/// the alloc churn matters.
#[cfg(feature = "cuda")]
fn add_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "add_elementwise_f32_cuda_wrapper: expected 2 inputs, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "add_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::add_elementwise_f32(lhs_cuda, rhs_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(SubElementwise, F32, Cuda)`. Same shape
/// as `add_elementwise_f32_cuda_wrapper`; only the underlying
/// byte-kernel call differs.
#[cfg(feature = "cuda")]
fn sub_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "sub_elementwise_f32_cuda_wrapper: expected 2 inputs, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "sub_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::sub_elementwise_f32(lhs_cuda, rhs_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(MulElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn mul_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "mul_elementwise_f32_cuda_wrapper: expected 2 inputs, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "mul_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::mul_elementwise_f32(lhs_cuda, rhs_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(DivElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn div_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "div_elementwise_f32_cuda_wrapper: expected 2 inputs, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "div_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::div_elementwise_f32(lhs_cuda, rhs_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(MaximumElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn maximum_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "maximum_elementwise_f32_cuda_wrapper: expected 2 inputs, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "maximum_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::maximum_elementwise_f32(lhs_cuda, rhs_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(MinimumElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn minimum_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "minimum_elementwise_f32_cuda_wrapper: expected 2 inputs, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "minimum_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::minimum_elementwise_f32(lhs_cuda, rhs_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Phase 7.5 CUDA registration. Wires CUDA byte-kernel wrappers
/// into the unified binding table. Same shape as
/// `register_cpu_kernels` but on the Cuda backend.
#[cfg(feature = "cuda")]
pub fn register_cuda_kernels(table: &mut KernelBindingTable) {
    use OpKind::*;
    let cuda = BackendId::Cuda;
    let f32_dt = DType::F32;
    table.register(AddElementwise,     f32_dt, cuda, add_elementwise_f32_cuda_wrapper);
    table.register(SubElementwise,     f32_dt, cuda, sub_elementwise_f32_cuda_wrapper);
    table.register(MulElementwise,     f32_dt, cuda, mul_elementwise_f32_cuda_wrapper);
    table.register(DivElementwise,     f32_dt, cuda, div_elementwise_f32_cuda_wrapper);
    table.register(MaximumElementwise, f32_dt, cuda, maximum_elementwise_f32_cuda_wrapper);
    table.register(MinimumElementwise, f32_dt, cuda, minimum_elementwise_f32_cuda_wrapper);
}

// =============================================================================
// Phase 7.5 B5+ — process-wide singleton (CapabilityRegistry + KernelBindingTable)
// =============================================================================

/// Process-wide [`CapabilityRegistry`]. Initialized on first access
/// via [`global_registry`]; the CPU backend is auto-registered
/// always (universal fallback). Other backends register themselves
/// during their initialization in fuel-graph-router or app startup.
///
/// Tests that need a private registry should construct one
/// directly with `CapabilityRegistry::new()` rather than touch the
/// global.
static GLOBAL_REGISTRY: OnceLock<RwLock<CapabilityRegistry>> = OnceLock::new();

/// Process-wide [`KernelBindingTable`]. Initialized on first access
/// with the CPU dispatch wrappers from [`register_cpu_kernels`].
/// Other backends extend it when they register.
static GLOBAL_BINDINGS: OnceLock<RwLock<KernelBindingTable>> = OnceLock::new();

fn default_cpu_caps() -> BackendCapabilities {
    use std::collections::HashSet;
    use OpKind::*;
    let mut op_dtype_support = HashSet::new();
    // The CPU coverage set must stay in lockstep with
    // `register_cpu_kernels`: each (op, dtype) pair registered there
    // is advertised here so capability-driven dispatch picks it.
    let f32_dt = DType::F32;
    for op in [
        AddElementwise,
        SubElementwise,
        MulElementwise,
        DivElementwise,
        ReluElementwise,
        NegElementwise,
        SqrElementwise,
        SqrtElementwise,
        RecipElementwise,
        AbsElementwise,
        TanhElementwise,
        ExpElementwise,
        LogElementwise,
        SinElementwise,
        CosElementwise,
        SigmoidElementwise,
        SiluElementwise,
        GeluElementwise,
        StepElementwise,
        SumReduce,
        MaxReduce,
        MinReduce,
        MeanReduce,
        MatMul,
        Conv2D,
        ConvTranspose2D,
        ReduceSumTo,
        FusedLinear,
        FlashAttn,
        PagedAttn,
        Affine,
        ClampElementwise,
        PowIElementwise,
        MaximumElementwise,
        MinimumElementwise,
        Concat,
        SoftmaxLastDim,
        RmsNormLastDim,
        LayerNormLastDim,
        IndexSelect,
        Gather,
        Rope,
        IndexAdd,
        ScatterAdd,
        QMatMul,
    ] {
        op_dtype_support.insert((op, f32_dt));
    }
    // f64 elementwise families. Reductions, matmul, conv, composed
    // ops, indexing, and view ops will join the f64 set as their
    // kernels migrate.
    for op in [
        AddElementwise,
        SubElementwise,
        MulElementwise,
        DivElementwise,
        ReluElementwise,
        NegElementwise,
        SqrElementwise,
        SqrtElementwise,
        RecipElementwise,
        AbsElementwise,
        TanhElementwise,
        ExpElementwise,
        LogElementwise,
        SinElementwise,
        CosElementwise,
        SigmoidElementwise,
        SiluElementwise,
        GeluElementwise,
        StepElementwise,
        MaximumElementwise,
        MinimumElementwise,
        SumReduce,
        MaxReduce,
        MinReduce,
        MeanReduce,
        MatMul,
        Conv2D,
        ConvTranspose2D,
        ReduceSumTo,
        FusedLinear,
        FlashAttn,
        PagedAttn,
    ] {
        op_dtype_support.insert((op, DType::F64));
    }
    // bf16 and f16 elementwise families. Reductions/matmul/conv/
    // composed/indexing/Rope/scalar follow as kernels migrate.
    let half_elementwise_ops = [
        AddElementwise,
        SubElementwise,
        MulElementwise,
        DivElementwise,
        MaximumElementwise,
        MinimumElementwise,
        ReluElementwise,
        NegElementwise,
        SqrElementwise,
        SqrtElementwise,
        RecipElementwise,
        AbsElementwise,
        TanhElementwise,
        ExpElementwise,
        LogElementwise,
        SinElementwise,
        CosElementwise,
        SigmoidElementwise,
        SiluElementwise,
        GeluElementwise,
        StepElementwise,
    ];
    for op in half_elementwise_ops {
        op_dtype_support.insert((op, DType::BF16));
        op_dtype_support.insert((op, DType::F16));
    }
    // bf16/f16 reductions + matmul (kernels accumulate in f32).
    for op in [SumReduce, MaxReduce, MinReduce, MeanReduce, MatMul] {
        op_dtype_support.insert((op, DType::BF16));
        op_dtype_support.insert((op, DType::F16));
    }
    // bf16/f16 composed/fused ops (Softmax, RmsNorm, LayerNorm,
    // Rope, Conv2D, ConvTranspose2D, ReduceSumTo, FusedLinear,
    // FlashAttn, PagedAttn) — all use the f32-accumulator pattern.
    for op in [SoftmaxLastDim, RmsNormLastDim, LayerNormLastDim, Rope, Conv2D, ConvTranspose2D, ReduceSumTo, FusedLinear, FlashAttn, PagedAttn] {
        op_dtype_support.insert((op, DType::BF16));
        op_dtype_support.insert((op, DType::F16));
    }
    // Concat / IndexSelect / Gather are dtype-agnostic at the
    // byte level — advertised across the universal float/int set.
    for dt in [DType::F64, DType::BF16, DType::F16, DType::U32, DType::U8, DType::I16, DType::I32, DType::I64] {
        op_dtype_support.insert((Concat, dt));
        op_dtype_support.insert((IndexSelect, dt));
        op_dtype_support.insert((Gather, dt));
    }
    // IndexAdd / ScatterAdd / Affine / Clamp / PowI for f64/bf16/f16.
    for dt in [DType::F64, DType::BF16, DType::F16] {
        op_dtype_support.insert((IndexAdd, dt));
        op_dtype_support.insert((ScatterAdd, dt));
        op_dtype_support.insert((Affine, dt));
        op_dtype_support.insert((ClampElementwise, dt));
        op_dtype_support.insert((PowIElementwise, dt));
    }
    // F64 composed/fused ops (Softmax, RmsNorm, LayerNorm, Rope).
    for op in [SoftmaxLastDim, RmsNormLastDim, LayerNormLastDim, Rope] {
        op_dtype_support.insert((op, DType::F64));
    }
    // Argmax/argmin always produce U32 indices.
    for op in [ArgMaxDim, ArgMinDim] {
        op_dtype_support.insert((op, DType::U32));
    }
    // Cast advertises every target dtype that has a wrapper
    // registered. The capability matrix says "this backend can
    // produce an F64 output via Cast"; the source-side coverage
    // is managed inside each wrapper's match arms.
    for target in [DType::F32, DType::F64, DType::BF16, DType::F16] {
        op_dtype_support.insert((Cast, target));
    }
    BackendCapabilities {
        backend_id: BackendId::Cpu,
        device_location: DeviceLocation::Cpu,
        op_dtype_support,
        required_alignment: 64,
        access_granularity_bits: 8,
        transfer_paths: vec![(DeviceLocation::Cpu, TransferPath::SameDevice)],
    }
}

/// Read-lock the process-wide capability registry. CPU is auto-
/// registered on first access; subsequent backends register
/// themselves via [`register_backend_capabilities`].
pub fn global_registry() -> std::sync::RwLockReadGuard<'static, CapabilityRegistry> {
    GLOBAL_REGISTRY
        .get_or_init(|| {
            let mut r = CapabilityRegistry::new();
            r.register(default_cpu_caps());
            RwLock::new(r)
        })
        .read()
        .unwrap()
}

/// Add a backend's capabilities to the process-wide registry.
/// Typically called by each backend's init / probe path during
/// app startup. Idempotent for the same `(backend_id,
/// device_location)` is the *caller's* responsibility — the
/// registry happily appends duplicates and the lookup picks the
/// first-registered match.
pub fn register_backend_capabilities(caps: BackendCapabilities) {
    let lock = GLOBAL_REGISTRY.get_or_init(|| {
        let mut r = CapabilityRegistry::new();
        r.register(default_cpu_caps());
        RwLock::new(r)
    });
    lock.write().unwrap().register(caps);
}

/// Read-lock the process-wide kernel-binding table. CPU dispatch
/// wrappers are auto-registered on first access.
pub fn global_bindings() -> std::sync::RwLockReadGuard<'static, KernelBindingTable> {
    GLOBAL_BINDINGS
        .get_or_init(|| {
            let mut t = KernelBindingTable::new();
            register_cpu_kernels(&mut t);
            RwLock::new(t)
        })
        .read()
        .unwrap()
}

/// Add a backend's dispatch wrappers to the process-wide binding
/// table. Each backend exposes a `register_*_kernels(table)`
/// function (see [`register_cpu_kernels`]); per-backend init paths
/// call this to plug their wrappers into the global table.
pub fn extend_global_bindings(register: impl FnOnce(&mut KernelBindingTable)) {
    let lock = GLOBAL_BINDINGS.get_or_init(|| {
        let mut t = KernelBindingTable::new();
        register_cpu_kernels(&mut t);
        RwLock::new(t)
    });
    register(&mut lock.write().unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_caps() -> BackendCapabilities {
        BackendCapabilities {
            backend_id: BackendId::Cpu,
            device_location: DeviceLocation::Cpu,
            op_dtype_support: [
                (OpKind::MatMul, DType::F32),
                (OpKind::MatMul, DType::F64),
                (OpKind::AddElementwise, DType::F32),
                (OpKind::AddElementwise, DType::F64),
            ]
            .into_iter()
            .collect(),
            required_alignment: 64,
            access_granularity_bits: 8,
            transfer_paths: vec![(DeviceLocation::Cpu, TransferPath::SameDevice)],
        }
    }

    fn cuda_caps() -> BackendCapabilities {
        BackendCapabilities {
            backend_id: BackendId::Cuda,
            device_location: DeviceLocation::Cuda { gpu_id: 0 },
            op_dtype_support: [
                (OpKind::MatMul, DType::F32),
                (OpKind::MatMul, DType::F16),
                (OpKind::AddElementwise, DType::F32),
            ]
            .into_iter()
            .collect(),
            required_alignment: 256,
            access_granularity_bits: 8,
            transfer_paths: vec![
                (DeviceLocation::Cpu, TransferPath::DeviceCopy),
                (DeviceLocation::Cuda { gpu_id: 0 }, TransferPath::SameDevice),
            ],
        }
    }

    /// Smoke: empty registry has nothing.
    #[test]
    fn empty_registry() {
        let r = CapabilityRegistry::new();
        assert!(r.backends().is_empty());
        let result = r.find_backend_for(OpKind::MatMul, DType::F32);
        assert!(result.is_err());
    }

    /// Find_backends returns matching backends in registration order.
    #[test]
    fn find_backends_order_preserved() {
        let mut r = CapabilityRegistry::new();
        r.register(cuda_caps());  // registered first → wins ties
        r.register(cpu_caps());

        let backends = r.find_backends(OpKind::MatMul, DType::F32);
        assert_eq!(backends.len(), 2);
        assert_eq!(backends[0].backend_id, BackendId::Cuda);
        assert_eq!(backends[1].backend_id, BackendId::Cpu);
    }

    /// Find_backend_for picks first match.
    #[test]
    fn find_backend_first_wins() {
        let mut r = CapabilityRegistry::new();
        r.register(cuda_caps());
        r.register(cpu_caps());

        let chosen = r.find_backend_for(OpKind::MatMul, DType::F32).unwrap();
        assert_eq!(chosen.backend_id, BackendId::Cuda);
    }

    /// CPU-only registry falls back when GPU dtypes aren't supported.
    #[test]
    fn cpu_handles_what_gpu_doesnt() {
        let mut r = CapabilityRegistry::new();
        r.register(cuda_caps());
        r.register(cpu_caps());

        // CUDA doesn't support F64; CPU does.
        let chosen = r.find_backend_for(OpKind::MatMul, DType::F64).unwrap();
        assert_eq!(chosen.backend_id, BackendId::Cpu);
    }

    /// Unsupported (op, dtype) returns NoBackendForOp with diagnostic data.
    #[test]
    fn unsupported_combo_errors_with_diagnostic() {
        let mut r = CapabilityRegistry::new();
        r.register(cpu_caps());

        let err = match r.find_backend_for(OpKind::MatMul, DType::BF16) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("MatMul") || msg.contains("matmul"),
            "error names op: {msg}"
        );
        assert!(msg.contains("BF16"), "error names dtype: {msg}");
        assert!(msg.contains("Cpu"), "error names available backends: {msg}");
    }

    /// TransferMatrix preserves SameDevice for src == dst always.
    #[test]
    fn transfer_matrix_same_device() {
        let r = CapabilityRegistry::new();
        let m = r.build_transfer_matrix();
        assert_eq!(
            m.path(DeviceLocation::Cpu, DeviceLocation::Cpu),
            Some(TransferPath::SameDevice)
        );
    }

    /// TransferMatrix entries from registered backends.
    #[test]
    fn transfer_matrix_built_from_caps() {
        let mut r = CapabilityRegistry::new();
        r.register(cuda_caps());
        let m = r.build_transfer_matrix();

        // CUDA → CPU is DeviceCopy (advertised).
        assert_eq!(
            m.path(DeviceLocation::Cuda { gpu_id: 0 }, DeviceLocation::Cpu),
            Some(TransferPath::DeviceCopy)
        );
    }

    /// path_or_staging falls back to HostStaging for unadvertised
    /// pairs.
    #[test]
    fn host_staging_fallback() {
        let r = CapabilityRegistry::new();
        let m = r.build_transfer_matrix();
        // No CPU registered; CPU→Cuda has no entry; staging fallback.
        assert_eq!(
            m.path_or_staging(DeviceLocation::Cpu, DeviceLocation::Cuda { gpu_id: 0 }),
            TransferPath::HostStaging
        );
    }

    /// resolve_target_backend picks the first registered backend
    /// supporting (op, dtype).
    #[test]
    fn resolve_picks_first_match() {
        let mut r = CapabilityRegistry::new();
        r.register(cuda_caps());
        r.register(cpu_caps());
        assert_eq!(
            resolve_target_backend(&r, OpKind::MatMul, DType::F32).unwrap(),
            BackendId::Cuda
        );
    }

    /// resolve_target_backend falls back to CPU for dtypes GPU doesn't support.
    #[test]
    fn resolve_falls_back_to_cpu() {
        let mut r = CapabilityRegistry::new();
        r.register(cuda_caps());
        r.register(cpu_caps());
        assert_eq!(
            resolve_target_backend(&r, OpKind::MatMul, DType::F64).unwrap(),
            BackendId::Cpu
        );
    }

    /// resolve_target_backend errors with NoBackendForOp on miss.
    #[test]
    fn resolve_errors_on_unsupported() {
        let mut r = CapabilityRegistry::new();
        r.register(cpu_caps());
        let result = resolve_target_backend(&r, OpKind::MatMul, DType::BF16);
        assert!(result.is_err());
    }

    /// Residency-aware resolver picks a backend with input-local
    /// residency when available.
    #[test]
    fn resolve_residency_aware_prefers_local() {
        let mut r = CapabilityRegistry::new();
        // CUDA registered first (would normally win ties for f32).
        r.register(cuda_caps());
        r.register(cpu_caps());

        // Inputs live on CPU; residency-aware prefers CPU even
        // though CUDA also supports (MatMul, F32).
        let chosen = resolve_target_backend_residency_aware(
            &r,
            OpKind::MatMul,
            DType::F32,
            &[DeviceLocation::Cpu],
        )
        .unwrap();
        assert_eq!(chosen, BackendId::Cpu);
    }

    /// Residency-aware falls back to first match when no residency
    /// match exists.
    #[test]
    fn resolve_residency_aware_no_local_falls_back() {
        let mut r = CapabilityRegistry::new();
        r.register(cuda_caps());
        r.register(cpu_caps());

        // Inputs on a Vulkan device — no candidate matches; falls
        // back to first (CUDA).
        let chosen = resolve_target_backend_residency_aware(
            &r,
            OpKind::MatMul,
            DType::F32,
            &[DeviceLocation::Vulkan { gpu_id: 0 }],
        )
        .unwrap();
        assert_eq!(chosen, BackendId::Cuda);
    }

    // -------- B5: kernel binding table + CPU dispatch wrappers --------

    /// Smoke: register_cpu_kernels populates the binding table.
    #[test]
    fn register_cpu_kernels_populates_table() {
        let mut table = KernelBindingTable::new();
        assert!(table.is_empty());
        register_cpu_kernels(&mut table);
        assert!(!table.is_empty());
        // The (AddElementwise, F32, Cpu) binding lands.
        let _kernel = table
            .lookup(OpKind::AddElementwise, DType::F32, BackendId::Cpu)
            .expect("registered");
    }

    /// Lookup miss returns NoBackendForOp with diagnostic data.
    #[test]
    fn binding_table_lookup_miss_errors() {
        let mut table = KernelBindingTable::new();
        register_cpu_kernels(&mut table);
        // I64 isn't registered for any elementwise op — must error.
        // (BF16/F16/F32/F64 all have AddElementwise wrappers as of
        // Phase C's multi-dtype expansion.)
        let result = table.lookup(OpKind::AddElementwise, DType::I64, BackendId::Cpu);
        assert!(result.is_err());
    }

    /// End-to-end: register, resolve target backend, look up the
    /// kernel, allocate input/output Storages, call the kernel,
    /// verify output bytes match the elementwise sum. This is the
    /// proof-of-concept for the entire B-phase dispatch path.
    #[test]
    fn b5_end_to_end_add_elementwise_f32() {
        // 1. Build registry + capability advertisement.
        let mut registry = CapabilityRegistry::new();
        registry.register(cpu_caps());

        // 2. Resolve which backend handles (AddElementwise, F32).
        let backend = resolve_target_backend(
            &registry,
            OpKind::AddElementwise,
            DType::F32,
        )
        .expect("resolve");
        assert_eq!(backend, BackendId::Cpu);

        // 3. Build binding table + register CPU wrappers.
        let mut bindings = KernelBindingTable::new();
        register_cpu_kernels(&mut bindings);

        // 4. Look up the kernel for (op, dtype, backend).
        let kernel = bindings
            .lookup(OpKind::AddElementwise, DType::F32, backend)
            .expect("lookup");

        // 5. Allocate input + output Storages.
        let lhs = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let rhs = crate::from_slice_cpu(&[10.0_f32, 20.0, 30.0, 40.0]);
        let out = crate::alloc_cpu_zeroed(DType::F32, 4).expect("alloc");

        let inputs = vec![Arc::new(RwLock::new(lhs)), Arc::new(RwLock::new(rhs))];
        let mut outputs = vec![Arc::new(RwLock::new(out))];

        // 6. Call the dispatch wrapper.
        kernel(&inputs, &mut outputs, &OpParams::None).expect("kernel");

        // 7. Verify output bytes match elementwise sum.
        let out_guard = outputs[0].read().unwrap();
        if let BackendStorage::Cpu(c) = &out_guard.inner {
            let typed: &[f32] = c.as_slice().expect("cast");
            assert_eq!(typed, &[11.0, 22.0, 33.0, 44.0]);
        } else {
            panic!("expected CPU output");
        }
    }

    /// E2E: wrong number of inputs surfaces a typed error.
    #[test]
    fn b5_wrapper_arity_check() {
        let mut bindings = KernelBindingTable::new();
        register_cpu_kernels(&mut bindings);
        let kernel = bindings
            .lookup(OpKind::AddElementwise, DType::F32, BackendId::Cpu)
            .unwrap();

        // Only one input — should error, not panic.
        let lhs = crate::from_slice_cpu(&[1.0_f32, 2.0]);
        let out = crate::alloc_cpu_zeroed(DType::F32, 2).unwrap();
        let inputs = vec![Arc::new(RwLock::new(lhs))];
        let mut outputs = vec![Arc::new(RwLock::new(out))];

        let result = kernel(&inputs, &mut outputs, &OpParams::None);
        assert!(result.is_err());
    }

    /// Global registry auto-registers CPU on first access.
    #[test]
    fn global_registry_auto_registers_cpu() {
        // First access initializes the registry with CPU caps.
        let r = global_registry();
        assert!(!r.backends().is_empty());
        // CPU is registered.
        let cpu = r.backends().iter().find(|c| c.backend_id == BackendId::Cpu);
        assert!(cpu.is_some(), "CPU should be auto-registered");
    }

    /// Global bindings auto-registers the CPU AddElementwise+F32
    /// wrapper on first access.
    #[test]
    fn global_bindings_auto_registers_cpu_wrappers() {
        let b = global_bindings();
        let result = b.lookup(OpKind::AddElementwise, DType::F32, BackendId::Cpu);
        assert!(result.is_ok(), "CPU AddElementwise+F32 should be registered");
    }
}
