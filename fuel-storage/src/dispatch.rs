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
use fuel_core_types::{DType, DeviceLocation, Error, Layout, Result};

use crate::kernel::{KernelBindingTable, KernelRef, OpParams};
#[cfg(feature = "cuda")]
use crate::kernel::KernelCaps;
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
        // Capability-level lookups still operate on a single output
        // dtype (binding-table multi-dtype keys are an execution-time
        // concern). Wrap the single dtype in a 1-vec for the error so
        // both error sites share one variant shape.
        Err(Error::NoBackendForOp {
            op,
            dtypes: vec![dtype],
            available_backends: self.backends.iter().map(|c| c.backend_id).collect(),
            supported_combinations: self
                .backends
                .iter()
                .flat_map(|c| {
                    c.op_dtype_support
                        .iter()
                        .map(|&(o, d)| (c.backend_id, o, vec![d]))
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
            _layouts: &[Layout],
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
            _layouts: &[Layout],
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

// Comparison family — typed input, U8 output. The wrapper signature
// is identical to a regular binary wrapper (3 byte buffers); only the
// kernel internally casts inputs to `&[T]` and output to `&mut [u8]`.
// Binding-table key is `[T, T, U8]` so the executor allocates a U8-
// sized output buffer (1 byte per element) instead of T-sized.
cpu_binary_wrapper!(eq_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::eq_f32_u8, "eq_elementwise");
cpu_binary_wrapper!(eq_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::eq_f64_u8, "eq_elementwise");
cpu_binary_wrapper!(eq_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::eq_bf16_u8, "eq_elementwise");
cpu_binary_wrapper!(eq_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::eq_f16_u8, "eq_elementwise");

cpu_binary_wrapper!(ne_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::ne_f32_u8, "ne_elementwise");
cpu_binary_wrapper!(ne_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::ne_f64_u8, "ne_elementwise");
cpu_binary_wrapper!(ne_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::ne_bf16_u8, "ne_elementwise");
cpu_binary_wrapper!(ne_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::ne_f16_u8, "ne_elementwise");

cpu_binary_wrapper!(lt_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::lt_f32_u8, "lt_elementwise");
cpu_binary_wrapper!(lt_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::lt_f64_u8, "lt_elementwise");
cpu_binary_wrapper!(lt_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::lt_bf16_u8, "lt_elementwise");
cpu_binary_wrapper!(lt_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::lt_f16_u8, "lt_elementwise");

cpu_binary_wrapper!(le_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::le_f32_u8, "le_elementwise");
cpu_binary_wrapper!(le_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::le_f64_u8, "le_elementwise");
cpu_binary_wrapper!(le_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::le_bf16_u8, "le_elementwise");
cpu_binary_wrapper!(le_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::le_f16_u8, "le_elementwise");

cpu_binary_wrapper!(gt_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::gt_f32_u8, "gt_elementwise");
cpu_binary_wrapper!(gt_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::gt_f64_u8, "gt_elementwise");
cpu_binary_wrapper!(gt_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::gt_bf16_u8, "gt_elementwise");
cpu_binary_wrapper!(gt_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::gt_f16_u8, "gt_elementwise");

cpu_binary_wrapper!(ge_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::ge_f32_u8, "ge_elementwise");
cpu_binary_wrapper!(ge_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::ge_f64_u8, "ge_elementwise");
cpu_binary_wrapper!(ge_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::ge_bf16_u8, "ge_elementwise");
cpu_binary_wrapper!(ge_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::ge_f16_u8, "ge_elementwise");

/// Build a `(3 inputs, 1 output)` CPU dispatch wrapper for the
/// ternary [`Op::Where`] family. Inputs are `(cond, a, b)` byte
/// buffers; output is the typed `T` result. The wrapper signature
/// matches [`KernelRef`]; the kernel itself does the typed casts
/// internally.
macro_rules! cpu_where_wrapper {
    ($wrapper:ident, $kernel:path, $op_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            _params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 3 inputs (cond, a, b), got {}",
                    $op_name, inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 1 output, got {}",
                    $op_name, outputs.len(),
                ))
                .bt());
            }
            let cond_guard = read_storage(&inputs[0])?;
            if cond_guard.dtype != DType::U8 {
                return Err(Error::Msg(format!(
                    "{}: cond must be U8, got {:?}",
                    $op_name, cond_guard.dtype,
                ))
                .bt());
            }
            let a_guard = read_storage(&inputs[1])?;
            let b_guard = read_storage(&inputs[2])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let cond_cpu = cpu_input(&cond_guard)?;
            let a_cpu = cpu_input(&a_guard)?;
            let b_cpu = cpu_input(&b_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(cond_cpu, a_cpu, b_cpu, out_cpu)
        }
    };
}

cpu_where_wrapper!(where_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::where_f32, "where");
cpu_where_wrapper!(where_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::where_f64, "where");
cpu_where_wrapper!(where_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::where_bf16, "where");
cpu_where_wrapper!(where_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::where_f16, "where");

// Rounding family (Floor / Ceil / Round) — same-dtype unary; standard
// `cpu_unary_wrapper!`.
cpu_unary_wrapper!(floor_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::floor_f32, "floor_elementwise");
cpu_unary_wrapper!(floor_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::floor_f64, "floor_elementwise");
cpu_unary_wrapper!(floor_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::floor_bf16, "floor_elementwise");
cpu_unary_wrapper!(floor_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::floor_f16, "floor_elementwise");

cpu_unary_wrapper!(ceil_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::ceil_f32, "ceil_elementwise");
cpu_unary_wrapper!(ceil_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::ceil_f64, "ceil_elementwise");
cpu_unary_wrapper!(ceil_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::ceil_bf16, "ceil_elementwise");
cpu_unary_wrapper!(ceil_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::ceil_f16, "ceil_elementwise");

cpu_unary_wrapper!(round_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::round_f32, "round_elementwise");
cpu_unary_wrapper!(round_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::round_f64, "round_elementwise");
cpu_unary_wrapper!(round_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::round_bf16, "round_elementwise");
cpu_unary_wrapper!(round_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::round_f16, "round_elementwise");

cpu_unary_wrapper!(sign_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::sign_f32, "sign_elementwise");
cpu_unary_wrapper!(sign_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::sign_f64, "sign_elementwise");
cpu_unary_wrapper!(sign_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::sign_bf16, "sign_elementwise");
cpu_unary_wrapper!(sign_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::sign_f16, "sign_elementwise");

cpu_unary_wrapper!(erf_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::erf_f32, "erf_elementwise");
cpu_unary_wrapper!(erf_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::erf_f64, "erf_elementwise");
cpu_unary_wrapper!(erf_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::erf_bf16, "erf_elementwise");
cpu_unary_wrapper!(erf_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::erf_f16, "erf_elementwise");

cpu_unary_wrapper!(gelu_erf_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::gelu_erf_f32, "gelu_erf_elementwise");
cpu_unary_wrapper!(gelu_erf_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::gelu_erf_f64, "gelu_erf_elementwise");
cpu_unary_wrapper!(gelu_erf_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::gelu_erf_bf16, "gelu_erf_elementwise");
cpu_unary_wrapper!(gelu_erf_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::gelu_erf_f16, "gelu_erf_elementwise");

cpu_binary_wrapper!(pow_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::pow_f32, "pow_elementwise");
cpu_binary_wrapper!(pow_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::pow_f64, "pow_elementwise");
cpu_binary_wrapper!(pow_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::pow_bf16, "pow_elementwise");
cpu_binary_wrapper!(pow_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::pow_f16, "pow_elementwise");

cpu_unary_wrapper!(rsqrt_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::rsqrt_f32, "rsqrt_elementwise");
cpu_unary_wrapper!(rsqrt_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::rsqrt_f64, "rsqrt_elementwise");
cpu_unary_wrapper!(rsqrt_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::rsqrt_bf16, "rsqrt_elementwise");
cpu_unary_wrapper!(rsqrt_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::rsqrt_f16, "rsqrt_elementwise");

cpu_binary_wrapper!(rem_elementwise_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::rem_f32, "rem_elementwise");
cpu_binary_wrapper!(rem_elementwise_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::rem_f64, "rem_elementwise");
cpu_binary_wrapper!(rem_elementwise_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::rem_bf16, "rem_elementwise");
cpu_binary_wrapper!(rem_elementwise_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::rem_f16, "rem_elementwise");

/// Dispatch wrapper for `(Flip, *, Cpu)`. Dtype-agnostic at the byte
/// level — `dtype_size` flows from the output Storage. Geometry
/// (`outer_count`, `dim_size`, `inner_count`) is precomputed by
/// `op_to_op_params` from the input shape + flip dim.
fn flip_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "flip wrapper expects 1 input + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let (outer, dim_size, inner) = match params {
        OpParams::Flip { outer_count, dim_size, inner_count } => {
            (*outer_count, *dim_size, *inner_count)
        }
        other => {
            return Err(Error::Msg(format!(
                "flip wrapper expects OpParams::Flip, got {other:?}",
            ))
            .bt())
        }
    };
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let dtype_size = out_guard.dtype.size_in_bytes();
    let in_cpu = cpu_input(&in_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::flip_cpu(
        in_cpu, out_cpu, outer, dim_size, inner, dtype_size,
    )
}

/// Build a `(1 input, 1 output)` CPU dispatch wrapper for the
/// per-dtype CumSum kernels. The kernel name is bound at macro
/// invocation; geometry comes from `OpParams::CumSum`.
macro_rules! cpu_cumsum_wrapper {
    ($wrapper:ident, $kernel:path, $op_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 1 input + 1 output, got {} + {}",
                    $op_name, inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let (outer, dim_size, inner) = match params {
                OpParams::CumSum { outer_count, dim_size, inner_count } => {
                    (*outer_count, *dim_size, *inner_count)
                }
                other => {
                    return Err(Error::Msg(format!(
                        "{} wrapper expects OpParams::CumSum, got {other:?}",
                        $op_name,
                    ))
                    .bt())
                }
            };
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(in_cpu, out_cpu, outer, dim_size, inner)
        }
    };
}

cpu_cumsum_wrapper!(cumsum_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::cumsum_f32, "cumsum_f32");
cpu_cumsum_wrapper!(cumsum_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::cumsum_f64, "cumsum_f64");
cpu_cumsum_wrapper!(cumsum_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::cumsum_bf16, "cumsum_bf16");
cpu_cumsum_wrapper!(cumsum_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::cumsum_f16, "cumsum_f16");

/// Triu / Tril share one byte-level kernel; the wrapper picks
/// keep_upper from the OpKind at dispatch time. Dtype-agnostic.
fn triu_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    triangular_wrapper_inner(inputs, outputs, params, /*keep_upper*/ true, "triu_cpu_wrapper")
}

fn tril_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    triangular_wrapper_inner(inputs, outputs, params, /*keep_upper*/ false, "tril_cpu_wrapper")
}

fn triangular_wrapper_inner(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
    keep_upper: bool,
    op_name: &str,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "{op_name}: expects 1 input + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        )).bt());
    }
    let (batch, rows, cols, diag) = match params {
        OpParams::Triangular { batch_count, rows, cols, diagonal } => {
            (*batch_count, *rows, *cols, *diagonal)
        }
        other => {
            return Err(Error::Msg(format!(
                "{op_name}: expects OpParams::Triangular, got {other:?}",
            )).bt());
        }
    };
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let dtype_size = out_guard.dtype.size_in_bytes();
    let in_cpu = cpu_input(&in_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::triangular_cpu(
        in_cpu, out_cpu, batch, rows, cols, diag, keep_upper, dtype_size,
    )
}

/// Per-dtype log-softmax wrapper. Geometry comes from
/// `OpParams::LogSoftmaxLastDim`.
macro_rules! cpu_log_softmax_wrapper {
    ($wrapper:ident, $kernel:path, $op_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 1 input + 1 output, got {} + {}",
                    $op_name, inputs.len(), outputs.len(),
                )).bt());
            }
            let (outer, last_dim) = match params {
                OpParams::LogSoftmaxLastDim { outer_count, last_dim } => (*outer_count, *last_dim),
                other => {
                    return Err(Error::Msg(format!(
                        "{} wrapper expects OpParams::LogSoftmaxLastDim, got {other:?}",
                        $op_name,
                    )).bt());
                }
            };
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(in_cpu, out_cpu, outer, last_dim)
        }
    };
}

cpu_log_softmax_wrapper!(log_softmax_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::log_softmax_last_dim_f32, "log_softmax_f32");
cpu_log_softmax_wrapper!(log_softmax_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::log_softmax_last_dim_f64, "log_softmax_f64");
cpu_log_softmax_wrapper!(log_softmax_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::log_softmax_last_dim_bf16, "log_softmax_bf16");
cpu_log_softmax_wrapper!(log_softmax_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::log_softmax_last_dim_f16, "log_softmax_f16");

/// Per-dtype log-softmax-backward wrapper. Two inputs (y, g); same
/// geometry as forward.
macro_rules! cpu_log_softmax_backward_wrapper {
    ($wrapper:ident, $kernel:path, $op_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 2 inputs (y, g) + 1 output, got {} + {}",
                    $op_name, inputs.len(), outputs.len(),
                )).bt());
            }
            let (outer, last_dim) = match params {
                OpParams::LogSoftmaxLastDim { outer_count, last_dim } => (*outer_count, *last_dim),
                other => {
                    return Err(Error::Msg(format!(
                        "{} wrapper expects OpParams::LogSoftmaxLastDim, got {other:?}",
                        $op_name,
                    )).bt());
                }
            };
            let y_guard = read_storage(&inputs[0])?;
            let g_guard = read_storage(&inputs[1])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let y_cpu = cpu_input(&y_guard)?;
            let g_cpu = cpu_input(&g_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(y_cpu, g_cpu, out_cpu, outer, last_dim)
        }
    };
}

cpu_log_softmax_backward_wrapper!(log_softmax_backward_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::log_softmax_last_dim_backward_f32, "log_softmax_backward_f32");
cpu_log_softmax_backward_wrapper!(log_softmax_backward_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::log_softmax_last_dim_backward_f64, "log_softmax_backward_f64");
cpu_log_softmax_backward_wrapper!(log_softmax_backward_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::log_softmax_last_dim_backward_bf16, "log_softmax_backward_bf16");
cpu_log_softmax_backward_wrapper!(log_softmax_backward_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::log_softmax_last_dim_backward_f16, "log_softmax_backward_f16");

/// Single dtype-agnostic MaskedFill wrapper. Reads `fill_bytes`
/// (pre-encoded by `op_to_op_params`) and dtype_size from the output.
fn masked_fill_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "masked_fill_cpu_wrapper: expects 2 inputs (x, mask) + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        )).bt());
    }
    let fill_bytes = match params {
        OpParams::MaskedFill { fill_bytes } => fill_bytes.clone(),
        other => {
            return Err(Error::Msg(format!(
                "masked_fill_cpu_wrapper: expects OpParams::MaskedFill, got {other:?}",
            )).bt());
        }
    };
    let in_guard = read_storage(&inputs[0])?;
    let mask_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let dtype_size = out_guard.dtype.size_in_bytes();
    let in_cpu = cpu_input(&in_guard)?;
    let mask_cpu = cpu_input(&mask_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::masked_fill_cpu(
        in_cpu, mask_cpu, out_cpu, &fill_bytes, dtype_size,
    )
}

/// Single dtype-agnostic dispatch wrapper for Pad. The kernel is
/// byte-level (`fill_bytes` is pre-encoded for the output dtype in
/// `op_to_op_params`); this wrapper just reads dtype_size from the
/// output Storage and passes through. One wrapper covers every
/// dtype the binding-table is registered for.
fn pad_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "pad wrapper expects 1 input + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let (in_shape, out_shape, padding, mode_tag, fill_bytes) = match params {
        OpParams::Pad { in_shape, out_shape, padding, mode_tag, fill_bytes } => {
            (in_shape, out_shape, padding, *mode_tag, fill_bytes)
        }
        other => {
            return Err(Error::Msg(format!(
                "pad wrapper expects OpParams::Pad, got {other:?}",
            ))
            .bt())
        }
    };
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let dtype_size = out_guard.dtype.size_in_bytes();
    let in_cpu = cpu_input(&in_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    // Per-mode forward dispatch.
    match mode_tag {
        0 => fuel_cpu_backend::byte_kernels::pad_const_cpu(
            in_cpu, out_cpu, in_shape, out_shape, padding, dtype_size, fill_bytes,
        ),
        1 => {
            // Reflect: validate before/after <= n-1 per axis (otherwise
            // the reflection runs off the input).
            for (k, (&n, &(b, a))) in in_shape.iter().zip(padding.iter()).enumerate() {
                if n > 0 && (b > n - 1 || a > n - 1) {
                    return Err(Error::Msg(format!(
                        "pad reflect: axis {k} has dim_size {n}; before ({b}) and \
                         after ({a}) must each be <= dim_size - 1",
                    )).bt());
                }
            }
            fuel_cpu_backend::byte_kernels::pad_reflect_cpu(
                in_cpu, out_cpu, in_shape, out_shape, padding, dtype_size,
            )
        }
        2 => fuel_cpu_backend::byte_kernels::pad_replicate_cpu(
            in_cpu, out_cpu, in_shape, out_shape, padding, dtype_size,
        ),
        other => Err(Error::Msg(format!(
            "pad: unknown mode_tag {other}",
        )).bt()),
    }
}

/// Build a `(1 input, 1 output)` per-dtype dispatch wrapper for
/// `Op::PadBackward`. Per-dtype (unlike forward Pad) because the
/// backward kernel does typed addition (accumulation per input slot).
macro_rules! cpu_pad_backward_wrapper {
    ($wrapper:ident, $kernel:path, $op_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 1 input + 1 output, got {} + {}",
                    $op_name, inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let (in_shape, out_shape, padding, mode_tag) = match params {
                OpParams::PadBackward { in_shape, out_shape, padding, mode_tag } => {
                    (in_shape, out_shape, padding, *mode_tag)
                }
                other => {
                    return Err(Error::Msg(format!(
                        "{} wrapper expects OpParams::PadBackward, got {other:?}",
                        $op_name,
                    ))
                    .bt())
                }
            };
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let in_cpu = cpu_input(&in_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(in_cpu, out_cpu, in_shape, out_shape, padding, mode_tag)
        }
    };
}

cpu_pad_backward_wrapper!(pad_backward_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::pad_backward_f32, "pad_backward_f32");
cpu_pad_backward_wrapper!(pad_backward_f64_cpu_wrapper, fuel_cpu_backend::byte_kernels::pad_backward_f64, "pad_backward_f64");
cpu_pad_backward_wrapper!(pad_backward_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::pad_backward_bf16, "pad_backward_bf16");
cpu_pad_backward_wrapper!(pad_backward_f16_cpu_wrapper, fuel_cpu_backend::byte_kernels::pad_backward_f16, "pad_backward_f16");

/// Dispatch wrapper for `(Roll, *, Cpu)`. Dtype-agnostic at the byte
/// level. Same shape as Flip plus a signed `shift`.
fn roll_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "roll wrapper expects 1 input + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let (outer, dim_size, inner, shift) = match params {
        OpParams::Roll { outer_count, dim_size, inner_count, shift } => {
            (*outer_count, *dim_size, *inner_count, *shift)
        }
        other => {
            return Err(Error::Msg(format!(
                "roll wrapper expects OpParams::Roll, got {other:?}",
            ))
            .bt())
        }
    };
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let dtype_size = out_guard.dtype.size_in_bytes();
    let in_cpu = cpu_input(&in_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::roll_cpu(
        in_cpu, out_cpu, outer, dim_size, inner, shift, dtype_size,
    )
}

/// Generate a CPU argextremum wrapper. Output dtype is U32; the
/// binding-table key is keyed on the OUTPUT dtype = U32. The
/// wrapper validates the input is F32 (only F32 is wired today).
macro_rules! cpu_arg_dim_wrapper {
    ($wrapper:ident, $kernel:path, $op_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 1 input + 1 output, got {} + {}",
                    $op_name, inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let dims = match params {
                OpParams::Reduce { dims, .. } => dims,
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
            let input_layout = layouts.first().ok_or_else(|| {
                Error::Msg(format!(
                    "{} wrapper: layouts empty (executor must pass input layout at [0])",
                    $op_name,
                ))
                .bt()
            })?;
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
    _layouts: &[Layout],
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
    _layouts: &[Layout],
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
    _layouts: &[Layout],
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
    _layouts: &[Layout],
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
    _layouts: &[Layout],
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
            _layouts: &[Layout],
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
            _layouts: &[Layout],
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
            _layouts: &[Layout],
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
    _layouts: &[Layout],
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
    _layouts: &[Layout],
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
            _layouts: &[Layout],
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
            _layouts: &[Layout],
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
            _layouts: &[Layout],
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
            _layouts: &[Layout],
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
            _layouts: &[Layout],
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
            _layouts: &[Layout],
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
            layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{}: 1 input + 1 output", $op_name,
                ))
                .bt());
            }
            let dims = match params {
                OpParams::Reduce { dims, .. } => dims,
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
            let input_layout = layouts.first().ok_or_else(|| {
                Error::Msg(format!(
                    "{}: layouts empty (executor must pass input layout at [0])",
                    $op_name,
                ))
                .bt()
            })?;
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
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    let in_dtype = read_storage(&inputs[0])?.dtype;
    match in_dtype {
        DType::F32 => argmax_dim_u32_cpu_wrapper(inputs, outputs, _layouts, params),
        DType::F64 => argmax_dim_f64_only_wrapper(inputs, outputs, _layouts, params),
        DType::BF16 => argmax_dim_bf16_only_wrapper(inputs, outputs, _layouts, params),
        DType::F16 => argmax_dim_f16_only_wrapper(inputs, outputs, _layouts, params),
        other => Err(Error::Msg(format!(
            "argmax_dim: unsupported input dtype {other:?}",
        ))
        .bt()),
    }
}

fn argmin_dim_u32_cpu_dispatch(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    let in_dtype = read_storage(&inputs[0])?.dtype;
    match in_dtype {
        DType::F32 => argmin_dim_u32_cpu_wrapper(inputs, outputs, _layouts, params),
        DType::F64 => argmin_dim_f64_only_wrapper(inputs, outputs, _layouts, params),
        DType::BF16 => argmin_dim_bf16_only_wrapper(inputs, outputs, _layouts, params),
        DType::F16 => argmin_dim_f16_only_wrapper(inputs, outputs, _layouts, params),
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
    _layouts: &[Layout],
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
    _layouts: &[Layout],
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
    _layouts: &[Layout],
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
            layouts: &[Layout],
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
            let dims = match params {
                OpParams::Reduce { dims, .. } => dims,
                other => {
                    return Err(Error::Msg(format!(
                        "{} wrapper expects OpParams::Reduce, got {:?}",
                        $op_name, other,
                    ))
                    .bt())
                }
            };
            let input_layout = layouts.first().ok_or_else(|| {
                Error::Msg(format!(
                    "{} wrapper: layouts empty (executor must pass input layout at [0])",
                    $op_name,
                ))
                .bt()
            })?;
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
            _layouts: &[Layout],
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
            _layouts: &[Layout],
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
            _layouts: &[Layout],
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
            _layouts: &[Layout],
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

/// Dispatch wrapper for `(ReduceMaxTo, *, Cpu)`. Single input → single
/// output; shapes flow through `OpParams::ReduceMaxTo`.
macro_rules! cpu_reduce_max_to_wrapper {
    ($name:ident, $kernel:path) => {
        fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "reduce_max_to wrapper expects 1 input, got {}",
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "reduce_max_to wrapper expects 1 output, got {}",
                    outputs.len(),
                ))
                .bt());
            }
            let (input_shape, output_shape) = match params {
                OpParams::ReduceMaxTo { input_shape, output_shape } => {
                    (input_shape.clone(), output_shape.clone())
                }
                other => {
                    return Err(Error::Msg(format!(
                        "reduce_max_to wrapper expects OpParams::ReduceMaxTo, got {other:?}",
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

cpu_reduce_max_to_wrapper!(reduce_max_to_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::reduce_max_to_f32);
cpu_reduce_max_to_wrapper!(reduce_max_to_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::reduce_max_to_f64);
cpu_reduce_max_to_wrapper!(reduce_max_to_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::reduce_max_to_bf16);
cpu_reduce_max_to_wrapper!(reduce_max_to_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::reduce_max_to_f16);

/// Dispatch wrapper for `(FusedLinear, *, Cpu)`. Three inputs
/// (lhs, rhs, bias). Reuses `OpParams::Matmul` for shape.
macro_rules! cpu_fused_linear_wrapper {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
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
            _layouts: &[Layout],
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
            _layouts: &[Layout],
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
    _layouts: &[Layout],
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
            _layouts: &[Layout],
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
    _layouts: &[Layout],
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
    let f64_dt = DType::F64;
    let bf16_dt = DType::BF16;
    let f16_dt  = DType::F16;
    let u32_dt = DType::U32;

    // Per-operand dtype-list shape helpers. The list captures all
    // operands the kernel sees — inputs in order, then outputs.
    // Variadic Concat uses the `unary` shape as a canonical
    // shorthand for "uniform-dtype across N inputs + output."
    let unary  = |t: DType| [t, t];                             // (in, out)
    let binary = |t: DType| [t, t, t];                          // (lhs, rhs, out)
    let u8_dt  = DType::U8;
    let compare = |t: DType| [t, t, u8_dt];                     // (lhs, rhs, U8 mask)
    let rope_dts = |t: DType| [t, t, t, t];                     // (x, cos, sin, out)
    let conv2d_no_bias   = |t: DType| [t, t, t];                // (x, w, out)
    let conv2d_with_bias = |t: DType| [t, t, t, t];             // (x, w, bias, out)
    let fused_linear     = |t: DType| [t, t, t, t];             // (lhs, rhs, bias, out)
    let flash_attn_no_alibi   = |t: DType| [t, t, t, t];        // (q, k, v, out)
    let flash_attn_with_alibi = |t: DType| [t, t, t, t, t];     // (q, k, v, alibi, out)
    let paged_attn_no_alibi   = |t: DType| [t, t, t, u32_dt, u32_dt, t];      // q,kc,vc,bt,cl,out
    let paged_attn_with_alibi = |t: DType| [t, t, t, u32_dt, u32_dt, t, t];   // +alibi
    let index_select  = |t: DType| [t, u32_dt, t];              // (data, indices, out)
    let gather_dts    = |t: DType| [t, u32_dt, t];              // (data, indices, out)
    let index_add_dts = |t: DType| [t, u32_dt, t, t];           // (base, indices, src, out)
    let scatter_add   = |t: DType| [t, u32_dt, t, t];           // (base, indices, src, out)

    // Elementwise binary / unary — F32.
    table.register(AddElementwise,   &binary(f32_dt), cpu, add_elementwise_f32_cpu_wrapper);
    table.register(SubElementwise,   &binary(f32_dt), cpu, sub_elementwise_f32_cpu_wrapper);
    table.register(MulElementwise,   &binary(f32_dt), cpu, mul_elementwise_f32_cpu_wrapper);
    table.register(DivElementwise,   &binary(f32_dt), cpu, div_elementwise_f32_cpu_wrapper);

    table.register(ReluElementwise,    &unary(f32_dt), cpu, relu_elementwise_f32_cpu_wrapper);
    table.register(NegElementwise,     &unary(f32_dt), cpu, neg_elementwise_f32_cpu_wrapper);
    table.register(SqrElementwise,     &unary(f32_dt), cpu, sqr_elementwise_f32_cpu_wrapper);
    table.register(SqrtElementwise,    &unary(f32_dt), cpu, sqrt_elementwise_f32_cpu_wrapper);
    table.register(RecipElementwise,   &unary(f32_dt), cpu, recip_elementwise_f32_cpu_wrapper);
    table.register(AbsElementwise,     &unary(f32_dt), cpu, abs_elementwise_f32_cpu_wrapper);
    table.register(TanhElementwise,    &unary(f32_dt), cpu, tanh_elementwise_f32_cpu_wrapper);
    table.register(ExpElementwise,     &unary(f32_dt), cpu, exp_elementwise_f32_cpu_wrapper);
    table.register(LogElementwise,     &unary(f32_dt), cpu, log_elementwise_f32_cpu_wrapper);
    table.register(SinElementwise,     &unary(f32_dt), cpu, sin_elementwise_f32_cpu_wrapper);
    table.register(CosElementwise,     &unary(f32_dt), cpu, cos_elementwise_f32_cpu_wrapper);
    table.register(SigmoidElementwise, &unary(f32_dt), cpu, sigmoid_elementwise_f32_cpu_wrapper);
    table.register(SiluElementwise,    &unary(f32_dt), cpu, silu_elementwise_f32_cpu_wrapper);
    table.register(GeluElementwise,    &unary(f32_dt), cpu, gelu_elementwise_f32_cpu_wrapper);
    table.register(StepElementwise,    &unary(f32_dt), cpu, step_elementwise_f32_cpu_wrapper);

    // Elementwise binary / unary — F64.
    table.register(AddElementwise,     &binary(f64_dt), cpu, add_elementwise_f64_cpu_wrapper);
    table.register(SubElementwise,     &binary(f64_dt), cpu, sub_elementwise_f64_cpu_wrapper);
    table.register(MulElementwise,     &binary(f64_dt), cpu, mul_elementwise_f64_cpu_wrapper);
    table.register(DivElementwise,     &binary(f64_dt), cpu, div_elementwise_f64_cpu_wrapper);
    table.register(ReluElementwise,    &unary(f64_dt), cpu, relu_elementwise_f64_cpu_wrapper);
    table.register(NegElementwise,     &unary(f64_dt), cpu, neg_elementwise_f64_cpu_wrapper);
    table.register(SqrElementwise,     &unary(f64_dt), cpu, sqr_elementwise_f64_cpu_wrapper);
    table.register(SqrtElementwise,    &unary(f64_dt), cpu, sqrt_elementwise_f64_cpu_wrapper);
    table.register(RecipElementwise,   &unary(f64_dt), cpu, recip_elementwise_f64_cpu_wrapper);
    table.register(AbsElementwise,     &unary(f64_dt), cpu, abs_elementwise_f64_cpu_wrapper);
    table.register(TanhElementwise,    &unary(f64_dt), cpu, tanh_elementwise_f64_cpu_wrapper);
    table.register(ExpElementwise,     &unary(f64_dt), cpu, exp_elementwise_f64_cpu_wrapper);
    table.register(LogElementwise,     &unary(f64_dt), cpu, log_elementwise_f64_cpu_wrapper);
    table.register(SinElementwise,     &unary(f64_dt), cpu, sin_elementwise_f64_cpu_wrapper);
    table.register(CosElementwise,     &unary(f64_dt), cpu, cos_elementwise_f64_cpu_wrapper);
    table.register(SigmoidElementwise, &unary(f64_dt), cpu, sigmoid_elementwise_f64_cpu_wrapper);
    table.register(SiluElementwise,    &unary(f64_dt), cpu, silu_elementwise_f64_cpu_wrapper);
    table.register(GeluElementwise,    &unary(f64_dt), cpu, gelu_elementwise_f64_cpu_wrapper);
    table.register(StepElementwise,    &unary(f64_dt), cpu, step_elementwise_f64_cpu_wrapper);

    // Reductions.
    table.register(SumReduce,          &unary(f32_dt), cpu, sum_reduce_f32_cpu_wrapper);
    table.register(MaxReduce,          &unary(f32_dt), cpu, max_reduce_f32_cpu_wrapper);
    table.register(MinReduce,          &unary(f32_dt), cpu, min_reduce_f32_cpu_wrapper);
    table.register(MeanReduce,         &unary(f32_dt), cpu, mean_reduce_f32_cpu_wrapper);
    table.register(SumReduce,          &unary(f64_dt), cpu, sum_reduce_f64_cpu_wrapper);
    table.register(MaxReduce,          &unary(f64_dt), cpu, max_reduce_f64_cpu_wrapper);
    table.register(MinReduce,          &unary(f64_dt), cpu, min_reduce_f64_cpu_wrapper);
    table.register(MeanReduce,         &unary(f64_dt), cpu, mean_reduce_f64_cpu_wrapper);

    table.register(MatMul,             &binary(f32_dt),  cpu, matmul_f32_cpu_wrapper);
    table.register(MatMul,             &binary(f64_dt),  cpu, matmul_f64_cpu_wrapper);
    table.register(MatMul,             &binary(bf16_dt), cpu, matmul_bf16_cpu_wrapper);
    table.register(MatMul,             &binary(f16_dt),  cpu, matmul_f16_cpu_wrapper);

    // bf16 + f16 reductions — accumulate in f32 for stability.
    table.register(SumReduce,          &unary(bf16_dt), cpu, sum_reduce_bf16_cpu_wrapper);
    table.register(MaxReduce,          &unary(bf16_dt), cpu, max_reduce_bf16_cpu_wrapper);
    table.register(MinReduce,          &unary(bf16_dt), cpu, min_reduce_bf16_cpu_wrapper);
    table.register(MeanReduce,         &unary(bf16_dt), cpu, mean_reduce_bf16_cpu_wrapper);
    table.register(SumReduce,          &unary(f16_dt),  cpu, sum_reduce_f16_cpu_wrapper);
    table.register(MaxReduce,          &unary(f16_dt),  cpu, max_reduce_f16_cpu_wrapper);
    table.register(MinReduce,          &unary(f16_dt),  cpu, min_reduce_f16_cpu_wrapper);
    table.register(MeanReduce,         &unary(f16_dt),  cpu, mean_reduce_f16_cpu_wrapper);

    // Cast — CPU wrappers are still keyed on the *target* dtype and
    // dispatch internally on the source. Register `[T, T]` for each
    // target dtype to preserve current behavior; the binding table's
    // multi-dtype key shape lets callers pass `[src, dst]` and still
    // hit the right wrapper because src is matched inside.
    //
    // TODO(phase 7.5 cast cleanup): split each cast wrapper into
    // (src, dst) pairs once enough source dtypes are exercised; the
    // binding key already supports it.
    table.register(Cast, &unary(DType::F32),  cpu, cast_to_f32_cpu_wrapper);
    table.register(Cast, &unary(DType::F64),  cpu, cast_to_f64_cpu_wrapper);
    table.register(Cast, &unary(DType::BF16), cpu, cast_to_bf16_cpu_wrapper);
    table.register(Cast, &unary(DType::F16),  cpu, cast_to_f16_cpu_wrapper);

    // Conv2D — register both no-bias (3 operands) and with-bias
    // (4 operands) shapes per dtype; the wrapper handles both.
    for (dt, w) in [
        (f32_dt,  conv2d_f32_cpu_wrapper  as KernelRef),
        (f64_dt,  conv2d_f64_cpu_wrapper),
        (bf16_dt, conv2d_bf16_cpu_wrapper),
        (f16_dt,  conv2d_f16_cpu_wrapper),
    ] {
        table.register(Conv2D, &conv2d_no_bias(dt),   cpu, w);
        table.register(Conv2D, &conv2d_with_bias(dt), cpu, w);
    }
    for (dt, w) in [
        (f32_dt,  conv_transpose2d_f32_cpu_wrapper as KernelRef),
        (f64_dt,  conv_transpose2d_f64_cpu_wrapper),
        (bf16_dt, conv_transpose2d_bf16_cpu_wrapper),
        (f16_dt,  conv_transpose2d_f16_cpu_wrapper),
    ] {
        table.register(ConvTranspose2D, &conv2d_no_bias(dt),   cpu, w);
        table.register(ConvTranspose2D, &conv2d_with_bias(dt), cpu, w);
    }

    table.register(ReduceSumTo, &unary(f32_dt),  cpu, reduce_sum_to_f32_cpu_wrapper);
    table.register(ReduceSumTo, &unary(f64_dt),  cpu, reduce_sum_to_f64_cpu_wrapper);
    table.register(ReduceSumTo, &unary(bf16_dt), cpu, reduce_sum_to_bf16_cpu_wrapper);
    table.register(ReduceSumTo, &unary(f16_dt),  cpu, reduce_sum_to_f16_cpu_wrapper);

    table.register(ReduceMaxTo, &unary(f32_dt),  cpu, reduce_max_to_f32_cpu_wrapper);
    table.register(ReduceMaxTo, &unary(f64_dt),  cpu, reduce_max_to_f64_cpu_wrapper);
    table.register(ReduceMaxTo, &unary(bf16_dt), cpu, reduce_max_to_bf16_cpu_wrapper);
    table.register(ReduceMaxTo, &unary(f16_dt),  cpu, reduce_max_to_f16_cpu_wrapper);

    // FusedLinear: 3 inputs (lhs, rhs, bias) → out.
    table.register(FusedLinear, &fused_linear(f32_dt),  cpu, fused_linear_f32_cpu_wrapper);
    table.register(FusedLinear, &fused_linear(f64_dt),  cpu, fused_linear_f64_cpu_wrapper);
    table.register(FusedLinear, &fused_linear(bf16_dt), cpu, fused_linear_bf16_cpu_wrapper);
    table.register(FusedLinear, &fused_linear(f16_dt),  cpu, fused_linear_f16_cpu_wrapper);

    // FlashAttn — register both 3-input (q,k,v) and 4-input
    // (q,k,v,alibi) shapes per dtype.
    for (dt, w) in [
        (f32_dt,  flash_attn_f32_cpu_wrapper  as KernelRef),
        (f64_dt,  flash_attn_f64_cpu_wrapper),
        (bf16_dt, flash_attn_bf16_cpu_wrapper),
        (f16_dt,  flash_attn_f16_cpu_wrapper),
    ] {
        table.register(FlashAttn, &flash_attn_no_alibi(dt),   cpu, w);
        table.register(FlashAttn, &flash_attn_with_alibi(dt), cpu, w);
    }

    // PagedAttn — block_table + ctx_lens are always U32; alibi is
    // optional.
    for (dt, w) in [
        (f32_dt,  paged_attn_f32_cpu_wrapper  as KernelRef),
        (f64_dt,  paged_attn_f64_cpu_wrapper),
        (bf16_dt, paged_attn_bf16_cpu_wrapper),
        (f16_dt,  paged_attn_f16_cpu_wrapper),
    ] {
        table.register(PagedAttn, &paged_attn_no_alibi(dt),   cpu, w);
        table.register(PagedAttn, &paged_attn_with_alibi(dt), cpu, w);
    }

    table.register(Affine,             &unary(f32_dt), cpu, affine_f32_cpu_wrapper);
    table.register(ClampElementwise,   &unary(f32_dt), cpu, clamp_elementwise_f32_cpu_wrapper);
    table.register(PowIElementwise,    &unary(f32_dt), cpu, powi_elementwise_f32_cpu_wrapper);
    table.register(MaximumElementwise, &binary(f32_dt), cpu, maximum_elementwise_f32_cpu_wrapper);
    table.register(MinimumElementwise, &binary(f32_dt), cpu, minimum_elementwise_f32_cpu_wrapper);
    table.register(MaximumElementwise, &binary(f64_dt), cpu, maximum_elementwise_f64_cpu_wrapper);
    table.register(MinimumElementwise, &binary(f64_dt), cpu, minimum_elementwise_f64_cpu_wrapper);

    // Comparison family (output dtype = U8). Each kernel produces a
    // U8 mask (`1` where the predicate holds, `0` otherwise).
    table.register(EqualElementwise, &compare(f32_dt),  cpu, eq_elementwise_f32_cpu_wrapper);
    table.register(EqualElementwise, &compare(f64_dt),  cpu, eq_elementwise_f64_cpu_wrapper);
    table.register(EqualElementwise, &compare(bf16_dt), cpu, eq_elementwise_bf16_cpu_wrapper);
    table.register(EqualElementwise, &compare(f16_dt),  cpu, eq_elementwise_f16_cpu_wrapper);

    table.register(NotEqualElementwise, &compare(f32_dt),  cpu, ne_elementwise_f32_cpu_wrapper);
    table.register(NotEqualElementwise, &compare(f64_dt),  cpu, ne_elementwise_f64_cpu_wrapper);
    table.register(NotEqualElementwise, &compare(bf16_dt), cpu, ne_elementwise_bf16_cpu_wrapper);
    table.register(NotEqualElementwise, &compare(f16_dt),  cpu, ne_elementwise_f16_cpu_wrapper);

    table.register(LessElementwise, &compare(f32_dt),  cpu, lt_elementwise_f32_cpu_wrapper);
    table.register(LessElementwise, &compare(f64_dt),  cpu, lt_elementwise_f64_cpu_wrapper);
    table.register(LessElementwise, &compare(bf16_dt), cpu, lt_elementwise_bf16_cpu_wrapper);
    table.register(LessElementwise, &compare(f16_dt),  cpu, lt_elementwise_f16_cpu_wrapper);

    table.register(LessEqualElementwise, &compare(f32_dt),  cpu, le_elementwise_f32_cpu_wrapper);
    table.register(LessEqualElementwise, &compare(f64_dt),  cpu, le_elementwise_f64_cpu_wrapper);
    table.register(LessEqualElementwise, &compare(bf16_dt), cpu, le_elementwise_bf16_cpu_wrapper);
    table.register(LessEqualElementwise, &compare(f16_dt),  cpu, le_elementwise_f16_cpu_wrapper);

    table.register(GreaterElementwise, &compare(f32_dt),  cpu, gt_elementwise_f32_cpu_wrapper);
    table.register(GreaterElementwise, &compare(f64_dt),  cpu, gt_elementwise_f64_cpu_wrapper);
    table.register(GreaterElementwise, &compare(bf16_dt), cpu, gt_elementwise_bf16_cpu_wrapper);
    table.register(GreaterElementwise, &compare(f16_dt),  cpu, gt_elementwise_f16_cpu_wrapper);

    table.register(GreaterEqualElementwise, &compare(f32_dt),  cpu, ge_elementwise_f32_cpu_wrapper);
    table.register(GreaterEqualElementwise, &compare(f64_dt),  cpu, ge_elementwise_f64_cpu_wrapper);
    table.register(GreaterEqualElementwise, &compare(bf16_dt), cpu, ge_elementwise_bf16_cpu_wrapper);
    table.register(GreaterEqualElementwise, &compare(f16_dt),  cpu, ge_elementwise_f16_cpu_wrapper);

    // Ternary select. Binding-table dtype list is `[U8, T, T, T]`:
    // cond + lhs + rhs + output. Per-dtype kernel for each `T`.
    let where_dts = |t: DType| [u8_dt, t, t, t];
    table.register(Where, &where_dts(f32_dt),  cpu, where_f32_cpu_wrapper);
    table.register(Where, &where_dts(f64_dt),  cpu, where_f64_cpu_wrapper);
    table.register(Where, &where_dts(bf16_dt), cpu, where_bf16_cpu_wrapper);
    table.register(Where, &where_dts(f16_dt),  cpu, where_f16_cpu_wrapper);

    // Rounding family — standard unary shape `[T, T]`.
    table.register(FloorElementwise, &unary(f32_dt),  cpu, floor_elementwise_f32_cpu_wrapper);
    table.register(FloorElementwise, &unary(f64_dt),  cpu, floor_elementwise_f64_cpu_wrapper);
    table.register(FloorElementwise, &unary(bf16_dt), cpu, floor_elementwise_bf16_cpu_wrapper);
    table.register(FloorElementwise, &unary(f16_dt),  cpu, floor_elementwise_f16_cpu_wrapper);

    table.register(CeilElementwise, &unary(f32_dt),  cpu, ceil_elementwise_f32_cpu_wrapper);
    table.register(CeilElementwise, &unary(f64_dt),  cpu, ceil_elementwise_f64_cpu_wrapper);
    table.register(CeilElementwise, &unary(bf16_dt), cpu, ceil_elementwise_bf16_cpu_wrapper);
    table.register(CeilElementwise, &unary(f16_dt),  cpu, ceil_elementwise_f16_cpu_wrapper);

    table.register(RoundElementwise, &unary(f32_dt),  cpu, round_elementwise_f32_cpu_wrapper);
    table.register(RoundElementwise, &unary(f64_dt),  cpu, round_elementwise_f64_cpu_wrapper);
    table.register(RoundElementwise, &unary(bf16_dt), cpu, round_elementwise_bf16_cpu_wrapper);
    table.register(RoundElementwise, &unary(f16_dt),  cpu, round_elementwise_f16_cpu_wrapper);

    table.register(SignElementwise, &unary(f32_dt),  cpu, sign_elementwise_f32_cpu_wrapper);
    table.register(SignElementwise, &unary(f64_dt),  cpu, sign_elementwise_f64_cpu_wrapper);
    table.register(SignElementwise, &unary(bf16_dt), cpu, sign_elementwise_bf16_cpu_wrapper);
    table.register(SignElementwise, &unary(f16_dt),  cpu, sign_elementwise_f16_cpu_wrapper);

    table.register(ErfElementwise, &unary(f32_dt),  cpu, erf_elementwise_f32_cpu_wrapper);
    table.register(ErfElementwise, &unary(f64_dt),  cpu, erf_elementwise_f64_cpu_wrapper);
    table.register(ErfElementwise, &unary(bf16_dt), cpu, erf_elementwise_bf16_cpu_wrapper);
    table.register(ErfElementwise, &unary(f16_dt),  cpu, erf_elementwise_f16_cpu_wrapper);

    table.register(GeluErfElementwise, &unary(f32_dt),  cpu, gelu_erf_elementwise_f32_cpu_wrapper);
    table.register(GeluErfElementwise, &unary(f64_dt),  cpu, gelu_erf_elementwise_f64_cpu_wrapper);
    table.register(GeluErfElementwise, &unary(bf16_dt), cpu, gelu_erf_elementwise_bf16_cpu_wrapper);
    table.register(GeluErfElementwise, &unary(f16_dt),  cpu, gelu_erf_elementwise_f16_cpu_wrapper);

    table.register(PowElementwise, &binary(f32_dt),  cpu, pow_elementwise_f32_cpu_wrapper);
    table.register(PowElementwise, &binary(f64_dt),  cpu, pow_elementwise_f64_cpu_wrapper);
    table.register(PowElementwise, &binary(bf16_dt), cpu, pow_elementwise_bf16_cpu_wrapper);
    table.register(PowElementwise, &binary(f16_dt),  cpu, pow_elementwise_f16_cpu_wrapper);

    table.register(RsqrtElementwise, &unary(f32_dt),  cpu, rsqrt_elementwise_f32_cpu_wrapper);
    table.register(RsqrtElementwise, &unary(f64_dt),  cpu, rsqrt_elementwise_f64_cpu_wrapper);
    table.register(RsqrtElementwise, &unary(bf16_dt), cpu, rsqrt_elementwise_bf16_cpu_wrapper);
    table.register(RsqrtElementwise, &unary(f16_dt),  cpu, rsqrt_elementwise_f16_cpu_wrapper);

    table.register(RemElementwise, &binary(f32_dt),  cpu, rem_elementwise_f32_cpu_wrapper);
    table.register(RemElementwise, &binary(f64_dt),  cpu, rem_elementwise_f64_cpu_wrapper);
    table.register(RemElementwise, &binary(bf16_dt), cpu, rem_elementwise_bf16_cpu_wrapper);
    table.register(RemElementwise, &binary(f16_dt),  cpu, rem_elementwise_f16_cpu_wrapper);

    // Flip and Roll are dtype-agnostic at the byte level — the
    // wrappers read `dtype_size` from the output Storage. Register
    // one wrapper per dtype so the binding-table key matches the
    // execute-time dtype list.
    table.register(Flip, &unary(f32_dt),  cpu, flip_cpu_wrapper);
    table.register(Flip, &unary(f64_dt),  cpu, flip_cpu_wrapper);
    table.register(Flip, &unary(bf16_dt), cpu, flip_cpu_wrapper);
    table.register(Flip, &unary(f16_dt),  cpu, flip_cpu_wrapper);
    table.register(Flip, &unary(u32_dt),  cpu, flip_cpu_wrapper);
    table.register(Flip, &unary(u8_dt),   cpu, flip_cpu_wrapper);

    table.register(Roll, &unary(f32_dt),  cpu, roll_cpu_wrapper);
    table.register(Roll, &unary(f64_dt),  cpu, roll_cpu_wrapper);
    table.register(Roll, &unary(bf16_dt), cpu, roll_cpu_wrapper);
    table.register(Roll, &unary(f16_dt),  cpu, roll_cpu_wrapper);
    table.register(Roll, &unary(u32_dt),  cpu, roll_cpu_wrapper);
    table.register(Roll, &unary(u8_dt),   cpu, roll_cpu_wrapper);

    // CumSum is per-dtype (typed accumulation, not byte copy).
    table.register(CumSum, &unary(f32_dt),  cpu, cumsum_f32_cpu_wrapper);
    table.register(CumSum, &unary(f64_dt),  cpu, cumsum_f64_cpu_wrapper);
    table.register(CumSum, &unary(bf16_dt), cpu, cumsum_bf16_cpu_wrapper);
    table.register(CumSum, &unary(f16_dt),  cpu, cumsum_f16_cpu_wrapper);

    // Triu / Tril share one byte-level kernel (dtype-agnostic).
    table.register(Triu, &unary(f32_dt),  cpu, triu_cpu_wrapper);
    table.register(Triu, &unary(f64_dt),  cpu, triu_cpu_wrapper);
    table.register(Triu, &unary(bf16_dt), cpu, triu_cpu_wrapper);
    table.register(Triu, &unary(f16_dt),  cpu, triu_cpu_wrapper);
    table.register(Triu, &unary(u32_dt),  cpu, triu_cpu_wrapper);
    table.register(Triu, &unary(u8_dt),   cpu, triu_cpu_wrapper);
    table.register(Tril, &unary(f32_dt),  cpu, tril_cpu_wrapper);
    table.register(Tril, &unary(f64_dt),  cpu, tril_cpu_wrapper);
    table.register(Tril, &unary(bf16_dt), cpu, tril_cpu_wrapper);
    table.register(Tril, &unary(f16_dt),  cpu, tril_cpu_wrapper);
    table.register(Tril, &unary(u32_dt),  cpu, tril_cpu_wrapper);
    table.register(Tril, &unary(u8_dt),   cpu, tril_cpu_wrapper);

    // LogSoftmaxLastDim — per-dtype.
    table.register(LogSoftmaxLastDim, &unary(f32_dt),  cpu, log_softmax_f32_cpu_wrapper);
    table.register(LogSoftmaxLastDim, &unary(f64_dt),  cpu, log_softmax_f64_cpu_wrapper);
    table.register(LogSoftmaxLastDim, &unary(bf16_dt), cpu, log_softmax_bf16_cpu_wrapper);
    table.register(LogSoftmaxLastDim, &unary(f16_dt),  cpu, log_softmax_f16_cpu_wrapper);

    // LogSoftmaxLastDimBackward — per-dtype, two inputs (y, g) → out.
    table.register(LogSoftmaxLastDimBackward, &binary(f32_dt),  cpu, log_softmax_backward_f32_cpu_wrapper);
    table.register(LogSoftmaxLastDimBackward, &binary(f64_dt),  cpu, log_softmax_backward_f64_cpu_wrapper);
    table.register(LogSoftmaxLastDimBackward, &binary(bf16_dt), cpu, log_softmax_backward_bf16_cpu_wrapper);
    table.register(LogSoftmaxLastDimBackward, &binary(f16_dt),  cpu, log_softmax_backward_f16_cpu_wrapper);

    // MaskedFill — dtype-agnostic byte kernel; binding-table key is
    // [T, U8, T] (x dtype, mask U8, output == x).
    let masked_dtypes = |t: DType| [t, DType::U8, t];
    table.register(MaskedFill, &masked_dtypes(f32_dt),  cpu, masked_fill_cpu_wrapper);
    table.register(MaskedFill, &masked_dtypes(f64_dt),  cpu, masked_fill_cpu_wrapper);
    table.register(MaskedFill, &masked_dtypes(bf16_dt), cpu, masked_fill_cpu_wrapper);
    table.register(MaskedFill, &masked_dtypes(f16_dt),  cpu, masked_fill_cpu_wrapper);
    table.register(MaskedFill, &masked_dtypes(u32_dt),  cpu, masked_fill_cpu_wrapper);
    table.register(MaskedFill, &masked_dtypes(u8_dt),   cpu, masked_fill_cpu_wrapper);

    // Pad (Constant mode wired; Reflect/Replicate fall through to a
    // clean error inside the wrapper). Single dtype-agnostic wrapper
    // registered per dtype — kernel reads dtype_size from output.
    table.register(Pad, &unary(f32_dt),  cpu, pad_cpu_wrapper);
    table.register(Pad, &unary(f64_dt),  cpu, pad_cpu_wrapper);
    table.register(Pad, &unary(bf16_dt), cpu, pad_cpu_wrapper);
    table.register(Pad, &unary(f16_dt),  cpu, pad_cpu_wrapper);
    table.register(Pad, &unary(u32_dt),  cpu, pad_cpu_wrapper);
    table.register(Pad, &unary(u8_dt),   cpu, pad_cpu_wrapper);

    // PadBackward — per-dtype since accumulation is typed.
    table.register(PadBackward, &unary(f32_dt),  cpu, pad_backward_f32_cpu_wrapper);
    table.register(PadBackward, &unary(f64_dt),  cpu, pad_backward_f64_cpu_wrapper);
    table.register(PadBackward, &unary(bf16_dt), cpu, pad_backward_bf16_cpu_wrapper);
    table.register(PadBackward, &unary(f16_dt),  cpu, pad_backward_f16_cpu_wrapper);

    // bf16 + f16 elementwise — via-f32 round-trip kernels.
    table.register(AddElementwise,     &binary(bf16_dt), cpu, add_elementwise_bf16_cpu_wrapper);
    table.register(SubElementwise,     &binary(bf16_dt), cpu, sub_elementwise_bf16_cpu_wrapper);
    table.register(MulElementwise,     &binary(bf16_dt), cpu, mul_elementwise_bf16_cpu_wrapper);
    table.register(DivElementwise,     &binary(bf16_dt), cpu, div_elementwise_bf16_cpu_wrapper);
    table.register(MaximumElementwise, &binary(bf16_dt), cpu, maximum_elementwise_bf16_cpu_wrapper);
    table.register(MinimumElementwise, &binary(bf16_dt), cpu, minimum_elementwise_bf16_cpu_wrapper);
    table.register(ReluElementwise,    &unary(bf16_dt),  cpu, relu_elementwise_bf16_cpu_wrapper);
    table.register(NegElementwise,     &unary(bf16_dt),  cpu, neg_elementwise_bf16_cpu_wrapper);
    table.register(SqrElementwise,     &unary(bf16_dt),  cpu, sqr_elementwise_bf16_cpu_wrapper);
    table.register(SqrtElementwise,    &unary(bf16_dt),  cpu, sqrt_elementwise_bf16_cpu_wrapper);
    table.register(RecipElementwise,   &unary(bf16_dt),  cpu, recip_elementwise_bf16_cpu_wrapper);
    table.register(AbsElementwise,     &unary(bf16_dt),  cpu, abs_elementwise_bf16_cpu_wrapper);
    table.register(TanhElementwise,    &unary(bf16_dt),  cpu, tanh_elementwise_bf16_cpu_wrapper);
    table.register(ExpElementwise,     &unary(bf16_dt),  cpu, exp_elementwise_bf16_cpu_wrapper);
    table.register(LogElementwise,     &unary(bf16_dt),  cpu, log_elementwise_bf16_cpu_wrapper);
    table.register(SinElementwise,     &unary(bf16_dt),  cpu, sin_elementwise_bf16_cpu_wrapper);
    table.register(CosElementwise,     &unary(bf16_dt),  cpu, cos_elementwise_bf16_cpu_wrapper);
    table.register(SigmoidElementwise, &unary(bf16_dt),  cpu, sigmoid_elementwise_bf16_cpu_wrapper);
    table.register(SiluElementwise,    &unary(bf16_dt),  cpu, silu_elementwise_bf16_cpu_wrapper);
    table.register(GeluElementwise,    &unary(bf16_dt),  cpu, gelu_elementwise_bf16_cpu_wrapper);
    table.register(StepElementwise,    &unary(bf16_dt),  cpu, step_elementwise_bf16_cpu_wrapper);

    table.register(AddElementwise,     &binary(f16_dt), cpu, add_elementwise_f16_cpu_wrapper);
    table.register(SubElementwise,     &binary(f16_dt), cpu, sub_elementwise_f16_cpu_wrapper);
    table.register(MulElementwise,     &binary(f16_dt), cpu, mul_elementwise_f16_cpu_wrapper);
    table.register(DivElementwise,     &binary(f16_dt), cpu, div_elementwise_f16_cpu_wrapper);
    table.register(MaximumElementwise, &binary(f16_dt), cpu, maximum_elementwise_f16_cpu_wrapper);
    table.register(MinimumElementwise, &binary(f16_dt), cpu, minimum_elementwise_f16_cpu_wrapper);
    table.register(ReluElementwise,    &unary(f16_dt),  cpu, relu_elementwise_f16_cpu_wrapper);
    table.register(NegElementwise,     &unary(f16_dt),  cpu, neg_elementwise_f16_cpu_wrapper);
    table.register(SqrElementwise,     &unary(f16_dt),  cpu, sqr_elementwise_f16_cpu_wrapper);
    table.register(SqrtElementwise,    &unary(f16_dt),  cpu, sqrt_elementwise_f16_cpu_wrapper);
    table.register(RecipElementwise,   &unary(f16_dt),  cpu, recip_elementwise_f16_cpu_wrapper);
    table.register(AbsElementwise,     &unary(f16_dt),  cpu, abs_elementwise_f16_cpu_wrapper);
    table.register(TanhElementwise,    &unary(f16_dt),  cpu, tanh_elementwise_f16_cpu_wrapper);
    table.register(ExpElementwise,     &unary(f16_dt),  cpu, exp_elementwise_f16_cpu_wrapper);
    table.register(LogElementwise,     &unary(f16_dt),  cpu, log_elementwise_f16_cpu_wrapper);
    table.register(SinElementwise,     &unary(f16_dt),  cpu, sin_elementwise_f16_cpu_wrapper);
    table.register(CosElementwise,     &unary(f16_dt),  cpu, cos_elementwise_f16_cpu_wrapper);
    table.register(SigmoidElementwise, &unary(f16_dt),  cpu, sigmoid_elementwise_f16_cpu_wrapper);
    table.register(SiluElementwise,    &unary(f16_dt),  cpu, silu_elementwise_f16_cpu_wrapper);
    table.register(GeluElementwise,    &unary(f16_dt),  cpu, gelu_elementwise_f16_cpu_wrapper);
    table.register(StepElementwise,    &unary(f16_dt),  cpu, step_elementwise_f16_cpu_wrapper);

    // Concat is a variadic uniform-dtype op (N inputs, all the same
    // dtype, plus output). Register the canonical `[T, T]` shorthand
    // per supported dtype; the lookup site collapses the actual N+1
    // dtype list to this same shorthand.
    for dt in [
        DType::F32, DType::F64, DType::BF16, DType::F16,
        DType::U32, DType::U8, DType::I16, DType::I32, DType::I64,
    ] {
        table.register(Concat, &unary(dt), cpu, concat_cpu_wrapper);
    }

    table.register(SoftmaxLastDim,   &unary(f32_dt),  cpu, softmax_last_dim_f32_cpu_wrapper);
    table.register(SoftmaxLastDim,   &unary(bf16_dt), cpu, softmax_last_dim_bf16_cpu_wrapper);
    table.register(SoftmaxLastDim,   &unary(f16_dt),  cpu, softmax_last_dim_f16_cpu_wrapper);
    table.register(SoftmaxLastDim,   &unary(f64_dt),  cpu, softmax_last_dim_f64_cpu_wrapper);
    table.register(RmsNormLastDim,   &unary(f32_dt),  cpu, rms_norm_last_dim_f32_cpu_wrapper);
    table.register(RmsNormLastDim,   &unary(bf16_dt), cpu, rms_norm_last_dim_bf16_cpu_wrapper);
    table.register(RmsNormLastDim,   &unary(f16_dt),  cpu, rms_norm_last_dim_f16_cpu_wrapper);
    table.register(RmsNormLastDim,   &unary(f64_dt),  cpu, rms_norm_last_dim_f64_cpu_wrapper);
    table.register(LayerNormLastDim, &unary(f32_dt),  cpu, layer_norm_last_dim_f32_cpu_wrapper);
    table.register(LayerNormLastDim, &unary(bf16_dt), cpu, layer_norm_last_dim_bf16_cpu_wrapper);
    table.register(LayerNormLastDim, &unary(f16_dt),  cpu, layer_norm_last_dim_f16_cpu_wrapper);
    table.register(LayerNormLastDim, &unary(f64_dt),  cpu, layer_norm_last_dim_f64_cpu_wrapper);

    // IndexSelect / Gather: data + U32 indices → data.
    for dt in [
        DType::F32, DType::F64, DType::BF16, DType::F16,
        DType::U32, DType::U8, DType::I16, DType::I32, DType::I64,
    ] {
        table.register(IndexSelect, &index_select(dt), cpu, index_select_cpu_wrapper);
        table.register(Gather,      &gather_dts(dt),   cpu, gather_cpu_wrapper);
    }

    // Rope: x + cos + sin → out, all same dtype.
    table.register(Rope, &rope_dts(f32_dt),  cpu, rope_f32_cpu_wrapper);
    table.register(Rope, &rope_dts(bf16_dt), cpu, rope_bf16_cpu_wrapper);
    table.register(Rope, &rope_dts(f16_dt),  cpu, rope_f16_cpu_wrapper);
    table.register(Rope, &rope_dts(f64_dt),  cpu, rope_f64_cpu_wrapper);

    // QMatMul: F32 activations, U32 weight blocks, F32 output.
    table.register(QMatMul, &[f32_dt, u32_dt, f32_dt], cpu, qmatmul_f32_cpu_wrapper);

    // IndexAdd / ScatterAdd: base + U32 indices + src → out (base shape).
    table.register(IndexAdd,   &index_add_dts(f32_dt),  cpu, index_add_f32_cpu_wrapper);
    table.register(IndexAdd,   &index_add_dts(f64_dt),  cpu, index_add_f64_cpu_wrapper);
    table.register(IndexAdd,   &index_add_dts(bf16_dt), cpu, index_add_bf16_cpu_wrapper);
    table.register(IndexAdd,   &index_add_dts(f16_dt),  cpu, index_add_f16_cpu_wrapper);
    table.register(ScatterAdd, &scatter_add(f32_dt),    cpu, scatter_add_f32_cpu_wrapper);
    table.register(ScatterAdd, &scatter_add(f64_dt),    cpu, scatter_add_f64_cpu_wrapper);
    table.register(ScatterAdd, &scatter_add(bf16_dt),   cpu, scatter_add_bf16_cpu_wrapper);
    table.register(ScatterAdd, &scatter_add(f16_dt),    cpu, scatter_add_f16_cpu_wrapper);

    // ArgMax/ArgMin: input dtype varies, output is U32. The dispatch
    // wrapper still does its internal input-dtype match (preserves
    // current behavior). Register `[input_dt, U32]` once per input
    // dtype the dispatcher handles so the binding table can also
    // route directly when we collapse the wrapper later.
    for dt in [f32_dt, f64_dt, bf16_dt, f16_dt] {
        table.register(ArgMaxDim, &[dt, u32_dt], cpu, argmax_dim_u32_cpu_dispatch);
        table.register(ArgMinDim, &[dt, u32_dt], cpu, argmin_dim_u32_cpu_dispatch);
    }

    table.register(Affine,             &unary(f64_dt),  cpu, affine_f64_cpu_wrapper);
    table.register(Affine,             &unary(bf16_dt), cpu, affine_bf16_cpu_wrapper);
    table.register(Affine,             &unary(f16_dt),  cpu, affine_f16_cpu_wrapper);
    table.register(ClampElementwise,   &unary(f64_dt),  cpu, clamp_f64_cpu_wrapper);
    table.register(ClampElementwise,   &unary(bf16_dt), cpu, clamp_bf16_cpu_wrapper);
    table.register(ClampElementwise,   &unary(f16_dt),  cpu, clamp_f16_cpu_wrapper);
    table.register(PowIElementwise,    &unary(f64_dt),  cpu, powi_f64_cpu_wrapper);
    table.register(PowIElementwise,    &unary(bf16_dt), cpu, powi_bf16_cpu_wrapper);
    table.register(PowIElementwise,    &unary(f16_dt),  cpu, powi_f16_cpu_wrapper);
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

/// Helper: extract `(lhs_layout, rhs_layout)` from a binary kernel's
/// `layouts` slice. Layouts is laid out as `[lhs, rhs, output]` for
/// binary ops; this peels off the two inputs. Errors with `wrapper`
/// in the message if the slice is too short — direct callers (tests)
/// must construct a 3-element slice.
#[cfg(feature = "cuda")]
fn binary_input_layouts<'a>(
    wrapper: &'static str,
    layouts: &'a [Layout],
) -> Result<(&'a Layout, &'a Layout)> {
    if layouts.len() < 3 {
        return Err(Error::Msg(format!(
            "{wrapper}: expected layouts of len 3 (lhs, rhs, output), got {}",
            layouts.len(),
        ))
        .bt());
    }
    Ok((&layouts[0], &layouts[1]))
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
    layouts: &[Layout],
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
    let (lhs_layout, rhs_layout) = binary_input_layouts("add_elementwise_f32_cuda_wrapper", layouts)?;
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::add_elementwise_f32(
        lhs_cuda, rhs_cuda, lhs_layout, rhs_layout,
    )?;
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
    layouts: &[Layout],
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
    let (lhs_layout, rhs_layout) = binary_input_layouts("sub_elementwise_f32_cuda_wrapper", layouts)?;
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::sub_elementwise_f32(
        lhs_cuda, rhs_cuda, lhs_layout, rhs_layout,
    )?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(MulElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn mul_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
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
    let (lhs_layout, rhs_layout) = binary_input_layouts("mul_elementwise_f32_cuda_wrapper", layouts)?;
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::mul_elementwise_f32(
        lhs_cuda, rhs_cuda, lhs_layout, rhs_layout,
    )?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(DivElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn div_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
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
    let (lhs_layout, rhs_layout) = binary_input_layouts("div_elementwise_f32_cuda_wrapper", layouts)?;
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::div_elementwise_f32(
        lhs_cuda, rhs_cuda, lhs_layout, rhs_layout,
    )?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(MaximumElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn maximum_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
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
    let (lhs_layout, rhs_layout) = binary_input_layouts("maximum_elementwise_f32_cuda_wrapper", layouts)?;
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::maximum_elementwise_f32(
        lhs_cuda, rhs_cuda, lhs_layout, rhs_layout,
    )?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(MinimumElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn minimum_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
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
    let (lhs_layout, rhs_layout) = binary_input_layouts("minimum_elementwise_f32_cuda_wrapper", layouts)?;
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::minimum_elementwise_f32(
        lhs_cuda, rhs_cuda, lhs_layout, rhs_layout,
    )?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(ReluElementwise, F32, Cuda)`. First CUDA
/// unary op through the unified binding table; subsequent unary
/// fanout entries reuse the shared `unary_elementwise_f32` helper
/// in `fuel-cuda-backend::byte_kernels`, so they are one-line
/// delegations + a wrapper of this exact shape.
#[cfg(feature = "cuda")]
fn relu_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "relu_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "relu_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::relu_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(NegElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn neg_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "neg_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "neg_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::neg_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(SqrElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn sqr_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "sqr_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "sqr_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::sqr_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(SqrtElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn sqrt_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "sqrt_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "sqrt_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::sqrt_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(RecipElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn recip_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "recip_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "recip_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::recip_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(AbsElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn abs_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "abs_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "abs_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::abs_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(TanhElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn tanh_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "tanh_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "tanh_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::tanh_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(ExpElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn exp_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "exp_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "exp_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::exp_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(LogElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn log_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "log_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "log_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::log_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(SinElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn sin_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "sin_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "sin_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::sin_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(CosElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn cos_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "cos_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "cos_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::cos_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(SigmoidElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn sigmoid_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "sigmoid_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "sigmoid_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::sigmoid_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(SiluElementwise, F32, Cuda)`.
#[cfg(feature = "cuda")]
fn silu_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "silu_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "silu_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::silu_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(GeluElementwise, F32, Cuda)`.
/// Maps to the tanh-approximation kernel `ugelu_f32` (matches
/// the CPU `OpKind::GeluElementwise` semantics).
#[cfg(feature = "cuda")]
fn gelu_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "gelu_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "gelu_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::gelu_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(StepElementwise, F32, Cuda)`. Heaviside
/// step: `1.0` where `x > 0`, `0.0` otherwise — matches the CPU
/// `step_f32` semantics. Backed by `ustep_f32`, which is introduced
/// to `fuel-cuda-kernels::UNARY` alongside this wrapper.
#[cfg(feature = "cuda")]
fn step_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "step_elementwise_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "step_elementwise_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::step_elementwise_f32(src_cuda)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(SumReduce, F32, Cuda)`. First reduction op
/// through the unified binding table; mirrors the CPU
/// `cpu_reduce_wrapper` macro. Reads the input layout from
/// `layouts[0]` (executor side-channel) and the reduce dims from
/// `OpParams::Reduce`. Subsequent Max/Min/Mean wrappers share the
/// same shape.
#[cfg(feature = "cuda")]
fn sum_reduce_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "sum_reduce_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "sum_reduce_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let dims = match params {
        OpParams::Reduce { dims, .. } => dims,
        other => {
            return Err(Error::Msg(format!(
                "sum_reduce_f32_cuda_wrapper: expected OpParams::Reduce, got {:?}",
                other,
            ))
            .bt())
        }
    };
    let input_layout = layouts.first().ok_or_else(|| {
        Error::Msg("sum_reduce_f32_cuda_wrapper: layouts empty".to_string()).bt()
    })?;
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::sum_reduce_f32(src_cuda, input_layout, dims)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(MaxReduce, F32, Cuda)`. Same shape as
/// `sum_reduce_f32_cuda_wrapper`; only the byte-kernel call differs.
#[cfg(feature = "cuda")]
fn max_reduce_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "max_reduce_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "max_reduce_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let dims = match params {
        OpParams::Reduce { dims, .. } => dims,
        other => {
            return Err(Error::Msg(format!(
                "max_reduce_f32_cuda_wrapper: expected OpParams::Reduce, got {:?}",
                other,
            ))
            .bt())
        }
    };
    let input_layout = layouts.first().ok_or_else(|| {
        Error::Msg("max_reduce_f32_cuda_wrapper: layouts empty".to_string()).bt()
    })?;
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::max_reduce_f32(src_cuda, input_layout, dims)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(MinReduce, F32, Cuda)`. Same shape as
/// `sum_reduce_f32_cuda_wrapper`; only the byte-kernel call differs.
#[cfg(feature = "cuda")]
fn min_reduce_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "min_reduce_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "min_reduce_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let dims = match params {
        OpParams::Reduce { dims, .. } => dims,
        other => {
            return Err(Error::Msg(format!(
                "min_reduce_f32_cuda_wrapper: expected OpParams::Reduce, got {:?}",
                other,
            ))
            .bt())
        }
    };
    let input_layout = layouts.first().ok_or_else(|| {
        Error::Msg("min_reduce_f32_cuda_wrapper: layouts empty".to_string()).bt()
    })?;
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::min_reduce_f32(src_cuda, input_layout, dims)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(MeanReduce, F32, Cuda)`. Mirrors the CPU
/// approach: composes `fast_sum_f32` + an `affine_f32` scale-by-
/// `(1/divisor)`. Same wrapper shape as the other reduction
/// wrappers; the byte-kernel handles the two-launch composition.
#[cfg(feature = "cuda")]
fn mean_reduce_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 {
        return Err(Error::Msg(format!(
            "mean_reduce_f32_cuda_wrapper: expected 1 input, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "mean_reduce_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let dims = match params {
        OpParams::Reduce { dims, .. } => dims,
        other => {
            return Err(Error::Msg(format!(
                "mean_reduce_f32_cuda_wrapper: expected OpParams::Reduce, got {:?}",
                other,
            ))
            .bt())
        }
    };
    let input_layout = layouts.first().ok_or_else(|| {
        Error::Msg("mean_reduce_f32_cuda_wrapper: layouts empty".to_string()).bt()
    })?;
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::mean_reduce_f32(src_cuda, input_layout, dims)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(ReduceSumTo, F32, Cuda)`. Maps the
/// broadcast-aligned target shape from `OpParams::ReduceSumTo` to
/// reduce dims and dispatches through the existing `fast_sum_f32`
/// kernel. PR 3.5-followup: drops the CPU-fallback round-trip cost
/// the lowered SoftmaxLastDim was paying on CUDA.
#[cfg(feature = "cuda")]
fn reduce_sum_to_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "reduce_sum_to_f32_cuda_wrapper: expected 1 input + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let (input_shape, output_shape) = match params {
        OpParams::ReduceSumTo { input_shape, output_shape } => {
            (input_shape.clone(), output_shape.clone())
        }
        other => {
            return Err(Error::Msg(format!(
                "reduce_sum_to_f32_cuda_wrapper: expected OpParams::ReduceSumTo, got {other:?}",
            ))
            .bt())
        }
    };
    let input_layout = layouts.first().ok_or_else(|| {
        Error::Msg("reduce_sum_to_f32_cuda_wrapper: layouts empty".to_string()).bt()
    })?;
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::reduce_sum_to_f32(
        src_cuda, input_layout, &input_shape, &output_shape,
    )?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(ReduceMaxTo, F32, Cuda)`. Symmetric of
/// `reduce_sum_to_f32_cuda_wrapper` — only the byte-kernel call
/// differs (`fast_max_f32` instead of `fast_sum_f32`).
#[cfg(feature = "cuda")]
fn reduce_max_to_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "reduce_max_to_f32_cuda_wrapper: expected 1 input + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let (input_shape, output_shape) = match params {
        OpParams::ReduceMaxTo { input_shape, output_shape } => {
            (input_shape.clone(), output_shape.clone())
        }
        other => {
            return Err(Error::Msg(format!(
                "reduce_max_to_f32_cuda_wrapper: expected OpParams::ReduceMaxTo, got {other:?}",
            ))
            .bt())
        }
    };
    let input_layout = layouts.first().ok_or_else(|| {
        Error::Msg("reduce_max_to_f32_cuda_wrapper: layouts empty".to_string()).bt()
    })?;
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::reduce_max_to_f32(
        src_cuda, input_layout, &input_shape, &output_shape,
    )?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(ArgMaxDim, [F32, U32], Cuda)`. Output is
/// U32 indices into the reduce dim; the reduce dim is dropped from
/// the output shape. Mirrors the CPU `argmax_dim_u32_cpu_dispatch`
/// but specialized to F32 source dtype.
#[cfg(feature = "cuda")]
fn argmax_dim_u32_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    arg_extremum_dim_f32_cuda_wrapper(inputs, outputs, layouts, params, true)
}

/// Dispatch wrapper for `(ArgMinDim, [F32, U32], Cuda)`. Sister of
/// `argmax_dim_u32_f32_cuda_wrapper`.
#[cfg(feature = "cuda")]
fn argmin_dim_u32_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    arg_extremum_dim_f32_cuda_wrapper(inputs, outputs, layouts, params, false)
}

/// Shared body for argmax/argmin F32 wrappers. Flag picks between
/// the byte-kernel entry; the rest of the validation + decoding is
/// identical.
#[cfg(feature = "cuda")]
fn arg_extremum_dim_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
    params: &OpParams,
    is_argmax: bool,
) -> Result<()> {
    let op_name = if is_argmax { "argmax_dim" } else { "argmin_dim" };
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "{op_name}_f32_cuda_wrapper: expected 1 input + 1 output, got {} + {}",
            inputs.len(),
            outputs.len(),
        ))
        .bt());
    }
    let dim = match params {
        OpParams::Reduce { dims, keepdim: _ } => {
            if dims.len() != 1 {
                return Err(Error::Msg(format!(
                    "{op_name}_f32_cuda_wrapper: argmax/argmin reduces a single dim; got {dims:?}",
                ))
                .bt());
            }
            dims[0]
        }
        other => {
            return Err(Error::Msg(format!(
                "{op_name}_f32_cuda_wrapper: expected OpParams::Reduce, got {other:?}",
            ))
            .bt())
        }
    };
    let input_layout = layouts.first().ok_or_else(|| {
        Error::Msg(format!("{op_name}_f32_cuda_wrapper: layouts empty")).bt()
    })?;
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = if is_argmax {
        fuel_cuda_backend::byte_kernels::argmax_dim_f32(src_cuda, input_layout, dim)?
    } else {
        fuel_cuda_backend::byte_kernels::argmin_dim_f32(src_cuda, input_layout, dim)?
    };
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(Concat, F32, Cuda)`. Concatenate N F32
/// inputs along one dim via the `concat_f32` PTX kernel (one launch
/// per input, accumulating `input_idx_offset`). Mirrors the CPU
/// `concat_cpu_wrapper`.
#[cfg(feature = "cuda")]
fn concat_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.is_empty() {
        return Err(Error::Msg(
            "concat_f32_cuda_wrapper: at least 1 input required".to_string(),
        )
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "concat_f32_cuda_wrapper: expected 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let (outer_count, input_dim_sizes, inner_count) = match params {
        OpParams::Concat { outer_count, input_dim_sizes, inner_count } => {
            (*outer_count, input_dim_sizes.clone(), *inner_count)
        }
        other => {
            return Err(Error::Msg(format!(
                "concat_f32_cuda_wrapper: expected OpParams::Concat, got {other:?}",
            ))
            .bt())
        }
    };
    if input_dim_sizes.len() != inputs.len() {
        return Err(Error::Msg(format!(
            "concat_f32_cuda_wrapper: OpParams declares {} inputs but the work item carries {}",
            input_dim_sizes.len(),
            inputs.len(),
        ))
        .bt());
    }
    let in_guards: Vec<_> = inputs
        .iter()
        .map(read_storage)
        .collect::<Result<Vec<_>>>()?;
    let mut in_cudas: Vec<&fuel_cuda_backend::CudaStorageBytes> = Vec::with_capacity(in_guards.len());
    for g in &in_guards {
        in_cudas.push(cuda_input(g)?);
    }
    let mut out_guard = write_storage(&outputs[0])?;
    let result = fuel_cuda_backend::byte_kernels::concat_f32(
        &in_cudas, outer_count, &input_dim_sizes, inner_count,
    )?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(PowIElementwise, F32, Cuda)`. Element-wise
/// `y = x^exp` for an integer `exp`, via the `upowi_f32` PTX kernel
/// (square-and-multiply for bit-exact parity with `f32::powi`).
/// Mirrors the CPU `powi_elementwise_f32_cpu_wrapper`.
#[cfg(feature = "cuda")]
fn powi_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "powi_elementwise_f32_cuda_wrapper: expected 1 input + 1 output, got {} + {}",
            inputs.len(),
            outputs.len(),
        ))
        .bt());
    }
    let exp = match params {
        OpParams::PowI { exp } => *exp,
        other => {
            return Err(Error::Msg(format!(
                "powi_elementwise_f32_cuda_wrapper: expected OpParams::PowI, got {other:?}",
            ))
            .bt())
        }
    };
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::powi_f32(src_cuda, exp)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(ClampElementwise, F32, Cuda)`. Element-wise
/// `y = min(max(x, lo), hi)` via the `uclamp_f32` PTX kernel (UNARY_OP2),
/// on byte-shaped CUDA storage. Mirrors the CPU
/// `clamp_elementwise_f32_cpu_wrapper`.
#[cfg(feature = "cuda")]
fn clamp_elementwise_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "clamp_elementwise_f32_cuda_wrapper: expected 1 input + 1 output, got {} + {}",
            inputs.len(),
            outputs.len(),
        ))
        .bt());
    }
    let (lo, hi) = match params {
        OpParams::Clamp { min, max } => (*min as f32, *max as f32),
        other => {
            return Err(Error::Msg(format!(
                "clamp_elementwise_f32_cuda_wrapper: expected OpParams::Clamp, got {other:?}",
            ))
            .bt())
        }
    };
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::clamp_f32(src_cuda, lo, hi)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(Gather, [F32, U32, F32], Cuda)`. Mirrors
/// the CPU `gather_cpu_wrapper`; pulls `(source_shape, output_shape,
/// dim)` from `OpParams::Gather`. Indices must be U32 (matches the
/// `gather_u32_f32` PTX kernel symbol).
#[cfg(feature = "cuda")]
fn gather_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "gather_f32_cuda_wrapper: expected 2 inputs + 1 output, got {} + {}",
            inputs.len(),
            outputs.len(),
        ))
        .bt());
    }
    let (source_shape, output_shape, dim) = match params {
        OpParams::Gather { source_shape, output_shape, dim } => {
            (source_shape.clone(), output_shape.clone(), *dim)
        }
        other => {
            return Err(Error::Msg(format!(
                "gather_f32_cuda_wrapper: expected OpParams::Gather, got {other:?}",
            ))
            .bt())
        }
    };
    let src_guard = read_storage(&inputs[0])?;
    let ids_guard = read_storage(&inputs[1])?;
    if ids_guard.dtype != DType::U32 {
        return Err(Error::Msg(format!(
            "gather_f32_cuda_wrapper: indices must be U32, got {:?}",
            ids_guard.dtype,
        ))
        .bt());
    }
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&src_guard)?;
    let ids_cuda = cuda_input(&ids_guard)?;
    let result = fuel_cuda_backend::byte_kernels::gather_f32(
        src_cuda, ids_cuda, &source_shape, &output_shape, dim,
    )?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(IndexSelect, [F32, U32, F32], Cuda)`. The
/// `OpParams::IndexSelect` carries the four pre-computed counts the
/// kernel needs (`outer_count`, `source_dim_size`, `n_indices`,
/// `inner_count`) — the executor's `OpParams::for_node` derives these
/// from the source layout and selected dim before dispatch reaches us.
/// Mirrors the CPU `index_select_cpu_wrapper`; the indices must be U32
/// (matches the `is_u32_f32` PTX kernel; the U8/I64 index variants
/// have their own kernel symbols and would register as separate
/// binding-table entries when needed).
#[cfg(feature = "cuda")]
fn index_select_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "index_select_f32_cuda_wrapper: expected 2 inputs (source, indices), got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "index_select_f32_cuda_wrapper: expected 1 output, got {}",
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
                "index_select_f32_cuda_wrapper: expected OpParams::IndexSelect, got {other:?}",
            ))
            .bt())
        }
    };
    let src_guard = read_storage(&inputs[0])?;
    let ids_guard = read_storage(&inputs[1])?;
    if ids_guard.dtype != DType::U32 {
        return Err(Error::Msg(format!(
            "index_select_f32_cuda_wrapper: indices must be U32, got {:?}",
            ids_guard.dtype,
        ))
        .bt());
    }
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&src_guard)?;
    let ids_cuda = cuda_input(&ids_guard)?;
    let result = fuel_cuda_backend::byte_kernels::index_select_f32(
        src_cuda, ids_cuda, outer_count, source_dim_size, n_indices, inner_count,
    )?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(Affine, F32, Cuda)`. Element-wise
/// `y = mul * x + add` via the `affine_f32` PTX kernel, on byte-shaped
/// CUDA storage. Mirrors the CPU `affine_f32_cpu_wrapper`.
#[cfg(feature = "cuda")]
fn affine_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "affine_f32_cuda_wrapper: expected 1 input + 1 output, got {} + {}",
            inputs.len(),
            outputs.len(),
        ))
        .bt());
    }
    let (mul, add) = match params {
        OpParams::Affine { mul, add } => (*mul as f32, *add as f32),
        other => {
            return Err(Error::Msg(format!(
                "affine_f32_cuda_wrapper: expected OpParams::Affine, got {:?}",
                other,
            ))
            .bt())
        }
    };
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::affine_f32(src_cuda, mul, add)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(Cast, [src_dt, dst_dt], Cuda)`. First op
/// where input dtype != output dtype — exercises the multi-dtype
/// binding-table key. The dtypes flow from the input/output Storages
/// (the binding-table lookup already filtered to a registered pair),
/// so the wrapper just hands them to `byte_kernels::cast`.
///
/// One wrapper covers every registered (src, dst) pair: each pair
/// gets its own binding-table entry pointing at this same function,
/// so the lookup picks the entry by exact dtype match and the kernel
/// name is built from the storages' actual dtypes inside the kernel.
#[cfg(feature = "cuda")]
fn cast_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "cast_cuda_wrapper: expected 1 input + 1 output, got {} + {}",
            inputs.len(),
            outputs.len(),
        ))
        .bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let src_dtype = in_guard.dtype;
    let dst_dtype = out_guard.dtype;
    let src_cuda = cuda_input(&in_guard)?;
    let result = fuel_cuda_backend::byte_kernels::cast(src_cuda, src_dtype, dst_dtype)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(MatMul, F32, Cuda)`. First non-PTX, non-
/// element-wise op through the unified binding table — the underlying
/// implementation is cuBLAS `gemm_strided_batched_ex`, not a launched
/// kernel from `fuel-cuda-kernels`. Mirrors the CPU
/// `matmul_f32_cpu_wrapper` for `OpParams::Matmul` destructuring.
#[cfg(feature = "cuda")]
fn matmul_f32_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "matmul_f32_cuda_wrapper: expected 2 inputs, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "matmul_f32_cuda_wrapper: expected 1 output, got {}",
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
                "matmul_f32_cuda_wrapper: expected OpParams::Matmul, got {:?}",
                other,
            ))
            .bt())
        }
    };
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::matmul_f32(
        lhs_cuda,
        rhs_cuda,
        lhs_batch_dims,
        rhs_batch_dims,
        m,
        n,
        k,
    )?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(MatMul, BF16, Cuda)`. Routes through
/// the CUTLASS `LayoutSku::Rrr` SKU in `fuel-cuda-backend`. Net-new
/// bf16 CUDA matmul coverage on the byte-storage substrate (no
/// existing cuBLAS bf16 binding-table entry to be sibling with —
/// Phase 7.6 step 9 migrates primitive ops to the
/// `FusedKernelRegistry` shape where multiple alternatives at one
/// decision point become the architecture-target).
#[cfg(feature = "cuda")]
fn matmul_bf16_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "matmul_bf16_cuda_wrapper: expected 2 inputs, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "matmul_bf16_cuda_wrapper: expected 1 output, got {}",
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
                "matmul_bf16_cuda_wrapper: expected OpParams::Matmul, got {:?}",
                other,
            ))
            .bt())
        }
    };
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::matmul_bf16(
        lhs_cuda,
        rhs_cuda,
        lhs_batch_dims,
        rhs_batch_dims,
        m,
        n,
        k,
    )?;
    let out_cuda = cuda_output(&mut out_guard)?;
    *out_cuda = result;
    Ok(())
}

/// Dispatch wrapper for `(MatMul, F16, Cuda)`. Mirror of
/// `matmul_bf16_cuda_wrapper` at `f16`; same CUTLASS Rrr path.
#[cfg(feature = "cuda")]
fn matmul_f16_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "matmul_f16_cuda_wrapper: expected 2 inputs, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "matmul_f16_cuda_wrapper: expected 1 output, got {}",
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
                "matmul_f16_cuda_wrapper: expected OpParams::Matmul, got {:?}",
                other,
            ))
            .bt())
        }
    };
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cuda = cuda_input(&lhs_guard)?;
    let rhs_cuda = cuda_input(&rhs_guard)?;
    let result = fuel_cuda_backend::byte_kernels::matmul_f16(
        lhs_cuda,
        rhs_cuda,
        lhs_batch_dims,
        rhs_batch_dims,
        m,
        n,
        k,
    )?;
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
    let bf16_dt = DType::BF16;
    let f16_dt = DType::F16;
    let unary  = |t: DType| [t, t];
    let binary = |t: DType| [t, t, t];

    // Binary F32 elementwise ops opt in to strided_input — the
    // PTX BINARY_OP kernels walk per-input strides, so non-contiguous
    // inputs (broadcast, transpose) reach the wrapper as metadata-only
    // views rather than going through executor-side auto-Contiguize.
    let strided = KernelCaps::strided_input();
    table.register_with_caps(AddElementwise,     &binary(f32_dt), cuda, add_elementwise_f32_cuda_wrapper, strided);
    table.register_with_caps(SubElementwise,     &binary(f32_dt), cuda, sub_elementwise_f32_cuda_wrapper, strided);
    table.register_with_caps(MulElementwise,     &binary(f32_dt), cuda, mul_elementwise_f32_cuda_wrapper, strided);
    table.register_with_caps(DivElementwise,     &binary(f32_dt), cuda, div_elementwise_f32_cuda_wrapper, strided);
    table.register_with_caps(MaximumElementwise, &binary(f32_dt), cuda, maximum_elementwise_f32_cuda_wrapper, strided);
    table.register_with_caps(MinimumElementwise, &binary(f32_dt), cuda, minimum_elementwise_f32_cuda_wrapper, strided);

    table.register(ReluElementwise,    &unary(f32_dt), cuda, relu_elementwise_f32_cuda_wrapper);
    table.register(NegElementwise,     &unary(f32_dt), cuda, neg_elementwise_f32_cuda_wrapper);
    table.register(SqrElementwise,     &unary(f32_dt), cuda, sqr_elementwise_f32_cuda_wrapper);
    table.register(SqrtElementwise,    &unary(f32_dt), cuda, sqrt_elementwise_f32_cuda_wrapper);
    table.register(RecipElementwise,   &unary(f32_dt), cuda, recip_elementwise_f32_cuda_wrapper);
    table.register(AbsElementwise,     &unary(f32_dt), cuda, abs_elementwise_f32_cuda_wrapper);
    table.register(TanhElementwise,    &unary(f32_dt), cuda, tanh_elementwise_f32_cuda_wrapper);
    table.register(ExpElementwise,     &unary(f32_dt), cuda, exp_elementwise_f32_cuda_wrapper);
    table.register(LogElementwise,     &unary(f32_dt), cuda, log_elementwise_f32_cuda_wrapper);
    table.register(SinElementwise,     &unary(f32_dt), cuda, sin_elementwise_f32_cuda_wrapper);
    table.register(CosElementwise,     &unary(f32_dt), cuda, cos_elementwise_f32_cuda_wrapper);
    table.register(SigmoidElementwise, &unary(f32_dt), cuda, sigmoid_elementwise_f32_cuda_wrapper);
    table.register(SiluElementwise,    &unary(f32_dt), cuda, silu_elementwise_f32_cuda_wrapper);
    table.register(GeluElementwise,    &unary(f32_dt), cuda, gelu_elementwise_f32_cuda_wrapper);
    table.register(StepElementwise,    &unary(f32_dt), cuda, step_elementwise_f32_cuda_wrapper);

    table.register(SumReduce,          &unary(f32_dt), cuda, sum_reduce_f32_cuda_wrapper);
    table.register(MaxReduce,          &unary(f32_dt), cuda, max_reduce_f32_cuda_wrapper);
    table.register(MinReduce,          &unary(f32_dt), cuda, min_reduce_f32_cuda_wrapper);
    table.register(MeanReduce,         &unary(f32_dt), cuda, mean_reduce_f32_cuda_wrapper);

    table.register(ReduceSumTo,        &unary(f32_dt), cuda, reduce_sum_to_f32_cuda_wrapper);
    table.register(ReduceMaxTo,        &unary(f32_dt), cuda, reduce_max_to_f32_cuda_wrapper);

    table.register(MatMul,             &binary(f32_dt), cuda, matmul_f32_cuda_wrapper);
    table.register(MatMul,             &binary(bf16_dt), cuda, matmul_bf16_cuda_wrapper);
    table.register(MatMul,             &binary(f16_dt), cuda, matmul_f16_cuda_wrapper);
    table.register(Affine,             &unary(f32_dt),  cuda, affine_f32_cuda_wrapper);
    table.register(ClampElementwise,   &unary(f32_dt),  cuda, clamp_elementwise_f32_cuda_wrapper);
    table.register(PowIElementwise,    &unary(f32_dt),  cuda, powi_elementwise_f32_cuda_wrapper);
    table.register(Concat,             &unary(f32_dt),  cuda, concat_f32_cuda_wrapper);

    // IndexSelect / Gather: gather data from an F32 source via U32
    // indices. Dtype binding key matches the CPU shape:
    // `[source_dt, U32, source_dt]`. Other (source, index) pairs
    // (F32×U8, F32×I64, F64×U32, …) register as their own entries
    // when their PTX kernels and CPU mirrors are wired through.
    let u32_dt = DType::U32;
    table.register(IndexSelect, &[f32_dt, u32_dt, f32_dt], cuda, index_select_f32_cuda_wrapper);
    table.register(Gather,      &[f32_dt, u32_dt, f32_dt], cuda, gather_f32_cuda_wrapper);

    // ArgMaxDim / ArgMinDim: F32 source → U32 indices into the reduce dim.
    // Dtype binding key matches the CPU `argmax_dim_u32_cpu_dispatch` shape:
    // `[source_dt, U32]`. Other source dtypes (F64 / BF16 / F16) register
    // as their own entries when their PTX paths are wired through.
    table.register(ArgMaxDim, &[f32_dt, u32_dt], cuda, argmax_dim_u32_f32_cuda_wrapper);
    table.register(ArgMinDim, &[f32_dt, u32_dt], cuda, argmin_dim_u32_f32_cuda_wrapper);

    // Cast — one binding-table entry per (src, dst) pair the cast.cu
    // PTX defines. Pairs gated on `__CUDA_ARCH__` in the .cu source
    // still register here unconditionally; if a specific kernel
    // wasn't compiled into the in-process PTX (e.g. BF16 paths on a
    // pre-Ampere GPU) the kernel-load step inside
    // `byte_kernels::cast` surfaces a clear error with the kernel
    // name. FP8 pairs are skipped — DType::F8E4M3 isn't yet
    // exercised through the unified path; trivial follow-up to add.
    const CAST_PAIRS: &[(DType, DType)] = &[
        // Always available (sm_500+).
        (DType::U32, DType::U32), (DType::U32, DType::U8),  (DType::U32, DType::I64),
        (DType::U32, DType::F32), (DType::U32, DType::F64),
        (DType::U8,  DType::U32), (DType::U8,  DType::U8),  (DType::U8,  DType::I64),
        (DType::U8,  DType::F32), (DType::U8,  DType::F64),
        (DType::I64, DType::U32), (DType::I64, DType::U8),  (DType::I64, DType::I64),
        (DType::I64, DType::F32), (DType::I64, DType::F64),
        (DType::F32, DType::U8),  (DType::F32, DType::U32), (DType::F32, DType::I64),
        (DType::F32, DType::F32), (DType::F32, DType::F64),
        (DType::F64, DType::U8),  (DType::F64, DType::U32), (DType::F64, DType::I64),
        (DType::F64, DType::F32), (DType::F64, DType::F64),
        // sm_530+ (F16).
        (DType::F16, DType::F16), (DType::F16, DType::U8),  (DType::F16, DType::U32),
        (DType::F16, DType::F32), (DType::F16, DType::F64),
        (DType::U8,  DType::F16), (DType::U32, DType::F16), (DType::F32, DType::F16),
        (DType::F64, DType::F16),
        // sm_800+ (BF16).
        (DType::BF16, DType::BF16),
        (DType::BF16, DType::U32), (DType::BF16, DType::F32), (DType::BF16, DType::F64),
        (DType::BF16, DType::U8),  (DType::BF16, DType::F16),
        (DType::U8,   DType::BF16), (DType::U32, DType::BF16), (DType::F32, DType::BF16),
        (DType::F64,  DType::BF16), (DType::F16, DType::BF16),
    ];
    for &(src, dst) in CAST_PAIRS {
        table.register(Cast, &[src, dst], cuda, cast_cuda_wrapper);
    }
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
        ReduceMaxTo,
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
        ReduceMaxTo,
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
    // Rope, Conv2D, ConvTranspose2D, ReduceSumTo, ReduceMaxTo,
    // FusedLinear, FlashAttn, PagedAttn) — all use the f32-accumulator
    // pattern.
    for op in [SoftmaxLastDim, RmsNormLastDim, LayerNormLastDim, Rope, Conv2D, ConvTranspose2D, ReduceSumTo, ReduceMaxTo, FusedLinear, FlashAttn, PagedAttn] {
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

/// Phase 7.6 step 6 — register the always-built fused-op kernels into
/// the [`crate::fused::FusedKernelRegistry`]. Called by
/// [`crate::fused::default_kernel_registry`]; kept here so the
/// crate-private CPU dispatch wrappers stay co-located with their
/// registration.
///
/// Today's coverage: `FUSED_LINEAR` × `Cpu` × {F32, F64, BF16, F16}.
/// Backend crates (fuel-cuda-backend, fuel-vulkan-backend) extend by
/// composing against the registry from their own startup paths or via
/// the step-9 binding-table refactor.
pub fn register_default_fused_kernels(r: &mut crate::fused::FusedKernelRegistry) {
    use crate::fused::{cost_fused_linear_cpu, FUSED_LINEAR_CPU_PRECISION};
    use crate::register_fused;
    use fuel_graph::registry::FusedOps;

    // Dtype tuples mirror the binding-table shape:
    //   (lhs, rhs, bias, out) — all four agree per FusedLinear's contract.
    const FL_F32:  &[DType] = &[DType::F32,  DType::F32,  DType::F32,  DType::F32];
    const FL_F64:  &[DType] = &[DType::F64,  DType::F64,  DType::F64,  DType::F64];
    const FL_BF16: &[DType] = &[DType::BF16, DType::BF16, DType::BF16, DType::BF16];
    const FL_F16:  &[DType] = &[DType::F16,  DType::F16,  DType::F16,  DType::F16];

    let cpu = BackendId::Cpu;
    register_fused!(r, FusedOps::FUSED_LINEAR, cpu, FL_F32,
        fused_linear_f32_cpu_wrapper,
        cost = cost_fused_linear_cpu,
        precision = FUSED_LINEAR_CPU_PRECISION);
    register_fused!(r, FusedOps::FUSED_LINEAR, cpu, FL_F64,
        fused_linear_f64_cpu_wrapper,
        cost = cost_fused_linear_cpu,
        precision = FUSED_LINEAR_CPU_PRECISION);
    register_fused!(r, FusedOps::FUSED_LINEAR, cpu, FL_BF16,
        fused_linear_bf16_cpu_wrapper,
        cost = cost_fused_linear_cpu,
        precision = FUSED_LINEAR_CPU_PRECISION);
    register_fused!(r, FusedOps::FUSED_LINEAR, cpu, FL_F16,
        fused_linear_f16_cpu_wrapper,
        cost = cost_fused_linear_cpu,
        precision = FUSED_LINEAR_CPU_PRECISION);
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
            .lookup(OpKind::AddElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Cpu)
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
        let result = table.lookup(OpKind::AddElementwise, &[DType::I64, DType::I64, DType::I64], BackendId::Cpu);
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

        // 4. Look up the kernel for (op, dtypes, backend).
        let kernel = bindings
            .lookup(
                OpKind::AddElementwise,
                &[DType::F32, DType::F32, DType::F32],
                backend,
            )
            .expect("lookup");

        // 5. Allocate input + output Storages.
        let lhs = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let rhs = crate::from_slice_cpu(&[10.0_f32, 20.0, 30.0, 40.0]);
        let out = crate::alloc_cpu_zeroed(DType::F32, 4).expect("alloc");

        let inputs = vec![Arc::new(RwLock::new(lhs)), Arc::new(RwLock::new(rhs))];
        let mut outputs = vec![Arc::new(RwLock::new(out))];

        // 6. Call the dispatch wrapper.
        kernel(&inputs, &mut outputs, &[], &OpParams::None).expect("kernel");

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
            .lookup(OpKind::AddElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Cpu)
            .unwrap();

        // Only one input — should error, not panic.
        let lhs = crate::from_slice_cpu(&[1.0_f32, 2.0]);
        let out = crate::alloc_cpu_zeroed(DType::F32, 2).unwrap();
        let inputs = vec![Arc::new(RwLock::new(lhs))];
        let mut outputs = vec![Arc::new(RwLock::new(out))];

        let result = kernel(&inputs, &mut outputs, &[], &OpParams::None);
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
        let result = b.lookup(OpKind::AddElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Cpu);
        assert!(result.is_ok(), "CPU AddElementwise+F32 should be registered");
    }
}
