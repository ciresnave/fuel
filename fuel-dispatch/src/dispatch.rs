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
//! own. The canonical process-wide instance lives below
//! ([`global_registry`] / [`global_bindings`]).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use fuel_ir::backend::{BackendCapabilities, SubstrateClass, TransferPath};
use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, DeviceLocation, Error, Layout, Result};

use crate::kernel::{KernelBindingTable, KernelRef, OpParams};
#[cfg(feature = "cuda")]
use crate::kernel::KernelCaps;
use fuel_memory::{BackendStorage, Storage};

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
    /// [`Error::NoBackendForOp`](fuel_ir::Error::NoBackendForOp)
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
    /// [`Error::UnsupportedTransfer`](fuel_ir::Error::UnsupportedTransfer)
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
pub fn cpu_input(s: &Storage) -> Result<&fuel_cpu_backend::CpuStorageBytes> {
    match &s.inner {
        BackendStorage::Cpu(c) => Ok(c),
        #[allow(unreachable_patterns)]
        _ => Err(Error::Msg(
            "cpu kernel wrapper called with non-CPU input".to_string(),
        )
        .bt()),
    }
}

/// Helper: extract `&mut CpuStorageBytes` from `&mut Storage`. `pub`
/// (along with [`cpu_input`], [`read_storage`], [`write_storage`]) so
/// external backend crates (fuel-mkl-cpu-backend, fuel-aocl-cpu-backend)
/// can build their own binding-table wrappers without reimplementing
/// the lock + dtype-match shape.
pub fn cpu_output(s: &mut Storage) -> Result<&mut fuel_cpu_backend::CpuStorageBytes> {
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
pub fn read_storage(
    arc: &Arc<RwLock<Storage>>,
) -> Result<std::sync::RwLockReadGuard<'_, Storage>> {
    arc.read()
        .map_err(|_| Error::Msg("kernel wrapper: storage RwLock poisoned (read)".to_string()).bt())
}

pub fn write_storage(
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
        OpParams::Flip { outer_count, dim_size, inner_count, .. } => {
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
                OpParams::CumSum { outer_count, dim_size, inner_count, .. } => {
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

/// Softmax-last-dim backward — 2 inputs (y, g) + 1 output, reuses
/// `OpParams::SoftmaxLastDim` (same outer × last_dim shape contract
/// as the forward).
macro_rules! cpu_softmax_last_dim_backward_wrapper {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "softmax_last_dim_backward wrapper expects 2 inputs (y, g) + 1 output, got {} + {}",
                    inputs.len(), outputs.len(),
                )).bt());
            }
            let (outer, last_dim) = match params {
                OpParams::SoftmaxLastDim { outer_count, last_dim } => (*outer_count, *last_dim),
                other => {
                    return Err(Error::Msg(format!(
                        "softmax_last_dim_backward wrapper expects OpParams::SoftmaxLastDim, got {other:?}",
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

cpu_softmax_last_dim_backward_wrapper!(
    softmax_last_dim_backward_f32_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::softmax_last_dim_backward_f32);
cpu_softmax_last_dim_backward_wrapper!(
    softmax_last_dim_backward_f64_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::softmax_last_dim_backward_f64);
cpu_softmax_last_dim_backward_wrapper!(
    softmax_last_dim_backward_bf16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::softmax_last_dim_backward_bf16);
cpu_softmax_last_dim_backward_wrapper!(
    softmax_last_dim_backward_f16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::softmax_last_dim_backward_f16);

/// LayerNorm / RmsNorm backward share `OpParams::NormLastDim` (with
/// eps). Two inputs (x, g) + 1 output, same outer × last_dim shape.
macro_rules! cpu_norm_last_dim_backward_wrapper {
    ($wrapper:ident, $kernel:path, $op_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 2 inputs (x, g) + 1 output, got {} + {}",
                    $op_name, inputs.len(), outputs.len(),
                )).bt());
            }
            let (outer, last_dim, eps) = match params {
                OpParams::NormLastDim { outer_count, last_dim, eps } => {
                    (*outer_count, *last_dim, *eps)
                }
                other => {
                    return Err(Error::Msg(format!(
                        "{} wrapper expects OpParams::NormLastDim, got {other:?}",
                        $op_name,
                    )).bt());
                }
            };
            let x_guard = read_storage(&inputs[0])?;
            let g_guard = read_storage(&inputs[1])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let x_cpu = cpu_input(&x_guard)?;
            let g_cpu = cpu_input(&g_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(x_cpu, g_cpu, out_cpu, outer, last_dim, eps)
        }
    };
}

cpu_norm_last_dim_backward_wrapper!(
    layer_norm_last_dim_backward_f32_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::layer_norm_last_dim_backward_f32, "layer_norm_backward_f32");
cpu_norm_last_dim_backward_wrapper!(
    layer_norm_last_dim_backward_f64_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::layer_norm_last_dim_backward_f64, "layer_norm_backward_f64");
cpu_norm_last_dim_backward_wrapper!(
    layer_norm_last_dim_backward_bf16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::layer_norm_last_dim_backward_bf16, "layer_norm_backward_bf16");
cpu_norm_last_dim_backward_wrapper!(
    layer_norm_last_dim_backward_f16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::layer_norm_last_dim_backward_f16, "layer_norm_backward_f16");

cpu_norm_last_dim_backward_wrapper!(
    rms_norm_last_dim_backward_f32_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::rms_norm_last_dim_backward_f32, "rms_norm_backward_f32");
cpu_norm_last_dim_backward_wrapper!(
    rms_norm_last_dim_backward_f64_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::rms_norm_last_dim_backward_f64, "rms_norm_backward_f64");
cpu_norm_last_dim_backward_wrapper!(
    rms_norm_last_dim_backward_bf16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::rms_norm_last_dim_backward_bf16, "rms_norm_backward_bf16");
cpu_norm_last_dim_backward_wrapper!(
    rms_norm_last_dim_backward_f16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::rms_norm_last_dim_backward_f16, "rms_norm_backward_f16");

/// ReduceMaxTo backward — 2 inputs (x, upstream) + 1 output. Carries
/// the shape pair via `OpParams::ReduceMaxToBackward`.
macro_rules! cpu_reduce_max_to_backward_wrapper {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "reduce_max_to_backward wrapper expects 2 inputs (x, upstream) + 1 output, got {} + {}",
                    inputs.len(), outputs.len(),
                )).bt());
            }
            let (input_shape, output_shape) = match params {
                OpParams::ReduceMaxToBackward { input_shape, output_shape } => {
                    (input_shape.as_slice(), output_shape.as_slice())
                }
                other => {
                    return Err(Error::Msg(format!(
                        "reduce_max_to_backward wrapper expects OpParams::ReduceMaxToBackward, got {other:?}",
                    )).bt());
                }
            };
            let x_guard = read_storage(&inputs[0])?;
            let up_guard = read_storage(&inputs[1])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let x_cpu = cpu_input(&x_guard)?;
            let up_cpu = cpu_input(&up_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(x_cpu, up_cpu, out_cpu, input_shape, output_shape)
        }
    };
}

cpu_reduce_max_to_backward_wrapper!(
    reduce_max_to_backward_f32_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::reduce_max_to_backward_f32);
cpu_reduce_max_to_backward_wrapper!(
    reduce_max_to_backward_f64_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::reduce_max_to_backward_f64);
cpu_reduce_max_to_backward_wrapper!(
    reduce_max_to_backward_bf16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::reduce_max_to_backward_bf16);
cpu_reduce_max_to_backward_wrapper!(
    reduce_max_to_backward_f16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::reduce_max_to_backward_f16);

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
        OpParams::Roll { outer_count, dim_size, inner_count, shift, .. } => {
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
fn write_slice_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "write_slice wrapper expects 1 input (source) + 1 output (dest), \
             got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let (dest_shape, ranges) = match params {
        OpParams::WriteSlice { dest_shape, ranges } => (dest_shape, ranges),
        other => {
            return Err(Error::Msg(format!(
                "write_slice wrapper expects OpParams::WriteSlice, got {other:?}",
            ))
            .bt())
        }
    };
    let src_guard = read_storage(&inputs[0])?;
    let src_cpu = cpu_input(&src_guard)?;
    let mut dest_guard = write_storage(&outputs[0])?;
    let dtype_size = dest_guard.dtype.size_in_bytes();
    let dest_cpu = cpu_output(&mut dest_guard)?;
    fuel_cpu_backend::byte_kernels::write_slice_cpu(
        src_cpu, dest_cpu, dest_shape, ranges, dtype_size,
    )
}

/// Dispatch wrapper for `(WriteSliceRotating, *, Cpu)`. Same shape
/// as `write_slice_cpu_wrapper` plus a second input carrying the
/// dynamic position scalar.
fn write_slice_rotating_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "write_slice_rotating wrapper expects 2 inputs (source, position) + 1 output (dest), \
             got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let (dest_shape, axis, modulus, ranges) = match params {
        OpParams::WriteSliceRotating { dest_shape, axis, modulus, ranges } => {
            (dest_shape, *axis, *modulus, ranges)
        }
        other => {
            return Err(Error::Msg(format!(
                "write_slice_rotating wrapper expects OpParams::WriteSliceRotating, got {other:?}",
            ))
            .bt())
        }
    };
    let src_guard = read_storage(&inputs[0])?;
    let src_cpu = cpu_input(&src_guard)?;
    let pos_guard = read_storage(&inputs[1])?;
    let pos_cpu = cpu_input(&pos_guard)?;
    let mut dest_guard = write_storage(&outputs[0])?;
    let dtype_size = dest_guard.dtype.size_in_bytes();
    let dest_cpu = cpu_output(&mut dest_guard)?;
    fuel_cpu_backend::byte_kernels::write_slice_rotating_cpu(
        src_cpu, pos_cpu, dest_cpu, dest_shape, axis, modulus, ranges, dtype_size,
    )
}

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
        OpParams::Concat { outer_count, input_dim_sizes, inner_count, .. } => {
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

/// Dispatch wrapper macro for in-place affine on CPU. Inputs is
/// empty (the executor's `WorkItemKind::InplaceKernel` arm passes the
/// target Arc as `outputs[0]`); the kernel reads + writes through the
/// single write lock.
macro_rules! cpu_affine_inplace_wrapper {
    ($name:ident, $kernel:ident, $T:ty) => {
        fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if !inputs.is_empty() || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    concat!(stringify!($name),
                        ": expected 0 inputs + 1 output (target adopted by executor), got {} + {}"),
                    inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let (mul, add) = match params {
                OpParams::Affine { mul, add } => (*mul as $T, *add as $T),
                other => {
                    return Err(Error::Msg(format!(
                        concat!(stringify!($name), ": expected OpParams::Affine, got {:?}"),
                        other,
                    ))
                    .bt())
                }
            };
            let mut out_guard = write_storage(&outputs[0])?;
            let out_cpu = cpu_output(&mut out_guard)?;
            fuel_cpu_backend::byte_kernels::$kernel(out_cpu, mul, add)
        }
    };
    ($name:ident, $kernel:ident, $T:ty, half) => {
        fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if !inputs.is_empty() || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    concat!(stringify!($name),
                        ": expected 0 inputs + 1 output (target adopted by executor), got {} + {}"),
                    inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let (mul, add) = match params {
                OpParams::Affine { mul, add } => (*mul, *add),
                other => {
                    return Err(Error::Msg(format!(
                        concat!(stringify!($name), ": expected OpParams::Affine, got {:?}"),
                        other,
                    ))
                    .bt())
                }
            };
            let mut out_guard = write_storage(&outputs[0])?;
            let out_cpu = cpu_output(&mut out_guard)?;
            fuel_cpu_backend::byte_kernels::$kernel(out_cpu, mul, add)
        }
    };
}

cpu_affine_inplace_wrapper!(inplace_affine_f32_cpu_wrapper, affine_inplace_f32, f32);
cpu_affine_inplace_wrapper!(inplace_affine_f64_cpu_wrapper, affine_inplace_f64, f64);
cpu_affine_inplace_wrapper!(inplace_affine_bf16_cpu_wrapper, affine_inplace_bf16, f64, half);
cpu_affine_inplace_wrapper!(inplace_affine_f16_cpu_wrapper,  affine_inplace_f16,  f64, half);

/// Dispatch wrapper macro for in-place clamp on CPU. Same shape as
/// `cpu_affine_inplace_wrapper!` (no inputs, target as outputs[0]),
/// but pulls `(min, max)` from `OpParams::Clamp`.
macro_rules! cpu_clamp_inplace_wrapper {
    ($name:ident, $kernel:ident, $T:ty) => {
        fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if !inputs.is_empty() || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    concat!(stringify!($name),
                        ": expected 0 inputs + 1 output (target adopted by executor), got {} + {}"),
                    inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let (min, max) = match params {
                OpParams::Clamp { min, max } => (*min as $T, *max as $T),
                other => {
                    return Err(Error::Msg(format!(
                        concat!(stringify!($name), ": expected OpParams::Clamp, got {:?}"),
                        other,
                    ))
                    .bt())
                }
            };
            let mut out_guard = write_storage(&outputs[0])?;
            let out_cpu = cpu_output(&mut out_guard)?;
            fuel_cpu_backend::byte_kernels::$kernel(out_cpu, min, max)
        }
    };
    ($name:ident, $kernel:ident, $T:ty, half) => {
        fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if !inputs.is_empty() || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    concat!(stringify!($name),
                        ": expected 0 inputs + 1 output (target adopted by executor), got {} + {}"),
                    inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let (min, max) = match params {
                OpParams::Clamp { min, max } => (*min, *max),
                other => {
                    return Err(Error::Msg(format!(
                        concat!(stringify!($name), ": expected OpParams::Clamp, got {:?}"),
                        other,
                    ))
                    .bt())
                }
            };
            let mut out_guard = write_storage(&outputs[0])?;
            let out_cpu = cpu_output(&mut out_guard)?;
            fuel_cpu_backend::byte_kernels::$kernel(out_cpu, min, max)
        }
    };
}

cpu_clamp_inplace_wrapper!(clamp_inplace_f32_cpu_wrapper,  clamp_inplace_f32,  f32);
cpu_clamp_inplace_wrapper!(clamp_inplace_f64_cpu_wrapper,  clamp_inplace_f64,  f64);
cpu_clamp_inplace_wrapper!(clamp_inplace_bf16_cpu_wrapper, clamp_inplace_bf16, f64, half);
cpu_clamp_inplace_wrapper!(clamp_inplace_f16_cpu_wrapper,  clamp_inplace_f16,  f64, half);

/// Dispatch wrapper macro for in-place powi on CPU. Pulls `exp` from
/// `OpParams::PowI`.
macro_rules! cpu_powi_inplace_wrapper {
    ($name:ident, $kernel:ident) => {
        fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if !inputs.is_empty() || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    concat!(stringify!($name),
                        ": expected 0 inputs + 1 output (target adopted by executor), got {} + {}"),
                    inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let exp = match params {
                OpParams::PowI { exp } => *exp,
                other => {
                    return Err(Error::Msg(format!(
                        concat!(stringify!($name), ": expected OpParams::PowI, got {:?}"),
                        other,
                    ))
                    .bt())
                }
            };
            let mut out_guard = write_storage(&outputs[0])?;
            let out_cpu = cpu_output(&mut out_guard)?;
            fuel_cpu_backend::byte_kernels::$kernel(out_cpu, exp)
        }
    };
}

cpu_powi_inplace_wrapper!(powi_inplace_f32_cpu_wrapper,  powi_inplace_f32);
cpu_powi_inplace_wrapper!(powi_inplace_f64_cpu_wrapper,  powi_inplace_f64);
cpu_powi_inplace_wrapper!(powi_inplace_bf16_cpu_wrapper, powi_inplace_bf16);
cpu_powi_inplace_wrapper!(powi_inplace_f16_cpu_wrapper,  powi_inplace_f16);

/// Dispatch wrapper macro for in-place elementwise unary ops on CPU.
/// Same shape as `cpu_affine_inplace_wrapper!` (inputs empty, target
/// adopted as outputs[0] by the executor), but the kernel takes no
/// scalar params — just the target buffer.
macro_rules! cpu_unary_inplace_wrapper {
    ($name:ident, $kernel:ident) => {
        fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            _params: &OpParams,
        ) -> Result<()> {
            if !inputs.is_empty() || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    concat!(stringify!($name),
                        ": expected 0 inputs + 1 output (target adopted by executor), got {} + {}"),
                    inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let mut out_guard = write_storage(&outputs[0])?;
            let out_cpu = cpu_output(&mut out_guard)?;
            fuel_cpu_backend::byte_kernels::$kernel(out_cpu)
        }
    };
}

cpu_unary_inplace_wrapper!(relu_inplace_f32_cpu_wrapper,    relu_inplace_f32);
cpu_unary_inplace_wrapper!(silu_inplace_f32_cpu_wrapper,    silu_inplace_f32);
cpu_unary_inplace_wrapper!(gelu_inplace_f32_cpu_wrapper,    gelu_inplace_f32);
cpu_unary_inplace_wrapper!(tanh_inplace_f32_cpu_wrapper,    tanh_inplace_f32);
cpu_unary_inplace_wrapper!(sigmoid_inplace_f32_cpu_wrapper, sigmoid_inplace_f32);

cpu_unary_inplace_wrapper!(relu_inplace_f64_cpu_wrapper,    relu_inplace_f64);
cpu_unary_inplace_wrapper!(silu_inplace_f64_cpu_wrapper,    silu_inplace_f64);
cpu_unary_inplace_wrapper!(gelu_inplace_f64_cpu_wrapper,    gelu_inplace_f64);
cpu_unary_inplace_wrapper!(tanh_inplace_f64_cpu_wrapper,    tanh_inplace_f64);
cpu_unary_inplace_wrapper!(sigmoid_inplace_f64_cpu_wrapper, sigmoid_inplace_f64);

cpu_unary_inplace_wrapper!(relu_inplace_bf16_cpu_wrapper,    relu_inplace_bf16);
cpu_unary_inplace_wrapper!(silu_inplace_bf16_cpu_wrapper,    silu_inplace_bf16);
cpu_unary_inplace_wrapper!(gelu_inplace_bf16_cpu_wrapper,    gelu_inplace_bf16);
cpu_unary_inplace_wrapper!(tanh_inplace_bf16_cpu_wrapper,    tanh_inplace_bf16);
cpu_unary_inplace_wrapper!(sigmoid_inplace_bf16_cpu_wrapper, sigmoid_inplace_bf16);

cpu_unary_inplace_wrapper!(relu_inplace_f16_cpu_wrapper,    relu_inplace_f16);
cpu_unary_inplace_wrapper!(silu_inplace_f16_cpu_wrapper,    silu_inplace_f16);
cpu_unary_inplace_wrapper!(gelu_inplace_f16_cpu_wrapper,    gelu_inplace_f16);
cpu_unary_inplace_wrapper!(tanh_inplace_f16_cpu_wrapper,    tanh_inplace_f16);
cpu_unary_inplace_wrapper!(sigmoid_inplace_f16_cpu_wrapper, sigmoid_inplace_f16);

// In-place unary op family expansion (2026-05-30) — 16 new ops × 4
// dtypes. Each wrapper is identical in shape to the original 5-op
// starter set; the chassis handles per-dtype math.
cpu_unary_inplace_wrapper!(neg_inplace_f32_cpu_wrapper,    neg_inplace_f32);
cpu_unary_inplace_wrapper!(neg_inplace_f64_cpu_wrapper,    neg_inplace_f64);
cpu_unary_inplace_wrapper!(neg_inplace_bf16_cpu_wrapper,   neg_inplace_bf16);
cpu_unary_inplace_wrapper!(neg_inplace_f16_cpu_wrapper,    neg_inplace_f16);

cpu_unary_inplace_wrapper!(abs_inplace_f32_cpu_wrapper,    abs_inplace_f32);
cpu_unary_inplace_wrapper!(abs_inplace_f64_cpu_wrapper,    abs_inplace_f64);
cpu_unary_inplace_wrapper!(abs_inplace_bf16_cpu_wrapper,   abs_inplace_bf16);
cpu_unary_inplace_wrapper!(abs_inplace_f16_cpu_wrapper,    abs_inplace_f16);

cpu_unary_inplace_wrapper!(sqr_inplace_f32_cpu_wrapper,    sqr_inplace_f32);
cpu_unary_inplace_wrapper!(sqr_inplace_f64_cpu_wrapper,    sqr_inplace_f64);
cpu_unary_inplace_wrapper!(sqr_inplace_bf16_cpu_wrapper,   sqr_inplace_bf16);
cpu_unary_inplace_wrapper!(sqr_inplace_f16_cpu_wrapper,    sqr_inplace_f16);

cpu_unary_inplace_wrapper!(sqrt_inplace_f32_cpu_wrapper,   sqrt_inplace_f32);
cpu_unary_inplace_wrapper!(sqrt_inplace_f64_cpu_wrapper,   sqrt_inplace_f64);
cpu_unary_inplace_wrapper!(sqrt_inplace_bf16_cpu_wrapper,  sqrt_inplace_bf16);
cpu_unary_inplace_wrapper!(sqrt_inplace_f16_cpu_wrapper,   sqrt_inplace_f16);

cpu_unary_inplace_wrapper!(rsqrt_inplace_f32_cpu_wrapper,  rsqrt_inplace_f32);
cpu_unary_inplace_wrapper!(rsqrt_inplace_f64_cpu_wrapper,  rsqrt_inplace_f64);
cpu_unary_inplace_wrapper!(rsqrt_inplace_bf16_cpu_wrapper, rsqrt_inplace_bf16);
cpu_unary_inplace_wrapper!(rsqrt_inplace_f16_cpu_wrapper,  rsqrt_inplace_f16);

cpu_unary_inplace_wrapper!(recip_inplace_f32_cpu_wrapper,  recip_inplace_f32);
cpu_unary_inplace_wrapper!(recip_inplace_f64_cpu_wrapper,  recip_inplace_f64);
cpu_unary_inplace_wrapper!(recip_inplace_bf16_cpu_wrapper, recip_inplace_bf16);
cpu_unary_inplace_wrapper!(recip_inplace_f16_cpu_wrapper,  recip_inplace_f16);

cpu_unary_inplace_wrapper!(exp_inplace_f32_cpu_wrapper,    exp_inplace_f32);
cpu_unary_inplace_wrapper!(exp_inplace_f64_cpu_wrapper,    exp_inplace_f64);
cpu_unary_inplace_wrapper!(exp_inplace_bf16_cpu_wrapper,   exp_inplace_bf16);
cpu_unary_inplace_wrapper!(exp_inplace_f16_cpu_wrapper,    exp_inplace_f16);

cpu_unary_inplace_wrapper!(log_inplace_f32_cpu_wrapper,    log_inplace_f32);
cpu_unary_inplace_wrapper!(log_inplace_f64_cpu_wrapper,    log_inplace_f64);
cpu_unary_inplace_wrapper!(log_inplace_bf16_cpu_wrapper,   log_inplace_bf16);
cpu_unary_inplace_wrapper!(log_inplace_f16_cpu_wrapper,    log_inplace_f16);

cpu_unary_inplace_wrapper!(sin_inplace_f32_cpu_wrapper,    sin_inplace_f32);
cpu_unary_inplace_wrapper!(sin_inplace_f64_cpu_wrapper,    sin_inplace_f64);
cpu_unary_inplace_wrapper!(sin_inplace_bf16_cpu_wrapper,   sin_inplace_bf16);
cpu_unary_inplace_wrapper!(sin_inplace_f16_cpu_wrapper,    sin_inplace_f16);

cpu_unary_inplace_wrapper!(cos_inplace_f32_cpu_wrapper,    cos_inplace_f32);
cpu_unary_inplace_wrapper!(cos_inplace_f64_cpu_wrapper,    cos_inplace_f64);
cpu_unary_inplace_wrapper!(cos_inplace_bf16_cpu_wrapper,   cos_inplace_bf16);
cpu_unary_inplace_wrapper!(cos_inplace_f16_cpu_wrapper,    cos_inplace_f16);

cpu_unary_inplace_wrapper!(sign_inplace_f32_cpu_wrapper,   sign_inplace_f32);
cpu_unary_inplace_wrapper!(sign_inplace_f64_cpu_wrapper,   sign_inplace_f64);
cpu_unary_inplace_wrapper!(sign_inplace_bf16_cpu_wrapper,  sign_inplace_bf16);
cpu_unary_inplace_wrapper!(sign_inplace_f16_cpu_wrapper,   sign_inplace_f16);

cpu_unary_inplace_wrapper!(floor_inplace_f32_cpu_wrapper,  floor_inplace_f32);
cpu_unary_inplace_wrapper!(floor_inplace_f64_cpu_wrapper,  floor_inplace_f64);
cpu_unary_inplace_wrapper!(floor_inplace_bf16_cpu_wrapper, floor_inplace_bf16);
cpu_unary_inplace_wrapper!(floor_inplace_f16_cpu_wrapper,  floor_inplace_f16);

cpu_unary_inplace_wrapper!(ceil_inplace_f32_cpu_wrapper,   ceil_inplace_f32);
cpu_unary_inplace_wrapper!(ceil_inplace_f64_cpu_wrapper,   ceil_inplace_f64);
cpu_unary_inplace_wrapper!(ceil_inplace_bf16_cpu_wrapper,  ceil_inplace_bf16);
cpu_unary_inplace_wrapper!(ceil_inplace_f16_cpu_wrapper,   ceil_inplace_f16);

cpu_unary_inplace_wrapper!(round_inplace_f32_cpu_wrapper,  round_inplace_f32);
cpu_unary_inplace_wrapper!(round_inplace_f64_cpu_wrapper,  round_inplace_f64);
cpu_unary_inplace_wrapper!(round_inplace_bf16_cpu_wrapper, round_inplace_bf16);
cpu_unary_inplace_wrapper!(round_inplace_f16_cpu_wrapper,  round_inplace_f16);

cpu_unary_inplace_wrapper!(erf_inplace_f32_cpu_wrapper,    erf_inplace_f32);
cpu_unary_inplace_wrapper!(erf_inplace_f64_cpu_wrapper,    erf_inplace_f64);
cpu_unary_inplace_wrapper!(erf_inplace_bf16_cpu_wrapper,   erf_inplace_bf16);
cpu_unary_inplace_wrapper!(erf_inplace_f16_cpu_wrapper,    erf_inplace_f16);

cpu_unary_inplace_wrapper!(gelu_erf_inplace_f32_cpu_wrapper,  gelu_erf_inplace_f32);
cpu_unary_inplace_wrapper!(gelu_erf_inplace_f64_cpu_wrapper,  gelu_erf_inplace_f64);
cpu_unary_inplace_wrapper!(gelu_erf_inplace_bf16_cpu_wrapper, gelu_erf_inplace_bf16);
cpu_unary_inplace_wrapper!(gelu_erf_inplace_f16_cpu_wrapper,  gelu_erf_inplace_f16);

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

/// Build a `(2 inputs `(x, upstream)`, 1 output, OpParams::PowI)` CPU
/// dispatch wrapper for the per-dtype PowIBackward kernels.
macro_rules! cpu_powi_backward_wrapper {
    ($wrapper:ident, $kernel:path, $op_name:literal) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "{} wrapper expects 2 inputs + 1 output, got {} + {}",
                    $op_name, inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let exp = match params {
                OpParams::PowI { exp } => *exp,
                other => {
                    return Err(Error::Msg(format!(
                        "{} wrapper expects OpParams::PowI, got {other:?}",
                        $op_name,
                    ))
                    .bt())
                }
            };
            let x_guard = read_storage(&inputs[0])?;
            let up_guard = read_storage(&inputs[1])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let x_cpu = cpu_input(&x_guard)?;
            let up_cpu = cpu_input(&up_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(x_cpu, up_cpu, out_cpu, exp)
        }
    };
}

cpu_powi_backward_wrapper!(
    powi_backward_f32_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::powi_backward_f32,
    "powi_backward_f32"
);
cpu_powi_backward_wrapper!(
    powi_backward_f64_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::powi_backward_f64,
    "powi_backward_f64"
);
cpu_powi_backward_wrapper!(
    powi_backward_bf16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::powi_backward_bf16,
    "powi_backward_bf16"
);
cpu_powi_backward_wrapper!(
    powi_backward_f16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::powi_backward_f16,
    "powi_backward_f16"
);

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

/// Per-dtype FusedSoftmaxCrossEntropy dispatch wrapper. Two inputs
/// (logits T, targets I64) → one F32 output (the FSCE declared dtype,
/// regardless of logits dtype — losses accumulate in f64, narrow to
/// f32). Translates the `Reduction` enum from the FusedOpParams
/// payload (which lives on the graph node) to the kernel's `u8` tag —
/// the kernel intentionally stays free of fuel-graph dependencies.
macro_rules! cpu_fused_softmax_cross_entropy_wrapper {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 {
                return Err(Error::Msg(format!(
                    "fused_softmax_cross_entropy wrapper expects 2 inputs (logits, targets), got {}",
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "fused_softmax_cross_entropy wrapper expects 1 output, got {}",
                    outputs.len(),
                ))
                .bt());
            }
            let (n_rows, vocab, reduction, ignore_index) = match params {
                OpParams::FusedSoftmaxCrossEntropy {
                    n_rows, vocab, reduction, ignore_index,
                } => (*n_rows, *vocab, *reduction, *ignore_index),
                other => {
                    return Err(Error::Msg(format!(
                        "fused_softmax_cross_entropy wrapper expects \
                         OpParams::FusedSoftmaxCrossEntropy, got {other:?}",
                    ))
                    .bt());
                }
            };
            let reduction_tag = match reduction {
                fuel_graph::registry::Reduction::Mean => fuel_cpu_backend::byte_kernels::REDUCTION_MEAN,
                fuel_graph::registry::Reduction::Sum  => fuel_cpu_backend::byte_kernels::REDUCTION_SUM,
                fuel_graph::registry::Reduction::None => fuel_cpu_backend::byte_kernels::REDUCTION_NONE,
            };
            let logits_guard = read_storage(&inputs[0])?;
            let targets_guard = read_storage(&inputs[1])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let logits_cpu = cpu_input(&logits_guard)?;
            let targets_cpu = cpu_input(&targets_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(
                logits_cpu, targets_cpu, out_cpu,
                n_rows, vocab, reduction_tag, ignore_index,
            )
        }
    };
}

cpu_fused_softmax_cross_entropy_wrapper!(
    fused_softmax_cross_entropy_f32_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::fused_softmax_cross_entropy_f32
);
cpu_fused_softmax_cross_entropy_wrapper!(
    fused_softmax_cross_entropy_f64_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::fused_softmax_cross_entropy_f64
);
cpu_fused_softmax_cross_entropy_wrapper!(
    fused_softmax_cross_entropy_bf16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::fused_softmax_cross_entropy_bf16
);
cpu_fused_softmax_cross_entropy_wrapper!(
    fused_softmax_cross_entropy_f16_cpu_wrapper,
    fuel_cpu_backend::byte_kernels::fused_softmax_cross_entropy_f16
);

/// Per-dtype CausalConv1d dispatch wrapper. Three inputs (x, weight,
/// bias) → one output. Geometry + `use_silu` flow through
/// `OpParams::CausalConv1d`. All four inputs/outputs share the same
/// dtype `T`; the binding-table key is `[T, T, T, T]`.
macro_rules! cpu_causal_conv1d_wrapper {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "causal_conv1d wrapper expects 3 inputs (x, weight, bias), got {}",
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "causal_conv1d wrapper expects 1 output, got {}", outputs.len(),
                ))
                .bt());
            }
            let (batch, channels, seq_in, seq_out, kernel, use_silu) = match params {
                OpParams::CausalConv1d {
                    batch, channels, seq_in, seq_out, kernel, use_silu,
                } => (*batch, *channels, *seq_in, *seq_out, *kernel, *use_silu),
                other => {
                    return Err(Error::Msg(format!(
                        "causal_conv1d wrapper expects OpParams::CausalConv1d, got {other:?}",
                    ))
                    .bt());
                }
            };
            let x_guard = read_storage(&inputs[0])?;
            let w_guard = read_storage(&inputs[1])?;
            let bias_guard = read_storage(&inputs[2])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let x_cpu = cpu_input(&x_guard)?;
            let w_cpu = cpu_input(&w_guard)?;
            let bias_cpu = cpu_input(&bias_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(
                x_cpu, w_cpu, bias_cpu, out_cpu,
                batch, channels, seq_in, seq_out, kernel, use_silu,
            )
        }
    };
}

cpu_causal_conv1d_wrapper!(causal_conv1d_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::causal_conv1d_f32);
cpu_causal_conv1d_wrapper!(causal_conv1d_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::causal_conv1d_f64);
cpu_causal_conv1d_wrapper!(causal_conv1d_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::causal_conv1d_bf16);
cpu_causal_conv1d_wrapper!(causal_conv1d_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::causal_conv1d_f16);

/// Per-dtype SelectiveScan dispatch wrapper. Five inputs (u, delta,
/// a, b, c) → one output (y). Geometry + `delta_softplus` flow
/// through `OpParams::SelectiveScan`. All six tensors share dtype `T`;
/// the binding-table key is `[T; 6]`.
macro_rules! cpu_selective_scan_wrapper {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 5 {
                return Err(Error::Msg(format!(
                    "selective_scan wrapper expects 5 inputs (u, delta, a, b, c), got {}",
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "selective_scan wrapper expects 1 output, got {}", outputs.len(),
                ))
                .bt());
            }
            let (batch, seqlen, dim, dstate, delta_softplus) = match params {
                OpParams::SelectiveScan {
                    batch, seqlen, dim, dstate, delta_softplus,
                } => (*batch, *seqlen, *dim, *dstate, *delta_softplus),
                other => {
                    return Err(Error::Msg(format!(
                        "selective_scan wrapper expects OpParams::SelectiveScan, got {other:?}",
                    ))
                    .bt());
                }
            };
            let u_guard = read_storage(&inputs[0])?;
            let delta_guard = read_storage(&inputs[1])?;
            let a_guard = read_storage(&inputs[2])?;
            let b_guard = read_storage(&inputs[3])?;
            let c_guard = read_storage(&inputs[4])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let u_cpu = cpu_input(&u_guard)?;
            let delta_cpu = cpu_input(&delta_guard)?;
            let a_cpu = cpu_input(&a_guard)?;
            let b_cpu = cpu_input(&b_guard)?;
            let c_cpu = cpu_input(&c_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(
                u_cpu, delta_cpu, a_cpu, b_cpu, c_cpu, out_cpu,
                batch, seqlen, dim, dstate, delta_softplus,
            )
        }
    };
}

cpu_selective_scan_wrapper!(selective_scan_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::selective_scan_f32);
cpu_selective_scan_wrapper!(selective_scan_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::selective_scan_f64);
cpu_selective_scan_wrapper!(selective_scan_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::selective_scan_bf16);
cpu_selective_scan_wrapper!(selective_scan_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::selective_scan_f16);

/// Per-dtype SsdChunkScan dispatch wrapper. Five inputs (x, dt, a,
/// b, c) → one output (y). Geometry + `chunk_size` flow through
/// `OpParams::SsdChunkScan`. All six tensors share dtype `T`; the
/// binding-table key is `[T; 6]`.
macro_rules! cpu_ssd_chunk_scan_wrapper {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 5 {
                return Err(Error::Msg(format!(
                    "ssd_chunk_scan wrapper expects 5 inputs (x, dt, a, b, c), got {}",
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "ssd_chunk_scan wrapper expects 1 output, got {}", outputs.len(),
                ))
                .bt());
            }
            let (batch, seqlen, heads, head_dim, state_dim, chunk_size) = match params {
                OpParams::SsdChunkScan {
                    batch, seqlen, heads, head_dim, state_dim, chunk_size,
                } => (*batch, *seqlen, *heads, *head_dim, *state_dim, *chunk_size),
                other => {
                    return Err(Error::Msg(format!(
                        "ssd_chunk_scan wrapper expects OpParams::SsdChunkScan, got {other:?}",
                    ))
                    .bt());
                }
            };
            let x_guard = read_storage(&inputs[0])?;
            let dt_guard = read_storage(&inputs[1])?;
            let a_guard = read_storage(&inputs[2])?;
            let b_guard = read_storage(&inputs[3])?;
            let c_guard = read_storage(&inputs[4])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let x_cpu = cpu_input(&x_guard)?;
            let dt_cpu = cpu_input(&dt_guard)?;
            let a_cpu = cpu_input(&a_guard)?;
            let b_cpu = cpu_input(&b_guard)?;
            let c_cpu = cpu_input(&c_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(
                x_cpu, dt_cpu, a_cpu, b_cpu, c_cpu, out_cpu,
                batch, seqlen, heads, head_dim, state_dim, chunk_size,
            )
        }
    };
}

cpu_ssd_chunk_scan_wrapper!(ssd_chunk_scan_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::ssd_chunk_scan_f32);
cpu_ssd_chunk_scan_wrapper!(ssd_chunk_scan_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::ssd_chunk_scan_f64);
cpu_ssd_chunk_scan_wrapper!(ssd_chunk_scan_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::ssd_chunk_scan_bf16);
cpu_ssd_chunk_scan_wrapper!(ssd_chunk_scan_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::ssd_chunk_scan_f16);

/// Per-dtype Nf4Matmul wrapper. Three inputs (activations, w_packed
/// U8, absmax F32) → one output of the activations' dtype.
/// `block_size` flows through `OpParams::Nf4Matmul`.
macro_rules! cpu_nf4_matmul_wrapper {
    ($wrapper:ident, $kernel:path) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "nf4_matmul wrapper expects 3 inputs (activations, w_packed, absmax), got {}",
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "nf4_matmul wrapper expects 1 output, got {}", outputs.len(),
                ))
                .bt());
            }
            let (batch, m, n, k, block_size) = match params {
                OpParams::Nf4Matmul { batch, m, n, k, block_size } => {
                    (*batch, *m, *n, *k, *block_size)
                }
                other => {
                    return Err(Error::Msg(format!(
                        "nf4_matmul wrapper expects OpParams::Nf4Matmul, got {other:?}",
                    ))
                    .bt());
                }
            };
            let a_guard = read_storage(&inputs[0])?;
            let w_guard = read_storage(&inputs[1])?;
            let abs_guard = read_storage(&inputs[2])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a_cpu = cpu_input(&a_guard)?;
            let w_cpu = cpu_input(&w_guard)?;
            let abs_cpu = cpu_input(&abs_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(a_cpu, w_cpu, abs_cpu, out_cpu, batch, m, n, k, block_size)
        }
    };
}

cpu_nf4_matmul_wrapper!(nf4_matmul_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::nf4_matmul_f32);
cpu_nf4_matmul_wrapper!(nf4_matmul_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::nf4_matmul_f16);
cpu_nf4_matmul_wrapper!(nf4_matmul_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::nf4_matmul_bf16);

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
            let (b, hq, hkv, sq, sk, d, k_len, scale, causal, wl, wr, softcap) = match params {
                OpParams::FlashAttn {
                    b, hq, hkv, sq, sk, d, k_len,
                    softmax_scale, causal,
                    window_size_left, window_size_right, softcap,
                } => (
                    *b, *hq, *hkv, *sq, *sk, *d, *k_len,
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
                b, hq, hkv, sq, sk, d, k_len,
                scale, causal, wl, wr, softcap,
            )
        }
    };
}

cpu_flash_attn_wrapper!(flash_attn_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_f32);
cpu_flash_attn_wrapper!(flash_attn_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_f64);
cpu_flash_attn_wrapper!(flash_attn_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::flash_attn_bf16);
cpu_flash_attn_wrapper!(flash_attn_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_f16);

/// Dispatch wrapper for `(FlashAttnBackward{Q,K,V}, *, Cpu)`. Four or
/// five inputs `(q, k, v, do, [alibi])`; one output (one of dQ/dK/dV).
/// The CPU kernel always computes all three gradients; this wrapper
/// only persists the requested one.
macro_rules! cpu_flash_attn_backward_wrapper {
    ($wrapper:ident, $kernel:path, $which:expr) => {
        fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 4 && inputs.len() != 5 {
                return Err(Error::Msg(format!(
                    "flash_attn_backward wrapper expects 4 or 5 inputs (q, k, v, do, [alibi]), got {}",
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "flash_attn_backward wrapper expects 1 output, got {}",
                    outputs.len(),
                ))
                .bt());
            }
            let (b, hq, hkv, sq, sk, d, scale, causal, wl, wr, softcap) = match params {
                OpParams::FlashAttn {
                    b, hq, hkv, sq, sk, d,
                    softmax_scale, causal,
                    window_size_left, window_size_right, softcap,
                    // Backward is the static (full-K) training path; the
                    // recompute attends the full K extent. k_len ignored.
                    k_len: _,
                } => (
                    *b, *hq, *hkv, *sq, *sk, *d,
                    *softmax_scale, *causal,
                    *window_size_left, *window_size_right, *softcap,
                ),
                other => {
                    return Err(Error::Msg(format!(
                        "flash_attn_backward wrapper expects OpParams::FlashAttn, got {other:?}",
                    ))
                    .bt())
                }
            };
            let q_guard = read_storage(&inputs[0])?;
            let k_guard = read_storage(&inputs[1])?;
            let v_guard = read_storage(&inputs[2])?;
            let do_guard = read_storage(&inputs[3])?;
            let alibi_guard = match inputs.get(4) {
                Some(arc) => Some(read_storage(arc)?),
                None => None,
            };
            let mut out_guard = write_storage(&outputs[0])?;
            let q_cpu = cpu_input(&q_guard)?;
            let k_cpu = cpu_input(&k_guard)?;
            let v_cpu = cpu_input(&v_guard)?;
            let do_cpu = cpu_input(&do_guard)?;
            let alibi_cpu = match &alibi_guard {
                Some(g) => Some(cpu_input(g)?),
                None => None,
            };
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(
                q_cpu, k_cpu, v_cpu, do_cpu, alibi_cpu, out_cpu, $which,
                b, hq, hkv, sq, sk, d,
                scale, causal, wl, wr, softcap,
            )
        }
    };
}

cpu_flash_attn_backward_wrapper!(flash_attn_backward_q_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_backward_f32,  fuel_cpu_backend::byte_kernels::FaBackwardWhich::Q);
cpu_flash_attn_backward_wrapper!(flash_attn_backward_k_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_backward_f32,  fuel_cpu_backend::byte_kernels::FaBackwardWhich::K);
cpu_flash_attn_backward_wrapper!(flash_attn_backward_v_f32_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_backward_f32,  fuel_cpu_backend::byte_kernels::FaBackwardWhich::V);
cpu_flash_attn_backward_wrapper!(flash_attn_backward_q_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_backward_f64,  fuel_cpu_backend::byte_kernels::FaBackwardWhich::Q);
cpu_flash_attn_backward_wrapper!(flash_attn_backward_k_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_backward_f64,  fuel_cpu_backend::byte_kernels::FaBackwardWhich::K);
cpu_flash_attn_backward_wrapper!(flash_attn_backward_v_f64_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_backward_f64,  fuel_cpu_backend::byte_kernels::FaBackwardWhich::V);
cpu_flash_attn_backward_wrapper!(flash_attn_backward_q_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::flash_attn_backward_bf16, fuel_cpu_backend::byte_kernels::FaBackwardWhich::Q);
cpu_flash_attn_backward_wrapper!(flash_attn_backward_k_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::flash_attn_backward_bf16, fuel_cpu_backend::byte_kernels::FaBackwardWhich::K);
cpu_flash_attn_backward_wrapper!(flash_attn_backward_v_bf16_cpu_wrapper, fuel_cpu_backend::byte_kernels::flash_attn_backward_bf16, fuel_cpu_backend::byte_kernels::FaBackwardWhich::V);
cpu_flash_attn_backward_wrapper!(flash_attn_backward_q_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_backward_f16,  fuel_cpu_backend::byte_kernels::FaBackwardWhich::Q);
cpu_flash_attn_backward_wrapper!(flash_attn_backward_k_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_backward_f16,  fuel_cpu_backend::byte_kernels::FaBackwardWhich::K);
cpu_flash_attn_backward_wrapper!(flash_attn_backward_v_f16_cpu_wrapper,  fuel_cpu_backend::byte_kernels::flash_attn_backward_f16,  fuel_cpu_backend::byte_kernels::FaBackwardWhich::V);

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
        DType::F64    => fuel_cpu_backend::byte_kernels::cast_f64_to_f32,
        DType::BF16   => fuel_cpu_backend::byte_kernels::cast_bf16_to_f32,
        DType::F16    => fuel_cpu_backend::byte_kernels::cast_f16_to_f32,
        DType::F8E4M3 => fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_f32,
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
        DType::F32    => fuel_cpu_backend::byte_kernels::cast_f32_to_bf16,
        DType::F8E4M3 => fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_bf16,
    },
);
cpu_cast_wrapper!(
    cast_to_f16_cpu_wrapper,
    DType::F16,
    "f16",
    {
        DType::F32    => fuel_cpu_backend::byte_kernels::cast_f32_to_f16,
        DType::F8E4M3 => fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_f16,
    },
);
cpu_cast_wrapper!(
    cast_to_f8e4m3_cpu_wrapper,
    DType::F8E4M3,
    "f8e4m3",
    {
        DType::F32  => fuel_cpu_backend::byte_kernels::cast_f32_to_f8e4m3,
        DType::F16  => fuel_cpu_backend::byte_kernels::cast_f16_to_f8e4m3,
        DType::BF16 => fuel_cpu_backend::byte_kernels::cast_bf16_to_f8e4m3,
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
cpu_matmul_wrapper!(matmul_i8_cpu_wrapper,   fuel_cpu_backend::byte_kernels::matmul_i8,   "matmul_i8");
cpu_matmul_wrapper!(matmul_u8_cpu_wrapper,   fuel_cpu_backend::byte_kernels::matmul_u8,   "matmul_u8");

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

/// Dispatch wrapper for `(OpKind::Copy, [T, T], Cpu)` — copy from a
/// CPU source storage into a freshly-allocated output on any target
/// backend. The executor allocates the output on `target_location`
/// via [`crate::pipelined::WorkItemKind::Copy`] before this wrapper
/// runs; the wrapper switches on the output's `BackendStorage`
/// variant to pick the H2D path.
///
/// - CPU output → memcpy (CPU→CPU).
/// - CUDA output → `CudaStorageBytes::write_from_host` (H2D via
///   `cuMemcpyHtoD`).
/// - Vulkan output → `VulkanBackend::write_bytes` (H2D via staging
///   buffer + `vkCmdCopyBuffer`).
///
/// Bridge-retirement Phase 2 (CPU→CPU) + Phase 3b (CPU→GPU). The
/// wrapper-internal match on output variant is the pragmatic shape
/// while the binding-table key `(op, dtypes, source_backend)` doesn't
/// yet encode the target. A future binding-table-key extension would
/// split this into per-(source, target) sub-wrappers without changing
/// the call surface.
///
/// Replaces:
/// * The per-variant `match self` in `BackendStorage::read_to_cpu_bytes`
///   (Phase 2 deletion).
/// * `fuel-core::pipelined_bridge::upload_host_buffer`'s per-
///   `DeviceLocation` match (Phase 3b deletion).
fn copy_from_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "copy_from_cpu_wrapper: expected 1 input + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        )).bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let src = cpu_input(&in_guard)?;
    let mut out_guard = write_storage(&outputs[0])?;
    // Truncate source bytes to the destination's byte count — the
    // executor allocates the output exactly `node.shape.elem_count *
    // dtype.size_in_bytes()` bytes. Host buffers from Op::Const slots
    // may be larger (shared storage across views); we copy only the
    // destination-sized prefix. Mirrors the deleted
    // `upload_host_buffer`'s `truncate_to` parameter.
    let n_out = out_guard.inner.len_bytes();
    let n_src = src.len_bytes();
    let n = n_src.min(n_out);
    let src_slice = &src.bytes()[..n];
    match &mut out_guard.inner {
        BackendStorage::Cpu(dst) => {
            dst.bytes_mut()[..n].copy_from_slice(src_slice);
        }
        #[cfg(feature = "cuda")]
        BackendStorage::Cuda(dst) => {
            // CUDA write_from_host requires src.len() == dst.len_bytes.
            // The executor sized the output to `n_out`; we truncated
            // src to `n` = min(n_src, n_out). When n < n_out the
            // remaining bytes stay uninit; for the Op::Const upload
            // path this matches the deleted upload_host_buffer's
            // behavior (truncated host bytes uploaded as-is).
            if n == n_out {
                dst.write_from_host(src_slice)?;
            } else {
                // Pad: copy src into a staging vec sized exactly n_out
                // with trailing zeros. Rare path (size mismatch).
                let mut staged = vec![0_u8; n_out];
                staged[..n].copy_from_slice(src_slice);
                dst.write_from_host(&staged)?;
            }
        }
        #[cfg(feature = "vulkan")]
        BackendStorage::Vulkan(dst) => {
            let backend = dst.backend().ok_or_else(|| Error::Msg(
                "copy_from_cpu_wrapper: Vulkan output has no attached \
                 backend handle. The executor's Op::Copy arm must allocate \
                 via VulkanBackend::alloc_bytes_handle (which attaches \
                 the handle).".to_string()
            ).bt())?.clone();
            if n == n_out {
                backend.write_bytes(dst, src_slice)?;
            } else {
                let mut staged = vec![0_u8; n_out];
                staged[..n].copy_from_slice(src_slice);
                backend.write_bytes(dst, &staged)?;
            }
        }
        #[allow(unreachable_patterns)]
        other => {
            return Err(Error::Msg(format!(
                "copy_from_cpu_wrapper: output backend not wired ({other:?}); \
                 CPU + CUDA + Vulkan covered, Metal extends when its \
                 byte-storage substrate is ready.",
            )).bt());
        }
    }
    Ok(())
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
    let fa_bw_no_alibi   = |t: DType| [t, t, t, t, t];          // (q, k, v, do, out)
    let fa_bw_with_alibi = |t: DType| [t, t, t, t, t, t];       // (q, k, v, do, alibi, out)
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
    // Integer MatMul — i32 accumulator, saturating cast back to T on
    // store. Mirrors baracuda's `gemm_{s8,u8}_rrr_sm80_run` contract.
    table.register(MatMul,             &binary(DType::I8), cpu, matmul_i8_cpu_wrapper);
    table.register(MatMul,             &binary(u8_dt),     cpu, matmul_u8_cpu_wrapper);

    // bf16 + f16 reductions — accumulate in f32 for stability.
    table.register(SumReduce,          &unary(bf16_dt), cpu, sum_reduce_bf16_cpu_wrapper);
    table.register(MaxReduce,          &unary(bf16_dt), cpu, max_reduce_bf16_cpu_wrapper);
    table.register(MinReduce,          &unary(bf16_dt), cpu, min_reduce_bf16_cpu_wrapper);
    table.register(MeanReduce,         &unary(bf16_dt), cpu, mean_reduce_bf16_cpu_wrapper);
    table.register(SumReduce,          &unary(f16_dt),  cpu, sum_reduce_f16_cpu_wrapper);
    table.register(MaxReduce,          &unary(f16_dt),  cpu, max_reduce_f16_cpu_wrapper);
    table.register(MinReduce,          &unary(f16_dt),  cpu, min_reduce_f16_cpu_wrapper);
    table.register(MeanReduce,         &unary(f16_dt),  cpu, mean_reduce_f16_cpu_wrapper);

    // Cast — register every (src, dst) pair the per-target wrapper
    // handles internally. The wrapper dispatches by source dtype via
    // a `match`; the binding-table key needs to match the actual
    // dtypes the executor produces (`[src_dt, dst_dt]`).
    //
    // Identity pairs (`[T, T]`) are not registered here because the
    // wrappers' internal match doesn't include the identity arm —
    // Fuel's graph optimizer elides identity Cast before dispatch.
    let cast_to_f32 = cast_to_f32_cpu_wrapper as KernelRef;
    table.register(Cast, &[DType::F64,  DType::F32], cpu, cast_to_f32);
    table.register(Cast, &[DType::BF16, DType::F32], cpu, cast_to_f32);
    table.register(Cast, &[DType::F16,  DType::F32], cpu, cast_to_f32);
    table.register(Cast, &[DType::F32,  DType::F64], cpu, cast_to_f64_cpu_wrapper);
    table.register(Cast, &[DType::F32,  DType::BF16], cpu, cast_to_bf16_cpu_wrapper);
    table.register(Cast, &[DType::F32,  DType::F16], cpu, cast_to_f16_cpu_wrapper);

    // F8E4M3 ↔ {F32, F16, BF16} — mirrors baracuda alpha.29's
    // CastSubBytePlan surface. CPU side pivots F16/BF16 through f32
    // (see byte_kernels.rs cast section).
    let cast_to_f8 = cast_to_f8e4m3_cpu_wrapper as KernelRef;
    table.register(Cast, &[DType::F8E4M3, DType::F32],    cpu, cast_to_f32);
    table.register(Cast, &[DType::F8E4M3, DType::BF16],   cpu, cast_to_bf16_cpu_wrapper);
    table.register(Cast, &[DType::F8E4M3, DType::F16],    cpu, cast_to_f16_cpu_wrapper);
    table.register(Cast, &[DType::F32,    DType::F8E4M3], cpu, cast_to_f8);
    table.register(Cast, &[DType::BF16,   DType::F8E4M3], cpu, cast_to_f8);
    table.register(Cast, &[DType::F16,    DType::F8E4M3], cpu, cast_to_f8);

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

    // FlashAttn backward — three OpKinds (Q/K/V), four dtypes, two
    // alibi shapes each. The CPU wrapper computes all three gradients
    // every call and copies the requested one into `out`; expect ~3×
    // the cost of a single-gradient backward kernel until a fused
    // multi-output variant lands.
    for (dt, wq, wk, wv) in [
        (f32_dt,  flash_attn_backward_q_f32_cpu_wrapper  as KernelRef,
                  flash_attn_backward_k_f32_cpu_wrapper  as KernelRef,
                  flash_attn_backward_v_f32_cpu_wrapper  as KernelRef),
        (f64_dt,  flash_attn_backward_q_f64_cpu_wrapper,
                  flash_attn_backward_k_f64_cpu_wrapper,
                  flash_attn_backward_v_f64_cpu_wrapper),
        (bf16_dt, flash_attn_backward_q_bf16_cpu_wrapper,
                  flash_attn_backward_k_bf16_cpu_wrapper,
                  flash_attn_backward_v_bf16_cpu_wrapper),
        (f16_dt,  flash_attn_backward_q_f16_cpu_wrapper,
                  flash_attn_backward_k_f16_cpu_wrapper,
                  flash_attn_backward_v_f16_cpu_wrapper),
    ] {
        table.register(FlashAttnBackwardQ, &fa_bw_no_alibi(dt),   cpu, wq);
        table.register(FlashAttnBackwardQ, &fa_bw_with_alibi(dt), cpu, wq);
        table.register(FlashAttnBackwardK, &fa_bw_no_alibi(dt),   cpu, wk);
        table.register(FlashAttnBackwardK, &fa_bw_with_alibi(dt), cpu, wk);
        table.register(FlashAttnBackwardV, &fa_bw_no_alibi(dt),   cpu, wv);
        table.register(FlashAttnBackwardV, &fa_bw_with_alibi(dt), cpu, wv);
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

    // In-place affine — Phase 3c of in-place ops infrastructure.
    // Binding-table key shape mirrors the non-inplace Affine ([T, T])
    // because `build_lookup_dtypes` produces that for an Op::Fused
    // INPLACE_AFFINE node with 1 input + 1 output of the same dtype.
    // The wrapper rejects non-empty `inputs` (the executor's
    // InplaceKernel arm passes the target as `outputs[0]` instead),
    // but the binding-table KEY is what matters for lookup.
    table.register(InplaceAffine, &unary(f32_dt),  cpu, inplace_affine_f32_cpu_wrapper);
    table.register(InplaceAffine, &unary(f64_dt),  cpu, inplace_affine_f64_cpu_wrapper);
    table.register(InplaceAffine, &unary(bf16_dt), cpu, inplace_affine_bf16_cpu_wrapper);
    table.register(InplaceAffine, &unary(f16_dt),  cpu, inplace_affine_f16_cpu_wrapper);

    // ClampInplace + PowIInplace — Phase 3e scalar-param family. Same
    // [T, T] key shape as InplaceAffine; OpParams::{Clamp, PowI} carry
    // the scalars to the wrapper.
    table.register(ClampInplace, &unary(f32_dt),  cpu, clamp_inplace_f32_cpu_wrapper);
    table.register(ClampInplace, &unary(f64_dt),  cpu, clamp_inplace_f64_cpu_wrapper);
    table.register(ClampInplace, &unary(bf16_dt), cpu, clamp_inplace_bf16_cpu_wrapper);
    table.register(ClampInplace, &unary(f16_dt),  cpu, clamp_inplace_f16_cpu_wrapper);

    table.register(PowIInplace, &unary(f32_dt),  cpu, powi_inplace_f32_cpu_wrapper);
    table.register(PowIInplace, &unary(f64_dt),  cpu, powi_inplace_f64_cpu_wrapper);
    table.register(PowIInplace, &unary(bf16_dt), cpu, powi_inplace_bf16_cpu_wrapper);
    table.register(PowIInplace, &unary(f16_dt),  cpu, powi_inplace_f16_cpu_wrapper);

    // FusedSoftmaxCrossEntropy: 2 inputs (logits T, targets I64) →
    // 1 output (F32, the FSCE declared dtype regardless of logits T).
    // The lookup key `[T, I64, F32]` matches what `build_lookup_dtypes`
    // produces for this node shape. T ∈ {F32, F64, BF16, F16}.
    for (logits_dt, w) in [
        (DType::F32,  fused_softmax_cross_entropy_f32_cpu_wrapper  as KernelRef),
        (DType::F64,  fused_softmax_cross_entropy_f64_cpu_wrapper),
        (DType::BF16, fused_softmax_cross_entropy_bf16_cpu_wrapper),
        (DType::F16,  fused_softmax_cross_entropy_f16_cpu_wrapper),
    ] {
        table.register(
            FusedSoftmaxCrossEntropy,
            &[logits_dt, DType::I64, DType::F32],
            cpu,
            w,
        );
    }

    // CausalConv1d: 3 inputs (x, weight, bias) + 1 output, uniform
    // dtype T ∈ {F32, F64, BF16, F16}.
    table.register(
        CausalConv1d, &[DType::F32, DType::F32, DType::F32, DType::F32],
        cpu, causal_conv1d_f32_cpu_wrapper,
    );
    table.register(
        CausalConv1d, &[DType::F64, DType::F64, DType::F64, DType::F64],
        cpu, causal_conv1d_f64_cpu_wrapper,
    );
    table.register(
        CausalConv1d, &[DType::BF16, DType::BF16, DType::BF16, DType::BF16],
        cpu, causal_conv1d_bf16_cpu_wrapper,
    );
    table.register(
        CausalConv1d, &[DType::F16, DType::F16, DType::F16, DType::F16],
        cpu, causal_conv1d_f16_cpu_wrapper,
    );

    // SelectiveScan: 5 inputs (u, delta, a, b, c) + 1 output, uniform
    // dtype T ∈ {F32, F64, BF16, F16}.
    for (dt, w) in [
        (DType::F32,  selective_scan_f32_cpu_wrapper  as KernelRef),
        (DType::F64,  selective_scan_f64_cpu_wrapper),
        (DType::BF16, selective_scan_bf16_cpu_wrapper),
        (DType::F16,  selective_scan_f16_cpu_wrapper),
    ] {
        table.register(SelectiveScan, &[dt, dt, dt, dt, dt, dt], cpu, w);
    }

    // SsdChunkScan: 5 inputs (x, dt, a, b, c) + 1 output, uniform
    // dtype T ∈ {F32, F64, BF16, F16}.
    for (dt, w) in [
        (DType::F32,  ssd_chunk_scan_f32_cpu_wrapper  as KernelRef),
        (DType::F64,  ssd_chunk_scan_f64_cpu_wrapper),
        (DType::BF16, ssd_chunk_scan_bf16_cpu_wrapper),
        (DType::F16,  ssd_chunk_scan_f16_cpu_wrapper),
    ] {
        table.register(SsdChunkScan, &[dt, dt, dt, dt, dt, dt], cpu, w);
    }

    // Nf4Matmul: 3 inputs (activations T, w_packed U8, absmax F32) + 1
    // output T, where T ∈ {F32, F16, BF16}. The binding-table key
    // shape is [T, U8, F32, T].
    table.register(
        Nf4Matmul,
        &[DType::F32, DType::U8, DType::F32, DType::F32],
        cpu,
        nf4_matmul_f32_cpu_wrapper,
    );
    table.register(
        Nf4Matmul,
        &[DType::F16, DType::U8, DType::F32, DType::F16],
        cpu,
        nf4_matmul_f16_cpu_wrapper,
    );
    table.register(
        Nf4Matmul,
        &[DType::BF16, DType::U8, DType::F32, DType::BF16],
        cpu,
        nf4_matmul_bf16_cpu_wrapper,
    );

    // In-place unary activations — Phase 3e of in-place ops. Same
    // `[T, T]` key shape as the non-inplace cousins (the natural
    // shape build_lookup_dtypes produces for a 1-input + 1-output
    // node with matching dtypes). Full 4-dtype coverage (f32/f64/bf16/
    // f16); bf16+f16 route through the chassis's f32-pivot blanket
    // impls so the numerics match the non-inplace cousins bit-for-bit.
    table.register(ReluInplace,    &unary(f32_dt), cpu, relu_inplace_f32_cpu_wrapper);
    table.register(SiluInplace,    &unary(f32_dt), cpu, silu_inplace_f32_cpu_wrapper);
    table.register(GeluInplace,    &unary(f32_dt), cpu, gelu_inplace_f32_cpu_wrapper);
    table.register(TanhInplace,    &unary(f32_dt), cpu, tanh_inplace_f32_cpu_wrapper);
    table.register(SigmoidInplace, &unary(f32_dt), cpu, sigmoid_inplace_f32_cpu_wrapper);

    table.register(ReluInplace,    &unary(f64_dt), cpu, relu_inplace_f64_cpu_wrapper);
    table.register(SiluInplace,    &unary(f64_dt), cpu, silu_inplace_f64_cpu_wrapper);
    table.register(GeluInplace,    &unary(f64_dt), cpu, gelu_inplace_f64_cpu_wrapper);
    table.register(TanhInplace,    &unary(f64_dt), cpu, tanh_inplace_f64_cpu_wrapper);
    table.register(SigmoidInplace, &unary(f64_dt), cpu, sigmoid_inplace_f64_cpu_wrapper);

    table.register(ReluInplace,    &unary(bf16_dt), cpu, relu_inplace_bf16_cpu_wrapper);
    table.register(SiluInplace,    &unary(bf16_dt), cpu, silu_inplace_bf16_cpu_wrapper);
    table.register(GeluInplace,    &unary(bf16_dt), cpu, gelu_inplace_bf16_cpu_wrapper);
    table.register(TanhInplace,    &unary(bf16_dt), cpu, tanh_inplace_bf16_cpu_wrapper);
    table.register(SigmoidInplace, &unary(bf16_dt), cpu, sigmoid_inplace_bf16_cpu_wrapper);

    table.register(ReluInplace,    &unary(f16_dt), cpu, relu_inplace_f16_cpu_wrapper);
    table.register(SiluInplace,    &unary(f16_dt), cpu, silu_inplace_f16_cpu_wrapper);
    table.register(GeluInplace,    &unary(f16_dt), cpu, gelu_inplace_f16_cpu_wrapper);
    table.register(TanhInplace,    &unary(f16_dt), cpu, tanh_inplace_f16_cpu_wrapper);
    table.register(SigmoidInplace, &unary(f16_dt), cpu, sigmoid_inplace_f16_cpu_wrapper);

    // In-place unary op family expansion (2026-05-30) — 16 new ops
    // × 4 dtypes = 64 new (OpKind, [T, T], Cpu) entries.
    for (op, regs) in [
        (NegInplace,    [neg_inplace_f32_cpu_wrapper,    neg_inplace_f64_cpu_wrapper,    neg_inplace_bf16_cpu_wrapper,    neg_inplace_f16_cpu_wrapper]),
        (AbsInplace,    [abs_inplace_f32_cpu_wrapper,    abs_inplace_f64_cpu_wrapper,    abs_inplace_bf16_cpu_wrapper,    abs_inplace_f16_cpu_wrapper]),
        (SqrInplace,    [sqr_inplace_f32_cpu_wrapper,    sqr_inplace_f64_cpu_wrapper,    sqr_inplace_bf16_cpu_wrapper,    sqr_inplace_f16_cpu_wrapper]),
        (SqrtInplace,   [sqrt_inplace_f32_cpu_wrapper,   sqrt_inplace_f64_cpu_wrapper,   sqrt_inplace_bf16_cpu_wrapper,   sqrt_inplace_f16_cpu_wrapper]),
        (RsqrtInplace,  [rsqrt_inplace_f32_cpu_wrapper,  rsqrt_inplace_f64_cpu_wrapper,  rsqrt_inplace_bf16_cpu_wrapper,  rsqrt_inplace_f16_cpu_wrapper]),
        (RecipInplace,  [recip_inplace_f32_cpu_wrapper,  recip_inplace_f64_cpu_wrapper,  recip_inplace_bf16_cpu_wrapper,  recip_inplace_f16_cpu_wrapper]),
        (ExpInplace,    [exp_inplace_f32_cpu_wrapper,    exp_inplace_f64_cpu_wrapper,    exp_inplace_bf16_cpu_wrapper,    exp_inplace_f16_cpu_wrapper]),
        (LogInplace,    [log_inplace_f32_cpu_wrapper,    log_inplace_f64_cpu_wrapper,    log_inplace_bf16_cpu_wrapper,    log_inplace_f16_cpu_wrapper]),
        (SinInplace,    [sin_inplace_f32_cpu_wrapper,    sin_inplace_f64_cpu_wrapper,    sin_inplace_bf16_cpu_wrapper,    sin_inplace_f16_cpu_wrapper]),
        (CosInplace,    [cos_inplace_f32_cpu_wrapper,    cos_inplace_f64_cpu_wrapper,    cos_inplace_bf16_cpu_wrapper,    cos_inplace_f16_cpu_wrapper]),
        (SignInplace,   [sign_inplace_f32_cpu_wrapper,   sign_inplace_f64_cpu_wrapper,   sign_inplace_bf16_cpu_wrapper,   sign_inplace_f16_cpu_wrapper]),
        (FloorInplace,  [floor_inplace_f32_cpu_wrapper,  floor_inplace_f64_cpu_wrapper,  floor_inplace_bf16_cpu_wrapper,  floor_inplace_f16_cpu_wrapper]),
        (CeilInplace,   [ceil_inplace_f32_cpu_wrapper,   ceil_inplace_f64_cpu_wrapper,   ceil_inplace_bf16_cpu_wrapper,   ceil_inplace_f16_cpu_wrapper]),
        (RoundInplace,  [round_inplace_f32_cpu_wrapper,  round_inplace_f64_cpu_wrapper,  round_inplace_bf16_cpu_wrapper,  round_inplace_f16_cpu_wrapper]),
        (ErfInplace,    [erf_inplace_f32_cpu_wrapper,    erf_inplace_f64_cpu_wrapper,    erf_inplace_bf16_cpu_wrapper,    erf_inplace_f16_cpu_wrapper]),
        (GeluErfInplace,[gelu_erf_inplace_f32_cpu_wrapper, gelu_erf_inplace_f64_cpu_wrapper, gelu_erf_inplace_bf16_cpu_wrapper, gelu_erf_inplace_f16_cpu_wrapper]),
    ] {
        for (dt, wrapper) in [f32_dt, f64_dt, bf16_dt, f16_dt].into_iter().zip(regs.into_iter()) {
            table.register(op, &unary(dt), cpu, wrapper);
        }
    }

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

    // WriteSlice — Phase 7.6 step 9c E.3.2. The kernel is dtype-
    // agnostic at the byte level (it just memcpy's slabs); per-dtype
    // entries exist so the binding-table lookup matches the executor's
    // `[T_src, T_out]` canonicalized key. Coverage: every dtype the
    // KV cache might hold today + integer dtypes for future index-
    // table use cases.
    table.register(WriteSlice, &unary(f32_dt),  cpu, write_slice_cpu_wrapper);
    table.register(WriteSlice, &unary(f64_dt),  cpu, write_slice_cpu_wrapper);
    table.register(WriteSlice, &unary(bf16_dt), cpu, write_slice_cpu_wrapper);
    table.register(WriteSlice, &unary(f16_dt),  cpu, write_slice_cpu_wrapper);
    table.register(WriteSlice, &unary(u32_dt),  cpu, write_slice_cpu_wrapper);
    table.register(WriteSlice, &unary(u8_dt),   cpu, write_slice_cpu_wrapper);

    // WriteSliceRotating — Phase C. Sliding-window KV cache writes.
    // Same dtype surface as WriteSlice; the binding-table key is
    // `[T_src, T_out]` (position scalar is a separate kernel input,
    // not part of the lookup key — see PipelinedExecutor's
    // WriteSliceRotating arm).
    table.register(WriteSliceRotating, &unary(f32_dt),  cpu, write_slice_rotating_cpu_wrapper);
    table.register(WriteSliceRotating, &unary(f64_dt),  cpu, write_slice_rotating_cpu_wrapper);
    table.register(WriteSliceRotating, &unary(bf16_dt), cpu, write_slice_rotating_cpu_wrapper);
    table.register(WriteSliceRotating, &unary(f16_dt),  cpu, write_slice_rotating_cpu_wrapper);
    table.register(WriteSliceRotating, &unary(u32_dt),  cpu, write_slice_rotating_cpu_wrapper);
    table.register(WriteSliceRotating, &unary(u8_dt),   cpu, write_slice_rotating_cpu_wrapper);

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

    // Phase 7.6 step 6 follow-up — backward helpers gain byte-level
    // CPU coverage. Each takes 2 inputs (y/x, g/upstream) + 1 output;
    // dtype tuple matches the binary `[T, T, T]` shape.
    table.register(SoftmaxLastDimBackward,    &binary(f32_dt),  cpu, softmax_last_dim_backward_f32_cpu_wrapper);
    table.register(SoftmaxLastDimBackward,    &binary(f64_dt),  cpu, softmax_last_dim_backward_f64_cpu_wrapper);
    table.register(SoftmaxLastDimBackward,    &binary(bf16_dt), cpu, softmax_last_dim_backward_bf16_cpu_wrapper);
    table.register(SoftmaxLastDimBackward,    &binary(f16_dt),  cpu, softmax_last_dim_backward_f16_cpu_wrapper);
    table.register(LayerNormLastDimBackward,  &binary(f32_dt),  cpu, layer_norm_last_dim_backward_f32_cpu_wrapper);
    table.register(LayerNormLastDimBackward,  &binary(f64_dt),  cpu, layer_norm_last_dim_backward_f64_cpu_wrapper);
    table.register(LayerNormLastDimBackward,  &binary(bf16_dt), cpu, layer_norm_last_dim_backward_bf16_cpu_wrapper);
    table.register(LayerNormLastDimBackward,  &binary(f16_dt),  cpu, layer_norm_last_dim_backward_f16_cpu_wrapper);
    table.register(RmsNormLastDimBackward,    &binary(f32_dt),  cpu, rms_norm_last_dim_backward_f32_cpu_wrapper);
    table.register(RmsNormLastDimBackward,    &binary(f64_dt),  cpu, rms_norm_last_dim_backward_f64_cpu_wrapper);
    table.register(RmsNormLastDimBackward,    &binary(bf16_dt), cpu, rms_norm_last_dim_backward_bf16_cpu_wrapper);
    table.register(RmsNormLastDimBackward,    &binary(f16_dt),  cpu, rms_norm_last_dim_backward_f16_cpu_wrapper);
    table.register(ReduceMaxToBackward,       &binary(f32_dt),  cpu, reduce_max_to_backward_f32_cpu_wrapper);
    table.register(ReduceMaxToBackward,       &binary(f64_dt),  cpu, reduce_max_to_backward_f64_cpu_wrapper);
    table.register(ReduceMaxToBackward,       &binary(bf16_dt), cpu, reduce_max_to_backward_bf16_cpu_wrapper);
    table.register(ReduceMaxToBackward,       &binary(f16_dt),  cpu, reduce_max_to_backward_f16_cpu_wrapper);

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

    // Op::Copy — D2H / SameDevice byte-level transfer. The CPU
    // registration here is the CPU→CPU memcpy noop; per-source-
    // backend registrations live in their dispatch crates
    // (`register_cuda_kernels`, `vulkan_dispatch::register_vulkan_kernels`).
    // Bridge-retirement Phase 2 (post-9c): every realize root that
    // isn't on CPU gets an `Op::Copy { target: Cpu }` spliced in
    // before realize; the kernel dispatch resolves on the *source*
    // backend's BackendId. CPU→CPU is registered for uniformity (the
    // splice doesn't fire when device == Cpu, but kernels registered
    // here let direct executor tests round-trip through OpKind::Copy
    // on the universal fallback).
    let copy_dtypes = [
        f32_dt, f64_dt, bf16_dt, f16_dt, u32_dt, u8_dt, DType::I16, DType::I32, DType::I64,
    ];
    for dt in copy_dtypes {
        table.register(Copy, &[dt, dt], cpu, copy_from_cpu_wrapper);
    }
    table.register(ClampElementwise,   &unary(f64_dt),  cpu, clamp_f64_cpu_wrapper);
    table.register(ClampElementwise,   &unary(bf16_dt), cpu, clamp_bf16_cpu_wrapper);
    table.register(ClampElementwise,   &unary(f16_dt),  cpu, clamp_f16_cpu_wrapper);
    table.register(PowIElementwise,    &unary(f64_dt),  cpu, powi_f64_cpu_wrapper);
    table.register(PowIElementwise,    &unary(bf16_dt), cpu, powi_bf16_cpu_wrapper);
    table.register(PowIElementwise,    &unary(f16_dt),  cpu, powi_f16_cpu_wrapper);

    // PowI backward — `(x, upstream) → grad_x`. Two inputs, same
    // dtype on both inputs + output. Routes Op::Fused(POWI_BACKWARD,
    // _) emitted by autograd through the binding table.
    table.register(PowIElementwiseBackward, &binary(f32_dt),  cpu, powi_backward_f32_cpu_wrapper);
    table.register(PowIElementwiseBackward, &binary(f64_dt),  cpu, powi_backward_f64_cpu_wrapper);
    table.register(PowIElementwiseBackward, &binary(bf16_dt), cpu, powi_backward_bf16_cpu_wrapper);
    table.register(PowIElementwiseBackward, &binary(f16_dt),  cpu, powi_backward_f16_cpu_wrapper);

    // Phase 7.6 step 7b — populate `PrecisionGuarantee` for every
    // CPU primitive registration. Every `table.register(...)` call
    // above defaulted to `PrecisionGuarantee::UNAUDITED`; this fill
    // pass upgrades them to `PRIMITIVE_DETERMINISTIC_CPU` (bit-stable
    // per hardware, deterministic iteration order, F32 accumulator
    // for half-precision). Sites that need a weaker claim should
    // call `table.register_with_precision(...)` explicitly *before*
    // this point — those won't be overwritten because the fill only
    // touches UNAUDITED entries.
    //
    // The architecture-target shape is precision-per-call-site,
    // but the 335+ CPU primitive registrations all share the same
    // deterministic property; bulk-applying the claim here keeps
    // the call sites concise while still ensuring every entry
    // carries an explicit, non-UNAUDITED PrecisionGuarantee that the
    // step-7b coverage lint can enforce.
    table.fill_unset_cpu_precision(crate::fused::PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU);

    // Phase 7.6 step 8 — populate cost functions for every CPU
    // primitive registration. Every `table.register(...)` call
    // above defaulted to `unknown_cost`; this fill pass dispatches
    // each entry to its OpKind-appropriate cost-family function
    // (per `crate::cost::default_cost_for_op_kind`). Sites that
    // need a non-default cost claim should call
    // `table.register_full(...)` with their own `CostFn` *before*
    // this point — those won't be overwritten because the fill
    // only touches entries still bound to `unknown_cost`.
    //
    // Architecturally this is the same shape as the precision fill
    // above: bulk-apply the family-default at the end so the 335
    // CPU registration call sites stay concise, while the lint
    // (`precision_guarantee_lint_bit_stable_cpu_coverage_primitives`
    // companion: `cost_lint_per_op_kind_cpu_coverage`) enforces
    // that every OpKind ends up with a non-default cost function.
    table.fill_unset_cpu_cost(crate::cost::default_cost_for_op_kind);
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
pub(crate) fn cuda_input(s: &Storage) -> Result<&fuel_cuda_backend::CudaStorageBytes> {
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
pub(crate) fn cuda_output(s: &mut Storage) -> Result<&mut fuel_cuda_backend::CudaStorageBytes> {
    match &mut s.inner {
        BackendStorage::Cuda(c) => Ok(c),
        #[allow(unreachable_patterns)]
        _ => Err(Error::Msg(
            "cuda kernel wrapper called with non-CUDA output".to_string(),
        )
        .bt()),
    }
}

/// Dispatch wrapper for `(OpKind::Copy, [T, T], Cuda)` — copy FROM a
/// CUDA source storage. The key is the SOURCE backend; the
/// destination is whatever the executor pre-allocated the output as
/// (see [`crate::pipelined::WorkItemKind::Copy`]), so the wrapper
/// routes on the output's variant:
///
/// - **CPU output** → D2H via `CudaStorageBytes::to_cpu_bytes`
///   (the realize-root path; the original sole behavior).
/// - **CUDA output** → device-to-device via
///   `CudaStorageBytes::slot_copy_to_new` (one `memcpy_dtod`).
///   Same-device copies are emitted by `insert_safety_copies` for
///   destructive ops on GPU-placed tensors and by the residency
///   machinery — before this routing existed, any such Copy errored
///   with "cpu kernel wrapper called with non-CPU output"
///   (residency_eviction_live caught it).
///
/// Dtype-agnostic at the byte level — one wrapper covers every dtype
/// registered at this key.
///
/// Phase 2 of the bridge-retirement trajectory (post-9c). Replaces
/// the CUDA branch of `BackendStorage::read_to_cpu_bytes` (deleted
/// alongside).
#[cfg(feature = "cuda")]
fn copy_from_cuda_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "copy_from_cuda_wrapper: expected 1 input + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        )).bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let cuda_src = cuda_input(&in_guard)?;
    let mut out_guard = write_storage(&outputs[0])?;
    match &mut out_guard.inner {
        BackendStorage::Cpu(_) => {
            // Step E A4b-3: the FINER cross-device D2H. The executor waited the
            // source node's CompletionHandle before dispatching this Copy/Move
            // (`pipelined::wait_producer_handle`), so the producer is done; use
            // `to_cpu_bytes_finer` (no whole-device `synchronize`) so the OTHER
            // sub-DAG's independent in-flight CUDA work is NOT force-drained.
            // `cuMemcpyDtoH_v2` is itself host-synchronous + legacy-stream-ordered
            // (see `CudaStorageBytes::to_cpu_bytes_finer`), so the read stays
            // byte-exact. This wrapper is reached ONLY via the executor's
            // Copy/Move WorkItem (it is the kernel registered at
            // `(OpKind::Copy, [dt,dt], Cuda)`), never standalone — so the finer
            // contract always holds here.
            let bytes = cuda_src.to_cpu_bytes_finer()?;
            let dst = cpu_output(&mut out_guard)?;
            let n = bytes.len().min(dst.len_bytes());
            dst.bytes_mut()[..n].copy_from_slice(&bytes[..n]);
            Ok(())
        }
        BackendStorage::Cuda(_) => {
            // Device-to-device: replace the pre-allocated output's
            // bytes with a fresh DtoD copy of the full source buffer.
            let copied = cuda_src.slot_copy_to_new(0, cuda_src.len_bytes())?;
            let dst = cuda_output(&mut out_guard)?;
            *dst = copied;
            Ok(())
        }
        #[allow(unreachable_patterns)]
        other => Err(Error::Msg(format!(
            "copy_from_cuda_wrapper: unsupported output substrate {:?} \
             (CUDA sources copy to CPU or CUDA outputs only; cross-vendor \
             GPU transfer goes through host staging as two Copy hops)",
            std::mem::discriminant(other),
        )).bt()),
    }
}

/// Phase 7.5 CUDA registration — now `Op::Copy` D2H only.
///
/// **Scope (post-alpha.67, 2026-06-10):** every compute kernel has
/// migrated to `register_baracuda_cuda_kernels` — baracuda is the
/// single CUDA kernel home. The final migrations (answering
/// `docs/baracuda-ask-fp-gemm-reduce-to-2026-06-10.md`):
///
/// - **MatMul** (f32/f64/f16/bf16) → Phase 74 `gemm_dense` facade;
///   the local cuBLAS f32 path and CUTLASS bf16/f16 byte paths
///   retired with `byte_kernels`.
/// - **ReduceSumTo / ReduceMaxTo** (4 dtypes) → baracuda's
///   `reduce_{sum,max}_to_*` (sys symbols shipped since alpha.46).
/// - **CausalConv1d** → registration moved (was always
///   baracuda-backed via the Fuel-prepad bridge).
///
/// What stays: **Op::Copy** — D2H byte-buffer transfer, a
/// Fuel-specific cross-device path that lives at this layer rather
/// than in any kernel crate.
#[cfg(feature = "cuda")]
pub fn register_cuda_kernels(table: &mut KernelBindingTable) {
    use OpKind::*;
    let cuda = BackendId::Cuda;

    // Op::Copy D2H — register at `(OpKind::Copy, [dt, dt], Cuda)` for
    // every dtype the byte-storage substrate supports. Source-backend
    // key (= Cuda); the wrapper produces a CPU output, copying through
    // `CudaStorageBytes::to_cpu_bytes`. Bridge-retirement Phase 2.
    let copy_dtypes = [
        DType::F32, DType::BF16, DType::F16, DType::U32,
        DType::F64, DType::U8, DType::I16, DType::I32, DType::I64,
    ];
    for dt in copy_dtypes {
        table.register(Copy, &[dt, dt], cuda, copy_from_cuda_wrapper);
    }
}

// =============================================================================
// Phase 7.5 B5+ — process-wide singleton (CapabilityRegistry + KernelBindingTable)
// =============================================================================

/// Process-wide [`CapabilityRegistry`]. Initialized on first access
/// via [`global_registry`]; the CPU backend is auto-registered
/// always (universal fallback). Other backends register themselves
/// during their initialization or app startup.
///
/// Tests that need a private registry should construct one
/// directly with `CapabilityRegistry::new()` rather than touch the
/// global.
static GLOBAL_REGISTRY: OnceLock<RwLock<CapabilityRegistry>> = OnceLock::new();

/// Process-wide topology generation counter. Bumped whenever the
/// set of registered backends / loaded kernels changes — so a
/// [`fuel_core::topology::SystemTopology`] snapshot built at
/// generation N self-invalidates the next time a consumer asks for
/// a fresh view and the counter is N+1 or greater. See the
/// system-topology session prompt for the contract.
///
/// The two existing global-registry mutation sites
/// ([`register_backend_capabilities`] and [`extend_global_bindings`])
/// bump this counter unconditionally. Tests and a future device-loss
/// detector hook in via [`bump_topology_generation`].
static TOPOLOGY_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Read the current topology generation. Used by SystemTopology's
/// `current()` to decide whether its cached snapshot is still valid.
pub fn topology_generation() -> u64 {
    TOPOLOGY_GENERATION.load(Ordering::Acquire)
}

/// Bump the topology generation counter. Every component that can
/// change the visible set of backends or devices should call this
/// after the mutation lands so consumers see a self-healing
/// `SystemTopology::current()` next call. Cheap (one atomic add);
/// `Release` ordering pairs with the `Acquire` load in
/// `topology_generation`.
pub fn bump_topology_generation() {
    TOPOLOGY_GENERATION.fetch_add(1, Ordering::Release);
}

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
        Copy,
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
        storage_substrate: SubstrateClass::HostBytes,
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
    bump_topology_generation();
}

/// Read-lock the process-wide kernel-binding table. CPU dispatch
/// wrappers are auto-registered on first access. When built with the
/// `cuda` feature, the CUDA PTX path + the baracuda-kernels-sys path
/// are also auto-registered — production callers picking up the
/// global table see all available backends without manual init.
pub fn global_bindings() -> std::sync::RwLockReadGuard<'static, KernelBindingTable> {
    GLOBAL_BINDINGS
        .get_or_init(|| {
            let mut t = KernelBindingTable::new();
            register_cpu_kernels(&mut t);
            register_optional_backends(&mut t);
            RwLock::new(t)
        })
        .read()
        .unwrap()
}

/// Auto-register every cargo-feature-gated backend that built. CPU is
/// always present (registered above); CUDA and Vulkan paths are
/// conditionally added. Future Metal / etc. paths hook in here.
fn register_optional_backends(table: &mut KernelBindingTable) {
    #[cfg(feature = "cuda")]
    {
        register_cuda_kernels(table);
        crate::baracuda_dispatch::register_baracuda_cuda_kernels(table);
    }
    #[cfg(feature = "vulkan")]
    {
        crate::vulkan_dispatch::register_vulkan_kernels(table);
    }
    let _ = table;
}

/// Add a backend's dispatch wrappers to the process-wide binding
/// table. Each backend exposes a `register_*_kernels(table)`
/// function (see [`register_cpu_kernels`]); per-backend init paths
/// call this to plug their wrappers into the global table.
pub fn extend_global_bindings(register: impl FnOnce(&mut KernelBindingTable)) {
    let lock = GLOBAL_BINDINGS.get_or_init(|| {
        let mut t = KernelBindingTable::new();
        register_cpu_kernels(&mut t);
        register_optional_backends(&mut t);
        RwLock::new(t)
    });
    register(&mut lock.write().unwrap());
    bump_topology_generation();
}

/// Phase 7.6 step 6 — register the always-built fused-op kernels into
/// the [`crate::fused::FusedKernelRegistry`]. Called by
/// [`crate::fused::default_kernel_registry`]; kept here so the
/// crate-private CPU dispatch wrappers stay co-located with their
/// registration.
///
/// Today's coverage (Phase 7.6 step 6 + backward-helper follow-up — 2026-05-11):
/// - `FUSED_LINEAR` × `Cpu` × {F32, F64, BF16, F16} — 4 impls
/// - `CONV2D` × `Cpu` × {F32, F64, BF16, F16} × {no-bias, with-bias} — 8 impls
/// - `SOFTMAX_LAST_DIM` × `Cpu` × {F32, F64, BF16, F16} — 4 impls
/// - `RMS_NORM_LAST_DIM` × `Cpu` × {F32, F64, BF16, F16} — 4 impls
/// - `LAYER_NORM_LAST_DIM` × `Cpu` × {F32, F64, BF16, F16} — 4 impls
/// - `ROPE` × `Cpu` × {F32, F64, BF16, F16} — 4 impls
/// - `CONV_TRANSPOSE2D` × `Cpu` × {F32, F64, BF16, F16} × {no-bias, with-bias} — 8 impls
/// - `FLASH_ATTN` × `Cpu` × {F32, F64, BF16, F16} × {no-alibi, with-alibi} — 8 impls
/// - `FLASH_ATTN_BACKWARD_{Q,K,V}` × `Cpu` × {F32, F64, BF16, F16} × {no-alibi, with-alibi} — 24 impls
/// - `PAGED_ATTN` × `Cpu` × {F32, F64, BF16, F16} × {no-alibi, with-alibi} — 8 impls
/// - `QMATMUL` × `Cpu` × {F32 activations + U32 weights} — 1 impl
/// - `SOFTMAX_LAST_DIM_BACKWARD` × `Cpu` × {F32, F64, BF16, F16} — 4 impls
/// - `LAYER_NORM_LAST_DIM_BACKWARD` × `Cpu` × {F32, F64, BF16, F16} — 4 impls
/// - `RMS_NORM_LAST_DIM_BACKWARD` × `Cpu` × {F32, F64, BF16, F16} — 4 impls
/// - `REDUCE_MAX_TO_BACKWARD` × `Cpu` × {F32, F64, BF16, F16} — 4 impls
/// - plus the later arrivals registered below: `POWI_BACKWARD`,
///   `INPLACE_AFFINE`, `FUSED_SOFTMAX_CROSS_ENTROPY`, `CAUSAL_CONV1D`,
///   `SELECTIVE_SCAN`, `SSD_CHUNK_SCAN` (4 impls each) and
///   `NF4_MATMUL` (3 impls).
///
/// Total: 120 CPU BackendImpls registered across **all 24** registered
/// fused ops. The architecture v1.0 §05 bit-stable coverage
/// commitment is now compiler-enforced for the full fused-op set
/// (no `KNOWN_GAPS` allowlist in the step-7 lint).
///
/// Backend crates (fuel-cuda-backend, fuel-vulkan-backend) extend by
/// composing against the registry from their own startup paths or via
/// the step-9 binding-table refactor.
pub fn register_default_fused_kernels(r: &mut crate::fused::FusedKernelRegistry) {
    use crate::fused::{
        cost_attn_backward_cpu, cost_attn_cpu, cost_causal_conv1d_cpu,
        cost_conv2d_cpu,
        cost_conv_transpose2d_cpu, cost_fused_linear_cpu,
        cost_fused_softmax_cross_entropy_cpu,
        cost_inplace_affine_cpu, cost_nf4_matmul_cpu,
        cost_norm_family_cpu, cost_powi_backward_cpu,
        cost_qmatmul_cpu, cost_reduce_max_to_backward_cpu, cost_rope_cpu,
        cost_selective_scan_cpu, cost_ssd_chunk_scan_cpu,
        ATTN_BACKWARD_CPU_PRECISION,
        ATTN_CPU_PRECISION, CAUSAL_CONV1D_CPU_PRECISION,
        CONV2D_CPU_PRECISION,
        CONV_TRANSPOSE2D_CPU_PRECISION, FUSED_LINEAR_CPU_PRECISION,
        FUSED_SOFTMAX_CROSS_ENTROPY_CPU_PRECISION,
        INPLACE_AFFINE_CPU_PRECISION, NF4_MATMUL_CPU_PRECISION,
        NORM_FAMILY_CPU_PRECISION,
        POWI_BACKWARD_CPU_PRECISION,
        QMATMUL_CPU_PRECISION, REDUCE_MAX_TO_BACKWARD_CPU_PRECISION,
        ROPE_CPU_PRECISION, SELECTIVE_SCAN_CPU_PRECISION,
        SSD_CHUNK_SCAN_CPU_PRECISION,
    };
    use crate::register_fused;
    use fuel_graph::registry::FusedOps;

    // Dtype tuples mirror the binding-table shape:
    //   FusedLinear: (lhs, rhs, bias, out) — all four agree.
    const FL_F32:  &[DType] = &[DType::F32,  DType::F32,  DType::F32,  DType::F32];
    const FL_F64:  &[DType] = &[DType::F64,  DType::F64,  DType::F64,  DType::F64];
    const FL_BF16: &[DType] = &[DType::BF16, DType::BF16, DType::BF16, DType::BF16];
    const FL_F16:  &[DType] = &[DType::F16,  DType::F16,  DType::F16,  DType::F16];

    // Conv2D: two shapes per dtype — no-bias (x, w, out) and
    // with-bias (x, w, bias, out). The CPU wrapper handles both.
    const CV_F32_NOB:  &[DType] = &[DType::F32,  DType::F32,  DType::F32];
    const CV_F32_BIAS: &[DType] = &[DType::F32,  DType::F32,  DType::F32,  DType::F32];
    const CV_F64_NOB:  &[DType] = &[DType::F64,  DType::F64,  DType::F64];
    const CV_F64_BIAS: &[DType] = &[DType::F64,  DType::F64,  DType::F64,  DType::F64];
    const CV_BF16_NOB:  &[DType] = &[DType::BF16, DType::BF16, DType::BF16];
    const CV_BF16_BIAS: &[DType] = &[DType::BF16, DType::BF16, DType::BF16, DType::BF16];
    const CV_F16_NOB:  &[DType] = &[DType::F16,  DType::F16,  DType::F16];
    const CV_F16_BIAS: &[DType] = &[DType::F16,  DType::F16,  DType::F16,  DType::F16];

    // Unary (in, out) — used by Softmax/RmsNorm/LayerNorm.
    const UNARY_F32:  &[DType] = &[DType::F32,  DType::F32];
    const UNARY_F64:  &[DType] = &[DType::F64,  DType::F64];
    const UNARY_BF16: &[DType] = &[DType::BF16, DType::BF16];
    const UNARY_F16:  &[DType] = &[DType::F16,  DType::F16];

    // Rope (x, cos, sin, out) — all four dtypes agree.
    const ROPE_F32:  &[DType] = &[DType::F32,  DType::F32,  DType::F32,  DType::F32];
    const ROPE_F64:  &[DType] = &[DType::F64,  DType::F64,  DType::F64,  DType::F64];
    const ROPE_BF16: &[DType] = &[DType::BF16, DType::BF16, DType::BF16, DType::BF16];
    const ROPE_F16:  &[DType] = &[DType::F16,  DType::F16,  DType::F16,  DType::F16];

    // FlashAttn: (q, k, v, [alibi], out) — no-alibi 4-tuple,
    // with-alibi 5-tuple. Same wrapper handles both.
    const FA_F32_NOA:  &[DType] = &[DType::F32,  DType::F32,  DType::F32,  DType::F32];
    const FA_F32_A:    &[DType] = &[DType::F32,  DType::F32,  DType::F32,  DType::F32,  DType::F32];
    const FA_F64_NOA:  &[DType] = &[DType::F64,  DType::F64,  DType::F64,  DType::F64];
    const FA_F64_A:    &[DType] = &[DType::F64,  DType::F64,  DType::F64,  DType::F64,  DType::F64];
    const FA_BF16_NOA: &[DType] = &[DType::BF16, DType::BF16, DType::BF16, DType::BF16];
    const FA_BF16_A:   &[DType] = &[DType::BF16, DType::BF16, DType::BF16, DType::BF16, DType::BF16];
    const FA_F16_NOA:  &[DType] = &[DType::F16,  DType::F16,  DType::F16,  DType::F16];
    const FA_F16_A:    &[DType] = &[DType::F16,  DType::F16,  DType::F16,  DType::F16,  DType::F16];

    // FlashAttnBackward{Q,K,V}: (q, k, v, do, [alibi], out) —
    // no-alibi 5-tuple, with-alibi 6-tuple. Mirrors the binding-table
    // `fa_bw_no_alibi` / `fa_bw_with_alibi` key shapes.
    const FAB_F32_NOA:  &[DType] = &[DType::F32,  DType::F32,  DType::F32,  DType::F32,  DType::F32];
    const FAB_F32_A:    &[DType] = &[DType::F32,  DType::F32,  DType::F32,  DType::F32,  DType::F32,  DType::F32];
    const FAB_F64_NOA:  &[DType] = &[DType::F64,  DType::F64,  DType::F64,  DType::F64,  DType::F64];
    const FAB_F64_A:    &[DType] = &[DType::F64,  DType::F64,  DType::F64,  DType::F64,  DType::F64,  DType::F64];
    const FAB_BF16_NOA: &[DType] = &[DType::BF16, DType::BF16, DType::BF16, DType::BF16, DType::BF16];
    const FAB_BF16_A:   &[DType] = &[DType::BF16, DType::BF16, DType::BF16, DType::BF16, DType::BF16, DType::BF16];
    const FAB_F16_NOA:  &[DType] = &[DType::F16,  DType::F16,  DType::F16,  DType::F16,  DType::F16];
    const FAB_F16_A:    &[DType] = &[DType::F16,  DType::F16,  DType::F16,  DType::F16,  DType::F16,  DType::F16];

    // PagedAttn: (q, kc, vc, bt:U32, cl:U32, [alibi], out).
    const PA_F32_NOA:  &[DType] = &[DType::F32,  DType::F32,  DType::F32,  DType::U32, DType::U32, DType::F32];
    const PA_F32_A:    &[DType] = &[DType::F32,  DType::F32,  DType::F32,  DType::U32, DType::U32, DType::F32,  DType::F32];
    const PA_F64_NOA:  &[DType] = &[DType::F64,  DType::F64,  DType::F64,  DType::U32, DType::U32, DType::F64];
    const PA_F64_A:    &[DType] = &[DType::F64,  DType::F64,  DType::F64,  DType::U32, DType::U32, DType::F64,  DType::F64];
    const PA_BF16_NOA: &[DType] = &[DType::BF16, DType::BF16, DType::BF16, DType::U32, DType::U32, DType::BF16];
    const PA_BF16_A:   &[DType] = &[DType::BF16, DType::BF16, DType::BF16, DType::U32, DType::U32, DType::BF16, DType::BF16];
    const PA_F16_NOA:  &[DType] = &[DType::F16,  DType::F16,  DType::F16,  DType::U32, DType::U32, DType::F16];
    const PA_F16_A:    &[DType] = &[DType::F16,  DType::F16,  DType::F16,  DType::U32, DType::U32, DType::F16,  DType::F16];

    // QMatMul: (a:F32 activations, w_q:U32 bytes, out:F32). Only F32
    // is wired today.
    const QM_F32: &[DType] = &[DType::F32, DType::U32, DType::F32];

    // Backward helpers — `[T, T, T]` for the binary (input0, input1, out)
    // shape. SoftmaxBackward, LayerNormBackward, RmsNormBackward,
    // ReduceMaxToBackward all share this dtype-tuple structure.
    const BW_F32:  &[DType] = &[DType::F32,  DType::F32,  DType::F32];
    const BW_F64:  &[DType] = &[DType::F64,  DType::F64,  DType::F64];
    const BW_BF16: &[DType] = &[DType::BF16, DType::BF16, DType::BF16];
    const BW_F16:  &[DType] = &[DType::F16,  DType::F16,  DType::F16];

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

    // Conv2D — eight registrations: {F32,F64,BF16,F16} × {no-bias, with-bias}.
    // The same wrapper handles both shapes; the dtype tuple distinguishes
    // them in the kernel registry so the route picker matches per-input-count.
    register_fused!(r, FusedOps::CONV2D, cpu, CV_F32_NOB,
        conv2d_f32_cpu_wrapper,
        cost = cost_conv2d_cpu,
        precision = CONV2D_CPU_PRECISION);
    register_fused!(r, FusedOps::CONV2D, cpu, CV_F32_BIAS,
        conv2d_f32_cpu_wrapper,
        cost = cost_conv2d_cpu,
        precision = CONV2D_CPU_PRECISION);
    register_fused!(r, FusedOps::CONV2D, cpu, CV_F64_NOB,
        conv2d_f64_cpu_wrapper,
        cost = cost_conv2d_cpu,
        precision = CONV2D_CPU_PRECISION);
    register_fused!(r, FusedOps::CONV2D, cpu, CV_F64_BIAS,
        conv2d_f64_cpu_wrapper,
        cost = cost_conv2d_cpu,
        precision = CONV2D_CPU_PRECISION);
    register_fused!(r, FusedOps::CONV2D, cpu, CV_BF16_NOB,
        conv2d_bf16_cpu_wrapper,
        cost = cost_conv2d_cpu,
        precision = CONV2D_CPU_PRECISION);
    register_fused!(r, FusedOps::CONV2D, cpu, CV_BF16_BIAS,
        conv2d_bf16_cpu_wrapper,
        cost = cost_conv2d_cpu,
        precision = CONV2D_CPU_PRECISION);
    register_fused!(r, FusedOps::CONV2D, cpu, CV_F16_NOB,
        conv2d_f16_cpu_wrapper,
        cost = cost_conv2d_cpu,
        precision = CONV2D_CPU_PRECISION);
    register_fused!(r, FusedOps::CONV2D, cpu, CV_F16_BIAS,
        conv2d_f16_cpu_wrapper,
        cost = cost_conv2d_cpu,
        precision = CONV2D_CPU_PRECISION);

    // Phase 7.6 step 6 (continued): SoftmaxLastDim × 4 dtypes.
    register_fused!(r, FusedOps::SOFTMAX_LAST_DIM, cpu, UNARY_F32,
        softmax_last_dim_f32_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::SOFTMAX_LAST_DIM, cpu, UNARY_F64,
        softmax_last_dim_f64_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::SOFTMAX_LAST_DIM, cpu, UNARY_BF16,
        softmax_last_dim_bf16_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::SOFTMAX_LAST_DIM, cpu, UNARY_F16,
        softmax_last_dim_f16_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);

    // RmsNormLastDim × 4 dtypes.
    register_fused!(r, FusedOps::RMS_NORM_LAST_DIM, cpu, UNARY_F32,
        rms_norm_last_dim_f32_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::RMS_NORM_LAST_DIM, cpu, UNARY_F64,
        rms_norm_last_dim_f64_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::RMS_NORM_LAST_DIM, cpu, UNARY_BF16,
        rms_norm_last_dim_bf16_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::RMS_NORM_LAST_DIM, cpu, UNARY_F16,
        rms_norm_last_dim_f16_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);

    // LayerNormLastDim × 4 dtypes.
    register_fused!(r, FusedOps::LAYER_NORM_LAST_DIM, cpu, UNARY_F32,
        layer_norm_last_dim_f32_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::LAYER_NORM_LAST_DIM, cpu, UNARY_F64,
        layer_norm_last_dim_f64_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::LAYER_NORM_LAST_DIM, cpu, UNARY_BF16,
        layer_norm_last_dim_bf16_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::LAYER_NORM_LAST_DIM, cpu, UNARY_F16,
        layer_norm_last_dim_f16_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);

    // Rope × 4 dtypes.
    register_fused!(r, FusedOps::ROPE, cpu, ROPE_F32,
        rope_f32_cpu_wrapper,
        cost = cost_rope_cpu,
        precision = ROPE_CPU_PRECISION);
    register_fused!(r, FusedOps::ROPE, cpu, ROPE_F64,
        rope_f64_cpu_wrapper,
        cost = cost_rope_cpu,
        precision = ROPE_CPU_PRECISION);
    register_fused!(r, FusedOps::ROPE, cpu, ROPE_BF16,
        rope_bf16_cpu_wrapper,
        cost = cost_rope_cpu,
        precision = ROPE_CPU_PRECISION);
    register_fused!(r, FusedOps::ROPE, cpu, ROPE_F16,
        rope_f16_cpu_wrapper,
        cost = cost_rope_cpu,
        precision = ROPE_CPU_PRECISION);

    // ConvTranspose2D × 4 dtypes × {no-bias, with-bias}. The CPU
    // wrapper handles both — same dispatch pattern as Conv2D.
    register_fused!(r, FusedOps::CONV_TRANSPOSE2D, cpu, CV_F32_NOB,
        conv_transpose2d_f32_cpu_wrapper,
        cost = cost_conv_transpose2d_cpu,
        precision = CONV_TRANSPOSE2D_CPU_PRECISION);
    register_fused!(r, FusedOps::CONV_TRANSPOSE2D, cpu, CV_F32_BIAS,
        conv_transpose2d_f32_cpu_wrapper,
        cost = cost_conv_transpose2d_cpu,
        precision = CONV_TRANSPOSE2D_CPU_PRECISION);
    register_fused!(r, FusedOps::CONV_TRANSPOSE2D, cpu, CV_F64_NOB,
        conv_transpose2d_f64_cpu_wrapper,
        cost = cost_conv_transpose2d_cpu,
        precision = CONV_TRANSPOSE2D_CPU_PRECISION);
    register_fused!(r, FusedOps::CONV_TRANSPOSE2D, cpu, CV_F64_BIAS,
        conv_transpose2d_f64_cpu_wrapper,
        cost = cost_conv_transpose2d_cpu,
        precision = CONV_TRANSPOSE2D_CPU_PRECISION);
    register_fused!(r, FusedOps::CONV_TRANSPOSE2D, cpu, CV_BF16_NOB,
        conv_transpose2d_bf16_cpu_wrapper,
        cost = cost_conv_transpose2d_cpu,
        precision = CONV_TRANSPOSE2D_CPU_PRECISION);
    register_fused!(r, FusedOps::CONV_TRANSPOSE2D, cpu, CV_BF16_BIAS,
        conv_transpose2d_bf16_cpu_wrapper,
        cost = cost_conv_transpose2d_cpu,
        precision = CONV_TRANSPOSE2D_CPU_PRECISION);
    register_fused!(r, FusedOps::CONV_TRANSPOSE2D, cpu, CV_F16_NOB,
        conv_transpose2d_f16_cpu_wrapper,
        cost = cost_conv_transpose2d_cpu,
        precision = CONV_TRANSPOSE2D_CPU_PRECISION);
    register_fused!(r, FusedOps::CONV_TRANSPOSE2D, cpu, CV_F16_BIAS,
        conv_transpose2d_f16_cpu_wrapper,
        cost = cost_conv_transpose2d_cpu,
        precision = CONV_TRANSPOSE2D_CPU_PRECISION);

    // FlashAttn × 4 dtypes × {no-alibi, with-alibi}.
    register_fused!(r, FusedOps::FLASH_ATTN, cpu, FA_F32_NOA,
        flash_attn_f32_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN, cpu, FA_F32_A,
        flash_attn_f32_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN, cpu, FA_F64_NOA,
        flash_attn_f64_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN, cpu, FA_F64_A,
        flash_attn_f64_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN, cpu, FA_BF16_NOA,
        flash_attn_bf16_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN, cpu, FA_BF16_A,
        flash_attn_bf16_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN, cpu, FA_F16_NOA,
        flash_attn_f16_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN, cpu, FA_F16_A,
        flash_attn_f16_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);

    // FlashAttnBackward{Q,K,V} × 4 dtypes × {no-alibi, with-alibi}.
    // Reuses the binding-table dispatch wrappers — the CPU kernel
    // computes all three gradients each call and the wrapper persists
    // the one matching the OpKind; the cost model accounts for that.
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_Q, cpu, FAB_F32_NOA,
        flash_attn_backward_q_f32_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_Q, cpu, FAB_F32_A,
        flash_attn_backward_q_f32_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_Q, cpu, FAB_F64_NOA,
        flash_attn_backward_q_f64_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_Q, cpu, FAB_F64_A,
        flash_attn_backward_q_f64_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_Q, cpu, FAB_BF16_NOA,
        flash_attn_backward_q_bf16_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_Q, cpu, FAB_BF16_A,
        flash_attn_backward_q_bf16_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_Q, cpu, FAB_F16_NOA,
        flash_attn_backward_q_f16_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_Q, cpu, FAB_F16_A,
        flash_attn_backward_q_f16_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);

    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_K, cpu, FAB_F32_NOA,
        flash_attn_backward_k_f32_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_K, cpu, FAB_F32_A,
        flash_attn_backward_k_f32_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_K, cpu, FAB_F64_NOA,
        flash_attn_backward_k_f64_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_K, cpu, FAB_F64_A,
        flash_attn_backward_k_f64_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_K, cpu, FAB_BF16_NOA,
        flash_attn_backward_k_bf16_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_K, cpu, FAB_BF16_A,
        flash_attn_backward_k_bf16_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_K, cpu, FAB_F16_NOA,
        flash_attn_backward_k_f16_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_K, cpu, FAB_F16_A,
        flash_attn_backward_k_f16_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);

    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_V, cpu, FAB_F32_NOA,
        flash_attn_backward_v_f32_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_V, cpu, FAB_F32_A,
        flash_attn_backward_v_f32_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_V, cpu, FAB_F64_NOA,
        flash_attn_backward_v_f64_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_V, cpu, FAB_F64_A,
        flash_attn_backward_v_f64_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_V, cpu, FAB_BF16_NOA,
        flash_attn_backward_v_bf16_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_V, cpu, FAB_BF16_A,
        flash_attn_backward_v_bf16_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_V, cpu, FAB_F16_NOA,
        flash_attn_backward_v_f16_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::FLASH_ATTN_BACKWARD_V, cpu, FAB_F16_A,
        flash_attn_backward_v_f16_cpu_wrapper,
        cost = cost_attn_backward_cpu,
        precision = ATTN_BACKWARD_CPU_PRECISION);

    // PagedAttn × 4 dtypes × {no-alibi, with-alibi}.
    register_fused!(r, FusedOps::PAGED_ATTN, cpu, PA_F32_NOA,
        paged_attn_f32_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);
    register_fused!(r, FusedOps::PAGED_ATTN, cpu, PA_F32_A,
        paged_attn_f32_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);
    register_fused!(r, FusedOps::PAGED_ATTN, cpu, PA_F64_NOA,
        paged_attn_f64_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);
    register_fused!(r, FusedOps::PAGED_ATTN, cpu, PA_F64_A,
        paged_attn_f64_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);
    register_fused!(r, FusedOps::PAGED_ATTN, cpu, PA_BF16_NOA,
        paged_attn_bf16_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);
    register_fused!(r, FusedOps::PAGED_ATTN, cpu, PA_BF16_A,
        paged_attn_bf16_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);
    register_fused!(r, FusedOps::PAGED_ATTN, cpu, PA_F16_NOA,
        paged_attn_f16_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);
    register_fused!(r, FusedOps::PAGED_ATTN, cpu, PA_F16_A,
        paged_attn_f16_cpu_wrapper,
        cost = cost_attn_cpu,
        precision = ATTN_CPU_PRECISION);

    // QMatMul × F32 activations × U32 weights (1 impl — only F32 is
    // wired in the legacy executor today).
    register_fused!(r, FusedOps::QMATMUL, cpu, QM_F32,
        qmatmul_f32_cpu_wrapper,
        cost = cost_qmatmul_cpu,
        precision = QMATMUL_CPU_PRECISION);

    // Phase 7.6 step 6 follow-up — backward helpers gain CPU
    // BackendImpls now that byte-level wrappers exist. Each takes
    // 2 inputs + 1 output, dtype tuple `[T, T, T]`. Softmax /
    // Layer / Rms backwards share `cost_norm_family_cpu` +
    // `NORM_FAMILY_CPU_PRECISION` (same outer × last_dim shape as
    // their forward); ReduceMaxToBackward has its own (5-pass
    // recomputed-max + tie-share gate).
    register_fused!(r, FusedOps::SOFTMAX_LAST_DIM_BACKWARD, cpu, BW_F32,
        softmax_last_dim_backward_f32_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::SOFTMAX_LAST_DIM_BACKWARD, cpu, BW_F64,
        softmax_last_dim_backward_f64_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::SOFTMAX_LAST_DIM_BACKWARD, cpu, BW_BF16,
        softmax_last_dim_backward_bf16_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::SOFTMAX_LAST_DIM_BACKWARD, cpu, BW_F16,
        softmax_last_dim_backward_f16_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);

    register_fused!(r, FusedOps::LAYER_NORM_LAST_DIM_BACKWARD, cpu, BW_F32,
        layer_norm_last_dim_backward_f32_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::LAYER_NORM_LAST_DIM_BACKWARD, cpu, BW_F64,
        layer_norm_last_dim_backward_f64_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::LAYER_NORM_LAST_DIM_BACKWARD, cpu, BW_BF16,
        layer_norm_last_dim_backward_bf16_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::LAYER_NORM_LAST_DIM_BACKWARD, cpu, BW_F16,
        layer_norm_last_dim_backward_f16_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);

    register_fused!(r, FusedOps::RMS_NORM_LAST_DIM_BACKWARD, cpu, BW_F32,
        rms_norm_last_dim_backward_f32_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::RMS_NORM_LAST_DIM_BACKWARD, cpu, BW_F64,
        rms_norm_last_dim_backward_f64_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::RMS_NORM_LAST_DIM_BACKWARD, cpu, BW_BF16,
        rms_norm_last_dim_backward_bf16_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);
    register_fused!(r, FusedOps::RMS_NORM_LAST_DIM_BACKWARD, cpu, BW_F16,
        rms_norm_last_dim_backward_f16_cpu_wrapper,
        cost = cost_norm_family_cpu,
        precision = NORM_FAMILY_CPU_PRECISION);

    register_fused!(r, FusedOps::REDUCE_MAX_TO_BACKWARD, cpu, BW_F32,
        reduce_max_to_backward_f32_cpu_wrapper,
        cost = cost_reduce_max_to_backward_cpu,
        precision = REDUCE_MAX_TO_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::REDUCE_MAX_TO_BACKWARD, cpu, BW_F64,
        reduce_max_to_backward_f64_cpu_wrapper,
        cost = cost_reduce_max_to_backward_cpu,
        precision = REDUCE_MAX_TO_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::REDUCE_MAX_TO_BACKWARD, cpu, BW_BF16,
        reduce_max_to_backward_bf16_cpu_wrapper,
        cost = cost_reduce_max_to_backward_cpu,
        precision = REDUCE_MAX_TO_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::REDUCE_MAX_TO_BACKWARD, cpu, BW_F16,
        reduce_max_to_backward_f16_cpu_wrapper,
        cost = cost_reduce_max_to_backward_cpu,
        precision = REDUCE_MAX_TO_BACKWARD_CPU_PRECISION);

    register_fused!(r, FusedOps::POWI_BACKWARD, cpu, BW_F32,
        powi_backward_f32_cpu_wrapper,
        cost = cost_powi_backward_cpu,
        precision = POWI_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::POWI_BACKWARD, cpu, BW_F64,
        powi_backward_f64_cpu_wrapper,
        cost = cost_powi_backward_cpu,
        precision = POWI_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::POWI_BACKWARD, cpu, BW_BF16,
        powi_backward_bf16_cpu_wrapper,
        cost = cost_powi_backward_cpu,
        precision = POWI_BACKWARD_CPU_PRECISION);
    register_fused!(r, FusedOps::POWI_BACKWARD, cpu, BW_F16,
        powi_backward_f16_cpu_wrapper,
        cost = cost_powi_backward_cpu,
        precision = POWI_BACKWARD_CPU_PRECISION);

    // INPLACE_AFFINE — `x = mul · x + add`, single-input + same-dtype
    // output. The binding-table key shape is `[T, T]` (mirrors the
    // non-inplace Affine OpKind so `build_lookup_dtypes` produces the
    // same canonical key). 4 dtypes: f32, f64, bf16, f16.
    register_fused!(r, FusedOps::INPLACE_AFFINE, cpu, UNARY_F32,
        inplace_affine_f32_cpu_wrapper,
        cost = cost_inplace_affine_cpu,
        precision = INPLACE_AFFINE_CPU_PRECISION);
    register_fused!(r, FusedOps::INPLACE_AFFINE, cpu, UNARY_F64,
        inplace_affine_f64_cpu_wrapper,
        cost = cost_inplace_affine_cpu,
        precision = INPLACE_AFFINE_CPU_PRECISION);
    register_fused!(r, FusedOps::INPLACE_AFFINE, cpu, UNARY_BF16,
        inplace_affine_bf16_cpu_wrapper,
        cost = cost_inplace_affine_cpu,
        precision = INPLACE_AFFINE_CPU_PRECISION);
    register_fused!(r, FusedOps::INPLACE_AFFINE, cpu, UNARY_F16,
        inplace_affine_f16_cpu_wrapper,
        cost = cost_inplace_affine_cpu,
        precision = INPLACE_AFFINE_CPU_PRECISION);

    // FUSED_SOFTMAX_CROSS_ENTROPY — three-tuple (logits T, targets
    // I64, out F32). T ∈ {F32, F64, BF16, F16}; output dtype stays
    // F32 across all variants (the FSCE declared dtype — losses
    // accumulate in F64 and narrow to F32). The cost model + precision
    // guarantee are shared: per-row work is dtype-agnostic and the F64
    // accumulator gives the same precision contract for every T.
    const FSCE_F32:  &[DType] = &[DType::F32,  DType::I64, DType::F32];
    const FSCE_F64:  &[DType] = &[DType::F64,  DType::I64, DType::F32];
    const FSCE_BF16: &[DType] = &[DType::BF16, DType::I64, DType::F32];
    const FSCE_F16:  &[DType] = &[DType::F16,  DType::I64, DType::F32];
    register_fused!(r, FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY, cpu, FSCE_F32,
        fused_softmax_cross_entropy_f32_cpu_wrapper,
        cost = cost_fused_softmax_cross_entropy_cpu,
        precision = FUSED_SOFTMAX_CROSS_ENTROPY_CPU_PRECISION);
    register_fused!(r, FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY, cpu, FSCE_F64,
        fused_softmax_cross_entropy_f64_cpu_wrapper,
        cost = cost_fused_softmax_cross_entropy_cpu,
        precision = FUSED_SOFTMAX_CROSS_ENTROPY_CPU_PRECISION);
    register_fused!(r, FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY, cpu, FSCE_BF16,
        fused_softmax_cross_entropy_bf16_cpu_wrapper,
        cost = cost_fused_softmax_cross_entropy_cpu,
        precision = FUSED_SOFTMAX_CROSS_ENTROPY_CPU_PRECISION);
    register_fused!(r, FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY, cpu, FSCE_F16,
        fused_softmax_cross_entropy_f16_cpu_wrapper,
        cost = cost_fused_softmax_cross_entropy_cpu,
        precision = FUSED_SOFTMAX_CROSS_ENTROPY_CPU_PRECISION);

    // CAUSAL_CONV1D — four-tuple (x, weight, bias, out), 4 dtype
    // variants. F32/F64 accumulate natively; F16/BF16 use F32
    // accumulator + narrow on store.
    const CC1D_F32:  &[DType] = &[DType::F32,  DType::F32,  DType::F32,  DType::F32];
    const CC1D_F64:  &[DType] = &[DType::F64,  DType::F64,  DType::F64,  DType::F64];
    const CC1D_BF16: &[DType] = &[DType::BF16, DType::BF16, DType::BF16, DType::BF16];
    const CC1D_F16:  &[DType] = &[DType::F16,  DType::F16,  DType::F16,  DType::F16];
    register_fused!(r, FusedOps::CAUSAL_CONV1D, cpu, CC1D_F32,
        causal_conv1d_f32_cpu_wrapper,
        cost = cost_causal_conv1d_cpu,
        precision = CAUSAL_CONV1D_CPU_PRECISION);
    register_fused!(r, FusedOps::CAUSAL_CONV1D, cpu, CC1D_F64,
        causal_conv1d_f64_cpu_wrapper,
        cost = cost_causal_conv1d_cpu,
        precision = CAUSAL_CONV1D_CPU_PRECISION);
    register_fused!(r, FusedOps::CAUSAL_CONV1D, cpu, CC1D_BF16,
        causal_conv1d_bf16_cpu_wrapper,
        cost = cost_causal_conv1d_cpu,
        precision = CAUSAL_CONV1D_CPU_PRECISION);
    register_fused!(r, FusedOps::CAUSAL_CONV1D, cpu, CC1D_F16,
        causal_conv1d_f16_cpu_wrapper,
        cost = cost_causal_conv1d_cpu,
        precision = CAUSAL_CONV1D_CPU_PRECISION);

    // SELECTIVE_SCAN — six-tuple (u, delta, a, b, c, out), 4 dtype
    // variants. F64 accumulator regardless of T; F16/BF16 narrow on
    // store; F32/F64 lossless.
    const SS_F32:  &[DType] = &[DType::F32,  DType::F32,  DType::F32,  DType::F32,  DType::F32,  DType::F32];
    const SS_F64:  &[DType] = &[DType::F64,  DType::F64,  DType::F64,  DType::F64,  DType::F64,  DType::F64];
    const SS_BF16: &[DType] = &[DType::BF16, DType::BF16, DType::BF16, DType::BF16, DType::BF16, DType::BF16];
    const SS_F16:  &[DType] = &[DType::F16,  DType::F16,  DType::F16,  DType::F16,  DType::F16,  DType::F16];
    register_fused!(r, FusedOps::SELECTIVE_SCAN, cpu, SS_F32,
        selective_scan_f32_cpu_wrapper,
        cost = cost_selective_scan_cpu,
        precision = SELECTIVE_SCAN_CPU_PRECISION);
    register_fused!(r, FusedOps::SELECTIVE_SCAN, cpu, SS_F64,
        selective_scan_f64_cpu_wrapper,
        cost = cost_selective_scan_cpu,
        precision = SELECTIVE_SCAN_CPU_PRECISION);
    register_fused!(r, FusedOps::SELECTIVE_SCAN, cpu, SS_BF16,
        selective_scan_bf16_cpu_wrapper,
        cost = cost_selective_scan_cpu,
        precision = SELECTIVE_SCAN_CPU_PRECISION);
    register_fused!(r, FusedOps::SELECTIVE_SCAN, cpu, SS_F16,
        selective_scan_f16_cpu_wrapper,
        cost = cost_selective_scan_cpu,
        precision = SELECTIVE_SCAN_CPU_PRECISION);

    // SSD_CHUNK_SCAN — six-tuple (x, dt, a, b, c, out), 4 dtype
    // variants. v1 single-chunk only (chunk_size == seqlen). F64
    // accumulator regardless of T; F16/BF16 narrow on store.
    const SCS_F32:  &[DType] = &[DType::F32,  DType::F32,  DType::F32,  DType::F32,  DType::F32,  DType::F32];
    const SCS_F64:  &[DType] = &[DType::F64,  DType::F64,  DType::F64,  DType::F64,  DType::F64,  DType::F64];
    const SCS_BF16: &[DType] = &[DType::BF16, DType::BF16, DType::BF16, DType::BF16, DType::BF16, DType::BF16];
    const SCS_F16:  &[DType] = &[DType::F16,  DType::F16,  DType::F16,  DType::F16,  DType::F16,  DType::F16];
    register_fused!(r, FusedOps::SSD_CHUNK_SCAN, cpu, SCS_F32,
        ssd_chunk_scan_f32_cpu_wrapper,
        cost = cost_ssd_chunk_scan_cpu,
        precision = SSD_CHUNK_SCAN_CPU_PRECISION);
    register_fused!(r, FusedOps::SSD_CHUNK_SCAN, cpu, SCS_F64,
        ssd_chunk_scan_f64_cpu_wrapper,
        cost = cost_ssd_chunk_scan_cpu,
        precision = SSD_CHUNK_SCAN_CPU_PRECISION);
    register_fused!(r, FusedOps::SSD_CHUNK_SCAN, cpu, SCS_BF16,
        ssd_chunk_scan_bf16_cpu_wrapper,
        cost = cost_ssd_chunk_scan_cpu,
        precision = SSD_CHUNK_SCAN_CPU_PRECISION);
    register_fused!(r, FusedOps::SSD_CHUNK_SCAN, cpu, SCS_F16,
        ssd_chunk_scan_f16_cpu_wrapper,
        cost = cost_ssd_chunk_scan_cpu,
        precision = SSD_CHUNK_SCAN_CPU_PRECISION);

    // NF4_MATMUL — four-tuple (activations T, w_packed U8, absmax
    // F32, out T), 3 dtype variants (T ∈ {F32, F16, BF16}).
    const NF4_F32:  &[DType] = &[DType::F32,  DType::U8, DType::F32, DType::F32];
    const NF4_F16:  &[DType] = &[DType::F16,  DType::U8, DType::F32, DType::F16];
    const NF4_BF16: &[DType] = &[DType::BF16, DType::U8, DType::F32, DType::BF16];
    register_fused!(r, FusedOps::NF4_MATMUL, cpu, NF4_F32,
        nf4_matmul_f32_cpu_wrapper,
        cost = cost_nf4_matmul_cpu,
        precision = NF4_MATMUL_CPU_PRECISION);
    register_fused!(r, FusedOps::NF4_MATMUL, cpu, NF4_F16,
        nf4_matmul_f16_cpu_wrapper,
        cost = cost_nf4_matmul_cpu,
        precision = NF4_MATMUL_CPU_PRECISION);
    register_fused!(r, FusedOps::NF4_MATMUL, cpu, NF4_BF16,
        nf4_matmul_bf16_cpu_wrapper,
        cost = cost_nf4_matmul_cpu,
        precision = NF4_MATMUL_CPU_PRECISION);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 7.6 step 7b — the architecture v1.0 §05 "always-built
    /// backend bit-stable coverage commitment" lint, extended from
    /// the fused-op half (step 7a) to **primitive** ops.
    ///
    /// Every `OpKind` variant that flows through `KernelBindingTable`
    /// MUST have at least one CPU registration with
    /// `precision.bit_stable_on_same_hardware == true`. The
    /// architecture's correctness anchor is the always-built backend
    /// giving every downstream consumer a deterministic
    /// implementation to fall back on, so cross-backend equivalence
    /// tests have a fixed reference.
    ///
    /// The lint runs as a unit test so violations surface in CI
    /// rather than at runtime. Adding a new `OpKind` variant without
    /// a matching CPU registration fails this test immediately.
    ///
    /// As of step 7b (2026-05-11), every `OpKind` variant has CPU
    /// coverage via `register_cpu_kernels` + the
    /// `fill_unset_cpu_precision` pass at the end of that function.
    /// `KNOWN_GAPS` is empty.
    ///
    /// Note on the OpKind enumeration: there's no `strum`-style
    /// derive on the enum, so the test hardcodes the variant list.
    /// New OpKind variants must be added here AND to a matching
    /// CPU registration in `register_cpu_kernels`. The two-touch
    /// requirement is a feature — the test becomes the canonical
    /// "all kernels accounted for" reference.
    #[test]
    fn precision_guarantee_lint_bit_stable_cpu_coverage_primitives() {
        use fuel_ir::dispatch::OpKind;
        use fuel_ir::probe::BackendId;

        // Allowlist for OpKind variants that *can't* have bit-stable
        // CPU coverage. Empty as of step 7b (every variant routes
        // through deterministic kernels). New entries must come with
        // a documented reason — and a follow-up to close the gap.
        const KNOWN_GAPS: &[(OpKind, &str)] = &[];

        // The canonical list of every OpKind variant. Mirrors the
        // enum definition in fuel-core-types::dispatch. Adding a
        // new variant requires adding it here; that's the explicit
        // contract for "this OpKind exists and is expected to have
        // CPU coverage."
        const ALL_OP_KINDS: &[OpKind] = &[
            OpKind::MatMul,
            OpKind::AddElementwise, OpKind::SubElementwise,
            OpKind::MulElementwise, OpKind::DivElementwise,
            OpKind::ReluElementwise, OpKind::NegElementwise,
            OpKind::SqrElementwise, OpKind::SqrtElementwise,
            OpKind::RecipElementwise, OpKind::AbsElementwise,
            OpKind::TanhElementwise, OpKind::ExpElementwise,
            OpKind::LogElementwise, OpKind::SinElementwise,
            OpKind::CosElementwise, OpKind::SigmoidElementwise,
            OpKind::SiluElementwise, OpKind::GeluElementwise,
            OpKind::StepElementwise,
            OpKind::SumReduce, OpKind::MaxReduce,
            OpKind::MinReduce, OpKind::MeanReduce,
            OpKind::Cast,
            OpKind::Conv2D, OpKind::ConvTranspose2D,
            OpKind::ReduceSumTo, OpKind::ReduceMaxTo,
            OpKind::FusedLinear,
            OpKind::FlashAttn, OpKind::PagedAttn,
            OpKind::Affine, OpKind::ClampElementwise,
            OpKind::PowIElementwise,
            OpKind::MaximumElementwise, OpKind::MinimumElementwise,
            OpKind::EqualElementwise, OpKind::NotEqualElementwise,
            OpKind::LessElementwise, OpKind::LessEqualElementwise,
            OpKind::GreaterElementwise, OpKind::GreaterEqualElementwise,
            OpKind::Where,
            OpKind::FloorElementwise, OpKind::CeilElementwise,
            OpKind::RoundElementwise, OpKind::SignElementwise,
            OpKind::ErfElementwise, OpKind::GeluErfElementwise,
            OpKind::PowElementwise, OpKind::RsqrtElementwise,
            OpKind::RemElementwise,
            OpKind::Flip, OpKind::Roll, OpKind::CumSum,
            OpKind::Pad, OpKind::PadBackward,
            OpKind::Triu, OpKind::Tril,
            OpKind::LogSoftmaxLastDim, OpKind::LogSoftmaxLastDimBackward,
            OpKind::MaskedFill, OpKind::Concat,
            OpKind::SoftmaxLastDim, OpKind::SoftmaxLastDimBackward,
            OpKind::RmsNormLastDim, OpKind::RmsNormLastDimBackward,
            OpKind::LayerNormLastDim, OpKind::LayerNormLastDimBackward,
            OpKind::ReduceMaxToBackward,
            OpKind::IndexSelect, OpKind::Gather,
            OpKind::Rope,
            OpKind::IndexAdd, OpKind::ScatterAdd,
            OpKind::ArgMaxDim, OpKind::ArgMinDim,
            OpKind::QMatMul,
            OpKind::Copy,
            OpKind::FusedSoftmaxCrossEntropy,
            OpKind::CausalConv1d,
            OpKind::SelectiveScan,
            OpKind::SsdChunkScan,
            OpKind::Nf4Matmul,
        ];

        // Populate the binding table the same way the production
        // path does — via register_cpu_kernels including the
        // fill_unset_cpu_precision pass at the end.
        let mut table = KernelBindingTable::new();
        register_cpu_kernels(&mut table);

        // Group all CPU registrations by OpKind so we can check
        // coverage per variant.
        let mut by_op_kind: std::collections::HashMap<
            OpKind,
            Vec<crate::fused::PrecisionGuarantee>,
        > = std::collections::HashMap::new();
        for (op, _dtypes, backend, precision) in table.iter_precision() {
            if backend == BackendId::Cpu {
                by_op_kind.entry(op).or_default().push(precision);
            }
        }

        let mut failures: Vec<String> = Vec::new();
        let mut covered = 0usize;
        let mut allowlisted = 0usize;

        for op in ALL_OP_KINDS.iter().copied() {
            if let Some((_, reason)) = KNOWN_GAPS.iter().find(|(g, _)| *g == op) {
                allowlisted += 1;
                if by_op_kind.contains_key(&op) {
                    failures.push(format!(
                        "OpKind::{op:?} is on the KNOWN_GAPS allowlist but DOES \
                         have a CPU registration now. Reason given was: \
                         {reason:?}. Remove the allowlist entry to enable the \
                         bit-stable lint for this op.",
                    ));
                }
                continue;
            }
            let precisions = by_op_kind.get(&op);
            let has_bit_stable_cpu = precisions.is_some_and(|ps| {
                ps.iter().any(|p| p.bit_stable_on_same_hardware)
            });
            if has_bit_stable_cpu {
                covered += 1;
            } else {
                failures.push(format!(
                    "OpKind::{op:?} has no bit-stable CPU registration. \
                     Architecture v1.0 §05 requires the always-built backend \
                     (fuel-cpu-backend) to provide at least one \
                     `bit_stable_on_same_hardware: true` kernel per primitive \
                     op. Either add a CPU registration in \
                     register_cpu_kernels (the fill_unset_cpu_precision pass \
                     at the end will upgrade it to PRIMITIVE_DETERMINISTIC_CPU \
                     automatically), or add a line to KNOWN_GAPS above with a \
                     documented reason.",
                ));
            }
        }

        assert!(
            failures.is_empty(),
            "Architecture v1.0 bit-stable CPU coverage lint (primitives) failed:\n{}",
            failures.join("\n"),
        );
        assert_eq!(
            allowlisted, 0,
            "KNOWN_GAPS allowlist is empty by design; saw {allowlisted} allowlisted",
        );
        // Sanity: ALL_OP_KINDS should match the production
        // registration set. If a new variant lands in the enum
        // without being added here, the test's value is reduced —
        // catch the case by asserting the count.
        assert!(
            covered > 0,
            "lint covered 0 OpKinds — table appears empty",
        );
        // Sanity: every OpKind we enumerated should have ≥1 dtype
        // registration. The fill pass should have populated
        // precision for each.
        for op in ALL_OP_KINDS {
            assert!(
                by_op_kind.contains_key(op),
                "OpKind::{op:?} has no CPU registration at all — either add \
                 one in register_cpu_kernels or document the gap in \
                 KNOWN_GAPS.",
            );
        }
    }

    /// Phase 7.6 step 8 — the "every OpKind has a real cost
    /// function" coverage lint. Companion to step 7b's bit-stable
    /// precision lint; the architectural commitment is that every
    /// primitive op carries both a `PrecisionGuarantee` and a
    /// `CostFn` (Layer-1 cost model: FLOPs + bandwidth + launch
    /// overhead) by the time the binding table is consulted at
    /// dispatch time.
    ///
    /// The lint runs as a unit test. Violations surface in CI
    /// rather than at runtime — a new `OpKind` variant without a
    /// matching arm in
    /// [`crate::cost::default_cost_for_op_kind`] fails this test
    /// immediately because the fill pass leaves it bound to
    /// `unknown_cost`.
    ///
    /// `KNOWN_GAPS` is empty as of step 8 (2026-05-12). Adding a
    /// future variant without immediate cost coverage needs a
    /// documented allowlist entry AND a follow-up.
    #[test]
    fn cost_lint_per_op_kind_cpu_coverage() {
        use fuel_ir::dispatch::OpKind;
        use fuel_ir::probe::BackendId;

        const KNOWN_GAPS: &[(OpKind, &str)] = &[];

        const ALL_OP_KINDS: &[OpKind] = &[
            OpKind::MatMul,
            OpKind::AddElementwise, OpKind::SubElementwise,
            OpKind::MulElementwise, OpKind::DivElementwise,
            OpKind::ReluElementwise, OpKind::NegElementwise,
            OpKind::SqrElementwise, OpKind::SqrtElementwise,
            OpKind::RecipElementwise, OpKind::AbsElementwise,
            OpKind::TanhElementwise, OpKind::ExpElementwise,
            OpKind::LogElementwise, OpKind::SinElementwise,
            OpKind::CosElementwise, OpKind::SigmoidElementwise,
            OpKind::SiluElementwise, OpKind::GeluElementwise,
            OpKind::StepElementwise,
            OpKind::SumReduce, OpKind::MaxReduce,
            OpKind::MinReduce, OpKind::MeanReduce,
            OpKind::Cast,
            OpKind::Conv2D, OpKind::ConvTranspose2D,
            OpKind::ReduceSumTo, OpKind::ReduceMaxTo,
            OpKind::FusedLinear,
            OpKind::FlashAttn, OpKind::PagedAttn,
            OpKind::Affine, OpKind::ClampElementwise,
            OpKind::PowIElementwise,
            OpKind::MaximumElementwise, OpKind::MinimumElementwise,
            OpKind::EqualElementwise, OpKind::NotEqualElementwise,
            OpKind::LessElementwise, OpKind::LessEqualElementwise,
            OpKind::GreaterElementwise, OpKind::GreaterEqualElementwise,
            OpKind::Where,
            OpKind::FloorElementwise, OpKind::CeilElementwise,
            OpKind::RoundElementwise, OpKind::SignElementwise,
            OpKind::ErfElementwise, OpKind::GeluErfElementwise,
            OpKind::PowElementwise, OpKind::RsqrtElementwise,
            OpKind::RemElementwise,
            OpKind::Flip, OpKind::Roll, OpKind::CumSum,
            OpKind::Pad, OpKind::PadBackward,
            OpKind::Triu, OpKind::Tril,
            OpKind::LogSoftmaxLastDim, OpKind::LogSoftmaxLastDimBackward,
            OpKind::MaskedFill, OpKind::Concat,
            OpKind::SoftmaxLastDim, OpKind::SoftmaxLastDimBackward,
            OpKind::RmsNormLastDim, OpKind::RmsNormLastDimBackward,
            OpKind::LayerNormLastDim, OpKind::LayerNormLastDimBackward,
            OpKind::ReduceMaxToBackward,
            OpKind::IndexSelect, OpKind::Gather,
            OpKind::Rope,
            OpKind::IndexAdd, OpKind::ScatterAdd,
            OpKind::ArgMaxDim, OpKind::ArgMinDim,
            OpKind::QMatMul,
            OpKind::Copy,
            OpKind::FusedSoftmaxCrossEntropy,
            OpKind::CausalConv1d,
            OpKind::SelectiveScan,
            OpKind::SsdChunkScan,
            OpKind::Nf4Matmul,
        ];

        let mut table = KernelBindingTable::new();
        register_cpu_kernels(&mut table);

        // Group cost functions by OpKind for CPU registrations.
        let mut by_op_kind: std::collections::HashMap<
            OpKind,
            Vec<crate::kernel::CostFn>,
        > = std::collections::HashMap::new();
        for (op, _dtypes, backend, cost) in table.iter_cost() {
            if backend == BackendId::Cpu {
                by_op_kind.entry(op).or_default().push(cost);
            }
        }

        let unknown_sentinel = crate::kernel::unknown_cost as usize;

        let mut failures: Vec<String> = Vec::new();
        for op in ALL_OP_KINDS.iter().copied() {
            if let Some((_, reason)) = KNOWN_GAPS.iter().find(|(g, _)| *g == op) {
                if let Some(costs) = by_op_kind.get(&op) {
                    let has_real = costs.iter().any(|c| (*c as usize) != unknown_sentinel);
                    if has_real {
                        failures.push(format!(
                            "OpKind::{op:?} is on the cost-lint KNOWN_GAPS \
                             allowlist but DOES have a non-default cost fn \
                             now. Reason given: {reason:?}. Remove the \
                             allowlist entry.",
                        ));
                    }
                }
                continue;
            }
            let costs = by_op_kind.get(&op);
            let has_real_cost = costs.is_some_and(|cs| {
                cs.iter().any(|c| (*c as usize) != unknown_sentinel)
            });
            if !has_real_cost {
                failures.push(format!(
                    "OpKind::{op:?} has no non-default cost fn in any CPU \
                     registration. Either add an arm in \
                     `crate::cost::default_cost_for_op_kind` (the \
                     fill_unset_cpu_cost pass at the end of \
                     register_cpu_kernels will pick it up automatically), \
                     register an explicit cost via `register_full(...)`, \
                     or add a line to KNOWN_GAPS with a documented reason.",
                ));
            }
        }

        assert!(
            failures.is_empty(),
            "Phase 7.6 step 8 cost-coverage lint failed:\n{}",
            failures.join("\n"),
        );
        // Sanity: every OpKind we enumerated should have ≥1 CPU
        // registration (asserted separately by the step-7b lint,
        // but re-checked here for symmetry).
        for op in ALL_OP_KINDS {
            assert!(
                by_op_kind.contains_key(op),
                "OpKind::{op:?} has no CPU registration at all.",
            );
        }
    }

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
            storage_substrate: SubstrateClass::HostBytes,
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
            storage_substrate: SubstrateClass::CudaUntyped,
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

    /// F8E4M3 ↔ {F32, F16, BF16} casts (alpha.29 CastSubBytePlan sibling
    /// on the CPU side) are all reachable through the binding table.
    #[test]
    fn cpu_f8e4m3_cast_pairs_registered() {
        let mut table = KernelBindingTable::new();
        register_cpu_kernels(&mut table);
        for other in [DType::F32, DType::F16, DType::BF16] {
            for (src, dst) in [(DType::F8E4M3, other), (other, DType::F8E4M3)] {
                table
                    .lookup(OpKind::Cast, &[src, dst], BackendId::Cpu)
                    .unwrap_or_else(|e| panic!("Cast {src:?} → {dst:?} (CPU) not registered: {e}"));
            }
        }
    }

    /// Integer MatMul (i8 / u8) registrations land under the same
    /// `(MatMul, [T, T, T], Cpu)` key shape as the float variants.
    #[test]
    fn cpu_int_matmul_registered() {
        let mut table = KernelBindingTable::new();
        register_cpu_kernels(&mut table);
        for t in [DType::I8, DType::U8] {
            table
                .lookup(OpKind::MatMul, &[t, t, t], BackendId::Cpu)
                .unwrap_or_else(|e| panic!("MatMul {t:?} (CPU) not registered: {e}"));
        }
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
        let lhs = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let rhs = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0, 30.0, 40.0]);
        let out = fuel_memory::alloc_cpu_zeroed(DType::F32, 4).expect("alloc");

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
        let lhs = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0]);
        let out = fuel_memory::alloc_cpu_zeroed(DType::F32, 2).unwrap();
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

    /// When the `cuda` feature is on, `register_cuda_kernels` (the
    /// post-retirement residue: cuBLAS MatMul / ReduceTo / Copy /
    /// CausalConv1d) AND `register_baracuda_cuda_kernels` are both
    /// auto-registered into the global table. The PTX-duplicate
    /// unary/binary registrations were stripped in the
    /// fuel-cuda-kernels retirement (commit d9898fec), so elementwise
    /// keys carry exactly ONE alternative — baracuda's.
    #[cfg(feature = "cuda")]
    #[test]
    fn global_bindings_auto_registers_cuda_paths() {
        let b = global_bindings();
        // register_cuda_kernels residue: F32 MatMul via cuBLAS.
        let cublas = b.lookup(
            OpKind::MatMul,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cuda,
        );
        assert!(cublas.is_ok(), "cuBLAS F32 MatMul should be auto-registered on Cuda");

        // Baracuda path: int8 MatMul exists only via baracuda. If this
        // resolves, both paths fired.
        let baracuda_int8 = b.lookup(
            OpKind::MatMul,
            &[DType::I8, DType::I8, DType::I8],
            BackendId::Cuda,
        );
        assert!(
            baracuda_int8.is_ok(),
            "baracuda int8 MatMul should be auto-registered on Cuda",
        );

        // Post-strip invariant: F32 unary Neg resolves with exactly
        // one alternative (baracuda is the single source of truth for
        // CUDA elementwise; a second entry would mean a duplicate
        // registration crept back in).
        let alts = b.lookup_alternatives(
            OpKind::NegElementwise,
            &[DType::F32, DType::F32],
            BackendId::Cuda,
        );
        assert!(
            alts.len() == 1,
            "expected exactly 1 CUDA Neg F32 alternative (baracuda); got {}",
            alts.len(),
        );
    }

    /// Picker-alternatives audit harness — enumerates the global
    /// binding table at process start and prints every `(op, dtypes)`
    /// key with `>1` registered alternative (either multi-backend or
    /// multi-impl within one backend). `--ignored` because it's
    /// diagnostic, not a regression check; it always passes and
    /// produces output for the audit doc via `--nocapture`.
    ///
    /// Run: `cargo test -p fuel-storage --features cuda,vulkan \
    ///     audit_multi_backend_coverage -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn audit_multi_backend_coverage() {
        let b = global_bindings();

        // Group by (op_kind, dtypes); each entry maps to the list of
        // (backend, n_alts_within_backend). Iterates via iter_precision
        // because that already exposes per-alternative tuples.
        // OpKind/DType aren't Ord, so we use Vec-based group lookup.
        let mut grouped: Vec<((OpKind, Vec<DType>), Vec<(BackendId, usize)>)> =
            Vec::new();
        for (op, dtypes, backend, _precision) in b.iter_precision() {
            let key = (op, dtypes.to_vec());
            if let Some((_, entry)) = grouped.iter_mut().find(|(k, _)| *k == key) {
                if let Some(slot) = entry.iter_mut().find(|(be, _)| *be == backend) {
                    slot.1 += 1;
                } else {
                    entry.push((backend, 1));
                }
            } else {
                grouped.push((key, vec![(backend, 1)]));
            }
        }
        // Sort by op debug-string, then dtype list, for stable output.
        grouped.sort_by(|a, b| {
            let oa = format!("{:?}", a.0.0);
            let ob = format!("{:?}", b.0.0);
            oa.cmp(&ob).then_with(|| {
                let da = format!("{:?}", a.0.1);
                let db = format!("{:?}", b.0.1);
                da.cmp(&db)
            })
        });

        let total_keys = grouped.len();
        let mut multi_backend = 0;
        let mut multi_impl_single_backend = 0;
        let mut single_alt = 0;

        println!("\n=== Picker-alternatives audit ===");
        println!("Total unique (op, dtypes) keys: {total_keys}");
        println!("Total alternatives (sum):       {}", b.len());
        println!();
        println!("Keys with >1 alternative (per Judge-picking working set):");
        for ((op, dtypes), backends) in &grouped {
            let total_alts: usize = backends.iter().map(|(_, n)| *n).sum();
            if total_alts <= 1 {
                single_alt += 1;
                continue;
            }
            if backends.len() > 1 {
                multi_backend += 1;
            } else {
                multi_impl_single_backend += 1;
            }
            let backend_summary: Vec<String> = backends
                .iter()
                .map(|(be, n)| {
                    if *n == 1 {
                        format!("{be:?}")
                    } else {
                        format!("{be:?}×{n}")
                    }
                })
                .collect();
            println!(
                "  {op:?} {dtypes:?} → {} alts: [{}]",
                total_alts,
                backend_summary.join(", "),
            );
        }
        println!();
        println!("--- summary ---");
        println!("Single-alternative keys (no pick to make): {single_alt}");
        println!("Multi-backend keys (cross-backend pick):   {multi_backend}");
        println!("Multi-impl single-backend keys:            {multi_impl_single_backend}");
    }
}
