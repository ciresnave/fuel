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
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use fuel_ir::backend::{BackendCapabilities, SubstrateClass, TransferPath};
use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, DeviceLocation, Error, Layout, Result};

use crate::kernel::{KernelBindingTable, KernelRef, MatmulM, OpParams};
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
        // pub(crate) so the FKC vertical-slice test (fkc::register) can name
        // the real production wrapper and assert an imported binding IS it.
        pub(crate) fn $wrapper(
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
        // pub(crate) so the FKC link registry (fkc::cpu_link) can name the real
        // production wrapper for the elementwise-unary contract's entry points.
        pub(crate) fn $wrapper(
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
        // pub(crate) so the FKC link table (fkc::cpu_link) can name the real
        // production wrapper as `crate::dispatch::where_<dt>_cpu_wrapper`.
        pub(crate) fn $wrapper(
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
pub(crate) fn flip_cpu_wrapper(
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
        pub(crate) fn $wrapper(
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
        pub(crate) fn $wrapper(
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
        // `pub(crate)` so the FKC `link_registry` (`fkc::cpu_link`) can name the
        // real production wrapper and the importer can bind it FROM the contract.
        pub(crate) fn $wrapper(
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
        // `pub(crate)` so the FKC `link_registry` (`fkc::cpu_link`) can name the
        // real production wrapper and the importer can bind it FROM the contract.
        pub(crate) fn $wrapper(
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
        // `pub(crate)` so the FKC `link_registry` (`fkc::cpu_link`) can name the
        // real production wrapper and the importer can bind it FROM the contract.
        pub(crate) fn $wrapper(
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
/// `pub(crate)` so the FKC `link_registry` (`fkc::cpu_link`) can name the real
/// production wrapper and the importer can bind it FROM the contract.
macro_rules! cpu_reduce_max_to_backward_wrapper {
    ($wrapper:ident, $kernel:path) => {
        pub(crate) fn $wrapper(
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
pub(crate) fn masked_fill_cpu_wrapper(
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
pub(crate) fn pad_cpu_wrapper(
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
        pub(crate) fn $wrapper(
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
pub(crate) fn roll_cpu_wrapper(
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
/// `pub(crate)` so the FKC `CpuLinkRegistry` can bind this symbol from the
/// indexing contract.
pub(crate) fn index_add_f32_cpu_wrapper(
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
/// `pub(crate)` so the FKC `CpuLinkRegistry` can bind this symbol from the
/// indexing contract.
pub(crate) fn scatter_add_f32_cpu_wrapper(
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
/// `pub(crate)` so the FKC `CpuLinkRegistry` (`fkc::cpu_link`) can bind this
/// symbol from the rope contract.
pub(crate) fn rope_f32_cpu_wrapper(
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
/// `pub(crate)` so the FKC `CpuLinkRegistry` can bind this symbol from the
/// indexing contract (the fabricated `gather_cpu_<dt>` symbols all resolve here).
pub(crate) fn gather_cpu_wrapper(
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
/// `pub(crate)` so the FKC `CpuLinkRegistry` can bind this symbol from the
/// indexing contract (the fabricated `index_select_cpu_<dt>` symbols resolve here).
pub(crate) fn index_select_cpu_wrapper(
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
        pub(crate) fn $wrapper(
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
        pub(crate) fn $wrapper(
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
/// and `OpParams::Rope` carries the geometry. `pub(crate)` so the FKC
/// `CpuLinkRegistry` (`fkc::cpu_link`) can bind these symbols from the rope
/// contract.
macro_rules! cpu_rope_wrapper {
    ($wrapper:ident, $kernel:path) => {
        pub(crate) fn $wrapper(
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
///
/// `pub(crate)` so the FKC `CpuLinkRegistry` can resolve the linear-quant fused
/// bundle's `qmatmul_cpu` `entry_point` symbol to this wrapper
/// (`CPU_FUSED_LINEAR_QUANT_ENTRY_POINTS`).
pub(crate) fn qmatmul_f32_cpu_wrapper(
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

/// Dispatch wrapper for `(WriteSlice, *, Cpu)`. Dtype-agnostic — the
/// underlying kernel is `write_slice_cpu(... dtype_size)`. The wrapper
/// reads dtype_size from the output (dest) Storage's dtype tag. Takes
/// 1 input (source) + 1 output (dest, mutated in place); the slab
/// offset rides in `OpParams::WriteSlice`. `pub(crate)` so the
/// shape-ops FKC contract can resolve its `write_slice_cpu` entry point
/// to this fn via [`crate::fkc::CPU_SHAPE_OPS_ENTRY_POINTS`].
pub(crate) fn write_slice_cpu_wrapper(
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
        OpParams::WriteSlice { dest_shape, ranges, .. } => (dest_shape, ranges),
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
/// dynamic position scalar (a runtime U32 operand, NOT part of the
/// binding-table lookup key — see `build_lookup_dtypes`). `pub(crate)`
/// so the shape-ops FKC contract can resolve its
/// `write_slice_rotating_cpu` entry point to this fn via
/// [`crate::fkc::CPU_SHAPE_OPS_ENTRY_POINTS`].
pub(crate) fn write_slice_rotating_cpu_wrapper(
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

/// Dispatch wrapper for `(WriteSliceDoff, *, Cpu)`. Like
/// `write_slice_rotating_cpu_wrapper` but the second input is a
/// rank-0 `I64` offset (the device-resident start on `axis`; read
/// host-side here on CPU) and there is no modulus/wrap. `pub(crate)`
/// so the shape-ops FKC contract can resolve its `write_slice_doff_cpu`
/// entry point to this fn via [`crate::fkc::CPU_SHAPE_OPS_ENTRY_POINTS`].
pub(crate) fn write_slice_doff_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "write_slice_doff wrapper expects 2 inputs (source, offset) + 1 output (dest), \
             got {} + {}",
            inputs.len(), outputs.len(),
        ))
        .bt());
    }
    let (dest_shape, axis, ranges) = match params {
        OpParams::WriteSliceDoff { dest_shape, axis, ranges } => {
            (dest_shape, *axis, ranges)
        }
        other => {
            return Err(Error::Msg(format!(
                "write_slice_doff wrapper expects OpParams::WriteSliceDoff, got {other:?}",
            ))
            .bt())
        }
    };
    let src_guard = read_storage(&inputs[0])?;
    let src_cpu = cpu_input(&src_guard)?;
    let off_guard = read_storage(&inputs[1])?;
    let off_cpu = cpu_input(&off_guard)?;
    let mut dest_guard = write_storage(&outputs[0])?;
    let dtype_size = dest_guard.dtype.size_in_bytes();
    let dest_cpu = cpu_output(&mut dest_guard)?;
    fuel_cpu_backend::byte_kernels::write_slice_doff_cpu(
        src_cpu, off_cpu, dest_cpu, dest_shape, axis, ranges, dtype_size,
    )
}

pub(crate) fn concat_cpu_wrapper(
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
        // `pub(crate)` so the FKC `CpuLinkRegistry` (`fkc::cpu_link`) can bind
        // these symbols from the indexing contract.
        pub(crate) fn $wrapper(
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
        // `pub(crate)` so the FKC `CpuLinkRegistry` can bind these symbols.
        pub(crate) fn $wrapper(
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
        // pub(crate) so the FKC link registry (fkc::cpu_link) can name the
        // real production wrapper for the affine-clamp-powi contract import.
        pub(crate) fn $wrapper(
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
        // pub(crate) so the FKC link registry (fkc::cpu_link) can name the
        // real production wrapper for the affine-clamp-powi contract import.
        pub(crate) fn $wrapper(
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
        // pub(crate) so the FKC link registry (fkc::cpu_link) can name the
        // real production wrapper for the affine-clamp-powi contract import.
        pub(crate) fn $wrapper(
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
        // pub(crate) so the FKC link registry (fkc::cpu_link) can name the
        // real production wrapper for the affine-clamp-powi contract import.
        pub(crate) fn $wrapper(
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
pub(crate) fn affine_f32_cpu_wrapper(
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
        pub(crate) fn $name(
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
        pub(crate) fn $name(
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
        pub(crate) fn $name(
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
        pub(crate) fn $name(
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
        pub(crate) fn $name(
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
        pub(crate) fn $name(
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
pub(crate) fn clamp_elementwise_f32_cpu_wrapper(
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
pub(crate) fn powi_elementwise_f32_cpu_wrapper(
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
        // pub(crate) so the FKC link registry (fkc::cpu_link) can name the
        // real production wrapper for the affine-clamp-powi contract import.
        pub(crate) fn $wrapper(
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
        pub(crate) fn $wrapper(
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
        // `pub(crate)` so the FKC `link_registry` (`fkc::cpu_link`) can name the
        // real production wrapper and the importer can bind it FROM the contract.
        pub(crate) fn $wrapper(
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
/// `pub(crate)` so the FKC `link_registry` (`fkc::cpu_link`) can name the real
/// production wrapper and the importer can bind it FROM the contract.
macro_rules! cpu_conv2d_wrapper {
    ($name:ident, $kernel:path) => {
        pub(crate) fn $name(
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
/// `pub(crate)` so the FKC `link_registry` (`fkc::cpu_link`) can name the real
/// production wrapper and the importer can bind it FROM the contract.
macro_rules! cpu_conv_transpose2d_wrapper {
    ($name:ident, $kernel:path) => {
        pub(crate) fn $name(
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
/// `pub(crate)` so the FKC `link_registry` (`fkc::cpu_link`) can name the real
/// production wrapper and the importer can bind it FROM the contract.
macro_rules! cpu_reduce_sum_to_wrapper {
    ($name:ident, $kernel:path) => {
        pub(crate) fn $name(
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
/// `pub(crate)` so the FKC `link_registry` (`fkc::cpu_link`) can name the real
/// production wrapper and the importer can bind it FROM the contract.
macro_rules! cpu_reduce_max_to_wrapper {
    ($name:ident, $kernel:path) => {
        pub(crate) fn $name(
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
        // `pub(crate)` so the FKC `CpuLinkRegistry` can resolve the matmul
        // contract's `fused_linear_<dt>` `entry_point` symbols to these wrappers.
        pub(crate) fn $wrapper(
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
                OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k, .. } => {
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
        pub(crate) fn $wrapper(
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
        pub(crate) fn $wrapper(
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
        pub(crate) fn $wrapper(
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

/// Per-input-dtype NonZeroIndices dispatch wrapper. One input `x`
/// (the value tensor) → one bundled output `[indices [capacity] U32 ;
/// count [1] U32]`. `capacity` flows through `OpParams::NonZeroIndices`;
/// the binding-table key is `[input_dtype, U32]` (input + the primary
/// output-slot dtype). The `count_sym` field is consumed by the executor
/// after the kernel runs (the SymEnv bind seam), not here.
macro_rules! cpu_nonzero_indices_wrapper {
    ($wrapper:ident, $kernel:path) => {
        pub(crate) fn $wrapper(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "nonzero_indices wrapper expects 1 input, got {}",
                    inputs.len(),
                ))
                .bt());
            }
            if outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "nonzero_indices wrapper expects 1 output, got {}",
                    outputs.len(),
                ))
                .bt());
            }
            let capacity = match params {
                OpParams::NonZeroIndices { capacity, .. } => *capacity,
                other => {
                    return Err(Error::Msg(format!(
                        "nonzero_indices wrapper expects OpParams::NonZeroIndices, got {other:?}",
                    ))
                    .bt());
                }
            };
            let x_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let x_cpu = cpu_input(&x_guard)?;
            let out_cpu = cpu_output(&mut out_guard)?;
            $kernel(x_cpu, out_cpu, capacity)
        }
    };
}

cpu_nonzero_indices_wrapper!(nonzero_indices_f32_cpu_wrapper, fuel_cpu_backend::byte_kernels::nonzero_indices_f32);
cpu_nonzero_indices_wrapper!(nonzero_indices_u32_cpu_wrapper, fuel_cpu_backend::byte_kernels::nonzero_indices_u32);

/// Per-dtype SsdChunkScan dispatch wrapper. Five inputs (x, dt, a,
/// b, c) → one output (y). Geometry + `chunk_size` flow through
/// `OpParams::SsdChunkScan`. All six tensors share dtype `T`; the
/// binding-table key is `[T; 6]`.
macro_rules! cpu_ssd_chunk_scan_wrapper {
    ($wrapper:ident, $kernel:path) => {
        pub(crate) fn $wrapper(
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
        // `pub(crate)` so the FKC `CpuLinkRegistry` can resolve the attention
        // contract's forward-FlashAttn `entry_point` symbols to these wrappers.
        pub(crate) fn $wrapper(
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
        // `pub(crate)` so the FKC `CpuLinkRegistry` can resolve the attention
        // contract's FlashAttnBackward{Q,K,V} `entry_point` symbols to these wrappers.
        pub(crate) fn $wrapper(
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
        // `pub(crate)` so the FKC `CpuLinkRegistry` (`CPU_ATTENTION_ENTRY_POINTS`)
        // can resolve the paged sections' `byte_kernels::paged_attn_<dt>` symbols
        // to these wrappers (the attention family is contract-sourced; §3.9.1).
        pub(crate) fn $wrapper(
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

// Cast wrappers — one per TARGET dtype, each matching every other source
// dtype. Together they cover the full 11×10 = 110 directed pair matrix
// (identity pairs excluded; the optimizer elides them). Every real numeric
// dtype is a complete closed cast basis.
cpu_cast_wrapper!(
    cast_to_f32_cpu_wrapper,
    DType::F32,
    "f32",
    {
        DType::F64     => fuel_cpu_backend::byte_kernels::cast_f64_to_f32,
        DType::F16     => fuel_cpu_backend::byte_kernels::cast_f16_to_f32,
        DType::BF16    => fuel_cpu_backend::byte_kernels::cast_bf16_to_f32,
        DType::F8E4M3  => fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_f32,
        DType::U8      => fuel_cpu_backend::byte_kernels::cast_u8_to_f32,
        DType::I8      => fuel_cpu_backend::byte_kernels::cast_i8_to_f32,
        DType::U32     => fuel_cpu_backend::byte_kernels::cast_u32_to_f32,
        DType::I16     => fuel_cpu_backend::byte_kernels::cast_i16_to_f32,
        DType::I32     => fuel_cpu_backend::byte_kernels::cast_i32_to_f32,
        DType::I64     => fuel_cpu_backend::byte_kernels::cast_i64_to_f32,
    },
);
cpu_cast_wrapper!(
    cast_to_f64_cpu_wrapper,
    DType::F64,
    "f64",
    {
        DType::F32     => fuel_cpu_backend::byte_kernels::cast_f32_to_f64,
        DType::F16     => fuel_cpu_backend::byte_kernels::cast_f16_to_f64,
        DType::BF16    => fuel_cpu_backend::byte_kernels::cast_bf16_to_f64,
        DType::F8E4M3  => fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_f64,
        DType::U8      => fuel_cpu_backend::byte_kernels::cast_u8_to_f64,
        DType::I8      => fuel_cpu_backend::byte_kernels::cast_i8_to_f64,
        DType::U32     => fuel_cpu_backend::byte_kernels::cast_u32_to_f64,
        DType::I16     => fuel_cpu_backend::byte_kernels::cast_i16_to_f64,
        DType::I32     => fuel_cpu_backend::byte_kernels::cast_i32_to_f64,
        DType::I64     => fuel_cpu_backend::byte_kernels::cast_i64_to_f64,
    },
);
cpu_cast_wrapper!(
    cast_to_bf16_cpu_wrapper,
    DType::BF16,
    "bf16",
    {
        DType::F32     => fuel_cpu_backend::byte_kernels::cast_f32_to_bf16,
        DType::F64     => fuel_cpu_backend::byte_kernels::cast_f64_to_bf16,
        DType::F16     => fuel_cpu_backend::byte_kernels::cast_f16_to_bf16,
        DType::F8E4M3  => fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_bf16,
        DType::U8      => fuel_cpu_backend::byte_kernels::cast_u8_to_bf16,
        DType::I8      => fuel_cpu_backend::byte_kernels::cast_i8_to_bf16,
        DType::U32     => fuel_cpu_backend::byte_kernels::cast_u32_to_bf16,
        DType::I16     => fuel_cpu_backend::byte_kernels::cast_i16_to_bf16,
        DType::I32     => fuel_cpu_backend::byte_kernels::cast_i32_to_bf16,
        DType::I64     => fuel_cpu_backend::byte_kernels::cast_i64_to_bf16,
    },
);
cpu_cast_wrapper!(
    cast_to_f16_cpu_wrapper,
    DType::F16,
    "f16",
    {
        DType::F32     => fuel_cpu_backend::byte_kernels::cast_f32_to_f16,
        DType::F64     => fuel_cpu_backend::byte_kernels::cast_f64_to_f16,
        DType::BF16    => fuel_cpu_backend::byte_kernels::cast_bf16_to_f16,
        DType::F8E4M3  => fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_f16,
        DType::U8      => fuel_cpu_backend::byte_kernels::cast_u8_to_f16,
        DType::I8      => fuel_cpu_backend::byte_kernels::cast_i8_to_f16,
        DType::U32     => fuel_cpu_backend::byte_kernels::cast_u32_to_f16,
        DType::I16     => fuel_cpu_backend::byte_kernels::cast_i16_to_f16,
        DType::I32     => fuel_cpu_backend::byte_kernels::cast_i32_to_f16,
        DType::I64     => fuel_cpu_backend::byte_kernels::cast_i64_to_f16,
    },
);
cpu_cast_wrapper!(
    cast_to_f8e4m3_cpu_wrapper,
    DType::F8E4M3,
    "f8e4m3",
    {
        DType::F32     => fuel_cpu_backend::byte_kernels::cast_f32_to_f8e4m3,
        DType::F64     => fuel_cpu_backend::byte_kernels::cast_f64_to_f8e4m3,
        DType::F16     => fuel_cpu_backend::byte_kernels::cast_f16_to_f8e4m3,
        DType::BF16    => fuel_cpu_backend::byte_kernels::cast_bf16_to_f8e4m3,
        DType::U8      => fuel_cpu_backend::byte_kernels::cast_u8_to_f8e4m3,
        DType::I8      => fuel_cpu_backend::byte_kernels::cast_i8_to_f8e4m3,
        DType::U32     => fuel_cpu_backend::byte_kernels::cast_u32_to_f8e4m3,
        DType::I16     => fuel_cpu_backend::byte_kernels::cast_i16_to_f8e4m3,
        DType::I32     => fuel_cpu_backend::byte_kernels::cast_i32_to_f8e4m3,
        DType::I64     => fuel_cpu_backend::byte_kernels::cast_i64_to_f8e4m3,
    },
);
cpu_cast_wrapper!(
    cast_to_u8_cpu_wrapper,
    DType::U8,
    "u8",
    {
        DType::F32     => fuel_cpu_backend::byte_kernels::cast_f32_to_u8,
        DType::F64     => fuel_cpu_backend::byte_kernels::cast_f64_to_u8,
        DType::F16     => fuel_cpu_backend::byte_kernels::cast_f16_to_u8,
        DType::BF16    => fuel_cpu_backend::byte_kernels::cast_bf16_to_u8,
        DType::F8E4M3  => fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_u8,
        DType::I8      => fuel_cpu_backend::byte_kernels::cast_i8_to_u8,
        DType::U32     => fuel_cpu_backend::byte_kernels::cast_u32_to_u8,
        DType::I16     => fuel_cpu_backend::byte_kernels::cast_i16_to_u8,
        DType::I32     => fuel_cpu_backend::byte_kernels::cast_i32_to_u8,
        DType::I64     => fuel_cpu_backend::byte_kernels::cast_i64_to_u8,
    },
);
cpu_cast_wrapper!(
    cast_to_i8_cpu_wrapper,
    DType::I8,
    "i8",
    {
        DType::F32     => fuel_cpu_backend::byte_kernels::cast_f32_to_i8,
        DType::F64     => fuel_cpu_backend::byte_kernels::cast_f64_to_i8,
        DType::F16     => fuel_cpu_backend::byte_kernels::cast_f16_to_i8,
        DType::BF16    => fuel_cpu_backend::byte_kernels::cast_bf16_to_i8,
        DType::F8E4M3  => fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_i8,
        DType::U8      => fuel_cpu_backend::byte_kernels::cast_u8_to_i8,
        DType::U32     => fuel_cpu_backend::byte_kernels::cast_u32_to_i8,
        DType::I16     => fuel_cpu_backend::byte_kernels::cast_i16_to_i8,
        DType::I32     => fuel_cpu_backend::byte_kernels::cast_i32_to_i8,
        DType::I64     => fuel_cpu_backend::byte_kernels::cast_i64_to_i8,
    },
);
cpu_cast_wrapper!(
    cast_to_u32_cpu_wrapper,
    DType::U32,
    "u32",
    {
        DType::F32     => fuel_cpu_backend::byte_kernels::cast_f32_to_u32,
        DType::F64     => fuel_cpu_backend::byte_kernels::cast_f64_to_u32,
        DType::F16     => fuel_cpu_backend::byte_kernels::cast_f16_to_u32,
        DType::BF16    => fuel_cpu_backend::byte_kernels::cast_bf16_to_u32,
        DType::F8E4M3  => fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_u32,
        DType::U8      => fuel_cpu_backend::byte_kernels::cast_u8_to_u32,
        DType::I8      => fuel_cpu_backend::byte_kernels::cast_i8_to_u32,
        DType::I16     => fuel_cpu_backend::byte_kernels::cast_i16_to_u32,
        DType::I32     => fuel_cpu_backend::byte_kernels::cast_i32_to_u32,
        DType::I64     => fuel_cpu_backend::byte_kernels::cast_i64_to_u32,
    },
);
cpu_cast_wrapper!(
    cast_to_i16_cpu_wrapper,
    DType::I16,
    "i16",
    {
        DType::F32     => fuel_cpu_backend::byte_kernels::cast_f32_to_i16,
        DType::F64     => fuel_cpu_backend::byte_kernels::cast_f64_to_i16,
        DType::F16     => fuel_cpu_backend::byte_kernels::cast_f16_to_i16,
        DType::BF16    => fuel_cpu_backend::byte_kernels::cast_bf16_to_i16,
        DType::F8E4M3  => fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_i16,
        DType::U8      => fuel_cpu_backend::byte_kernels::cast_u8_to_i16,
        DType::I8      => fuel_cpu_backend::byte_kernels::cast_i8_to_i16,
        DType::U32     => fuel_cpu_backend::byte_kernels::cast_u32_to_i16,
        DType::I32     => fuel_cpu_backend::byte_kernels::cast_i32_to_i16,
        DType::I64     => fuel_cpu_backend::byte_kernels::cast_i64_to_i16,
    },
);
cpu_cast_wrapper!(
    cast_to_i32_cpu_wrapper,
    DType::I32,
    "i32",
    {
        DType::F32     => fuel_cpu_backend::byte_kernels::cast_f32_to_i32,
        DType::F64     => fuel_cpu_backend::byte_kernels::cast_f64_to_i32,
        DType::F16     => fuel_cpu_backend::byte_kernels::cast_f16_to_i32,
        DType::BF16    => fuel_cpu_backend::byte_kernels::cast_bf16_to_i32,
        DType::F8E4M3  => fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_i32,
        DType::U8      => fuel_cpu_backend::byte_kernels::cast_u8_to_i32,
        DType::I8      => fuel_cpu_backend::byte_kernels::cast_i8_to_i32,
        DType::U32     => fuel_cpu_backend::byte_kernels::cast_u32_to_i32,
        DType::I16     => fuel_cpu_backend::byte_kernels::cast_i16_to_i32,
        DType::I64     => fuel_cpu_backend::byte_kernels::cast_i64_to_i32,
    },
);
cpu_cast_wrapper!(
    cast_to_i64_cpu_wrapper,
    DType::I64,
    "i64",
    {
        DType::F32     => fuel_cpu_backend::byte_kernels::cast_f32_to_i64,
        DType::F64     => fuel_cpu_backend::byte_kernels::cast_f64_to_i64,
        DType::F16     => fuel_cpu_backend::byte_kernels::cast_f16_to_i64,
        DType::BF16    => fuel_cpu_backend::byte_kernels::cast_bf16_to_i64,
        DType::F8E4M3  => fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_i64,
        DType::U8      => fuel_cpu_backend::byte_kernels::cast_u8_to_i64,
        DType::I8      => fuel_cpu_backend::byte_kernels::cast_i8_to_i64,
        DType::U32     => fuel_cpu_backend::byte_kernels::cast_u32_to_i64,
        DType::I16     => fuel_cpu_backend::byte_kernels::cast_i16_to_i64,
        DType::I32     => fuel_cpu_backend::byte_kernels::cast_i32_to_i64,
    },
);

/// Dispatch wrapper for `(MatMul, F32, Cpu)`. Extracts the
/// `OpParams::Matmul { m, n, k }` and forwards to the typed
/// kernel. Both inputs are guaranteed contiguous f32 by the
/// executor's auto-Contiguize pass.
///
/// `pub(crate)` so the FKC `CpuLinkRegistry` (`fkc::cpu_link`,
/// [`crate::fkc::CPU_MATMUL_ENTRY_POINTS`]) can resolve the matmul contract's
/// `entry_point` symbol to this exact wrapper fn-pointer.
pub(crate) fn matmul_f32_cpu_wrapper(
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
    let (lhs_batch_dims, rhs_batch_dims, m, n, k, m_compute) = match params {
        OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k, m_compute } => {
            (lhs_batch_dims, rhs_batch_dims, *m, *n, *k, m_compute)
        }
        other => {
            return Err(Error::Msg(format!(
                "matmul wrapper expects OpParams::Matmul, got {other:?}",
            ))
            .bt())
        }
    };
    // Data-determined-M (sparse MoE): `m` is the row capacity; compute only
    // `rows`. `Deferred` must have been resolved at execute before the
    // kernel runs (see resolve_deferred_matmul).
    let rows = match m_compute {
        MatmulM::All => m,
        MatmulM::Rows(c) => *c,
        MatmulM::Deferred(d) => {
            return Err(Error::Msg(format!(
                "matmul_f32 wrapper: unresolved data-determined row count {d:?} \
                 (must be resolved at execute before the kernel runs)",
            ))
            .bt());
        }
    };
    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cpu = cpu_input(&lhs_guard)?;
    let rhs_cpu = cpu_input(&rhs_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;
    fuel_cpu_backend::byte_kernels::matmul_f32_capacity(
        lhs_cpu,
        rhs_cpu,
        out_cpu,
        lhs_batch_dims,
        rhs_batch_dims,
        m,
        rows,
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
        // `pub(crate)` so the FKC `CpuLinkRegistry` can resolve the matmul
        // contract's `entry_point` symbols to these wrapper fn-pointers.
        pub(crate) fn $wrapper(
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
                OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k, .. } => {
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
///
/// `pub(crate)` so the FKC `CpuLinkRegistry` can resolve the matmul contract's
/// `entry_point` symbol to this exact wrapper fn-pointer.
pub(crate) fn matmul_f64_cpu_wrapper(
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
        OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k, .. } => {
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

/// The authored CPU elementwise-binary kernel contract, embedded into the
/// binary. This is the PRODUCTION `include_str!` (distinct from the identical
/// path used in the `fkc::register` test module); `register_cpu_binary_from_contract`
/// parses + lowers it and binds the family FROM THE CONTRACT.
const CPU_ELEMENTWISE_BINARY_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/elementwise-binary.fkc.md");

/// Register the CPU elementwise-binary family (8 ops × 4 dtypes = 32
/// bindings) by IMPORTING its FKC kernel contract — the FIRST production
/// FKC consumer. FKC is unconditional core infrastructure, so this is the
/// ONE registration path for the family: there is no hand-written fallback
/// (deleting it was the point — a build that lost the importer would silently
/// lose the family).
///
/// The authored contract `docs/kernel-contracts/cpu/elementwise-binary.fkc.md`
/// is parsed + lowered and each `entry_point` symbol resolved through the
/// production [`crate::fkc::CpuLinkRegistry`] to the exact same wrapper
/// fn-pointers the CPU backend exposes (`add_elementwise_f32_cpu_wrapper`, …;
/// the very map [`crate::fkc::CPU_BINARY_ENTRY_POINTS`] holds). Relative to the
/// deleted hand-written registrations this is behavior-preserving: identical
/// kernels + caps; the binding's `kernel_source` becomes the contract's
/// `"portable-cpu"` tag and its precision the contract's audited claim. Cost is
/// preserved because this runs BEFORE the `fill_unset_cpu_cost` pass, which
/// upgrades the imported entries' `unknown_cost` sentinel to the same OpKind
/// cost fn every other CPU primitive gets.
///
/// Fused-registry decision: the elementwise-binary family declares NO fused
/// ops (`provider.fused` is empty), and `register_cpu_kernels` threads only a
/// `KernelBindingTable` — there is no paired global `FusedKernelRegistry` at
/// this seam (the static `crate::fused::default_kernel_registry()` is built
/// independently). So `register_into`'s required fused argument is a local
/// throwaway that provably stays empty.
///
/// Init-boundary fail-fast: a parse/lower/link failure of the embedded,
/// authored contract is a programmer error surfaced once here via `expect`
/// (mirroring the `finalize().expect(...)` convention in [`global_bindings`]).
/// It cannot fail for a runtime-data reason — the contract is `include_str!`'d
/// into the binary and the link registry is exhaustive for this family.
fn register_cpu_binary_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_ELEMENTWISE_BINARY_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU elementwise-binary contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "elementwise-binary contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU elementwise-binary contract must register into the binding table",
    );
}

/// The authored CPU affine / clamp / powi kernel contract, embedded into the
/// binary (the PRODUCTION `include_str!`). `register_cpu_affine_clamp_powi_from_contract`
/// parses + lowers it and binds the family FROM THE CONTRACT.
const CPU_AFFINE_CLAMP_POWI_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/affine-clamp-powi.fkc.md");

/// Register the CPU out-of-place scalar-param family (affine/clamp/powi × 4
/// dtypes + powi_backward × 4 = 16 bindings) by IMPORTING its FKC kernel
/// contract — the second production FKC consumer, mirroring
/// [`register_cpu_binary_from_contract`]. FKC is unconditional core
/// infrastructure, so this is the ONE registration path for the family: the
/// hand-written `table.register(...)` calls it used to carry are DELETED.
///
/// Each `entry_point` symbol resolves through the production
/// [`crate::fkc::CpuLinkRegistry`] (now chaining
/// [`crate::fkc::CPU_AFFINE_CLAMP_POWI_ENTRY_POINTS`]) to the exact same wrapper
/// fn-pointers the CPU backend exposes. Behavior-preserving vs. the deleted
/// hand-written path: identical kernels + caps (contiguous-only); the binding's
/// `kernel_source` becomes the contract's `"portable-cpu"` tag and its precision
/// the contract's audited bit-stable claim. Cost is preserved because this runs
/// BEFORE `fill_unset_cpu_cost`, which upgrades the imported entries'
/// `unknown_cost` sentinel to the same OpKind cost fn every CPU primitive gets.
///
/// The scalar params (affine mul/add, clamp min/max, powi exp) ride in
/// `OpParams`, NOT the dtype-list, so the imported binding keys are `[t, t]` for
/// the single-input forward ops and `[t, t, t]` for the two-input
/// `powi_backward` — identical to the deleted `&unary(t)` / `&binary(t)` regs.
///
/// The family declares NO fused ops (the FusedOps::POWI_BACKWARD fused-registry
/// registration is a SEPARATE seam, untouched), so `register_into`'s required
/// fused argument is a local throwaway that provably stays empty.
fn register_cpu_affine_clamp_powi_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_AFFINE_CLAMP_POWI_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU affine/clamp/powi contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "affine/clamp/powi contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU affine/clamp/powi contract must register into the binding table",
    );
}

/// The authored CPU elementwise-unary kernel contract, embedded into the binary
/// (the PRODUCTION `include_str!`). `register_cpu_unary_from_contract` parses +
/// lowers it and binds the family FROM THE CONTRACT.
const CPU_ELEMENTWISE_UNARY_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/elementwise-unary.fkc.md");

/// Register the CPU elementwise-unary family (22 ops × 4 dtypes = 88 bindings)
/// by IMPORTING its FKC kernel contract — the third production FKC consumer,
/// mirroring [`register_cpu_binary_from_contract`] /
/// [`register_cpu_affine_clamp_powi_from_contract`]. FKC is unconditional core
/// infrastructure, so this is the ONE registration path for the family: the
/// hand-written `table.register(...)` calls it used to carry are DELETED.
///
/// This is the **first** consumer of the §3.4 multi-dtype fan-out: each per-op
/// section declares a BASE `entry_point` (e.g. `…::relu`) and enumerates
/// `dtypes: [F32, F64, BF16, F16]`; the importer fans it into one binding per
/// dtype, resolving `<base>_<dtype>` through the production
/// [`crate::fkc::CpuLinkRegistry`] (now chaining
/// [`crate::fkc::CPU_UNARY_ENTRY_POINTS`]) to the exact same wrapper fn-pointers
/// the CPU backend exposes. Behavior-preserving vs. the deleted hand-written
/// path: identical kernels + caps (contiguous-only, `[t, t]` keys); the
/// binding's `kernel_source` becomes the contract's `"portable-cpu"` tag and its
/// precision the contract's bit-stable claim. Cost is preserved because this
/// runs BEFORE `fill_unset_cpu_cost`, which upgrades the imported entries'
/// `unknown_cost` sentinel to the same OpKind cost fn every CPU primitive gets.
///
/// `gelu_tanh` (`OpKind::GeluElementwise`, base `gelu`) and `gelu_erf`
/// (`OpKind::GeluErfElementwise`, base `gelu_erf`) stay DISTINCT — the exact-erf
/// GELU must never be confused with the tanh approximation under a Judge epsilon.
///
/// The family declares NO fused ops, so `register_into`'s required fused argument
/// is a local throwaway that provably stays empty.
fn register_cpu_unary_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_ELEMENTWISE_UNARY_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU elementwise-unary contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "elementwise-unary contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU elementwise-unary contract must register into the binding table",
    );
}

/// The authored CPU compare + where kernel contract, embedded into the binary
/// (the PRODUCTION `include_str!`). `register_cpu_compare_where_from_contract`
/// parses + lowers it and binds the family FROM THE CONTRACT.
const CPU_COMPARE_WHERE_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/compare-where.fkc.md");

/// Register the CPU compare (6 ops × 4 dtypes = 24) + where (1 op × 4 dtypes =
/// 4) family — 28 bindings — by IMPORTING its FKC kernel contract, the fourth
/// production FKC consumer (mirroring the binary / affine-clamp-powi / unary
/// importers). FKC is unconditional core infrastructure, so this is the ONE
/// registration path for the family: the hand-written `table.register(...)`
/// calls it used to carry are DELETED.
///
/// Two contract shapes, both resolved through the production
/// [`crate::fkc::CpuLinkRegistry`] (now chaining
/// [`crate::fkc::CPU_COMPARE_ENTRY_POINTS`] + [`crate::fkc::CPU_WHERE_ENTRY_POINTS`]):
///  - the 24 per-(op,dtype) COMPARE thunks are single-dtype sections → each
///    resolves its declared `_u8`-suffixed symbol (`eq_f32_u8`, …) AS-IS; the
///    binding key is `[T, T, U8]` (output is the `fixed(U8)` mask), identical
///    to the deleted `&compare(t)` regs.
///  - the single WHERE section rides the §3.4 multi-dtype fan-out — its BASE
///    `entry_point` `…::where` expands to `where_{f32,f64,bf16,f16}`, one
///    binding per dtype, key `[U8, T, T, T]` (cond U8 + `passthrough(a)` → T),
///    identical to the deleted `&where_dts(t)` regs.
///
/// Behavior-preserving vs. the deleted hand-written path: identical kernels +
/// caps (contiguous-only); the binding's `kernel_source` becomes the contract's
/// `"portable-cpu"` tag and its precision the contract's audited bit-stable
/// claim. Cost is preserved because this runs BEFORE `fill_unset_cpu_cost`,
/// which upgrades the imported entries' `unknown_cost` sentinel to the same
/// OpKind cost fn every CPU primitive gets. The `## compare` chassis umbrella is
/// `registrable: false` (§3.10) and never registers. The family declares NO
/// fused ops, so `register_into`'s required fused argument is a local throwaway
/// that provably stays empty.
fn register_cpu_compare_where_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_COMPARE_WHERE_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU compare/where contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "compare/where contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU compare/where contract must register into the binding table",
    );
}

/// The authored CPU reduce kernel contract, embedded into the binary (the
/// PRODUCTION `include_str!`). `register_cpu_reduce_from_contract` parses +
/// lowers it and binds the family FROM THE CONTRACT.
const CPU_REDUCE_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/reduce.fkc.md");

/// Register the CPU per-axis reduce family (Sum/Mean/Max/Min × 4 dtypes = 16
/// bindings) by IMPORTING its FKC kernel contract, the fifth production FKC
/// consumer (mirroring the binary / affine-clamp-powi / unary / compare-where
/// importers). FKC is unconditional core infrastructure, so this is the ONE
/// registration path for the family: the hand-written `table.register(...)`
/// calls it used to carry are DELETED.
///
/// Each of the 16 per-(op, dtype) sections (`## sum_reduce_f32`, …) is a SPECIFIC
/// single-dtype contract with a concrete `entry_point` (`…::sum_reduce_f32`), so
/// none of them fan — the importer resolves each declared symbol AS-IS through
/// the production [`crate::fkc::CpuLinkRegistry`] (now chaining
/// [`crate::fkc::CPU_REDUCE_ENTRY_POINTS`]) to the exact wrapper fn-pointer the
/// CPU backend exposes. The binding key is `[T, T]` (input + `passthrough(input)`
/// output; the reduce axes + keepdim ride in `OpParams::Reduce`, NOT the
/// dtype-list), identical to the deleted `&unary(t)` regs.
///
/// Behavior-preserving vs. the deleted hand-written path: identical kernels +
/// caps (contiguous-only); the binding's `kernel_source` becomes the contract's
/// `"portable-cpu"` tag and its precision the contract's audited bit-stable
/// claim. Cost is preserved because this runs BEFORE `fill_unset_cpu_cost`, which
/// upgrades the imported entries' `unknown_cost` sentinel to the same OpKind cost
/// fn every CPU primitive gets. The `## reduce` chassis umbrella is
/// `registrable: false` (§3.10) and never registers (without it the chassis would
/// double-register `SumReduce`/`[F32]` → `DuplicateKernelRef` at init). The
/// f32-only `argmax_dim_f32` / `argmin_dim_f32` sections are `registrable: false`
/// (DEFERRED — production registers `Arg{Max,Min}Dim` for ALL input dtypes via
/// the hand-written `arg{max,min}_dim_u32_cpu_dispatch`, which stays untouched
/// below). The family declares NO fused ops, so `register_into`'s required fused
/// argument is a local throwaway that provably stays empty.
fn register_cpu_reduce_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_REDUCE_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU reduce contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "reduce contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU reduce contract must register into the binding table",
    );
}

/// The authored CPU reduce-to kernel contract, embedded into the binary (the
/// PRODUCTION `include_str!`). `register_cpu_reduce_to_from_contract` parses +
/// lowers it and binds the family FROM THE CONTRACT.
const CPU_REDUCE_TO_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/reduce-to.fkc.md");

/// Register the CPU broadcast-target reduce-to family (ReduceSumTo /
/// ReduceMaxTo × 4 dtypes = 8, key `[T, T]`; ReduceMaxToBackward × 4 dtypes = 4,
/// key `[T, T, T]` = 12 bindings) by IMPORTING its FKC kernel contract, the
/// sixth production FKC consumer (mirroring the binary / affine-clamp-powi /
/// unary / compare-where / reduce importers). FKC is unconditional core
/// infrastructure, so this is the ONE registration path for the family: the
/// hand-written `table.register(...)` calls it used to carry are DELETED.
///
/// Each of the 12 per-(op, dtype) sections (`## reduce_sum_to_f32`, …) is a
/// SPECIFIC single-dtype contract with a concrete `entry_point`
/// (`…::reduce_sum_to_f32`), so none of them fan — the importer resolves each
/// declared symbol AS-IS through the production [`crate::fkc::CpuLinkRegistry`]
/// (now chaining [`crate::fkc::CPU_REDUCE_TO_ENTRY_POINTS`]) to the exact
/// wrapper fn-pointer the CPU backend exposes. The forward binding key is
/// `[T, T]` (input + `passthrough(input)` output — the target
/// `input_shape`/`output_shape` ride in `OpParams::ReduceSumTo` /
/// `OpParams::ReduceMaxTo`, NOT the dtype-list), identical to the deleted
/// `&unary(t)` regs; the backward key is `[T, T, T]` (x, upstream +
/// `passthrough(x)` output), identical to the deleted `&binary(t)` regs.
///
/// Behavior-preserving vs. the deleted hand-written path: identical kernels +
/// caps (contiguous-only); the binding's `kernel_source` becomes the contract's
/// `"portable-cpu"` tag and its precision the contract's audited bit-stable
/// claim. Cost is preserved because this runs BEFORE `fill_unset_cpu_cost`,
/// which upgrades the imported entries' `unknown_cost` sentinel to the same
/// OpKind cost fn every CPU primitive gets (ReduceMaxToBackward keeps its own
/// `cost_reduce_max_to_backward_cpu` fill). The `## reduce_to` chassis umbrella
/// is `registrable: false` (§3.10) and never registers (without it the chassis
/// would double-register `ReduceSumTo`/`[F32]` → `DuplicateKernelRef` at init).
/// The family declares NO fused ops, so `register_into`'s required fused
/// argument is a local throwaway that provably stays empty.
fn register_cpu_reduce_to_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_REDUCE_TO_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU reduce-to contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "reduce-to contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU reduce-to contract must register into the binding table",
    );
}

/// The authored CPU norm (forward) kernel contract, embedded into the binary
/// (the PRODUCTION `include_str!`). `register_cpu_norm_from_contract` parses +
/// lowers it and binds the family FROM THE CONTRACT.
const CPU_NORM_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/norm.fkc.md");

/// Register the CPU last-dim NORM (forward) family (Softmax / LogSoftmax /
/// RmsNorm / LayerNorm × 4 dtypes = 16 bindings, key `[T, T]`) by IMPORTING its
/// FKC kernel contract, the seventh production FKC consumer (mirroring the
/// binary / affine-clamp-powi / unary / compare-where / reduce / reduce-to
/// importers). FKC is unconditional core infrastructure, so this is the ONE
/// registration path for the family: the hand-written `table.register(...)`
/// calls it used to carry are DELETED.
///
/// Each of the 16 per-(op, dtype) sections (`## softmax_last_dim_f32`,
/// `## rms_norm_last_dim_f32`, …) is a SPECIFIC single-dtype contract with a
/// concrete `entry_point` (`…::softmax_last_dim_f32`), so none of them fan — the
/// importer resolves each declared symbol AS-IS through the production
/// [`crate::fkc::CpuLinkRegistry`] (now chaining [`crate::fkc::CPU_NORM_ENTRY_POINTS`])
/// to the exact wrapper fn-pointer the CPU backend exposes. The binding key is
/// `[T, T]` (a SINGLE input + `passthrough(input)` output — the RMS/LayerNorm
/// kernels carry NO affine gamma/beta operand; they are the bare normalization,
/// and `outer_count` / `last_dim` / `eps` ride in
/// `OpParams::{SoftmaxLastDim,LogSoftmaxLastDim,NormLastDim}`, NOT the dtype-list),
/// identical to the deleted `&unary(t)` regs.
///
/// Behavior-preserving vs. the deleted hand-written path: identical kernels +
/// caps (contiguous-only); the binding's `kernel_source` becomes the contract's
/// `"portable-cpu"` tag and its precision the contract's audited bit-stable
/// claim. Cost is preserved because this runs BEFORE `fill_unset_cpu_cost`, which
/// upgrades the imported entries' `unknown_cost` sentinel to the same OpKind cost
/// fn every CPU primitive gets. This contract has NO `##` chassis umbrella
/// section, so there is nothing marked `registrable: false` and no double-register
/// risk. The forward NORM ops are ALSO registered in the `FusedKernelRegistry`
/// (`register_default_fused_kernels`, `FusedOps::{SOFTMAX,RMS_NORM,LAYER_NORM}_LAST_DIM`)
/// — that is a SEPARATE registry seam and stays untouched; this migration only
/// moves the `KernelBindingTable` primitive path. The BACKWARD forms live in a
/// separate norm-backward contract and their hand-written regs stay authoritative.
/// The family declares NO fused ops, so `register_into`'s required fused argument
/// is a local throwaway that provably stays empty.
fn register_cpu_norm_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_NORM_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU norm contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "norm forward contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU norm contract must register into the binding table",
    );
}

/// The authored CPU norm-BACKWARD kernel contract, embedded into the binary
/// (the PRODUCTION `include_str!`). `register_cpu_norm_backward_from_contract`
/// parses + lowers it and binds the family FROM THE CONTRACT.
const CPU_NORM_BACKWARD_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/norm-backward.fkc.md");

/// Register the CPU last-dim NORM-BACKWARD family (Softmax / LogSoftmax /
/// RmsNorm / LayerNorm backward × 4 dtypes = 16 bindings, key `[T, T, T]`) by
/// IMPORTING its FKC kernel contract, the eighth production FKC consumer (the
/// BACKWARD sibling of `register_cpu_norm_from_contract`). FKC is unconditional
/// core infrastructure, so this is the ONE registration path for the family: the
/// hand-written `table.register(...)` calls it used to carry are DELETED.
///
/// Each of the 16 per-(op, dtype) sections (`## softmax_last_dim_backward_f32`,
/// `## rms_norm_last_dim_backward_f32`, …) is a SPECIFIC single-dtype contract
/// with a concrete `entry_point` (`…::softmax_last_dim_backward_f32`), so none of
/// them fan — the importer resolves each declared symbol AS-IS through the
/// production [`crate::fkc::CpuLinkRegistry`] (now chaining
/// [`crate::fkc::CPU_NORM_BACKWARD_ENTRY_POINTS`]) to the exact wrapper
/// fn-pointer the CPU backend exposes. The binding key is `[T, T, T]` — the BARE
/// backward takes TWO inputs (softmax/log-softmax: forward output `y` + upstream
/// gradient `g`; layer/rms-norm: forward input `x` + `g`, stats recomputed from
/// `x` + `eps`) and writes ONE `passthrough(y|x)` output; outer_count / last_dim /
/// eps ride in `OpParams::{SoftmaxLastDim,LogSoftmaxLastDim,NormLastDim}`, NOT the
/// dtype-list — identical to the deleted `&binary(t)` regs.
///
/// Behavior-preserving vs. the deleted hand-written path: identical kernels +
/// caps (contiguous-only); the binding's `kernel_source` becomes the contract's
/// `"portable-cpu"` tag and its precision the contract's audited bit-stable
/// claim. Cost is preserved because this runs BEFORE `fill_unset_cpu_cost`, which
/// upgrades the imported entries' `unknown_cost` sentinel to the same OpKind cost
/// fn every CPU primitive gets. This contract has NO `##` chassis umbrella
/// section, so there is nothing marked `registrable: false` and no double-register
/// risk. The backward NORM ops are ALSO registered in the `FusedKernelRegistry`
/// (`register_default_fused_kernels`, `FusedOps::{SOFTMAX,LAYER_NORM,RMS_NORM}_
/// LAST_DIM_BACKWARD`) — that is a SEPARATE registry seam and stays untouched;
/// this migration only moves the `KernelBindingTable` primitive path. The family
/// declares NO fused ops, so `register_into`'s required fused argument is a local
/// throwaway that provably stays empty.
fn register_cpu_norm_backward_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_NORM_BACKWARD_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU norm-backward contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "norm backward contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU norm-backward contract must register into the binding table",
    );
}

/// The authored CPU RoPE kernel contract, embedded into the binary (the
/// PRODUCTION `include_str!`). `register_cpu_rope_from_contract` parses +
/// lowers it and binds the family FROM THE CONTRACT.
const CPU_ROPE_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/rope.fkc.md");

/// Register the CPU RoPE family (rotary position embedding; 1 op × 4 dtypes =
/// 4 bindings, key `[T, T, T, T]`) by IMPORTING its FKC kernel contract, the
/// ninth production FKC consumer. FKC is unconditional core infrastructure, so
/// this is the ONE registration path for the family: the hand-written
/// `table.register(Rope, ...)` calls it used to carry are DELETED.
///
/// Each of the 4 per-dtype sections (`## rope_f32`, `## rope_f64`,
/// `## rope_bf16`, `## rope_f16`) is a SPECIFIC single-dtype contract with a
/// concrete `entry_point` (`…::rope_f32`), so none of them fan — the importer
/// resolves each declared symbol AS-IS through the production
/// [`crate::fkc::CpuLinkRegistry`] (now chaining
/// [`crate::fkc::CPU_ROPE_ENTRY_POINTS`]) to the exact wrapper fn-pointer the
/// CPU backend exposes. The binding key is `[T, T, T, T]` — RoPE takes THREE
/// inputs (`x` + the precomputed `cos`/`sin` tables of shape `[seq, head_dim]`,
/// all one dtype; the tables broadcast over the `outer_count` axis by the
/// kernel re-indexing them per outer, NOT a stride-0 view) and writes ONE
/// `passthrough(x)` output; outer_count / seq / head_dim ride in
/// `OpParams::Rope`, NOT the dtype-list — identical to the deleted
/// `rope_dts(t)` regs.
///
/// Behavior-preserving vs. the deleted hand-written path: identical kernels +
/// caps (contiguous-only, even `head_dim`); the binding's `kernel_source`
/// becomes the contract's `"portable-cpu"` tag and its precision the contract's
/// bit-stable claim. Cost is preserved because this runs BEFORE
/// `fill_unset_cpu_cost`, which upgrades the imported entries' `unknown_cost`
/// sentinel to the same OpKind cost fn every CPU primitive gets. This contract
/// has NO `##` chassis umbrella section, so there is nothing marked
/// `registrable: false` and no double-register risk. RoPE is ALSO registered in
/// the `FusedKernelRegistry` (`register_default_fused_kernels`,
/// `FusedOps::ROPE`) — that is a SEPARATE registry seam and stays untouched;
/// this migration only moves the `KernelBindingTable` primitive path. The
/// family declares NO fused ops, so `register_into`'s required fused argument is
/// a local throwaway that provably stays empty.
fn register_cpu_rope_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_ROPE_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU rope contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "rope contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU rope contract must register into the binding table",
    );
}

/// The authored CPU SSM / Mamba kernel contract, embedded into the binary (the
/// PRODUCTION `include_str!`). `register_cpu_ssm_from_contract` parses + lowers
/// it and binds the MIGRATED subset of the family FROM THE CONTRACT.
const CPU_SSM_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/ssm.fkc.md");

/// Register the ENTIRE CPU SSM / Mamba family —
/// FusedSoftmaxCrossEntropy (key `[T, I64, F32]`) + CausalConv1d (key
/// `[T, T, T, T]`) + SelectiveScan + SsdChunkScan (key `[T; 6]`), 4 ops ×
/// 4 dtypes = 16 bindings — by IMPORTING its FKC kernel contract, the tenth
/// production FKC consumer. FKC is unconditional core infrastructure, so this is
/// the ONE registration path for the whole family: every hand-written
/// `table.register(...)` call is DELETED.
///
/// Each per-(op, dtype) section (`## fused_softmax_cross_entropy_f32`,
/// `## selective_scan_f32`, …) is a SPECIFIC single-dtype contract with a
/// concrete `entry_point` (`…::selective_scan_f32`), so none of them fan — the
/// importer resolves each declared symbol AS-IS through the production
/// [`crate::fkc::CpuLinkRegistry`] (chaining [`crate::fkc::CPU_SSM_ENTRY_POINTS`])
/// to the exact wrapper fn-pointer the CPU backend exposes. FSCE takes TWO inputs
/// (logits T + I64 targets) → ONE `fixed(F32)` output; CausalConv1d takes THREE
/// inputs (x, weight, bias) → ONE `passthrough(x)` output; every geometry/knob
/// rides in `OpParams`, NOT the dtype-list — identical keys to the deleted
/// hand-written regs.
///
/// **The two SCAN ops (SelectiveScan / SsdChunkScan) return a `return.bundle`**
/// multi-output (Option C: one packed buffer `[y ; last_state]`). The importer's
/// key-builder (`fkc/lower.rs` `assemble_dtype_variants`) appends the bundle's
/// PRIMARY-slot dtype (`passthrough(u)` / `passthrough(x)` → T) to the key tail,
/// so a 5-input scan section keys `[T; 6]` (5 inputs + the one bundled output
/// slot) — byte-for-byte the deleted hand-written
/// `table.register(SelectiveScan/SsdChunkScan, &[dt; 6], ...)` regs. All four
/// ops are now `registrable: true` (the default), so `lower_file` imports the
/// whole family.
///
/// Behavior-preserving vs. the deleted hand-written path: identical kernels +
/// caps (contiguous-only); the binding's
/// `kernel_source` becomes the contract's `"portable-cpu"` tag and its
/// precision the contract's bit-stable claim. Cost is preserved because this
/// runs BEFORE `fill_unset_cpu_cost`, which upgrades the imported entries'
/// `unknown_cost` sentinel to the same OpKind cost fn every CPU primitive gets.
/// The family declares NO fused ops (the "fused" in FusedSoftmaxCrossEntropy is
/// an intra-op softmax+NLL fusion, not a graph `FusedOpId`), so `register_into`'s
/// required fused argument is a local throwaway that provably stays empty.
fn register_cpu_ssm_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_SSM_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU ssm contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "ssm contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU ssm contract must register into the binding table",
    );
}

/// The authored CPU 2D-convolution kernel contract, embedded into the binary
/// (the PRODUCTION `include_str!`). `register_cpu_conv_from_contract` parses +
/// lowers it and binds the MIGRATED (with-bias) subset of the family FROM THE
/// CONTRACT.
const CPU_CONV_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/conv.fkc.md");

/// Register the CPU 2D-convolution family FULLY — Conv2D + ConvTranspose2D at
/// BOTH the no-bias key `[T, T, T]` (x, weight + out) and the with-bias key
/// `[T, T, T, T]` (x, weight, bias + passthrough(x) output), 2 ops × 4 dtypes ×
/// 2 operand-counts = 16 bindings — by IMPORTING its FKC kernel contract, the
/// eleventh production FKC consumer. FKC is unconditional core infrastructure,
/// so this is the ONE registration path for the whole family: every hand-written
/// `table.register(...)` conv call is DELETED.
///
/// Each per-(op, dtype) section (`## conv2d_f32`, `## conv_transpose2d_f32`, …)
/// is a SPECIFIC single-dtype contract with a concrete `entry_point`
/// (`…::conv2d_f32`), so none of them dtype-fan — the importer resolves each
/// declared symbol AS-IS through the production [`crate::fkc::CpuLinkRegistry`]
/// (now chaining [`crate::fkc::CPU_CONV_ENTRY_POINTS`]) to the exact wrapper
/// fn-pointer the CPU backend exposes. Conv2D's weight is `[Cout, Cin/groups,
/// Kh, Kw]`; ConvTranspose2D's is the transposed `[Cin, Cout/groups, Kh, Kw]`
/// and it carries an extra `output_padding` op-param — but both are the same
/// (x, weight, optional bias) → one-output (passthrough(x)) accept shape, and
/// every geometry knob rides in `OpParams::{Conv2D, ConvTranspose2D}`, NOT the
/// dtype-list.
///
/// **The NO-BIAS key is now CONTRACT-SOURCED.** The conv contract declares
/// `bias` as `optional: true`, and the FKC importer's key-builder
/// (`fkc/lower.rs` `assemble_dtype_variants`) now supports optional operands: an
/// optional LAST input fans each section into TWO keys — one OMITTING `bias`
/// (`[T, T, T]`) and one INCLUDING it (`[T, T, T, T]`) — BOTH resolving the SAME
/// `entry_point`/wrapper (the same CPU wrapper handles 2 or 3 inputs). The
/// former no-bias deferral (hand-written `table.register(Conv2D /
/// ConvTranspose2D, &[dt; 3], ...)` regs) is closed and those regs deleted.
///
/// Behavior-preserving vs. the deleted hand-written path (for both operand-count
/// keys): identical kernels + caps (contiguous-only NCHW); the
/// binding's `kernel_source` becomes the contract's `"portable-cpu"` tag and its
/// precision the contract's bit-stable claim. Cost is preserved because this
/// runs BEFORE `fill_unset_cpu_cost`, which upgrades the imported entries'
/// `unknown_cost` sentinel to the same OpKind cost fn every CPU primitive gets.
/// The family declares NO fused ops (the SEPARATE FusedKernelRegistry
/// `FusedOps::{CONV2D, CONV_TRANSPOSE2D}` seam is hand-written, in
/// `register_default_fused_kernels`, and stays untouched), so `register_into`'s
/// required fused argument is a local throwaway that provably stays empty.
fn register_cpu_conv_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_CONV_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU conv contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "conv contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU conv contract must register into the binding table",
    );
}

/// The authored CPU **padding** kernel contract, embedded into the binary (the
/// PRODUCTION `include_str!`). `register_cpu_padding_from_contract` parses +
/// lowers it and binds the FULL family (mode-unified forward `Pad` +
/// `PadBackward`) FROM THE CONTRACT.
const CPU_PADDING_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/padding.fkc.md");

/// Register the FULL CPU padding family — mode-unified forward `Pad` × 6 dtypes
/// = 6 + `PadBackward` × 4 dtypes = 4 bindings (all key `[T, T]`) — by IMPORTING
/// its FKC kernel contract, the twelfth production FKC consumer.
///
/// - **Forward `Pad`** — the ONE mode-unified `## pad` section declares a BASE
///   `entry_point` (`…::pad_cpu`, a synthetic umbrella — the three real mode
///   byte-kernels are `pad_{const,reflect,replicate}_cpu`) + the 6 production
///   dtypes (`U8/U32/BF16/F16/F32/F64`), so the importer's §3.4 fan-out resolves
///   `pad_cpu_<dtype>` — every dtype variant mapping to the ONE mode-dispatching
///   `pad_cpu_wrapper` (which selects Constant/Reflect/Replicate at runtime via
///   `mode_tag`, incl. the reflect `before/after <= n-1` validation) through the
///   production [`crate::fkc::CpuLinkRegistry`] (chaining
///   [`crate::fkc::CPU_PADDING_ENTRY_POINTS`]). Key `[T, T]` (input +
///   `passthrough(input)` output; the in_shape/out_shape/padding/mode_tag/
///   fill_bytes ride in `OpParams::Pad`, NOT the dtype-list) — byte-for-byte the
///   deleted hand-written `table.register(Pad, &unary(t), …)` regs. The three
///   per-mode sections (`pad_const_cpu` / `pad_reflect_cpu` / `pad_replicate_cpu`)
///   and the `pad_walk_cpu` helper are `registrable: false` (§3.10 describe-only
///   mode documentation) and never resolve.
/// - **`PadBackward`** — each per-dtype section (`## pad_backward_f32`, …) carries
///   a SPECIFIC single-dtype `entry_point` (`…::pad_backward_f32`) resolved AS-IS
///   (no fan), key `[T, T]` (grad_out + `passthrough(grad_out)` grad_in; the
///   in_shape/out_shape/padding/mode_tag ride in `OpParams::PadBackward`) —
///   byte-for-byte the deleted `table.register(PadBackward, &unary(t), …)` regs.
///   `PadBackward` is per-dtype (unlike the dtype-agnostic forward `Pad`) because
///   gradient accumulation is typed (bf16/f16/f32 widen the scratch accumulator
///   to f64).
///
/// Behavior-preserving vs. the deleted hand-written path: identical kernels +
/// caps (contiguous-only); the binding's `kernel_source` becomes the contract's
/// `"portable-cpu"` tag and its precision the contract's bit-stable claim. Cost
/// is preserved because this runs BEFORE `fill_unset_cpu_cost`, which upgrades
/// the imported entries' `unknown_cost` sentinel to the same OpKind cost fn every
/// CPU primitive gets. The family declares NO fused ops, so `register_into`'s
/// required fused argument is a local throwaway that provably stays empty.
fn register_cpu_padding_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_PADDING_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU padding contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "padding contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU padding contract must register into the binding table",
    );
}

/// The authored CPU **shape-ops** kernel contract, embedded into the binary (the
/// PRODUCTION `include_str!`). `register_cpu_shape_ops_from_contract` parses +
/// lowers it and binds the MIGRATED subset of the family FROM THE CONTRACT.
const CPU_SHAPE_OPS_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/shape-ops.fkc.md");

/// Register the MIGRATED CPU shape-ops subset — Flip + Roll + Concat +
/// MaskedFill + WriteSlice + WriteSliceRotating (dtype-agnostic byte kernels,
/// fanned per production dtype) and the four per-dtype CumSum kernels — by
/// IMPORTING its FKC kernel contract, the thirteenth production FKC consumer.
///
/// - **Flip** / **Roll** (key `[T, T]`, 6 dtypes each) and **Concat** (key
///   `[T, T]`, 9 dtypes) and **MaskedFill** (key `[T, U8, T]`, 6 dtypes) and
///   **WriteSlice** / **WriteSliceRotating** (key `[T, T]`, 6 dtypes each) are
///   ONE dtype-agnostic wrapper each. Their sections declare a BASE `entry_point`
///   (`…::flip_cpu`, …) + an enumerated `dtypes` list trimmed to the wired set,
///   so the §3.4 fan-out resolves `<base>_<dtype>` (`flip_cpu_f32`, …) — a
///   fabricated per-dtype symbol that every variant maps to the SAME wrapper via
///   [`crate::fkc::CPU_SHAPE_OPS_ENTRY_POINTS`]. MaskedFill's `input` is the sole
///   varying operand; `mask` stays the fixed U8 slot and `out: passthrough(input)`,
///   so the fan emits `[T, U8, T]` — byte-for-byte the deleted
///   `table.register(MaskedFill, &masked_dtypes(t), …)` regs. **WriteSlice /
///   WriteSliceRotating** model `dest` as the in-place OUTPUT slot (not a key
///   input) and — for the rotating op — the U32 `position` as a NON-KEY runtime
///   operand (both handled by the executor's dedicated `Op::WriteSlice{,Rotating}`
///   arms), so a source-only + `out: passthrough(source)` section keys
///   `[T_source, T_out]` = `[T, T]` — byte-for-byte the deleted
///   `table.register(WriteSlice{,Rotating}, &unary(t), …)` regs and matching
///   `build_lookup_dtypes`' canonicalization exactly.
/// - **CumSum** (key `[T, T]`, 4 dtypes) is per-dtype typed accumulation, so each
///   `cumsum_<dt>` section carries a SPECIFIC single-dtype `entry_point` resolved
///   AS-IS (no fan) to its OWN typed wrapper.
///
/// **DEFERRED (describe-only, NOT imported).** `contiguize` and `triangular` are
/// `registrable: false` chassis/describe-only sections (no `OpKind::Contiguize` —
/// it is an executor materialize pass; `triangular` is the umbrella backing the
/// two hand-written `Triu`/`Tril` OpKinds with distinct wrappers, not a keyable
/// section). Those hand-written `Triu`/`Tril` regs stay authoritative.
///
/// Behavior-preserving vs. the deleted hand-written path: identical kernels +
/// caps (contiguous-only); the binding's `kernel_source` becomes the contract's
/// `"portable-cpu"` tag and its precision the contract's bit-stable claim. Cost
/// is preserved because this runs BEFORE `fill_unset_cpu_cost`, which upgrades
/// the imported entries' `unknown_cost` sentinel to the same OpKind cost fn every
/// CPU primitive gets. The family declares NO fused ops, so `register_into`'s
/// required fused argument is a local throwaway that provably stays empty.
fn register_cpu_shape_ops_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_SHAPE_OPS_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU shape-ops contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "shape-ops contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU shape-ops contract must register into the binding table",
    );
}

/// The authored CPU **matmul** kernel contract, embedded into the binary (the
/// PRODUCTION `include_str!`). `register_cpu_matmul_from_contract` parses +
/// lowers it and binds the FULL portable family FROM THE CONTRACT.
const CPU_MATMUL_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/matmul.fkc.md");

/// Register the CPU **matmul** family FULLY — bare batched `MatMul` (6 dtypes,
/// key `[T, T, T]`) + fused `FusedLinear` (matmul + bias-add, 4 dtypes, key
/// `[T, T, T, T]`) = 10 bindings — by IMPORTING its FKC kernel contract, the
/// fourteenth production FKC consumer. FKC is unconditional core infrastructure,
/// so this is the ONE registration path for the PORTABLE kernels: every
/// hand-written `table.register(MatMul / FusedLinear, …)` portable-CPU call is
/// DELETED.
///
/// Each per-(op, dtype) section (`## matmul_f32`, `## fused_linear_f32`, …) is a
/// SPECIFIC single-dtype contract with a concrete `entry_point`
/// (`…::matmul_f32`), so none of them dtype-fan — the importer resolves each
/// declared symbol AS-IS through the production [`crate::fkc::CpuLinkRegistry`]
/// (now chaining [`crate::fkc::CPU_MATMUL_ENTRY_POINTS`]) to the exact wrapper
/// fn-pointer the CPU backend exposes. MatMul is 2-input (lhs, rhs) →
/// `passthrough(lhs)` output; FusedLinear is 3-input (a, b, bias) →
/// `passthrough(a)` output with `bias` a REQUIRED 1-D `[N]` operand (NOT
/// `optional`, so a single 4-slot key per dtype — no optional {absent, present}
/// fan). Both reuse `OpParams::Matmul` for shape, so every batch-geometry knob
/// (lhs_batch_dims/rhs_batch_dims/m/n/k) rides in `OpParams`, NOT the dtype-list.
/// The float variants (F32/F64/BF16/F16) accumulate in f32/native; the integer
/// MatMul variants (I8/U8) accumulate in i32 and SATURATE on store.
///
/// **Alternatives at the same key.** The binding table supports MULTIPLE ranked
/// kernels per `(op, dtypes, backend)` key. This importer registers ONLY the
/// portable (`kernel_source: "portable-cpu"`) kernels the contract covers; the
/// MKL / AOCL BLAS siblings live in SEPARATE external backend crates
/// (`fuel-mkl-cpu-backend` / `fuel-aocl-cpu-backend`), register through the
/// exported dispatch helpers with their own `"mkl"`/`"aocl"` `kernel_source`
/// tags as ranked alternatives at these SAME keys, and are out of scope here
/// (untouched — never registered in this crate's default build).
///
/// Behavior-preserving vs. the deleted hand-written path: identical kernels +
/// caps (contiguous-only, GQA-divisible batch); the binding's `kernel_source`
/// becomes the contract's `"portable-cpu"` tag and its precision the contract's
/// bit-stable claim. Cost is preserved because this runs BEFORE
/// `fill_unset_cpu_cost`, which upgrades the imported entries' `unknown_cost`
/// sentinel to the same OpKind cost fn every CPU primitive gets. The family
/// declares NO fused ops (the SEPARATE `FusedKernelRegistry`
/// `FusedOps::FUSED_LINEAR` seam is hand-written, in
/// `register_default_fused_kernels`, and stays untouched — "fused" in
/// `FusedLinear` names an intra-op matmul+bias fusion, NOT a graph `FusedOpId`),
/// so `register_into`'s required fused argument is a local throwaway that
/// provably stays empty. The quant `QMatMul` / `Nf4Matmul` OpKinds have their
/// own contracts and stay hand-written.
fn register_cpu_matmul_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_MATMUL_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU matmul contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "matmul contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU matmul contract must register into the binding table",
    );
}

/// The authored CPU **attention** kernel contract, embedded into the binary (the
/// PRODUCTION `include_str!`). `register_cpu_attention_from_contract` parses +
/// lowers it and binds the MIGRATED (`KernelBindingTable`) subset FROM THE
/// CONTRACT.
const CPU_ATTENTION_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/attention.fkc.md");

/// Register the CPU **attention** family's `KernelBindingTable` path FROM its FKC
/// kernel contract, the fifteenth production FKC consumer: forward `FlashAttn`
/// (4 dtypes) + `FlashAttnBackward{Q,K,V}` (3 selectors × 4 dtypes), each
/// `alibi_slopes`-optional section fanning into BOTH the no-alibi and with-alibi
/// keys = 4 × 4 × 2 = 32 bindings. FKC is unconditional core infrastructure, so
/// this is the ONE registration path for the migrated subset: the hand-written
/// `table.register(FlashAttn / FlashAttnBackward{Q,K,V}, …)` regs (both alibi
/// keys) are DELETED.
///
/// Each per-(op, dtype) section (`## flash_attn_f32`,
/// `## flash_attn_backward_q_f32`, …) declares a SPECIFIC single-dtype
/// `entry_point`, so none of them dtype-fan — the importer resolves each declared
/// symbol AS-IS through the production [`crate::fkc::CpuLinkRegistry`] (now
/// chaining [`crate::fkc::CPU_ATTENTION_ENTRY_POINTS`]) to the exact wrapper
/// fn-pointer. Because the contract marks `alibi_slopes` as `optional: true` (the
/// LAST input), the importer's key-builder (`fkc/lower.rs`
/// `assemble_dtype_variants`) fans each section into BOTH the no-alibi key
/// (`[q,k,v,out]` forward / `[q,k,v,do,out]` backward) and the with-alibi key
/// (`+alibi`) — both resolving the SAME `entry_point`/wrapper (the CPU wrapper
/// handles the presence/absence of the alibi operand). Forward FlashAttn AND the
/// three backward selectors share the single `OpParams::FlashAttn` carrier (there
/// is NO dedicated backward `OpParams` variant), so every softmax/mask geometry
/// knob rides in `OpParams`, NOT the dtype-list.
///
/// **PagedAttn is REGISTRABLE (`registrable: true`) and contract-sourced too.**
/// Its paged KV pool carries an `fdx.gather: paged_blocks` operand (§3.9.1), but
/// that block is import METADATA describing what an FDX view of the pool will
/// someday carry (FDX §6.9: "Description only: no cost, no decision"), NOT a
/// registration dependency: the as-built ABI passes `block_table` /
/// `context_lens` as ordinary U32 graph inputs + the geometry in
/// `OpParams::PagedAttn`, and the kernel reads them directly. So the importer
/// validates the gather block for COHERENCE (kind / requires_ext /
/// symbolic_extent / real block_table+context_lens roles) and registers the four
/// paged sections onto the `KernelBindingTable` at `(OpKind::PagedAttn,
/// [T,T,T,U32,U32,T](+[T]), Cpu)` — the optional-operand fan builds both alibi
/// keys, byte-for-byte the deleted hand-written `paged_attn_{no,with}_alibi`
/// regs. What remains [consumer-ahead] is only the FDX VIEW/materialize seam
/// (`view_with_gather` + `Capability::DlpackExtGather`), which has no consumer
/// yet. (The SEPARATE `FusedKernelRegistry` `PAGED_ATTN` seam stays hand-written,
/// exactly like `FLASH_ATTN`.)
///
/// Behavior-preserving vs. the deleted hand-written path (both alibi keys):
/// identical kernels + caps (contiguous-only); the binding's `kernel_source`
/// becomes the contract's `"portable-cpu"` tag and its precision the contract's
/// bit-stable claim. Cost is preserved because this runs BEFORE
/// `fill_unset_cpu_cost`, which upgrades the imported entries' `unknown_cost`
/// sentinel to the same OpKind cost fn every CPU primitive gets (the CPU cost
/// dispatcher has `FlashAttn`/`PagedAttn` arms; the backward selectors keep
/// `unknown_cost`, exactly as under the deleted hand-written regs — they are not
/// on the cost-coverage lint's enumerated set). The family declares NO fused ops
/// (all sections are `op_kind:`, incl. the now-registrable PagedAttn), so
/// `register_into`'s required fused argument is a local throwaway that provably
/// stays empty. The
/// SEPARATE `FusedKernelRegistry` FLASH_ATTN* / PAGED_ATTN seam
/// (`register_default_fused_kernels`) is hand-written and stays untouched.
fn register_cpu_attention_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_ATTENTION_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU attention contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "attention contract declares no fused ops (all op_kind, incl. PagedAttn)",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU attention contract must register into the binding table",
    );
}

/// The authored CPU **in-place scalar-param** kernel contract, embedded into the
/// binary (the PRODUCTION `include_str!`). `register_cpu_inplace_from_contract`
/// parses + lowers it and binds the WHOLE family FROM THE CONTRACT.
const CPU_INPLACE_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/inplace-unary-affine.fkc.md");

/// Register the CPU **in-place scalar-param** family's `KernelBindingTable` path
/// FROM its FKC kernel contract, the sixteenth production FKC consumer: 21
/// in-place unary ops (`<Op>Inplace`, each per-op section fanning `[F32,F64,
/// BF16,F16]` = 84 bindings) + `InplaceAffine` / `ClampInplace` / `PowIInplace`
/// (per-dtype single sections, 4 each = 12) = 96 bindings. FKC is unconditional
/// core infrastructure, so this is the ONE registration path for the whole
/// family: the hand-written `table.register(<Op>Inplace / InplaceAffine /
/// ClampInplace / PowIInplace, &unary(dt), …)` regs are DELETED.
///
/// Each per-op unary section (`## relu_inplace`, …) declares a BASE `entry_point`
/// (`…::relu_inplace`) + enumerates `dtypes: [F32,F64,BF16,F16]`, so the
/// importer's §3.4 multi-dtype fan-out resolves `<base>_<dtype>`
/// (`relu_inplace_f32`) through the production [`crate::fkc::CpuLinkRegistry`]
/// (now chaining [`crate::fkc::CPU_INPLACE_ENTRY_POINTS`]) to the exact wrapper
/// fn-pointer. The affine / clamp / powi sections are per-dtype SINGLE sections,
/// so they do NOT fan — their specific `<op>_inplace_<dt>` symbol resolves AS-IS
/// (the affine rows carry the three-way naming skew: symbol `affine_inplace_<dt>`
/// → wrapper `inplace_affine_<dt>_cpu_wrapper`, words swapped). Every binding
/// keys `[T, T]` (the single `out` operand + its `passthrough(out)` mirror; the
/// executor's `WorkItemKind::InplaceKernel` arm passes the target as
/// `outputs[0]`), byte-for-byte the deleted `&unary(dt)` regs. Scalar params
/// (affine `mul`/`add`, clamp `min`/`max`, powi `exp`) ride in
/// `OpParams::{Affine, Clamp, PowI}`, NOT the dtype-list.
///
/// The `## unary_inplace` chassis umbrella is `registrable: false` (§3.10
/// describe-only) — it binds no OpKind (each named op pins its own distinct
/// `<Op>Inplace`), so it never lowers/resolves and is excluded from
/// registration.
///
/// Behavior-preserving vs. the deleted hand-written path: identical kernels +
/// caps (contiguous-only, empty caps); the binding's `kernel_source` becomes the
/// contract's `"portable-cpu"` tag and its precision the contract's bit-stable
/// claim. Cost is preserved because this runs BEFORE `fill_unset_cpu_cost`: the
/// in-place OpKinds are NOT on the CPU cost-coverage lint's enumerated set and
/// have no arm in `default_cost_for_op_kind`, so they keep the `unknown_cost`
/// sentinel exactly as under the deleted hand-written regs. The family declares
/// NO fused ops (all sections are `op_kind:` or the describe-only umbrella), so
/// `register_into`'s required fused argument is a local throwaway that provably
/// stays empty.
fn register_cpu_inplace_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_INPLACE_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU in-place contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "in-place contract declares no fused ops (all op_kind or describe-only umbrella)",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU in-place contract must register into the binding table",
    );
}

/// The authored CPU **cast** kernel contract, embedded into the binary (the
/// PRODUCTION `include_str!`). `register_cpu_cast_from_contract` parses + lowers
/// it and binds the FULL directed-pair matrix FROM THE CONTRACT.
const CPU_CAST_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/cast.fkc.md");

/// Register the CPU **cast** family FULLY — the COMPLETE directed-pair matrix,
/// every ordered pair of the 11 real numeric dtypes
/// {F32,F64,F16,BF16,F8E4M3,U8,I8,U32,I16,I32,I64} with identity excluded =
/// 11 × 10 = 110 bindings (all key `[SRC, DST]`) — by IMPORTING its FKC kernel
/// contract.
///
/// Each per-pair section (`## cast_f64_to_f32`, …) declares a SPECIFIC
/// single-dtype `src` input + a `fixed(DST)` output, so none of them dtype-fan —
/// the importer's key-builder emits `[SRC, DST]` (the src input dtype + the
/// `fixed(DST)` output dtype, byte-for-byte the deleted hand-written
/// `table.register(Cast, &[SRC, DST], …)` regs) and resolves the section's
/// SPECIFIC `cast_<src>_to_<dst>` byte-kernel `entry_point` AS-IS through the
/// production [`crate::fkc::CpuLinkRegistry`] (chaining
/// [`crate::fkc::CPU_CAST_ENTRY_POINTS`]). Because the binding lookup is keyed on
/// the TARGET dtype, all 10 of a target's source pairs bind the SAME per-target
/// `cast_to_<dst>_cpu_wrapper` (which `match`es on the source dtype internally) —
/// 10 distinct real byte-kernel entry_points → 1 wrapper (the synthetic-umbrella
/// precedent). This is the SOLE registration path for the whole family; every
/// hand-written `table.register(Cast, …)` reg is DELETED. Identity pairs
/// (`[T, T]`) are never registered — the optimizer elides identity `Cast` before
/// dispatch, and the per-target wrappers carry no identity arm.
///
/// Behavior-preserving vs. the deleted hand-written path: identical kernels +
/// caps (contiguous-only); the binding's `kernel_source` becomes the contract's
/// `"portable-cpu"` tag and its precision the contract's bit-stable claim. Cost
/// is preserved because this runs BEFORE `fill_unset_cpu_cost`, which upgrades
/// the imported entries' `unknown_cost` sentinel to the same OpKind cost fn every
/// CPU primitive gets. The family declares NO fused ops, so `register_into`'s
/// required fused argument is a local throwaway that provably stays empty.
fn register_cpu_cast_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_CAST_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU cast contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "cast contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU cast contract must register into the binding table",
    );
}

/// The authored CPU **indexing / gather / scatter** kernel contract, embedded
/// into the binary (the PRODUCTION `include_str!`).
/// `register_cpu_indexing_from_contract` parses + lowers it and binds the FULL
/// family (IndexSelect + Gather + IndexAdd + ScatterAdd) FROM THE CONTRACT.
const CPU_INDEXING_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/cpu/indexing.fkc.md");

/// Register the FULL CPU indexing family — IndexSelect (9 dtypes, key
/// `[T, U32, T]`) + Gather (9 dtypes, key `[T, U32, T]`) + IndexAdd (4 dtypes,
/// key `[T, U32, T, T]`) + ScatterAdd (4 dtypes, key `[T, U32, T, T]`) = 26
/// bindings — by IMPORTING its FKC kernel contract, the eighteenth production
/// FKC consumer.
///
/// - **IndexSelect** / **Gather** — the ONE section each declares a BASE
///   `entry_point` (`…::index_select_cpu` / `…::gather_cpu`, a dtype-agnostic
///   byte copy) + an enumerated `dtypes` list, so the importer's §3.4 fan-out
///   resolves `<base>_<dtype>` — every dtype variant mapping to the ONE
///   dtype-agnostic `index_select_cpu_wrapper` / `gather_cpu_wrapper` through the
///   production [`crate::fkc::CpuLinkRegistry`] (chaining
///   [`crate::fkc::cpu_link::CPU_INDEXING_ENTRY_POINTS`]). The `indices` operand
///   is the FIXED U32 slot and `out: passthrough(source)`, so the fan emits key
///   `[T, U32, T]` — byte-for-byte the deleted hand-written
///   `table.register(IndexSelect/Gather, &index_select(dt)/&gather_dts(dt), …)`
///   regs. The contract's dtype list was trimmed to production's 9 wired dtypes
///   (F32/F64/BF16/F16/U32/U8/I16/I32/I64); I8 is describable (byte-agnostic) but
///   NOT wired in production, so it is dropped from the contract.
/// - **IndexAdd** / **ScatterAdd** — per-dtype typed accumulation (f32/f64
///   native, bf16/f16 widen to an f32 accumulator; out seeded from `base` then
///   `+= src`), so each `index_add_<dt>` / `scatter_add_<dt>` section carries a
///   SPECIFIC single-dtype `entry_point` resolved AS-IS (no fan), key
///   `[T, U32, T, T]` (`base`, U32 `indices`, `src`, `passthrough(base)` output)
///   — byte-for-byte the deleted `table.register(IndexAdd/ScatterAdd, …)` regs.
///
/// The `indices` slot being a FIXED single-dtype (U32) in every section means
/// there is NO independent index-dtype axis and no multi-axis
/// `FanoutDtypeMismatch` deferral — the whole family migrates (unlike the metal
/// indexing situation).
///
/// Behavior-preserving vs. the deleted hand-written path: identical kernels +
/// caps (contiguous-only); the binding's `kernel_source` becomes the contract's
/// `"portable-cpu"` tag and its precision the contract's bit-stable claim. Cost
/// is preserved because this runs BEFORE `fill_unset_cpu_cost`, which upgrades
/// the imported entries' `unknown_cost` sentinel to the same OpKind cost fn every
/// CPU primitive gets. The family declares NO fused ops, so `register_into`'s
/// required fused argument is a local throwaway that provably stays empty.
fn register_cpu_indexing_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(CPU_INDEXING_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored CPU indexing contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "indexing contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider.register_into(table, &mut fused).expect(
        "CPU indexing contract must register into the binding table",
    );
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
    // (MatMul's (lhs, rhs, out) key shape is now built by the FKC importer from
    // docs/kernel-contracts/cpu/matmul.fkc.md — see
    // register_cpu_matmul_from_contract; the hand-written `binary` closure was
    // removed with its regs.)
    let u8_dt  = DType::U8;
    // (Rope's (x, cos, sin, out) key shape is now built by the FKC importer
    // from docs/kernel-contracts/cpu/rope.fkc.md — see
    // register_cpu_rope_from_contract; the hand-written `rope_dts` closure was
    // removed with its regs.)
    // (Conv2D/ConvTranspose2D are now FULLY built by the FKC importer from
    // docs/kernel-contracts/cpu/conv.fkc.md — see register_cpu_conv_from_contract.
    // The contract marks `bias` as `optional: true`, so the importer's key-builder
    // now emits BOTH the no-bias (x, w, out) and with-bias (x, w, bias, out) keys
    // per (op, dtype); the hand-written `conv2d_with_bias` / `conv2d_no_bias`
    // closures + their regs were all removed.)
    // (FusedLinear's (lhs, rhs, bias, out) key shape is now built by the FKC
    // importer from docs/kernel-contracts/cpu/matmul.fkc.md — see
    // register_cpu_matmul_from_contract; the hand-written `fused_linear` closure
    // was removed with its regs.)
    // FlashAttn / FlashAttnBackward{Q,K,V} / PagedAttn key-shape closures were
    // DELETED with the migration to FKC-contract registration (the importer's
    // optional-operand fan builds both alibi keys from the contract).
    // (IndexSelect/Gather's (data, indices, out) = [T, U32, T] and IndexAdd/
    // ScatterAdd's (base, indices, src, out) = [T, U32, T, T] key shapes are now
    // built by the FKC importer from docs/kernel-contracts/cpu/indexing.fkc.md —
    // see register_cpu_indexing_from_contract; the hand-written `index_select` /
    // `gather_dts` / `index_add_dts` / `scatter_add` closures were removed with
    // their regs.)

    // Elementwise binary family (8 ops × 4 dtypes = 32 bindings:
    // Add/Sub/Mul/Div/Maximum/Minimum/Pow/Rem × F32/F64/BF16/F16).
    //
    // This family is the FIRST production FKC-contract consumer: it is
    // IMPORTED from its kernel contract
    // (docs/kernel-contracts/cpu/elementwise-binary.fkc.md), resolving every
    // `entry_point` symbol through the production `CpuLinkRegistry` to the
    // exact same wrapper fn-pointers the CPU backend exposes. The hand-written
    // `table.register(...)` calls this family used to carry are DELETED — FKC
    // is unconditional core infrastructure, so this is the sole registration
    // path (no build config, no fallback). Placed here — BEFORE the
    // `fill_unset_cpu_*` passes — so the imported entries pick up the same CPU
    // cost fill every other CPU primitive gets.
    register_cpu_binary_from_contract(table);

    // Out-of-place scalar-param family (affine/clamp/powi × 4 dtypes +
    // powi_backward × 4 = 16 bindings). Second production FKC-contract
    // consumer: IMPORTED from docs/kernel-contracts/cpu/affine-clamp-powi.fkc.md
    // via the same `CpuLinkRegistry`. The hand-written `table.register(...)`
    // calls (Affine / ClampElementwise / PowIElementwise / PowIElementwiseBackward)
    // are DELETED — this is the sole registration path. Placed BEFORE the
    // `fill_unset_cpu_*` passes so the imported entries pick up the CPU cost
    // fill. (The in-place InplaceAffine/ClampInplace/PowIInplace family and the
    // FusedOps::POWI_BACKWARD fused-registry seam are separate and untouched.)
    register_cpu_affine_clamp_powi_from_contract(table);

    // Elementwise unary (22 ops × 4 dtypes = 88 bindings: relu/neg/sqr/sqrt/
    // recip/abs/tanh/exp/log/sin/cos/sigmoid/silu/step/gelu/floor/ceil/round/
    // sign/erf/gelu_erf/rsqrt × F32/F64/BF16/F16). Third production FKC-contract
    // consumer and the FIRST user of the §3.4 multi-dtype fan-out: IMPORTED from
    // docs/kernel-contracts/cpu/elementwise-unary.fkc.md via the same
    // `CpuLinkRegistry`. Each per-op section declares a BASE entry_point that the
    // importer expands to `<base>_<dtype>`. The hand-written `table.register(...)`
    // calls (the F32/F64/BF16/F16 unary blocks + the rounding/sign/erf/gelu_erf/
    // rsqrt block) are DELETED — this is the sole registration path. Placed
    // BEFORE the `fill_unset_cpu_*` passes so the imported entries pick up the CPU
    // cost fill. gelu_tanh (GeluElementwise) and gelu_erf (GeluErfElementwise)
    // stay DISTINCT.
    register_cpu_unary_from_contract(table);

    // Compare (6 ops × 4 dtypes = 24, key [T, T, U8]) + where (1 op × 4 dtypes
    // = 4, key [U8, T, T, T]) = 28 bindings. Fourth production FKC-contract
    // consumer: IMPORTED from docs/kernel-contracts/cpu/compare-where.fkc.md via
    // the same `CpuLinkRegistry`. The 24 compare thunks resolve their declared
    // `_u8`-suffixed symbols AS-IS; the single `where_kernel` section rides the
    // §3.4 multi-dtype fan-out (BASE `…::where` → `where_{f32,f64,bf16,f16}`)
    // and the `passthrough(a)` output rule. The hand-written `table.register(...)`
    // calls (the compare block + the `where_dts` block) are DELETED — this is the
    // sole registration path. Placed BEFORE the `fill_unset_cpu_*` passes so the
    // imported entries pick up the CPU cost fill.
    register_cpu_compare_where_from_contract(table);

    // Per-axis reduce (Sum/Mean/Max/Min × 4 dtypes = 16 bindings, key [T, T]).
    // Fifth production FKC-contract consumer: IMPORTED from
    // docs/kernel-contracts/cpu/reduce.fkc.md via the same `CpuLinkRegistry`.
    // Each per-(op,dtype) section carries a SPECIFIC single-dtype entry_point
    // (`sum_reduce_f32`, …) resolved AS-IS (no fan-out). The hand-written
    // `table.register(...)` calls (the F32/F64 + BF16/F16 reduce blocks) are
    // DELETED — this is the sole registration path. Placed BEFORE the
    // `fill_unset_cpu_*` passes so the imported entries pick up the CPU cost
    // fill. The `## reduce` chassis is `registrable: false` (§3.10) and the
    // f32-only argmax/argmin sections are deferred (registrable: false); the
    // hand-written Arg{Max,Min}Dim regs below stay authoritative.
    register_cpu_reduce_from_contract(table);

    // Broadcast-target reduce-to (ReduceSumTo / ReduceMaxTo × 4 dtypes, key
    // [T, T]; ReduceMaxToBackward × 4 dtypes, key [T, T, T] = 12 bindings).
    // Sixth production FKC-contract consumer: IMPORTED from
    // docs/kernel-contracts/cpu/reduce-to.fkc.md via the same `CpuLinkRegistry`.
    // Each per-(op,dtype) section carries a SPECIFIC single-dtype entry_point
    // (`reduce_sum_to_f32`, …) resolved AS-IS (no fan-out). The hand-written
    // `table.register(...)` calls (the ReduceSumTo/ReduceMaxTo forward block +
    // the ReduceMaxToBackward block) are DELETED — this is the sole registration
    // path. Placed BEFORE the `fill_unset_cpu_*` passes so the imported entries
    // pick up the CPU cost fill. The `## reduce_to` chassis is
    // `registrable: false` (§3.10) and never registers.
    register_cpu_reduce_to_from_contract(table);

    // Last-dim NORM forward (Softmax / LogSoftmax / RmsNorm / LayerNorm × 4
    // dtypes = 16 bindings, key [T, T]). Seventh production FKC-contract
    // consumer: IMPORTED from docs/kernel-contracts/cpu/norm.fkc.md via the same
    // `CpuLinkRegistry`. Each per-(op,dtype) section carries a SPECIFIC
    // single-dtype entry_point (`softmax_last_dim_f32`, `rms_norm_last_dim_f32`,
    // …) resolved AS-IS (no fan-out). The RMS/LayerNorm kernels are the BARE
    // normalization (no affine gamma/beta operand); outer_count/last_dim/eps ride
    // in OpParams, so the key stays [T, T]. The hand-written `table.register(...)`
    // calls (the LogSoftmaxLastDim block above + the SoftmaxLastDim / RmsNormLastDim
    // / LayerNormLastDim block below) are DELETED — this is the sole registration
    // path. Placed BEFORE the `fill_unset_cpu_*` passes so the imported entries
    // pick up the CPU cost fill. The norm ops' SEPARATE FusedKernelRegistry seam
    // (register_default_fused_kernels) and the norm-BACKWARD hand-written regs stay
    // untouched.
    register_cpu_norm_from_contract(table);

    // Last-dim NORM BACKWARD (Softmax / LogSoftmax / RmsNorm / LayerNorm backward
    // × 4 dtypes = 16 bindings, key [T, T, T]). Eighth production FKC-contract
    // consumer and the BACKWARD sibling of the norm-forward importer above:
    // IMPORTED from docs/kernel-contracts/cpu/norm-backward.fkc.md via the same
    // `CpuLinkRegistry`. Each per-(op,dtype) section carries a SPECIFIC
    // single-dtype entry_point (`softmax_last_dim_backward_f32`,
    // `rms_norm_last_dim_backward_f32`, …) resolved AS-IS (no fan-out). The BARE
    // backward takes TWO inputs (y/x + upstream gradient g) → ONE passthrough
    // output, so the key is [T, T, T]; outer_count/last_dim/eps ride in OpParams.
    // The hand-written `table.register(...)` calls (the LogSoftmaxLastDimBackward
    // block + the SoftmaxLastDimBackward / LayerNormLastDimBackward /
    // RmsNormLastDimBackward block below) are DELETED — this is the sole
    // registration path. Placed BEFORE the `fill_unset_cpu_*` passes so the
    // imported entries pick up the CPU cost fill. The norm-backward ops' SEPARATE
    // FusedKernelRegistry seam (register_default_fused_kernels) stays untouched.
    register_cpu_norm_backward_from_contract(table);

    // RoPE (rotary position embedding; 1 op × 4 dtypes = 4 bindings, key
    // [T, T, T, T]). Ninth production FKC-contract consumer: IMPORTED from
    // docs/kernel-contracts/cpu/rope.fkc.md via the same `CpuLinkRegistry`.
    // Each per-dtype section carries a SPECIFIC single-dtype entry_point
    // (`rope_f32`, …) resolved AS-IS (no fan-out). RoPE takes THREE inputs
    // (x + the precomputed cos/sin tables, all one dtype) → ONE passthrough(x)
    // output, so the key is [T, T, T, T]; outer_count/seq/head_dim ride in
    // OpParams::Rope. The hand-written `table.register(Rope, ...)` calls (the
    // rope_dts block below) are DELETED — this is the sole registration path.
    // Placed BEFORE the `fill_unset_cpu_*` passes so the imported entries pick
    // up the CPU cost fill. RoPE's SEPARATE FusedKernelRegistry seam
    // (register_default_fused_kernels, FusedOps::ROPE) stays untouched.
    register_cpu_rope_from_contract(table);

    // CPU SSM / Mamba family — the FULL family: FusedSoftmaxCrossEntropy
    // (key [T, I64, F32]) + CausalConv1d (key [T, T, T, T]) + SelectiveScan +
    // SsdChunkScan (key [T; 6]), 4 ops × 4 dtypes = 16 bindings. Tenth
    // production FKC-contract consumer: IMPORTED from
    // docs/kernel-contracts/cpu/ssm.fkc.md via the same `CpuLinkRegistry`. Each
    // per-(op,dtype) section carries a SPECIFIC single-dtype entry_point
    // (`fused_softmax_cross_entropy_f32`, `selective_scan_f32`, …) resolved AS-IS
    // (no fan-out). The two SCAN ops return a `return.bundle` multi-output
    // (Option C, one buffer [y ; last_state]); the importer now appends the
    // bundle's primary-slot dtype to the key tail, so they key [T; 6]
    // byte-for-byte the deleted hand-written regs. Every hand-written
    // `table.register(...)` call for the whole family is DELETED — this is its
    // sole registration path. Placed BEFORE the `fill_unset_cpu_*` passes so the
    // imported entries pick up the CPU cost fill.
    register_cpu_ssm_from_contract(table);

    // NonZeroIndices — the keystone primitive for data-dependent dynamic
    // shapes (SSM decode / MoE sparsity / MLA KV compression). One input
    // `x` → one bundled output `[indices [capacity] U32 ; count [1] U32]`;
    // the binding key is `[input_dtype, U32]` (input + primary output-slot
    // dtype). Registered DIRECTLY (not yet FKC-contract-sourced): the
    // op's U32 output slot is a *constant* dtype, not a `passthrough` of a
    // T input, which the current contract key-builder does not express —
    // so this bring-up primitive registers by hand and is a candidate to
    // migrate once a data-dependent-ops contract family grows. Placed
    // BEFORE the `fill_unset_cpu_*` passes so the entries pick up the CPU
    // precision + cost fill (UNAUDITED → PRIMITIVE_DETERMINISTIC_CPU,
    // unknown_cost → the OpKind cost fn).
    table.register(NonZeroIndices, &[f32_dt, u32_dt], cpu, nonzero_indices_f32_cpu_wrapper);
    table.register(NonZeroIndices, &[u32_dt, u32_dt], cpu, nonzero_indices_u32_cpu_wrapper);

    // CPU 2D-convolution family — FULLY contract-sourced: Conv2D +
    // ConvTranspose2D at BOTH the no-bias key [T, T, T] (x, weight, out) AND the
    // with-bias key [T, T, T, T] (x, weight, bias + passthrough(x) output), 2 ops
    // × 4 dtypes × 2 operand-counts = 16 bindings. Eleventh production
    // FKC-contract consumer: IMPORTED from docs/kernel-contracts/cpu/conv.fkc.md
    // via the same `CpuLinkRegistry`. Each per-(op,dtype) section carries a
    // SPECIFIC single-dtype entry_point (`conv2d_f32`, `conv_transpose2d_f32`, …)
    // resolved AS-IS (no dtype fan-out); every geometry knob rides in
    // OpParams::{Conv2D, ConvTranspose2D}. The contract marks `bias` as
    // `optional: true`, so the importer's optional-operand key-builder fans each
    // section into both the no-bias and with-bias keys (same wrapper handles 2 or
    // 3 inputs) — ALL hand-written conv `table.register(...)` calls are DELETED,
    // this is the sole registration path for the whole family. Placed BEFORE the
    // `fill_unset_cpu_*` passes so the imported entries pick up the CPU cost fill.
    // The conv ops' SEPARATE FusedKernelRegistry seam
    // (register_default_fused_kernels, FusedOps::{CONV2D, CONV_TRANSPOSE2D}) stays
    // untouched.
    register_cpu_conv_from_contract(table);

    // CPU padding family (FULL) — mode-unified forward Pad × 6 dtypes = 6 +
    // PadBackward × 4 dtypes = 4 bindings (all key [T, T]). Twelfth production
    // FKC-contract consumer: IMPORTED from docs/kernel-contracts/cpu/padding.fkc.md
    // via the same `CpuLinkRegistry`. The ONE mode-unified `## pad` forward section
    // declares a BASE entry_point (`pad_cpu`) + the 6 wired dtypes (U8/U32/BF16/
    // F16/F32/F64), so the fan resolves `pad_cpu_<dtype>` (all → the one
    // mode-dispatching `pad_cpu_wrapper`, which selects Constant/Reflect/Replicate
    // at runtime via mode_tag); each per-dtype backward section carries a SPECIFIC
    // single-dtype entry_point (`pad_backward_f32`, …) resolved AS-IS (no fan).
    // in_shape/out_shape/padding/mode_tag/fill_bytes ride in OpParams::{Pad,
    // PadBackward}. The three per-mode forward sections + the pad_walk_cpu helper
    // are `registrable: false` describe-only. ALL hand-written forward + backward
    // `table.register(Pad/PadBackward, …)` regs are DELETED — this is their sole
    // registration path. Placed BEFORE the `fill_unset_cpu_*` passes so the
    // imported entries pick up the CPU cost fill.
    register_cpu_padding_from_contract(table);

    // CPU shape-ops family (migratable subset) — Flip + Roll + Concat +
    // MaskedFill + WriteSlice + WriteSliceRotating (dtype-agnostic byte kernels
    // fanned per production dtype: Flip/Roll/MaskedFill/WriteSlice/
    // WriteSliceRotating 6 dtypes, Concat 9) + the four per-dtype CumSum kernels.
    // Thirteenth production FKC-contract consumer: IMPORTED from
    // docs/kernel-contracts/cpu/shape-ops.fkc.md via the same `CpuLinkRegistry`.
    // Flip/Roll/Concat/MaskedFill/WriteSlice/WriteSliceRotating sections declare a
    // BASE entry_point + a dtypes list trimmed to the wired set, so the fan
    // resolves `<base>_<dtype>` (all → the one dtype-agnostic wrapper); CumSum's
    // per-dtype sections resolve their SPECIFIC `cumsum_<dt>` symbol AS-IS. Keys:
    // Flip/Roll/Concat/CumSum/WriteSlice/WriteSliceRotating [T, T], MaskedFill
    // [T, U8, T]. WriteSlice/WriteSliceRotating model `dest` as the in-place
    // OUTPUT slot (and rotating's `position` as a non-key runtime operand), so the
    // imported key matches build_lookup_dtypes' canonicalized [T_src, T_out]. The
    // migrated hand-written regs are DELETED — this is their sole registration
    // path. DEFERRED (hand-written kept): Triu/Tril (the contract carries only the
    // `triangular` describe-only chassis) + Contiguize (no OpKind — an executor
    // materialize pass). Placed BEFORE the `fill_unset_cpu_*` passes so the
    // imported entries pick up the CPU cost fill.
    register_cpu_shape_ops_from_contract(table);

    // CPU matmul family — FULLY contract-sourced (PORTABLE kernels): bare
    // batched MatMul (6 dtypes F32/F64/BF16/F16/I8/U8, key [T, T, T]) + fused
    // FusedLinear (matmul + bias-add, 4 dtypes F32/F64/BF16/F16, key
    // [T, T, T, T]) = 10 bindings. Fourteenth production FKC-contract consumer:
    // IMPORTED from docs/kernel-contracts/cpu/matmul.fkc.md via the same
    // `CpuLinkRegistry`. Each per-(op,dtype) section carries a SPECIFIC
    // single-dtype entry_point (`matmul_f32`, `fused_linear_f32`, …) resolved
    // AS-IS (no fan-out); the batch geometry (lhs_batch_dims/rhs_batch_dims/
    // m/n/k) rides in OpParams::Matmul (FusedLinear reuses it). Integer MatMul
    // (I8/U8) accumulates in i32 and saturates on store. Every hand-written
    // portable `table.register(MatMul / FusedLinear, …)` call is DELETED — this
    // is the sole registration path for the portable family. Placed BEFORE the
    // `fill_unset_cpu_*` passes so the imported entries pick up the CPU cost
    // fill. ALTERNATIVES-AT-KEY: the MKL/AOCL BLAS siblings register at these
    // SAME keys from SEPARATE external crates with their own `"mkl"`/`"aocl"`
    // kernel_source tags — SEPARATE and untouched (not registered in this build).
    // The SEPARATE FusedKernelRegistry FusedOps::FUSED_LINEAR seam
    // (register_default_fused_kernels) is hand-written and stays untouched; the
    // quant QMatMul/Nf4Matmul OpKinds have their own contracts and stay
    // hand-written below.
    register_cpu_matmul_from_contract(table);

    // The CPU ATTENTION family's KernelBindingTable path — forward FlashAttn (4
    // dtypes) + FlashAttnBackward{Q,K,V} (3 selectors × 4 dtypes) + PagedAttn (4
    // dtypes), each alibi-optional section fanning into BOTH the no-alibi and
    // with-alibi keys = 40 bindings — is now registered FROM
    // docs/kernel-contracts/cpu/attention.fkc.md by
    // register_cpu_attention_from_contract. The contract marks `alibi_slopes`
    // as `optional: true`, so the importer's key-builder fans each (op, dtype)
    // section into both operand-count keys; FKC is the SOLE path for the migrated
    // family. All hand-written FlashAttn + FlashAttnBackward{Q,K,V} + PagedAttn
    // regs (both alibi keys) were DELETED. PagedAttn's fdx.gather pool is import
    // METADATA (§3.9.1: it dispatches via ordinary U32 block_table/context_lens
    // operands + OpParams::PagedAttn), so it registers here too; only the FDX VIEW
    // seam stays [consumer-ahead]. The SEPARATE FusedKernelRegistry
    // FLASH_ATTN*/PAGED_ATTN seam (register_default_fused_kernels) is untouched.
    // Placed BEFORE the `fill_unset_cpu_*` passes so the imported entries pick up
    // the CPU cost fill.
    register_cpu_attention_from_contract(table);

    // The CPU IN-PLACE scalar-param family's KernelBindingTable path — 21 in-place
    // unary ops (each per-op section fanning [F32,F64,BF16,F16] = 84) +
    // InplaceAffine / ClampInplace / PowIInplace (per-dtype single sections, 4 each
    // = 12) = 96 bindings, all keyed [T, T] — is now registered FROM
    // docs/kernel-contracts/cpu/inplace-unary-affine.fkc.md by
    // register_cpu_inplace_from_contract. FKC is the SOLE path for the whole
    // family; all hand-written <Op>Inplace / InplaceAffine / ClampInplace /
    // PowIInplace regs were DELETED. The `unary_inplace` chassis section is
    // describe-only (registrable: false). Placed BEFORE the `fill_unset_cpu_*`
    // passes; the in-place OpKinds keep the `unknown_cost` sentinel (no arm in
    // default_cost_for_op_kind, not on the cost-coverage lint set) — identical to
    // the deleted hand-written regs.
    register_cpu_inplace_from_contract(table);

    // (bf16 + f16 reductions are now registered from the reduce contract above,
    // alongside the f32/f64 variants — accumulate in f32 for stability.)

    // Cast — the COMPLETE directed-pair matrix: every ordered pair of the 11
    // real numeric dtypes {F32,F64,F16,BF16,F8E4M3,U8,I8,U32,I16,I32,I64} with
    // identity excluded = 11 × 10 = 110 bindings (all key `[SRC, DST]`) — is now
    // registered FROM docs/kernel-contracts/cpu/cast.fkc.md by
    // register_cpu_cast_from_contract. Each per-pair section declares a
    // single-dtype `src` input + a `fixed(DST)` output (no dtype fan), so the
    // importer keys `[SRC, DST]` and resolves the SPECIFIC `cast_<src>_to_<dst>`
    // byte-kernel entry_point AS-IS; because the lookup is keyed on the TARGET
    // dtype, all 10 of a target's source pairs bind the SAME per-target
    // `cast_to_<dst>_cpu_wrapper` (10 real entry_points → 1 wrapper). FKC is the
    // SOLE path for the whole family; all hand-written `table.register(Cast, …)`
    // regs were DELETED. Identity pairs are never registered — the optimizer
    // elides identity Cast before dispatch. The MX dummy dtypes
    // (F6E2M3,F6E3M2,F4,F8E8M0) have no Rust scalar type and are excluded. Placed
    // BEFORE the `fill_unset_cpu_*` passes so the imported entries pick up the CPU
    // cost/precision fill.
    register_cpu_cast_from_contract(table);

    // CPU indexing / gather / scatter family — IndexSelect + Gather
    // (dtype-agnostic byte copies fanned per dtype: 9 dtypes each, key
    // [T, U32, T], the fabricated `<base>_<dt>` symbol resolving to the ONE
    // wrapper) + IndexAdd + ScatterAdd (per-dtype typed accumulation: 4 dtypes
    // each, key [T, U32, T, T], `<op>_<dt>` resolved AS-IS). Eighteenth
    // production FKC consumer: IMPORTED from
    // docs/kernel-contracts/cpu/indexing.fkc.md via the same `CpuLinkRegistry`.
    // The `indices` slot is a FIXED U32 operand in every section (no independent
    // index-dtype axis), so there is no multi-axis fan-out deferral — the whole
    // family migrates. The contract's index_select/gather dtype list is trimmed
    // to production's 9 wired dtypes (I8 describable but NOT wired). ALL
    // hand-written `table.register(IndexSelect/Gather/IndexAdd/ScatterAdd, …)`
    // regs are DELETED — this is their sole registration path. Placed BEFORE the
    // `fill_unset_cpu_*` passes so the imported entries pick up the CPU
    // cost/precision fill.
    register_cpu_indexing_from_contract(table);

    // Conv2D / ConvTranspose2D — BOTH the no-bias key [T, T, T] (x, weight, out)
    // AND the with-bias key [T, T, T, T] (x, weight, bias, out) are now registered
    // FROM the conv FKC contract (`register_cpu_conv_from_contract`, above). The
    // contract marks `bias` as `optional: true`, so the importer's key-builder
    // fans each (op, dtype) section into both operand-count keys — FKC is the SOLE
    // path for the whole conv family (16 bindings). All hand-written conv regs were
    // DELETED (the no-bias deferral is closed).

    // ReduceSumTo / ReduceMaxTo (× 4 dtypes, key [T, T]) are now registered
    // FROM the reduce-to FKC contract (`register_cpu_reduce_to_from_contract`,
    // above) — the hand-written regs were DELETED (FKC is the sole path). The
    // ReduceMaxToBackward regs (key [T, T, T]) below were likewise deleted.

    // FusedLinear (3 inputs lhs, rhs, bias → out; key [T, T, T, T]) is now
    // registered FROM the matmul FKC contract
    // (`register_cpu_matmul_from_contract`, above) — the hand-written regs were
    // DELETED (FKC is the sole path for the portable family). The SEPARATE
    // FusedKernelRegistry FusedOps::FUSED_LINEAR seam
    // (`register_default_fused_kernels`) is hand-written and stays untouched.

    // FlashAttn (forward) + FlashAttnBackward{Q,K,V} + PagedAttn — both the
    // no-alibi and with-alibi keys per dtype — are now registered FROM the
    // attention FKC contract (`register_cpu_attention_from_contract`, near the top
    // of this fn). The contract marks `alibi_slopes` as `optional: true`, so the
    // importer's key-builder fans each (op, dtype) section into both operand-count
    // keys — FKC is the SOLE path for the migrated attention family (40 bindings:
    // FlashAttn 8 + FlashAttnBackward{Q,K,V} 24 + PagedAttn 8). All hand-written
    // FlashAttn + FlashAttnBackward{Q,K,V} + PagedAttn regs (both alibi keys) were
    // DELETED. PagedAttn's `fdx.gather: paged_blocks` pool is import METADATA
    // (§3.9.1): block_table/context_lens ride as ordinary U32 operands +
    // OpParams::PagedAttn, so registration does not depend on the FDX gather VIEW
    // (that stays [consumer-ahead]).

    // (Affine F32 is now registered from the affine-clamp-powi contract above.)

    // The in-place scalar-param family — InplaceAffine / ClampInplace /
    // PowIInplace (each key [T, T], scalars in OpParams::{Affine, Clamp, PowI}) —
    // is now registered FROM docs/kernel-contracts/cpu/inplace-unary-affine.fkc.md
    // by register_cpu_inplace_from_contract near the top of this fn (alongside the
    // 21 in-place unary ops). The affine sections' op_kind was corrected from the
    // out-of-place `Affine` to `InplaceAffine`; all hand-written in-place scalar
    // regs were DELETED with the migration.

    // The ENTIRE CPU SSM / Mamba family — FusedSoftmaxCrossEntropy (key
    // [T, I64, F32]), CausalConv1d (key [T, T, T, T]), and the two SCAN ops
    // SelectiveScan / SsdChunkScan (key [T; 6] — 5 inputs + the ONE bundled
    // output slot) — is now registered from docs/kernel-contracts/cpu/ssm.fkc.md
    // by register_cpu_ssm_from_contract near the top of this fn. The scans'
    // `return.bundle` multi-output is now key-buildable by the importer (it
    // appends the bundle's primary-slot dtype to the key tail), so their
    // hand-written `table.register(SelectiveScan/SsdChunkScan, &[dt; 6], ...)`
    // loops that used to live here were DELETED with the migration.

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

    // The 21 in-place unary activations (Relu/Silu/Gelu/Tanh/Sigmoid/Neg/Abs/
    // Sqr/Sqrt/Rsqrt/Recip/Exp/Log/Sin/Cos/Sign/Floor/Ceil/Round/Erf/GeluErf ×
    // F32/F64/BF16/F16 = 84 bindings, each key [T, T]) are now registered FROM
    // docs/kernel-contracts/cpu/inplace-unary-affine.fkc.md by
    // register_cpu_inplace_from_contract near the top of this fn: each per-op
    // section declares a BASE entry_point `<op>_inplace` + enumerates
    // [F32,F64,BF16,F16], so the importer's §3.4 dtype fan resolves
    // `<op>_inplace_<dt>` to the exact wrapper. All hand-written in-place unary
    // regs (the 5-op starter set + the 16-op expansion loop) were DELETED with
    // the migration; the shared `unary_inplace` chassis section is describe-only.

    // (ClampElementwise / PowIElementwise F32 are now registered from the
    // affine-clamp-powi contract above.)

    // Comparison (6 ops × 4 dtypes, key [T, T, U8]) + ternary select `where`
    // (4 dtypes, key [U8, T, T, T]) are now registered FROM the compare-where
    // FKC contract via register_cpu_compare_where_from_contract (near the top of
    // this fn) — the hand-written regs (and the `compare` / `where_dts`
    // dtype-list closures) are DELETED.

    // Rounding / sign / transcendental unary ops (floor/ceil/round/sign/erf/
    // gelu_erf/rsqrt × F32/F64/BF16/F16) are registered from the elementwise-unary
    // FKC contract via register_cpu_unary_from_contract (near the top of this fn).

    // (Flip / Roll × 6 dtypes [T, T] and CumSum × 4 dtypes [T, T] are now
    // registered FROM the shape-ops FKC contract via
    // register_cpu_shape_ops_from_contract near the top of this fn — Flip/Roll are
    // one dtype-agnostic wrapper fanned per dtype; CumSum's per-dtype typed
    // sections resolve AS-IS. The hand-written regs were DELETED — FKC is the sole
    // path.)

    // (WriteSlice + WriteSliceRotating × 6 dtypes, key [T, T], are now registered
    // FROM the shape-ops FKC contract via register_cpu_shape_ops_from_contract
    // near the top of this fn — one dtype-agnostic wrapper each, fanned per dtype.
    // Both model `dest` as the in-place OUTPUT slot (and rotating's `position` as a
    // non-key runtime operand), so the imported key matches build_lookup_dtypes'
    // canonicalized `[T_src, T_out]`. The hand-written regs were DELETED — FKC is
    // the sole path.)

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

    // (LogSoftmaxLastDim × 4 dtypes is now registered FROM the norm FKC contract
    // via register_cpu_norm_from_contract near the top of this fn, alongside
    // Softmax / RmsNorm / LayerNorm forward.)

    // (SoftmaxLastDimBackward / LogSoftmaxLastDimBackward / LayerNormLastDimBackward
    // / RmsNormLastDimBackward × 4 dtypes = 16 bindings, key [T, T, T]) are now
    // registered FROM the norm-backward FKC contract via
    // register_cpu_norm_backward_from_contract near the top of this fn — the BARE
    // backward (two inputs y/x + upstream gradient g → one passthrough output;
    // outer_count/last_dim/eps in OpParams). Their SEPARATE FusedKernelRegistry
    // seam (register_default_fused_kernels) stays untouched.
    // ReduceMaxToBackward (× 4 dtypes, key [T, T, T]) is now registered FROM the
    // reduce-to FKC contract (`register_cpu_reduce_to_from_contract`); the
    // hand-written regs were DELETED (FKC is the sole path).

    // (MaskedFill × 6 dtypes, key [T, U8, T] — x dtype, mask U8, output == x — is
    // now registered FROM the shape-ops FKC contract via
    // register_cpu_shape_ops_from_contract near the top of this fn: `input` is the
    // sole varying operand that drives the fan, `mask` is the fixed U8 slot, and
    // `out: passthrough(input)`. The hand-written regs + the `masked_dtypes`
    // closure were DELETED — FKC is the sole path.)

    // (Pad forward (Constant/Reflect/Replicate) × 6 dtypes — key [T, T], one
    // dtype-agnostic `pad_cpu_wrapper` dispatching all three modes at runtime via
    // `mode_tag` — is now registered FROM the padding FKC contract via
    // register_cpu_padding_from_contract near the top of this fn. The contract's
    // ONE mode-unified `## pad` section fans its 6 wired dtypes (U8/U32/BF16/F16/
    // F32/F64), ALL resolving `pad_cpu_wrapper`; the three per-mode sections +
    // the pad_walk_cpu helper are `registrable: false` describe-only. The
    // hand-written regs were DELETED — FKC is the sole path for the forward half.)

    // (PadBackward × 4 dtypes — key [T, T], typed accumulation — is likewise
    // registered FROM the padding FKC contract via
    // register_cpu_padding_from_contract near the top of this fn; the
    // hand-written regs were DELETED — FKC is the sole path for the backward
    // half.)

    // (Elementwise unary BF16 + F16 — like F32/F64 — are registered from the
    // elementwise-unary FKC contract via register_cpu_unary_from_contract near
    // the top of this fn; the importer fans each per-op section over all four
    // dtypes. The binary bf16/f16 ops likewise come from the elementwise-binary
    // contract.)

    // (Concat — a variadic uniform-dtype op (N inputs, all the same dtype, plus
    // output) — is now registered FROM the shape-ops FKC contract via
    // register_cpu_shape_ops_from_contract near the top of this fn: the importer
    // treats the variadic list as ONE representative input, so the fan builds the
    // canonical `[T, T]` shorthand key per supported dtype (9 dtypes; the lookup
    // site collapses the actual N+1 dtype list to this same shorthand). The
    // hand-written loop was DELETED — FKC is the sole path.)

    // (SoftmaxLastDim / RmsNormLastDim / LayerNormLastDim × 4 dtypes are now
    // registered FROM the norm FKC contract via register_cpu_norm_from_contract
    // near the top of this fn — key [T, T], bare normalization, no affine operand.
    // Their SEPARATE FusedKernelRegistry seam and the norm-backward regs stay
    // hand-written.)

    // (IndexSelect / Gather × 9 dtypes (key [T, U32, T]) are now registered FROM
    // the indexing FKC contract via register_cpu_indexing_from_contract above —
    // one dtype-agnostic wrapper each, fanned per dtype from a BASE entry_point.
    // The hand-written loop was DELETED — FKC is the sole path.)

    // (Rope × 4 dtypes is now registered FROM the rope FKC contract via
    // register_cpu_rope_from_contract near the top of this fn — key
    // [T, T, T, T] (x, cos, sin + passthrough output), geometry in
    // OpParams::Rope. Its SEPARATE FusedKernelRegistry seam
    // (register_default_fused_kernels, FusedOps::ROPE) stays hand-written.)

    // QMatMul: F32 activations, U32 weight blocks, F32 output.
    table.register(QMatMul, &[f32_dt, u32_dt, f32_dt], cpu, qmatmul_f32_cpu_wrapper);

    // (IndexAdd / ScatterAdd × 4 dtypes (key [T, U32, T, T]) are now registered
    // FROM the indexing FKC contract via register_cpu_indexing_from_contract
    // above — each per-dtype `<op>_<dt>` entry_point resolved AS-IS to its OWN
    // typed wrapper. The hand-written regs were DELETED — FKC is the sole path.)

    // ArgMax/ArgMin: input dtype varies, output is U32. The dispatch
    // wrapper still does its internal input-dtype match (preserves
    // current behavior). Register `[input_dt, U32]` once per input
    // dtype the dispatcher handles so the binding table can also
    // route directly when we collapse the wrapper later.
    for dt in [f32_dt, f64_dt, bf16_dt, f16_dt] {
        table.register(ArgMaxDim, &[dt, u32_dt], cpu, argmax_dim_u32_cpu_dispatch);
        table.register(ArgMinDim, &[dt, u32_dt], cpu, argmin_dim_u32_cpu_dispatch);
    }

    // (Affine F64/BF16/F16 are now registered from the affine-clamp-powi
    // contract above.)

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
    // (ClampElementwise / PowIElementwise F64/BF16/F16 and the two-input
    // PowIElementwiseBackward × 4 dtypes are now registered from the
    // affine-clamp-powi contract above. The FusedOps::POWI_BACKWARD
    // fused-registry seam remains separate and untouched.)

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
            // Device-to-device, write-into (CapturedRun build-out, 4b-ζ):
            // overwrite the executor-provided output's bytes in place via
            // `CudaStorageBytes::copy_from_device` — no allocation, so this
            // branch is capture-safe (mirrors the Affine/Concat/Contiguize
            // conversions from 4b-α/γ) AND a strict improvement for ordinary
            // (non-capture) realize: the executor already hands this wrapper
            // a validly-sized pre-allocated output, so writing into it
            // produces byte-identical results with one fewer allocation.
            let dst = cuda_output(&mut out_guard)?;
            dst.copy_from_device(cuda_src)?;
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

// =============================================================================
// Step E Phase C / B1 — per-device in-flight async-op counter (the load signal)
// =============================================================================

/// Process-wide per-[`DeviceLocation`] in-flight async-op counter — the
/// "queue depth / slot utilization" load signal `06-runtime` names. The
/// same idiom as [`TOPOLOGY_GENERATION`]: a lazily-initialized
/// process-wide table the executor mutates and a runtime selector reads.
///
/// **What it counts.** The number of GPU operations the executor has
/// SUBMITTED but not yet observed completed, per device. It is a
/// *fuel-internal* count derived from the A4b async-completion handle
/// lifecycle ([`compiled::execute_compiled`]'s CUDA event +
/// `pipelined`'s eager-submitted Vulkan batches), NOT a driver query
/// (`cuStreamQuery` is a busy/idle bool, not a depth). One `+1` per
/// async submit ([`inflight_inc`]), one `-1` per completion-handle
/// retirement ([`inflight_dec`], fired from the handle's `Drop`).
///
/// **Narrowed by "event only where waited" (A4b open question #4).** A CUDA
/// node the `pipelined` executor's `WaitSet` pre-scan elides never builds a
/// `CudaCompletion` at all (`produce_pending` returns `Ready` directly, no
/// `Event`, no `inc`/`dec`) — see `compiled::execute_compiled_with_wait_hint`.
/// So on `OrderSource::Default`/`Optimized` realizes this counter now tracks
/// only WAITED CUDA work (cross-device `Copy`/`Move` producers, plus whatever
/// handles are still alive when sampled) rather than literally every enqueued
/// op — same-device-only kernels never touch it, elided or not, on the theory
/// that they were never individually observable before this change either
/// (only swept up in the old per-node drain). `OrderSource::Streaming` — the
/// ONLY order source the production load-aware picker
/// (`ranker::DeviceLoadSelector` / `ChainedSelector`'s `load_tier` leg)
/// actually reads THIS counter through — never elides (falls back to
/// event-every-node), so the picker's own live-load signal is unaffected;
/// the narrowing only bites a caller reading `inflight_count` for a
/// *different*, concurrently-running `Default`/`Optimized` realize on the
/// same device, which would now under-report that realize's true same-device
/// backlog. No production caller does this today.
///
/// **Honesty / correctness.** This is a SCHEDULING HINT, never a
/// correctness gate — correctness rests entirely on the A4b waits (which for
/// elided nodes now come from `pipelined::sync_active_cuda_devices`'s
/// realize-end blanket stream sync instead of a per-node `Event`). So
/// the atomics are [`Ordering::Relaxed`] (no memory to synchronize) and
/// [`inflight_dec`] is underflow-saturating: a hint that briefly
/// mis-reads costs at most a sub-optimal route, never a wrong result.
///
/// **CPU.** Keyed uniformly by `DeviceLocation`, but the CPU dispatch
/// path is synchronous and never builds an async completion handle, so
/// `inflight_count(DeviceLocation::Cpu)` is always 0 — no special-case
/// needed. A `BackendStreams` impl reports CPU as `None` (no queue
/// concept) regardless (see `fuel-core::pipelined_bridge`).
static DEVICE_INFLIGHT: OnceLock<RwLock<HashMap<DeviceLocation, AtomicU32>>> = OnceLock::new();

fn device_inflight_table() -> &'static RwLock<HashMap<DeviceLocation, AtomicU32>> {
    DEVICE_INFLIGHT.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Increment the in-flight async-op count for `loc` — called when the
/// executor SUBMITS an async op (the same site that builds an A4b
/// completion handle). Cheap: a single relaxed `fetch_add` once the
/// per-device slot exists (the slot is created once, under a brief write
/// lock, on the first submit to a device). A poisoned lock is swallowed
/// (the counter is a hint; a poisoned table just stops tracking — never
/// a panic on a production path).
pub fn inflight_inc(loc: DeviceLocation) {
    let table = device_inflight_table();
    // Fast path: slot already exists — read lock + relaxed add, no map mutation.
    if let Ok(guard) = table.read() {
        if let Some(slot) = guard.get(&loc) {
            slot.fetch_add(1, Ordering::Relaxed);
            return;
        }
    }
    // Slow path: create the slot. `entry().or_insert` is idempotent under the
    // write lock, so a racing inc that took the same slow path is harmless —
    // whichever lands second sees the slot and adds to it.
    if let Ok(mut guard) = table.write() {
        guard
            .entry(loc)
            .or_insert_with(|| AtomicU32::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }
}

/// Decrement the in-flight async-op count for `loc` — called when an A4b
/// completion handle RETIRES (its `Drop`). Underflow-saturating: if the
/// slot is missing or already 0 this is a no-op, so an unexpected extra
/// dec can never wrap the count to `u32::MAX` and poison future
/// scheduling. (Under the balance invariant — every `inc` is matched by
/// exactly one `dec` via the handle's `Drop` — this saturation never
/// actually fires; it is pure defense.)
pub fn inflight_dec(loc: DeviceLocation) {
    let table = device_inflight_table();
    if let Ok(guard) = table.read() {
        if let Some(slot) = guard.get(&loc) {
            // saturating: never wrap below 0.
            let _ = slot.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            });
        }
    }
}

/// Read the current in-flight async-op count for `loc`. The signal a
/// load-aware selector consults (Step E Phase C / C2, via
/// `BackendStreams::pending_work_count`). 0 for a device that has never
/// submitted async work (including CPU, which never submits any).
pub fn inflight_count(loc: DeviceLocation) -> u32 {
    let table = device_inflight_table();
    table
        .read()
        .ok()
        .and_then(|guard| guard.get(&loc).map(|slot| slot.load(Ordering::Relaxed)))
        .unwrap_or(0)
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
    let (compute_throughput_flops_per_ns, mem_bandwidth_bytes_per_ns) =
        crate::ranker::default_backend_rates(BackendId::Cpu);
    BackendCapabilities {
        backend_id: BackendId::Cpu,
        device_location: DeviceLocation::Cpu,
        op_dtype_support,
        required_alignment: 64,
        access_granularity_bits: 8,
        transfer_paths: vec![(DeviceLocation::Cpu, TransferPath::SameDevice)],
        storage_substrate: SubstrateClass::HostBytes,
        compute_throughput_flops_per_ns,
        mem_bandwidth_bytes_per_ns,
    }
}

/// Default storage substrate for a backend, matching the storage type
/// each backend actually produces. Mirrors `topology::default_substrate_for`
/// (kept in lockstep; that one is private to `topology.rs`).
fn substrate_for_backend(backend: BackendId) -> SubstrateClass {
    match backend {
        BackendId::Cpu => SubstrateClass::HostBytes,
        BackendId::Cuda => SubstrateClass::CudaUntyped,
        BackendId::Vulkan => SubstrateClass::VulkanBuffer,
        BackendId::Metal => SubstrateClass::MetalBuffer,
        // `BackendId` is `#[non_exhaustive]`; a downstream variant
        // defaults to host bytes until it declares its own substrate.
        _ => SubstrateClass::HostBytes,
    }
}

/// FKC cost unification — Part A.
///
/// Derive a [`BackendCapabilities`] for `backend` (at `device_location`)
/// from the kernel binding `table`. This is the general (any-backend)
/// analogue of [`default_cpu_caps`]: instead of a hand-maintained list,
/// it reads the backend's REGISTERED KERNELS out of the binding table
/// and advertises exactly the `(OpKind, DType)` pairs those kernels
/// cover.
///
/// The `(op, dtype)` derivation mirrors
/// [`fuel_core::topology::SystemTopology`]'s `binding_op_coverage`: each
/// binding key carries per-operand dtypes (inputs then outputs); the
/// classic `(op, dtype)` shape takes the OUTPUT dtype — the last entry
/// in the key's dtype list. Single-dtype keys (most elementwise ops)
/// are unambiguous; multi-dtype keys (e.g. QMatMul `[F32, U32, F32]`)
/// still produce the consumer-relevant output-dtype entry.
///
/// **Why this fixes the placement cost model.** The placement cost
/// composer (`ranker::cost::compute_static_costs`) skips any candidate
/// whose backend has no `BackendCapabilities` in the registry —
/// leaving its `static_cost` at the default ZERO, so an uncapped GPU
/// candidate prices as free and can spuriously out-rank a real CPU
/// cost. Registering the caps this function derives makes
/// `capabilities_for(gpu)` return `Some`, so the GPU candidate's
/// FLOP/byte cost fn actually runs (see also
/// [`KernelBindingTable::fill_unset_cost_for_backend`], which ensures
/// those kernels carry a non-`unknown_cost` cost fn).
///
/// Alignment / granularity use conservative device defaults (256-byte
/// alignment for GPU buffers — covers CUDA's 256B and Vulkan's typical
/// storage-buffer alignment; byte-addressable). Transfer paths
/// advertise the universal `HostStaging` link back to the CPU so the
/// `TransferMatrix` can price GPU↔CPU crossings; richer paths (P2P,
/// shared memory) are a future refinement the backend can declare
/// explicitly. Never panics.
pub fn derive_backend_caps(
    backend: BackendId,
    device_location: DeviceLocation,
    table: &KernelBindingTable,
) -> BackendCapabilities {
    use std::collections::HashSet;
    let mut op_dtype_support: HashSet<(OpKind, DType)> = HashSet::new();
    for (op, dtypes, this_backend) in table.iter_keys() {
        if this_backend != backend {
            continue;
        }
        if let Some(&output_dt) = dtypes.last() {
            op_dtype_support.insert((op, output_dt));
        }
    }
    // GPU substrate + a universal host-staging path back to CPU so the
    // transfer matrix can price GPU↔CPU. CPU keeps its SameDevice self
    // edge (this function is general, but CPU has its own hand-tuned
    // `default_cpu_caps`; the CPU branch here is just defensive).
    let transfer_paths = if device_location == DeviceLocation::Cpu {
        vec![(DeviceLocation::Cpu, TransferPath::SameDevice)]
    } else {
        vec![(DeviceLocation::Cpu, TransferPath::HostStaging)]
    };
    let (required_alignment, access_granularity_bits) = match backend {
        BackendId::Cpu => (64, 8),
        // GPU buffer allocations: 256-byte alignment is a safe
        // superset (CUDA cudaMalloc is 256B-aligned; Vulkan storage
        // buffers meet this). Byte-addressable.
        _ => (256, 8),
    };
    let (compute_throughput_flops_per_ns, mem_bandwidth_bytes_per_ns) =
        crate::ranker::default_backend_rates(backend);
    BackendCapabilities {
        backend_id: backend,
        device_location,
        op_dtype_support,
        required_alignment,
        access_granularity_bits,
        transfer_paths,
        storage_substrate: substrate_for_backend(backend),
        compute_throughput_flops_per_ns,
        mem_bandwidth_bytes_per_ns,
    }
}

/// FKC cost unification — Part A.
///
/// Register the [`BackendCapabilities`] for every non-CPU backend that
/// has kernels in `table` into the process-wide [`CapabilityRegistry`],
/// deriving each backend's caps from its registered kernels via
/// [`derive_backend_caps`]. Called once, at the same init boundary that
/// populates the global binding table (see [`global_bindings`]), AFTER
/// the kernels are registered — so the derivation sees the full kernel
/// set.
///
/// This is the registration half of Part A: it closes the gap where
/// only the CPU auto-registered caps, leaving CUDA/Vulkan candidates
/// priced at zero in the placement DP. CPU is skipped here because it
/// already auto-registers its hand-tuned [`default_cpu_caps`] via
/// [`global_registry`].
///
/// Idempotency: the registry appends and lookups pick the first match,
/// so this should run exactly once per backend. The single init
/// boundary in [`global_bindings`] guarantees that.
fn register_derived_gpu_caps(table: &KernelBindingTable) {
    use std::collections::HashSet;
    // Which non-CPU backends have at least one kernel registered?
    let gpu_backends: HashSet<BackendId> = table
        .iter_keys()
        .map(|(_, _, backend)| backend)
        .filter(|b| *b != BackendId::Cpu)
        .collect();
    for backend in gpu_backends {
        let device_location = match backend {
            BackendId::Cuda => DeviceLocation::Cuda { gpu_id: 0 },
            BackendId::Vulkan => DeviceLocation::Vulkan { gpu_id: 0 },
            BackendId::Metal => DeviceLocation::Metal { gpu_id: 0 },
            // Non-exhaustive: an unknown backend with kernels still
            // gets caps derived; default its device to CPU (defensive).
            _ => DeviceLocation::Cpu,
        };
        register_backend_capabilities(derive_backend_caps(backend, device_location, table));
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
            // Single init-boundary fail-fast: a duplicate `KernelRef`
            // in the hand-written static tables is a programmer error,
            // surfaced once here after all backends register — the
            // never-panic replacement for the former inline registration
            // panic (the dynamic FKC importer will call `finalize()?`).
            t.finalize()
                .expect("KernelBindingTable: duplicate kernel registration at init");
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
        // FKC cost unification — Part A. The CUDA registrations above
        // use `table.register(...)`, defaulting cost to `unknown_cost`
        // (→ zero cost). Bulk-fill the OpKind-family FLOP/byte cost fn
        // for every CUDA entry still on the sentinel — the same pass
        // Vulkan already runs at the end of `register_vulkan_kernels`,
        // and the same `default_cost_for_op_kind` families the CPU fill
        // uses (op FLOPs/bytes are backend-agnostic; per-backend
        // throughput is a Layer-2 / Part-C refinement). Without this,
        // even a *capped* CUDA candidate would price at zero.
        table.fill_unset_cost_for_backend(
            BackendId::Cuda,
            crate::cost::default_cost_for_op_kind,
        );
    }
    #[cfg(feature = "vulkan")]
    {
        crate::vulkan_dispatch::register_vulkan_kernels(table);
    }

    // FKC cost unification — Part A. Now that every compiled-in
    // backend's kernels are registered (and their cost fns filled),
    // derive + register the `BackendCapabilities` for every non-CPU
    // backend that has kernels, so the placement cost model's
    // `capabilities_for(gpu)` returns `Some` and the GPU candidate's
    // cost fn actually runs (instead of being skipped → priced at
    // zero). CPU already auto-registers its hand-tuned caps via
    // `global_registry`. Touches the separate `GLOBAL_REGISTRY` lock,
    // not this table's — no re-entrancy.
    register_derived_gpu_caps(table);

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
        t.finalize()
            .expect("KernelBindingTable: duplicate kernel registration at init");
        RwLock::new(t)
    });
    {
        let mut guard = lock.write().unwrap();
        register(&mut guard);
        // Re-validate after the extender adds its wrappers (same
        // never-panic fail-fast as the init boundary above).
        guard
            .finalize()
            .expect("KernelBindingTable: duplicate kernel registration after extend");
    }
    bump_topology_generation();
}

/// The authored linear / quant-matmul FUSED kernel bundle, embedded into the
/// binary (the PRODUCTION `include_str!`). `register_cpu_linear_quant_fused_from_contract`
/// parses + lowers it and registers the MIGRATED subset FROM THE CONTRACT.
const FUSED_LINEAR_QUANT_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/fused/linear-quant.fkc.md");

/// Register the linear / quant-matmul FUSED family — `FUSED_LINEAR` (4 dtypes,
/// key `[T, T, T, T]`) + `QMATMUL` (1 impl, key `[F32, U32, F32]`) +
/// `INPLACE_AFFINE` (4 dtypes, key `[T, T]`) + `FUSED_SOFTMAX_CROSS_ENTROPY`
/// (4 dtypes, key `[T, I64, F32]`) = **13 CPU `BackendImpl`s** — into the
/// [`crate::fused::FusedKernelRegistry`] by IMPORTING its `audited: true` FKC
/// contract (`docs/kernel-contracts/fused/linear-quant.fkc.md`), resolved
/// through the production [`crate::fkc::CpuLinkRegistry`] (chaining
/// [`crate::fkc::CPU_FUSED_LINEAR_QUANT_ENTRY_POINTS`]). FKC is unconditional
/// core infrastructure, so this is the ONE registration path for these four
/// fused ops — every hand-written `register_fused!(FUSED_LINEAR / QMATMUL /
/// INPLACE_AFFINE / FUSED_SOFTMAX_CROSS_ENTROPY, …)` call is DELETED.
///
/// The bundle's fifth section `nf4_matmul` is `registrable: false` (its
/// `fdx.quant.family: AFFINE_BLOCK` is consumer-ahead, §6), so it never
/// lowers/registers and NF4's hand-written `FusedOps::NF4_MATMUL` regs (below,
/// unchanged) stay authoritative. QMATMUL's contract keys its weight operand the
/// LOGICAL dispatch dtype `U32` (the physical GGML block byte-honesty rides the
/// `fdx.quant: GGML_BLOCK` block), so the fanned key is `[F32, U32, F32]` —
/// byte-for-byte the deleted `QM_F32` reg. FSCE's `logits` fans over
/// `{F32,F64,BF16,F16}` (targets I64, output always F32), matching the four
/// deleted `FSCE_<dt>` regs exactly.
///
/// The bundle is fused-only, so `register_into`'s required binding-table
/// argument is a local throwaway that provably stays empty. Behavior-preserving
/// vs. the deleted hand-written path: identical per-dtype kernels (bound by
/// pointer through the link registry), the contract's bit-stable `audited: true`
/// precision (same shape as the deleted `*_CPU_PRECISION` consts — no downgrade),
/// and the real `compute_revision` hash (hand-written stamped `UNTRACKED`). Cost
/// stays the Judge-bootstrapped `fused_unknown_cost` sentinel ([consumer-ahead]:
/// the fused cost trampoline is a follow-up slice — the same posture the primitive
/// FKC imports take before `fill_unset_cpu_cost`).
fn register_cpu_linear_quant_fused_from_contract(r: &mut crate::fused::FusedKernelRegistry) {
    let provider =
        crate::fkc::import_bundle_str(FUSED_LINEAR_QUANT_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored linear-quant fused contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.primitives.is_empty(),
        "linear-quant bundle declares only fused ops",
    );
    let mut table = KernelBindingTable::new();
    provider
        .register_into(&mut table, r)
        .expect("linear-quant fused contract must register into the fused registry");
}

/// The authored norm / softmax FUSED kernel bundle, embedded into the binary
/// (the PRODUCTION `include_str!`). `register_cpu_norm_softmax_fused_from_contract`
/// parses + lowers it and registers the FULL bundle FROM THE CONTRACT.
const FUSED_NORM_SOFTMAX_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/fused/norm-softmax.fkc.md");

/// Register the norm / softmax FUSED family — all EIGHT `fused_op` sections
/// (`SOFTMAX_LAST_DIM` / `RMS_NORM_LAST_DIM` / `LAYER_NORM_LAST_DIM`, key `[T, T]`;
/// `SOFTMAX_LAST_DIM_BACKWARD` / `LAYER_NORM_LAST_DIM_BACKWARD` /
/// `RMS_NORM_LAST_DIM_BACKWARD` / `REDUCE_MAX_TO_BACKWARD` / `POWI_BACKWARD`, key
/// `[T, T, T]`), each dtype-fanned over `{F32, F64, BF16, F16}` = **32 CPU
/// `BackendImpl`s** — into the [`crate::fused::FusedKernelRegistry`] by IMPORTING
/// its `audited: true` FKC contract (`docs/kernel-contracts/fused/norm-softmax.fkc.md`),
/// resolved through the production [`crate::fkc::CpuLinkRegistry`] (chaining
/// [`crate::fkc::CPU_FUSED_NORM_ENTRY_POINTS`], the seam-proven 32-row table).
/// FKC is unconditional core infrastructure, so this is the ONE registration
/// path for these eight fused ops — every hand-written `register_fused!(SOFTMAX /
/// RMS_NORM / LAYER_NORM_LAST_DIM (+backward) / REDUCE_MAX_TO_BACKWARD /
/// POWI_BACKWARD, …)` call is DELETED.
///
/// Behavior-preserving vs. the deleted hand-written path: identical per-dtype
/// kernels (bound by pointer through the link registry), the contract's
/// bit-stable `audited: true` precision — the **2026-07-03 maintainer flip
/// (CireSnave)** that relocates the `NORM_FAMILY_CPU_PRECISION` /
/// `REDUCE_MAX_TO_BACKWARD_CPU_PRECISION` / `POWI_BACKWARD_CPU_PRECISION`
/// bit-stable claim onto the contract (same author, same guarantee — no
/// downgrade), and the real `compute_revision` hash (hand-written stamped
/// `UNTRACKED`). Cost stays the Judge-bootstrapped `fused_unknown_cost` sentinel
/// (the fused cost trampoline is a follow-up slice).
fn register_cpu_norm_softmax_fused_from_contract(r: &mut crate::fused::FusedKernelRegistry) {
    let provider =
        crate::fkc::import_bundle_str(FUSED_NORM_SOFTMAX_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored norm-softmax fused contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.primitives.is_empty(),
        "norm-softmax bundle declares only fused ops",
    );
    let mut table = KernelBindingTable::new();
    provider
        .register_into(&mut table, r)
        .expect("norm-softmax fused contract must register into the fused registry");
}

/// The authored conv / RoPE / SSM FUSED kernel bundle, embedded into the binary
/// (the PRODUCTION `include_str!`). `register_cpu_conv_rope_fused_from_contract`
/// parses + lowers it and registers the FULL bundle FROM THE CONTRACT.
const FUSED_CONV_ROPE_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/fused/conv-rope.fkc.md");

/// Register the conv / RoPE / SSM FUSED family — all SIX `fused_op` sections,
/// each dtype-fanned over `{F32, F64, BF16, F16}` = **32 CPU `BackendImpl`s** —
/// into the [`crate::fused::FusedKernelRegistry`] by IMPORTING its `audited: true`
/// FKC contract (`docs/kernel-contracts/fused/conv-rope.fkc.md`), resolved
/// through the production [`crate::fkc::CpuLinkRegistry`] (chaining
/// [`crate::fkc::CPU_FUSED_CONV_ROPE_ENTRY_POINTS`], 24 rows). FKC is
/// unconditional core infrastructure, so this is the ONE registration path for
/// these six fused ops — every hand-written `register_fused!(ROPE / CONV2D /
/// CONV_TRANSPOSE2D / CAUSAL_CONV1D / SELECTIVE_SCAN / SSD_CHUNK_SCAN, …)` call
/// is DELETED.
///
/// The 24 symbol rows fan into **32 impls** (ROPE 4 + CONV2D 8 +
/// CONV_TRANSPOSE2D 8 + CAUSAL_CONV1D 4 + SELECTIVE_SCAN 4 + SSD_CHUNK_SCAN 4):
/// - CONV2D / CONV_TRANSPOSE2D mark `bias` `optional: true`, so the importer's
///   key-builder fans EACH per-dtype section into BOTH the no-bias key `[T, T, T]`
///   and the with-bias key `[T, T, T, T]` — 4 rows → 8 impls each, byte-for-byte
///   the deleted `CV_*_NOB` + `CV_*_BIAS` regs. **CONV_TRANSPOSE2D's contract was
///   WIDENED to declare the optional `bias` operand** (it previously described
///   only the no-bias form): production registered both tuples and the CPU
///   transposed-conv scatter kernel seeds the output with `bias[co]` (or `0`), so
///   the widening is truthful and preserves the 8-impl count the
///   `default_kernel_registry_step6_coverage` gate asserts.
/// - SELECTIVE_SCAN / SSD_CHUNK_SCAN declare a `return.bundle` multi-output
///   (Option C, `[y ; last_state]`), so the key-builder appends the bundle's
///   primary-slot dtype (`passthrough(u)` / `passthrough(x)` → T) to the 5-input
///   tail — each keys `[T; 6]` byte-for-byte the deleted `SS_*` / `SCS_*` regs.
///
/// Behavior-preserving vs. the deleted hand-written path: identical per-dtype
/// kernels (bound by pointer), the contract's bit-stable `audited: true`
/// precision — the **2026-07-03 maintainer flip (CireSnave)** relocating the
/// `ROPE_CPU_PRECISION` / `CONV2D_CPU_PRECISION` / `CONV_TRANSPOSE2D_CPU_PRECISION`
/// / `CAUSAL_CONV1D_CPU_PRECISION` / `SELECTIVE_SCAN_CPU_PRECISION` /
/// `SSD_CHUNK_SCAN_CPU_PRECISION` bit-stable claims onto the contract (same
/// author, same guarantee — no downgrade), and the real `compute_revision` hash
/// (hand-written stamped `UNTRACKED`). Cost stays the Judge-bootstrapped
/// `fused_unknown_cost` sentinel.
fn register_cpu_conv_rope_fused_from_contract(r: &mut crate::fused::FusedKernelRegistry) {
    let provider =
        crate::fkc::import_bundle_str(FUSED_CONV_ROPE_CONTRACT, &crate::fkc::CpuLinkRegistry)
            .expect(
                "authored conv-rope fused contract must import \
                 (embedded via include_str!, resolved through CpuLinkRegistry)",
            );
    debug_assert!(
        provider.primitives.is_empty(),
        "conv-rope bundle declares only fused ops",
    );
    let mut table = KernelBindingTable::new();
    provider
        .register_into(&mut table, r)
        .expect("conv-rope fused contract must register into the fused registry");
}

/// Phase 7.6 step 6 — register the always-built fused-op kernels into
/// the [`crate::fused::FusedKernelRegistry`]. Called by
/// [`crate::fused::default_kernel_registry`]; kept here so the
/// crate-private CPU dispatch wrappers stay co-located with their
/// registration.
///
/// Today's coverage. Three `audited: true` FKC contracts are IMPORTED at the top
/// (contract-sourced, replacing the deleted hand-written `register_fused!`
/// entries); the attention family + deferred NF4 stay hand-written:
/// - **linear-quant** (`register_cpu_linear_quant_fused_from_contract`) — 13
///   impls: `FUSED_LINEAR` (4) + `QMATMUL` (1) + `INPLACE_AFFINE` (4) +
///   `FUSED_SOFTMAX_CROSS_ENTROPY` (4). The 5th section `NF4_MATMUL` is
///   `registrable: false` (AFFINE_BLOCK consumer-ahead) → stays hand-written.
/// - **norm-softmax** (`register_cpu_norm_softmax_fused_from_contract`) — 32
///   impls: `SOFTMAX_LAST_DIM` / `RMS_NORM_LAST_DIM` / `LAYER_NORM_LAST_DIM`
///   (+ their `_BACKWARD`) + `REDUCE_MAX_TO_BACKWARD` + `POWI_BACKWARD`
///   (8 sections × 4 dtypes).
/// - **conv-rope** (`register_cpu_conv_rope_fused_from_contract`) — 32 impls:
///   `ROPE` (4) + `CONV2D` (8, no-bias|with-bias) + `CONV_TRANSPOSE2D` (8,
///   no-bias|with-bias — bias operand WIDENED into the contract) +
///   `CAUSAL_CONV1D` (4) + `SELECTIVE_SCAN` (4, bundle) + `SSD_CHUNK_SCAN`
///   (4, bundle).
/// - hand-written below: `FLASH_ATTN` (8) + `FLASH_ATTN_BACKWARD_{Q,K,V}` (24) +
///   `PAGED_ATTN` (8) + `NF4_MATMUL` (3).
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
    // NOTE: only the attention family (FLASH_ATTN / FLASH_ATTN_BACKWARD_{Q,K,V} /
    // PAGED_ATTN) + the deferred NF4_MATMUL keep hand-written cost fns + precision
    // consts here. The norm-softmax, linear-quant, and conv-rope bundles are
    // IMPORTED from their `audited: true` FKC contracts (below), so their cost fns
    // + `*_CPU_PRECISION` consts are no longer imported into this fn.
    use crate::fused::{
        cost_attn_backward_cpu, cost_attn_cpu,
        cost_nf4_matmul_cpu,
        ATTN_BACKWARD_CPU_PRECISION,
        ATTN_CPU_PRECISION,
        NF4_MATMUL_CPU_PRECISION,
    };
    use crate::register_fused;
    use fuel_graph::registry::FusedOps;

    // Dtype tuples mirror the binding-table shape.
    // (FusedLinear's `(lhs, rhs, bias, out)` tuples are gone — FUSED_LINEAR is
    // now imported from the linear-quant FKC contract. Conv2D/ConvTranspose2D
    // `CV_*`, Softmax/RmsNorm/LayerNorm `UNARY_*`, `ROPE_*`, the norm/reduce/powi
    // backward `BW_*`, and the SSM `CC1D_*`/`SS_*`/`SCS_*` tuples are ALSO gone —
    // CONV2D/CONV_TRANSPOSE2D/ROPE + the norm-softmax bundle + CAUSAL_CONV1D/
    // SELECTIVE_SCAN/SSD_CHUNK_SCAN are now imported from the norm-softmax and
    // conv-rope FKC contracts, below.)

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

    // (QMatMul's `(a:F32, w_q:U32, out:F32)` tuple is gone — QMATMUL is now
    // imported from the linear-quant FKC contract. The `[T, T, T]` backward-helper
    // `BW_*` tuples are gone too — SOFTMAX/LAYER/RMS_NORM_LAST_DIM_BACKWARD +
    // REDUCE_MAX_TO_BACKWARD + POWI_BACKWARD are now imported from the norm-softmax
    // FKC contract.)

    let cpu = BackendId::Cpu;

    // FUSED_LINEAR (4) + QMATMUL (1) + INPLACE_AFFINE (4) +
    // FUSED_SOFTMAX_CROSS_ENTROPY (4) = 13 CPU impls are IMPORTED from the
    // `audited: true` linear-quant FKC contract (resolved through the production
    // CpuLinkRegistry). This REPLACES the deleted hand-written register_fused!
    // entries for those four fused ops. The bundle's fifth section, nf4_matmul,
    // is `registrable: false` (AFFINE_BLOCK consumer-ahead, §6), so NF4's
    // hand-written FusedOps::NF4_MATMUL regs below stay authoritative.
    register_cpu_linear_quant_fused_from_contract(r);

    // SOFTMAX / RMS_NORM / LAYER_NORM_LAST_DIM (+ backward) +
    // REDUCE_MAX_TO_BACKWARD + POWI_BACKWARD = 32 CPU impls are IMPORTED from the
    // `audited: true` norm-softmax FKC contract (8 sections × 4 dtypes), resolved
    // through the production CpuLinkRegistry (CPU_FUSED_NORM_ENTRY_POINTS). This
    // REPLACES the deleted hand-written register_fused! entries for those eight
    // fused ops.
    register_cpu_norm_softmax_fused_from_contract(r);

    // ROPE + CONV2D + CONV_TRANSPOSE2D + CAUSAL_CONV1D + SELECTIVE_SCAN +
    // SSD_CHUNK_SCAN = 32 CPU impls are IMPORTED from the `audited: true`
    // conv-rope FKC contract (6 sections; CONV2D/CONV_TRANSPOSE2D fan the optional
    // bias into no-bias + with-bias keys → 8 impls each; SELECTIVE_SCAN/
    // SSD_CHUNK_SCAN key [T;6] via the return.bundle primary slot), resolved
    // through the production CpuLinkRegistry (CPU_FUSED_CONV_ROPE_ENTRY_POINTS).
    // This REPLACES the deleted hand-written register_fused! entries for those six
    // fused ops.
    register_cpu_conv_rope_fused_from_contract(r);

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

    // (QMATMUL × F32 activations × U32 weights — 1 impl — is now IMPORTED from
    // the linear-quant FKC contract, above. The `[T, T, T]` backward helpers
    // SOFTMAX/LAYER/RMS_NORM_LAST_DIM_BACKWARD + REDUCE_MAX_TO_BACKWARD +
    // POWI_BACKWARD are IMPORTED from the norm-softmax FKC contract, and
    // CAUSAL_CONV1D + SELECTIVE_SCAN + SSD_CHUNK_SCAN (with INPLACE_AFFINE +
    // FUSED_SOFTMAX_CROSS_ENTROPY) are IMPORTED from the conv-rope / linear-quant
    // FKC contracts — all above.)

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

    /// BORN-RED → GREEN: the `audited: true` linear-quant FUSED bundle is
    /// PRODUCTION-migrated into the `FusedKernelRegistry` FROM its FKC contract.
    ///
    /// `register_default_fused_kernels` now imports
    /// `docs/kernel-contracts/fused/linear-quant.fkc.md` through the production
    /// `CpuLinkRegistry` (chaining `CPU_FUSED_LINEAR_QUANT_ENTRY_POINTS`) INSTEAD
    /// of the hand-written `register_fused!(FUSED_LINEAR / QMATMUL / INPLACE_AFFINE
    /// / FUSED_SOFTMAX_CROSS_ENTROPY, …)` entries (deleted). Each migrated
    /// `(FusedOpId, Cpu, dtypes)` resolves to the EXACT per-dtype production
    /// wrapper (pointer identity) AND carries the contract's REAL revision hash.
    ///
    /// **Discriminator = `revision != UNTRACKED`.** Both the deleted hand-written
    /// regs and the imported impls bind the SAME wrapper fn-pointer, so pointer
    /// identity holds in BOTH states — it is not the red discriminator. The
    /// hand-written `register_fused!` path stamps `KernelRevisionHash::UNTRACKED`
    /// (the macro default); only the FKC import path stamps the contract's real
    /// `compute_revision` hash. RED (before the import wired + hand-written still
    /// present): `lookup_by_dtypes` returns the first-registered (hand-written)
    /// impl → `revision == UNTRACKED` → `assert_ne!` fails. GREEN (hand-written
    /// deleted, import wired): the imported impl is the sole source → real
    /// revision → passes.
    ///
    /// Guards: the unmigrated `FLASH_ATTN` still resolves its hand-written impl
    /// (UNTRACKED, unchanged); the DEFERRED `NF4_MATMUL` (contract section
    /// `registrable: false`, `AFFINE_BLOCK` consumer-ahead) also stays
    /// hand-written (UNTRACKED) — proving the migration is scoped to exactly the
    /// four registrable sections.
    #[test]
    fn linear_quant_fused_family_migrated_to_fkc_contract() {
        use crate::fused::{FusedKernelRegistry, KernelRevisionHash};
        use fuel_graph::registry::FusedOps;

        let mut r = FusedKernelRegistry::new();
        register_default_fused_kernels(&mut r);

        // --- FUSED_LINEAR: 4 dtypes, key [T, T, T, T] ---
        for (dt, expected) in [
            (DType::F32,  fused_linear_f32_cpu_wrapper as usize),
            (DType::F64,  fused_linear_f64_cpu_wrapper as usize),
            (DType::BF16, fused_linear_bf16_cpu_wrapper as usize),
            (DType::F16,  fused_linear_f16_cpu_wrapper as usize),
        ] {
            let got = r
                .lookup_by_dtypes(FusedOps::FUSED_LINEAR, BackendId::Cpu, &[dt, dt, dt, dt])
                .unwrap_or_else(|| panic!("FUSED_LINEAR {dt:?} migrated impl present"));
            assert_eq!(
                got.kernel as usize, expected,
                "FUSED_LINEAR {dt:?} binds its exact per-dtype production wrapper",
            );
            assert_ne!(
                got.revision, KernelRevisionHash::UNTRACKED,
                "FKC-imported FUSED_LINEAR {dt:?} carries the contract's real revision \
                 (hand-written register_fused! stamps UNTRACKED)",
            );
        }

        // --- QMATMUL: 1 impl, key [F32, U32, F32] (logical dispatch dtype U32) ---
        let qm = r
            .lookup_by_dtypes(
                FusedOps::QMATMUL,
                BackendId::Cpu,
                &[DType::F32, DType::U32, DType::F32],
            )
            .expect("QMATMUL migrated impl present at [F32, U32, F32]");
        assert_eq!(qm.kernel as usize, qmatmul_f32_cpu_wrapper as usize);
        assert_ne!(qm.revision, KernelRevisionHash::UNTRACKED);

        // --- INPLACE_AFFINE: 4 dtypes, key [T, T] ---
        for (dt, expected) in [
            (DType::F32,  inplace_affine_f32_cpu_wrapper as usize),
            (DType::F64,  inplace_affine_f64_cpu_wrapper as usize),
            (DType::BF16, inplace_affine_bf16_cpu_wrapper as usize),
            (DType::F16,  inplace_affine_f16_cpu_wrapper as usize),
        ] {
            let got = r
                .lookup_by_dtypes(FusedOps::INPLACE_AFFINE, BackendId::Cpu, &[dt, dt])
                .unwrap_or_else(|| panic!("INPLACE_AFFINE {dt:?} migrated impl present"));
            assert_eq!(got.kernel as usize, expected);
            assert_ne!(got.revision, KernelRevisionHash::UNTRACKED);
        }

        // --- FUSED_SOFTMAX_CROSS_ENTROPY: 4 dtypes, key [T, I64, F32] ---
        for (dt, expected) in [
            (DType::F32,  fused_softmax_cross_entropy_f32_cpu_wrapper as usize),
            (DType::F64,  fused_softmax_cross_entropy_f64_cpu_wrapper as usize),
            (DType::BF16, fused_softmax_cross_entropy_bf16_cpu_wrapper as usize),
            (DType::F16,  fused_softmax_cross_entropy_f16_cpu_wrapper as usize),
        ] {
            let got = r
                .lookup_by_dtypes(
                    FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY,
                    BackendId::Cpu,
                    &[dt, DType::I64, DType::F32],
                )
                .unwrap_or_else(|| panic!("FSCE {dt:?} migrated impl present"));
            assert_eq!(got.kernel as usize, expected);
            assert_ne!(got.revision, KernelRevisionHash::UNTRACKED);
        }

        // --- GUARD: unmigrated FLASH_ATTN still resolves its hand-written impl ---
        let fa = r
            .lookup_by_dtypes(
                FusedOps::FLASH_ATTN,
                BackendId::Cpu,
                &[DType::F32, DType::F32, DType::F32, DType::F32],
            )
            .expect("FLASH_ATTN hand-written impl present");
        assert_eq!(fa.kernel as usize, flash_attn_f32_cpu_wrapper as usize);
        assert_eq!(
            fa.revision, KernelRevisionHash::UNTRACKED,
            "unmigrated FLASH_ATTN keeps its hand-written UNTRACKED revision",
        );

        // --- GUARD: DEFERRED NF4_MATMUL stays hand-written (registrable: false) ---
        let nf4 = r
            .lookup_by_dtypes(
                FusedOps::NF4_MATMUL,
                BackendId::Cpu,
                &[DType::F32, DType::U8, DType::F32, DType::F32],
            )
            .expect("NF4_MATMUL hand-written impl present (AFFINE_BLOCK deferred)");
        assert_eq!(nf4.kernel as usize, nf4_matmul_f32_cpu_wrapper as usize);
        assert_eq!(
            nf4.revision, KernelRevisionHash::UNTRACKED,
            "deferred NF4_MATMUL keeps its hand-written UNTRACKED revision",
        );
    }

    /// BORN-RED → GREEN: the `audited: true` norm/softmax FUSED bundle is
    /// PRODUCTION-migrated into the `FusedKernelRegistry` FROM its FKC contract.
    ///
    /// `register_default_fused_kernels` now imports
    /// `docs/kernel-contracts/fused/norm-softmax.fkc.md` through the production
    /// `CpuLinkRegistry` (chaining `CPU_FUSED_NORM_ENTRY_POINTS`) INSTEAD of the
    /// hand-written `register_fused!(SOFTMAX / RMS_NORM / LAYER_NORM_LAST_DIM
    /// (+backward) / REDUCE_MAX_TO_BACKWARD / POWI_BACKWARD, …)` entries (deleted).
    /// Each migrated `(FusedOpId, Cpu, dtypes)` resolves to the EXACT per-dtype
    /// production wrapper (pointer identity) AND carries the contract's REAL
    /// revision hash.
    ///
    /// **Discriminator = `revision != UNTRACKED`.** Both paths bind the SAME
    /// wrapper fn-pointer, so pointer identity holds in BOTH states — it is not
    /// the red discriminator. The hand-written `register_fused!` path stamps
    /// `KernelRevisionHash::UNTRACKED`; only the FKC import path stamps the
    /// contract's real `compute_revision`. RED (before wired + hand-written still
    /// present): `lookup_by_dtypes` returns the hand-written impl → `UNTRACKED` →
    /// `assert_ne!` fails. GREEN (deleted + imported): real revision → passes.
    ///
    /// Also asserts the **2026-07-03 maintainer flip rode through**: the imported
    /// precision is bit-stable (`bit_stable_on_same_hardware == true`), NOT the
    /// `UNAUDITED` an `audited: false` contract would have lowered to — the whole
    /// point of the flip.
    #[test]
    fn norm_softmax_fused_family_migrated_to_fkc_contract() {
        use crate::fused::{FusedKernelRegistry, KernelRevisionHash};
        use fuel_graph::registry::FusedOps;

        let mut r = FusedKernelRegistry::new();
        register_default_fused_kernels(&mut r);

        // FORWARD (key [T, T]) — Softmax / RmsNorm / LayerNorm last-dim.
        let forward = [
            (
                FusedOps::SOFTMAX_LAST_DIM,
                [
                    softmax_last_dim_f32_cpu_wrapper as usize,
                    softmax_last_dim_f64_cpu_wrapper as usize,
                    softmax_last_dim_bf16_cpu_wrapper as usize,
                    softmax_last_dim_f16_cpu_wrapper as usize,
                ],
            ),
            (
                FusedOps::RMS_NORM_LAST_DIM,
                [
                    rms_norm_last_dim_f32_cpu_wrapper as usize,
                    rms_norm_last_dim_f64_cpu_wrapper as usize,
                    rms_norm_last_dim_bf16_cpu_wrapper as usize,
                    rms_norm_last_dim_f16_cpu_wrapper as usize,
                ],
            ),
            (
                FusedOps::LAYER_NORM_LAST_DIM,
                [
                    layer_norm_last_dim_f32_cpu_wrapper as usize,
                    layer_norm_last_dim_f64_cpu_wrapper as usize,
                    layer_norm_last_dim_bf16_cpu_wrapper as usize,
                    layer_norm_last_dim_f16_cpu_wrapper as usize,
                ],
            ),
        ];
        for (id, wrappers) in forward {
            for (dt, expected) in [DType::F32, DType::F64, DType::BF16, DType::F16]
                .iter()
                .zip(wrappers)
            {
                let got = r
                    .lookup_by_dtypes(id, BackendId::Cpu, &[*dt, *dt])
                    .unwrap_or_else(|| panic!("{id:?} {dt:?} migrated impl present"));
                assert_eq!(
                    got.kernel as usize, expected,
                    "{id:?} {dt:?} binds its exact per-dtype production wrapper",
                );
                assert_ne!(
                    got.revision, KernelRevisionHash::UNTRACKED,
                    "FKC-imported {id:?} {dt:?} carries the contract's real revision \
                     (hand-written register_fused! stamps UNTRACKED)",
                );
                assert!(
                    got.precision.bit_stable_on_same_hardware,
                    "the 2026-07-03 audited:true flip rode through as bit-stable for {id:?} {dt:?} \
                     (NOT the UNAUDITED an audited:false contract would have lowered to)",
                );
            }
        }

        // BACKWARD (key [T, T, T]) — Softmax / Layer / Rms backward +
        // ReduceMaxTo / PowI backward (backward-of-primitive helpers).
        let backward = [
            (
                FusedOps::SOFTMAX_LAST_DIM_BACKWARD,
                [
                    softmax_last_dim_backward_f32_cpu_wrapper as usize,
                    softmax_last_dim_backward_f64_cpu_wrapper as usize,
                    softmax_last_dim_backward_bf16_cpu_wrapper as usize,
                    softmax_last_dim_backward_f16_cpu_wrapper as usize,
                ],
            ),
            (
                FusedOps::LAYER_NORM_LAST_DIM_BACKWARD,
                [
                    layer_norm_last_dim_backward_f32_cpu_wrapper as usize,
                    layer_norm_last_dim_backward_f64_cpu_wrapper as usize,
                    layer_norm_last_dim_backward_bf16_cpu_wrapper as usize,
                    layer_norm_last_dim_backward_f16_cpu_wrapper as usize,
                ],
            ),
            (
                FusedOps::RMS_NORM_LAST_DIM_BACKWARD,
                [
                    rms_norm_last_dim_backward_f32_cpu_wrapper as usize,
                    rms_norm_last_dim_backward_f64_cpu_wrapper as usize,
                    rms_norm_last_dim_backward_bf16_cpu_wrapper as usize,
                    rms_norm_last_dim_backward_f16_cpu_wrapper as usize,
                ],
            ),
            (
                FusedOps::REDUCE_MAX_TO_BACKWARD,
                [
                    reduce_max_to_backward_f32_cpu_wrapper as usize,
                    reduce_max_to_backward_f64_cpu_wrapper as usize,
                    reduce_max_to_backward_bf16_cpu_wrapper as usize,
                    reduce_max_to_backward_f16_cpu_wrapper as usize,
                ],
            ),
            (
                FusedOps::POWI_BACKWARD,
                [
                    powi_backward_f32_cpu_wrapper as usize,
                    powi_backward_f64_cpu_wrapper as usize,
                    powi_backward_bf16_cpu_wrapper as usize,
                    powi_backward_f16_cpu_wrapper as usize,
                ],
            ),
        ];
        for (id, wrappers) in backward {
            for (dt, expected) in [DType::F32, DType::F64, DType::BF16, DType::F16]
                .iter()
                .zip(wrappers)
            {
                let got = r
                    .lookup_by_dtypes(id, BackendId::Cpu, &[*dt, *dt, *dt])
                    .unwrap_or_else(|| panic!("{id:?} {dt:?} migrated impl present"));
                assert_eq!(got.kernel as usize, expected);
                assert_ne!(
                    got.revision, KernelRevisionHash::UNTRACKED,
                    "FKC-imported {id:?} {dt:?} carries the contract's real revision",
                );
                assert!(
                    got.precision.bit_stable_on_same_hardware,
                    "the audited:true flip rode through as bit-stable for {id:?} {dt:?}",
                );
            }
        }

        // GUARD: an UNMIGRATED fused op (FLASH_ATTN) still resolves its
        // hand-written impl at UNTRACKED — proves the migration is scoped.
        let fa = r
            .lookup_by_dtypes(
                FusedOps::FLASH_ATTN,
                BackendId::Cpu,
                &[DType::F32, DType::F32, DType::F32, DType::F32],
            )
            .expect("FLASH_ATTN hand-written impl present");
        assert_eq!(
            fa.revision, KernelRevisionHash::UNTRACKED,
            "unmigrated FLASH_ATTN keeps its hand-written UNTRACKED revision",
        );
    }

    /// BORN-RED → GREEN: the `audited: true` conv / RoPE / SSM FUSED bundle is
    /// PRODUCTION-migrated into the `FusedKernelRegistry` FROM its FKC contract.
    ///
    /// `register_default_fused_kernels` now imports
    /// `docs/kernel-contracts/fused/conv-rope.fkc.md` through the production
    /// `CpuLinkRegistry` (chaining `CPU_FUSED_CONV_ROPE_ENTRY_POINTS`) INSTEAD of
    /// the hand-written `register_fused!(ROPE / CONV2D / CONV_TRANSPOSE2D /
    /// CAUSAL_CONV1D / SELECTIVE_SCAN / SSD_CHUNK_SCAN, …)` entries (deleted). Each
    /// migrated impl binds the EXACT per-dtype wrapper (pointer identity) AND
    /// carries the contract's REAL revision (hand-written stamps UNTRACKED — the
    /// red discriminator). Covers the two multi-key cases:
    /// - CONV2D / CONV_TRANSPOSE2D fan the optional bias into BOTH the no-bias
    ///   key `[T, T, T]` and the with-bias key `[T, T, T, T]` (8 impls each);
    /// - SELECTIVE_SCAN / SSD_CHUNK_SCAN key `[T; 6]` (5 inputs + the bundle's
    ///   primary output slot).
    ///
    /// Also asserts the **2026-07-03 maintainer flip rode through** as bit-stable
    /// (NOT `UNAUDITED`).
    #[test]
    fn conv_rope_fused_family_migrated_to_fkc_contract() {
        use crate::fused::{FusedKernelRegistry, KernelRevisionHash};
        use fuel_graph::registry::FusedOps;

        let mut r = FusedKernelRegistry::new();
        register_default_fused_kernels(&mut r);

        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        // ROPE — key [T, T, T, T] (x, cos, sin + out).
        let rope_w = [
            rope_f32_cpu_wrapper as usize,
            rope_f64_cpu_wrapper as usize,
            rope_bf16_cpu_wrapper as usize,
            rope_f16_cpu_wrapper as usize,
        ];
        for (dt, expected) in dts.iter().zip(rope_w) {
            let got = r
                .lookup_by_dtypes(FusedOps::ROPE, BackendId::Cpu, &[*dt, *dt, *dt, *dt])
                .unwrap_or_else(|| panic!("ROPE {dt:?} migrated impl present"));
            assert_eq!(got.kernel as usize, expected);
            assert_ne!(got.revision, KernelRevisionHash::UNTRACKED, "ROPE {dt:?} real revision");
            assert!(
                got.precision.bit_stable_on_same_hardware,
                "audited:true rode through bit-stable for ROPE {dt:?}",
            );
        }

        // CONV2D / CONV_TRANSPOSE2D — optional bias ⇒ BOTH no-bias [T,T,T] and
        // with-bias [T,T,T,T] resolve the SAME per-dtype wrapper (8 impls each).
        let conv = [
            (
                FusedOps::CONV2D,
                [
                    conv2d_f32_cpu_wrapper as usize,
                    conv2d_f64_cpu_wrapper as usize,
                    conv2d_bf16_cpu_wrapper as usize,
                    conv2d_f16_cpu_wrapper as usize,
                ],
            ),
            (
                FusedOps::CONV_TRANSPOSE2D,
                [
                    conv_transpose2d_f32_cpu_wrapper as usize,
                    conv_transpose2d_f64_cpu_wrapper as usize,
                    conv_transpose2d_bf16_cpu_wrapper as usize,
                    conv_transpose2d_f16_cpu_wrapper as usize,
                ],
            ),
        ];
        for (id, wrappers) in conv {
            for (dt, expected) in dts.iter().zip(wrappers) {
                for key in [vec![*dt, *dt, *dt], vec![*dt, *dt, *dt, *dt]] {
                    let with_bias = key.len() == 4;
                    let got = r
                        .lookup_by_dtypes(id, BackendId::Cpu, &key)
                        .unwrap_or_else(|| {
                            panic!("{id:?} {dt:?} (with_bias={with_bias}) migrated impl present")
                        });
                    assert_eq!(got.kernel as usize, expected);
                    assert_ne!(
                        got.revision, KernelRevisionHash::UNTRACKED,
                        "{id:?} {dt:?} (with_bias={with_bias}) real revision",
                    );
                    assert!(
                        got.precision.bit_stable_on_same_hardware,
                        "audited:true rode through bit-stable for {id:?} {dt:?} (with_bias={with_bias})",
                    );
                }
            }
        }

        // CAUSAL_CONV1D — key [T, T, T, T] (x, weight, bias + out).
        let cc1d_w = [
            causal_conv1d_f32_cpu_wrapper as usize,
            causal_conv1d_f64_cpu_wrapper as usize,
            causal_conv1d_bf16_cpu_wrapper as usize,
            causal_conv1d_f16_cpu_wrapper as usize,
        ];
        for (dt, expected) in dts.iter().zip(cc1d_w) {
            let got = r
                .lookup_by_dtypes(
                    FusedOps::CAUSAL_CONV1D,
                    BackendId::Cpu,
                    &[*dt, *dt, *dt, *dt],
                )
                .unwrap_or_else(|| panic!("CAUSAL_CONV1D {dt:?} migrated impl present"));
            assert_eq!(got.kernel as usize, expected);
            assert_ne!(got.revision, KernelRevisionHash::UNTRACKED);
            assert!(got.precision.bit_stable_on_same_hardware);
        }

        // SELECTIVE_SCAN / SSD_CHUNK_SCAN — return.bundle ⇒ key [T; 6]
        // (5 inputs + the bundled primary output slot).
        let scan = [
            (
                FusedOps::SELECTIVE_SCAN,
                [
                    selective_scan_f32_cpu_wrapper as usize,
                    selective_scan_f64_cpu_wrapper as usize,
                    selective_scan_bf16_cpu_wrapper as usize,
                    selective_scan_f16_cpu_wrapper as usize,
                ],
            ),
            (
                FusedOps::SSD_CHUNK_SCAN,
                [
                    ssd_chunk_scan_f32_cpu_wrapper as usize,
                    ssd_chunk_scan_f64_cpu_wrapper as usize,
                    ssd_chunk_scan_bf16_cpu_wrapper as usize,
                    ssd_chunk_scan_f16_cpu_wrapper as usize,
                ],
            ),
        ];
        for (id, wrappers) in scan {
            for (dt, expected) in dts.iter().zip(wrappers) {
                let got = r
                    .lookup_by_dtypes(id, BackendId::Cpu, &[*dt; 6])
                    .unwrap_or_else(|| panic!("{id:?} {dt:?} migrated impl present"));
                assert_eq!(got.kernel as usize, expected);
                assert_ne!(
                    got.revision, KernelRevisionHash::UNTRACKED,
                    "{id:?} {dt:?} real revision",
                );
                assert!(
                    got.precision.bit_stable_on_same_hardware,
                    "audited:true rode through bit-stable for {id:?} {dt:?}",
                );
            }
        }

        // GUARD: the DEFERRED NF4_MATMUL (linear-quant `registrable: false`) stays
        // hand-written at UNTRACKED — proves this migration is scoped to conv-rope.
        let nf4 = r
            .lookup_by_dtypes(
                FusedOps::NF4_MATMUL,
                BackendId::Cpu,
                &[DType::F32, DType::U8, DType::F32, DType::F32],
            )
            .expect("NF4_MATMUL hand-written impl present");
        assert_eq!(
            nf4.revision, KernelRevisionHash::UNTRACKED,
            "deferred NF4_MATMUL keeps its hand-written UNTRACKED revision",
        );
    }

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
            compute_throughput_flops_per_ns: 1.0,
            mem_bandwidth_bytes_per_ns: 4.0,
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
            compute_throughput_flops_per_ns: 30.0,
            mem_bandwidth_bytes_per_ns: 40.0,
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

    /// FIRST PRODUCTION FKC CONSUMER (born-red gate). `global_bindings()`
    /// registers the CPU elementwise-binary family (8 ops × 4 dtypes = 32
    /// bindings) FROM ITS KERNEL CONTRACT
    /// (`docs/kernel-contracts/cpu/elementwise-binary.fkc.md`) — the sole
    /// registration path, now that the hand-written `table.register(...)`
    /// calls for this family are DELETED.
    ///
    /// For each of the 32 `(op, [dt; 3], Cpu)` keys this asserts:
    ///  - the binding resolves to the EXACT production wrapper fn-pointer
    ///    (behavior-preserving execution),
    ///  - `kernel_source == "portable-cpu"` — the contract's provenance tag
    ///    (the deleted hand-written path stamped `""`). THIS is the
    ///    discriminator that makes the test go red without the import wired:
    ///    with the hand-written binary regs removed and the import absent, the
    ///    family is simply missing from `global_bindings()`,
    ///  - caps stay empty (`strided_input == false`) and the contract's
    ///    audited bit-stable precision claim rode through.
    #[test]
    fn global_bindings_registers_binary_family_from_contract() {
        let table = global_bindings();

        // (op, [f32, f64, bf16, f16] production wrappers) for all 8 ops.
        let families: &[(OpKind, [crate::kernel::KernelRef; 4])] = &[
            (OpKind::AddElementwise, [
                add_elementwise_f32_cpu_wrapper, add_elementwise_f64_cpu_wrapper,
                add_elementwise_bf16_cpu_wrapper, add_elementwise_f16_cpu_wrapper]),
            (OpKind::SubElementwise, [
                sub_elementwise_f32_cpu_wrapper, sub_elementwise_f64_cpu_wrapper,
                sub_elementwise_bf16_cpu_wrapper, sub_elementwise_f16_cpu_wrapper]),
            (OpKind::MulElementwise, [
                mul_elementwise_f32_cpu_wrapper, mul_elementwise_f64_cpu_wrapper,
                mul_elementwise_bf16_cpu_wrapper, mul_elementwise_f16_cpu_wrapper]),
            (OpKind::DivElementwise, [
                div_elementwise_f32_cpu_wrapper, div_elementwise_f64_cpu_wrapper,
                div_elementwise_bf16_cpu_wrapper, div_elementwise_f16_cpu_wrapper]),
            (OpKind::MaximumElementwise, [
                maximum_elementwise_f32_cpu_wrapper, maximum_elementwise_f64_cpu_wrapper,
                maximum_elementwise_bf16_cpu_wrapper, maximum_elementwise_f16_cpu_wrapper]),
            (OpKind::MinimumElementwise, [
                minimum_elementwise_f32_cpu_wrapper, minimum_elementwise_f64_cpu_wrapper,
                minimum_elementwise_bf16_cpu_wrapper, minimum_elementwise_f16_cpu_wrapper]),
            (OpKind::PowElementwise, [
                pow_elementwise_f32_cpu_wrapper, pow_elementwise_f64_cpu_wrapper,
                pow_elementwise_bf16_cpu_wrapper, pow_elementwise_f16_cpu_wrapper]),
            (OpKind::RemElementwise, [
                rem_elementwise_f32_cpu_wrapper, rem_elementwise_f64_cpu_wrapper,
                rem_elementwise_bf16_cpu_wrapper, rem_elementwise_f16_cpu_wrapper]),
        ];
        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        let mut checked = 0usize;
        for (op, wrappers) in families {
            for (dt, expected) in dts.iter().zip(wrappers.iter()) {
                let alts = table.lookup_alternatives(*op, &[*dt, *dt, *dt], BackendId::Cpu);
                let entry = alts
                    .iter()
                    .find(|e| e.kernel as usize == *expected as usize)
                    .unwrap_or_else(|| {
                        panic!(
                            "{op:?}/{dt:?}/Cpu: the production wrapper must be bound \
                             FROM the elementwise-binary contract in global_bindings() \
                             — found {} alternative(s) with sources {:?}",
                            alts.len(),
                            alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                        )
                    });
                assert_eq!(
                    entry.kernel_source, "portable-cpu",
                    "{op:?}/{dt:?}: binary family must be contract-sourced \
                     (kernel_source=\"portable-cpu\"); got {:?}",
                    entry.kernel_source,
                );
                assert!(
                    !entry.caps.strided_input,
                    "{op:?}/{dt:?}: caps preserved (contiguous-only, strided_input=false)",
                );
                assert!(
                    entry.precision.bit_stable_on_same_hardware,
                    "{op:?}/{dt:?}: the contract's audited bit-stable claim rode through",
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 32, "all 8 ops × 4 dtypes checked");
    }

    /// FKC born-red gate for the CPU affine / clamp / powi family.
    /// `global_bindings()` registers the out-of-place scalar-param family
    /// (affine/clamp/powi × 4 dtypes + powi_backward × 4 = 16 bindings) FROM
    /// ITS KERNEL CONTRACT (`docs/kernel-contracts/cpu/affine-clamp-powi.fkc.md`)
    /// — the sole registration path, now that the hand-written
    /// `table.register(...)` calls are DELETED. Asserts each (op, dtype, Cpu)
    /// key resolves to the EXACT production wrapper fn-pointer, is
    /// contract-sourced (`kernel_source == "portable-cpu"`), caps stayed
    /// contiguous-only, and the audited bit-stable precision claim rode through.
    #[test]
    fn global_bindings_registers_affine_clamp_powi_family_from_contract() {
        let table = global_bindings();
        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        // Single-input forward ops → key [dt, dt].
        let forward: &[(OpKind, [crate::kernel::KernelRef; 4])] = &[
            (OpKind::Affine, [
                affine_f32_cpu_wrapper, affine_f64_cpu_wrapper,
                affine_bf16_cpu_wrapper, affine_f16_cpu_wrapper]),
            (OpKind::ClampElementwise, [
                clamp_elementwise_f32_cpu_wrapper, clamp_f64_cpu_wrapper,
                clamp_bf16_cpu_wrapper, clamp_f16_cpu_wrapper]),
            (OpKind::PowIElementwise, [
                powi_elementwise_f32_cpu_wrapper, powi_f64_cpu_wrapper,
                powi_bf16_cpu_wrapper, powi_f16_cpu_wrapper]),
        ];

        let mut checked = 0usize;
        for (op, wrappers) in forward {
            for (dt, expected) in dts.iter().zip(wrappers.iter()) {
                let alts = table.lookup_alternatives(*op, &[*dt, *dt], BackendId::Cpu);
                let entry = alts
                    .iter()
                    .find(|e| e.kernel as usize == *expected as usize)
                    .unwrap_or_else(|| {
                        panic!(
                            "{op:?}/{dt:?}/Cpu: the production wrapper must be bound \
                             FROM the affine-clamp-powi contract in global_bindings() \
                             — found {} alternative(s) with sources {:?}",
                            alts.len(),
                            alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                        )
                    });
                assert_eq!(
                    entry.kernel_source, "portable-cpu",
                    "{op:?}/{dt:?}: family must be contract-sourced \
                     (kernel_source=\"portable-cpu\"); got {:?}",
                    entry.kernel_source,
                );
                assert!(
                    !entry.caps.strided_input,
                    "{op:?}/{dt:?}: caps preserved (contiguous-only, strided_input=false)",
                );
                assert!(
                    entry.precision.bit_stable_on_same_hardware,
                    "{op:?}/{dt:?}: the contract's audited bit-stable claim rode through",
                );
                checked += 1;
            }
        }

        // Two-input backward op → key [dt, dt, dt].
        let bw: [crate::kernel::KernelRef; 4] = [
            powi_backward_f32_cpu_wrapper, powi_backward_f64_cpu_wrapper,
            powi_backward_bf16_cpu_wrapper, powi_backward_f16_cpu_wrapper];
        for (dt, expected) in dts.iter().zip(bw.iter()) {
            let alts = table.lookup_alternatives(
                OpKind::PowIElementwiseBackward, &[*dt, *dt, *dt], BackendId::Cpu);
            let entry = alts
                .iter()
                .find(|e| e.kernel as usize == *expected as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "PowIElementwiseBackward/{dt:?}/Cpu: the production wrapper must \
                         be bound FROM the affine-clamp-powi contract in global_bindings() \
                         — found {} alternative(s) with sources {:?}",
                        alts.len(),
                        alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                    )
                });
            assert_eq!(
                entry.kernel_source, "portable-cpu",
                "PowIElementwiseBackward/{dt:?}: must be contract-sourced; got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "PowIElementwiseBackward/{dt:?}: caps preserved (contiguous-only)",
            );
            assert!(
                entry.precision.bit_stable_on_same_hardware,
                "PowIElementwiseBackward/{dt:?}: the audited bit-stable claim rode through",
            );
            checked += 1;
        }

        assert_eq!(checked, 16, "3 forward ops × 4 dtypes + powi_backward × 4");
    }

    /// FKC born-red gate for the CPU elementwise-unary family.
    /// `global_bindings()` registers the elementwise-unary family (22 ops × 4
    /// dtypes = 88 bindings) FROM ITS KERNEL CONTRACT
    /// (`docs/kernel-contracts/cpu/elementwise-unary.fkc.md`) via the §3.4
    /// multi-dtype fan-out (each per-op section declares a BASE `entry_point`
    /// that the importer expands to `<base>_<dtype>`) — the sole registration
    /// path, now that the hand-written `table.register(...)` calls are DELETED.
    /// Asserts each (op, dtype, Cpu) key resolves to the EXACT production
    /// wrapper fn-pointer, is contract-sourced (`kernel_source == "portable-cpu"`),
    /// caps stayed contiguous-only, and the audited bit-stable precision claim
    /// rode through. `gelu_tanh` (`GeluElementwise`, base `gelu`) and `gelu_erf`
    /// (`GeluErfElementwise`, base `gelu_erf`) are kept DISTINCT.
    #[test]
    fn global_bindings_registers_unary_family_from_contract() {
        let table = global_bindings();
        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        // (op, [f32, f64, bf16, f16] production wrappers) for all 22 ops.
        let families: &[(OpKind, [crate::kernel::KernelRef; 4])] = &[
            (OpKind::ReluElementwise, [
                relu_elementwise_f32_cpu_wrapper, relu_elementwise_f64_cpu_wrapper,
                relu_elementwise_bf16_cpu_wrapper, relu_elementwise_f16_cpu_wrapper]),
            (OpKind::NegElementwise, [
                neg_elementwise_f32_cpu_wrapper, neg_elementwise_f64_cpu_wrapper,
                neg_elementwise_bf16_cpu_wrapper, neg_elementwise_f16_cpu_wrapper]),
            (OpKind::SqrElementwise, [
                sqr_elementwise_f32_cpu_wrapper, sqr_elementwise_f64_cpu_wrapper,
                sqr_elementwise_bf16_cpu_wrapper, sqr_elementwise_f16_cpu_wrapper]),
            (OpKind::SqrtElementwise, [
                sqrt_elementwise_f32_cpu_wrapper, sqrt_elementwise_f64_cpu_wrapper,
                sqrt_elementwise_bf16_cpu_wrapper, sqrt_elementwise_f16_cpu_wrapper]),
            (OpKind::RecipElementwise, [
                recip_elementwise_f32_cpu_wrapper, recip_elementwise_f64_cpu_wrapper,
                recip_elementwise_bf16_cpu_wrapper, recip_elementwise_f16_cpu_wrapper]),
            (OpKind::AbsElementwise, [
                abs_elementwise_f32_cpu_wrapper, abs_elementwise_f64_cpu_wrapper,
                abs_elementwise_bf16_cpu_wrapper, abs_elementwise_f16_cpu_wrapper]),
            (OpKind::TanhElementwise, [
                tanh_elementwise_f32_cpu_wrapper, tanh_elementwise_f64_cpu_wrapper,
                tanh_elementwise_bf16_cpu_wrapper, tanh_elementwise_f16_cpu_wrapper]),
            (OpKind::ExpElementwise, [
                exp_elementwise_f32_cpu_wrapper, exp_elementwise_f64_cpu_wrapper,
                exp_elementwise_bf16_cpu_wrapper, exp_elementwise_f16_cpu_wrapper]),
            (OpKind::LogElementwise, [
                log_elementwise_f32_cpu_wrapper, log_elementwise_f64_cpu_wrapper,
                log_elementwise_bf16_cpu_wrapper, log_elementwise_f16_cpu_wrapper]),
            (OpKind::SinElementwise, [
                sin_elementwise_f32_cpu_wrapper, sin_elementwise_f64_cpu_wrapper,
                sin_elementwise_bf16_cpu_wrapper, sin_elementwise_f16_cpu_wrapper]),
            (OpKind::CosElementwise, [
                cos_elementwise_f32_cpu_wrapper, cos_elementwise_f64_cpu_wrapper,
                cos_elementwise_bf16_cpu_wrapper, cos_elementwise_f16_cpu_wrapper]),
            (OpKind::SigmoidElementwise, [
                sigmoid_elementwise_f32_cpu_wrapper, sigmoid_elementwise_f64_cpu_wrapper,
                sigmoid_elementwise_bf16_cpu_wrapper, sigmoid_elementwise_f16_cpu_wrapper]),
            (OpKind::SiluElementwise, [
                silu_elementwise_f32_cpu_wrapper, silu_elementwise_f64_cpu_wrapper,
                silu_elementwise_bf16_cpu_wrapper, silu_elementwise_f16_cpu_wrapper]),
            (OpKind::StepElementwise, [
                step_elementwise_f32_cpu_wrapper, step_elementwise_f64_cpu_wrapper,
                step_elementwise_bf16_cpu_wrapper, step_elementwise_f16_cpu_wrapper]),
            (OpKind::GeluElementwise, [
                gelu_elementwise_f32_cpu_wrapper, gelu_elementwise_f64_cpu_wrapper,
                gelu_elementwise_bf16_cpu_wrapper, gelu_elementwise_f16_cpu_wrapper]),
            (OpKind::FloorElementwise, [
                floor_elementwise_f32_cpu_wrapper, floor_elementwise_f64_cpu_wrapper,
                floor_elementwise_bf16_cpu_wrapper, floor_elementwise_f16_cpu_wrapper]),
            (OpKind::CeilElementwise, [
                ceil_elementwise_f32_cpu_wrapper, ceil_elementwise_f64_cpu_wrapper,
                ceil_elementwise_bf16_cpu_wrapper, ceil_elementwise_f16_cpu_wrapper]),
            (OpKind::RoundElementwise, [
                round_elementwise_f32_cpu_wrapper, round_elementwise_f64_cpu_wrapper,
                round_elementwise_bf16_cpu_wrapper, round_elementwise_f16_cpu_wrapper]),
            (OpKind::SignElementwise, [
                sign_elementwise_f32_cpu_wrapper, sign_elementwise_f64_cpu_wrapper,
                sign_elementwise_bf16_cpu_wrapper, sign_elementwise_f16_cpu_wrapper]),
            (OpKind::ErfElementwise, [
                erf_elementwise_f32_cpu_wrapper, erf_elementwise_f64_cpu_wrapper,
                erf_elementwise_bf16_cpu_wrapper, erf_elementwise_f16_cpu_wrapper]),
            (OpKind::GeluErfElementwise, [
                gelu_erf_elementwise_f32_cpu_wrapper, gelu_erf_elementwise_f64_cpu_wrapper,
                gelu_erf_elementwise_bf16_cpu_wrapper, gelu_erf_elementwise_f16_cpu_wrapper]),
            (OpKind::RsqrtElementwise, [
                rsqrt_elementwise_f32_cpu_wrapper, rsqrt_elementwise_f64_cpu_wrapper,
                rsqrt_elementwise_bf16_cpu_wrapper, rsqrt_elementwise_f16_cpu_wrapper]),
        ];

        let mut checked = 0usize;
        for (op, wrappers) in families {
            for (dt, expected) in dts.iter().zip(wrappers.iter()) {
                let alts = table.lookup_alternatives(*op, &[*dt, *dt], BackendId::Cpu);
                let entry = alts
                    .iter()
                    .find(|e| e.kernel as usize == *expected as usize)
                    .unwrap_or_else(|| {
                        panic!(
                            "{op:?}/{dt:?}/Cpu: the production wrapper must be bound \
                             FROM the elementwise-unary contract in global_bindings() \
                             — found {} alternative(s) with sources {:?}",
                            alts.len(),
                            alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                        )
                    });
                assert_eq!(
                    entry.kernel_source, "portable-cpu",
                    "{op:?}/{dt:?}: unary family must be contract-sourced \
                     (kernel_source=\"portable-cpu\"); got {:?}",
                    entry.kernel_source,
                );
                assert!(
                    !entry.caps.strided_input,
                    "{op:?}/{dt:?}: caps preserved (contiguous-only, strided_input=false)",
                );
                assert!(
                    entry.precision.bit_stable_on_same_hardware,
                    "{op:?}/{dt:?}: the contract's audited bit-stable claim rode through",
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 88, "all 22 ops × 4 dtypes checked");
    }

    /// FKC production consumer (born-red gate). `global_bindings()` registers
    /// the CPU compare + where family FROM ITS KERNEL CONTRACT
    /// (`docs/kernel-contracts/cpu/compare-where.fkc.md`) — the sole path now
    /// that the hand-written `table.register(...)` calls for this family are
    /// DELETED.
    ///
    /// Two key shapes, and getting them right IS the point of the test:
    ///  - COMPARE (6 ops × 4 dtypes = 24): the U8-mask operand-dtype list
    ///    `[T, T, U8]` (`return.out` is always `fixed(U8)`), NOT `[T, T, T]`.
    ///  - WHERE (1 op × 4 dtypes = 4): the ternary-select list `[U8, T, T, T]`
    ///    (cond U8 + a/b/out share T). `where` exercises BOTH the §3.4
    ///    multi-dtype fan-out (one contract section → 4 per-dtype bindings) AND
    ///    the `passthrough(a)` fix (output mirrors `a`=T, not the U8 cond).
    ///
    /// For each key: resolves to the EXACT production wrapper fn-pointer and
    /// `kernel_source == "portable-cpu"` — the contract provenance tag (the
    /// deleted hand-written path stamped `""`; THIS is the discriminator that
    /// goes red until the import is wired) — with caps contiguous and the
    /// audited bit-stable precision riding through.
    #[test]
    fn global_bindings_registers_compare_where_family_from_contract() {
        let table = global_bindings();
        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        // -- COMPARE: 6 ops × 4 dtypes, key [T, T, U8] --
        let compare_families: &[(OpKind, [crate::kernel::KernelRef; 4])] = &[
            (OpKind::EqualElementwise, [
                eq_elementwise_f32_cpu_wrapper, eq_elementwise_f64_cpu_wrapper,
                eq_elementwise_bf16_cpu_wrapper, eq_elementwise_f16_cpu_wrapper]),
            (OpKind::NotEqualElementwise, [
                ne_elementwise_f32_cpu_wrapper, ne_elementwise_f64_cpu_wrapper,
                ne_elementwise_bf16_cpu_wrapper, ne_elementwise_f16_cpu_wrapper]),
            (OpKind::LessElementwise, [
                lt_elementwise_f32_cpu_wrapper, lt_elementwise_f64_cpu_wrapper,
                lt_elementwise_bf16_cpu_wrapper, lt_elementwise_f16_cpu_wrapper]),
            (OpKind::LessEqualElementwise, [
                le_elementwise_f32_cpu_wrapper, le_elementwise_f64_cpu_wrapper,
                le_elementwise_bf16_cpu_wrapper, le_elementwise_f16_cpu_wrapper]),
            (OpKind::GreaterElementwise, [
                gt_elementwise_f32_cpu_wrapper, gt_elementwise_f64_cpu_wrapper,
                gt_elementwise_bf16_cpu_wrapper, gt_elementwise_f16_cpu_wrapper]),
            (OpKind::GreaterEqualElementwise, [
                ge_elementwise_f32_cpu_wrapper, ge_elementwise_f64_cpu_wrapper,
                ge_elementwise_bf16_cpu_wrapper, ge_elementwise_f16_cpu_wrapper]),
        ];

        let mut checked = 0usize;
        for (op, wrappers) in compare_families {
            for (dt, expected) in dts.iter().zip(wrappers.iter()) {
                // U8-mask operand-dtype list: [T, T, U8] — NOT [T, T, T].
                let alts = table.lookup_alternatives(*op, &[*dt, *dt, DType::U8], BackendId::Cpu);
                let entry = alts
                    .iter()
                    .find(|e| e.kernel as usize == *expected as usize)
                    .unwrap_or_else(|| {
                        panic!(
                            "{op:?}/{dt:?}/Cpu: the production wrapper must be bound \
                             FROM the compare-where contract in global_bindings() \
                             — found {} alternative(s) with sources {:?}",
                            alts.len(),
                            alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                        )
                    });
                assert_eq!(
                    entry.kernel_source, "portable-cpu",
                    "{op:?}/{dt:?}: compare family must be contract-sourced \
                     (kernel_source=\"portable-cpu\"); got {:?}",
                    entry.kernel_source,
                );
                assert!(
                    !entry.caps.strided_input,
                    "{op:?}/{dt:?}: caps preserved (contiguous-only, strided_input=false)",
                );
                assert!(
                    entry.precision.bit_stable_on_same_hardware,
                    "{op:?}/{dt:?}: the contract's audited bit-stable claim rode through",
                );
                checked += 1;
            }
        }

        // -- WHERE: 1 op × 4 dtypes, key [U8, T, T, T] (fan-out + passthrough(a)) --
        let where_wrappers: [crate::kernel::KernelRef; 4] = [
            where_f32_cpu_wrapper, where_f64_cpu_wrapper,
            where_bf16_cpu_wrapper, where_f16_cpu_wrapper,
        ];
        for (dt, expected) in dts.iter().zip(where_wrappers.iter()) {
            // Ternary-select list: [U8 cond, T a, T b, T out].
            let alts = table.lookup_alternatives(OpKind::Where, &[DType::U8, *dt, *dt, *dt], BackendId::Cpu);
            let entry = alts
                .iter()
                .find(|e| e.kernel as usize == *expected as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "Where/{dt:?}/Cpu: the production wrapper must be bound FROM \
                         the compare-where contract (where fans to [U8,T,T,T]) in \
                         global_bindings() — found {} alternative(s) with sources {:?}",
                        alts.len(),
                        alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                    )
                });
            assert_eq!(
                entry.kernel_source, "portable-cpu",
                "Where/{dt:?}: where must be contract-sourced \
                 (kernel_source=\"portable-cpu\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "Where/{dt:?}: caps preserved (contiguous-only, strided_input=false)",
            );
            checked += 1;
        }

        assert_eq!(checked, 28, "24 compare (6×4) + 4 where keys checked");
    }

    /// FKC production consumer (born-red gate). `global_bindings()` registers
    /// the CPU per-axis reduce family (Sum/Mean/Max/Min × F32/F64/BF16/F16 = 16)
    /// FROM ITS KERNEL CONTRACT (`docs/kernel-contracts/cpu/reduce.fkc.md`) — the
    /// sole path now that the hand-written `table.register(...)` calls for this
    /// family are DELETED.
    ///
    /// The binding key is the `[T, T]` operand-dtype list (input +
    /// `passthrough(input)` output; the reduce axes + keepdim ride in
    /// `OpParams::Reduce`, NOT the dtype-list). Each per-(op,dtype) section
    /// carries a SPECIFIC single-dtype `entry_point` resolved AS-IS (no fan-out).
    ///
    /// For each key: resolves to the EXACT production wrapper fn-pointer and
    /// `kernel_source == "portable-cpu"` — the contract provenance tag (the
    /// deleted hand-written path stamped `""`; THIS is the discriminator that
    /// goes red until the import is wired) — with caps contiguous and the audited
    /// bit-stable precision riding through.
    #[test]
    fn global_bindings_registers_reduce_family_from_contract() {
        let table = global_bindings();
        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        // Sum/Mean/Max/Min × {f32,f64,bf16,f16}, key [T, T].
        let reduce_families: &[(OpKind, [crate::kernel::KernelRef; 4])] = &[
            (OpKind::SumReduce, [
                sum_reduce_f32_cpu_wrapper, sum_reduce_f64_cpu_wrapper,
                sum_reduce_bf16_cpu_wrapper, sum_reduce_f16_cpu_wrapper]),
            (OpKind::MeanReduce, [
                mean_reduce_f32_cpu_wrapper, mean_reduce_f64_cpu_wrapper,
                mean_reduce_bf16_cpu_wrapper, mean_reduce_f16_cpu_wrapper]),
            (OpKind::MaxReduce, [
                max_reduce_f32_cpu_wrapper, max_reduce_f64_cpu_wrapper,
                max_reduce_bf16_cpu_wrapper, max_reduce_f16_cpu_wrapper]),
            (OpKind::MinReduce, [
                min_reduce_f32_cpu_wrapper, min_reduce_f64_cpu_wrapper,
                min_reduce_bf16_cpu_wrapper, min_reduce_f16_cpu_wrapper]),
        ];

        let mut checked = 0usize;
        for (op, wrappers) in reduce_families {
            for (dt, expected) in dts.iter().zip(wrappers.iter()) {
                // [T, T] operand-dtype list (input + passthrough output).
                let alts = table.lookup_alternatives(*op, &[*dt, *dt], BackendId::Cpu);
                let entry = alts
                    .iter()
                    .find(|e| e.kernel as usize == *expected as usize)
                    .unwrap_or_else(|| {
                        panic!(
                            "{op:?}/{dt:?}/Cpu: the production wrapper must be bound \
                             FROM the reduce contract in global_bindings() \
                             — found {} alternative(s) with sources {:?}",
                            alts.len(),
                            alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                        )
                    });
                assert_eq!(
                    entry.kernel_source, "portable-cpu",
                    "{op:?}/{dt:?}: reduce family must be contract-sourced \
                     (kernel_source=\"portable-cpu\"); got {:?}",
                    entry.kernel_source,
                );
                assert!(
                    !entry.caps.strided_input,
                    "{op:?}/{dt:?}: caps preserved (contiguous-only, strided_input=false)",
                );
                assert!(
                    entry.precision.bit_stable_on_same_hardware,
                    "{op:?}/{dt:?}: the contract's audited bit-stable claim rode through",
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 16, "4 ops × 4 dtypes checked");
    }

    /// Born-red guard for the CPU reduce-to family FKC migration. Mirrors the
    /// reduce-family test across the two key shapes this family carries:
    /// `ReduceSumTo` / `ReduceMaxTo` are single input → `passthrough` output,
    /// key `[T, T]`; `ReduceMaxToBackward` is two inputs (x, upstream) +
    /// `passthrough(x)` output, key `[T, T, T]`. Each × {f32,f64,bf16,f16}.
    ///
    /// For each key: resolves to the EXACT production wrapper fn-pointer with
    /// `kernel_source == "portable-cpu"` — the contract provenance tag (the
    /// deleted hand-written path stamped `""`; THIS is the discriminator that
    /// goes red until the import is wired) — caps contiguous, and the audited
    /// bit-stable precision riding through.
    #[test]
    fn global_bindings_registers_reduce_to_family_from_contract() {
        let table = global_bindings();
        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        // ReduceSumTo / ReduceMaxTo — single input + passthrough output, key [T, T].
        let unary_families: &[(OpKind, [crate::kernel::KernelRef; 4])] = &[
            (OpKind::ReduceSumTo, [
                reduce_sum_to_f32_cpu_wrapper, reduce_sum_to_f64_cpu_wrapper,
                reduce_sum_to_bf16_cpu_wrapper, reduce_sum_to_f16_cpu_wrapper]),
            (OpKind::ReduceMaxTo, [
                reduce_max_to_f32_cpu_wrapper, reduce_max_to_f64_cpu_wrapper,
                reduce_max_to_bf16_cpu_wrapper, reduce_max_to_f16_cpu_wrapper]),
        ];
        // ReduceMaxToBackward — two inputs (x, upstream) + passthrough(x)
        // output, key [T, T, T].
        let backward_family: (OpKind, [crate::kernel::KernelRef; 4]) = (
            OpKind::ReduceMaxToBackward, [
                reduce_max_to_backward_f32_cpu_wrapper, reduce_max_to_backward_f64_cpu_wrapper,
                reduce_max_to_backward_bf16_cpu_wrapper, reduce_max_to_backward_f16_cpu_wrapper]);

        // Assert one (op, key, expected-wrapper) is bound FROM the contract.
        let assert_contract_bound =
            |op: OpKind, key: &[DType], expected: crate::kernel::KernelRef| {
                let alts = table.lookup_alternatives(op, key, BackendId::Cpu);
                let entry = alts
                    .iter()
                    .find(|e| e.kernel as usize == expected as usize)
                    .unwrap_or_else(|| {
                        panic!(
                            "{op:?}/{key:?}/Cpu: the production wrapper must be bound \
                             FROM the reduce-to contract in global_bindings() \
                             — found {} alternative(s) with sources {:?}",
                            alts.len(),
                            alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                        )
                    });
                assert_eq!(
                    entry.kernel_source, "portable-cpu",
                    "{op:?}/{key:?}: reduce-to family must be contract-sourced \
                     (kernel_source=\"portable-cpu\"); got {:?}",
                    entry.kernel_source,
                );
                assert!(
                    !entry.caps.strided_input,
                    "{op:?}/{key:?}: caps preserved (contiguous-only, strided_input=false)",
                );
                assert!(
                    entry.precision.bit_stable_on_same_hardware,
                    "{op:?}/{key:?}: the contract's audited bit-stable claim rode through",
                );
            };

        let mut checked = 0usize;
        for (op, wrappers) in unary_families {
            for (dt, expected) in dts.iter().zip(wrappers.iter()) {
                assert_contract_bound(*op, &[*dt, *dt], *expected);
                checked += 1;
            }
        }
        let (op, wrappers) = backward_family;
        for (dt, expected) in dts.iter().zip(wrappers.iter()) {
            assert_contract_bound(op, &[*dt, *dt, *dt], *expected);
            checked += 1;
        }
        assert_eq!(checked, 12, "3 ops × 4 dtypes checked");
    }

    /// Born-red guard for the CPU last-dim NORM (forward) family FKC migration.
    /// Softmax / LogSoftmax / RmsNorm / LayerNorm are each a SINGLE input →
    /// `passthrough` output, key `[T, T]` — the RMS/LayerNorm kernels carry NO
    /// affine gamma/beta operand (bare normalization), and outer_count/last_dim/
    /// eps ride in `OpParams`, so the key stays `[T, T]` (not a 3- or 4-operand
    /// key). Each × {f32,f64,bf16,f16} = 16 bindings.
    ///
    /// For each key: resolves to the EXACT production wrapper fn-pointer with
    /// `kernel_source == "portable-cpu"` — the contract provenance tag (the
    /// deleted hand-written path stamped `""`; THIS is the discriminator that
    /// goes red until the import is wired) — caps contiguous, and the audited
    /// bit-stable precision riding through.
    #[test]
    fn global_bindings_registers_norm_family_from_contract() {
        let table = global_bindings();
        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        // Softmax / LogSoftmax / RmsNorm / LayerNorm — single input +
        // passthrough output, key [T, T]. Wrapper order matches `dts`.
        let norm_families: &[(OpKind, [crate::kernel::KernelRef; 4])] = &[
            (OpKind::SoftmaxLastDim, [
                softmax_last_dim_f32_cpu_wrapper, softmax_last_dim_f64_cpu_wrapper,
                softmax_last_dim_bf16_cpu_wrapper, softmax_last_dim_f16_cpu_wrapper]),
            (OpKind::LogSoftmaxLastDim, [
                log_softmax_f32_cpu_wrapper, log_softmax_f64_cpu_wrapper,
                log_softmax_bf16_cpu_wrapper, log_softmax_f16_cpu_wrapper]),
            (OpKind::RmsNormLastDim, [
                rms_norm_last_dim_f32_cpu_wrapper, rms_norm_last_dim_f64_cpu_wrapper,
                rms_norm_last_dim_bf16_cpu_wrapper, rms_norm_last_dim_f16_cpu_wrapper]),
            (OpKind::LayerNormLastDim, [
                layer_norm_last_dim_f32_cpu_wrapper, layer_norm_last_dim_f64_cpu_wrapper,
                layer_norm_last_dim_bf16_cpu_wrapper, layer_norm_last_dim_f16_cpu_wrapper]),
        ];

        let mut checked = 0usize;
        for (op, wrappers) in norm_families {
            for (dt, expected) in dts.iter().zip(wrappers.iter()) {
                // [T, T] operand-dtype list (input + passthrough output).
                let alts = table.lookup_alternatives(*op, &[*dt, *dt], BackendId::Cpu);
                let entry = alts
                    .iter()
                    .find(|e| e.kernel as usize == *expected as usize)
                    .unwrap_or_else(|| {
                        panic!(
                            "{op:?}/{dt:?}/Cpu: the production wrapper must be bound \
                             FROM the norm contract in global_bindings() \
                             — found {} alternative(s) with sources {:?}",
                            alts.len(),
                            alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                        )
                    });
                assert_eq!(
                    entry.kernel_source, "portable-cpu",
                    "{op:?}/{dt:?}: norm family must be contract-sourced \
                     (kernel_source=\"portable-cpu\"); got {:?}",
                    entry.kernel_source,
                );
                assert!(
                    !entry.caps.strided_input,
                    "{op:?}/{dt:?}: caps preserved (contiguous-only, strided_input=false)",
                );
                assert!(
                    entry.precision.bit_stable_on_same_hardware,
                    "{op:?}/{dt:?}: the contract's audited bit-stable claim rode through",
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 16, "4 ops × 4 dtypes checked");
    }

    /// Born-red guard for the CPU last-dim NORM-BACKWARD family FKC migration.
    /// Softmax / LogSoftmax / LayerNorm / RmsNorm backward are each TWO inputs
    /// (the forward output-or-input `y`/`x` + the upstream gradient `g`) → ONE
    /// `passthrough` output — the BARE backward (no affine gamma/beta grad
    /// operand and no saved-stats operand; the norm kernels recompute mean/var
    /// or mean-square from `x` + `eps`), so the binding key is `[T, T, T]`
    /// (in, in, out) and outer_count / last_dim / eps ride in `OpParams`, NOT
    /// the dtype-list. Each × {f32,f64,bf16,f16} = 16 bindings.
    ///
    /// For each key: resolves to the EXACT production wrapper fn-pointer with
    /// `kernel_source == "portable-cpu"` — the contract provenance tag (the
    /// deleted hand-written path stamped `""`; THIS is the discriminator that
    /// goes red until the import is wired) — caps contiguous, and the audited
    /// bit-stable precision riding through.
    #[test]
    fn global_bindings_registers_norm_backward_family_from_contract() {
        let table = global_bindings();
        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        // Softmax / LogSoftmax / LayerNorm / RmsNorm backward — two inputs
        // (y/x, g) + passthrough output, key [T, T, T]. Wrapper order matches
        // `dts`. (LogSoftmax's wrapper fn-name `log_softmax_backward_*` differs
        // from its `log_softmax_last_dim_backward_*` contract symbol — the same
        // fn-vs-symbol split as the forward `log_softmax` case.)
        let backward_families: &[(OpKind, [crate::kernel::KernelRef; 4])] = &[
            (OpKind::SoftmaxLastDimBackward, [
                softmax_last_dim_backward_f32_cpu_wrapper, softmax_last_dim_backward_f64_cpu_wrapper,
                softmax_last_dim_backward_bf16_cpu_wrapper, softmax_last_dim_backward_f16_cpu_wrapper]),
            (OpKind::LogSoftmaxLastDimBackward, [
                log_softmax_backward_f32_cpu_wrapper, log_softmax_backward_f64_cpu_wrapper,
                log_softmax_backward_bf16_cpu_wrapper, log_softmax_backward_f16_cpu_wrapper]),
            (OpKind::LayerNormLastDimBackward, [
                layer_norm_last_dim_backward_f32_cpu_wrapper, layer_norm_last_dim_backward_f64_cpu_wrapper,
                layer_norm_last_dim_backward_bf16_cpu_wrapper, layer_norm_last_dim_backward_f16_cpu_wrapper]),
            (OpKind::RmsNormLastDimBackward, [
                rms_norm_last_dim_backward_f32_cpu_wrapper, rms_norm_last_dim_backward_f64_cpu_wrapper,
                rms_norm_last_dim_backward_bf16_cpu_wrapper, rms_norm_last_dim_backward_f16_cpu_wrapper]),
        ];

        let mut checked = 0usize;
        for (op, wrappers) in backward_families {
            for (dt, expected) in dts.iter().zip(wrappers.iter()) {
                // [T, T, T] operand-dtype list (two inputs + passthrough output).
                let alts = table.lookup_alternatives(*op, &[*dt, *dt, *dt], BackendId::Cpu);
                let entry = alts
                    .iter()
                    .find(|e| e.kernel as usize == *expected as usize)
                    .unwrap_or_else(|| {
                        panic!(
                            "{op:?}/{dt:?}/Cpu: the production wrapper must be bound \
                             FROM the norm-backward contract in global_bindings() \
                             — found {} alternative(s) with sources {:?}",
                            alts.len(),
                            alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                        )
                    });
                assert_eq!(
                    entry.kernel_source, "portable-cpu",
                    "{op:?}/{dt:?}: norm-backward family must be contract-sourced \
                     (kernel_source=\"portable-cpu\"); got {:?}",
                    entry.kernel_source,
                );
                assert!(
                    !entry.caps.strided_input,
                    "{op:?}/{dt:?}: caps preserved (contiguous-only, strided_input=false)",
                );
                assert!(
                    entry.precision.bit_stable_on_same_hardware,
                    "{op:?}/{dt:?}: the contract's audited bit-stable claim rode through",
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 16, "4 ops × 4 dtypes checked");
    }

    /// RoPE (rotary position embedding) is ONE primitive op (`OpKind::Rope`)
    /// monomorphized over the four float dtypes {F32, F64, BF16, F16}. Each
    /// dtype is a distinct single-dtype contract section (`## rope_f32`, …) with
    /// a concrete `entry_point` (`fuel_cpu_backend::byte_kernels::rope_f32`), so
    /// none of them fan — the importer resolves each declared symbol AS-IS. RoPE
    /// takes THREE inputs (`x` + the precomputed `cos`/`sin` tables, all one
    /// dtype; the tables broadcast over the outer axis by re-indexing, NOT a
    /// stride-0 view) and writes ONE `passthrough(x)` output, so the binding key
    /// is `[T, T, T, T]` (x, cos, sin, out); outer_count / seq / head_dim ride in
    /// `OpParams::Rope`, NOT the dtype-list — identical to the deleted
    /// `rope_dts(t)` regs.
    ///
    /// For each key: resolves to the EXACT production wrapper fn-pointer with
    /// `kernel_source == "portable-cpu"` — the contract provenance tag (the
    /// deleted hand-written path stamped `""`; THIS is the discriminator that
    /// goes red until the import is wired) — caps contiguous, and the audited
    /// bit-stable precision riding through.
    #[test]
    fn global_bindings_registers_rope_family_from_contract() {
        let table = global_bindings();
        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];
        // Wrapper order matches `dts`.
        let expected_wrappers: [crate::kernel::KernelRef; 4] = [
            rope_f32_cpu_wrapper,
            rope_f64_cpu_wrapper,
            rope_bf16_cpu_wrapper,
            rope_f16_cpu_wrapper,
        ];

        let mut checked = 0usize;
        for (dt, expected) in dts.iter().zip(expected_wrappers.iter()) {
            // [T, T, T, T] operand-dtype list (x, cos, sin + passthrough output).
            let alts =
                table.lookup_alternatives(OpKind::Rope, &[*dt, *dt, *dt, *dt], BackendId::Cpu);
            let entry = alts
                .iter()
                .find(|e| e.kernel as usize == *expected as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "Rope/{dt:?}/Cpu: the production wrapper must be bound \
                         FROM the rope contract in global_bindings() \
                         — found {} alternative(s) with sources {:?}",
                        alts.len(),
                        alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                    )
                });
            assert_eq!(
                entry.kernel_source, "portable-cpu",
                "Rope/{dt:?}: rope family must be contract-sourced \
                 (kernel_source=\"portable-cpu\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "Rope/{dt:?}: caps preserved (contiguous-only, strided_input=false)",
            );
            assert!(
                entry.precision.bit_stable_on_same_hardware,
                "Rope/{dt:?}: the contract's audited bit-stable claim rode through",
            );
            checked += 1;
        }
        assert_eq!(checked, 4, "1 op × 4 dtypes checked");
    }

    /// Gate for the CPU SSM / Mamba family FULLY migrated to FKC-contract
    /// registration. The WHOLE family — FusedSoftmaxCrossEntropy (key
    /// `[T, I64, F32]` — logits T + I64 targets → fixed(F32) output),
    /// CausalConv1d (key `[T, T, T, T]` — x, weight, bias + passthrough(x)
    /// output), and the two SCAN ops SelectiveScan / SsdChunkScan (key `[T; 6]`
    /// — 5 inputs + the ONE bundled output slot), 4 ops × 4 dtypes = 16
    /// bindings — is IMPORTED from `docs/kernel-contracts/cpu/ssm.fkc.md`.
    ///
    /// The scans' `[T; 6]` keys are now contract-sourced too: their sections
    /// return a `return.bundle` multi-output (Option C, one buffer
    /// `[y ; last_state]`), and the importer's key-builder appends the bundle's
    /// PRIMARY-slot dtype (`passthrough(u)` / `passthrough(x)` → T) to the key
    /// tail — byte-for-byte the deleted hand-written `&[dt; 6]` regs.
    ///
    /// For each of the 16 keys: resolves to the EXACT production wrapper
    /// fn-pointer with `kernel_source == "portable-cpu"` (the contract
    /// provenance tag; the deleted hand-written path stamped `""`), caps
    /// contiguous, and the contract's bit-stable precision riding through.
    #[test]
    fn global_bindings_registers_ssm_family_from_contract() {
        let table = global_bindings();
        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        // Assert one (op, key) binds `expected` FROM the contract (portable-cpu,
        // contiguous caps, bit-stable precision). `key` is the full binding key.
        let check = |op: OpKind, key: &[DType], expected: crate::kernel::KernelRef, label: &str| {
            let alts = table.lookup_alternatives(op, key, BackendId::Cpu);
            let entry = alts
                .iter()
                .find(|e| e.kernel as usize == expected as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "{label}/Cpu (key {key:?}): the production wrapper must be bound \
                         FROM the ssm contract in global_bindings() — found {} \
                         alternative(s) with sources {:?}",
                        alts.len(),
                        alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                    )
                });
            assert_eq!(
                entry.kernel_source, "portable-cpu",
                "{label}: family must be contract-sourced (kernel_source=\"portable-cpu\"); \
                 got {:?}",
                entry.kernel_source,
            );
            assert!(!entry.caps.strided_input, "{label}: caps preserved (contiguous-only)");
            assert!(
                entry.precision.bit_stable_on_same_hardware,
                "{label}: contract's bit-stable claim rode through",
            );
        };

        // Per-op wrappers, indexed to match `dts` order.
        let fsce: [crate::kernel::KernelRef; 4] = [
            fused_softmax_cross_entropy_f32_cpu_wrapper,
            fused_softmax_cross_entropy_f64_cpu_wrapper,
            fused_softmax_cross_entropy_bf16_cpu_wrapper,
            fused_softmax_cross_entropy_f16_cpu_wrapper,
        ];
        let conv: [crate::kernel::KernelRef; 4] = [
            causal_conv1d_f32_cpu_wrapper,
            causal_conv1d_f64_cpu_wrapper,
            causal_conv1d_bf16_cpu_wrapper,
            causal_conv1d_f16_cpu_wrapper,
        ];
        let sscan: [crate::kernel::KernelRef; 4] = [
            selective_scan_f32_cpu_wrapper,
            selective_scan_f64_cpu_wrapper,
            selective_scan_bf16_cpu_wrapper,
            selective_scan_f16_cpu_wrapper,
        ];
        let ssd: [crate::kernel::KernelRef; 4] = [
            ssd_chunk_scan_f32_cpu_wrapper,
            ssd_chunk_scan_f64_cpu_wrapper,
            ssd_chunk_scan_bf16_cpu_wrapper,
            ssd_chunk_scan_f16_cpu_wrapper,
        ];

        let mut checked = 0usize;
        for (i, dt) in dts.iter().enumerate() {
            // FSCE: [logits T, targets I64, out F32].
            check(
                OpKind::FusedSoftmaxCrossEntropy,
                &[*dt, DType::I64, DType::F32],
                fsce[i],
                "FusedSoftmaxCrossEntropy",
            );
            // CausalConv1d: [x, weight, bias, out] all T.
            check(OpKind::CausalConv1d, &[*dt; 4], conv[i], "CausalConv1d");
            // SelectiveScan: [u, delta, a, b, c, out] = [T; 6] (5 inputs + the
            // ONE bundled output slot, `passthrough(u)` → T).
            check(OpKind::SelectiveScan, &[*dt; 6], sscan[i], "SelectiveScan");
            // SsdChunkScan: [x, dt, a, b, c, out] = [T; 6] (5 inputs + the ONE
            // bundled output slot, `passthrough(x)` → T).
            check(OpKind::SsdChunkScan, &[*dt; 6], ssd[i], "SsdChunkScan");

            checked += 1;
        }
        assert_eq!(checked, 4, "4 ops × 4 dtypes checked (16 contract-sourced bindings)");
    }

    /// Gate for the CPU conv family FULLY migrated to FKC-contract registration.
    /// The whole family — BOTH the no-bias keys (`[T, T, T]` — x, weight + out)
    /// AND the with-bias keys (`[T, T, T, T]` — x, weight, bias + passthrough(x)
    /// output) of Conv2D and ConvTranspose2D, 2 ops × 4 dtypes × 2 operand-counts
    /// = 16 bindings — is IMPORTED from `docs/kernel-contracts/cpu/conv.fkc.md`.
    ///
    /// The no-bias keys are now contract-sourced too: the conv contract declares
    /// `bias` as `optional: true`, and the importer's key-builder
    /// (`fkc/lower.rs` `assemble_dtype_variants`) now fans an optional LAST input
    /// into BOTH the with- and without-operand keys, both resolving the SAME
    /// wrapper. The former hand-written no-bias `table.register(...)` regs are
    /// DELETED (deferral closed).
    ///
    /// For each of the 16 keys: resolves to the EXACT production wrapper
    /// fn-pointer with `kernel_source == "portable-cpu"` — the contract
    /// provenance tag — caps contiguous, and the contract's bit-stable precision
    /// riding through.
    #[test]
    fn global_bindings_registers_conv_family_from_contract() {
        let table = global_bindings();
        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        // Conv2D wrappers (order matches `dts`); the same wrapper serves both the
        // no-bias [x, weight, out] and with-bias [x, weight, bias, out] keys.
        let conv2d_wrappers: [crate::kernel::KernelRef; 4] = [
            conv2d_f32_cpu_wrapper,
            conv2d_f64_cpu_wrapper,
            conv2d_bf16_cpu_wrapper,
            conv2d_f16_cpu_wrapper,
        ];
        // ConvTranspose2D wrappers (order matches `dts`).
        let convt_wrappers: [crate::kernel::KernelRef; 4] = [
            conv_transpose2d_f32_cpu_wrapper,
            conv_transpose2d_f64_cpu_wrapper,
            conv_transpose2d_bf16_cpu_wrapper,
            conv_transpose2d_f16_cpu_wrapper,
        ];

        let mut checked = 0usize;
        for (i, dt) in dts.iter().enumerate() {
            for (op, expected) in [
                (OpKind::Conv2D, conv2d_wrappers[i]),
                (OpKind::ConvTranspose2D, convt_wrappers[i]),
            ] {
                // BOTH operand-count keys: no-bias [x, weight, out] AND with-bias
                // [x, weight, bias, out] — all T, both bound to the SAME wrapper
                // from the contract's optional-operand fan.
                let no_bias: &[DType] = &[*dt, *dt, *dt];
                let with_bias: &[DType] = &[*dt, *dt, *dt, *dt];
                for (key, label) in [(no_bias, "no-bias"), (with_bias, "with-bias")] {
                    let alts = table.lookup_alternatives(op, key, BackendId::Cpu);
                    let entry = alts
                        .iter()
                        .find(|e| e.kernel as usize == expected as usize)
                        .unwrap_or_else(|| {
                            panic!(
                                "{op:?}/{dt:?}/Cpu ({label}): the production wrapper must be \
                                 bound FROM the conv contract in global_bindings() — found {} \
                                 alternative(s) with sources {:?}",
                                alts.len(),
                                alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                            )
                        });
                    assert_eq!(
                        entry.kernel_source, "portable-cpu",
                        "{op:?}/{dt:?} ({label}): family must be contract-sourced \
                         (kernel_source=\"portable-cpu\"); got {:?}",
                        entry.kernel_source,
                    );
                    assert!(
                        !entry.caps.strided_input,
                        "{op:?}/{dt:?} ({label}): caps preserved (contiguous-only)",
                    );
                    assert!(
                        entry.precision.bit_stable_on_same_hardware,
                        "{op:?}/{dt:?} ({label}): contract's bit-stable claim rode through",
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(
            checked, 16,
            "2 ops × 4 dtypes × 2 operand-counts checked (16 contract-sourced bindings)"
        );
    }

    /// Gate for the CPU **padding** family's BACKWARD half migrated to
    /// FKC-contract registration. The four `PadBackward` per-dtype kernels
    /// (key `[T, T]` — grad_out + `passthrough(grad_out)` grad_in, T ∈
    /// {F32,F64,BF16,F16}) are IMPORTED from
    /// `docs/kernel-contracts/cpu/padding.fkc.md` — the `pad_backward_<dt>`
    /// single-dtype sections, each resolved AS-IS (no dtype fan-out) through the
    /// production `CpuLinkRegistry` (chaining `CPU_PADDING_ENTRY_POINTS`).
    ///
    /// The FORWARD `Pad` half (Constant/Reflect/Replicate) is ALSO migrated:
    /// the three runtime modes collapse to ONE mode-unified `(Pad, [T, T])`
    /// binding per dtype served by the single dtype-agnostic `pad_cpu_wrapper`
    /// (mode chosen at runtime via `mode_tag`). The contract's unified `## pad`
    /// section fans its 6 production dtypes (U8/U32/BF16/F16/F32/F64), ALL
    /// resolving `pad_cpu_wrapper` (the fabricated `pad_cpu_<dt>` symbol → the one
    /// wrapper, like Flip/Roll/Concat); the three per-mode sections
    /// (`pad_const_cpu`/`pad_reflect_cpu`/`pad_replicate_cpu`) and the
    /// `pad_walk_cpu` helper stay `registrable: false` describe-only mode docs.
    /// See `register_cpu_padding_from_contract`.
    ///
    /// For each of the 4 backward + 6 forward keys: resolves to the EXACT
    /// production wrapper with `kernel_source == "portable-cpu"` (the contract
    /// provenance tag), caps contiguous, and the contract's bit-stable precision
    /// riding through.
    #[test]
    fn global_bindings_registers_padding_family_from_contract() {
        let table = global_bindings();
        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];
        // PadBackward wrappers, indexed to match `dts` order.
        let pad_backward: [crate::kernel::KernelRef; 4] = [
            pad_backward_f32_cpu_wrapper,
            pad_backward_f64_cpu_wrapper,
            pad_backward_bf16_cpu_wrapper,
            pad_backward_f16_cpu_wrapper,
        ];

        let mut checked = 0usize;
        for (i, dt) in dts.iter().enumerate() {
            // PadBackward: [grad_out T, grad_in T] = [T, T] (passthrough(grad_out)).
            let key: &[DType] = &[*dt, *dt];
            let expected = pad_backward[i];
            let alts = table.lookup_alternatives(OpKind::PadBackward, key, BackendId::Cpu);
            let entry = alts
                .iter()
                .find(|e| e.kernel as usize == expected as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "PadBackward/{dt:?}/Cpu: the production wrapper must be bound \
                         FROM the padding contract in global_bindings() — found {} \
                         alternative(s) with sources {:?}",
                        alts.len(),
                        alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                    )
                });
            assert_eq!(
                entry.kernel_source, "portable-cpu",
                "PadBackward/{dt:?}: family must be contract-sourced \
                 (kernel_source=\"portable-cpu\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "PadBackward/{dt:?}: caps preserved (contiguous-only)",
            );
            assert!(
                entry.precision.bit_stable_on_same_hardware,
                "PadBackward/{dt:?}: contract's bit-stable claim rode through",
            );
            checked += 1;
        }

        // Forward Pad (Constant/Reflect/Replicate) — ONE mode-unified `(Pad, [T, T])`
        // binding per dtype served by the SINGLE dtype-agnostic `pad_cpu_wrapper`
        // (mode chosen at runtime via `mode_tag`). The contract's unified `## pad`
        // section fans its 6 production dtypes (U8/U32/BF16/F16/F32/F64), ALL
        // resolving `pad_cpu_wrapper` (fabricated `pad_cpu_<dt>` symbol → the one
        // wrapper, like Flip/Roll/Concat). in_shape/out_shape/padding/mode_tag/
        // fill_bytes ride in OpParams::Pad, NOT the dtype-list.
        let pad_forward_dts = [
            DType::U8, DType::U32, DType::BF16, DType::F16, DType::F32, DType::F64,
        ];
        for dt in pad_forward_dts {
            let key: &[DType] = &[dt, dt];
            let alts = table.lookup_alternatives(OpKind::Pad, key, BackendId::Cpu);
            let entry = alts
                .iter()
                .find(|e| e.kernel as usize == pad_cpu_wrapper as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "Pad/{dt:?}/Cpu: the production wrapper must be bound \
                         FROM the padding contract in global_bindings() — found {} \
                         alternative(s) with sources {:?}",
                        alts.len(),
                        alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                    )
                });
            assert_eq!(
                entry.kernel_source, "portable-cpu",
                "Pad/{dt:?}: forward family must be contract-sourced \
                 (kernel_source=\"portable-cpu\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "Pad/{dt:?}: caps preserved (contiguous-only)",
            );
            assert!(
                entry.precision.bit_stable_on_same_hardware,
                "Pad/{dt:?}: contract's bit-stable claim rode through",
            );
            checked += 1;
        }

        assert_eq!(
            checked, 10,
            "4 PadBackward + 6 forward Pad dtypes checked (contract-sourced bindings)"
        );
    }

    /// Gate for the CPU **shape-ops** family's migratable subset moved to
    /// FKC-contract registration. IMPORTED from
    /// `docs/kernel-contracts/cpu/shape-ops.fkc.md` via the production
    /// `CpuLinkRegistry` (chaining `CPU_SHAPE_OPS_ENTRY_POINTS`):
    /// - **Flip** / **Roll** — dtype-agnostic byte reorder, key `[T, T]`, one
    ///   `flip_cpu_wrapper` / `roll_cpu_wrapper` per dtype (fan over the 6
    ///   production dtypes F32/F64/BF16/F16/U32/U8; the contract's dtype list is
    ///   trimmed to match production, NOT the kernel's full byte-agnostic set).
    /// - **CumSum** — per-dtype typed accumulation (f32/f64 native, bf16/f16 with
    ///   an f32 accumulator), key `[T, T]`, its SPECIFIC `cumsum_<dt>` symbol
    ///   resolved AS-IS (no fan), 4 dtypes.
    /// - **MaskedFill** — dtype-agnostic data + U8 mask, key `[T, U8, T]`, one
    ///   `masked_fill_cpu_wrapper` per dtype (fan over the same 6 dtypes).
    /// - **Concat** — variadic uniform-dtype join collapsed to the `[T, T]`
    ///   shorthand key, one `concat_cpu_wrapper` per dtype (fan over the 9
    ///   production dtypes F32/F64/BF16/F16/U32/U8/I16/I32/I64).
    /// - **WriteSlice** / **WriteSliceRotating** — in-place rectangular / ring
    ///   scatter, key `[T, T]`, one dtype-agnostic wrapper each (fan over the same
    ///   6 dtypes). `dest` is the in-place OUTPUT slot (not a key input); for the
    ///   rotating op the U32 `position` is a NON-KEY runtime operand — so the
    ///   contract's source-only + `out: passthrough(source)` section keys
    ///   `[T_source, T_out]` = `[T, T]`, matching `build_lookup_dtypes` exactly.
    ///
    /// DEFERRED (hand-written / describe-only, NOT checked here): `Contiguize`
    /// (no `OpKind` — an executor materialize pass) and `Triu`/`Tril` (the contract
    /// carries only the `triangular` chassis umbrella, `registrable: false`, not
    /// two per-OpKind sections). See `register_cpu_shape_ops_from_contract`.
    ///
    /// For each migrated key: resolves to the EXACT production wrapper with
    /// `kernel_source == "portable-cpu"` (the contract provenance tag), caps
    /// contiguous, and the contract's bit-stable precision riding through.
    #[test]
    fn global_bindings_registers_shape_ops_family_from_contract() {
        let table = global_bindings();
        let mut checked = 0usize;

        // Assert a (op, key) resolves to `expected` with the contract provenance.
        let mut check = |op: OpKind, key: &[DType], expected: crate::kernel::KernelRef, label: &str| {
            let alts = table.lookup_alternatives(op, key, BackendId::Cpu);
            let entry = alts
                .iter()
                .find(|e| e.kernel as usize == expected as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "{op:?}/{label}/Cpu: the production wrapper must be bound \
                         FROM the shape-ops contract in global_bindings() — found {} \
                         alternative(s) with sources {:?}",
                        alts.len(),
                        alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                    )
                });
            assert_eq!(
                entry.kernel_source, "portable-cpu",
                "{op:?}/{label}: family must be contract-sourced \
                 (kernel_source=\"portable-cpu\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "{op:?}/{label}: caps preserved (contiguous-only)",
            );
            assert!(
                entry.precision.bit_stable_on_same_hardware,
                "{op:?}/{label}: contract's bit-stable claim rode through",
            );
            checked += 1;
        };

        // Flip / Roll — key [T, T], one dtype-agnostic wrapper fanned per dtype.
        let byte_dts = [
            DType::F32, DType::F64, DType::BF16, DType::F16, DType::U32, DType::U8,
        ];
        for dt in byte_dts {
            check(OpKind::Flip, &[dt, dt], flip_cpu_wrapper, "flip");
            check(OpKind::Roll, &[dt, dt], roll_cpu_wrapper, "roll");
            // MaskedFill — key [T, U8, T].
            check(
                OpKind::MaskedFill,
                &[dt, DType::U8, dt],
                masked_fill_cpu_wrapper,
                "masked_fill",
            );
        }

        // CumSum — per-dtype typed wrappers, key [T, T], resolved AS-IS.
        let cumsum_dts = [DType::F32, DType::F64, DType::BF16, DType::F16];
        let cumsum_wrappers: [crate::kernel::KernelRef; 4] = [
            cumsum_f32_cpu_wrapper,
            cumsum_f64_cpu_wrapper,
            cumsum_bf16_cpu_wrapper,
            cumsum_f16_cpu_wrapper,
        ];
        for (i, dt) in cumsum_dts.iter().enumerate() {
            check(OpKind::CumSum, &[*dt, *dt], cumsum_wrappers[i], "cumsum");
        }

        // Concat — variadic uniform-dtype collapsed to [T, T], 9 dtypes.
        let concat_dts = [
            DType::F32, DType::F64, DType::BF16, DType::F16,
            DType::U32, DType::U8, DType::I16, DType::I32, DType::I64,
        ];
        for dt in concat_dts {
            check(OpKind::Concat, &[dt, dt], concat_cpu_wrapper, "concat");
        }

        // WriteSlice / WriteSliceRotating — in-place rectangular / ring-buffer
        // scatter, key [T, T] (dest IS the output slot via in-place adoption;
        // WriteSliceRotating's `position` is a NON-KEY runtime U32 operand — see
        // `build_lookup_dtypes`). One dtype-agnostic wrapper each, fanned over the
        // 6 production dtypes (F32/F64/BF16/F16/U32/U8). The offset/modulus/axis/
        // ranges ride in OpParams::{WriteSlice,WriteSliceRotating}.
        let scatter_dts = [
            DType::F32, DType::F64, DType::BF16, DType::F16, DType::U32, DType::U8,
        ];
        for dt in scatter_dts {
            check(
                OpKind::WriteSlice,
                &[dt, dt],
                write_slice_cpu_wrapper,
                "write_slice",
            );
            check(
                OpKind::WriteSliceRotating,
                &[dt, dt],
                write_slice_rotating_cpu_wrapper,
                "write_slice_rotating",
            );
            // WriteSliceDoff — like WriteSliceRotating but the runtime operand is a
            // NON-KEY I64 `offset` (device-resident under CUDA), no wrap.
            check(
                OpKind::WriteSliceDoff,
                &[dt, dt],
                write_slice_doff_cpu_wrapper,
                "write_slice_doff",
            );
        }

        assert_eq!(
            checked,
            6 + 6 + 6 + 4 + 9 + 6 + 6 + 6,
            "flip(6) + roll(6) + masked_fill(6) + cumsum(4) + concat(9) + \
             write_slice(6) + write_slice_rotating(6) + write_slice_doff(6) contract-sourced bindings",
        );
    }

    /// Gate for the CPU **indexing / gather / scatter** family moved to
    /// FKC-contract registration. IMPORTED from
    /// `docs/kernel-contracts/cpu/indexing.fkc.md` via the production
    /// `CpuLinkRegistry` (chaining `CPU_INDEXING_ENTRY_POINTS`):
    /// - **IndexSelect** / **Gather** — dtype-agnostic byte copy, key
    ///   `[T, U32, T]` (`source`, fixed-U32 `indices`, `passthrough(source)`
    ///   output). One `index_select_cpu_wrapper` / `gather_cpu_wrapper` per dtype
    ///   (the contract's ONE section declares a BASE `entry_point`
    ///   (`…::index_select_cpu` / `…::gather_cpu`) fanned over its dtype list, the
    ///   fabricated `<base>_<dt>` symbol resolving to the ONE wrapper — the
    ///   pad/flip umbrella precedent). The contract's dtype list is trimmed to
    ///   production's 9 wired dtypes (F32/F64/BF16/F16/U32/U8/I16/I32/I64); I8 is
    ///   describable (byte-agnostic) but NOT wired in production, so it is dropped
    ///   from the contract, NOT the kernel's full byte-agnostic set.
    /// - **IndexAdd** / **ScatterAdd** — per-dtype typed accumulation (f32/f64
    ///   native, bf16/f16 via an f32 accumulator; out seeded from `base` then
    ///   `+= src`), key `[T, U32, T, T]` (`base`, fixed-U32 `indices`, `src`,
    ///   `passthrough(base)` output). Each `index_add_<dt>` / `scatter_add_<dt>`
    ///   section carries a SPECIFIC single-dtype `entry_point` resolved AS-IS (no
    ///   fan), 4 dtypes each, mapping to its OWN typed wrapper.
    ///
    /// The `indices` operand is a FIXED single-dtype (U32) slot in every section
    /// (like compare's U8 mask / paged's U32 block-table), so there is NO
    /// independent index-dtype axis and no multi-axis `FanoutDtypeMismatch`
    /// deferral — the whole family migrates.
    ///
    /// For each migrated key: resolves to the EXACT production wrapper with
    /// `kernel_source == "portable-cpu"` (the contract provenance tag), caps
    /// contiguous, and the contract's bit-stable precision riding through.
    #[test]
    fn global_bindings_registers_indexing_family_from_contract() {
        let table = global_bindings();
        let mut checked = 0usize;

        // Assert a (op, key) resolves to `expected` with the contract provenance.
        let mut check = |op: OpKind, key: &[DType], expected: crate::kernel::KernelRef, label: &str| {
            let alts = table.lookup_alternatives(op, key, BackendId::Cpu);
            let entry = alts
                .iter()
                .find(|e| e.kernel as usize == expected as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "{op:?}/{label}/Cpu: the production wrapper must be bound \
                         FROM the indexing contract in global_bindings() — found {} \
                         alternative(s) with sources {:?}",
                        alts.len(),
                        alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                    )
                });
            assert_eq!(
                entry.kernel_source, "portable-cpu",
                "{op:?}/{label}: family must be contract-sourced \
                 (kernel_source=\"portable-cpu\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "{op:?}/{label}: caps preserved (contiguous-only)",
            );
            assert!(
                entry.precision.bit_stable_on_same_hardware,
                "{op:?}/{label}: contract's bit-stable claim rode through",
            );
            checked += 1;
        };

        // IndexSelect / Gather — dtype-agnostic byte copy, key [T, U32, T], one
        // wrapper fanned per dtype (9 production dtypes; I8 trimmed).
        let byte_dts = [
            DType::F32, DType::F64, DType::BF16, DType::F16,
            DType::U32, DType::U8, DType::I16, DType::I32, DType::I64,
        ];
        for dt in byte_dts {
            check(
                OpKind::IndexSelect,
                &[dt, DType::U32, dt],
                index_select_cpu_wrapper,
                "index_select",
            );
            check(
                OpKind::Gather,
                &[dt, DType::U32, dt],
                gather_cpu_wrapper,
                "gather",
            );
        }

        // IndexAdd / ScatterAdd — per-dtype typed accumulation, key
        // [T, U32, T, T], each `<op>_<dt>` symbol resolved AS-IS (no fan).
        let acc_dts = [DType::F32, DType::F64, DType::BF16, DType::F16];
        let index_add_wrappers: [crate::kernel::KernelRef; 4] = [
            index_add_f32_cpu_wrapper,
            index_add_f64_cpu_wrapper,
            index_add_bf16_cpu_wrapper,
            index_add_f16_cpu_wrapper,
        ];
        let scatter_add_wrappers: [crate::kernel::KernelRef; 4] = [
            scatter_add_f32_cpu_wrapper,
            scatter_add_f64_cpu_wrapper,
            scatter_add_bf16_cpu_wrapper,
            scatter_add_f16_cpu_wrapper,
        ];
        for (i, dt) in acc_dts.iter().enumerate() {
            check(
                OpKind::IndexAdd,
                &[*dt, DType::U32, *dt, *dt],
                index_add_wrappers[i],
                "index_add",
            );
            check(
                OpKind::ScatterAdd,
                &[*dt, DType::U32, *dt, *dt],
                scatter_add_wrappers[i],
                "scatter_add",
            );
        }

        assert_eq!(
            checked,
            9 + 9 + 4 + 4,
            "index_select(9) + gather(9) + index_add(4) + scatter_add(4) \
             contract-sourced bindings",
        );
    }

    /// Gate for the CPU **matmul** family moved to FKC-contract registration.
    /// IMPORTED from `docs/kernel-contracts/cpu/matmul.fkc.md` via the production
    /// `CpuLinkRegistry` (chaining `CPU_MATMUL_ENTRY_POINTS`):
    /// - **MatMul** — bare batched matmul (`OpKind::MatMul`), key `[T, T, T]`
    ///   (lhs, rhs + `passthrough(lhs)` output). Each per-dtype section
    ///   (`## matmul_f32`, …) carries a SPECIFIC single-dtype `entry_point`
    ///   resolved AS-IS (no fan), 6 dtypes (F32/F64/BF16/F16 with an f32/native
    ///   accumulator + I8/U8 with an i32 accumulator + saturating store).
    /// - **FusedLinear** — fused matmul + bias-add (`OpKind::FusedLinear`), key
    ///   `[T, T, T, T]` (a, b, bias + `passthrough(a)` output). Reuses
    ///   `OpParams::Matmul` for shape; bias is a REQUIRED 1-D `[N]` operand (no
    ///   optional fan). 4 dtypes (F32/F64/BF16/F16).
    ///
    /// This family exercises the CRITICAL alternatives-at-the-same-key path: the
    /// binding table supports MULTIPLE ranked kernels per `(op, dtypes, backend)`
    /// key, and the MKL/AOCL BLAS siblings (external `fuel-mkl-cpu-backend` /
    /// `fuel-aocl-cpu-backend` crates, registered through the exported dispatch
    /// helpers with their own `"mkl"`/`"aocl"` `kernel_source` tags) may register
    /// at these SAME keys — SEPARATE and out of scope here. So this gate does NOT
    /// assert the portable wrapper is the SOLE alternative; it `find`s the
    /// portable `"portable-cpu"`-sourced entry among the alternatives.
    ///
    /// SEPARATE seams left untouched: the `FusedKernelRegistry`
    /// `FusedOps::FUSED_LINEAR` registration (`register_default_fused_kernels`) is
    /// hand-written and not FKC-imported (the matmul contract declares NO
    /// `fused_op` sections); the quant `QMatMul` / `Nf4Matmul` OpKinds have their
    /// own contracts and stay hand-written.
    ///
    /// For each migrated key: the portable wrapper is bound with
    /// `kernel_source == "portable-cpu"` (the contract provenance tag), caps
    /// contiguous, and the contract's bit-stable precision riding through.
    #[test]
    fn global_bindings_registers_matmul_family_from_contract() {
        let table = global_bindings();
        let mut checked = 0usize;

        // Assert a (op, key) resolves to `expected` (the portable wrapper) with
        // the contract provenance — `find`ing it among any BLAS-sibling
        // alternatives at the same key (never asserting single-alternative).
        let mut check = |op: OpKind, key: &[DType], expected: crate::kernel::KernelRef, label: &str| {
            let alts = table.lookup_alternatives(op, key, BackendId::Cpu);
            let entry = alts
                .iter()
                .find(|e| e.kernel as usize == expected as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "{op:?}/{label}/Cpu: the portable wrapper must be bound \
                         FROM the matmul contract in global_bindings() — found {} \
                         alternative(s) with sources {:?}",
                        alts.len(),
                        alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                    )
                });
            assert_eq!(
                entry.kernel_source, "portable-cpu",
                "{op:?}/{label}: portable kernel must be contract-sourced \
                 (kernel_source=\"portable-cpu\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "{op:?}/{label}: caps preserved (contiguous-only)",
            );
            assert!(
                entry.precision.bit_stable_on_same_hardware,
                "{op:?}/{label}: contract's bit-stable claim rode through",
            );
            checked += 1;
        };

        // MatMul — key [T, T, T], per-dtype wrapper resolved AS-IS. 6 dtypes.
        let mm_dts = [
            DType::F32, DType::F64, DType::BF16, DType::F16, DType::I8, DType::U8,
        ];
        let mm_wrappers: [crate::kernel::KernelRef; 6] = [
            matmul_f32_cpu_wrapper,
            matmul_f64_cpu_wrapper,
            matmul_bf16_cpu_wrapper,
            matmul_f16_cpu_wrapper,
            matmul_i8_cpu_wrapper,
            matmul_u8_cpu_wrapper,
        ];
        for (i, dt) in mm_dts.iter().enumerate() {
            check(OpKind::MatMul, &[*dt, *dt, *dt], mm_wrappers[i], "matmul");
        }

        // FusedLinear — key [T, T, T, T], per-dtype wrapper resolved AS-IS. 4 dtypes.
        let fl_dts = [DType::F32, DType::F64, DType::BF16, DType::F16];
        let fl_wrappers: [crate::kernel::KernelRef; 4] = [
            fused_linear_f32_cpu_wrapper,
            fused_linear_f64_cpu_wrapper,
            fused_linear_bf16_cpu_wrapper,
            fused_linear_f16_cpu_wrapper,
        ];
        for (i, dt) in fl_dts.iter().enumerate() {
            check(
                OpKind::FusedLinear,
                &[*dt, *dt, *dt, *dt],
                fl_wrappers[i],
                "fused_linear",
            );
        }

        assert_eq!(
            checked,
            6 + 4,
            "matmul(6) + fused_linear(4) contract-sourced bindings",
        );
    }

    /// Gate for the CPU **attention** family's `KernelBindingTable` path migrated
    /// to FKC-contract registration. FlashAttn (forward) + the three
    /// FlashAttnBackward{Q,K,V} selectors + PagedAttn — 5 ops × 4 dtypes ×
    /// {no-alibi, with-alibi} = 40 bindings — are IMPORTED from
    /// `docs/kernel-contracts/cpu/attention.fkc.md` (the `op_kind:` sections).
    /// Each section declares `alibi_slopes` as an `optional: true` LAST input, so
    /// the importer's optional-operand fan registers BOTH the no-alibi key
    /// (`[q,k,v(,do),out]` / `[q,kc,vc,bt,cl,out]`) and the with-alibi key
    /// (`+alibi`), both resolving the SAME wrapper — byte-for-byte the deleted
    /// hand-written `no_alibi`/`with_alibi` regs. Forward FlashAttn and the three
    /// backward selectors share the single `OpParams::FlashAttn` carrier (no
    /// dedicated backward variant); PagedAttn uses `OpParams::PagedAttn`.
    ///
    /// **PagedAttn is now REGISTRABLE + contract-sourced (FDX gather-sidecar arc,
    /// slice A).** Its `fdx.gather: paged_blocks` pool is import METADATA (§3.9.1):
    /// the block_table / context_lens ride as ordinary U32 operands +
    /// `OpParams::PagedAttn`, so registration does not depend on the FDX gather
    /// VIEW (which stays [consumer-ahead]). Its bindings are asserted
    /// contract-sourced alongside FlashAttn below — BORN-RED discriminator: the
    /// deleted hand-written regs stamped an empty `kernel_source`, so the
    /// `kernel_source == "portable-cpu"` assertion failed until the four paged
    /// sections were flipped `registrable: true` + wired into
    /// `CPU_ATTENTION_ENTRY_POINTS`. The SEPARATE `FusedKernelRegistry`
    /// FLASH_ATTN* / PAGED_ATTN seam (`register_default_fused_kernels`) is
    /// untouched (stays hand-written, exactly as before).
    ///
    /// For each of the 40 keys: resolves to the EXACT production wrapper with
    /// `kernel_source == "portable-cpu"` (the contract provenance tag), caps
    /// contiguous, and the contract's bit-stable precision riding through.
    #[test]
    fn global_bindings_registers_attention_family_from_contract() {
        let table = global_bindings();
        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        // Forward FlashAttn wrappers (order matches `dts`); the same wrapper
        // serves both the no-alibi [q,k,v,out] and with-alibi [q,k,v,alibi,out] keys.
        let fa_wrappers: [crate::kernel::KernelRef; 4] = [
            flash_attn_f32_cpu_wrapper,
            flash_attn_f64_cpu_wrapper,
            flash_attn_bf16_cpu_wrapper,
            flash_attn_f16_cpu_wrapper,
        ];
        // FlashAttnBackward{Q,K,V} wrappers (order matches `dts`).
        let fabq: [crate::kernel::KernelRef; 4] = [
            flash_attn_backward_q_f32_cpu_wrapper,
            flash_attn_backward_q_f64_cpu_wrapper,
            flash_attn_backward_q_bf16_cpu_wrapper,
            flash_attn_backward_q_f16_cpu_wrapper,
        ];
        let fabk: [crate::kernel::KernelRef; 4] = [
            flash_attn_backward_k_f32_cpu_wrapper,
            flash_attn_backward_k_f64_cpu_wrapper,
            flash_attn_backward_k_bf16_cpu_wrapper,
            flash_attn_backward_k_f16_cpu_wrapper,
        ];
        let fabv: [crate::kernel::KernelRef; 4] = [
            flash_attn_backward_v_f32_cpu_wrapper,
            flash_attn_backward_v_f64_cpu_wrapper,
            flash_attn_backward_v_bf16_cpu_wrapper,
            flash_attn_backward_v_f16_cpu_wrapper,
        ];
        // PagedAttn wrappers (order matches `dts`); the same wrapper serves both
        // the no-alibi [q,kc,vc,bt,cl,out] and with-alibi (+alibi) keys.
        let pa_wrappers: [crate::kernel::KernelRef; 4] = [
            paged_attn_f32_cpu_wrapper,
            paged_attn_f64_cpu_wrapper,
            paged_attn_bf16_cpu_wrapper,
            paged_attn_f16_cpu_wrapper,
        ];

        // Assert `op`/`key` resolves to `expected` with the contract provenance,
        // FINDING it among any alternatives at the key (never single-alternative).
        let check = |op: OpKind, key: &[DType], expected: crate::kernel::KernelRef, label: &str| {
            let alts = table.lookup_alternatives(op, key, BackendId::Cpu);
            let entry = alts
                .iter()
                .find(|e| e.kernel as usize == expected as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "{op:?}/{label}/Cpu: the production wrapper must be bound \
                         FROM the attention contract in global_bindings() — found {} \
                         alternative(s) with sources {:?}",
                        alts.len(),
                        alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                    )
                });
            assert_eq!(
                entry.kernel_source, "portable-cpu",
                "{op:?}/{label}: family must be contract-sourced \
                 (kernel_source=\"portable-cpu\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "{op:?}/{label}: caps preserved (contiguous-only)",
            );
            assert!(
                entry.precision.bit_stable_on_same_hardware,
                "{op:?}/{label}: contract's bit-stable claim rode through",
            );
        };

        let mut checked = 0usize;
        for (i, dt) in dts.iter().enumerate() {
            // Forward: no-alibi [q,k,v,out] + with-alibi [q,k,v,alibi,out].
            let fa_noa: &[DType] = &[*dt, *dt, *dt, *dt];
            let fa_a: &[DType] = &[*dt, *dt, *dt, *dt, *dt];
            check(OpKind::FlashAttn, fa_noa, fa_wrappers[i], "flash no-alibi");
            check(OpKind::FlashAttn, fa_a, fa_wrappers[i], "flash with-alibi");
            checked += 2;

            // Backward Q/K/V: no-alibi [q,k,v,do,out] + with-alibi [q,k,v,do,alibi,out].
            let bw_noa: &[DType] = &[*dt, *dt, *dt, *dt, *dt];
            let bw_a: &[DType] = &[*dt, *dt, *dt, *dt, *dt, *dt];
            for (op, ws) in [
                (OpKind::FlashAttnBackwardQ, &fabq),
                (OpKind::FlashAttnBackwardK, &fabk),
                (OpKind::FlashAttnBackwardV, &fabv),
            ] {
                check(op, bw_noa, ws[i], "backward no-alibi");
                check(op, bw_a, ws[i], "backward with-alibi");
                checked += 2;
            }

            // PagedAttn (FDX gather-sidecar arc, slice A): now CONTRACT-SOURCED
            // like FlashAttn. no-alibi [q,kc,vc,bt:U32,cl:U32,out] + with-alibi
            // (+alibi). BORN-RED discriminator: before the migration the paged
            // bindings were hand-written (`kernel_source == ""`); `check` requires
            // `kernel_source == "portable-cpu"`, so it fails until the four paged
            // sections are flipped `registrable: true` + resolved via
            // CPU_ATTENTION_ENTRY_POINTS. Proves the gather block is import
            // metadata, not a registration blocker.
            let pa_noa: &[DType] = &[*dt, *dt, *dt, DType::U32, DType::U32, *dt];
            let pa_a: &[DType] = &[*dt, *dt, *dt, DType::U32, DType::U32, *dt, *dt];
            check(OpKind::PagedAttn, pa_noa, pa_wrappers[i], "paged no-alibi");
            check(OpKind::PagedAttn, pa_a, pa_wrappers[i], "paged with-alibi");
            checked += 2;
        }
        assert_eq!(
            checked, 40,
            "FlashAttn(8) + FlashAttnBackward{{Q,K,V}}(24) + PagedAttn(8) = 40 \
             contract-sourced bindings",
        );
    }

    /// The CPU **in-place scalar-param** family is registered FROM its FKC
    /// contract (`docs/kernel-contracts/cpu/inplace-unary-affine.fkc.md`) via
    /// `register_cpu_inplace_from_contract`, NOT the deleted hand-written
    /// `table.register(<Op>Inplace / InplaceAffine / ClampInplace / PowIInplace,
    /// &unary(dt), …)` regs. 21 in-place unary ops (each fanning `[F32,F64,BF16,
    /// F16]` = 84) + `InplaceAffine` / `ClampInplace` / `PowIInplace` (4 dtypes
    /// each = 12) = 96 bindings, all keyed `[T, T]` (the single `out` operand +
    /// its `passthrough(out)` mirror; the executor's `InplaceKernel` arm passes
    /// the target as `outputs[0]`, so the wrapper takes 0 inputs + 1 output, but
    /// the binding-table KEY is `[T, T]`).
    ///
    /// For each of the 96 keys: resolves to the EXACT production wrapper with
    /// `kernel_source == "portable-cpu"` (the contract provenance tag), caps
    /// contiguous, and the contract's bit-stable precision riding through.
    #[test]
    fn global_bindings_registers_inplace_family_from_contract() {
        let table = global_bindings();
        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        // Assert `op`/`[dt,dt]` resolves to `expected` with the contract
        // provenance, FINDING it among any alternatives at the key.
        let check = |op: OpKind, dt: DType, expected: crate::kernel::KernelRef, label: &str| {
            let key: &[DType] = &[dt, dt];
            let alts = table.lookup_alternatives(op, key, BackendId::Cpu);
            let entry = alts
                .iter()
                .find(|e| e.kernel as usize == expected as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "{op:?}/{label}/{dt:?}/Cpu: the production wrapper must be bound \
                         FROM the in-place contract in global_bindings() — found {} \
                         alternative(s) with sources {:?}",
                        alts.len(),
                        alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                    )
                });
            assert_eq!(
                entry.kernel_source, "portable-cpu",
                "{op:?}/{label}/{dt:?}: family must be contract-sourced \
                 (kernel_source=\"portable-cpu\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "{op:?}/{label}/{dt:?}: caps preserved (contiguous-only)",
            );
            assert!(
                entry.precision.bit_stable_on_same_hardware,
                "{op:?}/{label}/{dt:?}: contract's bit-stable claim rode through",
            );
        };

        // 21 in-place unary ops × 4 dtypes (wrapper order matches `dts`).
        // Each row: (OpKind, label, [f32, f64, bf16, f16] wrappers).
        let unary_family: &[(OpKind, &str, [crate::kernel::KernelRef; 4])] = &[
            (OpKind::ReluInplace, "relu", [relu_inplace_f32_cpu_wrapper, relu_inplace_f64_cpu_wrapper, relu_inplace_bf16_cpu_wrapper, relu_inplace_f16_cpu_wrapper]),
            (OpKind::SiluInplace, "silu", [silu_inplace_f32_cpu_wrapper, silu_inplace_f64_cpu_wrapper, silu_inplace_bf16_cpu_wrapper, silu_inplace_f16_cpu_wrapper]),
            (OpKind::GeluInplace, "gelu", [gelu_inplace_f32_cpu_wrapper, gelu_inplace_f64_cpu_wrapper, gelu_inplace_bf16_cpu_wrapper, gelu_inplace_f16_cpu_wrapper]),
            (OpKind::TanhInplace, "tanh", [tanh_inplace_f32_cpu_wrapper, tanh_inplace_f64_cpu_wrapper, tanh_inplace_bf16_cpu_wrapper, tanh_inplace_f16_cpu_wrapper]),
            (OpKind::SigmoidInplace, "sigmoid", [sigmoid_inplace_f32_cpu_wrapper, sigmoid_inplace_f64_cpu_wrapper, sigmoid_inplace_bf16_cpu_wrapper, sigmoid_inplace_f16_cpu_wrapper]),
            (OpKind::NegInplace, "neg", [neg_inplace_f32_cpu_wrapper, neg_inplace_f64_cpu_wrapper, neg_inplace_bf16_cpu_wrapper, neg_inplace_f16_cpu_wrapper]),
            (OpKind::AbsInplace, "abs", [abs_inplace_f32_cpu_wrapper, abs_inplace_f64_cpu_wrapper, abs_inplace_bf16_cpu_wrapper, abs_inplace_f16_cpu_wrapper]),
            (OpKind::SqrInplace, "sqr", [sqr_inplace_f32_cpu_wrapper, sqr_inplace_f64_cpu_wrapper, sqr_inplace_bf16_cpu_wrapper, sqr_inplace_f16_cpu_wrapper]),
            (OpKind::SqrtInplace, "sqrt", [sqrt_inplace_f32_cpu_wrapper, sqrt_inplace_f64_cpu_wrapper, sqrt_inplace_bf16_cpu_wrapper, sqrt_inplace_f16_cpu_wrapper]),
            (OpKind::RsqrtInplace, "rsqrt", [rsqrt_inplace_f32_cpu_wrapper, rsqrt_inplace_f64_cpu_wrapper, rsqrt_inplace_bf16_cpu_wrapper, rsqrt_inplace_f16_cpu_wrapper]),
            (OpKind::RecipInplace, "recip", [recip_inplace_f32_cpu_wrapper, recip_inplace_f64_cpu_wrapper, recip_inplace_bf16_cpu_wrapper, recip_inplace_f16_cpu_wrapper]),
            (OpKind::ExpInplace, "exp", [exp_inplace_f32_cpu_wrapper, exp_inplace_f64_cpu_wrapper, exp_inplace_bf16_cpu_wrapper, exp_inplace_f16_cpu_wrapper]),
            (OpKind::LogInplace, "log", [log_inplace_f32_cpu_wrapper, log_inplace_f64_cpu_wrapper, log_inplace_bf16_cpu_wrapper, log_inplace_f16_cpu_wrapper]),
            (OpKind::SinInplace, "sin", [sin_inplace_f32_cpu_wrapper, sin_inplace_f64_cpu_wrapper, sin_inplace_bf16_cpu_wrapper, sin_inplace_f16_cpu_wrapper]),
            (OpKind::CosInplace, "cos", [cos_inplace_f32_cpu_wrapper, cos_inplace_f64_cpu_wrapper, cos_inplace_bf16_cpu_wrapper, cos_inplace_f16_cpu_wrapper]),
            (OpKind::SignInplace, "sign", [sign_inplace_f32_cpu_wrapper, sign_inplace_f64_cpu_wrapper, sign_inplace_bf16_cpu_wrapper, sign_inplace_f16_cpu_wrapper]),
            (OpKind::FloorInplace, "floor", [floor_inplace_f32_cpu_wrapper, floor_inplace_f64_cpu_wrapper, floor_inplace_bf16_cpu_wrapper, floor_inplace_f16_cpu_wrapper]),
            (OpKind::CeilInplace, "ceil", [ceil_inplace_f32_cpu_wrapper, ceil_inplace_f64_cpu_wrapper, ceil_inplace_bf16_cpu_wrapper, ceil_inplace_f16_cpu_wrapper]),
            (OpKind::RoundInplace, "round", [round_inplace_f32_cpu_wrapper, round_inplace_f64_cpu_wrapper, round_inplace_bf16_cpu_wrapper, round_inplace_f16_cpu_wrapper]),
            (OpKind::ErfInplace, "erf", [erf_inplace_f32_cpu_wrapper, erf_inplace_f64_cpu_wrapper, erf_inplace_bf16_cpu_wrapper, erf_inplace_f16_cpu_wrapper]),
            (OpKind::GeluErfInplace, "gelu_erf", [gelu_erf_inplace_f32_cpu_wrapper, gelu_erf_inplace_f64_cpu_wrapper, gelu_erf_inplace_bf16_cpu_wrapper, gelu_erf_inplace_f16_cpu_wrapper]),
        ];

        // Affine / clamp / powi — scalar-param single sections. Note the affine
        // wrapper name is `inplace_affine_<dt>_cpu_wrapper` (words swapped vs the
        // `affine_inplace_<dt>` symbol).
        let scalar_family: &[(OpKind, &str, [crate::kernel::KernelRef; 4])] = &[
            (OpKind::InplaceAffine, "affine", [inplace_affine_f32_cpu_wrapper, inplace_affine_f64_cpu_wrapper, inplace_affine_bf16_cpu_wrapper, inplace_affine_f16_cpu_wrapper]),
            (OpKind::ClampInplace, "clamp", [clamp_inplace_f32_cpu_wrapper, clamp_inplace_f64_cpu_wrapper, clamp_inplace_bf16_cpu_wrapper, clamp_inplace_f16_cpu_wrapper]),
            (OpKind::PowIInplace, "powi", [powi_inplace_f32_cpu_wrapper, powi_inplace_f64_cpu_wrapper, powi_inplace_bf16_cpu_wrapper, powi_inplace_f16_cpu_wrapper]),
        ];

        let mut checked = 0usize;
        for (op, label, ws) in unary_family.iter().chain(scalar_family.iter()) {
            for (i, dt) in dts.iter().enumerate() {
                check(*op, *dt, ws[i], label);
                checked += 1;
            }
        }
        assert_eq!(
            checked, 96,
            "21 in-place unary (×4) + InplaceAffine/ClampInplace/PowIInplace (×4) = 96 \
             contract-sourced bindings",
        );
    }

    /// Gate for the CPU **cast** family migrated to FKC-contract registration.
    /// The FULL directed-pair matrix — every ordered pair of the 11 real numeric
    /// dtypes {F32,F64,F16,BF16,F8E4M3,U8,I8,U32,I16,I32,I64}, identity excluded =
    /// 11 × 10 = 110 pairs — is IMPORTED from
    /// `docs/kernel-contracts/cpu/cast.fkc.md` (see
    /// `register_cpu_cast_from_contract`). Each per-pair section declares a
    /// single-dtype `src` input + `fixed(DST)` output, so it does NOT dtype-fan —
    /// the importer keys `[SRC, DST]` (byte-for-byte the deleted hand-written
    /// `table.register(Cast, &[SRC, DST], …)` regs) and resolves the SPECIFIC
    /// `cast_<src>_to_<dst>` byte-kernel symbol AS-IS through the production
    /// `CpuLinkRegistry` (chaining `CPU_CAST_ENTRY_POINTS`). Because the binding
    /// lookup is keyed on the TARGET dtype, every one of a target's 10 source
    /// pairs resolves to the SAME per-target `cast_to_<dst>_cpu_wrapper` (which
    /// dispatches on source internally) — the fanned byte-kernel symbols are the
    /// synthetic-umbrella precedent (10 distinct entry_points → 1 wrapper).
    ///
    /// For each of the 110 pairs: resolves to the EXACT production per-target
    /// wrapper with `kernel_source == "portable-cpu"` (the contract provenance
    /// tag), caps contiguous-only, and the contract's bit-stable precision riding
    /// through. FKC is the SOLE registration path for the whole family.
    #[test]
    fn global_bindings_registers_cast_family_from_contract() {
        let table = global_bindings();

        // Per-target wrapper (the binding-table key is keyed on the TARGET dtype;
        // each wrapper dispatches on the source dtype internally).
        let targets: [(DType, crate::kernel::KernelRef); 11] = [
            (DType::F32, cast_to_f32_cpu_wrapper),
            (DType::F64, cast_to_f64_cpu_wrapper),
            (DType::F16, cast_to_f16_cpu_wrapper),
            (DType::BF16, cast_to_bf16_cpu_wrapper),
            (DType::F8E4M3, cast_to_f8e4m3_cpu_wrapper),
            (DType::U8, cast_to_u8_cpu_wrapper),
            (DType::I8, cast_to_i8_cpu_wrapper),
            (DType::U32, cast_to_u32_cpu_wrapper),
            (DType::I16, cast_to_i16_cpu_wrapper),
            (DType::I32, cast_to_i32_cpu_wrapper),
            (DType::I64, cast_to_i64_cpu_wrapper),
        ];
        let all_dts: [DType; 11] = [
            DType::F32, DType::F64, DType::F16, DType::BF16, DType::F8E4M3,
            DType::U8, DType::I8, DType::U32, DType::I16, DType::I32, DType::I64,
        ];

        let mut checked = 0usize;
        for (dst, expected) in targets {
            for src in all_dts {
                if src == dst {
                    continue; // identity pairs are elided by the optimizer, not registered
                }
                let key: &[DType] = &[src, dst];
                let alts = table.lookup_alternatives(OpKind::Cast, key, BackendId::Cpu);
                let entry = alts
                    .iter()
                    .find(|e| e.kernel as usize == expected as usize)
                    .unwrap_or_else(|| {
                        panic!(
                            "Cast {src:?}->{dst:?}/Cpu: the production per-target wrapper must be \
                             bound FROM the cast contract in global_bindings() — found {} \
                             alternative(s) with sources {:?}",
                            alts.len(),
                            alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                        )
                    });
                assert_eq!(
                    entry.kernel_source, "portable-cpu",
                    "Cast {src:?}->{dst:?}: family must be contract-sourced \
                     (kernel_source=\"portable-cpu\"); got {:?}",
                    entry.kernel_source,
                );
                assert!(
                    !entry.caps.strided_input,
                    "Cast {src:?}->{dst:?}: caps preserved (contiguous-only)",
                );
                assert!(
                    entry.precision.bit_stable_on_same_hardware,
                    "Cast {src:?}->{dst:?}: contract's bit-stable claim rode through",
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked, 110,
            "11 target dtypes × 10 sources each = 110 directed cast pairs (identity excluded)"
        );
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

    // =========================================================================
    // Step E Phase C / B1 — DEVICE_INFLIGHT counter (the BALANCE gate)
    // =========================================================================
    //
    // The counter is PROCESS-WIDE, so these tests use sentinel `gpu_id`s no
    // real device uses (and that no other test in this binary touches) to stay
    // independent of any concurrent counter activity.

    /// The crux of B1: after a balanced sequence of `inc`/`dec`, the per-device
    /// count returns to its baseline. Mirrors the executor's invariant — every
    /// async submit (`inc`) is matched by exactly one completion-handle `Drop`
    /// (`dec`), so after a realize fully drains the counter is back where it
    /// started (the `drain_handles` empty-map assert, in counter form).
    #[test]
    fn inflight_inc_dec_balances_to_baseline() {
        let loc = DeviceLocation::Cuda { gpu_id: 99_001 };
        let base = inflight_count(loc);

        // N submits in flight → count rises by N.
        const N: u32 = 7;
        for i in 0..N {
            inflight_inc(loc);
            assert_eq!(inflight_count(loc), base + i + 1, "inc must raise the count");
        }
        assert_eq!(inflight_count(loc), base + N);

        // N completions retire → count falls back to baseline.
        for i in 0..N {
            inflight_dec(loc);
            assert_eq!(
                inflight_count(loc),
                base + N - i - 1,
                "dec must lower the count",
            );
        }
        assert_eq!(
            inflight_count(loc),
            base,
            "balanced inc/dec must return the count to its pre-sequence baseline",
        );
    }

    /// `inflight_dec` is underflow-saturating: decrementing a device whose count
    /// is already 0 (or that has never been touched) leaves it at 0, never wraps
    /// to `u32::MAX`. A stray dec is harmless to scheduling, never a poison.
    #[test]
    fn inflight_dec_is_underflow_safe() {
        let untouched = DeviceLocation::Vulkan { gpu_id: 99_002 };
        // Never incremented: dec must not create a wrapped count.
        inflight_dec(untouched);
        assert_eq!(inflight_count(untouched), 0, "dec on an untouched device stays 0");

        let loc = DeviceLocation::Vulkan { gpu_id: 99_003 };
        inflight_inc(loc);
        inflight_dec(loc);
        assert_eq!(inflight_count(loc), 0);
        // One extra dec past zero (the over-drain guard).
        inflight_dec(loc);
        assert_eq!(inflight_count(loc), 0, "extra dec past zero must saturate, not wrap");
    }

    /// Distinct `DeviceLocation`s have independent slots — incrementing one
    /// device's count never perturbs another's (the per-device keying the
    /// selector relies on to compare CUDA vs Vulkan load).
    #[test]
    fn inflight_counts_are_per_device_independent() {
        let cuda = DeviceLocation::Cuda { gpu_id: 99_004 };
        let vulkan = DeviceLocation::Vulkan { gpu_id: 99_004 };
        let base_c = inflight_count(cuda);
        let base_v = inflight_count(vulkan);

        inflight_inc(cuda);
        inflight_inc(cuda);
        inflight_inc(vulkan);

        assert_eq!(inflight_count(cuda), base_c + 2);
        assert_eq!(inflight_count(vulkan), base_v + 1);

        inflight_dec(cuda);
        inflight_dec(cuda);
        inflight_dec(vulkan);
        assert_eq!(inflight_count(cuda), base_c);
        assert_eq!(inflight_count(vulkan), base_v);
    }

    /// FKC cost unification — Part A. `derive_backend_caps` reads a
    /// backend's kernels out of the binding table and advertises the
    /// `(OpKind, OUTPUT-dtype)` pairs those kernels cover — the general
    /// analogue of the hand-maintained `default_cpu_caps`. Verifies:
    /// (1) only the requested backend's keys are picked; (2) the OUTPUT
    /// dtype (last operand) is what's advertised for multi-dtype keys;
    /// (3) substrate/device match the backend. Feature-independent: it
    /// builds a fresh table with synthetic wrappers.
    #[test]
    fn derive_backend_caps_mirrors_binding_table_output_dtypes() {
        fn stub(
            _i: &[Arc<RwLock<Storage>>],
            _o: &mut [Arc<RwLock<Storage>>],
            _l: &[Layout],
            _p: &OpParams,
        ) -> Result<()> {
            Ok(())
        }
        fn stub_b(
            _i: &[Arc<RwLock<Storage>>],
            _o: &mut [Arc<RwLock<Storage>>],
            _l: &[Layout],
            _p: &OpParams,
        ) -> Result<()> {
            Ok(())
        }
        fn stub_c(
            _i: &[Arc<RwLock<Storage>>],
            _o: &mut [Arc<RwLock<Storage>>],
            _l: &[Layout],
            _p: &OpParams,
        ) -> Result<()> {
            Ok(())
        }

        let mut table = KernelBindingTable::new();
        // A CPU key that MUST NOT bleed into the Vulkan derivation.
        table.register(OpKind::AddElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Cpu, stub);
        // Vulkan: an elementwise key (single output dtype) ...
        table.register(OpKind::MulElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Vulkan, stub_b);
        // ... and a MIXED-dtype key: inputs F16/F16, OUTPUT F32. The
        // derived pair must key on the OUTPUT dtype (F32), not an input.
        table.register(OpKind::MatMul, &[DType::F16, DType::F16, DType::F32], BackendId::Vulkan, stub_c);

        let vk0 = DeviceLocation::Vulkan { gpu_id: 0 };
        let caps = derive_backend_caps(BackendId::Vulkan, vk0, &table);

        assert_eq!(caps.backend_id, BackendId::Vulkan);
        assert_eq!(caps.device_location, vk0);
        assert_eq!(caps.storage_substrate, SubstrateClass::VulkanBuffer);
        // Elementwise output dtype advertised.
        assert!(caps.op_dtype_support.contains(&(OpKind::MulElementwise, DType::F32)));
        // Multi-dtype key advertises the OUTPUT dtype (F32), not F16.
        assert!(caps.op_dtype_support.contains(&(OpKind::MatMul, DType::F32)));
        assert!(!caps.op_dtype_support.contains(&(OpKind::MatMul, DType::F16)));
        // The CPU key is NOT attributed to Vulkan.
        assert!(!caps.op_dtype_support.contains(&(OpKind::AddElementwise, DType::F32)));
        // A universal host-staging path back to CPU is advertised so the
        // transfer matrix can price GPU↔CPU crossings.
        assert!(caps
            .transfer_paths
            .iter()
            .any(|&(dst, path)| dst == DeviceLocation::Cpu && path == TransferPath::HostStaging));
    }

    /// CapturedRun executor build-out (4b-ζ): `copy_from_cuda_wrapper`'s
    /// CUDA-output branch converted from allocate-and-replace
    /// (`slot_copy_to_new` + `*dst = copied`) to write-into
    /// (`CudaStorageBytes::copy_from_device`). Byte-compares the new
    /// write-into result against the OLD `slot_copy_to_new`-based behavior
    /// for a non-trivial (17-element, mixed-value) case, AND confirms the
    /// destination's buffer identity (device address) is preserved — the
    /// property capture-safety depends on (allocate-and-replace would swap
    /// in a fresh `Arc<DeviceBuffer<u8>>`, changing the address; write-into
    /// must not).
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn copy_from_cuda_wrapper_cuda_output_write_into_matches_old_behavior() {
        use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};

        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };

        let src_data: Vec<f32> = (0..17).map(|i| i as f32 * 1.5 - 3.0).collect();
        let src_bytes: Vec<u8> = src_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let src_cb = CudaStorageBytes::from_cpu_bytes(&dev, &src_bytes).expect("h2d src");
        let src_arc: Arc<RwLock<Storage>> =
            Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(src_cb), DType::F32)));

        // Pre-allocated output filled with distinguishable garbage — proves
        // the write-into path actually overwrites every byte, not just
        // reads through stale garbage by luck.
        let garbage: Vec<u8> = vec![0xAAu8; src_bytes.len()];
        let dst_cb = CudaStorageBytes::from_cpu_bytes(&dev, &garbage).expect("h2d garbage");
        let dst_arc: Arc<RwLock<Storage>> =
            Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(dst_cb), DType::F32)));

        // Destination buffer identity BEFORE the wrapper runs.
        let before_ptr = {
            let g = dst_arc.read().unwrap();
            let BackendStorage::Cuda(c) = &g.inner else { panic!("not cuda") };
            c.buffer() as *const _ as usize
        };

        let inputs = vec![Arc::clone(&src_arc)];
        let mut outputs = vec![Arc::clone(&dst_arc)];
        copy_from_cuda_wrapper(&inputs, &mut outputs, &[], &OpParams::None)
            .expect("copy_from_cuda_wrapper (write-into CUDA branch)");

        let after_ptr = {
            let g = dst_arc.read().unwrap();
            let BackendStorage::Cuda(c) = &g.inner else { panic!("not cuda") };
            c.buffer() as *const _ as usize
        };
        assert_eq!(
            before_ptr, after_ptr,
            "write-into must preserve the output's buffer identity (device address)"
        );

        let new_bytes = {
            let g = dst_arc.read().unwrap();
            let BackendStorage::Cuda(c) = &g.inner else { panic!("not cuda") };
            c.to_cpu_bytes().expect("d2h new")
        };

        // Independently reproduce the OLD allocate-and-replace behavior
        // (`slot_copy_to_new`) from the SAME source, for comparison.
        let old_bytes = {
            let g = src_arc.read().unwrap();
            let BackendStorage::Cuda(c) = &g.inner else { panic!("not cuda") };
            let copied = c
                .slot_copy_to_new(0, c.len_bytes())
                .expect("slot_copy_to_new (old allocate-and-replace path)");
            copied.to_cpu_bytes().expect("d2h old")
        };

        assert_eq!(
            new_bytes, old_bytes,
            "write-into result must byte-match the old slot_copy_to_new result"
        );
        assert_eq!(new_bytes, src_bytes, "and both must match the source bytes");
    }
}
