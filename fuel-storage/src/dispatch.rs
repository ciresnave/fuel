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

/// Dispatch wrapper for `(AddElementwise, F32, Cpu)`. Extracts
/// CpuStorageBytes from the BackendStorage::Cpu variant of inputs +
/// output, then calls the typed kernel in fuel-cpu-backend.
fn add_elementwise_f32_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "add_elementwise wrapper expects 2 inputs, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "add_elementwise wrapper expects 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let lhs_arc = inputs[0].clone();
    let rhs_arc = inputs[1].clone();
    let out_arc = outputs[0].clone();

    let lhs_guard = lhs_arc.read().unwrap();
    let rhs_guard = rhs_arc.read().unwrap();
    let mut out_guard = out_arc.write().unwrap();

    let lhs_cpu = cpu_input(&lhs_guard)?;
    let rhs_cpu = cpu_input(&rhs_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;

    fuel_cpu_backend::byte_kernels::add_f32(lhs_cpu, rhs_cpu, out_cpu)
}

/// Register CPU dispatch wrappers in the binding table. Call once
/// at process startup or on first table creation. The CPU backend
/// is the universal fallback; its bindings cover every standard
/// (op, dtype) combination after Phase C migration. B5 ships a
/// minimal set: just `(AddElementwise, F32)`.
pub fn register_cpu_kernels(table: &mut KernelBindingTable) {
    table.register(
        OpKind::AddElementwise,
        DType::F32,
        BackendId::Cpu,
        add_elementwise_f32_cpu_wrapper,
    );
    // Phase C: more (op, dtype) registrations land here as kernels
    // migrate.
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
    let mut op_dtype_support = HashSet::new();
    // Bootstrap: only AddElementwise+F32 registered today (matches
    // the binding table). Phase C grows this set as kernels
    // migrate. Eventually the full CPU coverage matrix lives here.
    op_dtype_support.insert((OpKind::AddElementwise, DType::F32));
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
        // BF16 isn't registered — error.
        let result = table.lookup(OpKind::AddElementwise, DType::BF16, BackendId::Cpu);
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
