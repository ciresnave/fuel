//! Phase 7.5 work item B3 step 2 — lazy realization of node-handle
//! Tensors through the executor stack.
//!
//! [`realize_into_storage`] is the dtype-erased entry point: given a
//! `fuel_graph::Tensor` whose graph slot is empty, walk the graph
//! through fuel-graph-cpu's executor, populate the slot with the
//! resulting `CpuStorage`, and return the slot's `Arc`.
//!
//! ## Why this shape
//!
//! The `realized_storage()` seam on `Tensor_` returns
//! `Arc<RwLock<Storage>>` — a dtype-erased Arc. So the realize
//! entry point it calls also has to take an erased input
//! (`&fuel_graph::Tensor`) and produce an erased output. The
//! per-dtype realize functions in fuel-graph-cpu are typed
//! (`realize_f32`, `realize_f64`, ...). The bridge between the two
//! worlds is one runtime → compile-time dispatch, which lives here
//! as the `DtypeRealizer` trait + a small lookup function.
//!
//! `DtypeRealizer` is a visitor-style trait: each impl knows its
//! `T` at compile time and calls the monomorphic `realize_<T>`. The
//! trait object dispatch from [`realizer_for_dtype`] replaces what
//! would otherwise be a match-on-dtype scattered at every call
//! site. Centralizing it in one named layer makes future expansion
//! (Vulkan/CUDA realize, additional dtypes) a matter of adding new
//! impls and one match arm — no change to callers.
//!
//! ## What's plumbed today
//!
//! Only `F32`, `F64`, `BF16`, `F16`, and `U32` are realizable —
//! those are the dtypes fuel-graph-cpu's `AnyTensor` enum supports.
//! Other dtypes return a typed error pointing at the gap. Closing
//! that gap is a separate concern (extending `AnyTensor` and the
//! per-op kernels in fuel-graph-cpu).
//!
//! ## What's NOT plumbed yet
//!
//! - **Multi-backend dispatch.** Today every realize routes through
//!   fuel-graph-cpu, producing a `CpuStorage`. Vulkan/CUDA realize
//!   would route through their respective `GraphExecutor`s and
//!   produce `VulkanStorage` / `CudaStorage`. That's a follow-up
//!   commit; the device polymorphism plugs into the same trait
//!   object, just with more impls or a richer `realize` signature.
//! - **Router / dispatch table.** [`crate::lazy::LazyTensor::realize_f32`]
//!   threads through the cached dispatch table when present. This
//!   entry point doesn't yet — it always uses raw fuel-graph-cpu.
//!   Wiring the table in is mechanical when needed.

use std::sync::{Arc, RwLock};

use fuel_core_types::{bail, DType, HostBuffer, Result, Storage};
use fuel_cpu_backend::CpuStorage;
use half::{bf16, f16};

/// Visitor-style trait that bridges from runtime-known `DType` to
/// compile-time-monomorphic `realize_<T>` calls. Each impl knows its
/// `T` at compile time; selecting the impl is a small table lookup
/// in [`realizer_for_dtype`].
pub trait DtypeRealizer: Sync {
    /// The `DType` this realizer handles. Used by call sites that
    /// want to assert the realized dtype matches expectations.
    fn dtype(&self) -> DType;

    /// Realize `link`'s graph through fuel-graph-cpu, downloading
    /// the resulting bytes as a `Storage` (CpuStorage today).
    ///
    /// Caller is responsible for installing the returned `Storage`
    /// into the graph's `storage_map` for `link.id()` if they want
    /// future reads to see it. [`realize_into_storage`] does this
    /// automatically.
    fn realize(&self, link: &fuel_graph::Tensor) -> Result<Storage>;
}

struct F32Realizer;
struct F64Realizer;
struct Bf16Realizer;
struct F16Realizer;

impl DtypeRealizer for F32Realizer {
    fn dtype(&self) -> DType { DType::F32 }
    fn realize(&self, link: &fuel_graph::Tensor) -> Result<Storage> {
        let typed = fuel_graph_cpu::realize_f32(link);
        let buf = HostBuffer::F32(typed.into_vec());
        Ok(Storage(Box::new(CpuStorage::from(buf))))
    }
}

impl DtypeRealizer for F64Realizer {
    fn dtype(&self) -> DType { DType::F64 }
    fn realize(&self, link: &fuel_graph::Tensor) -> Result<Storage> {
        let typed = fuel_graph_cpu::realize_f64(link);
        let buf = HostBuffer::F64(typed.into_vec());
        Ok(Storage(Box::new(CpuStorage::from(buf))))
    }
}

impl DtypeRealizer for Bf16Realizer {
    fn dtype(&self) -> DType { DType::BF16 }
    fn realize(&self, link: &fuel_graph::Tensor) -> Result<Storage> {
        let typed: fuel_reference_backend::RefTensor<bf16> =
            fuel_graph_cpu::realize_bf16(link);
        let buf = HostBuffer::BF16(typed.into_vec());
        Ok(Storage(Box::new(CpuStorage::from(buf))))
    }
}

impl DtypeRealizer for F16Realizer {
    fn dtype(&self) -> DType { DType::F16 }
    fn realize(&self, link: &fuel_graph::Tensor) -> Result<Storage> {
        let typed: fuel_reference_backend::RefTensor<f16> =
            fuel_graph_cpu::realize_f16(link);
        let buf = HostBuffer::F16(typed.into_vec());
        Ok(Storage(Box::new(CpuStorage::from(buf))))
    }
}

/// Look up the `DtypeRealizer` for a given runtime `DType`.
///
/// Returns `Err` for dtypes that fuel-graph-cpu's executor doesn't
/// support yet (everything except F32/F64/BF16/F16). The error
/// message names the dtype and points at the architectural gap.
///
/// The match is the runtime → compile-time dispatch boundary;
/// callers downstream get a `&'static dyn DtypeRealizer` and the
/// trait object's vtable handles the rest.
pub fn realizer_for_dtype(d: DType) -> Result<&'static dyn DtypeRealizer> {
    match d {
        DType::F32 => Ok(&F32Realizer),
        DType::F64 => Ok(&F64Realizer),
        DType::BF16 => Ok(&Bf16Realizer),
        DType::F16 => Ok(&F16Realizer),
        // U32 is handled by fuel-graph-cpu's AnyTensor internally
        // but not exposed via a public realize_u32 entry point;
        // adding one is the follow-up that lifts this gap.
        other => bail!(
            "lazy realize not yet supported for dtype {:?} — \
             fuel-graph-cpu's AnyTensor enum needs a public \
             realize_{:?} entry point first",
            other,
            other,
        ),
    }
}

/// Realize `link` through the executor and install the resulting
/// `Storage` in the graph's `storage_map` for `link.id()`. Returns
/// the slot's `Arc<RwLock<Storage>>` for the caller to read from.
///
/// Idempotent over a single graph: subsequent calls observe the
/// already-populated slot via slot-first dispatch and re-extract
/// the same Arc. Realization happens only on the first call per
/// (graph, NodeId) combination.
///
/// This is the entry point that B3 step 3 will plug into the
/// `realized_storage()` seam — when the seam sees an empty slot,
/// it calls this. Today this function is standalone; nothing
/// inside `Tensor_` calls it yet.
pub fn realize_into_storage(link: &fuel_graph::Tensor) -> Result<Arc<RwLock<Storage>>> {
    // Fast path: slot already populated (e.g. by a B2 factory or
    // a prior realize call on the same graph). Return its Arc
    // without re-running the executor.
    if let Some(arc) = link.storage_for() {
        return Ok(arc);
    }
    let realizer = realizer_for_dtype(link.dtype())?;
    let storage = realizer.realize(link)?;
    let arc = Arc::new(RwLock::new(storage));
    link.graph().write().unwrap().set_storage(link.id(), arc.clone());
    Ok(arc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use fuel_core_types::Shape;

    /// Realize a single Const node with its slot pre-populated by
    /// the factory: the entry point should hit the fast path and
    /// return the same Arc the slot already holds, without going
    /// through the executor.
    #[test]
    fn realize_into_storage_fast_path_returns_existing_slot() {
        let dev = Device::cpu();
        let link = fuel_graph::Tensor::from_f32(
            vec![1.0_f32, 2.0, 3.0],
            Shape::from_dims(&[3]),
            dev.as_dyn(),
        );
        let pre_arc = link.storage_for().expect("factory populates slot");
        let post_arc = realize_into_storage(&link).expect("fast path");
        assert!(Arc::ptr_eq(&pre_arc, &post_arc), "fast path returns same Arc");
    }

    /// Realize a Relu op whose slot is empty: the entry point
    /// walks the graph through fuel-graph-cpu, populates the slot,
    /// and returns the new Arc with the realized bytes.
    #[test]
    fn realize_into_storage_realizes_unary_op() {
        let dev = Device::cpu();
        let input = fuel_graph::Tensor::from_f32(
            vec![1.0_f32, -2.0, 3.0, -4.0],
            Shape::from_dims(&[4]),
            dev.as_dyn(),
        );
        let relu = input.relu();
        // Pre-realize: slot is empty.
        assert!(relu.storage_for().is_none(), "relu slot starts empty");

        let arc = realize_into_storage(&relu).expect("realize relu");
        assert!(relu.storage_for().is_some(), "slot populated after realize");

        // The realized bytes match relu's semantics.
        let guard = arc.read().unwrap();
        let cpu = guard.downcast_ref::<CpuStorage>().expect("cpu storage");
        match cpu.inner() {
            HostBuffer::F32(vec) => assert_eq!(vec, &[1.0, 0.0, 3.0, 0.0]),
            other => panic!("expected F32 host buffer, got {:?}", other),
        }
    }

    /// Realize a 2-input Add: both inputs slot-populated, the Add
    /// node empty. Executor walks the 3-node graph and produces
    /// the sum.
    #[test]
    fn realize_into_storage_realizes_binary_op() {
        let dev = Device::cpu();
        let a = fuel_graph::Tensor::from_f32(
            vec![1.0_f32, 2.0, 3.0],
            Shape::from_dims(&[3]),
            dev.as_dyn(),
        );
        let b = a.const_f32_like(vec![10.0_f32, 20.0, 30.0], Shape::from_dims(&[3]));
        let sum = a.add(&b);
        assert!(sum.storage_for().is_none(), "add slot starts empty");

        let arc = realize_into_storage(&sum).expect("realize add");
        let guard = arc.read().unwrap();
        let cpu = guard.downcast_ref::<CpuStorage>().expect("cpu storage");
        match cpu.inner() {
            HostBuffer::F32(vec) => assert_eq!(vec, &[11.0, 22.0, 33.0]),
            other => panic!("expected F32 host buffer, got {:?}", other),
        }
    }

    /// Idempotence: realizing twice in a row hits the fast path on
    /// the second call (slot is already populated) and returns the
    /// same Arc.
    #[test]
    fn realize_into_storage_is_idempotent() {
        let dev = Device::cpu();
        let input = fuel_graph::Tensor::from_f32(
            vec![1.0_f32, 2.0],
            Shape::from_dims(&[2]),
            dev.as_dyn(),
        );
        let neg = input.neg();
        let arc1 = realize_into_storage(&neg).expect("first realize");
        let arc2 = realize_into_storage(&neg).expect("second realize");
        assert!(Arc::ptr_eq(&arc1, &arc2), "second call returns same Arc");
    }

    /// Unsupported dtype produces a typed error rather than a
    /// panic — matches the "no panics in production" rule. Probe
    /// with I64 since fuel-graph-cpu's AnyTensor doesn't include it.
    #[test]
    fn realize_into_storage_errors_on_unsupported_dtype() {
        let err = match realizer_for_dtype(DType::I64) {
            Ok(_) => panic!("expected error for unsupported dtype, got Ok"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("I64") || msg.contains("not yet supported"),
            "error message names the gap: {msg}"
        );
    }
}
