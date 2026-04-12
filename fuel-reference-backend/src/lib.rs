//! # fuel-reference-backend
//!
//! A pure-Rust, correctness-first reference implementation of tensor
//! operations for the fuel ML framework.
//!
//! ## Purpose
//!
//! This crate exists for one reason: to serve as an independent correctness
//! oracle against which every other backend and every fused kernel is
//! validated. Its implementations are deliberately the simplest possible
//! expression of each operation — textbook math, scalar loops, no SIMD, no
//! GPU, no BLAS, no `rayon`, no cleverness.
//!
//! The slowness is the point. If the implementation here is too simple to be
//! wrong, it can be trusted as ground truth; other backends are validated by
//! comparing their outputs against this one on a matrix of inputs.
//!
//! ## Non-goals
//!
//! - **Speed.** Use `fuel-cpu-backend` for production CPU execution. The
//!   reference is expected to be many times slower and that is a feature.
//! - **Completeness across all dtypes.** The reference starts with `f32` and
//!   grows by dtype as validation needs require. Mechanical to extend.
//! - **Reuse.** This crate is strict about not being used for anything except
//!   correctness validation. If you find yourself reaching for it as "the
//!   simple CPU path," stop — use `fuel-cpu-backend` instead.
//!
//! ## Structure
//!
//! - [`RefTensor`] wraps a contiguous `Vec<T>` plus a [`Shape`]. It holds one
//!   logical tensor on the heap with no stride handling. Callers materialize
//!   any non-contiguous view into contiguous form before handing it here.
//! - [`ops`] provides the per-op reference implementations.
//! - [`exec`] walks a [`fuel_graph::Graph`] and executes each node using the
//!   reference implementations in [`ops`], returning a concrete
//!   [`RefTensor`] for any requested output. This is the bridge between the
//!   lazy graph layer and the textbook implementations that serve as the
//!   correctness oracle.
//!
//! ## Role in Phase 6
//!
//! `fuel-reference-backend` is the first hard prerequisite for Phase 6
//! (lazy execution and autonomous scheduling). Every other backend and every
//! fused kernel must pass oracle-equivalence against it before landing.

pub mod exec;
pub mod ops;

use fuel_core_types::Shape;
use std::sync::Arc;

/// A reference tensor: contiguous data on the heap, explicit shape, no
/// stride arithmetic. If you need to reference a non-contiguous view,
/// materialize it into contiguous form first.
///
/// Backing storage is `Arc<[T]>` so that cloning a `RefTensor` — which
/// happens every time the executor caches a node — is an atomic bump
/// instead of a full memcpy. This matters enormously for large model
/// weights that flow through the graph as `Const` nodes: a TinyLlama
/// forward pass moves ~4 GiB of weights per call, and without the Arc
/// share that becomes a 4 GiB memcpy every forward.
#[derive(Clone, Debug)]
pub struct RefTensor<T> {
    data:  Arc<[T]>,
    shape: Shape,
}

impl<T: Clone> RefTensor<T> {
    /// Build a [`RefTensor`] from raw contiguous data and a shape.
    ///
    /// Takes ownership of the `Vec<T>` and wraps it in an `Arc<[T]>`.
    /// This is the constructor every `ops::` function uses on its way
    /// out — each op builds a fresh `Vec<T>` and hands it off.
    ///
    /// Panics if the element count does not match the shape's total size.
    pub fn from_vec(data: Vec<T>, shape: impl Into<Shape>) -> Self {
        Self::from_arc(Arc::from(data), shape)
    }

    /// Build a [`RefTensor`] from an already shared `Arc<[T]>`. Used by
    /// the executor's `Const` path so weight buffers are never cloned
    /// on their way from the graph into the executor cache.
    pub fn from_arc(data: Arc<[T]>, shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        assert_eq!(
            data.len(),
            shape.elem_count(),
            "RefTensor::from_arc: data length {} does not match shape element count {}",
            data.len(),
            shape.elem_count(),
        );
        Self { data, shape }
    }

    /// Returns the underlying flat data as a slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns the underlying storage as an [`Arc<[T]>`]. Used when a
    /// consumer wants to hold the data cheaply instead of cloning it
    /// out into a fresh `Vec`.
    pub fn as_arc(&self) -> &Arc<[T]> {
        &self.data
    }

    /// Returns the shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the total element count.
    pub fn elem_count(&self) -> usize {
        self.data.len()
    }

    /// Consumes the tensor and returns its flat data as a `Vec<T>`.
    ///
    /// Because the underlying storage is now `Arc<[T]>`, the conversion
    /// back to a `Vec` always performs a copy. This is fine for the
    /// realize path — the only consumer — because final outputs like
    /// logits are much smaller than the weight tensors whose clones
    /// the `Arc` is there to avoid.
    pub fn into_vec(self) -> Vec<T> {
        self.data.to_vec()
    }
}
