//! Increment 1 (born-red) for the data-dependent dynamic-shapes keystone:
//! `Op::NonZeroIndices` produces, over a fixed-capacity buffer, the flat
//! indices of an input's nonzero elements plus the runtime count — the
//! multi-output bundle (slot 0 = `indices [capacity]` U32, slot 1 =
//! `count [1]` U32). This test exercises the CPU realize path end-to-end
//! and asserts both slots. (The mid-realize `SymEnv` bind of `count` is
//! increment 2 — not asserted here.)

use fuel_core::lazy::LazyTensor;
use fuel_ir::{Shape, SymGen};

/// A mask with three nonzeros at flat positions 1, 3, 4.
#[test]
fn nonzero_indices_f32_basic() {
    let dev = fuel_core::Device::cpu();
    // shape [2, 3]; flat = [0, 1, 0, 1, 1, 0] → nonzeros at 1, 3, 4.
    let x = LazyTensor::from_f32(
        vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        Shape::from_dims(&[2, 3]),
        &dev,
    );
    let mut symgen = SymGen::new();
    let count_sym = symgen.fresh();
    let (indices, count) = x.nonzero_indices_bundled(count_sym).unwrap();

    let count_host = count.realize_u32();
    assert_eq!(count_host, vec![3], "3 nonzero elements");

    let indices_host = indices.realize_u32();
    // capacity == elem_count == 6; first `count` entries are the valid
    // ascending flat indices; the tail is zero padding.
    assert_eq!(indices_host.len(), 6, "indices buffer sized to capacity");
    assert_eq!(&indices_host[..3], &[1, 3, 4], "flat nonzero positions");
}

/// All-zero input → count 0, no valid indices.
#[test]
fn nonzero_indices_all_zero() {
    let dev = fuel_core::Device::cpu();
    let x = LazyTensor::from_f32(vec![0.0; 4], Shape::from_dims(&[4]), &dev);
    let mut symgen = SymGen::new();
    let (indices, count) = x.nonzero_indices_bundled(symgen.fresh()).unwrap();
    assert_eq!(count.realize_u32(), vec![0], "no nonzeros");
    assert_eq!(indices.realize_u32().len(), 4, "capacity preserved");
}

/// All-nonzero input → count == capacity, identity index map.
#[test]
fn nonzero_indices_all_nonzero() {
    let dev = fuel_core::Device::cpu();
    let x = LazyTensor::from_f32(vec![1.0, 2.0, -3.0, 0.5], Shape::from_dims(&[4]), &dev);
    let mut symgen = SymGen::new();
    let (indices, count) = x.nonzero_indices_bundled(symgen.fresh()).unwrap();
    assert_eq!(count.realize_u32(), vec![4], "every element nonzero");
    assert_eq!(indices.realize_u32(), vec![0, 1, 2, 3], "identity index map");
}
