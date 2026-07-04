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

/// Increment 2b — the consumer half, end-to-end: a **data-determined**
/// `WriteSlice` offset. `NonZeroIndices` produces `count` mid-pass, and a
/// `WriteSlice` writes at `dyn_offset = count_sym`. Because the WriteSlice
/// consumes a producer output (`count`), the producer is an ancestor and
/// executes first — binding `count_sym` into `produced_syms` — so the
/// executor resolves the offset at execute time (not compile). This closes
/// the data-dependent dynamic-shapes loop: producer count → consumer extent.
#[test]
fn nonzero_indices_drives_data_determined_write_slice() {
    use fuel_ir::DynScalar;
    let dev = fuel_core::Device::cpu();
    // flat [0,1,0,1,1,0] → 3 nonzeros → count = 3.
    let x = LazyTensor::from_f32(
        vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        Shape::from_dims(&[6]),
        &dev,
    );
    let mut symgen = SymGen::new();
    let count_sym = symgen.fresh();
    let (_indices, count) = x.nonzero_indices_bundled(count_sym).unwrap();

    // dest [16] U32 zeros; write the 1-element `count` slab at the
    // data-determined offset `count_sym`. WriteSlice consumes `count` (a
    // producer output), so NonZeroIndices runs — binding count_sym = 3 —
    // before WriteSlice resolves the offset from produced_syms. `dest` must
    // live on the SAME graph as `count` (const_*_like), not a fresh graph.
    let dest = count.const_u32_like(vec![0u32; 16], Shape::from_dims(&[16]));
    let written = dest
        .write_slice_dyn(&count, vec![(0, 1)], 0, DynScalar::Sym(count_sym))
        .expect("build data-determined write_slice_dyn");

    let out = written.realize_u32();
    let mut expected = vec![0u32; 16];
    expected[3] = 3; // the count value (3) lands at the data-determined index 3
    assert_eq!(
        out, expected,
        "count(=3) must be written at the data-determined offset 3",
    );
}

/// The dropless-MoE payoff, end-to-end: a **data-determined-M matmul**
/// whose row count is `NonZeroIndices`'s count. The matmul consumes a
/// producer output (its LHS derives from `indices`), so the producer runs
/// first — binding `count_sym` — and the matmul computes exactly `count`
/// rows of its capacity-buffer output, the rest left zero. This is the FLOP
/// saving that makes sparse MoE dispatch worth it, now reachable through
/// the graph.
#[test]
fn nonzero_count_drives_dynamic_m_matmul() {
    use fuel_core::DType;
    use fuel_ir::DynScalar;
    let dev = fuel_core::Device::cpu();
    // [0,1,0,1,1,0] → indices = [1,3,4,0,0,0], count = 3, capacity = 6.
    let x = LazyTensor::from_f32(
        vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        Shape::from_dims(&[6]),
        &dev,
    );
    let mut symgen = SymGen::new();
    let count_sym = symgen.fresh();
    let (indices, _count) = x.nonzero_indices_bundled(count_sym).unwrap();

    // lhs [capacity=6, k=1] F32, derived from `indices` so the matmul
    // transitively depends on the NonZeroIndices producer (it runs first
    // and binds count_sym).
    let lhs = indices
        .to_dtype(DType::F32)
        .unwrap()
        .reshape(Shape::from_dims(&[6, 1]))
        .unwrap();
    // rhs [k=1, n=2].
    let rhs = lhs.const_f32_like(vec![10.0, 100.0], Shape::from_dims(&[1, 2]));
    let out = lhs
        .matmul_dyn_m(&rhs, DynScalar::Sym(count_sym))
        .expect("build dynamic-M matmul");

    let got = out.realize_f32();
    // Only count=3 rows computed: indices[0..3] = [1,3,4], each × [10,100];
    // rows 3..6 are the untouched (zeroed) capacity tail.
    let expected = vec![
        10.0, 100.0, // row 0: idx 1
        30.0, 300.0, // row 1: idx 3
        40.0, 400.0, // row 2: idx 4
        0.0, 0.0, // row 3 (not computed)
        0.0, 0.0, // row 4
        0.0, 0.0, // row 5
    ];
    assert_eq!(
        got, expected,
        "dynamic-M matmul must compute exactly count=3 rows, tail zeroed",
    );
}
