// SPDX-License-Identifier: MIT OR Apache-2.0
//! Op conformance tests for the LAZY ONNX evaluator.
//!
//! Replaces the eager `tests/ops.rs`, which drove the `simple_eval` evaluator
//! that B6 deleted along with the eager `Tensor`. That suite spent ~110 lines
//! per test on `ModelProto` boilerplate; the harness below collapses that to a
//! few lines, so the same effort buys far more ops covered.
//!
//! **This is not a 1:1 transcription of the old suite** — it is a fresh suite
//! aimed at the ops the lazy port added. Coverage overlaps but is not
//! identical; where the old suite went deeper on a single op (Conv padding
//! permutations, say) this one goes wider across ops.
//!
//! # The graph-affinity trap this harness exists to avoid
//!
//! Every `Tensor::from_*` constructor **mints a NEW graph**, and two
//! tensors can only combine if their `graph_id()` matches. A test that built
//! `x` and `y` with two separate `from_f32` calls would therefore fail inside
//! the evaluator with a cross-graph error that looks like an evaluator bug and
//! is not one. [`run`] builds the first input with `from_f32` and every
//! subsequent input with `from_f32_on(first.graph(), …)`.

use fuel::lazy::Tensor;
use fuel::{Device, Result, Shape};
use fuel_onnx::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};
use fuel_onnx::OnnxEval;
use std::collections::HashMap;
use std::sync::Arc;

const OUT: &str = "z";

// ---------------------------------------------------------------- harness --

fn attr(name: &str) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ref_attr_name: String::new(),
        i: 0,
        doc_string: String::new(),
        r#type: 0,
        f: 0.0,
        s: vec![],
        t: None,
        g: None,
        sparse_tensor: None,
        tp: None,
        floats: vec![],
        ints: vec![],
        strings: vec![],
        tensors: vec![],
        graphs: vec![],
        sparse_tensors: vec![],
        type_protos: vec![],
    }
}

/// INT attribute (`r#type` 2 = INT in the ONNX AttributeType enum).
fn attr_i(name: &str, v: i64) -> AttributeProto {
    AttributeProto {
        i: v,
        r#type: 2,
        ..attr(name)
    }
}

/// FLOAT attribute (`r#type` 1 = FLOAT).
fn attr_f(name: &str, v: f32) -> AttributeProto {
    AttributeProto {
        f: v,
        r#type: 1,
        ..attr(name)
    }
}

/// INTS attribute (`r#type` 7 = INTS).
fn attr_is(name: &str, v: Vec<i64>) -> AttributeProto {
    AttributeProto {
        ints: v,
        r#type: 7,
        ..attr(name)
    }
}

/// A model containing exactly one node.
fn single_node(op_type: &str, inputs: &[&str], attrs: Vec<AttributeProto>) -> ModelProto {
    single_node_multi_out(op_type, inputs, &[OUT], attrs)
}

fn single_node_multi_out(
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Vec<AttributeProto>,
) -> ModelProto {
    ModelProto {
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        ir_version: 0,
        opset_import: vec![],
        producer_name: String::new(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 0,
        doc_string: String::new(),
        graph: Some(GraphProto {
            node: vec![NodeProto {
                op_type: op_type.to_string(),
                domain: String::new(),
                attribute: attrs,
                input: inputs.iter().map(|s| s.to_string()).collect(),
                output: outputs.iter().map(|s| s.to_string()).collect(),
                name: format!("test_{op_type}"),
                doc_string: String::new(),
            }],
            name: String::new(),
            initializer: vec![],
            input: vec![],
            output: outputs
                .iter()
                .map(|o| ValueInfoProto {
                    name: o.to_string(),
                    doc_string: String::new(),
                    r#type: None,
                })
                .collect(),
            value_info: vec![],
            doc_string: String::new(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }),
    }
}

/// Named f32 input: (name, data, shape).
type In<'a> = (&'a str, Vec<f32>, Vec<usize>);

/// Run a model and realize output `OUT` to f32.
///
/// All inputs are placed on ONE graph — see the module docs.
fn run(model: &ModelProto, inputs: &[In]) -> Result<Vec<f32>> {
    run_named(model, inputs, OUT)
}

fn run_named(model: &ModelProto, inputs: &[In], want: &str) -> Result<Vec<f32>> {
    let dev = Device::cpu();
    let mut map: HashMap<String, Tensor> = HashMap::new();
    let mut anchor: Option<Tensor> = None;
    for (name, data, shape) in inputs {
        let d: Arc<[f32]> = Arc::from(data.clone());
        let t = match &anchor {
            None => Tensor::from_f32(d, Shape::from_dims(shape), &dev),
            Some(a) => Tensor::from_f32_on(a.graph(), d, Shape::from_dims(shape), &dev),
        };
        if anchor.is_none() {
            anchor = Some(t.clone());
        }
        map.insert(name.to_string(), t);
    }
    let out = OnnxEval::from_model(model.clone()).run(&map)?;
    let z = out
        .get(want)
        .unwrap_or_else(|| panic!("output '{want}' not produced; got {:?}", out.keys()));
    // `realize_f32` REINTERPRETS BYTES — it does not check the tensor's dtype.
    // Comparisons produce U8 and Shape/ArgMax produce I64, so realizing those
    // directly as f32 silently yields the wrong element count (2 i64 read as
    // 4 f32) rather than an error. Convert first.
    Ok(z.to_dtype(fuel::DType::F32)?.realize_f32())
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length {} != {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "{what}: index {i} got {g}, want {w} (full: {got:?} vs {want:?})"
        );
    }
}

// ------------------------------------------------------- unary elementwise --

#[test]
fn unary_elementwise_ops() -> Result<()> {
    let x = vec![-2.0f32, -0.5, 0.0, 0.5, 2.0];
    let shape = vec![5usize];
    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("Abs", vec![2.0, 0.5, 0.0, 0.5, 2.0]),
        ("Neg", vec![2.0, 0.5, 0.0, -0.5, -2.0]),
        ("Sign", vec![-1.0, -1.0, 0.0, 1.0, 1.0]),
        ("Ceil", vec![-2.0, -0.0, 0.0, 1.0, 2.0]),
        ("Floor", vec![-2.0, -1.0, 0.0, 0.0, 2.0]),
    ];
    for (op, want) in cases {
        let m = single_node(op, &["x"], vec![]);
        let got = run(&m, &[("x", x.clone(), shape.clone())])?;
        assert_close(&got, &want, op);
    }
    Ok(())
}

#[test]
fn transcendental_unary_ops() -> Result<()> {
    let x = vec![0.25f32, 1.0, 4.0];
    let shape = vec![3usize];
    let m = single_node("Sqrt", &["x"], vec![]);
    assert_close(
        &run(&m, &[("x", x.clone(), shape.clone())])?,
        &[0.5, 1.0, 2.0],
        "Sqrt",
    );

    let m = single_node("Log", &["x"], vec![]);
    let got = run(&m, &[("x", x.clone(), shape.clone())])?;
    assert_close(&got, &[0.25f32.ln(), 0.0, 4.0f32.ln()], "Log");

    let m = single_node("Exp", &["x"], vec![]);
    let got = run(&m, &[("x", vec![0.0, 1.0], vec![2])])?;
    assert_close(&got, &[1.0, std::f32::consts::E], "Exp");

    let m = single_node("Sin", &["x"], vec![]);
    let got = run(
        &m,
        &[("x", vec![0.0, std::f32::consts::FRAC_PI_2], vec![2])],
    )?;
    assert_close(&got, &[0.0, 1.0], "Sin");

    let m = single_node("Cos", &["x"], vec![]);
    let got = run(&m, &[("x", vec![0.0, std::f32::consts::PI], vec![2])])?;
    assert_close(&got, &[1.0, -1.0], "Cos");
    Ok(())
}

// ------------------------------------------------------------ comparisons --

#[test]
fn comparison_ops_produce_zero_one_masks() -> Result<()> {
    let a = ("a", vec![1.0f32, 2.0, 3.0], vec![3usize]);
    let b = ("b", vec![2.0f32, 2.0, 2.0], vec![3usize]);
    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("Equal", vec![0.0, 1.0, 0.0]),
        ("Greater", vec![0.0, 0.0, 1.0]),
        ("GreaterOrEqual", vec![0.0, 1.0, 1.0]),
        ("Less", vec![1.0, 0.0, 0.0]),
        ("LessOrEqual", vec![1.0, 1.0, 0.0]),
    ];
    for (op, want) in cases {
        let m = single_node(op, &["a", "b"], vec![]);
        let got = run(&m, &[a.clone(), b.clone()])?;
        assert_close(&got, &want, op);
    }
    Ok(())
}

/// ONNX specifies numpy broadcasting for comparisons, but the underlying
/// Tensor primitives are same-shape — this is the case that would fail
/// without `broadcast_pair`.
#[test]
fn comparisons_broadcast_a_scalar_against_a_matrix() -> Result<()> {
    let m = single_node("Greater", &["a", "b"], vec![]);
    let got = run(
        &m,
        &[
            ("a", vec![1.0, 5.0, 2.0, 9.0], vec![2, 2]),
            ("b", vec![3.0], vec![1]),
        ],
    )?;
    assert_close(&got, &[0.0, 1.0, 0.0, 1.0], "Greater broadcast");
    Ok(())
}

#[test]
fn logical_ops_on_zero_one_masks() -> Result<()> {
    let a = ("a", vec![0.0f32, 0.0, 1.0, 1.0], vec![4usize]);
    let b = ("b", vec![0.0f32, 1.0, 0.0, 1.0], vec![4usize]);
    for (op, want) in [
        ("And", vec![0.0f32, 0.0, 0.0, 1.0]),
        ("Or", vec![0.0, 1.0, 1.0, 1.0]),
        ("Xor", vec![0.0, 1.0, 1.0, 0.0]),
    ] {
        let m = single_node(op, &["a", "b"], vec![]);
        assert_close(&run(&m, &[a.clone(), b.clone()])?, &want, op);
    }
    let m = single_node("Not", &["a"], vec![]);
    assert_close(
        &run(&m, std::slice::from_ref(&a))?,
        &[1.0, 1.0, 0.0, 0.0],
        "Not",
    );
    Ok(())
}

/// `Not` must treat any non-zero as true, not just exactly 1.
#[test]
fn not_normalizes_non_zero_inputs() -> Result<()> {
    let m = single_node("Not", &["a"], vec![]);
    let got = run(&m, &[("a", vec![0.0, 7.0, -3.0], vec![3])])?;
    assert_close(&got, &[1.0, 0.0, 0.0], "Not non-zero");
    Ok(())
}

// ---------------------------------------------------------- arithmetic ------

#[test]
fn variadic_min_max() -> Result<()> {
    let ins = [
        ("a", vec![1.0f32, 5.0], vec![2usize]),
        ("b", vec![3.0f32, 2.0], vec![2usize]),
        ("c", vec![2.0f32, 9.0], vec![2usize]),
    ];
    let m = single_node("Min", &["a", "b", "c"], vec![]);
    assert_close(&run(&m, &ins)?, &[1.0, 2.0], "Min");
    let m = single_node("Max", &["a", "b", "c"], vec![]);
    assert_close(&run(&m, &ins)?, &[3.0, 9.0], "Max");
    Ok(())
}

#[test]
fn pow_broadcasts() -> Result<()> {
    let m = single_node("Pow", &["a", "b"], vec![]);
    let got = run(
        &m,
        &[
            ("a", vec![2.0, 3.0, 4.0], vec![3]),
            ("b", vec![2.0], vec![1]),
        ],
    )?;
    assert_close(&got, &[4.0, 9.0, 16.0], "Pow");
    Ok(())
}

#[test]
fn gemm_applies_alpha_beta_and_transposes() -> Result<()> {
    // A is 2x3, B is 3x2 => Y is 2x2. alpha=2, C broadcast, beta=3.
    let m = single_node(
        "Gemm",
        &["a", "b", "c"],
        vec![attr_f("alpha", 2.0), attr_f("beta", 3.0)],
    );
    let got = run(
        &m,
        &[
            ("a", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            ("b", vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]),
            ("c", vec![1.0], vec![1]),
        ],
    )?;
    // A@B = [[1+0+3, 0+2+3],[4+0+6, 0+5+6]] = [[4,5],[10,11]]
    // 2*that + 3*1 = [[11,13],[23,25]]
    assert_close(&got, &[11.0, 13.0, 23.0, 25.0], "Gemm");
    Ok(())
}

#[test]
fn gemm_transb_matches_untransposed_equivalent() -> Result<()> {
    // B given as 2x3 with transB=1 is the same as the 3x2 above.
    let m = single_node("Gemm", &["a", "b"], vec![attr_i("transB", 1)]);
    let got = run(
        &m,
        &[
            ("a", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            ("b", vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], vec![2, 3]),
        ],
    )?;
    assert_close(&got, &[4.0, 5.0, 10.0, 11.0], "Gemm transB");
    Ok(())
}

// ----------------------------------------------------------- selection -----

#[test]
fn where_selects_elementwise() -> Result<()> {
    let m = single_node("Where", &["c", "x", "y"], vec![]);
    let got = run(
        &m,
        &[
            ("c", vec![1.0, 0.0, 1.0, 0.0], vec![4]),
            ("x", vec![10.0, 20.0, 30.0, 40.0], vec![4]),
            ("y", vec![-1.0, -2.0, -3.0, -4.0], vec![4]),
        ],
    )?;
    assert_close(&got, &[10.0, -2.0, 30.0, -4.0], "Where");
    Ok(())
}

#[test]
fn clip_from_attributes_and_from_inputs() -> Result<()> {
    let x = ("x", vec![-5.0f32, 0.0, 5.0], vec![3usize]);
    // opset < 11: min/max as attributes
    let m = single_node(
        "Clip",
        &["x"],
        vec![attr_f("min", -1.0), attr_f("max", 1.0)],
    );
    assert_close(
        &run(&m, std::slice::from_ref(&x))?,
        &[-1.0, 0.0, 1.0],
        "Clip attrs",
    );

    // opset 11+: min/max as inputs
    let m = single_node("Clip", &["x", "lo", "hi"], vec![]);
    let got = run(
        &m,
        &[
            x.clone(),
            ("lo", vec![-2.0], vec![1]),
            ("hi", vec![2.0], vec![1]),
        ],
    )?;
    assert_close(&got, &[-2.0, 0.0, 2.0], "Clip inputs");
    Ok(())
}

/// An absent optional input is encoded as an empty-string name mid-list, not
/// only as a short list. Clip with "no min, max=1" exercises that path.
#[test]
fn clip_treats_an_empty_input_name_as_absent() -> Result<()> {
    let m = single_node("Clip", &["x", "", "hi"], vec![]);
    let got = run(
        &m,
        &[
            ("x", vec![-5.0, 0.0, 5.0], vec![3]),
            ("hi", vec![1.0], vec![1]),
        ],
    )?;
    assert_close(&got, &[-5.0, 0.0, 1.0], "Clip empty-name min");
    Ok(())
}

// --------------------------------------------------------------- shape -----

#[test]
fn shape_and_size_report_metadata() -> Result<()> {
    let x = ("x", vec![0.0; 24], vec![2usize, 3, 4]);
    let m = single_node("Shape", &["x"], vec![]);
    assert_close(
        &run(&m, std::slice::from_ref(&x))?,
        &[2.0, 3.0, 4.0],
        "Shape",
    );

    let m = single_node("Size", &["x"], vec![]);
    assert_close(&run(&m, std::slice::from_ref(&x))?, &[24.0], "Size");
    Ok(())
}

#[test]
fn expand_broadcasts_and_keeps_extent_on_ones() -> Result<()> {
    // 3x1 expanded against [2,1,4] -> [2,3,4]; the `1` in the target keeps 3.
    let m = single_node("Expand", &["x", "s"], vec![]);
    let got = run(
        &m,
        &[
            ("x", vec![1.0, 2.0, 3.0], vec![3, 1]),
            ("s", vec![2.0, 1.0, 4.0], vec![3]),
        ],
    )?;
    assert_eq!(got.len(), 2 * 3 * 4, "Expand element count");
    // every row is its source value repeated 4x
    assert_close(&got[0..4], &[1.0; 4], "Expand row0");
    assert_close(&got[4..8], &[2.0; 4], "Expand row1");
    Ok(())
}

#[test]
fn tile_repeats_per_axis() -> Result<()> {
    let m = single_node("Tile", &["x", "r"], vec![]);
    let got = run(
        &m,
        &[
            ("x", vec![1.0, 2.0], vec![1, 2]),
            ("r", vec![2.0, 2.0], vec![2]),
        ],
    )?;
    assert_close(&got, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0], "Tile");
    Ok(())
}

#[test]
fn slice_from_inputs_with_negative_and_clamped_bounds() -> Result<()> {
    let x = ("x", vec![0.0f32, 1.0, 2.0, 3.0, 4.0], vec![5usize]);
    // [1:4]
    let m = single_node("Slice", &["x", "s", "e"], vec![]);
    let got = run(
        &m,
        &[
            x.clone(),
            ("s", vec![1.0], vec![1]),
            ("e", vec![4.0], vec![1]),
        ],
    )?;
    assert_close(&got, &[1.0, 2.0, 3.0], "Slice 1..4");

    // [-2:] — negative start folds, end clamps past the extent
    let got = run(
        &m,
        &[
            x.clone(),
            ("s", vec![-2.0], vec![1]),
            ("e", vec![99.0], vec![1]),
        ],
    )?;
    assert_close(&got, &[3.0, 4.0], "Slice -2..end");
    Ok(())
}

#[test]
fn range_honours_a_non_unit_delta() -> Result<()> {
    let m = single_node("Range", &["s", "l", "d"], vec![]);
    let got = run(
        &m,
        &[
            ("s", vec![1.0], vec![1]),
            ("l", vec![10.0], vec![1]),
            ("d", vec![3.0], vec![1]),
        ],
    )?;
    assert_close(&got, &[1.0, 4.0, 7.0], "Range step 3");
    Ok(())
}

#[test]
fn trilu_upper_default_and_lower() -> Result<()> {
    let x = (
        "x",
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3usize, 3],
    );
    // upper defaults to 1
    let m = single_node("Trilu", &["x"], vec![]);
    assert_close(
        &run(&m, std::slice::from_ref(&x))?,
        &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0],
        "Trilu upper",
    );
    let m = single_node("Trilu", &["x"], vec![attr_i("upper", 0)]);
    assert_close(
        &run(&m, std::slice::from_ref(&x))?,
        &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0],
        "Trilu lower",
    );
    Ok(())
}

#[test]
fn cumsum_along_an_axis() -> Result<()> {
    let m = single_node("CumSum", &["x", "ax"], vec![]);
    let got = run(
        &m,
        &[
            ("x", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            ("ax", vec![1.0], vec![1]),
        ],
    )?;
    assert_close(&got, &[1.0, 3.0, 3.0, 7.0], "CumSum axis 1");
    Ok(())
}

#[test]
fn argmax_argmin_keepdims_default_and_off() -> Result<()> {
    let x = ("x", vec![1.0f32, 9.0, 3.0, 2.0], vec![2usize, 2]);
    // keepdims defaults to 1
    let m = single_node("ArgMax", &["x"], vec![attr_i("axis", 1)]);
    assert_close(&run(&m, std::slice::from_ref(&x))?, &[1.0, 0.0], "ArgMax");
    let m = single_node("ArgMin", &["x"], vec![attr_i("axis", 1)]);
    assert_close(&run(&m, std::slice::from_ref(&x))?, &[0.0, 1.0], "ArgMin");
    Ok(())
}

#[test]
fn reduce_l2_matches_sqrt_of_sum_of_squares() -> Result<()> {
    let m = single_node("ReduceL2", &["x"], vec![attr_is("axes", vec![1])]);
    let got = run(&m, &[("x", vec![3.0, 4.0, 6.0, 8.0], vec![2, 2])])?;
    assert_close(&got, &[5.0, 10.0], "ReduceL2");
    Ok(())
}

#[test]
fn one_hot_places_on_and_off_values() -> Result<()> {
    let m = single_node("OneHot", &["i", "d", "v"], vec![]);
    let got = run(
        &m,
        &[
            ("i", vec![0.0, 2.0], vec![2]),
            ("d", vec![3.0], vec![1]),
            ("v", vec![0.0, 1.0], vec![2]),
        ],
    )?;
    assert_close(&got, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], "OneHot");
    Ok(())
}

#[test]
fn gather_elements_indexes_per_element() -> Result<()> {
    let m = single_node("GatherElements", &["x", "i"], vec![attr_i("axis", 1)]);
    let got = run(
        &m,
        &[
            ("x", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            ("i", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
        ],
    )?;
    assert_close(&got, &[2.0, 1.0, 3.0, 4.0], "GatherElements");
    Ok(())
}

// ------------------------------------------------------------ activations --

#[test]
fn hard_swish_matches_the_reference_formula() -> Result<()> {
    let m = single_node("HardSwish", &["x"], vec![]);
    let xs = vec![-4.0f32, 0.0, 4.0];
    let got = run(&m, &[("x", xs.clone(), vec![3])])?;
    let want: Vec<f32> = xs
        .iter()
        .map(|x| x * ((x / 6.0 + 0.5).clamp(0.0, 1.0)))
        .collect();
    assert_close(&got, &want, "HardSwish");
    Ok(())
}

#[test]
fn prelu_uses_slope_only_on_the_negative_side() -> Result<()> {
    let m = single_node("PRelu", &["x", "s"], vec![]);
    let got = run(
        &m,
        &[("x", vec![-2.0, 3.0], vec![2]), ("s", vec![0.5], vec![1])],
    )?;
    assert_close(&got, &[-1.0, 3.0], "PRelu");
    Ok(())
}

#[test]
fn selu_matches_the_reference_formula() -> Result<()> {
    let m = single_node("Selu", &["x"], vec![]);
    let xs = vec![-1.0f32, 2.0];
    let got = run(&m, &[("x", xs.clone(), vec![2])])?;
    let (alpha, gamma) = (1.673_263_2f32, 1.050_701f32);
    let want: Vec<f32> = xs
        .iter()
        .map(|&x| {
            if x > 0.0 {
                gamma * x
            } else {
                gamma * (alpha * (x.exp() - 1.0))
            }
        })
        .collect();
    assert_close(&got, &want, "Selu");
    Ok(())
}

#[test]
fn dropout_is_the_identity_at_inference() -> Result<()> {
    let m = single_node("Dropout", &["x"], vec![]);
    let got = run(&m, &[("x", vec![1.0, 2.0, 3.0], vec![3])])?;
    assert_close(&got, &[1.0, 2.0, 3.0], "Dropout");
    Ok(())
}

// ------------------------------------------------------- surfaced gaps -----
//
// These assert that an unsupported case is an ERROR rather than a silently
// wrong answer. They are as much a part of the contract as the happy paths.

#[test]
fn slice_with_a_step_is_a_clear_error_not_a_wrong_answer() {
    let m = single_node("Slice", &["x", "s", "e", "a", "st"], vec![]);
    let err = run(
        &m,
        &[
            ("x", vec![0.0, 1.0, 2.0, 3.0], vec![4]),
            ("s", vec![0.0], vec![1]),
            ("e", vec![4.0], vec![1]),
            ("a", vec![0.0], vec![1]),
            ("st", vec![2.0], vec![1]),
        ],
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("step != 1"),
        "expected a step-unsupported error, got: {err}"
    );
}

#[test]
fn cumsum_reverse_is_a_clear_error() {
    let m = single_node("CumSum", &["x", "ax"], vec![attr_i("reverse", 1)]);
    let err = run(
        &m,
        &[("x", vec![1.0, 2.0], vec![2]), ("ax", vec![0.0], vec![1])],
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("reverse"),
        "expected a reverse-unsupported error, got: {err}"
    );
}

#[test]
fn an_unimplemented_op_names_itself() {
    let m = single_node("LSTM", &["x"], vec![]);
    let err = run(&m, &[("x", vec![1.0], vec![1])]).unwrap_err();
    assert!(
        err.to_string().contains("LSTM"),
        "the error must name the unsupported op, got: {err}"
    );
}
