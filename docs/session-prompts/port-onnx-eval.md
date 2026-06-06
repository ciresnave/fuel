# Port: fuel-onnx eval to lazy

## Eager source

- `fuel-onnx/src/eval.rs` (~800 LOC). Translates an ONNX graph
  to eager `Tensor` ops, holding a runtime symbol table mapping
  ONNX node names to materialized `Tensor`s.
- `fuel-onnx/src/lib.rs` — public surface (`SimpleEval`, etc.).

## Lazy module name

`fuel-onnx/src/lazy_eval.rs` (new file inside `fuel-onnx`, NOT
in fuel-core — preserves the crate boundary).

## Architecture summary

Walk the ONNX graph in topological order; for each node, look up
its op type and dispatch to the matching `LazyTensor` primitive,
inserting the resulting `LazyTensor` into the symbol table. At
the end, realize the requested output tensors.

Key ONNX ops to map:
- MatMul, Add, Sub, Mul, Div → LazyTensor binary
- Conv → LazyTensor::conv2d (rank-aware: 1D/2D/3D)
- ConvTranspose → LazyTensor::conv_transpose1d/2d
- BatchNormalization, LayerNormalization → norm primitives
- Relu, Gelu, Sigmoid, Tanh → activation primitives
- Reshape, Transpose, Squeeze, Unsqueeze → metadata-only ops
- Gather → index_select
- Softmax → softmax_last_dim or general softmax
- Concat, Split → concat / narrow
- Reduce* → mean / sum / max / min along dim
- Constant, ConstantOfShape → const_*_like
- Cast → astype
- Pad → pad_with_zeros (Constant only; Reflect / Replicate via
  the narrow+repeat+concat pattern from lazy_mimi_conv).

Surface:
- `pub struct LazyOnnxEval { graph: prost::Message, ... }`
- `pub fn run(&self, inputs: &HashMap<String, LazyTensor>) ->
  Result<HashMap<String, LazyTensor>>`

## Primitives needed

- All shipped. ONNX is just a different op-encoding for the same
  underlying primitives.

## Reusable modules

- Existing `fuel-onnx/src/eval.rs` for the topo walk + op-dispatch
  shape.
- `fuel-onnx/src/onnx.proto3` for the generated message types.
- `crate::lazy::LazyTensor` for the target representation.

## Open questions

- Initializer tensors in ONNX models: load as `const_f32_like`
  on the input graph anchor.
- Dynamic shapes: ONNX supports symbolic dimensions. For v1,
  require concrete shapes (error on symbolic-shape inputs).
- Quantized ONNX ops (QLinearMatMul, QuantizeLinear,
  DequantizeLinear): defer to a v2 sub-port; reject explicitly in
  v1 with a typed error.

## Splits

Recommended split:

1. Sub-port 1: Core arithmetic ops (MatMul, Add, Mul, ..., Reshape,
   Transpose, Reduce*) + initializer loading. Smallest viable
   subset.
2. Sub-port 2: Conv + ConvTranspose + Pad + Pooling.
3. Sub-port 3: BatchNorm + LayerNorm + activations + Softmax.
4. Sub-port 4: Quantized ops (if any consumer needs them).

## Test strategy

Per sub-port:
- ONNX golden test: serialize a tiny ONNX graph in-memory (one or
  two ops), feed known inputs, assert lazy eval output matches a
  hand-computed expected value.
- Real-model smoke: load a tiny ONNX model file (e.g. a 1-layer
  MNIST CNN) and verify forward output shape + finite values.

## References

- Eager source: `fuel-onnx/src/eval.rs`.
- ONNX spec: <https://onnx.ai/onnx/operators/>.
- LazyTensor primitive cheat sheet: `fuel-core/src/lazy.rs`.
