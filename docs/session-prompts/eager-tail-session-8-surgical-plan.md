# Session 8 — eager tail deletion: surgical plan (2026-06-12 audit)

Read-only audit of main @ `21b103bd`. Removes `BackpropOp` + the
eager autograd tape + `Var` from fuel-core, keeping `Tensor` as a
host-data container / node-handle wrapper. Companion to the
eager-tensor-retirement master plan (Phase F/H) and the
executor-unification re-audit (Session 8 line).

**Totals:** ~5,790 LOC removed (~5,100 source + ~650 tests + ~40
imports) out of 9,603 in the affected file set; ~3,800 LOC survive
(data containers, shape ops, lazy bridges). 14 commits, 2–3
sessions; commits 3–10 are mechanically independent and
parallelizable by feature area.

## 1. Keep vs delete — the API boundary

**KEPT (pure data-container, no BackpropOp):**
- Construction: `new(NdArray)` (tensor.rs:650), `from_slice` (807),
  `from_vec` (783), `from_iter` (692), `arange*` (713/729), `full`
  (668), `rand`/`randn` (516/603), `zeros`/`ones`/`*_like`
  (410/321/426/381/538/581), `eye` (4663), `triu2`/`tril2`
  (4647/4655), `from_storage` (855).
- Extraction: `to_vec0..3` (1475/3323/3356/3398), `to_scalar` (1446).
- Metadata: `shape`/`dims`/`dtype`/`device`/`rank`/`elem_count`/
  `dim`/`stride`/`layout`/`id`/`is_variable` (3447–3595).
- Shape views: `reshape` (4286), `unsqueeze` (4376), `squeeze`
  (4335), `transpose` (3916), `permute` (3949), `flatten*`
  (3769–3814), `broadcast_*` (4119–4168), `t` (3872).
- Copies: `copy` (4020), `contiguous` (4207), `force_contiguous`
  (4235), `to_device` (4083), `to_dtype` (4184), `detach` (4050).
- Indexing (no grad): `get` (3831), `get_on_dim` (3855),
  `strided_index` (3285), `strided_blocks` (3309).

**DELETED (tape-building compute):**
- Unary via `unary_op!` macro (tensor.rs:114-131): abs/neg/exp/log/
  sin/cos/sqrt/sqr/recip/tanh/relu/gelu/gelu_erf/sigmoid/silu +
  elu (1636), affine (1596), powf (1656), round_to (1430), cumsum
  (4674), flip (4786); upsample/interpolate (2273–2367); pooling
  (2473–2550).
- Binary via `binary_op!`/`binary_op_scalar!` (133–178); matmul
  family: matmul (2689), broadcast_matmul (2753), dot (2592), mv
  (2638), matvec (2669); scatter/gather (2881–3248); where_cond
  (2783); embedding (2821).
- Reductions (1898–2118, 3668–3724); comparisons + clamp
  (2135–2255).
- Grad-views: narrow (1747), chunk (1694), slice_scatter*
  (3029/3053/4694), pad_* (4442/4481), roll (1863), unfold (4799).
- Cat/stack/repeat/meshgrid (tensor_cat.rs:142/219; tensor.rs
  4417/1496/1550).
- In-place tape mutations: const_set (336), zero_set (351),
  one_set (366), scatter_set (2921), scatter_add_set (2997).
- Custom-op tensor methods `apply_op{1,2,3}[_no_bwd]`
  (custom_op.rs:209/238/268 region). KEEP the CustomOp1/2/3 traits.
- Eager `backward()` (backprop.rs:171) + `sorted_nodes` (36) +
  GradStore; `track_op` (tensor.rs:846).
- `Var` (variable.rs, entire 359-LOC module).

## 2. Per-file deletion map

| File | Strategy |
|---|---|
| tensor.rs (5101 LOC, 61 refs) | Keep data-container; delete all BackpropOp::new* compute methods + macros + `op: BackpropOp` field (tensor.rs:68, last) |
| op.rs (6 refs) | Delete BackpropOp struct+impl ONLY (op.rs:979-1034, 56 LOC). Op enum + variants STAY (graph schema) |
| custom_op.rs (7 refs) | Delete Tensor apply_op* methods (~185-280); KEEP CustomOp traits |
| conv.rs (5 refs) | Delete all eager conv methods (12-370); Op::Conv* variants stay |
| safetensors.rs (3 refs) | Delete BackpropOp::none() usage (const loads → lazy/from_storage pattern) |
| quantized/mod.rs (3 refs) | Same — QTensor is a const container |
| tensor_cat.rs (2 refs) | Delete eager cat/cat0/cat_contiguous; lazy cat exists |
| variable.rs (~28) | DELETE module + lib.rs re-exports |
| backprop.rs (~50) | DELETE module + lib.rs re-export |

## 3. Workspace consumers

- fuel_nn / transformers models / quantized_nn / 3 examples:
  already `_retired`-prefixed (Session 7 era; invisible to cargo
  discovery — verify `cargo build --workspace --examples` stays
  green).
- fuel-training: verify training_loop.rs for residual Var/backward
  refs; Trainer is lazy (train.rs ported in S5, commit 7d4e5e8c) —
  grep `GraphExecutor|BackpropOp|Var::` in train.rs first (expected
  clean).
- fuel-transformers: LogitsProcessor ported to `&[f32]`
  (34fb6190) — clean.
- fuel-pyo3 / fuel-onnx / fuel-wasm-* / fuel-inference: expected
  clean post-Phase-γ; verify by grep before commit 13.
- tensor-tools: pre-existing Device::Cpu break, unmaintained — out
  of scope, document.

## 4. Test triage

DELETE: fuel-core/tests/grad_tests.rs (~400 LOC, eager backward);
custom_op_tests.rs (~200, eager apply_op*); conv_tests.rs if purely
eager (verify); const_set/zero_set/one_set group in tensor_tests.rs
(~50). KEEP: remaining tensor_tests.rs (data-container), all
lazy.rs inline tests, phase_c_rotating_kv.

## 5. Commit sequence (smallest-risk order)

1. Delete BackpropOp struct+impl (op.rs:979-1034); `op:` field →
   stub (~60 LOC). NOTE: real order — land 3-10 first if commit 1
   alone won't compile; otherwise stub the field.
2. Delete variable.rs + re-exports (~360 LOC).
3. conv.rs eager methods (~350).
4. custom_op.rs tensor methods (~100).
5. tensor_cat.rs eager cat (~300).
6. tensor.rs unary/binary macros + generated methods (~800).
7. tensor.rs reductions (~300).
8. tensor.rs view/scatter/in-place-set (~500).
9. tensor.rs pooling/interpolation/roll/unfold (~400).
10. tensor.rs matmul family (~150).
11. Delete backprop.rs (~890).
12. Delete eager tests (~650).
13. lib.rs re-export cleanup + final grep gate:
    `BackpropOp|\.backward\(|Var::|const_set` → comments/fuel-graph/
    _retired/lazy-only.
14. (Optional) delete `op` field + `is_variable` field once grep
    shows zero readers (backprop.rs:54 `node.op()` and :48
    `is_variable()` are the last — gone after commit 11).

Commits 3–10 parallelize by area; 11–12 depend on 3–10.

## 6. Risks

1. `Tensor_.op` field read by backprop.rs:54 — field deletion only
   after commit 11; grep `\.op()` first.
2. `is_variable` dead after 11 — same protocol.
3. Semver: eager backward()/Var users break — Phase γ already made
   eager semi-internal; lazy backward (lazy.rs:2648) is the public
   story. Acceptable break, note in commit.
4. `_retired` trees: confirm cargo-invisible; their final drop is a
   SEPARATE follow-up after this session's audit gate.
5. train.rs: verified-by-comment lazy (S5); re-grep before start.
6. Doc-tests using eager backward (~3-5 expected): `cargo test
   --doc -p fuel-core` after commit 13; rewrite to lazy.

## CI gates per commit

`cargo check -p fuel-core` (+ touched crates) green per commit;
fuel-core --lib green after 11-12; `cargo test --doc -p fuel-core`
after 13; `cargo build --workspace --examples` after 12 (proves
_retired invisibility). Never workspace-wide check (tensor-tools).
