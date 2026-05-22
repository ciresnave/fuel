# V.3.E.5 follow-up — Full f64 transcendental precision on Vulkan

Two open items left over from the V.3.E.5 polynomial-transcendentals
work (commits `8e2267b7`, `37acd43a`, `6e7e9531`). Both are
worthwhile but neither blocks anything else in V.3.

Current state: `unary_f64.slang` covers the 13-op unary surface on
double via fdlibm-style exp, 9-term Taylor sin/cos, two-term Cody-
Waite for ln(2), three-term Cody-Waite for 2π / π/2. Worst-case
observed precision (relative error vs `libm`, RTX 4070):

| op | range | worst err |
|---|---|---|
| exp | full natural | 1.7e-14 (~75 ULP, limited by Horner-without-FMA) |
| log | full natural | 1.6e-15 (~7 ULP) |
| sin / cos | `\|x\| <= 100` | 6.8e-15 / 7.7e-16 |
| sin / cos | `\|x\| <= 1000` | 4.2e-14 / 2.9e-13 (small \|cos\| amplifies rel err) |
| sin | `\|x\| <= 5e6` | 1.1e-10 |
| sin / cos | `\|x\| > 6.6e6` | breaks down (k * TWO_PI_1 product overflows) |
| tanh | full | 4.4e-16 |
| sigmoid / silu | full | <=4.5e-16 |
| gelu | full | 0 (exact composition match) |

## Item A — Double-double Horner for exp (~1 ULP target)

The exp error of ~75 ULP is dominated by accumulating roundoff
through 5 Horner steps + 1 division + 2^k scaling. Reaching true
1 ULP requires running the polynomial in **double-double precision**:
each intermediate is a pair `(hi, lo)` with `|lo| < ULP(hi) / 2`.

### Primitives needed (Slang)

```slang
// Knuth's TwoSum
void two_sum(double a, double b, out double s, out double e) {
    s = a + b;
    double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

// Fast2Sum: only valid when |a| >= |b|
void fast2_sum(double a, double b, out double s, out double e) {
    s = a + b;
    e = b - (s - a);
}

// Dekker's product (no FMA — Vulkan doesn't expose double FMA via
// GLSL.std.450)
void two_prod(double a, double b, out double p, out double e) {
    const double SPLIT = 134217729.0;  // 2^27 + 1
    double a_t = SPLIT * a;
    double a_hi = a_t - (a_t - a);
    double a_lo = a - a_hi;
    double b_t = SPLIT * b;
    double b_hi = b_t - (b_t - b);
    double b_lo = b - b_hi;
    p = a * b;
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
}
```

### DD-aware Horner step

```slang
// (s_hi, s_lo) = (s_hi, s_lo) * r + c   for one Horner round
void dd_step(inout double s_hi, inout double s_lo, double r, double c) {
    double p_hi, p_lo;
    two_prod(s_hi, r, p_hi, p_lo);     // exact product
    p_lo += s_lo * r;                  // cross term
    double t_hi, t_lo;
    two_sum(p_hi, c, t_hi, t_lo);
    t_lo += p_lo;
    fast2_sum(t_hi, t_lo, s_hi, s_lo);
}
```

Cost per Horner step: ~20 ops vs current 2 ops (10x). For exp's
5-step polynomial: 100 ops in the polynomial alone (vs current ~10).
Acceptable on the dispatch path — kernels are bandwidth-bound, not
compute-bound, for unary ops.

### Tests
- Extend `vulkan_dispatch_unary_exp_f64` to tighten tolerance to
  `5e-16` (~3 ULP). Should pass after DD upgrade.

Estimated effort: ~80 LOC + targeted tests. ~3 hours including
debugging the DD primitives in Slang.

## Item B — Payne-Hanek for sin/cos (`|x|` to 2^53)

Beyond `|x| ≈ 6.6e6`, the very first Cody-Waite product
`k * TWO_PI_1` exceeds f64's 52-bit mantissa and loses bits. Full
Payne-Hanek (PH) reduction handles `|x|` up to 2^53 by avoiding
single-precision multiplication entirely.

### How PH works

For `x` and `2/π` (= `0.636619...`), we want `k = round(x * 2/π)`
and the fractional part `f = x * 2/π - k`. Then `z = f * π/2` is
the reduced argument (with quadrant given by `k mod 4`).

PH stores `2/π` as a multi-word integer (e.g., 14 × 24-bit chunks
giving ~336 bits of precision). For a given x:

1. Extract x's exponent via `frexp` (or bit-cast). This tells us
   which window of the `2/π` table is relevant — for `x ≈ 2^e`,
   the bits of `2/π` at positions ~`e-53` through `e+1` matter.
2. Multiply x's mantissa by the relevant 3-4 words of `2/π` (each
   `24 × 53 → 77`-bit product, kept as DD pair).
3. Sum the partial products with care.
4. Take the integer part (this is `k`); keep the fractional part `f`.
5. `z = f * π/2` via another DD multiplication.

### Tests
- Add `vulkan_dispatch_unary_sin_f64_huge` with inputs like:
  - `1e10`, `1e12`, `1e15` — regular large inputs
  - `2.0_f64.powi(53) - 1.0` — near `2^53` (PH's design limit)
  - `2.0_f64.powi(53) / 3.0` — adversarial argument (known to
    expose reduction bugs in poorly-written PH)
  - `huge * std::f64::consts::PI` — multiples of π where sin
    should be near zero (catastrophic-cancellation test)

Estimated effort: ~250 LOC including the `2/π` table (which can be
generated from a high-precision representation, e.g. via Python's
`mpmath`), the multi-word multiplication, and the adversarial tests.
~6-10 hours.

## Reference

Both algorithms are well-documented in:
- **fdlibm** (`e_exp.c`, `k_rem_pio2.c`, `e_sin.c`)
- Markstein, *IA-64 and Elementary Functions* (Prentice Hall, 2000) —
  comprehensive treatment of DD primitives + range reduction strategies
- Muller et al., *Handbook of Floating-Point Arithmetic* (Birkhäuser,
  2018) — the modern reference

## Where to look in fuel

- Kernel source: `fuel-kernels-source/kernels/unary_f64.slang`
- Compiled SPV: `fuel-vulkan-kernels/spv/unary_f64.spv`
- Slang→SPV pipeline: `fuel-kernels-source/kernels/compile.sh`
- Tests: `fuel-storage/tests/vulkan_dispatch_live.rs` (the
  `vulkan_dispatch_unary_*_f64` family)
- Registration: `fuel-storage/src/vulkan_dispatch.rs`
  (`unary_f64::*` module + `register_vulkan_kernels`)

## Slang gotchas to remember

- **Decimal double literals lose precision** — `0.1L` parses as
  f32 then promotes. Always use hex floating-point: `0x1.999...p-4L`.
- **GLSL.std.450 transcendentals don't accept double** — emit will
  silently produce invalid SPIR-V (spirv-val flags it). Use pure
  arithmetic only for double-precision math (see comments in
  `unary_f64.slang`).
- **OpenCL.std doesn't work in NVIDIA Windows Vulkan compute** —
  even though SPV is spec-valid, the driver rejects at
  `vkCreateComputePipeline`. Don't try this route.
