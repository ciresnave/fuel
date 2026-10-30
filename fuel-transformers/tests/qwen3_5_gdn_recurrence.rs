// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 115 step 1: the Gated DeltaNet recurrence against the HuggingFace reference.
//!
//! The vectors below were produced by `transformers 5.22.1`
//! `torch_recurrent_gated_delta_rule(.., use_qk_l2norm_in_kernel=True)` on seeded random f32
//! inputs (seed 20261030; `T=5`, then a 2-token continuation from the carried state; `nv=2`,
//! `dk=4`, `dv=3`). They are NOT produced by Fuel code.
//!
//! Two kinds of test: (1) the Fuel graph equals HF within a tolerance; (2) an independent
//! plain-Rust f64 reference with switchable defects, proving the vectors can tell the correct
//! recurrence from each wrong one (a missing decay, a missing L2 norm, a missing 1/sqrt(dk)), so
//! test (1) is not vacuous.

use fuel_ir::Shape;
use fuel_tensor::{Device, lazy::Tensor};
use fuel_transformers::models::lazy_qwen3_5_gdn::gated_delta_recurrence;

const T1: usize = 5;
const T2: usize = 2;
const NV: usize = 2;
const DK: usize = 4;
const DV: usize = 3;

const Q: [f32; 40] = [
    -1.6394445,
    -0.016313363,
    1.4623548,
    -0.74578553,
    1.1174381,
    1.3964626,
    -1.655074,
    0.66827816,
    0.86325294,
    0.6992989,
    -0.7299053,
    0.82289463,
    -1.142299,
    0.06767907,
    -0.44473934,
    -0.27340484,
    0.11987989,
    -0.94780725,
    -0.3279159,
    0.83761626,
    -1.8646075,
    1.803322,
    0.60644746,
    -0.3983483,
    -2.6973872,
    -0.15673871,
    0.239714,
    0.008164821,
    -0.66953456,
    -1.115636,
    1.4222956,
    -0.80084574,
    -0.35040966,
    -0.12073507,
    0.15783697,
    0.44979295,
    -0.5419527,
    -0.54787827,
    -0.51238966,
    0.0579031,
];
const K: [f32; 40] = [
    0.9345379,
    0.8603531,
    0.37964818,
    -0.68472195,
    -0.14710061,
    -0.52145135,
    1.1757443,
    -0.17041208,
    0.3731766,
    -0.4860111,
    0.7572428,
    1.86807,
    -0.6858818,
    -0.47848228,
    0.7557285,
    0.5293456,
    1.0610049,
    -0.38412157,
    -0.970854,
    0.9343291,
    1.2139685,
    -0.7221065,
    1.5996426,
    0.69754195,
    -0.54444355,
    -1.4824419,
    0.53115636,
    2.7606363,
    1.3685031,
    -0.12738013,
    0.781983,
    0.32285267,
    0.48552635,
    0.044227835,
    0.84347445,
    -1.353029,
    2.3728027,
    -1.8673592,
    -0.33085617,
    1.055443,
];
const V: [f32; 30] = [
    0.6590645,
    0.17785037,
    0.56532127,
    2.4827766,
    1.148929,
    0.16125944,
    3.149555,
    0.7527141,
    1.2427896,
    -1.1598794,
    0.3885989,
    1.1304204,
    0.9002782,
    -0.6434824,
    -1.4165632,
    1.0720285,
    -1.9244157,
    0.4526002,
    0.6746397,
    1.1969528,
    -0.3649201,
    1.5359906,
    0.32722408,
    1.7771283,
    -1.6774751,
    -1.5317832,
    -0.69863063,
    0.391871,
    1.5856043,
    0.2963992,
];
const G: [f32; 10] = [
    -0.8995108,
    -0.78139776,
    -1.4611971,
    -1.0935931,
    -0.9812899,
    -0.22111952,
    -0.52288127,
    -0.33212364,
    -0.8675736,
    -0.23950571,
];
const BETA: [f32; 10] = [
    0.69627, 0.5451562, 0.9847567, 0.2640409, 0.3820123, 0.61843497, 0.9672857, 0.2995001,
    0.10964894, 0.16686553,
];
const OUT: [f32; 30] = [
    -0.031834003,
    -0.008590493,
    -0.027306037,
    -0.6055864,
    -0.28024098,
    -0.039333593,
    0.47333854,
    0.11368351,
    0.1951807,
    -0.08114982,
    -0.016472457,
    0.022197729,
    0.42586195,
    -0.011763649,
    -0.0702303,
    -0.1648261,
    0.30997542,
    -0.027101595,
    -0.07736739,
    0.15623693,
    0.04393281,
    0.1286907,
    -0.09588448,
    0.09475373,
    0.157131,
    0.26636326,
    -0.01033946,
    -0.21294744,
    0.080383874,
    -0.15204519,
];
const STATE: [f32; 24] = [
    0.030379068,
    -0.14559793,
    -0.06197281,
    -0.081113055,
    -0.21058579,
    0.09618537,
    0.0053438693,
    0.07213648,
    0.050222088,
    0.4218924,
    0.51987886,
    -0.06784465,
    0.46449378,
    0.033007085,
    0.31559628,
    -0.14000303,
    -0.017754689,
    -0.1227091,
    0.43486276,
    -0.31477734,
    0.3666794,
    0.050774477,
    -0.07006439,
    0.16796245,
];
const Q2: [f32; 16] = [
    -0.7041543,
    -0.30621916,
    0.56074804,
    1.1551754,
    0.73619443,
    0.024862507,
    0.2920277,
    -0.34031913,
    1.0182482,
    0.6997738,
    1.3756733,
    -0.70988446,
    0.5794753,
    -1.7231885,
    -1.1936045,
    -2.4675624,
];
const K2: [f32; 16] = [
    -2.025611,
    -0.81803364,
    1.1185243,
    -0.35036138,
    0.84083015,
    -0.5553807,
    0.7224972,
    0.8271235,
    0.028763441,
    -0.7936317,
    -0.6574074,
    -0.71516937,
    0.032321487,
    0.44491988,
    -1.7036772,
    1.1239934,
];
const V2: [f32; 12] = [
    -0.99097955,
    1.6364063,
    1.1056025,
    -0.5827588,
    -0.7786963,
    -0.6727261,
    -2.377122,
    -0.71283597,
    0.5195914,
    0.14471939,
    1.550288,
    0.94404185,
];
const G2: [f32; 4] = [-0.28300524, -0.9709952, -1.4796275, -0.8209768];
const BETA2: [f32; 4] = [0.4171368, 0.2790277, 0.33503795, 0.6203925];
const OUT2: [f32; 12] = [
    0.02295988,
    0.36636722,
    0.104890235,
    0.05242929,
    -0.05124941,
    0.011389708,
    0.13963713,
    0.005469445,
    -0.03612382,
    -0.0014615557,
    -0.0724931,
    -0.060676806,
];
const STATE2: [f32; 24] = [
    0.060912635,
    -0.14860745,
    -0.08942322,
    0.51198155,
    0.060740292,
    -0.12813273,
    0.3712132,
    0.19759485,
    -0.037756033,
    0.53223073,
    0.1991101,
    -0.12616491,
    0.024320915,
    -0.029678833,
    0.002278631,
    0.037587013,
    0.22791362,
    0.1463501,
    -0.06916415,
    -0.8537186,
    -0.47811922,
    0.016815443,
    0.44009107,
    0.29136896,
];
// shapes: q,k [T,nv,dk]; v,out [T,nv,dv]; g,beta [T,nv]; state [nv,dk,dv]; T=5 then T2=2; nv=2 dk=4 dv=3

#[derive(Clone, Copy, PartialEq)]
enum Defect {
    None,
    NoDecay,
    NoL2Norm,
    NoQScale,
}

fn l2(x: &[f64], eps: f64) -> Vec<f64> {
    let n = (x.iter().map(|v| v * v).sum::<f64>() + eps).sqrt();
    x.iter().map(|v| v / n).collect()
}

/// Plain f64 recurrence; tensors are indexed `[t][h]` as `t * NV + h`. Returns (out, final state).
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    t_len: usize,
    state0: Option<&[f32]>,
    defect: Defect,
) -> (Vec<f64>, Vec<f64>) {
    let mut s: Vec<f64> = match state0 {
        Some(s) => s.iter().map(|&x| x as f64).collect(),
        None => vec![0.0; NV * DK * DV],
    };
    let mut out = vec![0.0; t_len * NV * DV];
    for t in 0..t_len {
        for h in 0..NV {
            let qi = (t * NV + h) * DK;
            let vi = (t * NV + h) * DV;
            let mut qv: Vec<f64> = q[qi..qi + DK].iter().map(|&x| x as f64).collect();
            let mut kv: Vec<f64> = k[qi..qi + DK].iter().map(|&x| x as f64).collect();
            if defect != Defect::NoL2Norm {
                qv = l2(&qv, 1e-6);
                kv = l2(&kv, 1e-6);
            }
            if defect != Defect::NoQScale {
                for x in qv.iter_mut() {
                    *x /= (DK as f64).sqrt();
                }
            }
            let sh = &mut s[h * DK * DV..(h + 1) * DK * DV];
            let decay = if defect == Defect::NoDecay {
                1.0
            } else {
                (g[t * NV + h] as f64).exp()
            };
            for x in sh.iter_mut() {
                *x *= decay;
            }
            let b = beta[t * NV + h] as f64;
            let mut delta = [0.0_f64; DV];
            for (j, d) in delta.iter_mut().enumerate() {
                let mem: f64 = (0..DK).map(|i| sh[i * DV + j] * kv[i]).sum();
                *d = (v[vi + j] as f64 - mem) * b;
            }
            for i in 0..DK {
                for (j, d) in delta.iter().enumerate() {
                    sh[i * DV + j] += kv[i] * d;
                }
            }
            for j in 0..DV {
                out[vi + j] = (0..DK).map(|i| sh[i * DV + j] * qv[i]).sum();
            }
        }
    }
    (out, s)
}

fn max_abs(a: &[f64], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - *y as f64).abs())
        .fold(0.0, f64::max)
}

fn max_abs_f32(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

/// f32 graph vs the HF f32 vectors. The vectors print 9 significant digits and the recurrence is a
/// handful of steps over a 4x3 state, so 1e-5 sits above rounding; `the_vectors_can_tell_each_defect_apart`
/// measures how far below any defect it is.
const TOL: f64 = 1e-5;

#[test]
fn the_f64_reference_matches_huggingface() {
    let (o, s) = reference(&Q, &K, &V, &G, &BETA, T1, None, Defect::None);
    assert!(max_abs(&o, &OUT) < TOL, "out {}", max_abs(&o, &OUT));
    assert!(max_abs(&s, &STATE) < TOL, "state {}", max_abs(&s, &STATE));
    let (o2, s2) = reference(&Q2, &K2, &V2, &G2, &BETA2, T2, Some(&STATE), Defect::None);
    assert!(max_abs(&o2, &OUT2) < TOL, "out2 {}", max_abs(&o2, &OUT2));
    assert!(
        max_abs(&s2, &STATE2) < TOL,
        "state2 {}",
        max_abs(&s2, &STATE2)
    );
}

#[test]
fn the_vectors_can_tell_each_defect_apart() {
    for (name, d) in [
        ("no decay", Defect::NoDecay),
        ("no l2norm", Defect::NoL2Norm),
        ("no q scale", Defect::NoQScale),
    ] {
        let (o, _) = reference(&Q, &K, &V, &G, &BETA, T1, None, d);
        let err = max_abs(&o, &OUT);
        assert!(
            err > 100.0 * TOL,
            "defect `{name}` is only {err:.2e} off: the vectors cannot see it"
        );
    }
}

fn run_fuel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    t_len: usize,
    state0: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let dev = Device::cpu();
    let qt = Tensor::from_f32(q.to_vec(), Shape::from_dims(&[t_len, NV, DK]), &dev).unwrap();
    let kt = qt
        .const_f32_like(k.to_vec(), Shape::from_dims(&[t_len, NV, DK]))
        .unwrap();
    let vt = qt
        .const_f32_like(v.to_vec(), Shape::from_dims(&[t_len, NV, DV]))
        .unwrap();
    let gt = qt
        .const_f32_like(g.to_vec(), Shape::from_dims(&[t_len, NV]))
        .unwrap();
    let bt = qt
        .const_f32_like(beta.to_vec(), Shape::from_dims(&[t_len, NV]))
        .unwrap();
    let st = state0.map(|s| {
        qt.const_f32_like(s.to_vec(), Shape::from_dims(&[NV, DK, DV]))
            .unwrap()
    });
    let (out, state) = gated_delta_recurrence(&qt, &kt, &vt, &gt, &bt, st.as_ref()).unwrap();
    let r = fuel_tensor::lazy::realize_many_f32(&[&out, &state]);
    (r[0].clone(), r[1].clone())
}

#[test]
fn the_fuel_recurrence_matches_huggingface_from_a_zero_state() {
    let (o, s) = run_fuel(&Q, &K, &V, &G, &BETA, T1, None);
    assert!(max_abs_f32(&o, &OUT) < TOL, "out {}", max_abs_f32(&o, &OUT));
    assert!(
        max_abs_f32(&s, &STATE) < TOL,
        "state {}",
        max_abs_f32(&s, &STATE)
    );
}

#[test]
fn the_fuel_recurrence_continues_from_a_carried_state() {
    let (o, s) = run_fuel(&Q2, &K2, &V2, &G2, &BETA2, T2, Some(&STATE));
    assert!(
        max_abs_f32(&o, &OUT2) < TOL,
        "out2 {}",
        max_abs_f32(&o, &OUT2)
    );
    assert!(
        max_abs_f32(&s, &STATE2) < TOL,
        "state2 {}",
        max_abs_f32(&s, &STATE2)
    );
}
