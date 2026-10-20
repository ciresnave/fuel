// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live-device probe for split-KV decode attention (ROADMAP item 17, row H12).
//!
//! The single-pass `flash_attn_f32` kernel launches one workgroup per
//! (b, h_q, q_i), so a decode step (sq = 1) with few heads leaves most of the
//! GPU idle and walks the whole key range serially. `flash_attn_f32_split_bytes`
//! cuts the key range into chunks that run as separate workgroups and merges
//! them in fixed order.
//!
//! * `split_matches_cpu_reference` checks both kernels against an f64 CPU
//!   reference (max abs error), including causal and ALiBi cases.
//! * `decode_latency_by_shape` prints per-call wall time (record + submit +
//!   wait + readback) for the unsplit kernel and several split counts. Variants
//!   are interleaved round-robin so a slow stretch of the (bimodal) GPU hits all
//!   of them alike. It asserts nothing about speed.
//!
//! ```sh
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -j 4 -p fuel-vulkan-backend \
//!     --release --test flash_attn_decode_latency -- --ignored --nocapture
//! ```

mod live_adapter;
use live_adapter::backend_or_skip;
use std::time::Instant;

fn bytes_of(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn floats_of(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 16_777_216.0) - 0.5
        })
        .collect()
}

/// f64 reference: softmax(q k^T * scale [+ alibi * (kj - qi)] [causal]) v.
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alibi: Option<&[f32]>,
    (hq, hkv, sq, sk, d): (usize, usize, usize, usize, usize),
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; hq * sq * d];
    for h in 0..hq {
        let kvh = h / (hq / hkv);
        for qi in 0..sq {
            let qrow = &q[(h * sq + qi) * d..][..d];
            let mut sc = vec![f64::NEG_INFINITY; sk];
            for (kj, slot) in sc.iter_mut().enumerate() {
                if causal && kj > qi {
                    continue;
                }
                let krow = &k[(kvh * sk + kj) * d..][..d];
                let dot: f64 = (0..d).map(|x| qrow[x] as f64 * krow[x] as f64).sum();
                let mut s = dot * scale as f64;
                if let Some(a) = alibi {
                    s += a[h] as f64 * (kj as f64 - qi as f64);
                }
                *slot = s;
            }
            let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if m == f64::NEG_INFINITY {
                continue;
            }
            let w: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
            let l: f64 = w.iter().sum();
            for x in 0..d {
                let acc: f64 = (0..sk)
                    .map(|kj| w[kj] * v[(kvh * sk + kj) * d + x] as f64)
                    .sum();
                out[(h * sq + qi) * d + x] = (acc / l) as f32;
            }
        }
    }
    out
}

#[test]
#[ignore]
fn split_matches_cpu_reference() {
    let Some(be) = backend_or_skip() else { return };
    // (hq, hkv, sq, sk, d, causal, alibi, splits)
    let cases = [
        (
            8usize, 8usize, 1usize, 4096usize, 128usize, false, false, 4usize,
        ),
        (32, 8, 1, 3000, 128, false, false, 8),
        (8, 4, 1, 1000, 64, false, true, 3),
        (4, 2, 16, 2048, 64, true, false, 2),
        (4, 2, 16, 2048, 64, true, true, 16),
        (8, 8, 1, 700, 128, false, false, 1),
    ];
    for (hq, hkv, sq, sk, d, causal, use_alibi, splits) in cases {
        let q = fill(hq * sq * d, 1);
        let k = fill(hkv * sk * d, 2);
        let v = fill(hkv * sk * d, 3);
        let al = fill(hq, 4).iter().map(|x| x * 0.02).collect::<Vec<_>>();
        let scale = 1.0 / (d as f32).sqrt();
        let want = reference(
            &q,
            &k,
            &v,
            use_alibi.then_some(&al[..]),
            (hq, hkv, sq, sk, d),
            scale,
            causal,
        );
        let qb = be.upload_bytes(&bytes_of(&q)).unwrap();
        let kb = be.upload_bytes(&bytes_of(&k)).unwrap();
        let vb = be.upload_bytes(&bytes_of(&v)).unwrap();
        let ab = be.upload_bytes(&bytes_of(&al)).unwrap();
        let alibi = use_alibi.then_some(&ab);
        let mut o1 = be.alloc_bytes(hq * sq * d * 4).unwrap();
        let mut o2 = be.alloc_bytes(hq * sq * d * 4).unwrap();
        be.flash_attn_f32_bytes(
            &qb, &kb, &vb, alibi, &mut o1, 1, hq, hkv, sq, sk, d, scale, causal,
        )
        .unwrap();
        be.flash_attn_f32_split_bytes(
            &qb, &kb, &vb, alibi, &mut o2, 1, hq, hkv, sq, sk, d, scale, causal, splits,
        )
        .unwrap();
        let g1 = floats_of(&be.download_bytes(&o1).unwrap());
        let g2 = floats_of(&be.download_bytes(&o2).unwrap());
        let err = |g: &[f32]| {
            g.iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        };
        let (e_unsplit, e_split) = (err(&g1), err(&g2));
        println!(
            "hq={hq} hkv={hkv} sq={sq} sk={sk} d={d} causal={causal} alibi={use_alibi} \
             splits={splits}: max|err| unsplit={e_unsplit:.2e} split={e_split:.2e}"
        );
        assert!(
            e_unsplit < 1e-4,
            "unsplit kernel is off its own reference: {e_unsplit}"
        );
        assert!(e_split < 1e-4, "split kernel off the reference: {e_split}");
    }
}

#[test]
#[ignore]
fn decode_latency_by_shape() {
    let Some(be) = backend_or_skip() else { return };
    let d = 128usize;
    let variants: [usize; 6] = [0, 2, 4, 8, 16, 32]; // 0 = unsplit kernel
    print!("{:>4} {:>4} {:>6}", "hq", "hkv", "sk");
    for s in variants {
        let name = if s == 0 {
            "unsplit".to_string()
        } else {
            format!("split{s}")
        };
        print!(" {name:>9}");
    }
    println!("   (median us per call incl. readback)");
    for &(hq, hkv) in &[(8usize, 8usize), (32, 8), (64, 8)] {
        for &sk in &[1024usize, 2048, 4096] {
            let q = be.upload_bytes(&bytes_of(&fill(hq * d, 1))).unwrap();
            let k = be.upload_bytes(&bytes_of(&fill(hkv * sk * d, 2))).unwrap();
            let v = be.upload_bytes(&bytes_of(&fill(hkv * sk * d, 3))).unwrap();
            let mut o = be.alloc_bytes(hq * d * 4).unwrap();
            let scale = 1.0 / (d as f32).sqrt();
            let mut run = |splits: usize| {
                let t0 = Instant::now();
                if splits == 0 {
                    be.flash_attn_f32_bytes(
                        &q, &k, &v, None, &mut o, 1, hq, hkv, 1, sk, d, scale, false,
                    )
                } else {
                    be.flash_attn_f32_split_bytes(
                        &q, &k, &v, None, &mut o, 1, hq, hkv, 1, sk, d, scale, false, splits,
                    )
                }
                .unwrap();
                be.download_bytes(&o).unwrap();
                t0.elapsed().as_secs_f64() * 1e6
            };
            let usable: Vec<usize> = variants
                .iter()
                .copied()
                .filter(|&s| s == 0 || sk.div_ceil(s) <= 1024)
                .collect();
            for &s in &usable {
                for _ in 0..5 {
                    run(s);
                }
            }
            let mut times: Vec<Vec<f64>> = vec![Vec::new(); usable.len()];
            for _ in 0..60 {
                for (i, &s) in usable.iter().enumerate() {
                    times[i].push(run(s));
                }
            }
            print!("{hq:>4} {hkv:>4} {sk:>6}");
            for s in variants {
                match usable.iter().position(|&u| u == s) {
                    Some(i) => {
                        let t = &mut times[i];
                        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        print!(" {:>9.1}", t[t.len() / 2]);
                    }
                    None => print!(" {:>9}", "-"),
                }
            }
            println!();
        }
    }
}
