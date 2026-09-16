// SPDX-License-Identifier: MIT OR Apache-2.0
//! GAP-315 FALSIFYING EXPERIMENT — the RECONCILIATION arm (A) and (B).
//!
//! ⚠️ PREDICTIONS WERE REGISTERED BEFORE THIS RAN:
//!
//!   (A) `layer_norm_affine`, `bias.len() != gain.len()`  -> PREDICT **NOT (d)**
//!       Reason: `hidden = gain.len()`, then the affine tensors are reshaped to
//!       a shape whose product is `hidden`. `reshape` RECONCILES element count
//!       against the actual slice length, so a wrong `bias.len()` cannot survive.
//!
//!   (B) `channel_affine_4d`, `gain.len() != channels`    -> PREDICT **NOT (d)**
//!       Reason: same structure, `.reshape(Shape::from_dims(&[1, channels, 1, 1]))?`.
//!
//! ⚠️ THESE TWO ARE THE REAL TEST OF THE PREDICATE. Reconciliation is the ONLY
//! reason to expect not-(d) here, so a MISS ON EITHER KILLS THE PREDICATE.
//! (The `stride_dims2` arm predicts (d) for an INDEPENDENT reason — `as usize`
//! wraps — so it would have been predictable without the predicate and carries
//! the least evidential weight of the three. The architect's point, recorded
//! here so a 3-for-3 is not read as three confirmations of one idea.)
//!
//! ⚠️ AND THE ASYMMETRY, STATED BEFORE THE RESULT: both real tests predict the
//! SAFE outcome. If both land, the predicate is shown to identify NON-(d); it is
//! NOT thereby shown to identify (d), because the only (d) call here is carried
//! by a different mechanism.
//!
//! Instrument, not a gate: asserts nothing.

use fuel_core::Device;
use fuel_core::lazy::Tensor;
use std::sync::Arc;

#[allow(
    dead_code,
    reason = "every field is read by the `{:?}` in the printed report; derive(Debug) \
does not count as a read for dead_code, and the report IS this instrument's output"
)]
#[derive(Debug)]
enum Outcome {
    BuildErr(String),
    Panic(String),
    OkRealized { dims: Vec<usize>, elems: usize },
    OkBuildRealizeErr { dims: Vec<usize> },
}

fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape.to_vec(), &Device::cpu()).unwrap()
}

fn probe<F>(f: F) -> Outcome
where
    F: FnOnce() -> std::result::Result<Tensor, fuel_ir::Error> + std::panic::UnwindSafe,
{
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(f);
    std::panic::set_hook(prev);
    match r {
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<non-string panic>".into());
            Outcome::Panic(msg.lines().next().unwrap_or("").to_string())
        }
        Ok(Err(e)) => Outcome::BuildErr(format!("{e}").lines().next().unwrap_or("").to_string()),
        Ok(Ok(t)) => {
            let dims = t.shape().dims().to_vec();
            let realized =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| t.realize_f32()));
            match realized {
                Ok(v) => Outcome::OkRealized {
                    dims,
                    elems: v.len(),
                },
                Err(_) => Outcome::OkBuildRealizeErr { dims },
            }
        }
    }
}

#[test]
#[ignore = "instrument, not a gate: asserts nothing; see the module doc"]
fn gap315_reconciliation_predicate_probe() {
    println!(
        "\n=== GAP-315 predicate experiment (A)+(B): reconciliation arm ===\n\
         debug_assertions = {}  ({})",
        cfg!(debug_assertions),
        if cfg!(debug_assertions) {
            "ARM A - guard PRESENT (positive control)"
        } else {
            "ARM B - guard ABSENT (what the shipped artifact does)"
        }
    );

    // ---- (A) layer_norm_affine: PREDICTED NOT (d) ----
    let g4: Arc<[f32]> = Arc::from(vec![1.0_f32; 4]);
    let b4: Arc<[f32]> = Arc::from(vec![0.0_f32; 4]);
    let b3: Arc<[f32]> = Arc::from(vec![0.0_f32; 3]); // WRONG: 3 != gain 4
    let b9: Arc<[f32]> = Arc::from(vec![0.0_f32; 9]); // WRONG the other way

    println!(
        "  [A-control] gain 4, bias 4 (VALID)   -> {:?}",
        probe({
            let (g, b) = (g4.clone(), b4.clone());
            move || cpu_f32(vec![1.0; 8], &[2, 4]).layer_norm_affine(g, b, 1e-5)
        })
    );
    println!(
        "  [A-short]   gain 4, bias 3           -> {:?}",
        probe({
            let (g, b) = (g4.clone(), b3.clone());
            move || cpu_f32(vec![1.0; 8], &[2, 4]).layer_norm_affine(g, b, 1e-5)
        })
    );
    println!(
        "  [A-long]    gain 4, bias 9           -> {:?}",
        probe({
            let (g, b) = (g4.clone(), b9.clone());
            move || cpu_f32(vec![1.0; 8], &[2, 4]).layer_norm_affine(g, b, 1e-5)
        })
    );

    // ---- (B) channel_affine_4d: PREDICTED NOT (d) ----
    // input [1, 3, 2, 2] -> channels = 3
    let g3: Arc<[f32]> = Arc::from(vec![1.0_f32; 3]);
    let bb3: Arc<[f32]> = Arc::from(vec![0.0_f32; 3]);
    let g2: Arc<[f32]> = Arc::from(vec![1.0_f32; 2]); // WRONG: 2 != channels 3
    let g8: Arc<[f32]> = Arc::from(vec![1.0_f32; 8]); // WRONG the other way

    println!(
        "  [B-control] gain 3, channels 3 (VALID) -> {:?}",
        probe({
            let (g, b) = (g3.clone(), bb3.clone());
            move || cpu_f32(vec![1.0; 12], &[1, 3, 2, 2]).channel_affine_4d(g, b)
        })
    );
    println!(
        "  [B-short]   gain 2, channels 3       -> {:?}",
        probe({
            let (g, b) = (g2.clone(), bb3.clone());
            move || cpu_f32(vec![1.0; 12], &[1, 3, 2, 2]).channel_affine_4d(g, b)
        })
    );
    println!(
        "  [B-long]    gain 8, channels 3       -> {:?}",
        probe({
            let (g, b) = (g8.clone(), bb3.clone());
            move || cpu_f32(vec![1.0; 12], &[1, 3, 2, 2]).channel_affine_4d(g, b)
        })
    );

    println!("  (measurement only - asserts nothing)");
}
