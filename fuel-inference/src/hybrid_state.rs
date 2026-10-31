// SPDX-License-Identifier: MIT OR Apache-2.0
//! The per-session fixed-size state of a hybrid model (ROADMAP row 117, design note
//! `docs/design-notes/2026-10-30-hybrid-decode-state-row-117.md` section 4, the functional option).
//!
//! A [`FixedState`] holds, for every layer of a [`HybridGeometry`] that keeps fixed-size state, one
//! host buffer per [`FixedSlot`]. A decode step takes the buffers as input and returns the next
//! ones; [`FixedState::replace`] swaps them in as one unit, so a step that fails (or returns the
//! wrong sizes) leaves the previous state intact and never a half-updated one. The buffers do not
//! grow: [`FixedState::bytes`] is the same after every step.
//!
//! Only f32 slots are supported; a recurrent state in another dtype is refused at construction
//! rather than silently stored as f32.

use crate::decode_state_spec::{FixedSlot, HybridGeometry, HybridLayer};
use fuel_ir::{DType, Error, Result};

/// One layer's buffers, one per fixed slot (empty for an attention layer).
pub type LayerBuffers = Vec<Vec<f32>>;

/// The fixed-size state of one session.
#[derive(Debug, Clone)]
pub struct FixedState {
    /// Element count each buffer must have, per layer per slot.
    sizes: Vec<Vec<usize>>,
    buffers: Vec<LayerBuffers>,
}

impl FixedState {
    /// Zeroed state for a new sequence. Refuses a slot whose dtype is not f32.
    pub fn zeroed(geometry: &HybridGeometry) -> Result<Self> {
        let mut sizes = Vec::with_capacity(geometry.layers().len());
        for (i, l) in geometry.layers().iter().enumerate() {
            sizes.push(match l {
                HybridLayer::Attention => Vec::new(),
                HybridLayer::Fixed(slots) => slots
                    .iter()
                    .map(|s| slot_elems(i, s))
                    .collect::<Result<_>>()?,
            });
        }
        let buffers = sizes
            .iter()
            .map(|l| l.iter().map(|n| vec![0.0_f32; *n]).collect())
            .collect();
        Ok(Self { sizes, buffers })
    }

    /// Layer count (attention layers included, with no buffers).
    pub fn n_layers(&self) -> usize {
        self.buffers.len()
    }

    /// The current buffers of layer `i`.
    pub fn layer(&self, i: usize) -> Option<&[Vec<f32>]> {
        self.buffers.get(i).map(Vec::as_slice)
    }

    /// Host bytes held. Constant for the life of the session.
    pub fn bytes(&self) -> usize {
        self.sizes.iter().flatten().sum::<usize>() * 4
    }

    /// Swap in the buffers a step produced, one entry per layer. All-or-nothing: on any mismatch
    /// of layer count, slot count or element count the state is unchanged and the error names the
    /// first offender.
    pub fn replace(&mut self, next: Vec<LayerBuffers>) -> Result<()> {
        if next.len() != self.sizes.len() {
            return Err(Error::Msg(format!(
                "FixedState::replace: {} layers given, the geometry has {}",
                next.len(),
                self.sizes.len()
            ))
            .bt());
        }
        for (i, (want, got)) in self.sizes.iter().zip(&next).enumerate() {
            let lens: Vec<usize> = got.iter().map(Vec::len).collect();
            if &lens != want {
                return Err(Error::Msg(format!(
                    "FixedState::replace: layer {i} gave buffer sizes {lens:?}, expected {want:?}"
                ))
                .bt());
            }
        }
        self.buffers = next;
        Ok(())
    }
}

fn slot_elems(layer: usize, s: &FixedSlot) -> Result<usize> {
    if s.dtype != DType::F32 {
        return Err(Error::Msg(format!(
            "FixedState: layer {layer} has a {:?} fixed slot; only f32 is supported",
            s.dtype
        ))
        .bt());
    }
    Ok(s.shape.iter().product())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_state_spec::LayerStateSpec;

    fn rec(n: usize) -> LayerStateSpec {
        LayerStateSpec::Fixed(vec![FixedSlot::new(vec![n], DType::F32)])
    }
    fn attn() -> LayerStateSpec {
        LayerStateSpec::KeyValue {
            n_kv_heads: 1,
            head_dim: 1,
        }
    }
    fn geom() -> HybridGeometry {
        HybridGeometry::from_specs(&[rec(3), rec(3), attn()]).unwrap()
    }

    /// The toy hybrid: two recurrent layers `s <- 0.5 s + x` (per channel) feeding one attention
    /// layer that sums every past layer-1 output (its "KV" is the history). One token at a time with
    /// a carried state and a growing history must equal the same maths over the whole sequence.
    struct Toy {
        kv: Vec<f32>,
        state: FixedState,
    }

    impl Toy {
        fn new() -> Self {
            Self {
                kv: Vec::new(),
                state: FixedState::zeroed(&geom()).unwrap(),
            }
        }

        fn step(&mut self, x: [f32; 3]) -> Result<f32> {
            let mut next: Vec<LayerBuffers> = vec![Vec::new(); 3];
            let mut cur = x;
            for (l, out) in next.iter_mut().enumerate().take(2) {
                let prev = self.state.layer(l).unwrap();
                let s: Vec<f32> = (0..3).map(|c| 0.5 * prev[0][c] + cur[c]).collect();
                cur = [s[0], s[1], s[2]];
                *out = vec![s];
            }
            self.state.replace(next)?;
            self.kv.push(cur.iter().sum());
            Ok(self.kv.iter().sum())
        }
    }

    fn whole(xs: &[[f32; 3]]) -> Vec<f32> {
        let mut states = [[0.0_f32; 3]; 2];
        let mut hist = 0.0;
        xs.iter()
            .map(|x| {
                let mut cur = *x;
                for s in states.iter_mut() {
                    for c in 0..3 {
                        s[c] = 0.5 * s[c] + cur[c];
                        cur[c] = s[c];
                    }
                }
                hist += cur.iter().sum::<f32>();
                hist
            })
            .collect()
    }

    fn inputs() -> Vec<[f32; 3]> {
        (0..6)
            .map(|t| [t as f32 + 1.0, 0.5 * t as f32, -1.0 + t as f32])
            .collect()
    }

    #[test]
    fn per_token_decode_equals_the_whole_sequence_at_every_position() {
        let xs = inputs();
        let want = whole(&xs);
        let mut toy = Toy::new();
        let got: Vec<f32> = xs.iter().map(|x| toy.step(*x).unwrap()).collect();
        assert_eq!(got, want);
        // Control: a fresh state for the last token alone does not reproduce its position.
        let mut fresh = Toy::new();
        assert_ne!(fresh.step(xs[5]).unwrap(), want[5]);
    }

    #[test]
    fn state_bytes_do_not_grow_with_tokens() {
        let mut toy = Toy::new();
        let before = toy.state.bytes();
        for x in inputs() {
            toy.step(x).unwrap();
        }
        assert_eq!(toy.state.bytes(), before);
        assert_eq!(before, 2 * 3 * 4);
        assert_eq!(toy.kv.len(), 6, "only the attention history grows");
    }

    #[test]
    fn a_bad_step_leaves_the_state_untouched() {
        let mut st = FixedState::zeroed(&geom()).unwrap();
        let good: Vec<LayerBuffers> = vec![vec![vec![1.0; 3]], vec![vec![2.0; 3]], Vec::new()];
        st.replace(good).unwrap();
        for bad in [
            vec![vec![vec![9.0; 3]]],
            vec![vec![vec![9.0; 2]], vec![vec![9.0; 3]], Vec::new()],
            vec![vec![vec![9.0; 3]], vec![vec![9.0; 3]], vec![vec![9.0; 3]]],
        ] {
            assert!(st.replace(bad).is_err());
            assert_eq!(st.layer(0).unwrap()[0], vec![1.0; 3]);
            assert_eq!(st.layer(1).unwrap()[0], vec![2.0; 3]);
        }
    }

    #[test]
    fn a_non_f32_slot_is_refused() {
        let g = HybridGeometry::from_specs(&[
            LayerStateSpec::Fixed(vec![FixedSlot::new(vec![4], DType::BF16)]),
            attn(),
        ])
        .unwrap();
        let e = FixedState::zeroed(&g).unwrap_err().to_string();
        assert!(e.contains("only f32"), "{e}");
    }
}
