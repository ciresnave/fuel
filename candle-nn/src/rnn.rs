//! Recurrent Neural Network layers.
//!
//! This module provides [`LSTM`] (Long Short-Term Memory) and [`GRU`] (Gated Recurrent Unit)
//! layers, both implementing the [`RNN`] trait. Weights are loaded through a
//! [`VarBuilder`](crate::VarBuilder) and follow PyTorch's naming conventions
//! (`weight_ih_l0`, `weight_hh_l0`, etc.).
use candle::{Context, DType, Device, IndexOp, Result, Tensor};

/// Trait for Recurrent Neural Networks.
///
/// Provides a common interface for stepping through a sequence one element at a time
/// ([`step`](Self::step)) or processing an entire sequence ([`seq`](Self::seq),
/// [`seq_init`](Self::seq_init)). Both [`LSTM`] and [`GRU`] implement this trait.
///
/// # Example
///
/// ```no_run
/// use candle_nn::{lstm, LSTMConfig, RNN, VarBuilder};
///
/// // let lstm = lstm(input_dim, hidden_dim, LSTMConfig::default(), vb)?;
/// // let state = lstm.zero_state(batch)?;
/// // let states = lstm.seq(&input)?; // process a full sequence
/// ```
#[allow(clippy::upper_case_acronyms)]
pub trait RNN {
    type State: Clone;

    /// A zero state from which the recurrent network is usually initialized.
    fn zero_state(&self, batch_dim: usize) -> Result<Self::State>;

    /// Applies a single step of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, features].
    fn step(&self, input: &Tensor, state: &Self::State) -> Result<Self::State>;

    /// Applies multiple steps of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, seq_len, features].
    /// The initial state is the result of applying zero_state.
    fn seq(&self, input: &Tensor) -> Result<Vec<Self::State>> {
        let batch_dim = input.dim(0)?;
        let state = self.zero_state(batch_dim)?;
        self.seq_init(input, &state)
    }

    /// Applies multiple steps of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, seq_len, features].
    fn seq_init(&self, input: &Tensor, init_state: &Self::State) -> Result<Vec<Self::State>> {
        let (_b_size, seq_len, _features) = input.dims3()?;
        let mut output = Vec::with_capacity(seq_len);
        for seq_index in 0..seq_len {
            let input = input.i((.., seq_index, ..))?.contiguous()?;
            let state = if seq_index == 0 {
                self.step(&input, init_state)?
            } else {
                self.step(&input, &output[seq_index - 1])?
            };
            output.push(state);
        }
        Ok(output)
    }

    /// Converts a sequence of state to a tensor.
    fn states_to_tensor(&self, states: &[Self::State]) -> Result<Tensor>;
}

/// The state for an LSTM network, containing the hidden state `h` and cell state `c`.
///
/// The hidden state `h` is also the output of the LSTM at each time step.
///
/// # Example
///
/// ```rust
/// use candle::{Tensor, Device, DType};
/// use candle_nn::rnn::LSTMState;
///
/// let h = Tensor::zeros((1, 8), DType::F32, &Device::Cpu)?;
/// let c = Tensor::zeros((1, 8), DType::F32, &Device::Cpu)?;
/// let state = LSTMState::new(h, c);
/// assert_eq!(state.h().dims(), &[1, 8]);
/// # Ok::<(), candle::Error>(())
/// ```
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct LSTMState {
    pub h: Tensor,
    pub c: Tensor,
}

impl LSTMState {
    /// Creates a new LSTM state from the given hidden state `h` and cell state `c`.
    pub fn new(h: Tensor, c: Tensor) -> Self {
        LSTMState { h, c }
    }

    /// The hidden state vector, which is also the output of the LSTM.
    pub fn h(&self) -> &Tensor {
        &self.h
    }

    /// The cell state vector.
    pub fn c(&self) -> &Tensor {
        &self.c
    }
}

/// Direction of an RNN layer, used for bidirectional models.
///
/// # Example
///
/// ```rust
/// use candle_nn::rnn::Direction;
///
/// let d = Direction::Forward;
/// assert!(matches!(d, Direction::Forward));
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub enum Direction {
    /// Process the sequence from first to last.
    Forward,
    /// Process the sequence from last to first.
    Backward,
}

/// Configuration for an [`LSTM`] layer.
///
/// Controls weight initialization, bias presence, layer index (for multi-layer stacking),
/// and direction (forward or backward for bidirectional models). The defaults use Kaiming
/// uniform initialization with zero biases.
///
/// # Example
///
/// ```rust
/// use candle_nn::rnn::{LSTMConfig, Direction};
///
/// let cfg = LSTMConfig::default();
/// assert!(matches!(cfg.direction, Direction::Forward));
/// assert_eq!(cfg.layer_idx, 0);
/// # Ok::<(), candle::Error>(())
/// ```
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub struct LSTMConfig {
    pub w_ih_init: super::Init,
    pub w_hh_init: super::Init,
    pub b_ih_init: Option<super::Init>,
    pub b_hh_init: Option<super::Init>,
    pub layer_idx: usize,
    pub direction: Direction,
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: Some(super::Init::Const(0.)),
            b_hh_init: Some(super::Init::Const(0.)),
            layer_idx: 0,
            direction: Direction::Forward,
        }
    }
}

impl LSTMConfig {
    /// Returns a config with biases disabled.
    pub fn default_no_bias() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: None,
            b_hh_init: None,
            layer_idx: 0,
            direction: Direction::Forward,
        }
    }
}

/// A Long Short-Term Memory (LSTM) layer.
///
/// The LSTM fuses the input-to-hidden and hidden-to-hidden weight matrices into a single
/// combined matrix at construction time, so each [`step`](RNN::step) call performs only one
/// matrix multiplication instead of two.
///
/// Weights follow PyTorch naming: `weight_ih_l{idx}`, `weight_hh_l{idx}`, and optional
/// `bias_ih_l{idx}` / `bias_hh_l{idx}`.
///
/// <https://en.wikipedia.org/wiki/Long_short-term_memory>
///
/// # Example
///
/// ```no_run
/// use candle_nn::{lstm, LSTMConfig, RNN, VarBuilder};
///
/// // let lstm_layer = lstm(input_size, hidden_size, LSTMConfig::default(), vb)?;
/// // let state = lstm_layer.zero_state(1)?;
/// // let states = lstm_layer.seq(&input)?;
/// ```
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub struct LSTM {
    // Pre-computed fused weight matrix: cat([w_ih, w_hh], dim=1), shape [4*hidden, input_size+hidden_size]
    w_combined: Tensor,
    // Pre-computed fused bias: b_ih + b_hh, shape [4*hidden]
    b_combined: Option<Tensor>,
    hidden_dim: usize,
    config: LSTMConfig,
    device: Device,
    dtype: DType,
}

impl LSTM {
    /// Creates a LSTM layer.
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        config: LSTMConfig,
        vb: crate::VarBuilder,
    ) -> Result<Self> {
        let layer_idx = config.layer_idx;
        let direction_str = match config.direction {
            Direction::Forward => "",
            Direction::Backward => "_reverse",
        };
        let w_ih = vb.get_with_hints(
            (4 * hidden_dim, in_dim),
            &format!("weight_ih_l{layer_idx}{direction_str}"), // Only a single layer is supported.
            config.w_ih_init,
        )?;
        let w_hh = vb.get_with_hints(
            (4 * hidden_dim, hidden_dim),
            &format!("weight_hh_l{layer_idx}{direction_str}"), // Only a single layer is supported.
            config.w_hh_init,
        )?;
        let b_ih = match config.b_ih_init {
            Some(init) => Some(vb.get_with_hints(
                4 * hidden_dim,
                &format!("bias_ih_l{layer_idx}{direction_str}"),
                init,
            )?),
            None => None,
        };
        let b_hh = match config.b_hh_init {
            Some(init) => Some(vb.get_with_hints(
                4 * hidden_dim,
                &format!("bias_hh_l{layer_idx}{direction_str}"),
                init,
            )?),
            None => None,
        };
        // Pre-compute fused weight matrix: cat along the column dimension so that
        // [input, h] @ w_combined^T replaces two separate matmuls.
        // w_ih: [4*hidden, in_dim], w_hh: [4*hidden, hidden_dim]
        // w_combined: [4*hidden, in_dim + hidden_dim]
        let w_combined = Tensor::cat(&[&w_ih, &w_hh], 1)?;
        // Pre-sum biases so we only do one broadcast_add per step.
        let b_combined = match (&b_ih, &b_hh) {
            (Some(b_ih), Some(b_hh)) => Some((b_ih + b_hh)?),
            (Some(b_ih), None) => Some(b_ih.clone()),
            (None, Some(b_hh)) => Some(b_hh.clone()),
            (None, None) => None,
        };
        Ok(Self {
            w_combined,
            b_combined,
            hidden_dim,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Returns a reference to this LSTM layer's configuration.
    pub fn config(&self) -> &LSTMConfig {
        &self.config
    }
}

/// Creates an [`LSTM`] layer. This is a convenience wrapper around [`LSTM::new`].
///
/// # Example
///
/// ```no_run
/// use candle_nn::{lstm, LSTMConfig, VarBuilder};
///
/// // let layer = lstm(input_size, hidden_size, LSTMConfig::default(), vb)?;
/// ```
pub fn lstm(
    in_dim: usize,
    hidden_dim: usize,
    config: LSTMConfig,
    vb: crate::VarBuilder,
) -> Result<LSTM> {
    LSTM::new(in_dim, hidden_dim, config, vb)
}

impl RNN for LSTM {
    type State = LSTMState;

    fn zero_state(&self, batch_dim: usize) -> Result<Self::State> {
        let zeros =
            Tensor::zeros((batch_dim, self.hidden_dim), self.dtype, &self.device)?;
        Ok(Self::State {
            h: zeros.clone(),
            c: zeros.clone(),
        })
    }

    fn step(&self, input: &Tensor, in_state: &Self::State) -> Result<Self::State> {
        let input_shape = input.shape().clone();
        let in_dim = self.w_combined.dim(1).unwrap_or(0).saturating_sub(self.hidden_dim);
        let result: Result<Self::State> = (|| {
            // Fuse the two matmuls into one:
            // combined_input: [batch, input_size + hidden_size]
            // w_combined:     [4*hidden, input_size + hidden_size]
            // gates:          [batch, 4*hidden]
            let combined_input = Tensor::cat(&[input, &in_state.h], 1)?;
            let gates = combined_input.matmul(&self.w_combined.t()?)?;
            let gates = match &self.b_combined {
                None => gates,
                Some(b) => gates.broadcast_add(b)?,
            };
            let chunks = gates.chunk(4, 1)?;
            let in_gate = crate::ops::sigmoid(&chunks[0])?;
            let forget_gate = crate::ops::sigmoid(&chunks[1])?;
            let cell_gate = chunks[2].tanh()?;
            let out_gate = crate::ops::sigmoid(&chunks[3])?;

            let next_c = ((forget_gate * &in_state.c)? + (in_gate * cell_gate)?)?;
            let next_h = (out_gate * next_c.tanh()?)?;
            Ok(LSTMState {
                c: next_c,
                h: next_h,
            })
        })();
        result.with_context(|| {
            format!(
                "LSTM(in={in_dim}, hidden={}): input shape {input_shape:?}",
                self.hidden_dim
            )
        })
    }

    fn states_to_tensor(&self, states: &[Self::State]) -> Result<Tensor> {
        let states = states.iter().map(|s| s.h.clone()).collect::<Vec<_>>();
        Tensor::stack(&states, 1)
    }
}

/// The state for a GRU network, containing a single hidden state tensor `h`.
///
/// # Example
///
/// ```rust
/// use candle::{Tensor, Device, DType};
/// use candle_nn::rnn::GRUState;
///
/// let h = Tensor::zeros((1, 16), DType::F32, &Device::Cpu)?;
/// let state = GRUState { h };
/// assert_eq!(state.h().dims(), &[1, 16]);
/// # Ok::<(), candle::Error>(())
/// ```
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct GRUState {
    pub h: Tensor,
}

impl GRUState {
    /// The hidden state vector, which is also the output of the GRU.
    pub fn h(&self) -> &Tensor {
        &self.h
    }
}

/// Configuration for a [`GRU`] layer.
///
/// Controls weight initialization and bias presence. The defaults use Kaiming uniform
/// initialization with zero biases.
///
/// # Example
///
/// ```rust
/// use candle_nn::rnn::GRUConfig;
///
/// let cfg = GRUConfig::default();
/// assert!(cfg.b_ih_init.is_some()); // biases enabled by default
/// # Ok::<(), candle::Error>(())
/// ```
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub struct GRUConfig {
    pub w_ih_init: super::Init,
    pub w_hh_init: super::Init,
    pub b_ih_init: Option<super::Init>,
    pub b_hh_init: Option<super::Init>,
}

impl Default for GRUConfig {
    fn default() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: Some(super::Init::Const(0.)),
            b_hh_init: Some(super::Init::Const(0.)),
        }
    }
}

impl GRUConfig {
    /// Returns a config with biases disabled.
    pub fn default_no_bias() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: None,
            b_hh_init: None,
        }
    }
}

/// A Gated Recurrent Unit (GRU) layer.
///
/// The GRU is a simpler alternative to LSTM with only two gates (reset and update) and no
/// separate cell state. Weights follow PyTorch naming: `weight_ih_l0`, `weight_hh_l0`,
/// and optional `bias_ih_l0` / `bias_hh_l0`.
///
/// <https://en.wikipedia.org/wiki/Gated_recurrent_unit>
///
/// # Example
///
/// ```no_run
/// use candle_nn::{gru, GRUConfig, RNN, VarBuilder};
///
/// // let gru_layer = gru(input_size, hidden_size, GRUConfig::default(), vb)?;
/// // let states = gru_layer.seq(&input)?;
/// ```
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub struct GRU {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Option<Tensor>,
    b_hh: Option<Tensor>,
    hidden_dim: usize,
    config: GRUConfig,
    device: Device,
    dtype: DType,
}

impl GRU {
    /// Creates a GRU layer.
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        config: GRUConfig,
        vb: crate::VarBuilder,
    ) -> Result<Self> {
        let w_ih = vb.get_with_hints(
            (3 * hidden_dim, in_dim),
            "weight_ih_l0", // Only a single layer is supported.
            config.w_ih_init,
        )?;
        let w_hh = vb.get_with_hints(
            (3 * hidden_dim, hidden_dim),
            "weight_hh_l0", // Only a single layer is supported.
            config.w_hh_init,
        )?;
        let b_ih = match config.b_ih_init {
            Some(init) => Some(vb.get_with_hints(3 * hidden_dim, "bias_ih_l0", init)?),
            None => None,
        };
        let b_hh = match config.b_hh_init {
            Some(init) => Some(vb.get_with_hints(3 * hidden_dim, "bias_hh_l0", init)?),
            None => None,
        };
        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden_dim,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Returns a reference to this GRU layer's configuration.
    pub fn config(&self) -> &GRUConfig {
        &self.config
    }
}

/// Creates a [`GRU`] layer. This is a convenience wrapper around [`GRU::new`].
///
/// # Example
///
/// ```no_run
/// use candle_nn::{gru, GRUConfig, VarBuilder};
///
/// // let layer = gru(input_size, hidden_size, GRUConfig::default(), vb)?;
/// ```
pub fn gru(
    in_dim: usize,
    hidden_dim: usize,
    config: GRUConfig,
    vb: crate::VarBuilder,
) -> Result<GRU> {
    GRU::new(in_dim, hidden_dim, config, vb)
}

impl RNN for GRU {
    type State = GRUState;

    fn zero_state(&self, batch_dim: usize) -> Result<Self::State> {
        let h =
            Tensor::zeros((batch_dim, self.hidden_dim), self.dtype, &self.device)?;
        Ok(Self::State { h })
    }

    fn step(&self, input: &Tensor, in_state: &Self::State) -> Result<Self::State> {
        let input_shape = input.shape().clone();
        let in_dim = self.w_ih.dim(1).unwrap_or(0);
        let result: Result<Self::State> = (|| {
            let w_ih = input.matmul(&self.w_ih.t()?)?;
            let w_hh = in_state.h.matmul(&self.w_hh.t()?)?;
            let w_ih = match &self.b_ih {
                None => w_ih,
                Some(b_ih) => w_ih.broadcast_add(b_ih)?,
            };
            let w_hh = match &self.b_hh {
                None => w_hh,
                Some(b_hh) => w_hh.broadcast_add(b_hh)?,
            };
            let chunks_ih = w_ih.chunk(3, 1)?;
            let chunks_hh = w_hh.chunk(3, 1)?;
            let r_gate = crate::ops::sigmoid(&(&chunks_ih[0] + &chunks_hh[0])?)?;
            let z_gate = crate::ops::sigmoid(&(&chunks_ih[1] + &chunks_hh[1])?)?;
            let n_gate = (&chunks_ih[2] + (r_gate * &chunks_hh[2])?)?.tanh();

            let next_h = ((&z_gate * &in_state.h)? - ((&z_gate - 1.)? * n_gate)?)?;
            Ok(GRUState { h: next_h })
        })();
        result.with_context(|| {
            format!(
                "GRU(in={in_dim}, hidden={}): input shape {input_shape:?}",
                self.hidden_dim
            )
        })
    }

    fn states_to_tensor(&self, states: &[Self::State]) -> Result<Tensor> {
        let states = states.iter().map(|s| s.h.clone()).collect::<Vec<_>>();
        Tensor::cat(&states, 1)
    }
}
