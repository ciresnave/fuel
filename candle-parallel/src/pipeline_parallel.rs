//! Pipeline parallelism: stage assignment and micro-batch scheduling.
//!
//! Pipeline parallelism (PP) partitions a model into consecutive *stages*,
//! each assigned to a different device. Data flows through stages sequentially,
//! like an assembly line. To maximise device utilisation, a mini-batch is split
//! into *micro-batches* that overlap: while stage 2 processes micro-batch 1,
//! stage 1 is already working on micro-batch 2.
//!
//! ## Schedules
//!
//! - **GPipe**: all-forward then all-backward. Simple but high bubble time.
//! - **1F1B** (One-Forward-One-Backward): interleaves forward and backward
//!   passes to reduce the pipeline bubble.
//!
//! This module provides the *scheduler* — it computes which micro-batch each
//! stage should process at each time step. Actual tensor transfer between
//! stages is the caller's responsibility.
//!
//! # Example
//!
//! ```rust
//! use candle_parallel::pipeline_parallel::{
//!     PipelineConfig, Schedule, ScheduleKind, StageOp,
//! };
//!
//! let config = PipelineConfig::new(4, 8); // 4 stages, 8 micro-batches
//! let schedule = config.build_schedule(ScheduleKind::OneForwardOneBackward);
//!
//! assert_eq!(schedule.num_stages(), 4);
//! assert_eq!(schedule.num_microbatches(), 8);
//!
//! // Stage 0 starts with forward of micro-batch 0
//! let first = &schedule.steps(0)[0];
//! assert_eq!(first.op, StageOp::Forward);
//! assert_eq!(first.microbatch, 0);
//! ```

use serde::{Deserialize, Serialize};

/// An operation a stage performs at one time step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StageOp {
    /// Forward pass on a micro-batch.
    Forward,
    /// Backward pass on a micro-batch.
    Backward,
    /// Idle (bubble).
    Idle,
}

/// A single scheduled step: what a stage does at one time slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduleStep {
    /// Stage index.
    pub stage: usize,
    /// Time step (0-indexed).
    pub time: usize,
    /// Operation.
    pub op: StageOp,
    /// Micro-batch index (meaningless if `op == Idle`).
    pub microbatch: usize,
}

/// Kind of pipeline schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScheduleKind {
    /// GPipe: all forwards, then all backwards.
    GPipe,
    /// 1F1B: interleaved forward-backward to reduce bubble.
    OneForwardOneBackward,
}

/// Pipeline schedule: a per-stage sequence of operations over time.
#[derive(Debug, Clone)]
pub struct Schedule {
    num_stages: usize,
    num_microbatches: usize,
    /// Steps indexed by `[stage][time]`.
    per_stage: Vec<Vec<ScheduleStep>>,
}

impl Schedule {
    /// Number of pipeline stages.
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// Number of micro-batches.
    pub fn num_microbatches(&self) -> usize {
        self.num_microbatches
    }

    /// Steps for a given stage in time order.
    pub fn steps(&self, stage: usize) -> &[ScheduleStep] {
        &self.per_stage[stage]
    }

    /// Total number of time steps.
    pub fn total_steps(&self) -> usize {
        self.per_stage.first().map_or(0, |s| s.len())
    }

    /// Count the number of idle (bubble) steps across all stages.
    pub fn bubble_steps(&self) -> usize {
        self.per_stage
            .iter()
            .flat_map(|s| s.iter())
            .filter(|s| s.op == StageOp::Idle)
            .count()
    }

    /// Bubble ratio: fraction of all `(stage × time)` slots that are idle.
    pub fn bubble_ratio(&self) -> f64 {
        let total = self.per_stage.iter().map(|s| s.len()).sum::<usize>();
        if total == 0 {
            return 0.0;
        }
        self.bubble_steps() as f64 / total as f64
    }

    /// All steps at a given time across all stages.
    pub fn at_time(&self, time: usize) -> Vec<&ScheduleStep> {
        self.per_stage
            .iter()
            .filter_map(|s| s.get(time))
            .collect()
    }
}

/// Pipeline parallelism configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Number of stages (partitions of the model).
    num_stages: usize,
    /// Number of micro-batches per mini-batch.
    num_microbatches: usize,
}

impl PipelineConfig {
    /// Create a pipeline configuration.
    ///
    /// # Panics
    ///
    /// Panics if `num_stages == 0` or `num_microbatches == 0`.
    pub fn new(num_stages: usize, num_microbatches: usize) -> Self {
        assert!(num_stages > 0, "num_stages must be > 0");
        assert!(num_microbatches > 0, "num_microbatches must be > 0");
        Self { num_stages, num_microbatches }
    }

    /// Number of stages.
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// Number of micro-batches.
    pub fn num_microbatches(&self) -> usize {
        self.num_microbatches
    }

    /// Build a schedule of the given kind.
    pub fn build_schedule(&self, kind: ScheduleKind) -> Schedule {
        match kind {
            ScheduleKind::GPipe => self.build_gpipe(),
            ScheduleKind::OneForwardOneBackward => self.build_1f1b(),
        }
    }

    /// GPipe schedule: all forwards first, then all backwards.
    ///
    /// Time steps: `num_stages + num_microbatches - 1` for forward phase,
    /// same for backward, total = 2 × (S + M - 1).
    fn build_gpipe(&self) -> Schedule {
        let s = self.num_stages;
        let m = self.num_microbatches;
        let phase_len = s + m - 1;
        let total = 2 * phase_len;

        let mut per_stage: Vec<Vec<ScheduleStep>> = Vec::with_capacity(s);
        for stage in 0..s {
            let mut steps = Vec::with_capacity(total);
            for t in 0..total {
                let step = if t < phase_len {
                    // Forward phase
                    let mb = t as isize - stage as isize;
                    if mb >= 0 && (mb as usize) < m {
                        ScheduleStep { stage, time: t, op: StageOp::Forward, microbatch: mb as usize }
                    } else {
                        ScheduleStep { stage, time: t, op: StageOp::Idle, microbatch: 0 }
                    }
                } else {
                    // Backward phase (reversed micro-batch order)
                    let bt = t - phase_len;
                    let mb = bt as isize - (s - 1 - stage) as isize;
                    if mb >= 0 && (mb as usize) < m {
                        let actual_mb = m - 1 - mb as usize;
                        ScheduleStep { stage, time: t, op: StageOp::Backward, microbatch: actual_mb }
                    } else {
                        ScheduleStep { stage, time: t, op: StageOp::Idle, microbatch: 0 }
                    }
                };
                steps.push(step);
            }
            per_stage.push(steps);
        }

        Schedule { num_stages: s, num_microbatches: m, per_stage }
    }

    /// 1F1B schedule: warmup forwards, then strictly interleaved 1-forward-1-backward,
    /// then cooldown backwards.
    fn build_1f1b(&self) -> Schedule {
        let s = self.num_stages;
        let m = self.num_microbatches;

        // Phase 1: warmup — stage i does (s - 1 - i) leading forwards
        // Phase 2: steady state — 1 forward then 1 backward per step
        // Phase 3: cooldown — remaining backwards

        let mut per_stage: Vec<Vec<ScheduleStep>> = vec![Vec::new(); s];

        for stage in 0..s {
            let warmup = s - 1 - stage;
            let mut fwd_idx: usize = 0;
            let mut bwd_idx: usize = 0;
            let mut time: usize = stage; // stagger start

            // Helper: pad stage with idle steps up to `time`.
            fn pad_idle(steps: &mut Vec<ScheduleStep>, stage: usize, time: usize) {
                while steps.len() < time {
                    let t = steps.len();
                    steps.push(ScheduleStep { stage, time: t, op: StageOp::Idle, microbatch: 0 });
                }
            }

            let steps = &mut per_stage[stage];

            // Warmup: forward-only
            for _ in 0..warmup.min(m) {
                pad_idle(steps, stage, time);
                steps.push(ScheduleStep {
                    stage, time, op: StageOp::Forward, microbatch: fwd_idx,
                });
                fwd_idx += 1;
                time += 1;
            }

            // Steady state: 1F1B
            while fwd_idx < m {
                // Forward
                pad_idle(steps, stage, time);
                steps.push(ScheduleStep {
                    stage, time, op: StageOp::Forward, microbatch: fwd_idx,
                });
                fwd_idx += 1;
                time += 1;

                // Backward
                if bwd_idx < m {
                    pad_idle(steps, stage, time);
                    steps.push(ScheduleStep {
                        stage, time, op: StageOp::Backward, microbatch: bwd_idx,
                    });
                    bwd_idx += 1;
                    time += 1;
                }
            }

            // Cooldown: remaining backwards
            while bwd_idx < m {
                pad_idle(steps, stage, time);
                steps.push(ScheduleStep {
                    stage, time, op: StageOp::Backward, microbatch: bwd_idx,
                });
                bwd_idx += 1;
                time += 1;
            }
        }

        // Pad all stages to the same length
        let max_len = per_stage.iter().map(|s| s.len()).max().unwrap_or(0);
        for (stage, steps) in per_stage.iter_mut().enumerate() {
            while steps.len() < max_len {
                let t = steps.len();
                steps.push(ScheduleStep { stage, time: t, op: StageOp::Idle, microbatch: 0 });
            }
        }

        Schedule { num_stages: s, num_microbatches: m, per_stage }
    }
}

/// Describes which layers belong to which pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageAssignment {
    /// Per-layer stage assignment: `assignments[layer_index] = stage_index`.
    pub assignments: Vec<usize>,
    /// Number of stages.
    pub num_stages: usize,
}

impl StageAssignment {
    /// Assign `num_layers` layers evenly across `num_stages` stages.
    pub fn uniform(num_layers: usize, num_stages: usize) -> Self {
        assert!(num_stages > 0 && num_layers >= num_stages,
                "need at least as many layers as stages");
        let per_stage = num_layers / num_stages;
        let remainder = num_layers % num_stages;
        let mut assignments = Vec::with_capacity(num_layers);
        let mut layer = 0;
        for stage in 0..num_stages {
            let count = per_stage + if stage < remainder { 1 } else { 0 };
            for _ in 0..count {
                assignments.push(stage);
                layer += 1;
            }
        }
        let _ = layer;
        Self { assignments, num_stages }
    }

    /// Which stage a layer belongs to.
    pub fn stage_of(&self, layer: usize) -> Option<usize> {
        self.assignments.get(layer).copied()
    }

    /// Layers belonging to a given stage.
    pub fn layers_in_stage(&self, stage: usize) -> Vec<usize> {
        self.assignments
            .iter()
            .enumerate()
            .filter(|(_, s)| **s == stage)
            .map(|(i, _)| i)
            .collect()
    }

    /// Number of layers per stage.
    pub fn layers_per_stage(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.num_stages];
        for &s in &self.assignments {
            counts[s] += 1;
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpipe_schedule_basic() {
        let config = PipelineConfig::new(2, 4);
        let sched = config.build_schedule(ScheduleKind::GPipe);

        assert_eq!(sched.num_stages(), 2);
        assert_eq!(sched.num_microbatches(), 4);

        // Stage 0 first step should be Forward micro-batch 0
        let first = &sched.steps(0)[0];
        assert_eq!(first.op, StageOp::Forward);
        assert_eq!(first.microbatch, 0);
    }

    #[test]
    fn gpipe_all_microbatches_processed() {
        let config = PipelineConfig::new(3, 5);
        let sched = config.build_schedule(ScheduleKind::GPipe);

        for stage in 0..3 {
            let fwd_mbs: Vec<usize> = sched.steps(stage).iter()
                .filter(|s| s.op == StageOp::Forward)
                .map(|s| s.microbatch)
                .collect();
            assert_eq!(fwd_mbs.len(), 5, "stage {stage} should have 5 forward steps");
            // Should process all microbatches 0..5
            let mut sorted = fwd_mbs.clone();
            sorted.sort();
            assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
        }
    }

    #[test]
    fn one_f1b_schedule_basic() {
        let config = PipelineConfig::new(4, 8);
        let sched = config.build_schedule(ScheduleKind::OneForwardOneBackward);

        assert_eq!(sched.num_stages(), 4);
        assert_eq!(sched.num_microbatches(), 8);

        // Stage 0 starts with forward
        let first_non_idle = sched.steps(0).iter()
            .find(|s| s.op != StageOp::Idle)
            .unwrap();
        assert_eq!(first_non_idle.op, StageOp::Forward);
        assert_eq!(first_non_idle.microbatch, 0);
    }

    #[test]
    fn one_f1b_all_microbatches_processed() {
        let config = PipelineConfig::new(3, 6);
        let sched = config.build_schedule(ScheduleKind::OneForwardOneBackward);

        for stage in 0..3 {
            let fwd_mbs: Vec<usize> = sched.steps(stage).iter()
                .filter(|s| s.op == StageOp::Forward)
                .map(|s| s.microbatch)
                .collect();
            let bwd_mbs: Vec<usize> = sched.steps(stage).iter()
                .filter(|s| s.op == StageOp::Backward)
                .map(|s| s.microbatch)
                .collect();
            assert_eq!(fwd_mbs.len(), 6, "stage {stage} forward count");
            assert_eq!(bwd_mbs.len(), 6, "stage {stage} backward count");
        }
    }

    #[test]
    fn one_f1b_less_bubble_than_gpipe() {
        let config = PipelineConfig::new(4, 16);
        let gpipe = config.build_schedule(ScheduleKind::GPipe);
        let f1b = config.build_schedule(ScheduleKind::OneForwardOneBackward);

        // 1F1B should have fewer or equal bubble steps
        assert!(f1b.bubble_ratio() <= gpipe.bubble_ratio(),
                "1F1B bubble={:.2}% should be ≤ GPipe bubble={:.2}%",
                f1b.bubble_ratio() * 100.0, gpipe.bubble_ratio() * 100.0);
    }

    #[test]
    fn bubble_ratio_range() {
        let config = PipelineConfig::new(2, 4);
        let sched = config.build_schedule(ScheduleKind::GPipe);
        let ratio = sched.bubble_ratio();
        assert!(ratio >= 0.0 && ratio <= 1.0);
    }

    #[test]
    fn at_time_query() {
        let config = PipelineConfig::new(2, 3);
        let sched = config.build_schedule(ScheduleKind::GPipe);
        let steps = sched.at_time(0);
        assert_eq!(steps.len(), 2); // both stages have a step at time 0
    }

    #[test]
    fn uniform_stage_assignment() {
        let assign = StageAssignment::uniform(12, 4);
        assert_eq!(assign.layers_per_stage(), vec![3, 3, 3, 3]);
        assert_eq!(assign.stage_of(0), Some(0));
        assert_eq!(assign.stage_of(3), Some(1));
        assert_eq!(assign.stage_of(11), Some(3));
    }

    #[test]
    fn uniform_stage_with_remainder() {
        let assign = StageAssignment::uniform(10, 3);
        // 10 / 3 = 3r1 → stages get 4, 3, 3
        assert_eq!(assign.layers_per_stage(), vec![4, 3, 3]);
    }

    #[test]
    fn layers_in_stage() {
        let assign = StageAssignment::uniform(6, 2);
        assert_eq!(assign.layers_in_stage(0), vec![0, 1, 2]);
        assert_eq!(assign.layers_in_stage(1), vec![3, 4, 5]);
    }

    #[test]
    #[should_panic]
    fn zero_stages_panics() {
        PipelineConfig::new(0, 4);
    }

    #[test]
    #[should_panic]
    fn zero_microbatches_panics() {
        PipelineConfig::new(4, 0);
    }

    #[test]
    fn single_stage_no_bubble() {
        let config = PipelineConfig::new(1, 4);
        let sched = config.build_schedule(ScheduleKind::GPipe);
        // With 1 stage, every step is productive
        let bubble = sched.bubble_steps();
        assert_eq!(bubble, 0);
    }
}
