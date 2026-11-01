// SPDX-License-Identifier: MIT OR Apache-2.0
// Fused softmax along the last dimension — f64.
//
// Formula:  y_j = exp(x_j - max) / sum_i(exp(x_i - max))
//
// Native f64 end-to-end (matches baracuda's CUDA pattern). Requires
// shaderFloat64 + GroupNonUniformArithmetic. Max is GLSL.std.450 FMax on
// doubles. exp is the double-double `poly_exp_d` from f64_dd_exp.slangh,
// NOT the native `exp(double)`: GLSL.std.450's Exp is restricted to 16/32-bit
// floats, NVIDIA's driver tolerated the out-of-spec call at about f32 accuracy,
// and Lavapipe returned inf (ROADMAP row 135). No OpenCL.std (broken on NVIDIA
// Windows Vulkan).
//
// Same Params shape (8 bytes) and 3-phase structure as the f32 kernel.

#include "f64_dd_exp.slangh"

struct Params {
    uint n_rows;
    uint n_cols;
};

[[vk::binding(0, 0)]] StructuredBuffer<double>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>     params;

groupshared double wg_max[64];
groupshared double wg_sum[64];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_off = row * params.n_cols;

    // Phase 1 — per-row max in f64. -DBL_MAX as the per-lane sentinel.
    double local_max = -1.7976931348623157e+308;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        double v = input[row_off + col];
        local_max = max(local_max, v);
    }

    double sg_max = WaveActiveMax(local_max);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_max[sg_id] = sg_max;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_sg = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        double v = -1.7976931348623157e+308;
        for (uint sgi = tid; sgi < n_sg; sgi += WaveGetLaneCount()) {
            v = max(v, wg_max[sgi]);
        }
        double m = WaveActiveMax(v);
        if (WaveIsFirstLane()) {
            wg_max[0] = m;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    double row_max = wg_max[0];

    // Phase 2 — exp + write + accumulate sum.
    double local_sum = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        double v = poly_exp_d(input[row_off + col] - row_max);
        output[row_off + col] = v;
        local_sum += v;
    }

    double sg_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) {
        wg_sum[sg_id] = sg_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (sg_id == 0u) {
        double v = 0.0;
        for (uint sgi = tid; sgi < n_sg; sgi += WaveGetLaneCount()) {
            v = v + wg_sum[sgi];
        }
        double s = WaveActiveSum(v);
        if (WaveIsFirstLane()) {
            wg_sum[0] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    double row_sum = wg_sum[0];

    // Phase 3 — normalize.
    double inv_sum = 1.0 / row_sum;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        output[row_off + col] *= inv_sum;
    }
}
