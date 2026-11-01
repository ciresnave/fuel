// SPDX-License-Identifier: MIT OR Apache-2.0
// Fused Q6_K × F32 gemv.
//
// Computes output[n] = sum_k A[k] * dequant(W_block[n, k/256][k%256]) for n in [0, N). W is stored
// as Q6_K super-blocks in [N, K/256] layout (N rows of K/256 blocks of 210 bytes, 256 elements each).
// This is the weight type that dominates a Q6_K Qwen3.5 file, including its tied embedding/output
// head (ROADMAP rows 118 and 119). Before this kernel Q6_K had no Vulkan path at all.
//
// Super-block layout (210 bytes; BlockQ6K in fuel-quantized, ggml block_q6_K):
//   [0..128)    u8[128]  ql: low 4 bits of each quant
//   [128..192)  u8[64]   qh: high 2 bits of each quant
//   [192..208)  i8[16]   scales, one per 16 elements
//   [208..210)  f16 d
// Element e (0..256): half h = e / 128, r = e % 128, group g = r / 32 (0..3), l = r % 32,
//   ql byte  = h*64 + l + (g & 1)*32   (low nibble when g < 2, high nibble when g >= 2)
//   qh byte  = 128 + h*32 + l,  2 bits at shift 2*g
//   scale    = i8 at 192 + h*8 + l/16 + 2*g
//   value    = d * scale * (((nibble) | (qh_bits << 4)) - 32)
// This is ggml's dequantize_row_q6_K written per element.
//
// One workgroup per output column (n), 128 threads, each stepping over K by 128; then a subgroup +
// shared-memory reduction whose cross-subgroup combine strides by the subgroup width (right on
// 8-lane devices, ROADMAP row 134). Simple per-element form: correct and checkable first; whether
// it is fast enough is a measurement.

struct Params {
    uint n;              // output length (N)
    uint k;              // contracted dim length (K, multiple of 256)
    uint blocks_per_row; // K / 256 — super-blocks per output row
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   a_f32;      // [K]
[[vk::binding(1, 0)]] ByteAddressBuffer         w_q6_k;     // [N, K/256] super-blocks (raw bytes)
[[vk::binding(2, 0)]] RWStructuredBuffer<float> c_f32;      // [N]
[[vk::binding(3, 0)]] ConstantBuffer<Params>    p;

static const uint BYTES_PER_BLOCK = 210;
static const uint QK_K = 256;

groupshared float subgroup_partials[32];

uint load_u8(uint byte_off)
{
    uint word_off = byte_off & ~3u;
    uint byte_in_word = byte_off & 3u;
    uint word = w_q6_k.Load(word_off);
    return (word >> (byte_in_word * 8u)) & 0xFFu;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void main(
    uint3 gid  : SV_GroupID,
    uint3 gtid : SV_GroupThreadID)
{
    uint n = gid.x;
    if (n >= p.n) return;

    uint tid = gtid.x;
    uint row_byte_off = n * p.blocks_per_row * BYTES_PER_BLOCK;

    float partial = 0.0;
    uint k_idx = tid;
    while (k_idx < p.k) {
        uint block_idx = k_idx / QK_K;
        uint e = k_idx % QK_K;
        uint h = e / 128u;
        uint r = e % 128u;
        uint g = r / 32u;
        uint l = r % 32u;

        uint b = row_byte_off + block_idx * BYTES_PER_BLOCK;

        uint ql_byte = load_u8(b + h * 64u + l + (g & 1u) * 32u);
        uint nibble = (g < 2u) ? (ql_byte & 0xFu) : (ql_byte >> 4u);
        uint qh_byte = load_u8(b + 128u + h * 32u + l);
        uint hi = (qh_byte >> (2u * g)) & 3u;
        int q = int(nibble | (hi << 4u)) - 32;

        int sc = int(load_u8(b + 192u + h * 8u + l / 16u + 2u * g));
        if (sc >= 128) { sc -= 256; }

        uint d_bits = load_u8(b + 208u) | (load_u8(b + 209u) << 8u);
        float d = f16tof32(d_bits);

        partial += a_f32[k_idx] * (d * float(sc) * float(q));
        k_idx += 128u;
    }

    float sg_sum = WaveActiveSum(partial);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        subgroup_partials[sg_id] = sg_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_subgroups = 128u / WaveGetLaneCount();
    if (sg_id == 0) {
        float v = 0.0;
        for (uint sgi = tid; sgi < n_subgroups; sgi += WaveGetLaneCount()) {
            v = v + subgroup_partials[sgi];
        }
        float total = WaveActiveSum(v);
        if (WaveIsFirstLane()) {
            c_f32[n] = total;
        }
    }
}
