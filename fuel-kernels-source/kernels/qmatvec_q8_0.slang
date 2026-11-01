// SPDX-License-Identifier: MIT OR Apache-2.0
// Fused Q8_0 × F32 gemv.
//
// Computes output[n] = sum_k A[k] * dequant(W_block[n, k/32][k%32]) for
// n in [0, N). Weight matrix W is stored as Q8_0 blocks in [N, K/32]
// layout (N rows of K/32 blocks of 34 bytes each = 32 elements each).
//
// This is the decode-phase (M=1) path for Q8_0 weights. Before this kernel a Q8_0
// QMatMul dequantized the WHOLE weight to an f32 scratch buffer on every use
// (4x the weight bytes written and read back); here the blocks stay quantized and
// are dequantized in the dot product. Same structure as qmatvec_q4_0.slang.
//
// Block layout (34 bytes per block, 32 elements):
//   [0:2)   f16 scale d
//   [2:34)  32 bytes of signed i8 quants
// Formula: elem = qs[k] * d.
//
// Blocks are not 4-byte aligned (34 bytes each), so every read goes through
// `load_u8`, an aligned word load plus a shift.
//
// One workgroup per output column (n). 128 threads per workgroup. Each thread
// accumulates a partial sum over its K slice, then a subgroup + shared-memory
// reduction produces the final output[n]. The cross-subgroup combine strides by
// the subgroup width, so it is correct on 8-lane devices too (ROADMAP row 134).

struct Params {
    uint n;              // output length (N)
    uint k;              // contracted dim length (K, must be multiple of 32)
    uint blocks_per_row; // K / 32 — number of Q8_0 blocks per output row
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   a_f32;      // [K]
[[vk::binding(1, 0)]] ByteAddressBuffer         w_q8_0;     // [N, K/32] blocks (raw bytes)
[[vk::binding(2, 0)]] RWStructuredBuffer<float> c_f32;      // [N]
[[vk::binding(3, 0)]] ConstantBuffer<Params>    p;

static const uint BYTES_PER_BLOCK = 34;
static const uint BLCK_SIZE = 32;

// Shared memory for cross-subgroup reduction (up to 32 subgroups at
// workgroup size 128; a 4-lane subgroup would need 32, narrower is unsupported).
groupshared float subgroup_partials[32];

// Load one byte at an arbitrary byte offset.
uint load_u8(uint byte_off)
{
    uint word_off = byte_off & ~3u;
    uint byte_in_word = byte_off & 3u;
    uint word = w_q8_0.Load(word_off);
    return (word >> (byte_in_word * 8u)) & 0xFFu;
}

// Load the f16 scale at an arbitrary byte offset (2 bytes).
float load_f16_scale(uint byte_off)
{
    uint lo = load_u8(byte_off);
    uint hi = load_u8(byte_off + 1);
    return f16tof32(lo | (hi << 8u));
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

    // Each output row n starts at byte offset n * blocks_per_row * BYTES_PER_BLOCK.
    uint row_byte_off = n * p.blocks_per_row * BYTES_PER_BLOCK;

    float partial = 0.0;

    // Thread-stride over K: 128 threads each step 128 in k.
    uint k_idx = tid;
    while (k_idx < p.k) {
        uint block_idx = k_idx / BLCK_SIZE;
        uint elem_in_block = k_idx % BLCK_SIZE;
        uint block_byte_off = row_byte_off + block_idx * BYTES_PER_BLOCK;
        float d = load_f16_scale(block_byte_off);
        int q = int(load_u8(block_byte_off + 2 + elem_in_block));
        if (q >= 128) { q -= 256; }
        partial += a_f32[k_idx] * (float(q) * d);
        k_idx += 128u;
    }

    // Subgroup reduction + cross-subgroup reduction through shared memory.
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
