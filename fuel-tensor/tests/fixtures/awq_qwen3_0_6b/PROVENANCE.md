# awq_qwen3_0_6b/slice.safetensors

Real bytes from `Orion-zhen/Qwen3-0.6B-AWQ` (revision
`9f607bee5fca19cba0bd2b383ba804c89385ea18`, an AutoAWQ GEMM checkpoint; model
licence Apache-2.0, as for Qwen3), file `model.safetensors`, layer
`model.layers.0.mlp.gate_proj` (full shapes `qweight [1024, 384]` I32,
`qzeros [8, 384]` I32, `scales [8, 3072]` F16, group size 128).

The slice is a verbatim byte copy: `qweight` rows 0..256 x words 0..8,
`qzeros` rows 0..2 x words 0..8, `scales` rows 0..2 x columns 0..64, i.e. a
valid layer with `IC = 256`, `OC = 64`, `group_size = 128`. Stored under the
tensor prefix `l`. Nothing was requantised or edited. The source file was read
from `C:\Models\Qwen3-0.6B-AWQ\model.safetensors` (first-MiB SHA-256 prefix
`04529031`; the revision is taken from lightbulb's
`scripts/awq_reference/README.md`, not re-verified against the hub here).
