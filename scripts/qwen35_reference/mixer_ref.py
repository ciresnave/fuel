# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerates fuel-transformers/tests/fixtures/qwen3_5_0_8b_gdn_mixer_reference.safetensors.

Runs Qwen/Qwen3.5-0.8B in float32 on CPU with HuggingFace transformers (5.22.1 when this was
written) over the 8 tokens below and records the input and output of the Gated DeltaNet token mixer
(`linear_attn`) of layers 0 and 1. Usage:
    python mixer_ref.py <checkpoint dir> <out.safetensors>
Needs: torch (CPU is enough), transformers, safetensors, numpy.
"""
import sys, numpy as np, torch
from transformers import Qwen3_5ForConditionalGeneration
from safetensors.numpy import save_file

d, out = sys.argv[1], sys.argv[2]
torch.set_grad_enabled(False)
m = Qwen3_5ForConditionalGeneration.from_pretrained(d, dtype=torch.float32).eval()
toks = [9707, 11, 1879, 0, 358, 1079, 264, 1273]
tm = m.model.language_model
rec = {}
def hook(name, mixer):
    def f(mod, args, kwargs, outp):
        x = args[0] if args else kwargs["hidden_states"]
        o = outp[0] if isinstance(outp, tuple) else outp
        rec[name + "_in"] = x.detach().float().numpy()[0]
        rec[name + "_out"] = o.detach().float().numpy()[0]
    return f
for i in (0, 1):
    tm.layers[i].linear_attn.register_forward_hook(hook(f"gdn{i}", True), with_kwargs=True)
m(input_ids=torch.tensor([toks]))
rec["tokens"] = np.array(toks, dtype=np.int64)
save_file({k: np.ascontiguousarray(v) for k, v in rec.items()}, out)
print({k: v.shape for k, v in rec.items()})
