# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerates fuel-transformers/tests/fixtures/qwen3_5_0_8b_forward_reference.safetensors.

Runs Qwen/Qwen3.5-0.8B in float32 on CPU with HuggingFace transformers (5.22.1 when this was
written) over the 8 tokens below. Records the embedding output, the output of decoder layers 0, 1,
3 and 23, the final norm output, and for the whole-sequence logits and for token-at-a-time cached
decode the top-10 token ids and values per position (the full logits are 8 x 248320 floats; the
top-10 keeps the file small). Usage:
    python forward_ref.py <checkpoint dir> <out.safetensors>
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
rec, on = {}, [True]
def hook(name):
    def f(mod, inp, outp):
        if on[0]:
            o = outp[0] if isinstance(outp, tuple) else outp
            rec[name] = o.detach().float().numpy()[0]
    return f
tm.embed_tokens.register_forward_hook(hook("embed"))
for i in (0, 1, 3, 23):
    tm.layers[i].register_forward_hook(hook(f"layer{i}"))
tm.norm.register_forward_hook(hook("final_norm"))
logits = m(input_ids=torch.tensor([toks])).logits.detach().float()[0]
on[0] = False
def top(l):
    v, i = torch.topk(l, 10, dim=-1)
    return v.numpy(), i.numpy().astype(np.int64)
rec["top_vals"], rec["top_ids"] = top(logits)
past, steps = None, []
for t in toks:
    o = m(input_ids=torch.tensor([[t]]), past_key_values=past, use_cache=True)
    past = o.past_key_values
    steps.append(o.logits.detach().float()[0, -1])
rec["step_top_vals"], rec["step_top_ids"] = top(torch.stack(steps))
rec["tokens"] = np.array(toks, dtype=np.int64)
save_file({k: np.ascontiguousarray(v) for k, v in rec.items()}, out)
print({k: v.shape for k, v in rec.items()})
