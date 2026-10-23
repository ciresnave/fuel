# qwen3_0_6b_awq_greedy.json

The greedy token ids that HuggingFace transformers 4.51.3 + AutoAWQ 0.2.9
(torch 2.5.1+cu121, RTX 4070 Laptop GPU, WSL2) produced for the prompt
`"The capital of France is"` with `Orion-zhen/Qwen3-0.6B-AWQ` revision
`9f607bee5fca19cba0bd2b383ba804c89385ea18`, 16 new tokens. The file's own
`provenance` block records the versions and the command.

Copied byte-for-byte from the `ciresnave/lightbulb` repository, commit
`92fc7ec22cab3ac480f7e22e309841b29439568d`, path
`tests/fixtures/awq_reference/v1/qwen3_0_6b_awq_greedy.json` (read with
`git show origin/main:<path>`; SHA-256 `0dbaa06780305e20961a7707bf66fe6a3d741abcf37ac55a014ab753e27ab8d3`
of that blob). It is independent of Fuel: nothing in Fuel produced it.

Licence note: the checkpoint is GPL-3.0 (its README front matter and
lightbulb's `scripts/awq_reference/README.md` both say so) and its base model
is Apache-2.0. The fixture holds sixteen integers and a short decoded string,
no weights and no code from either, and the checkpoint itself is never
committed here (the test reads it from `FUEL_QWEN3_AWQ_DIR`).
