# SPDX-License-Identifier: MIT OR Apache-2.0
# From huggingface/candle (MIT OR Apache-2.0), candle-examples/examples/flux/t5_tokenizer.py, body unchanged; see THIRD-PARTY-NOTICES.md.
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "google/t5-v1_1-xxl"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# The tokenizer will be saved in /tmp/tokenizer/tokenizer.json
tokenizer.save_pretrained("/tmp/tokenizer/")
