# SPDX-License-Identifier: MIT OR Apache-2.0
# From huggingface/candle (MIT OR Apache-2.0), candle-examples/examples/resnet/export_models.py, body unchanged; see THIRD-PARTY-NOTICES.md.
# This script exports pre-trained model weights in the safetensors format.
import numpy as np
import torch
import torchvision
from safetensors import torch as stt

m = torchvision.models.resnet50(pretrained=True)
stt.save_file(m.state_dict(), 'resnet50.safetensors')
m = torchvision.models.resnet101(pretrained=True)
stt.save_file(m.state_dict(), 'resnet101.safetensors')
m = torchvision.models.resnet152(pretrained=True)
stt.save_file(m.state_dict(), 'resnet152.safetensors')
