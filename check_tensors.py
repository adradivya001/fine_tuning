from safetensors import safe_open
import json

weights_path = 'gemma-final/model.safetensors'
with safe_open(weights_path, framework='np', device='cpu') as f:
    keys = list(f.keys())
    for k in keys:
        print(k)
