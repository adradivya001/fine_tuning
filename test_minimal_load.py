from safetensors import safe_open
import numpy as np

weights_path = 'gemma-final/model.safetensors'
with safe_open(weights_path, framework='np', device='cpu') as f:
    key = 'model.embed_tokens.weight'
    # Check if we can get shape without loading
    print(f"Keys: {list(f.keys())[:5]}")
    # In some versions of safetensors, you can use f.get_slice(key).get_shape() 
    # or look at the header.
    # Let's try to just use f.get_tensor(key).shape but immediately del
    import gc
    t = f.get_tensor(key)
    print(f"Shape: {t.shape}")
    del t
    gc.collect()
    print("Done")
