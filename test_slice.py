from safetensors import safe_open

weights_path = 'gemma-final/model.safetensors'
with safe_open(weights_path, framework='np', device='cpu') as f:
    key = 'model.embed_tokens.weight'
    # Use get_slice to avoid loading data
    slice_obj = f.get_slice(key)
    print(f"Slice Shape: {slice_obj.get_shape()}") # Some versions use .shape, some .get_shape()
    # Try .shape if .get_shape() fails
    try:
        print(f"Slice Shape attribute: {slice_obj.shape}")
    except:
        pass
