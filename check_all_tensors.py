from safetensors import safe_open

weights_path = 'gemma-final/model.safetensors'
with safe_open(weights_path, framework='np', device='cpu') as f:
    keys = set(f.keys())
    
    expected_per_layer = [
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight"
    ]
    
    for i in range(18):
        for suffix in expected_per_layer:
            key = f"model.layers.{i}.{suffix}"
            if key not in keys:
                print(f"Missing: {key}")
    
    if "model.embed_tokens.weight" not in keys: print("Missing: model.embed_tokens.weight")
    if "model.norm.weight" not in keys: print("Missing: model.norm.weight")
    if "lm_head.weight" not in keys: print("Missing: lm_head.weight")
