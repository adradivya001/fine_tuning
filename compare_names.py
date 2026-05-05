import gguf

arch = gguf.MODEL_ARCH.GEMMA
num_layers = 18
expected_names = []

# Embeddings
expected_names.append("token_embd.weight")

# Layers
for i in range(num_layers):
    # This is a bit of a guess on what llama.cpp expects, 
    # but let's see what the library says.
    # Actually, llama.cpp's Gemma loader expects:
    # 1. attn_norm
    # 2. attn_q
    # 3. attn_k
    # 4. attn_v
    # 5. attn_output
    # 6. ffn_norm
    # 7. ffn_gate
    # 8. ffn_up
    # 9. ffn_down
    expected_names.append(f"blk.{i}.attn_norm.weight")
    expected_names.append(f"blk.{i}.attn_q.weight")
    expected_names.append(f"blk.{i}.attn_k.weight")
    expected_names.append(f"blk.{i}.attn_v.weight")
    expected_names.append(f"blk.{i}.attn_output.weight")
    expected_names.append(f"blk.{i}.ffn_norm.weight")
    expected_names.append(f"blk.{i}.ffn_gate.weight")
    expected_names.append(f"blk.{i}.ffn_up.weight")
    expected_names.append(f"blk.{i}.ffn_down.weight")

# Output
expected_names.append("output_norm.weight")
expected_names.append("output.weight")

print(f"Total expected: {len(expected_names)}")

# Now check my actual GGUF
reader = gguf.GGUFReader("gemma_test.gguf")
actual_names = [t.name for t in reader.tensors]
print(f"Total actual: {len(actual_names)}")

missing = [n for n in expected_names if n not in actual_names]
extra = [n for n in actual_names if n not in expected_names]

print(f"Missing: {missing}")
print(f"Extra: {extra}")
