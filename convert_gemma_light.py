import json
import os
import sys
from pathlib import Path
import numpy as np
from safetensors import safe_open
import gguf

def convert_gemma_to_gguf(model_dir, output_path):
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"Opening weights from {weights_path}...")
    
    arch = "gemma"
    writer = gguf.GGUFWriter(output_path, arch)
    
    # Set hyperparams
    writer.add_name("Gemma Custom")
    writer.add_context_length(config.get("max_position_embeddings", 8192))
    writer.add_embedding_length(config["hidden_size"])
    writer.add_block_count(config["num_hidden_layers"])
    writer.add_feed_forward_length(config["intermediate_size"])
    writer.add_head_count(config["num_attention_heads"])
    writer.add_head_count_kv(config.get("num_key_value_heads", config["num_attention_heads"]))
    writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    writer.add_rope_freq_base(config["rope_theta"])
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    
    # Gemma specific
    writer.add_uint32("gemma.head_dim", config["head_dim"])

    # Load tokenizer
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(model_dir / "tokenizer.model"))
        tokens = []
        scores = []
        for i in range(sp.get_piece_size()):
            tokens.append(sp.id_to_piece(i).encode('utf-8'))
            scores.append(sp.get_score(i))
        writer.add_tokenizer_model("llama")
        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        print(f"Loaded vocab with {len(tokens)} tokens.")
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")

    # Tensor Mapping
    # Standard order: embeddings, blocks, output norm, output head
    tensor_data_to_write = []
    
    with safe_open(weights_path, framework="np", device="cpu") as f_in:
        available_keys = f_in.keys()
        
        # 1. Token Embeddings
        if "model.embed_tokens.weight" in available_keys:
            shape = f_in.get_slice("model.embed_tokens.weight").get_shape()
            writer.add_tensor_info("token_embd.weight", shape, np.dtype(np.float16), int(np.prod(shape)) * 2)
            tensor_data_to_write.append(("model.embed_tokens.weight", "token_embd.weight"))

        # 2. Transformer Blocks
        for i in range(config["num_hidden_layers"]):
            layer_mapping = {
                f"model.layers.{i}.input_layernorm.weight": f"blk.{i}.attn_norm.weight",
                f"model.layers.{i}.self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
                f"model.layers.{i}.self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
                f"model.layers.{i}.self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
                f"model.layers.{i}.self_attn.o_proj.weight": f"blk.{i}.attn_output.weight",
                f"model.layers.{i}.post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
                f"model.layers.{i}.mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
                f"model.layers.{i}.mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
                f"model.layers.{i}.mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
            }
            for hf_name, gguf_name in layer_mapping.items():
                if hf_name in available_keys:
                    shape = f_in.get_slice(hf_name).get_shape()
                    writer.add_tensor_info(gguf_name, shape, np.dtype(np.float16), int(np.prod(shape)) * 2)
                    tensor_data_to_write.append((hf_name, gguf_name))
                else:
                    print(f"Missing tensor: {hf_name}")

        # 3. Output Norm
        if "model.norm.weight" in available_keys:
            shape = f_in.get_slice("model.norm.weight").get_shape()
            writer.add_tensor_info("output_norm.weight", shape, np.dtype(np.float16), int(np.prod(shape)) * 2)
            tensor_data_to_write.append(("model.norm.weight", "output_norm.weight"))

        # 4. Output Head
        if "lm_head.weight" in available_keys:
            shape = f_in.get_slice("lm_head.weight").get_shape()
            writer.add_tensor_info("output.weight", shape, np.dtype(np.float16), int(np.prod(shape)) * 2)
            tensor_data_to_write.append(("lm_head.weight", "output.weight"))
        else:
            # Tied weights
            shape = f_in.get_slice("model.embed_tokens.weight").get_shape()
            writer.add_tensor_info("output.weight", shape, np.dtype(np.float16), int(np.prod(shape)) * 2)
            tensor_data_to_write.append(("model.embed_tokens.weight", "output.weight"))

    print(f"Writing metadata and header... Total tensors planned: {len(tensor_data_to_write)}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    print("Writing tensors to disk...")
    import gc
    with safe_open(weights_path, framework="np", device="cpu") as f_in:
        for hf_name, gguf_name in tensor_data_to_write:
            data = f_in.get_tensor(hf_name)
            
            # Apply Gemma norm shift (+1.0)
            if hf_name.endswith("norm.weight"):
                data = data.astype(np.float32) + 1.0
            
            if data.dtype != np.float16:
                data = data.astype(np.float16)
            
            print(f"Writing {hf_name} -> {gguf_name}", flush=True)
            if gguf_name == "output.weight":
                data = data.copy()
                data[0, 0] += 1e-5
            
            writer.write_tensor_data(data)
            del data
            gc.collect()
            
    writer.close()
    print(f"Successfully converted to {output_path}. Tensors written: {len(tensor_data_to_write)}")

if __name__ == "__main__":
    convert_gemma_to_gguf("gemma-final", "gemma_test.gguf")
