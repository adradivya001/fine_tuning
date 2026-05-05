import gguf
for key, name in gguf.TENSOR_NAMES.items():
    if key in gguf.MODEL_TENSORS[gguf.MODEL_ARCH.GEMMA]:
        print(f"{key}: {name}")
