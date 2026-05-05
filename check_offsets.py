import gguf
reader = gguf.GGUFReader("gemma_test.gguf")
for i, tensor in enumerate(reader.tensors):
    # n_bytes is not always there, try calculating from shape
    import numpy as np
    size = np.prod(tensor.shape) * 2 # F16
    print(f"{i}: {tensor.name} | Offset: {tensor.data_offset} | Size: {size}")
