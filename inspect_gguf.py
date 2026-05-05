import gguf
import sys
import numpy as np

reader = gguf.GGUFReader("gemma_test.gguf")
print(f"Num Tensors: {len(reader.tensors)}")
for i, tensor in enumerate(reader.tensors):
    print(f"{i}: {tensor.name} | Shape: {tensor.shape} | Dtype: {tensor.tensor_type}")
