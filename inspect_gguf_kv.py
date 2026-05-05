import gguf
import sys

reader = gguf.GGUFReader("gemma_test.gguf")
print("KV Pairs:")
for key in reader.fields:
    print(f"{key}: {reader.fields[key].parts[-1]}")
