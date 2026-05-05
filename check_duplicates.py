import gguf
reader = gguf.GGUFReader("gemma_test.gguf")
names = [t.name for t in reader.tensors]
print(f"Unique names: {len(set(names))}")
print(f"Total names: {len(names)}")
if len(set(names)) != len(names):
    from collections import Counter
    c = Counter(names)
    for name, count in c.items():
        if count > 1:
            print(f"Duplicate: {name}")
