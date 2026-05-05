with open('convert_hf_to_gguf.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('gguf.MODEL_ARCH.GEMMA4', 'gguf.MODEL_ARCH.GEMMA')
text = text.replace('gguf.MODEL_ARCH.GEMMAN', 'gguf.MODEL_ARCH.GEMMA3N')

with open('convert_hf_to_gguf_compat.py', 'w', encoding='utf-8') as f:
    f.write(text)
