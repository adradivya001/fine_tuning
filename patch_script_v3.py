import re

with open('convert_hf_to_gguf.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Delete classes that cause issues due to missing ARCH names
classes_to_delete = [
    'Gemma3NModel', 'Gemma4Model', 'Mistral4Model', 'Qwen3Model', 'Llama4Model'
]

for cls in classes_to_delete:
    # Match class definition up to the next class or end of file
    pattern = rf'class {cls}\(.*?\)[\s\S]*?(?=class |\Z)'
    content = re.sub(pattern, '', content)

# Also remove registrations
content = re.sub(rf'@ModelBase\.register\(".*?"\)\s+class (?:{"|".join(classes_to_delete)})', '', content)

with open('convert_hf_to_gguf_compat.py', 'w', encoding='utf-8') as f:
    f.write(content)
