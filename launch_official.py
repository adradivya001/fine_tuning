import gguf
from enum import IntEnum

# Monkey patch gguf.MODEL_ARCH to include missing members
# We'll add them to the IntEnum. 
# This is tricky because IntEnum is usually fixed, but we can try to re-create it or just add attributes.

class ExtendedModelArch(IntEnum):
    pass

# Copy existing members
members = {m.name: m.value for m in gguf.MODEL_ARCH}
# Add missing ones with dummy values (they won't be used for Gemma anyway)
missing = ['GEMMA4', 'GEMMAN', 'MISTRAL4', 'QWEN3', 'LLAMA4', 'MINIMAXM2', 'GEMMA3', 'GEMMA3N']
val = max(members.values()) + 1
for m in missing:
    if m not in members:
        members[m] = val
        val += 1

# Re-create the enum
gguf.MODEL_ARCH = IntEnum('MODEL_ARCH', members)

# Now import the rest and run
import sys
import os

# We need to run the script logic.
# Instead of importing, we'll just run it as a subprocess with this patch?
# No, we can't easily patch a subprocess.
# We'll write a "launcher" that patches and then imports the script.

with open('convert_hf_to_gguf.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Set arguments
sys.argv = ['convert_hf_to_gguf.py', 'gemma-final', '--outfile', 'gemma_official.gguf', '--outtype', 'f16']

# Execute the code in this patched environment
exec(code, {'__name__': '__main__', '__file__': 'convert_hf_to_gguf.py', 'gguf': gguf, 'sys': sys, 'os': os, 'argparse': __import__('argparse'), 'logging': __import__('logging'), 'json': __import__('json'), 're': __import__('re'), 'Path': __import__('pathlib').Path, 'IntEnum': __import__('enum').IntEnum})
