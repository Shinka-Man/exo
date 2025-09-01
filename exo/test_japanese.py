#!/usr/bin/env python3
import subprocess
import sys

prompt = "Pythonで1から10までの数字を出力するコードを書いてください"
cmd = [
    sys.executable, "-m", "exo.main",
    "--disable-tui",
    "--run-model", "qwen3-30b-a3b-8bit",
    "--prompt", prompt
]

print(f"Testing prompt: {prompt}")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
print("\nGenerated response:")
# Extract the response part
output = result.stdout
if "Generated response:" in output:
    response_start = output.index("Generated response:")
    print(output[response_start:])