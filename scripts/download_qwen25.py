import sys
from huggingface_hub import snapshot_download

print("Downloading Qwen/Qwen2.5-1.5B (Control baseline without QK-Norm)...")
try:
    path = snapshot_download("Qwen/Qwen2.5-1.5B", max_workers=4)
    print("Download complete:", path)
except Exception as e:
    print("Error:", e)
