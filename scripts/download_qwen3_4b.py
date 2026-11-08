import sys
import time
from huggingface_hub import snapshot_download

def main():
    model_id = "Qwen/Qwen3-4B"
    print(f"Starting snapshot download for {model_id} (I/O only, no GPU/CUDA)...", flush=True)
    t0 = time.time()
    try:
        path = snapshot_download(
            repo_id=model_id,
            repo_type="model"
        )
        elapsed = time.time() - t0
        print(f"SUCCESS: Snapshot downloaded to {path} in {elapsed:.1f}s", flush=True)
    except Exception as e:
        print(f"ERROR: Download failed with exception: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
