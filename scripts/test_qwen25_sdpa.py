import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.kv_engine import patch_qwen2_evict_sdpa, UnifiedEvictCache

def test_qwen25():
    patch_qwen2_evict_sdpa()
    model_id = "Qwen/Qwen2.5-1.5B"
    print("Loading tokenizer and model...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    prompt = "Question: What is 25 * 14? Let's solve it step by step:"
    inputs = tok(prompt, return_tensors="pt").to("cuda")

    # Run with UnifiedEvictCache (budget=128, snapkv_pp)
    cache = UnifiedEvictCache(
        config=model.config,
        token_budget=128,
        residual_length=32,
        eviction_mode="snapkv_pp",
    )

    print("Generating with UnifiedEvictCache...")
    with torch.no_grad():
        out = model.generate(**inputs, past_key_values=cache, max_new_tokens=60, do_sample=False)
    resp = tok.decode(out[0], skip_special_tokens=True)
    print("Output:\n", resp)
    print("Qwen2.5 test PASSED successfully!")

if __name__ == "__main__":
    test_qwen25()
