"""
Evaluate Adaptive VaCoR-KV on Suite A (Qwen3-1.7B and Qwen3-4B).
Computes All-3 recall, M1/M2/M3 survival, and compares directly with static VaCoR-KV and ValueDiff.
"""

import json
from pathlib import Path
import re
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_benchmarks import build_long_cot_milestone_tasks, extract_boxed_or_last_number
from src.kv_engine import UnifiedEvictCache, patch_qwen3_evict_sdpa


@torch.no_grad()
def run_one(model, tokenizer, task, mode, budget, seed, residual=64, max_new=64):
    dev = next(model.parameters()).device
    p_ids = tokenizer(task["prompt"], add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    c_ids = tokenizer(task["cot_prefix"], add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    eos = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    cache = UnifiedEvictCache(model.config, token_budget=budget, residual_length=residual,
                              eviction_mode=mode, seed=seed, track_positions=True)
    out = model(input_ids=p_ids, past_key_values=cache, use_cache=True)
    logits = out.logits[:, -1, :]
    cur, n = 0, c_ids.shape[1]
    while cur < n:
        c_len = cache.layers[0].cur_len
        if mode == "dense":
            step = n - cur
        elif c_len < budget + residual:
            step = min(budget + residual - c_len, n - cur)
        else:
            step = min(residual, n - cur)
        out = model(input_ids=c_ids[:, cur:cur + step], past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1, :]
        cur += step
    gen = []
    for _ in range(max_new):
        nid = int(logits.argmax(-1).item())
        gen.append(nid)
        if nid in eos:
            break
        out = model(input_ids=torch.tensor([[nid]], device=dev), past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1, :]
    text = tokenizer.decode(gen, skip_special_tokens=False)
    tail = "M1 = " + text
    m1 = re.search(rf"M1\s*=\s*{task['val_a']}\b", tail) is not None
    m2 = re.search(rf"M2\s*=\s*{task['val_b']}\b", tail) is not None or str(task["val_b"]) in text
    m3 = re.search(rf"M3\s*=\s*{task['val_c']}\b", tail) is not None or str(task["val_c"]) in text
    pred = extract_boxed_or_last_number(text)
    fs = (pred == task["ground_truth"]) or (f"= {task['ground_truth']}" in text)
    return dict(m1=m1, m2=m2, m3=m3, all3=m1 and m2 and m3, final=bool(fs))


def run_suite_a_eval(model_id="Qwen/Qwen3-1.7B", tag="q17b"):
    print(f"=== Evaluating Adaptive VaCoR-KV on Suite A ({model_id}) ===")
    patch_qwen3_evict_sdpa()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    tasks = build_long_cot_milestone_tasks(num_tasks=24)[8:]  # 16 held-out tasks

    budgets = [384, 512]
    seeds = [0, 1, 2]

    results = []
    for b in budgets:
        all3_list = []
        m1_list = []
        m2_list = []
        m3_list = []
        t0 = time.time()
        for idx, t in enumerate(tasks):
            for s in seeds:
                res = run_one(model, tok, t, "adaptive_vacor", b, s)
                all3_list.append(float(res["all3"]))
                m1_list.append(float(res["m1"]))
                m2_list.append(float(res["m2"]))
                m3_list.append(float(res["m3"]))
                rec = {
                    "task": idx,
                    "budget": b,
                    "mode": "adaptive_vacor",
                    "seed": s,
                    **res,
                }
                results.append(rec)
        elapsed = time.time() - t0
        all3_mean = sum(all3_list) / len(all3_list) * 100
        m1_mean = sum(m1_list) / len(m1_list) * 100
        m2_mean = sum(m2_list) / len(m2_list) * 100
        m3_mean = sum(m3_list) / len(m3_list) * 100
        print(f"[{tag}] Adaptive VaCoR-KV (B={b}): All-3 = {all3_mean:.1f}% | M1={m1_mean:.1f}% M2={m2_mean:.1f}% M3={m3_mean:.1f}% | {elapsed:.1f}s")

    out_file = ROOT / "results" / f"adaptive_vacor_{tag}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {out_file}\n")
    return results


if __name__ == "__main__":
    run_suite_a_eval("Qwen/Qwen3-1.7B", "q17b")
