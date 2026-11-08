"""
Extended Suite A: held-out milestone-recall tasks, multi-seed, with ablations.
Writes one JSON line per run to results/suite_a_ext_<tag>.jsonl (resumable, task-major order).

Held-out = tasks index >= 8 of `build_long_cot_milestone_tasks` (tasks 0-7 were used while tuning vacor).
Deterministic methods are run with 1 seed, stochastic ones with `--seeds` seeds.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_benchmarks import build_long_cot_milestone_tasks, extract_boxed_or_last_number
from src.kv_engine import UnifiedEvictCache, patch_qwen3_evict_sdpa

DETERMINISTIC = ["recency_pp", "snapkv_pp", "rkv_pp", "valuediff_pp", "valuediff_smooth_pp"]
STOCHASTIC = ["vase_pp", "random_pp", "vacor_kv", "vacor_pin", "vacor_antithetic_only"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--tag", default="q17b")
    ap.add_argument("--n_tasks", type=int, default=16)
    ap.add_argument("--budgets", type=int, nargs="+", default=[384, 512])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max_tasks_run", type=int, default=10**9)
    a = ap.parse_args()

    patch_qwen3_evict_sdpa()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    tasks = build_long_cot_milestone_tasks(num_tasks=8 + a.n_tasks)[8:]

    path = ROOT / "results" / f"suite_a_ext_{a.tag}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["task"], r["budget"], r["mode"], r["seed"]))

    for t in tasks[: a.max_tasks_run]:
        jobs = [("dense", 0, 0)]
        for b in a.budgets:
            for m in DETERMINISTIC:
                jobs.append((m, b, 0))
            for m in STOCHASTIC:
                for s in range(a.seeds):
                    jobs.append((m, b, s))
        for mode, b, s in jobs:
            key = (t["id"], b, mode, s)
            if key in done:
                continue
            t0 = time.time()
            r = run_one(model, tok, t, mode, b if b else 4096, 1000 * s + int(t["id"].split("_")[1]))
            r.update(task=t["id"], budget=b, mode=mode, seed=s, sec=round(time.time() - t0, 1), model=a.model)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            print(t["id"], b, mode, s, {k: r[k] for k in ("m1", "m2", "m3", "all3", "final")}, r["sec"], flush=True)


if __name__ == "__main__":
    main()
