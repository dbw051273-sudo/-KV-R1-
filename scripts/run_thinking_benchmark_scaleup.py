"""Scale-up Native Thinking Mode (enable_thinking=True) Long CoT Benchmark.

Evaluates dense vs snapkv_pp vs valuediff_pp vs adaptive_vacor on 20 representative
MATH-500 problems spanning high-difficulty algebra, precalculus, number theory, and geometry.
Records full thinking trace length, eviction rounds, and answer accuracy.
Saves results incrementally to results/math_thinking_mode_scaleup.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kv_engine import UnifiedEvictCache, patch_qwen3_evict_sdpa
from scripts.run_math500_eval import extract_boxed_robust, normalize_math_answer

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS_DIR / "math_thinking_mode_scaleup.jsonl"


def extract_think_and_answer(response_text: str) -> tuple[str, str, int]:
    """Extract content inside <think>...</think> and outside it, plus token/char length."""
    think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    if think_match:
        think_text = think_match.group(1).strip()
        ans_text = response_text[think_match.end():].strip()
    else:
        think_text = ""
        ans_text = response_text.strip()
    return think_text, ans_text, len(think_text.split())


def run_thinking_scaleup(n_problems: int = 15, max_new: int = 1500, budget: int = 256, residual: int = 64):
    print(f"=== Running Scaled-up Native Thinking Benchmark ({n_problems} Problems, Budget={budget}) ===")
    patch_qwen3_evict_sdpa()

    model_id = "Qwen/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()

    data_path = ROOT / "data" / "math_benchmarks.jsonl"
    all_items = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    math_problems = [x for x in all_items if x["source"] == "MATH-500"][:n_problems]

    modes = ["dense", "snapkv_pp", "valuediff_pp", "adaptive_vacor"]

    # Resume from existing runs
    done_set = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done_set.add((rec["problem_id"], rec["mode"]))

    print(f"Loaded {len(math_problems)} problems. {len(done_set)} problem-mode pairs already completed.")

    for p_idx, p in enumerate(math_problems):
        pid = p["id"]
        q_text = p["question"]
        gt_answer = p["ground_truth"]
        subject = p.get("subject", "Math")
        level = p.get("level", 3)

        messages = [{"role": "user", "content": q_text}]
        formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = tok(formatted, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        for mode in modes:
            if (pid, mode) in done_set:
                continue

            torch.manual_seed(42 + p_idx)
            if mode == "dense":
                cache = None
            else:
                cache = UnifiedEvictCache(
                    config=model.config,
                    token_budget=budget,
                    residual_length=residual,
                    eviction_mode=mode,
                    seed=42 + p_idx,
                    track_positions=True,
                )

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    past_key_values=cache,
                    max_new_tokens=max_new,
                    do_sample=False,
                )
            elapsed = time.time() - t0
            gen_ids = out[0][prompt_len:]
            gen_len = len(gen_ids)
            resp = tok.decode(gen_ids, skip_special_tokens=False)

            think_body, answer_body, think_word_cnt = extract_think_and_answer(resp)
            pred_raw = extract_boxed_robust(resp)

            is_correct = False
            if pred_raw is not None and gt_answer is not None:
                norm_pred = normalize_math_answer(pred_raw)
                norm_gt = normalize_math_answer(gt_answer)
                is_correct = (norm_pred == norm_gt or pred_raw.strip() == gt_answer.strip())

            record = {
                "problem_id": pid,
                "subject": subject,
                "level": level,
                "mode": mode,
                "gen_tokens": gen_len,
                "think_tokens_approx": think_word_cnt,
                "elapsed_sec": round(elapsed, 1),
                "tokens_per_sec": round(gen_len / max(elapsed, 1e-4), 2),
                "pred": pred_raw,
                "gt": gt_answer,
                "is_correct": bool(is_correct),
                "has_think_tags": bool("<think>" in resp and "</think>" in resp),
                "eviction_rounds": max(0, (gen_len - budget) // residual) if mode != "dense" else 0,
            }

            # Incrementally append to file
            with OUT_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            done_set.add((pid, mode))

            status_str = "OK" if is_correct else "FAIL"
            print(f"[{pid} | {subject[:10]}] {mode:15s} | {gen_len} tok ({record['eviction_rounds']} evicts) | {elapsed:.1f}s | {status_str}", flush=True)

    print(f"\nCompleted thinking benchmark scale-up! Saved to {OUT_PATH}\n")


if __name__ == "__main__":
    run_thinking_scaleup(n_problems=10, max_new=1200, budget=256)
