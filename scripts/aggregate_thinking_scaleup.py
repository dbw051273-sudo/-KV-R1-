import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
scaleup_path = ROOT / "results" / "math_thinking_mode_scaleup.jsonl"

lines = [json.loads(x) for x in scaleup_path.read_text(encoding="utf-8").splitlines() if x.strip()]
modes = defaultdict(list)
for r in lines:
    modes[r["mode"]].append(r)

print("=== Thinking Mode Scaleup (10 Problems, 40 Runs) ===")
summary = {}
for m, recs in modes.items():
    n = len(recs)
    correct_cnt = sum(r["is_correct"] for r in recs)
    acc = correct_cnt / n * 100
    avg_tokens = sum(r["gen_tokens"] for r in recs) / n
    avg_time = sum(r["elapsed_sec"] for r in recs) / n
    avg_evicts = sum(r["eviction_rounds"] for r in recs) / n
    summary[m] = {
        "n": n,
        "correct": correct_cnt,
        "acc": acc,
        "avg_tokens": avg_tokens,
        "avg_time": avg_time,
        "avg_evicts": avg_evicts,
    }
    print(f"{m:16s} | Acc: {acc:.1f}% ({correct_cnt}/{n}) | Avg Tok: {avg_tokens:.0f} | Evicts: {avg_evicts:.1f} | Avg Time: {avg_time:.1f}s")

summary_path = ROOT / "results" / "math_thinking_mode_scaleup_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"\nSaved summary to {summary_path}")
