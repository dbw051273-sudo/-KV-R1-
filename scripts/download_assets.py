import os
import json
from pathlib import Path
from collections import defaultdict
import huggingface_hub
from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import AutoTokenizer

def check_cached_models():
    hub_cache = Path.home() / ".cache" / "huggingface" / "hub"
    print("=" * 60)
    print(f"Scanning Hugging Face Hub Cache: {hub_cache}")
    print("=" * 60)
    if not hub_cache.exists():
        print("HF Hub cache directory does not exist yet.")
        return []
    
    qwen_models = []
    for item in hub_cache.iterdir():
        if item.is_dir() and "qwen" in item.name.lower():
            qwen_models.append(item.name)
            # Check size
            total_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
            print(f"  Found: {item.name} ({total_size / (1024**2):.2f} MB)")
            
    if not qwen_models:
        print("  No cached Qwen models found.")
    return qwen_models

def download_qwen_model():
    model_id = "Qwen/Qwen3-1.7B"
    print("\n" + "=" * 60)
    print(f"Downloading Model & Tokenizer: {model_id}")
    print("=" * 60)
    
    # Download full snapshot (weights, configs, tokenizers)
    snapshot_path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        resume_download=True
    )
    print(f"Snapshot successfully downloaded to: {snapshot_path}")
    
    # Verify tokenizer loads cleanly
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Tokenizer loaded successfully. Vocab size: {len(tokenizer)}")
    return snapshot_path

def build_benchmark_dataset():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "math_benchmarks.jsonl"
    
    print("\n" + "=" * 60)
    print("Downloading & Stratifying Math Benchmarks...")
    print("=" * 60)
    
    # 1. Load MATH-500
    print("Loading HuggingFaceH4/MATH-500 (test split)...")
    math500_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"Loaded MATH-500 with {len(math500_ds)} total items.")
    
    # Stratify by level (1 to 5)
    by_level = defaultdict(list)
    for idx, item in enumerate(math500_ds):
        lvl = item.get("level", "Unknown")
        by_level[lvl].append(item)
    
    selected_math500 = []
    # Target 60 items: 12 from each of Level 1, 2, 3, 4, 5
    target_levels = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
    for lvl in target_levels:
        items = by_level.get(lvl, [])
        # Sample 12 items uniformly across the level
        step = max(1, len(items) // 12)
        sampled = items[::step][:12]
        if len(sampled) < 12 and len(items) >= 12:
            sampled = items[:12]
        for s in sampled:
            selected_math500.append({
                "id": f"math500_{len(selected_math500)+1:03d}",
                "source": "MATH-500",
                "level": lvl,
                "subject": s.get("subject", "General Math"),
                "question": s.get("problem", ""),
                "ground_truth": s.get("answer", "")
            })
            if len(selected_math500) == 60:
                break
        if len(selected_math500) == 60:
            break
            
    # If not quite 60, fill from remaining
    if len(selected_math500) < 60:
        seen_questions = {x["question"] for x in selected_math500}
        for item in math500_ds:
            if item["problem"] not in seen_questions:
                selected_math500.append({
                    "id": f"math500_{len(selected_math500)+1:03d}",
                    "source": "MATH-500",
                    "level": item.get("level", "Unknown"),
                    "subject": item.get("subject", "General Math"),
                    "question": item.get("problem", ""),
                    "ground_truth": item.get("answer", "")
                })
                seen_questions.add(item["problem"])
            if len(selected_math500) == 60:
                break

    print(f"Stratified {len(selected_math500)} problems from MATH-500.")

    # 2. Load GSM8K
    print("Loading openai/gsm8k (main, test split)...")
    gsm8k_ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded GSM8K with {len(gsm8k_ds)} total items.")
    
    selected_gsm8k = []
    # Target 40 items: uniformly sample 40 from 1319 test items
    step = len(gsm8k_ds) // 40
    for i in range(40):
        item = gsm8k_ds[i * step]
        # Extract ground truth after '#### '
        raw_ans = item["answer"]
        gt = raw_ans.split("####")[-1].strip() if "####" in raw_ans else raw_ans.strip()
        selected_gsm8k.append({
            "id": f"gsm8k_{i+1:03d}",
            "source": "GSM8K",
            "level": "Grade School",
            "subject": "Word Problem Arithmetic",
            "question": item["question"],
            "ground_truth": gt
        })
    print(f"Sampled {len(selected_gsm8k)} problems from GSM8K.")

    all_benchmarks = selected_math500 + selected_gsm8k
    assert len(all_benchmarks) == 100, f"Expected 100 problems, got {len(all_benchmarks)}"

    with open(out_path, "w", encoding="utf-8") as f:
        for entry in all_benchmarks:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"\nSaved {len(all_benchmarks)} stratified problems to: {out_path}")
    print("Distribution:")
    counts = defaultdict(int)
    for b in all_benchmarks:
        counts[f"{b['source']} - {b['level']}"] += 1
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")

def main():
    check_cached_models()
    download_qwen_model()
    build_benchmark_dataset()
    print("\nAll assets downloaded and prepared successfully!")

if __name__ == "__main__":
    main()
