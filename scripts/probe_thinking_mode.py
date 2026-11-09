"""Probe Qwen3 thinking mode and token generation lengths."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def probe():
    model_id = "Qwen/Qwen3-1.7B"
    print(f"Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    prompt = "Problem: What is the least positive integer multiple of 30 that can be written with only the digits 0 and 2?"
    
    # Test 1: Non-thinking mode
    messages_non_think = [{"role": "user", "content": prompt}]
    text_non_think = tok.apply_chat_template(messages_non_think, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs_nt = tok(text_non_think, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        out_nt = model.generate(**inputs_nt, max_new_tokens=512, do_sample=False)
    resp_nt = tok.decode(out_nt[0][inputs_nt.input_ids.shape[1]:], skip_special_tokens=False)
    print(f"Non-thinking tokens: {out_nt.shape[1] - inputs_nt.input_ids.shape[1]}")
    print(f"Non-thinking output snippet: {resp_nt[:150]}...\n")

    # Test 2: Thinking mode
    messages_think = [{"role": "user", "content": prompt}]
    text_think = tok.apply_chat_template(messages_think, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs_t = tok(text_think, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out_t = model.generate(**inputs_t, max_new_tokens=1500, do_sample=False)
    resp_t = tok.decode(out_t[0][inputs_t.input_ids.shape[1]:], skip_special_tokens=False)
    gen_len = out_t.shape[1] - inputs_t.input_ids.shape[1]
    print(f"Thinking tokens: {gen_len}")
    print(f"Thinking output snippet: {resp_t[:250]}...\n")

if __name__ == "__main__":
    probe()
