from transformers import AutoConfig, AutoModelForCausalLM
import torch

c2 = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B")
c3 = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")

print("Qwen2.5-1.5B architect:", c2.architectures)
print("Qwen2.5-1.5B qk_norm in config?:", "qk_norm" in str(c2.to_dict()))

print("Qwen3-1.7B architect:", c3.architectures)
print("Qwen3-1.7B qk_norm in config?:", "qk_norm" in str(c3.to_dict()))

m2 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.bfloat16, device_map="cpu")
m3 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16, device_map="cpu")

l2_attn = m2.model.layers[0].self_attn
l3_attn = m3.model.layers[0].self_attn

print("Qwen2.5 layer 0 self_attn attributes:", [k for k in dir(l2_attn) if "norm" in k])
print("Qwen3 layer 0 self_attn attributes:", [k for k in dir(l3_attn) if "norm" in k])
