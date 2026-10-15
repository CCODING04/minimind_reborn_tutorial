import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("/tmp/m21_qwen3")
model = AutoModelForCausalLM.from_pretrained("/tmp/m21_qwen3", dtype=torch.float32).eval()  # CPU 上 fp32 最稳
prompt = tok.apply_chat_template([{"role": "user", "content": "用一句话介绍你自己"}], tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt")
torch.manual_seed(42)
out = model.generate(**ids, max_new_tokens=24, do_sample=True, temperature=0.8, top_p=0.9,
                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
print("架构:", model.config.architectures, "| 参数:", f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print("生成:", tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
