"""M19 观测：对一个随机 tiny 模型做一次 GRPO 式组采样（1 prompt × 4 completions）。"""
import torch
from transformers import AutoTokenizer

from minimind_reborn.inference.chat import render_chat
from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.model import MiniMindForCausalLM
from minimind_reborn.training.rollout import TorchRolloutEngine

torch.manual_seed(7)
tok = AutoTokenizer.from_pretrained("assets/tokenizer")
cfg = MiniMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
model = MiniMindForCausalLM(cfg).eval()  # 随机权重：只看结构，不看点数
engine = TorchRolloutEngine(model, tok, device="cpu")

prompt = render_chat(tok, [{"role": "user", "content": "1+1等于几？"}])
enc = tok([prompt], return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
result = engine.rollout(enc["input_ids"], enc["attention_mask"], num_generations=4, max_new_tokens=16, temperature=0.8)

print("prompt 长度:", result.prompt_lens.tolist(), "  completion_ids 形状:", tuple(result.completion_ids.shape))
print("per_token_logps 形状:", tuple(result.per_token_logps.shape), "（采样时策略对每个生成 token 的 logp）")
for i, c in enumerate(result.completions):
    n = int(result.completion_mask[i].sum())
    lp = result.per_token_logps[i, : n - 1].mean()
    print(f"[{i}] {c!r}  有效长度={n}  平均logp={lp:.3f}")
