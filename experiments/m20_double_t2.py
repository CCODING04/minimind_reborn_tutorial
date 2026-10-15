"""M20 双重温度插桩（2026-10-06 修复后）：apply_temperature 每步恰好一次。"""
import torch
import minimind_reborn.inference.generator as gen_mod
import minimind_reborn.inference.sampling as samp_mod
from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.model import MiniMindForCausalLM

CALLS = {"n": 0}
_orig = samp_mod.apply_temperature
def counting(logits, temperature):
    CALLS["n"] += 1
    return _orig(logits, temperature)
samp_mod.apply_temperature = counting    # 修复后 generator 已无外层调用，唯一点在 sample_token 内

torch.manual_seed(0)
cfg = MiniMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, vocab_size=6400, dropout=0.0)
model = MiniMindForCausalLM(cfg).eval()
g = torch.Generator().manual_seed(7)
out = gen_mod.generate(model, torch.zeros(1, 4, dtype=torch.long),
                       GenerateConfig(temperature=0.85, max_new_tokens=2),
                       eos_token_id=None, pad_token_id=0, generator=g)
print(f"generate 2 步 · apply_temperature 调用次数 = {CALLS['n']}")
