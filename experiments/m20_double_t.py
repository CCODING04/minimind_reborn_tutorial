# M20 双重温度插桩：钩住 apply_temperature 计数，复现"generate 路径上温度被应用两次"
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

# 两处引用各打一个钩：generator.py:81 的外层调用 + sampling.py:64 里 sample_token 的内层调用
gen_mod.apply_temperature = counting
samp_mod.apply_temperature = counting

torch.manual_seed(0)
cfg = MiniMindConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, vocab_size=64, max_position_embeddings=128, dropout=0.0)
model = MiniMindForCausalLM(cfg).eval()
prompt = torch.randint(0, 64, (1, 6))

gen = GenerateConfig(temperature=0.85, max_new_tokens=2)   # 与 M03 巡游同款旋钮读数
g = torch.Generator().manual_seed(123)
out = gen_mod.generate(model, prompt, gen, eos_token_id=None, pad_token_id=0, generator=g)
print(f"generate max_new_tokens=2 · apply_temperature 调用次数 = {CALLS['n']}（2 步 × 每步 2 次）")

CALLS["n"] = 0
tok = samp_mod.sample_token(torch.randn(1, 64), GenerateConfig(temperature=0.85))
print(f"sample_token 直调 1 次    · apply_temperature 调用次数 = {CALLS['n']}（单测路径只有一次——缝隙只在串联处）")

# 影响面量化：同一份 logits，T=0.85 一次 vs 两次（等效 T²=0.7225），softmax 后 top1 概率与熵
from minimind_reborn.inference.sampling import apply_temperature as real_t
torch.manual_seed(7)
logits = torch.randn(1, 64)
T = 0.85
p1 = torch.softmax(real_t(logits, T), -1)
p2 = torch.softmax(real_t(real_t(logits.clone(), T), T), -1)     # generate 路径的真实行为
top1 = p1.argmax()
ent = lambda p: -(p * p.clamp_min(1e-12).log()).sum().item()
print(f"T=0.85 → 有效温度 T²={T*T:.4f}")
print(f"一次缩放：top1(id={top1.item()}) 概率={p1[0, top1].item():.3f}  熵={ent(p1[0]):.3f}")
print(f"两次缩放：top1(id={p2.argmax().item()}) 概率={p2[0, top1].item():.3f}  熵={ent(p2[0]):.3f}")

# 贪心免疫：T≤0 时 apply_temperature 原样返回，argmax 对正数缩放也不变
greedy = GenerateConfig(temperature=0.0, max_new_tokens=2)
out_g = gen_mod.generate(model, prompt, greedy, eos_token_id=None, pad_token_id=0)
print(f"贪心路径（T=0）两次缩放都不发生，输出确定：{out_g['generated_ids'].squeeze(0).tolist()}")
