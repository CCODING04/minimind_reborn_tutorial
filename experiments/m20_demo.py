from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.sampling import (
    apply_temperature, apply_repetition_penalty, apply_top_k, apply_top_p, sample_token,
)
# M20 五连纯函数数值例子：每个函数一个最小输入，打印前后对照（可独立复算）
import torch

torch.manual_seed(7)

print("== 1 apply_temperature ==")
logits = torch.tensor([[1.0, 2.0, 4.0]])           # (1, 3)：三个候选的打分
print("T=0.5 :", apply_temperature(logits, 0.5).tolist())   # 除以 0.5 = 放大 2 倍 → 更"尖"
print("T=2.0 :", apply_temperature(logits, 2.0).tolist())   # 除以 2 = 压平 → 更"平"
print("T=0   :", apply_temperature(logits, 0.0).tolist())   # ≤0 原样返回（贪心交给 sample_token）

print("== 2 apply_repetition_penalty（正除负乘）==")
logits = torch.tensor([[2.0, -2.0, 3.0, 0.0]])    # 候选 0/1 已出现过
seen = torch.tensor([[0, 1]])
out = apply_repetition_penalty(logits.clone(), seen, penalty=2.0)
print("penalty=2.0 :", out.tolist())              # 正 2→1（除），负 -2→-4（乘），未见的 3/0 不动

print("== 3 apply_top_k ==")
logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
print("k=2 :", apply_top_k(logits.clone(), 2).tolist())     # 只留前 2 大，其余 -inf

print("== 4 apply_top_p（右移累积掩码）==")
logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))  # softmax 后恰好 [0.5,0.3,0.15,0.05]
out = apply_top_p(logits.clone(), 0.9)
kept = (out != float("-inf")).squeeze(0).tolist()
kept_mass = torch.softmax(torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05])), -1)[kept].sum().item()
print(f"p=0.9 保留前 {sum(kept)} 个（累积 0.5/0.8/0.95 跨过 0.9 的第 3 个也留下），保留质量={kept_mass:.2f}")

print("== 5 sample_token（贪心 vs 抽签）==")
logits = torch.tensor([[1.0, 3.0, 2.0]])
greedy = sample_token(logits, GenerateConfig(temperature=0.0))
draw = [int(sample_token(logits, GenerateConfig(temperature=1.0))[0, 0]) for _ in range(20)]
print("T=0 贪心 :", greedy.squeeze(0).tolist(), "（必选第一名 id=1）")
print("T=1 抽 20 次:", draw, "（按概率出现，绝大多数是 1，偶尔 2）")
