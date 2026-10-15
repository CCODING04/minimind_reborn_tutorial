"""M08 实验 B：use_moe 参数量对比——总参数 vs 每 token 激活参数（真实 768 配置）。"""
import torch

torch.manual_seed(0)

from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.feedforward import FeedForward, MOEFeedForward
from minimind_reborn.models.model import MiniMindForCausalLM

n = lambda m: sum(p.numel() for p in m.parameters())
cfg = MiniMindConfig()
ffn, expert = n(FeedForward(cfg)), n(FeedForward(cfg))     # 同构：一个专家 = 一份 dense FFN
moe = MOEFeedForward(MiniMindConfig(use_moe=True))
gate = n(moe.gate)                                          # router：768×4
d = n(MiniMindForCausalLM(cfg))
m = n(MiniMindForCausalLM(MiniMindConfig(use_moe=True)))
print(f"dense FFN 参数（3 矩阵 768×2432）: {ffn:,}")
print(f"单个专家参数（同为 3×768×2432）  : {expert:,}")
print(f"MoE 每层总参数（gate + 4 专家）  : {n(moe):,}  = gate {gate:,} + 4×{expert:,}")
print(f"MoE 每 token 激活（gate + 1 专家）: {gate + expert:,}  → 占总参数 {100*(gate+expert)/n(moe):.1f}%")

called = []
for i, e in enumerate(moe.experts):         # forward hook：模块被真调用时记一笔
    e.register_forward_hook(lambda mod, inp, out, idx=i: called.append(idx))
moe.eval()
with torch.no_grad():
    _ = moe(torch.randn(1, 1, 768))          # 只喂 1 个 token：只应激活 1 个专家
print(f"单 token 前向实际调用的专家: {sorted(called)}  （其余专家的参数一次都没碰）")

print(f"全模型 dense: {d:,}（{d/1e6:.2f}M）")
print(f"全模型 MoE : {m:,}（{m/1e6:.2f}M）  总参数 {m/d:.2f}×dense")
act = d + 8 * gate   # 每层：dense FFN 换成 gate + 1 专家（专家与 dense FFN 同构，净增仅 gate）
print(f"MoE 每 token 激活参数 ≈ {act:,}（{act/1e6:.2f}M）  ≈ dense 的 {act/d:.3f}×")
