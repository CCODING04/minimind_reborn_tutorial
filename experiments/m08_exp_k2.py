"""M08 实验 C：num_experts_per_tok=2 时的路由与激活（变式练习答案素材）。"""
import torch
import torch.nn.functional as F

from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.feedforward import MOEFeedForward

torch.manual_seed(0)
cfg = MiniMindConfig(use_moe=True, num_experts_per_tok=2)
moe = MOEFeedForward(cfg).train()
print(f"k=2: 每个 token 选 2 个专家，norm_topk_prob 重归一化")

x = torch.randn(2, 16, 768)
x_flat = x.view(-1, 768)
scores = F.softmax(moe.gate(x_flat), dim=-1)
topk_w, topk_idx = torch.topk(scores, k=2, dim=-1, sorted=False)
w_norm = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)     # 源码同款重归一化
print("token0 路由分数:", [round(v, 3) for v in scores[0].tolist()])
print("token0 选中:", topk_idx[0].tolist(), " 原始权重:", [round(v, 3) for v in topk_w[0].tolist()],
      " 重归一化:", [round(v, 3) for v in w_norm[0].tolist()], " 和=", round(w_norm[0].sum().item(), 6))
load = F.one_hot(topk_idx, 4).float().mean((0, 1)).flatten()   # 每 token 投 2 票，共 64 票
print("load（每专家得票占比）:", [round(v, 3) for v in load.tolist()],
      " 和=", round(load.sum().item(), 3), "（每个 token 投 2 票）")

called = []
for i, e in enumerate(moe.experts):
    e.register_forward_hook(lambda mod, inp, out, idx=i: called.append(idx))
moe.eval()
with torch.no_grad():
    _ = moe(torch.randn(1, 1, 768))
print(f"单 token 前向实际调用的专家: {sorted(called)}")
gate_n = sum(p.numel() for p in moe.gate.parameters())
expert_n = sum(p.numel() for p in moe.experts[0].parameters())
total = sum(p.numel() for p in moe.parameters())
print(f"每 token 激活: gate+2专家 = {gate_n + 2*expert_n:,}（k=1 时 {gate_n + expert_n:,}）"
      f" 占每层总参数 {100*(gate_n+2*expert_n)/total:.1f}%")
