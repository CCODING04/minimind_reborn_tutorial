"""M08 实验 A：路由分布 / aux loss / top-1 直通估计器。真实 768 配置。"""
import torch
import torch.nn.functional as F

from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.feedforward import MOEFeedForward

torch.manual_seed(0)
cfg = MiniMindConfig(use_moe=True)          # 768 hidden、4 专家、k=1（默认）
moe = MOEFeedForward(cfg).train()
print(f"config: experts={cfg.num_experts} k={cfg.num_experts_per_tok} "
      f"moe_intermediate={cfg.moe_intermediate_size} norm_topk={cfg.norm_topk_prob} coef={cfg.router_aux_loss_coef}")

x = torch.randn(2, 16, 768)                 # (B, T, C)：32 个 token
x_flat = x.view(-1, 768)
with torch.no_grad():
    scores = F.softmax(moe.gate(x_flat), dim=-1)   # (32, 4) 路由分布
    topk_w, topk_idx = torch.topk(scores, k=1, dim=-1)
    load = F.one_hot(topk_idx, 4).float().mean(0)  # 每位专家分到的 token 比例
    importance = scores.mean(0)                     # 平均路由分数
print("随机初始化 router——load（每专家 token 占比）:", [round(v, 3) for v in load.flatten().tolist()])
print("importance（每专家平均分数）   :", [round(v, 3) for v in importance.flatten().tolist()])
print("token0 的路由分数:", [round(v, 3) for v in scores[0].tolist()], "→ 选中专家", topk_idx[0, 0].item())

y = moe(x)                                   # 训练模式：内部记 aux_loss
print(f"aux_loss（train, 均匀路由）= {moe.aux_loss.item():.6f}")
moe.eval()
_ = moe(x)
print(f"aux_loss（eval）            = {moe.aux_loss.item():.6f}")

# ---- 偏心 router：专家 0 的打分行对齐输入的公共方向，制造"全归 0 号" ----
moe.train()
x2 = torch.randn(2, 16, 768) + 3.0           # 输入带共同的正方向（均值偏移）
x2_flat = x2.view(-1, 768)
with torch.no_grad():
    moe.gate.weight *= 0.01
    moe.gate.weight[0] = x2_flat.mean(0)     # 专家 0 的打分行 = 输入平均方向 → 对每个 token 都打高分
scores2 = F.softmax(moe.gate(x2_flat), dim=-1)
idx2 = torch.topk(scores2, 1, dim=-1)[1]
load2 = F.one_hot(idx2, 4).float().mean((0,)).flatten()
print("偏心后 load:", [round(v, 3) for v in load2.tolist()])
print("偏心后 importance:", [round(v, 3) for v in scores2.mean(0).flatten().tolist()])
y = moe(x2)
print(f"aux_loss（train, 偏心路由）= {moe.aux_loss.item():.6f}")

# ---- top-1 直通估计器：前向恒 1.0，梯度仍回 gate ----
moe2 = MOEFeedForward(cfg).train()
top1 = torch.topk(F.softmax(moe2.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]
w = top1 - top1.detach() + 1.0
print(f"直通权重前向值: 全部等于 1.0? {(w == 1.0).all().item()}  (前向数值上 weight≡1)")
moe2.gate.weight.grad = None
out = moe2(x)
out.pow(2).sum().backward()
print(f"反向后 gate.weight.grad 非零? {moe2.gate.weight.grad is not None and moe2.gate.weight.grad.abs().sum().item() > 0}")
