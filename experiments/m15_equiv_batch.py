"""等效 batch 的真算：DDP 的"两层平均"vs 单卡大 batch 的"一层平均"。
模拟 2 卡：卡 A 吃前 8 条样本，卡 B 吃后 8 条（DistributedSampler 等分片）。
"""
import torch

torch.manual_seed(42)
W = torch.randn(8, 3)            # 假装这是模型的全部参数（8 个，3 维输入）
X = torch.randn(16, 3)           # 16 条样本
y = torch.randn(16, 8)           # 每条样本的回归目标

def mean_grad(xs, ys):           # 一批样本的平均梯度（最小二乘）
    w = W.detach().clone().requires_grad_(True)
    loss = ((xs @ w.T - ys) ** 2).mean()
    loss.backward()
    return w.grad

g_big = mean_grad(X, y)                                  # 路线二：单卡 batch 16 累积 1
g_a = mean_grad(X[:8], y[:8])                            # 卡 A：8 条的平均梯度
g_b = mean_grad(X[8:], y[8:])                            # 卡 B：8 条的平均梯度
g_ddp = (g_a + g_b) / 2                                  # 路线一：all-reduce AVG（两卡等分片）

print("路线一（2 卡 ×4 batch ×2 累积 = 两层平均）与路线二（单卡 16）梯度最大差：",
      (g_ddp - g_big).abs().max().item())

# 不等分片的反例：卡 A 10 条、卡 B 6 条（手写切分没对齐时会遇到）
g_a2 = mean_grad(X[:10], y[:10])
g_b2 = mean_grad(X[10:], y[10:])
print("不等分片（10+6）时两层平均 vs 全局平均最大差：",
      (((g_a2 + g_b2) / 2) - g_big).abs().max().item())
