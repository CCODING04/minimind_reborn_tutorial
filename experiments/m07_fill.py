"""M07 填空自测：把三处空填上后，手动路径③应与 SDPA is_causal（路径①）逐元素一致。"""
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, H, T, D = 1, 2, 5, 4                            # batch、头数、序列长、head_dim
q, k, v = (torch.randn(B, H, T, D) for _ in range(3))   # (B, H, T, D)

ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # 路径① 参照答案

scores = (q @ k.transpose(-2, -1)).float() / ____               # 空① 缩放
scores += torch.full((T, T), float("-inf")).____                # 空② 因果 mask
out = F.softmax(scores, dim=____).type_as(q) @ v                # 空③ softmax 维度

print("allclose(atol=1e-6):", torch.allclose(out, ref, atol=1e-6),
      " max|Δ| =", (out - ref).abs().max().item())
