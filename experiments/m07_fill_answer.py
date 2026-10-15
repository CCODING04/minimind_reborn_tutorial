"""M07 填空自测：把三处空填上后，手动路径③应与 SDPA is_causal（路径①）逐元素一致。"""
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, H, T, D = 1, 2, 5, 4                       # batch、头数、序列长、head_dim
q = torch.randn(B, H, T, D)                   # (B, H, T, D)
k = torch.randn(B, H, T, D)
v = torch.randn(B, H, T, D)

# —— 参照答案：路径① SDPA is_causal（训练全段）——
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# —— 你的路径③：手动显式打分（三处空）——
scores = (q @ k.transpose(-2, -1)).float() / math.sqrt(D)                    # 空① 缩放
scores += torch.full((T, T), float("-inf")).triu(1)                     # 空② 因果 mask
out = F.softmax(scores, dim=-1).type_as(q) @ v                     # 空③ softmax 维度

print("out.shape =", tuple(out.shape), " ref.shape =", tuple(ref.shape))
print("allclose(atol=1e-6):", torch.allclose(out, ref, atol=1e-6),
      " max|Δ| =", (out - ref).abs().max().item())
