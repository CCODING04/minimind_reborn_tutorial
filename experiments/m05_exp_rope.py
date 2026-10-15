import torch
from minimind_reborn.models.layers import precompute_freqs_cis, apply_rotary_pos_emb

torch.manual_seed(0)
dim, end = 8, 64
cos, sin = precompute_freqs_cis(dim, end=end)            # cos/sin: (end, dim)
q, k = torch.randn(1, 1, 1, dim), torch.randn(1, 1, 1, dim)  # (b, seq, heads, head_dim)

def qk_at(m, n):
    """q 放在位置 m、k 放在位置 n，返回旋转后的内积"""
    qr, _ = apply_rotary_pos_emb(q, k, cos[m:m+1], sin[m:m+1])  # 只转 q 那一格
    _, kn = apply_rotary_pos_emb(k, k, cos[n:n+1], sin[n:n+1])  # 只转 k 那一格
    return (qr * kn).sum().item()

print("q·k（不旋转）= %.6f" % (q * k).sum().item())
for m, n in [(2, 6), (30, 34), (57, 61)]:
    print(f"q@{m:2d} k@{n:2d}（距离 {n-m}）: {qk_at(m, n):.6f}")
