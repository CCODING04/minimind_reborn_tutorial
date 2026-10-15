"""变式：chosen == rejected（同一文本两边）时，DPO 的 loss 与梯度是什么？"""
import torch
from minimind_reborn.training.dpo.loss import dpo_loss

torch.manual_seed(42)
# 1 对"完全相同"的回答：B=2（前半 chosen 后半 rejected），两边一字不差
pair = -torch.rand(1, 32)                          # 伪造每 token log prob ∈ (-1, 0)
logp = torch.cat([pair, pair], dim=0).requires_grad_(True)   # (2, 32)
mask = torch.ones(2, 32)

loss = dpo_loss(logp, logp, mask, beta=0.15)       # 同一张量既当 π 又当 ref
print(f"loss = {loss.item():.6f} | ln(2) = {torch.log(torch.tensor(2.0)):.6f}")
loss.backward()
print(f"grad|logp| 最大 = {logp.grad.abs().max().item():.3e}（精确为 0：{logp.grad.abs().max().item() == 0.0}）")

# 对照：只把 rejected 侧末 token 的 logp 压低 0.5 —— 偏好信号出现
ref = logp.detach()
pol = logp.detach().clone(); pol[1, -1] -= 0.5
pol.requires_grad_(True)
loss2 = dpo_loss(ref, pol, mask, beta=0.15)
loss2.backward()
print(f"对照（rejected 略差）loss = {loss2.item():.6f} < ln(2)；grad|pol| 最大 = {pol.grad.abs().max().item():.3e} ≠ 0")
