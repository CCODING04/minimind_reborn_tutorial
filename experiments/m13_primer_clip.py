import torch
g = torch.tensor([0.3, 4.0])           # 两个参数的梯度：一个正常，一个突然"变悬崖"
print(f"裁剪前范数 {g.norm():.2f}")     # 4.01 —— 照这个总步长迈一步就是跳崖
clipped = g * 1.0 / g.norm()           # clip_grad_norm_(…, max_norm=1.0) 的等价手算
print(f"裁剪后范数 {clipped.norm():.2f}，梯度 {clipped.tolist()}")
# 裁剪后范数 1.00，方向比例仍是 0.3 : 4.0 —— 只限总步长、不改方向；
# M01 冒烟日志里的 grad_norm 列，就是这把尺子量出的"裁剪时读数"
