"""M13 教学脚本：get_lr 曲线 10 个采样点 + warmup 变式对照。"""
import math
from minimind_reborn.training.common.lr import get_lr

base, total, warmup = 5e-4, 60, 10
print(f"base_lr={base:.0e} total_steps={total} warmup_steps={warmup} min_ratio=0.1")
print(f"{'step':>4} {'progress':>8} {'lr':>10}  bar")
for step in range(0, total + 1, 6):
    lr = get_lr(step, total, base, warmup_steps=warmup, min_ratio=0.1)
    bar = "#" * int(round(lr / base * 40))
    if step < warmup:
        phase = "warmup"
    else:
        phase = "cosine"
    print(f"{step:>4} {step/total:>7.0%} {lr:>10.3e}  {bar} {phase}")

print()
print("变式：warmup_steps 2 → 10，前 6 步的 lr 对比（total=60, base=5e-4）")
print(f"{'step':>4} {'warmup=2':>11} {'warmup=10':>11}")
for step in range(6):
    a = get_lr(step, 60, 5e-4, warmup_steps=2)
    b = get_lr(step, 60, 5e-4, warmup_steps=10)
    print(f"{step:>4} {a:>11.3e} {b:>11.3e}")
