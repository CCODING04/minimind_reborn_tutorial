from minimind_reborn.training.common.lr import get_lr

print(f"{'step':>4} {'warmup=2':>11} {'warmup=10':>11}")
for step in range(6):
    a = get_lr(step, 60, 5e-4, warmup_steps=2)     # (step, total, base, …)
    b = get_lr(step, 60, 5e-4, warmup_steps=10)
    print(f"{step:>4} {a:>11.3e} {b:>11.3e}")
