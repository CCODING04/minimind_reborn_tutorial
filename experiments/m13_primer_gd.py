x, lr = 2.0, 0.1                       # 出发点 x=2，学习率 0.1
for step in range(1, 4):
    grad = 2 * x                       # y=x² 的导数：脚下的坡度
    x = x - lr * grad                  # 沿坡度反方向迈一步
    print(f"step {step}: 坡度 {grad:+.2f} → x = {x:.4f}")
# step 1: 坡度 +4.00 → x = 1.6000
# step 2: 坡度 +3.20 → x = 1.2800
# step 3: 坡度 +2.56 → x = 1.0240   ← 每步离谷底 x=0 更近，且越近坡越缓、步子自动变小
