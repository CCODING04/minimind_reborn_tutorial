import torch
g = torch.tensor([2.0e-8], device="cuda")      # 一条典型的小梯度（fp32 里安然无恙）
print("fp32      : %.3e" % g.item())
print("fp16 直连 :", g.half().item(), " ← 下溢为 0")
print("bf16 直连 : %.3e" % g.bfloat16().item(), " ← 保住（指数位与 fp32 同宽）")
print("fp16×65536: %.4f" % (g * 65536).half().item(), " ← GradScaler 的活：先放大再表示")
s = torch.amp.GradScaler(device="cuda")
print("scaler 初始 scale =", s.get_scale(), "（之后按 inf/nan 探测动态升降）")
