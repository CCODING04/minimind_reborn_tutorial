import torch
torch.manual_seed(0)
eps = 1e-5
def norm(x):                                    # 与 layers.py 的 _norm 相同
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

dev = "cuda" if torch.cuda.is_available() else "cpu"
h = (torch.randn(8192, 768) * 30).to(dev)       # 模拟训练后期的 hidden（std=30），(8192, 768)
x = h.bfloat16()                                # 全 bf16 模型里 RMSNorm 的真实输入
ref    = norm(h.double()).float()               # fp64 参考答案
direct = norm(x).float()                        # 无纪律：统计量在 bf16 里算
discip = norm(x.float()).bfloat16().float()     # reborn：升 fp32 算完回落
for name, out in [("bf16 直算", direct), ("fp32 纪律", discip)]:
    print(f"{dev} · {name}: max_abs={(out-ref).abs().max():.4f} rel={(out-ref).norm()/ref.norm()*100:.3f}%")
r64 = torch.rsqrt(h.double().pow(2).mean(-1, keepdim=True) + eps)
print("scale 因子误差: bf16 %.4f%%  fp32 %.4f%%" % (
    ((torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps).double() - r64).abs() / r64).max() * 100,
    ((torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).double() - r64).abs() / r64).max() * 100))
