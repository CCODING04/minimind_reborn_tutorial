"""M13 教学脚本：同一训练步（前向+反向+优化）在 fp32/bf16/fp16 下的耗时对比。

走 Trainer 同款机制：resolve_dtype → autocast_context → build_scaler。
tiny = 冒烟 0.53M；64M = 官方结构（B=4, T=96 与 smoke_pretrain 一致）。
"""
import time
import torch
from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.model import MiniMindForCausalLM
from minimind_reborn.training.common.amp import autocast_context, build_scaler, resolve_dtype

TINY = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)


def bench(cfg_kwargs, dtype_name, steps=30, warmup=8, B=4, T=96):
    torch.manual_seed(0)
    model = MiniMindForCausalLM(MiniMindConfig(**cfg_kwargs)).cuda().train()
    dtype = resolve_dtype(dtype_name)
    ctx = autocast_context(dtype, "cuda")        # fp32 → nullcontext，bf16/fp16 → cuda autocast
    scaler = build_scaler(dtype)                  # fp32/bf16 → enabled=False（透传），fp16 → 真 scaler
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, 6400, (B, T), device="cuda")   # (B, T)：假语料，只测计算耗时

    def one_step():
        with ctx:
            loss = model(x, labels=x).loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    for _ in range(warmup):
        one_step()                                # CUDA 就绪 + cudnn benchmark 收敛
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / steps * 1000
    n = sum(p.numel() for p in model.parameters())
    peak = torch.cuda.max_memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    del model, opt
    torch.cuda.empty_cache()
    return n, ms, peak


print(f"{'model':>6} {'params':>9} {'dtype':>5} {'ms/step':>8} {'tok/s':>8} {'peak MB':>8}")
for tag, cfg in [("tiny", TINY), ("64M", {})]:
    for dt in ("fp32", "bf16", "fp16"):
        n, ms, peak = bench(cfg, dt)
        print(f"{tag:>6} {n/1e6:>8.2f}M {dt:>5} {ms:>8.2f} {4*96/ms*1000:>8.0f} {peak:>8.0f}")
