"""M08 实验 D：DDP 下未使用参数——真报错，以及 0·Σp 挂图技巧如何消掉它（单进程 world=1）。"""
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29517")
dist.init_process_group("gloo", rank=0, world_size=1)
torch.manual_seed(0)


class BiasedMoE(nn.Module):
    """玩具版 MoE：路由结果全部落专家 0（模拟偏心），专家 1 本步收不到任何 token。"""

    def __init__(self, d=16, hang=False):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.hang = hang

    def forward(self, x):                      # x: (T, d)
        y = self.experts[0](x)                 # 专家 1 本步未被路由到
        if self.hang:                          # reborn 的挂图技巧：0 乘也建立梯度依赖
            y[0, 0] += 0 * sum(p.sum() for p in self.experts[1].parameters())
        return y


for hang in (False, True):
    ddp = DDP(BiasedMoE(hang=hang))
    label = "无挂图技巧       " if not hang else "y[0,0]+=0·Σp 挂图"
    try:
        for step in range(2):                  # 第 1 步能过，第 2 步撞上"未使用参数"检查
            loss = ddp(torch.randn(8, 16)).pow(2).sum()
            loss.backward()
        print(f"{label}: 两步都通过")
    except RuntimeError as e:
        print(f"{label}: RuntimeError: {str(e).splitlines()[0]}")
