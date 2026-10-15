"""复现 fd9fdc8 修复前的 Trainer 行为：把 DDP.no_sync 猴子补丁成 nullcontext ——
每个 micro step 的 backward 都触发 allreduce（修复前 = 没有 no_sync 这回事）。"""
import runpy
import sys
from contextlib import nullcontext

from torch.nn.parallel.distributed import DistributedDataParallel

DistributedDataParallel.no_sync = lambda self: nullcontext()

sys.argv = ["tools/train.py"] + sys.argv[1:]
runpy.run_path("tools/train.py", run_name="__main__")
