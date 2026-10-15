"""M11 坑卡实测：真实 sidecar 的缓存键里没有 eval_ratio。"""
import time

from minimind_reborn.data.manifest import load_or_build
from pathlib import Path

SRC = Path.home() / "Code/WorkSpace/minimind-reproduce/dataset/sft_t2t_mini.jsonl"

t0 = time.perf_counter()
m = load_or_build(SRC, eval_ratio=0.005)  # sft_64m.yaml 的口径
print(f"load_or_build(eval_ratio=0.005) 耗时 {time.perf_counter()-t0:.2f}s（毫秒级=命中缓存，没重建）")
print(f"拿到的 eval 占比 {m.is_eval.mean():.5f}——是 smoke 的 0.001 桶，不是我要的 0.005")
