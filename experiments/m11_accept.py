"""M11 验收：同一行，索引 O(1) 取 与 顺序扫描 取，必须逐字节等价。"""
import json
import time
from pathlib import Path

from minimind_reborn.data.datasets import JsonlIndexedDataset

PATH = Path.home() / "Code/WorkSpace/minimind-reproduce/dataset/pretrain_t2t_mini.jsonl"
TARGET = 100_000

t0 = time.perf_counter()
ds = JsonlIndexedDataset(PATH)
print(f"OK 索引构建 {time.perf_counter()-t0:.2f}s，行数 {len(ds)}")

t0 = time.perf_counter()
row = ds.load_line(TARGET)  # O(1)：seek(offset) + readline
dt_idx = time.perf_counter() - t0

with PATH.open("rb") as f:  # O(N) 对照：从头数行
    for n, line in enumerate(f):
        if n == TARGET:
            row_scan = json.loads(line)
            break
dt_scan = time.perf_counter() - t0

assert row == row_scan, "两种姿势取到的行必须一致"
text = row["text"][:30]
print(f"OK 第 {TARGET} 行一致：{text!r}")
print(f"OK 索引 {dt_idx*1e6:.0f}µs vs 顺序扫描 {dt_scan*1000:.0f}ms（{dt_scan/dt_idx:.0f} 倍）")
print("ALL PASS")
