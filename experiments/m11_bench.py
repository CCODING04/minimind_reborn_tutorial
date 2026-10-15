"""M11 实验：同一 1.24GB pretrain_t2t_mini.jsonl，三种随机取样姿势对比。

用法：uv run python /tmp/m11_bench.py --mode naive_load|naive_scan|indexed
每个模式独立进程跑（/usr/bin/time -v 采峰值 RSS），输出机器可读行。
"""
from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from pathlib import Path

PATH = Path.home() / "Code/WorkSpace/minimind-reproduce/dataset/pretrain_t2t_mini.jsonl"
N_SAMPLE = 1000
TARGET = 100_000


def peak_mb() -> float:
    # Linux: ru_maxrss 单位 KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def bench_access(fn, n_lines: int) -> float:
    rng = random.Random(42)
    idxs = rng.sample(range(n_lines), N_SAMPLE)
    t0 = time.perf_counter()
    for i in idxs:
        fn(i)
    return (time.perf_counter() - t0) / N_SAMPLE * 1e6  # µs/次


_p = argparse.ArgumentParser()
_p.add_argument("--mode", required=True, choices=["naive_load", "naive_scan", "indexed"])
mode = _p.parse_args(sys.argv[1:]).mode

if mode == "naive_load":
    base = peak_mb()
    t0 = time.perf_counter()
    with PATH.open("rb") as f:
        data = [json.loads(line) for line in f if line.strip()]
    load_s = time.perf_counter() - t0
    print(f"[naive_load] 行数 {len(data)}")
    print(f"[naive_load] 全量解析耗时 {load_s:.2f}s")
    print(f"[naive_load] 峰值内存 {peak_mb():.0f}MB（进程基线 {base:.0f}MB → 数据净增 {peak_mb()-base:.0f}MB）")
    us = bench_access(lambda i: data[i]["text"][:40], len(data))
    print(f"[naive_load] 随机取样 {us:.2f}µs/次（{N_SAMPLE} 次平均）")
    print(f"[naive_load] 第 {TARGET} 行开头：{data[TARGET]['text'][:36]!r}")

elif mode == "naive_scan":
    def scan_get(i: int) -> dict:
        with PATH.open("rb") as f:  # 没有 offset：只能从头数行
            for n, line in enumerate(f):
                if n == i:
                    return json.loads(line)
        raise IndexError

    row = scan_get(TARGET)
    t0 = time.perf_counter()
    scan_get(TARGET)
    dt = time.perf_counter() - t0
    print(f"[naive_scan] 不预载、直接取第 {TARGET} 行：{dt*1000:.0f}ms/次（需从头扫过前 {TARGET} 行）")
    print(f"[naive_scan] 第 {TARGET} 行开头：{row['text'][:36]!r}")
    print(f"[naive_scan] 峰值内存 {peak_mb():.0f}MB（单行缓存，与文件大小无关）")

elif mode == "indexed":
    from minimind_reborn.data.datasets import JsonlIndexedDataset

    base = peak_mb()
    t0 = time.perf_counter()
    ds = JsonlIndexedDataset(PATH)
    build_s = time.perf_counter() - t0
    print(f"[indexed] 行数 {len(ds)}（与 wc -l 一致）")
    print(f"[indexed] 索引构建 {build_s:.2f}s（一次 O(N) 顺序扫描，只记偏移不解析 json）")
    print(f"[indexed] 峰值内存 {peak_mb():.0f}MB（进程基线 {base:.0f}MB → 索引净增 {peak_mb()-base:.0f}MB）")
    us = bench_access(lambda i: ds.load_line(i)["text"][:40], len(ds))
    print(f"[indexed] 随机取样 {us:.1f}µs/次（{N_SAMPLE} 次平均：open+seek+readline+json）")
    t0 = time.perf_counter()
    row = ds.load_line(TARGET)
    print(f"[indexed] 取第 {TARGET} 行 {(time.perf_counter()-t0)*1e6:.0f}µs：{row['text'][:36]!r}")
