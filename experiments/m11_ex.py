"""M11 变式：改一个字节（大小不变），manifest 指纹会怎样？"""
import subprocess
import time
from pathlib import Path

from minimind_reborn.data.manifest import _fingerprint, load_or_build, manifest_path

SRC = Path.home() / "Code/WorkSpace/minimind-reproduce/dataset/sft_t2t_mini.jsonl"
SLICE = Path("/tmp/m11_slice.jsonl")

subprocess.run(["head", "-n", "20000", str(SRC)], stdout=open(SLICE, "wb"), check=True)
manifest_path(SLICE).unlink(missing_ok=True)  # 干净起点：先删旁车

t0 = time.perf_counter()
m1 = load_or_build(SLICE, eval_ratio=0.1)  # 第一次：全量扫描建旁车
print(f"首建 {time.perf_counter()-t0:.2f}s  行数 {len(m1)}  eval 占比 {m1.is_eval.mean():.3f}")
print(f"指纹 = {m1.fingerprint}")

raw = SLICE.read_bytes()  # 原地改一个字节：「。」(3字节) → 「，」(3字节)，大小不变
i = raw.rindex("。".encode())
SLICE.write_bytes(raw[:i] + "，".encode() + raw[i + 3 :])
print(f"改后 stat：size {SLICE.stat().st_size}（不变）  指纹 = {_fingerprint(SLICE)}")

t0 = time.perf_counter()
m2 = load_or_build(SLICE, eval_ratio=0.5)  # 指纹已变：必须全量重建
print(f"重取(eval_ratio=0.5) {time.perf_counter()-t0:.2f}s  eval 占比 {m2.is_eval.mean():.3f}（重建，新口径生效）")
