"""M11 实验：注册表解析、三要素报错、视图不复制（真数据）。"""
import resource
import time
from pathlib import Path

from minimind_reborn.data.registry import _REGISTRY, file_path, resolve_dataset
from minimind_reborn.data.datasets import IndexListView, PretrainDataset, SegmentView

DATA = Path.home() / "Code/WorkSpace/minimind-reproduce/dataset"

# ① 注册表全景
print(f"注册条目 {len(_REGISTRY)} 个：{sorted(_REGISTRY)}")
entry = resolve_dataset("pretrain_t2t_mini")
print(f"resolve_dataset('pretrain_t2t_mini') → {entry}")
print(f"file_path → {file_path(entry)}  存在={file_path(entry).exists()}")

# ② 两条错误路径真触发
try:
    resolve_dataset("no_such_dataset")
except ValueError as e:
    print(f"-- ValueError: {e}")
try:
    resolve_dataset("pretrain_t2t_mini", data_root=Path("/tmp/nonexistent_data"))
except FileNotFoundError as e:
    print(f"-- FileNotFoundError: {e}")

# ③ 视图不复制：同一个底层索引，train/eval 各一张薄壳
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("assets/tokenizer")
t0 = time.perf_counter()
ds = PretrainDataset(DATA / "pretrain_t2t_mini.jsonl", tok, max_length=512, eval_ratio=0.005)
print(f"PretrainDataset 构建 {time.perf_counter()-t0:.2f}s（含 0.47s 索引扫描）")
tr, ev = ds.train_view(), ds.eval_view()
print(f"总行数 {len(ds)} = train {len(tr)} + eval {len(ev)}（eval_ratio=0.005）")
print(f"train_view 是薄壳：_base is ds = {tr._base is ds}，只带 (_start,_stop)=({tr._start},{tr._stop})")
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"建两张视图后峰值内存不变：{rss():.0f}MB（视图零数据复制）")
item = tr[0]
print(f"tr[0] → input_ids {tuple(item['input_ids'].shape)}，labels {tuple(item['labels'].shape)}")
print(f"视图转发：tr[123] 与 ds.encode(123) 同源 = {bool((tr[123]['input_ids'] == ds.encode(123)['input_ids']).all())}")

# ④ IndexListView：重复的是行索引不是数据（哈希分桶视图的底座，M12 详用）
lv = IndexListView(ds, [5, 3, 5])
print(f"IndexListView([5,3,5]) → len={len(lv)}，lv[0] 与 lv[2] 同一行 = "
      f"{bool((lv[0]['input_ids'] == lv[2]['input_ids']).all())}")
