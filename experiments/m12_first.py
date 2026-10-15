"""M12 实验：sft_t2t_mini.jsonl 第一条 → 渲染 → encode → 逐段标注哪段 -100。"""
import json

from transformers import AutoTokenizer

from minimind_reborn.data.datasets import encode_sft_sample, render_conversations
from minimind_reborn.data.loss_mask import build_labels, pattern_ids, scan_response_spans

tok = AutoTokenizer.from_pretrained("assets/tokenizer")
line = open("/home/admin02/Code/WorkSpace/minimind-reproduce/dataset/sft_t2t_mini.jsonl", "rb").readline()
convs = json.loads(line)["conversations"]

text = render_conversations(tok, convs, remove_empty_think=False)  # 固定抽签：空 think 原样保留
ids = tok(text).input_ids
prefix_ids, suffix_ids = pattern_ids(tok)
spans, dangling = scan_response_spans(ids, prefix_ids, suffix_ids)
labels = build_labels(ids, spans)

supervised = sum(1 for l in labels if l != -100)
print(f"渲染 {len(text)} 字符 → {len(ids)} token；{len(spans)} 个 assistant 区间，dangling={dangling}")
print(f"监督 token {supervised}/{len(ids)} = {supervised/len(ids):.0%}；区间：{spans}")

def brief(s):  # 段内文本截短展示（数据原样，只是打印截断）
    return (s[:38] + "…") if len(s) > 40 else s

segs, cur, start = [], labels[0] != -100, 0       # 连续同色段合并
for j in range(1, len(labels) + 1):
    hot = j < len(labels) and labels[j] != -100
    if j == len(labels) or hot != cur:
        segs.append((cur, start, j)); start, cur = j, hot
for hot, a, b in segs:
    print(f"[{a:3},{b:3}) {'监督' if hot else '忽略'} {brief(tok.decode(ids[a:b]))!r}")

out = encode_sft_sample(tok, prefix_ids, suffix_ids, convs, 256, remove_empty_think=False)
print("同一样本 C1 演示（预算 256）：", out["stats"])
