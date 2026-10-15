"""M12 变式：add_system_ratio 0.0 vs 1.0，同一行样本的标签与监督占比怎么变。"""
import json
import random

from transformers import AutoTokenizer

from minimind_reborn.data.datasets import SYSTEM_PROMPTS, render_conversations
from minimind_reborn.data.loss_mask import build_labels, pattern_ids, scan_response_spans

tok = AutoTokenizer.from_pretrained("assets/tokenizer")
line = open("/home/admin02/Code/WorkSpace/minimind-reproduce/dataset/sft_t2t_mini.jsonl", "rb").readline()
convs = json.loads(line)["conversations"]
prefix_ids, suffix_ids = pattern_ids(tok)


def probe(tag, ratio):
    random.seed(0)
    texts = set()
    for _ in range(6):  # 同一行抽 6 次，看渲染是否随抽签漂移
        c = convs
        if c[0].get("role") != "system" and random.random() < ratio:
            c = [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + c
        texts.add(render_conversations(tok, c, remove_empty_think=False))
    text = sorted(texts)[0]
    ids = tok(text).input_ids
    spans, _ = scan_response_spans(ids, prefix_ids, suffix_ids)
    labels = build_labels(ids, spans)
    sup = sum(1 for l in labels if l != -100)
    print(f"[{tag}] 渲染变体 {len(texts)} 种；全长 {len(ids)} token；监督 {sup} = {sup/len(ids):.0%}")
    print(f"    开头：{tok.decode(ids[:20])!r}")
    print(f"    前 12 个 label：{labels[:12]}")


probe("ratio=0.0", 0.0)
probe("ratio=1.0", 1.0)
