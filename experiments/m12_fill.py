"""M12 填空：把 scan_response_spans 的三个关键位补全（对照 reborn 参照答案）。"""
from transformers import AutoTokenizer

from minimind_reborn.data.loss_mask import find_response_spans, pattern_ids

tok = AutoTokenizer.from_pretrained("assets/tokenizer")
PREFIX, SUFFIX = pattern_ids(tok)  # ([1, 1388, 570, 811, 234], [2, 234])


def match_at(seq, pat, i):
    return seq[i : i + len(pat)] == pat  # 越界切片得短列表 → 自然 False


def my_scan(ids):
    """返回 (spans, dangling)：完整回答的 [start, stop)，残段只计数。"""
    spans, dangling, i, n = [], 0, 0, len(ids)
    while i < n:
        if match_at(ids, PREFIX, i):
            start = i + len(PREFIX)                             # 空①
            end = start
            while end < n and not match_at(ids, SUFFIX, end):   # 空②
                end += 1
            if end < n:
                spans.append((start, end + len(SUFFIX)))        # 空③
                i = spans[-1][1]
            else:
                dangling += 1                                   # 扫到头也没等到 suffix
                i = n
        else:
            i += 1
    return spans, dangling


msgs = [
    {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"},
    {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"},
]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
ids = tok(text).input_ids
spans_full, dg_full = my_scan(ids)
assert spans_full == find_response_spans(ids, PREFIX, SUFFIX) and dg_full == 0
spans_cut, dg_cut = my_scan(ids[:-4])
print("两轮完整：", len(spans_full), "个区间", spans_full)
print("砍掉末 4 token：", len(spans_cut), "个区间 +", dg_cut, "个残段（残段不计监督）")
print("与 reborn find_response_spans 逐位一致：ALL PASS")
