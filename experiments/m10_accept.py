"""M10 验收：encode→decode 完全还原 + 模板锚点（与 test_loss_mask 同机制）。"""
from minimind_reborn.data.loss_mask import pattern_ids, find_response_spans
from minimind_reborn.inference.chat import render_chat
from minimind_reborn.constants import ASSISTANT_PREFIX_TEXT, RESPONSE_SUFFIX_TEXT
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("assets/tokenizer")

# ① 往返一致性：encode → decode 必须完全还原（含生僻字/emoji 的字节回退）
texts = ["你好，世界", "Hello, world!", "def add(a, b): return a + b", "龘", "你好🤖"]
for t in texts:
    ids = tok(t).input_ids
    assert tok.decode(ids) == t, f"往返失败：{t!r}"
    print(f"OK 往返 {len(ids):2d} token  {t!r}")

# ② 模板锚点：渲染产物里 prefix/suffix 必须以完整 token 序列出现（loss mask 的地基）
prefix_ids, suffix_ids = pattern_ids(tok)
assert tok.decode(prefix_ids) == ASSISTANT_PREFIX_TEXT
assert tok.decode(suffix_ids) == RESPONSE_SUFFIX_TEXT
print(f"OK prefix={prefix_ids} {ASSISTANT_PREFIX_TEXT!r}")
print(f"OK suffix={suffix_ids} {RESPONSE_SUFFIX_TEXT!r}")

messages = [
    {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"},
    {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"},
]
ids = tok(render_chat(tok, messages, add_generation_prompt=False)).input_ids
spans = find_response_spans(ids, prefix_ids, suffix_ids)
assert len(spans) == 2, spans
print(f"OK 两轮渲染扫描出 {len(spans)} 个 assistant 区间")
print("ALL PASS")
