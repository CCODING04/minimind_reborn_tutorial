from minimind_reborn.inference.chat import render_chat
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("assets/tokenizer")
u = [{"role": "user", "content": "你好"}]

for tag, kwargs in [
    ("A 默认 add_generation_prompt=True（推理侧）", {}),
    ("B add_generation_prompt=False（训练侧）", {"add_generation_prompt": False}),
    ("D open_thinking=True（think 留白，M20 再见）", {"open_thinking": True}),
]:
    text = render_chat(tok, u, **kwargs)
    print(tag)
    print("  ", repr(text), f"→ {len(tok(text).input_ids)} token")

hist = u + [
    {"role": "assistant", "content": "我是 minimind。"},
    {"role": "user", "content": "你会什么？"},
]
print("C 带 history（旧轮次原样铺在前面，结尾仍递话头）")
print("  ", repr(render_chat(tok, hist)))
