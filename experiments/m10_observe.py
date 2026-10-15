from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("assets/tokenizer")  # reborn 词表加载点（engine.py 同款）
texts = ["你好，世界", "Hello, world!", "def add(a, b): return a + b", "龘", "你好🤖"]
for t in texts:
    ids = tok(t).input_ids
    pieces = [tok.decode([i]) for i in ids]             # 每个 token 单独还原成文本
    print(f"{t!r:30} {len(t)}字 {len(ids):2d} token  {pieces}")
