# M03 巡游：一次请求的一生——七站各打印一行证据（权重 out/pretrain_768.pth）
import torch
from torch.nn.functional import softmax
from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.chat import render_chat, split_reasoning
from minimind_reborn.inference.engine import MiniMindLLM
from minimind_reborn.inference.sampling import apply_temperature, apply_top_k, apply_top_p
from minimind_reborn.models.kv_cache import KVCache

llm = MiniMindLLM.build("pretrain")  # ③ engine 门面：权重→模型→dtype 回退→设备
ids = llm.tokenizer("你好，世界")["input_ids"]  # ①
print(f"① encode 「你好，世界」→ {ids} → decode 还原「{llm.tokenizer.decode(ids)}」 vocab={len(llm.tokenizer)}")
print(f"② render {render_chat(llm.tokenizer, [{'role': 'user', 'content': '你好'}])!r}")
print(f"③ engine device={llm.device} dtype={llm.dtype} 参数={sum(p.numel() for p in llm.model.parameters()) / 1e6:.2f}M")
inputs = llm.tokenizer(llm.tokenizer.bos_token + "大语言模型", return_tensors="pt").to(llm.device)  # (1, 4)
gen = GenerateConfig(temperature=0.85, top_k=50, top_p=0.95, max_new_tokens=6)
with torch.inference_mode():
    out = llm.generate_tokens(inputs["input_ids"], gen)  # ④ generator 内核
    c = llm.model.config
    cache = KVCache(num_layers=c.num_hidden_layers, max_batch_size=1,  # 复刻 generator.py 的分配参数
                    max_seq_len=4 + gen.max_new_tokens, num_kv_heads=c.num_key_value_heads,
                    head_dim=c.head_dim, dtype=llm.dtype, device=llm.device)
    print(f"④ generKV 预分配 k.shape={tuple(cache.k.shape)} 生成 {out['generated_ids'].shape[1]} tokens finish={out['finish_reasons'][0]}")
    logits = llm.model(**inputs).logits[:, -1, :]  # ⑤ 一次前向 → (1, 6400)
    print(f"⑤ forward logits.shape={tuple(logits.shape)}（hidden=768 × 8 层 → 每步只留最后一位）")
    t0 = softmax(logits[0], -1).topk(5)  # ⑥ 链前
    s = apply_top_p(apply_top_k(apply_temperature(logits, gen.temperature), gen.top_k), gen.top_p)  # 顺序契约
    fmt = lambda t: [f"{i}:{v:.3f}" for i, v in zip(t.indices.tolist(), t.values.tolist())]
    print(f"⑥ sample 链前 top5 {fmt(t0)}\n          链后 top5 {fmt(softmax(s[0], -1).topk(5))}（temperature→top_k→top_p 后重归一化）")
    text = llm.tokenizer.decode(out["sequences"][0, 4:].tolist(), skip_special_tokens=True)  # ⑦ 只解码生成段
    print(f"⑦ decode sequences[0, 4:] → {text!r} → split_reasoning={split_reasoning(text)[0]!r}")
