"""M07 三路径等价性验证：同权重 fp32 下，①SDPA is_causal / ②SDPA+mask+cache / ③manual 两两 allclose。

tiny 配置与 tests/test_models 的夹具同款（hidden=32、2 层、2 Q 头、1 KV 头）。
"""
import torch
import torch.nn.functional as F

from minimind_reborn.models import KVCache
from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.model import MiniMindForCausalLM

torch.manual_seed(7)
cfg = MiniMindConfig(
    hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
    vocab_size=64, max_position_embeddings=128, dropout=0.0,
)

a = MiniMindForCausalLM(cfg).eval()   # flash=True：无 cache 走①，带 cache 走②
b = MiniMindForCausalLM(cfg).eval()
b.load_state_dict(a.state_dict())
for layer in b.model.layers:
    layer.self_attn.flash = False      # 强制③ manual

# ——证据增强：数一数 SDPA 到底被谁调用——
calls = {"n": 0}
_real_sdpa = F.scaled_dot_product_attention
def counting_sdpa(*args, **kw):
    calls["n"] += 1
    return _real_sdpa(*args, **kw)
F.scaled_dot_product_attention = counting_sdpa
try:
    x = torch.randint(0, 64, (2, 7))
    with torch.no_grad():
        calls["n"] = 0
        out1 = a(x).logits                                   # ① 无 cache、无 mask
        n1 = calls["n"]
        ca = KVCache(2, 2, 24, 1, 16, torch.float32, "cpu")
        pre2 = a(x[:, :5], past_key_values=ca).logits        # ② prefill（写 5 列 cache）
        step2 = a(x[:, 5:6], past_key_values=ca).logits      # ② decode（每步 1 列）
        step3 = a(x[:, 6:7], past_key_values=ca).logits      # ② decode 第 2 步
        n2 = calls["n"] - n1
        calls["n"] = 0
        out3 = b(x).logits                                   # ③ manual 全段
        n3 = calls["n"]
    # ② 的 prefill 前 5 位 + 两步 decode 拼起来，对齐①③的 7 位输出
    out2 = torch.cat([pre2, step2, step3], dim=1)
    print(f"路径判定（SDPA 调用次数，2 层模型）: ①={n1}  ②={n2}(prefill 2 层 + 2 步 decode 各 2 层)  ③={n3}")
    for name, u, v in [("① vs ③", out1, out3), ("② vs ③", out2, out3), ("② vs ①", out2, out1)]:
        ok = torch.allclose(u, v, atol=1e-5)
        print(f"{name}  allclose={ok}  max|Δ|={(u - v).abs().max().item():.2e}")

    # —— 加 padding mask 的批次（②不带 cache / ③），对齐 test_sdpa_prefill_path_matches_manual 场景——
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
    with torch.no_grad():
        pa = a(x, attention_mask=mask).logits                # mask 含 0 → 落②
        pb = b(x, attention_mask=mask).logits                # ③
    print(f"② vs ③（含 padding mask） allclose={torch.allclose(pa, pb, atol=1e-5)}"
          f"  max|Δ|={(pa - pb).abs().max().item():.2e}")
finally:
    F.scaled_dot_product_attention = _real_sdpa
