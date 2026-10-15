import re
from collections import OrderedDict
from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.model import MiniMindForCausalLM

model = MiniMindForCausalLM(MiniMindConfig())   # 64M 默认结构：768 宽 × 8 层
groups: "OrderedDict[str, list]" = OrderedDict()
for name, p in model.named_parameters():
    key = name.removeprefix("model.")                 # embed_tokens.weight / layers.0.… / norm.weight / lm_head.weight
    key = re.sub(r"layers\.\d+", "layers.N", key)     # 8 层逐层归并到同一行
    groups.setdefault(key, [0, 0])
    groups[key][0] += p.numel()                       # 该行总参数
    groups[key][1] += 1                               # 出现的 key 数

total = sum(v[0] for v in groups.values())
for k, (n, c) in groups.items():
    tag = "" if c == 1 else f"  ← ×{c} 层"
    print(f"{k:44s}{n:>12,}{tag}")
print(f"{'合计':44s}{total:>12,}   = {total/1e6:.2f}M")
print("lm_head.weight 单独计账了吗：", "lm_head.weight" in groups)
