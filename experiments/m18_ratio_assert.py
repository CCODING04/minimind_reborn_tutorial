"""M18 验收：LoRA rank=16 参数占比断言（64M 主配方同款）。"""

from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.lora import apply_lora
from minimind_reborn.models.model import MiniMindForCausalLM

m = MiniMindForCausalLM(MiniMindConfig())  # 768×8 层，64M 主配方结构
base = sum(p.numel() for p in m.parameters())
apply_lora(m, rank=16)  # 与 lora_64m.yaml 同款 rank
lora = sum(p.numel() for n, p in m.named_parameters() if "lora" in n)
hooks = [n for n, mod in m.named_modules() if hasattr(mod, "lora") and n]
print(f"LoRA rank=16: {lora:,}/{base:,} = {100*lora/base:.3f}%（挂载点 {len(hooks)} 个：q/o_proj × 8 层）")

assert lora == 393_216, "rank=16 在 768×8 层应恰为 393,216"
assert len(hooks) == 16, "挂载点应为 16（8 层 × q_proj/o_proj）"
assert lora / base < 0.01, "可训练占比必须 <1%"
print("3 checks passed")
