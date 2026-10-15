"""M18 实验：① LoRA 挂载点与参数占比（64M 与 tiny 两档）；② merge 前后输出一致性。"""

import torch

from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.lora import apply_lora, lora_state_dict, merge_lora, save_lora
from minimind_reborn.models.model import MiniMindForCausalLM

torch.manual_seed(42)

# ---- ① 参数占比：64M 主配方（rank=16，与 lora_64m.yaml 一致）----
for tag, cfg, rank in [
    ("64M  (768×8层, rank=16)", MiniMindConfig(), 16),
    ("tiny (64×2层,  rank=8 )", MiniMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2), 8),
]:
    m = MiniMindForCausalLM(cfg)
    before = sum(p.numel() for p in m.parameters())
    apply_lora(m, rank=rank)
    hooked = [n for n, mod in m.named_modules() if hasattr(mod, "lora") and n]
    lora_n = sum(p.numel() for n, p in m.named_parameters() if "lora" in n)
    print(f"{tag}: 总参数 {before:,} | LoRA 参数 {lora_n:,} | 占比 {100*lora_n/before:.3f}%")
    print(f"  挂载点（{len(hooked)} 个）: {hooked}")

# ---- ② merge 一致性：64M，rank=16 ----
model = MiniMindForCausalLM(MiniMindConfig())
apply_lora(model, rank=16)
# 把 B 随机化，模拟"训了一阵的适配器"（B 非零，ΔW = B·A 才非零）
for n, p in model.named_parameters():
    if "lora.B" in n:
        p.data.normal_(0, 0.02)
x = torch.randint(0, 6400, (2, 16))
model.eval()
with torch.no_grad():
    y_adapter = model(x).logits  # 基座 + 旁路（W + B·A 的等价拆分执行）

save_lora(model, "/tmp/m18_lora.pth")
with torch.no_grad():  # 同一模型把 B 归零 = 纯基座视角（ΔW=B·A=0，旁路加 0）
    for n, p in model.named_parameters():
        if "lora.B" in n:
            p.zero_()
    y_base = model(x).logits

merge_lora(model, "/tmp/m18_lora.pth", "/tmp/m18_merged.pth")
model3 = MiniMindForCausalLM(MiniMindConfig())
sd = torch.load("/tmp/m18_merged.pth", map_location="cpu", weights_only=True)
model3.load_state_dict(sd)  # merged 权重按基座结构载入（无 lora 键）
with torch.no_grad():
    y_merged = model3(x).logits

d_base = (y_adapter - y_base).abs().max().item()
d_merge = (y_adapter - y_merged).abs().max().item()
print(f"\nmerge 一致性（64M, rank=16, fp16 适配器存档）:")
print(f"  adapter 路线 vs 干净基座     max|Δ| = {d_base:.4f}   ← 旁路确实改了输出")
print(f"  adapter 路线 vs merge 后基座 max|Δ| = {d_merge:.6f} ← 应≈0（半精度舍入级）")
print(f"  merged 权重里还有 lora 键?  {any('.lora.' in k for k in sd)}")
print(f"  lora 存档大小: {len(lora_state_dict(model))} 个张量 → B·A 逐层写回 W'")
