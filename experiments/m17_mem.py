"""显存账：DPO（策略+冻结参考，拼批）vs 单模型前向反向，64M、bf16 AMP、batch 8×1024。"""
import torch
from minimind_reborn.configuration.schemas import ModelConfig
from minimind_reborn.models.config import config_from_modelcfg
from minimind_reborn.models.model import MiniMindForCausalLM

mc = ModelConfig(hidden_size=768, num_hidden_layers=8, num_attention_heads=8,
                 num_key_value_heads=4, head_dim=96, intermediate_size=2432)
x = torch.randint(0, 6400, (8, 1024), device="cuda")   # 拼批：4 对 chosen/rejected
y = torch.randint(0, 6400, (8, 1024), device="cuda")
mask = torch.ones(8, 1024, device="cuda")

def report(tag):
    torch.cuda.synchronize()
    print(f"{tag}: 峰值显存 {torch.cuda.max_memory_allocated()/2**20:.0f} MiB（重置前）")
    torch.cuda.reset_peak_memory_stats()

# ① DPO：两份模型，策略带梯度，参考 no_grad
pol = MiniMindForCausalLM(config_from_modelcfg(mc)).cuda()
ref = MiniMindForCausalLM(config_from_modelcfg(mc)).cuda().eval().requires_grad_(False)
report("两份模型常驻（参数）")
from minimind_reborn.training.dpo.loss import dpo_loss, logits_to_log_probs
with torch.autocast("cuda", dtype=torch.bfloat16):
    with torch.no_grad():
        ref_lp = logits_to_log_probs(ref(x).logits, y)          # (8, 1024)
    out = pol(x)
    pol_lp = logits_to_log_probs(out.logits, y)
    loss = dpo_loss(ref_lp, pol_lp, mask, beta=0.15)
report("前向后（参考 no_grad + 策略带计算图）")
loss.backward()
report("反向后")
del pol, ref, ref_lp, pol_lp, out, loss
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# ② 对照：单模型 SFT 式（同 batch）
m = MiniMindForCausalLM(config_from_modelcfg(mc)).cuda()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(x, labels=y)
report("单模型（SFT 式 labels 交叉熵）前向后")
out.loss.backward()
report("单模型反向后")
