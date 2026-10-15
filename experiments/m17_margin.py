"""实测：64M DPO 前后 chosen/rejected 概率差（margin）变化。

ref / π_before = out/archive/era2/full_sft_768.pth（09-30 DPO run 加载的参考，md5 fb1dcf…）
π_after        = out/dpo_768.pth（09-30 15:40 run 的产物，md5 521a5b…）
"""
import torch
from minimind_reborn.configuration.schemas import ModelConfig
from minimind_reborn.data.registry import file_path, resolve_dataset
from minimind_reborn.data.datasets import DPODataset
from minimind_reborn.models.config import config_from_modelcfg
from minimind_reborn.models.model import MiniMindForCausalLM
from minimind_reborn.models.weights import load_inference_weights
from minimind_reborn.training.common.setup import load_tokenizer
from minimind_reborn.training.dpo.loss import logits_to_log_probs
import torch.nn.functional as F

torch.manual_seed(42)
mc = ModelConfig(hidden_size=768, num_hidden_layers=8, num_attention_heads=8,
                 num_key_value_heads=4, head_dim=96, intermediate_size=2432)
tok = load_tokenizer()
ds = DPODataset(file_path(resolve_dataset("dpo_mini")), tok, max_length=1024)
beta = 0.15

def build(weight_path):
    m = MiniMindForCausalLM(config_from_modelcfg(mc))
    load_inference_weights(m, weight_path, strict=True)
    return m.eval().requires_grad_(False).cuda()

ref = build("out/archive/era2/full_sft_768.pth")   # 冻结参考 = DPO 的锚
pol = {"DPO 前（=ref）": build("out/archive/era2/full_sft_768.pth"),
       "DPO 后": build("out/dpo_768.pth")}

@torch.no_grad()
def one_pass(m, b):
    x = torch.stack([b["x_chosen"], b["x_rejected"]]).cuda()   # (2, T) 拼批：上半 chosen 下半 rejected
    y = torch.stack([b["y_chosen"], b["y_rejected"]]).cuda()
    mask = torch.stack([b["mask_chosen"], b["mask_rejected"]]).cuda()
    return (logits_to_log_probs(m(x).logits, y) * mask).sum(dim=1)   # (2,) 回答区间总 logp

stats = {k: [] for k in pol}
for i in range(64):
    b = ds[i * 7 % len(ds)]
    ref_lp = one_pass(ref, b)                                   # (2,)
    for k, m in pol.items():
        lp = one_pass(m, b)
        z = beta * ((lp[0] - lp[1]) - (ref_lp[0] - ref_lp[1]))  # 每对的 DPO 判据量
        stats[k].append((z.item(), (-F.logsigmoid(z)).item()))

for k, v in stats.items():
    zs = torch.tensor([t[0] for t in v]); ls = torch.tensor([t[1] for t in v])
    print(f"{k}: margin z 均值 {zs.mean():+.4f} | σ(z) 偏好判对率 {torch.sigmoid(zs).mean():.4f} | 单对 DPO loss {ls.mean():.4f} | z>0 占比 {(zs>0).float().mean():.2%}")
