import math
import torch
from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.model import MiniMindForCausalLM

torch.manual_seed(42)
cfg = MiniMindConfig(hidden_size=64, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2)   # 词表仍是默认 6400
tiny = MiniMindForCausalLM(cfg).eval()
x = torch.randint(0, cfg.vocab_size, (3, 16))     # (B, T)：随机 token 当输入
with torch.no_grad():
    out = tiny(x, labels=x)                       # labels=input_ids：预测下一个 token
print("logits:", tuple(out.logits.shape), "| loss = %.4f" % out.loss.item())
print("ln(6400) = %.4f" % math.log(6400))
