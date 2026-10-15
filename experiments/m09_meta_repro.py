import torch
from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.model import MiniMindForCausalLM

torch.manual_seed(0)
cfg = MiniMindConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, vocab_size=64, max_position_embeddings=128)
good = MiniMindForCausalLM(cfg).eval()              # 正常初始化的孪生模型
with torch.device("meta"):                          # transformers>=5 建模型的姿势：只登记形状不占内存
    m = MiniMindForCausalLM(cfg)
m = m.to_empty(device="cpu")                        # 物化：分配内存但不填值
m.load_state_dict(good.state_dict())                # 参数与持久 buffer 复活；非持久 buffer 无人负责
x = torch.randint(0, 64, (2, 7))
with torch.no_grad():
    diff = (m(x).logits - good(x).logits).abs().max().item()
print("freqs_cos 与好模型一致?", torch.equal(m.model.freqs_cos, good.model.freqs_cos))
print("logits 最大偏差:", round(diff, 4))
