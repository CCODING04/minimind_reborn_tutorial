"""复刻 trainer._load_initial_weights 的 resume 段（trainer.py:442-444），
喂它一个被手删了 optimizer 件的 checkpoint，看真实报错。"""
import torch
from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.model import MiniMindForCausalLM
from minimind_reborn.training.common.optim import configure_optimizers
from minimind_reborn.training.common.checkpoint import load_checkpoint

# 与 smoke 同结构的 tiny 模型 + 真优化器（decay 分组同款）
model = MiniMindForCausalLM(MiniMindConfig(hidden_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2))
optimizer = configure_optimizers(model, lr=5e-4, weight_decay=0.1)
scaler = torch.amp.GradScaler(enabled=False)

# 把刚跑完实验的六件套挖掉一件再存回去
data = torch.load("checkpoints/pretrain_64_resume.pth", map_location="cpu", weights_only=False)
print("原件九键：", sorted(data.keys()))
del data["optimizer"]                    # ← 模拟"手删 optimizer 件"
torch.save(data, "/tmp/m14_no_optimizer.pth")

data = load_checkpoint("/tmp/m14_no_optimizer.pth")   # load_checkpoint 本身不报错
model.load_state_dict(data["model"], strict=True)     # 模型件：正常恢复
try:
    optimizer.load_state_dict(data["optimizer"])      # 下一行：KeyError
except KeyError as e:
    print("KeyError:", e)
