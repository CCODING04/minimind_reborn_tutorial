"""M06 坑卡 1 复现：把 KVCache.advance 打成空操作——输出逐位一致，只是喂入 token 数暴涨。

tiny 模型（未训练随机权重即可，两条路径比的是行为不是质量），prompt 9 token、生成 23 步，
forward hook 记录每次真正喂进 model.model 的 token 数。
"""
import torch
from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.generator import generate
from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.kv_cache import KVCache
from minimind_reborn.models.model import MiniMindForCausalLM

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = MiniMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
model = MiniMindForCausalLM(cfg).to(dev).eval()
PROMPT, NEW = 9, 24
prompt = torch.randint(0, 6400, (1, PROMPT), device=dev)

def run(tag):
    fed = []
    hook = model.model.register_forward_hook(
        lambda m, i, o: fed.append(i[0].shape[1]))      # 每次前向真正吃进的 token 数
    gen = GenerateConfig(temperature=0.0, max_new_tokens=NEW)
    out = generate(model, prompt, gen, eos_token_id=None, pad_token_id=0)
    hook.remove()
    print(f"{tag}：{len(fed)} 次前向，喂入 token 序列 = [{fed[0]}, {fed[1]}, {fed[2]}, …, {fed[-1]}]（总量 {sum(fed)}）")
    return out["sequences"]

seq_a = run("正常")

# 把 advance 打成空操作：游标停在 0，generator 每步都按"还没 prefill"喂整段
orig_advance = KVCache.advance
KVCache.advance = lambda self, *a, **k: None         # 模拟手写循环里忘推游标
try:
    seq_b = run("忘了 advance")
finally:
    KVCache.advance = orig_advance

print("输出仍然逐位一致 =", bool((seq_a == seq_b).all()))
