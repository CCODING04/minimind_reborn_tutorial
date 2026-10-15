from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.model import MiniMindForCausalLM
from minimind_reborn.models.weights import load_inference_weights

model = MiniMindForCausalLM(MiniMindConfig())            # 自建 64M：全部默认值
load_inference_weights(model, "out/official/pretrain_768.pth", strict=True)
print("官方 pretrain_768.pth strict=True 载入成功：无缺失 key、无多余 key")
print("参数量：%.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
