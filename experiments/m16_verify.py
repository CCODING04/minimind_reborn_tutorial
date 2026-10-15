"""里程碑 B 验证：自己训的 tiny 基座能被自己的推理链路加载并生成。"""
import torch
from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.engine import MiniMindLLM

torch.manual_seed(7)
llm = MiniMindLLM.build("pretrain", config_path="runs/smoke_pretrain/20261001_094752/config.json")
out = llm.generate_text([{"role": "user", "content": "北京是中国的"}],
                        GenerateConfig(temperature=0.8, max_new_tokens=48), pretrain_mode=True)
print("生成:", out["content"], "| finish:", out["finish_reason"], "| tokens:", out["generated_tokens"])
