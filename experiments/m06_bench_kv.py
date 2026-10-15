"""M06 对照实验：无缓存全量重算 vs reborn generate()（KV cache）——耗时/前向 token 数/逐位一致。

页面展示为 NEW=50 版本；表格里的 200/500 两轮是改 NEW 后重跑。
"""
import statistics, time, torch
from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.generator import generate
from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.model import MiniMindForCausalLM
from minimind_reborn.models.weights import load_inference_weights

torch.manual_seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = MiniMindConfig()                            # 64M 正主：768 × 8 层，8 头 / 4 KV 头
model = MiniMindForCausalLM(cfg)
load_inference_weights(model, "out/pretrain_768.pth", strict=True)
model = model.to(dev).eval()
PROMPT, NEW = 64, 50                              # 想扫更长，改 NEW 后重跑
prompt = torch.randint(0, 6400, (1, PROMPT), device=dev)

def no_cache():
    cur, fwd = prompt, 0
    for _ in range(NEW):                          # 每步全量前向：重算整个前缀的注意力
        with torch.inference_mode():
            logits = model(cur).logits[:, -1, :]  # (1, 6400)：只取最后一位，却算了全部位置
        fwd += cur.shape[1]
        cur = torch.cat([cur, logits.argmax(-1, keepdim=True)], dim=1)
    return cur, fwd

def with_cache():
    gen = GenerateConfig(temperature=0.0, max_new_tokens=NEW)
    out = generate(model, prompt, gen, eos_token_id=None, pad_token_id=0)
    return out["sequences"], PROMPT + NEW         # prefill 64 + 50 个单 token 步

def bench(fn):
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter(); fn()
    if dev == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000

no_cache(); with_cache()                          # 预热：首次内核加载不计入
seq_a, tok_a = no_cache(); seq_b, tok_b = with_cache()
assert (seq_a == seq_b).all()                     # 两条路径必须逐位一致（见"最硬的一条测试"）
print(statistics.median(bench(no_cache) for _ in range(5)),
      statistics.median(bench(with_cache) for _ in range(5)))
