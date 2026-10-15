"""M22 毕业示范实验（终版）：温度 0.7 vs 0.85 × 3 seed 的固定 8 题对照。

单变量设计：
- 自变量：temperature（0.7 vs 0.85）
- 因变量：distinct-2 均值、平均生成长度、finish=stop 比例、吞吐
- 控制变量：同一权重 out/full_sft_768.pth、top_p=0.95、top_k=50、max_new_tokens=384、
  固定 8 题（M21 eval_llm.py 的 PROMPTS）、seed ∈ {42,43,44}（两臂用同一组 seed）
"""

from __future__ import annotations

import torch

from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.engine import MiniMindLLM

PROMPTS = [
    "你有什么特长？",
    "为什么天空是蓝色的",
    "请用Python写一个计算斐波那契数列的函数",
    '解释一下"光合作用"的基本过程',
    "如果明天下雨，我应该如何出门",
    "比较一下猫和狗作为宠物的优缺点",
    "解释什么是机器学习",
    "推荐一些中国的美食",
]


def distinct2(text: str) -> float:
    """distinct-2：不重复 bigram 占全部 bigram 的比例（越低=复读越重）。"""
    chs = list(text)
    bigrams = [tuple(chs[i : i + 2]) for i in range(len(chs) - 1)]
    return len(set(bigrams)) / len(bigrams) if bigrams else 1.0


def run_arm(llm: MiniMindLLM, temperature: float, seed: int) -> dict:
    torch.manual_seed(seed)
    gen = GenerateConfig(temperature=temperature, top_p=0.95, max_new_tokens=384)
    lens, d2s, stops, total_s = [], [], 0, 0.0
    for p in PROMPTS:
        r = llm.generate_text([{"role": "user", "content": p}], gen)
        lens.append(r["generated_tokens"])
        d2s.append(distinct2(r["content"]))
        stops += r["finish_reason"] == "stop"
        total_s += r["latency_ms"] / 1000
    return {"d2": sum(d2s) / len(d2s), "len": sum(lens) / len(lens),
            "stops": stops, "tps": sum(lens) / total_s}


def main() -> None:
    llm = MiniMindLLM.build("full_sft")
    seeds = (42, 43, 44)
    print("temp  seed  distinct2  平均长度  stop/8  tok/s")
    for t in (0.7, 0.85):
        stats = []
        for s in seeds:
            r = run_arm(llm, t, s)
            stats.append(r)
            print(f"{t:.2f}  {s:>4}  {r['d2']:.4f}    {r['len']:6.1f}   {r['stops']}/8   {r['tps']:6.1f}")
        m = lambda k: sum(x[k] for x in stats) / len(stats)
        print(f"  → {t} 均值：distinct2={m('d2'):.4f}  长度={m('len'):.1f}  "
              f"stop={sum(x['stops'] for x in stats)}/{3*8}  吞吐={m('tps'):.1f}\n")


if __name__ == "__main__":
    main()
