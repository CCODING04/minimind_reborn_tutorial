# M20 变式：repetition_penalty 对贪心复读的影响（full_sft_768.pth，temperature=0 排除采样随机性）
from collections import Counter
from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.engine import MiniMindLLM

def dup_rate(text: str, n: int = 5) -> float:
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return sum(c for g, c in Counter(grams).items() if c > 1) / len(grams)

llm = MiniMindLLM.build("full_sft")
msgs = [{"role": "user", "content": "介绍一下大语言模型"}]
for rp in (1.0, 1.5):
    gen = GenerateConfig(temperature=0.0, repetition_penalty=rp, max_new_tokens=128)
    r = llm.generate_text(msgs, gen)
    c = r["content"]
    print(f"rp={rp}: 重复 5-gram 率={dup_rate(c):.1%} 「大语言模型」出现 {c.count('大语言模型')} 次 tokens={r['generated_tokens']}")
    print("  ", c[:80], "…")
