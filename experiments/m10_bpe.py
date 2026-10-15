"""迷你 BPE：8 行语料，手工跑前 3 轮合并（字符级，教学版）。"""
from collections import Counter

corpus = [
    "你好世界",
    "你好小模型",
    "世界你好",
    "你好你好",
    "小模型你好",
    "你好世界模型",
    "模型世界你好",
    "你好小模型世界",
]

# 每行先拆成字符序列；词表初始 = 语料里出现过的字符
seqs = [list(line) for line in corpus]
vocab = sorted({ch for s in seqs for ch in s})
print("初始词表（%d）：%s" % (len(vocab), " ".join(vocab)))

def top_pairs(n=3):
    cnt = Counter()
    for s in seqs:
        for a, b in zip(s, s[1:]):   # 相邻对：滑窗一格
            cnt[(a, b)] += 1
    return cnt.most_common(n)

for rnd in range(1, 4):
    pairs = top_pairs()
    print(f"\n第 {rnd} 轮 · 相邻对频次 top3：{pairs}")
    (a, b), c = pairs[0]
    merged = a + b
    new_seqs, hits = [], 0
    for s in seqs:
        out, i = [], 0
        while i < len(s):
            if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                out.append(merged); i += 2; hits += 1   # 合并：两枚旧 token 换一枚新 token
            else:
                out.append(s[i]); i += 1
        new_seqs.append(out)
    seqs = new_seqs
    vocab.append(merged)
    print(f"  合并 {a}+{b} → 「{merged}」（命中 {hits} 处）→ 词表 {len(vocab)} 个")
    print("  语料现状：", ["|".join(s) for s in seqs[:3]], "…")
print("\n3 轮后词表（%d）：%s" % (len(vocab), " ".join(vocab)))
