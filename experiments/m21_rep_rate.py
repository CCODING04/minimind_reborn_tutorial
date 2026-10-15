import re, sys

def ngram_rep_rate(text, n=3):
    cleaned = re.sub(r"\s", "", text)
    if len(cleaned) < n + 1:
        return 0.0
    grams = [cleaned[i:i+n] for i in range(len(cleaned) - n + 1)]
    return 1 - len(set(grams)) / len(grams)

for path in sys.argv[1:]:
    src = open(path, encoding="utf-8").read()
    answers = re.findall(r"🧠: (.*?)\n\[", src, re.S)
    reps = [ngram_rep_rate(a) for a in answers]
    lens = [len(a) for a in answers]
    n_stop = len(re.findall(r"\| stop\]", src))
    n_len = len(re.findall(r"\| length\]", src))
    print(f"{path}: 答案 {len(answers)} | 平均3gram重复率 {sum(reps)/len(reps):.3f} | 平均长度 {sum(lens)/len(lens):.0f} 字 | stop {n_stop} / length {n_len}")
