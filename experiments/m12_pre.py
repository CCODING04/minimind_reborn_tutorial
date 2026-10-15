"""M12 · Pretrain 短样本：pad 位的 label 处理。"""
from transformers import AutoTokenizer

from minimind_reborn.data.datasets import PretrainDataset

tok = AutoTokenizer.from_pretrained("assets/tokenizer")
pd = PretrainDataset("/home/admin02/Code/WorkSpace/minimind-reproduce/dataset/pretrain_t2t_mini.jsonl", tok, max_length=64)
best = min(range(50), key=lambda i: len(tok(pd.load_line(i)["text"]).input_ids))  # 前 50 行里挑最短的
it = pd[best]
ids, labs = it["input_ids"].tolist(), it["labels"].tolist()
n_real = sum(1 for l in labs if l != -100)
print(f"第 {best} 行 @max_length=64：真实 {n_real} token，pad {64 - n_real} 个")
print("ids    转折点前后：", ids[n_real - 2:n_real + 3])
print("labels 转折点前后：", labs[n_real - 2:n_real + 3])
print("首 token：", ids[0], "（bos）；最后一个真实 token：", ids[n_real - 1], "（eos）；pad id：", ids[-1])
print("正文区 label==id：", labs[:n_real] == ids[:n_real], "；pad 区全 -100：", all(l == -100 for l in labs[n_real:]))
