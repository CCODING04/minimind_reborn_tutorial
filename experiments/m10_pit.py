from minimind_reborn.inference.chat import render_chat
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("assets/tokenizer")
u = [{"role": "user", "content": "你好"}]
real = render_chat(tok, u)                                        # 真模板（单一事实源渲染）
hand = f"<|im_start|>user\n{u[0]['content']}<|im_end|>\n<|im_start|>assistant\n"   # 手拼：漏了空 think 块
hand2 = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant \n"              # 只多打了一个空格
ri, hi, h2 = (tok(t).input_ids for t in (real, hand, hand2))
print("真模板：", ri)
print("手拼版：", hi)
same = 0
for a, b in zip(ri, hi):
    if a != b:
        break
    same += 1
print(f"共同前缀 {same} 位后分岔：真 {len(ri)} token，手拼 {len(hi)} token（少了 <think> 空块）")
print("多个空格：", h2, "→ 结尾", repr(tok.decode([h2[-1]])), "≠ '\\n'，prefix 锚点从此失配")
