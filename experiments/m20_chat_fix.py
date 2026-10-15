# M20 chat 签名 bug 收尾：① 直连复现 ② 一行修复验证（猴子补丁形态，仓库只读）③ 契约断言
import inspect
from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.chat import ChatSession, render_chat
from minimind_reborn.inference.engine import MiniMindLLM

llm = MiniMindLLM.build("full_sft")
gen = GenerateConfig(temperature=0.1, max_new_tokens=8)

s_bad = ChatSession(llm.tokenizer, llm.generate_tokens, gen)
try:
    s_bad.respond("你好")
except TypeError as e:
    print("① 直连复现:", e)

# ② M02 的一行修复：重写 respond，仅第 8 行改为按真实签名关键字传参（仓库只读，用补丁形态验证）
def respond_fixed(self, user_text, *, open_thinking=False):
    self.history.append({"role": "user", "content": user_text})
    if self.pretrain_mode:
        prompt = self.tokenizer.bos_token + user_text
    else:
        self._trim_history()
        prompt = render_chat(self.tokenizer, self.history, open_thinking=open_thinking)
    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
    out = self.generate_fn(inputs["input_ids"], self.gen, attention_mask=inputs["attention_mask"])  # ← 唯一改动
    new_tokens = out["sequences"][0, inputs["input_ids"].shape[1]:]
    text = self._decode_truncated(new_tokens)
    self.history.append({"role": "assistant", "content": text})
    return text

ChatSession.respond = respond_fixed
s = ChatSession(llm.tokenizer, llm.generate_tokens, gen)
print("② 修复后 respond:", s.respond("你好"))
print("   history roles:", [m["role"] for m in s.history])

sig = inspect.signature(llm.generate_tokens)               # ③ 契约锁定：签名断言
positional = [p for p, v in sig.parameters.items()
              if v.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and v.default is inspect.Parameter.empty]
print("③ generate_tokens 必填位置参数:", positional[1:], "——其余全部 keyword-only（星号之后）")
