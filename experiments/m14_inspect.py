"""解剖六件套：每件的键结构与字节占比（本课 kill -9 实验同款 tiny checkpoint）。"""
import os
import torch

d = torch.load("/tmp/m14_ck.pth", map_location="cpu", weights_only=False)
print("payload 顶层键（9 个）：", sorted(d.keys()))
print("model   件：", len(d["model"]), "个张量键，如", list(d["model"])[:2])
print("optim   件：", sorted(d["optimizer"].keys()),
      "| state 覆盖", len(d["optimizer"]["state"]), "个参数 | param_groups",
      len(d["optimizer"]["param_groups"]), "组（decay/no_decay，M13）")
print("scaler  件：", d["scaler"])
print("epoch/step 件：", d["epoch"], "/", d["step"], "（micro step 口径）| world_size 件：", d["world_size"])
print("best    件：", d["best_val_loss"])
print("config  件：", list(d["config"].keys())[:6], "…")
print("rng     件：", {k: type(v).__name__ for k, v in d["rng"].items()})
torch.save({k: d[k] for k in ("model",)}, "/tmp/m14_only_model.pth")
for tag, f in [("六件套全套", "/tmp/m14_ck.pth"), ("只留 model 件", "/tmp/m14_only_model.pth")]:
    print(f"{tag}：{os.path.getsize(f)/1e6:.2f} MB")
