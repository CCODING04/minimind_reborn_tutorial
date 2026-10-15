# experiments/ · 课程实验脚本归档

课程各页引用的实验脚本统一收录于此（原始工作副本原在 `/tmp`；除标注"回填重建"外均逐字节归档、未改写）。
页面里的代码块是本目录脚本的**逐字切片**，终端输出是脚本的真实运行结果——完整代码经各页
标题里的 `codeview/experiments/<name>.html` 链接查看。

## 运行方式

脚本 `import minimind_reborn`，请在 **minimind_reborn 仓库根目录**用其 uv 环境运行：

```bash
cd <你的路径>/Reproduce/minimind_reborn
uv run python <课程仓库>/experiments/m08_exp_route.py
```

课程页面终端块里的命令 `uv run python experiments/m08_exp_route.py` 即此写法的简式
（路径按你的 checkout 位置调整）。多数脚本纯 CPU 秒级可跑；依赖数据/权重/GPU 的见下表"复跑"列。

## 结果审查（复跑验证）

- `expected/<name>.txt`：脚本在课程标注环境（torch 2.14.0+cu130）下的逐字输出存档（两次复跑逐字一致才收录）。
- `python3 tools/verify_experiments.py [--update] [--only <name>]`：在 reborn uv 环境复跑并逐行比对存档；
  页面终端块的输出应与存档一致（长日志按页面标注的 `…` 裁剪）。
- ✅ = 自动复跑比对通过；⚠️ manual = 输出本质不稳定（时延/采样/内存读数漂移）或重依赖（大数据/多卡/进程管理），
  不自动复跑，页面输出以**作者实测存档**为准，原因见 tools/verify_experiments.py 的 MANUAL 表。

## 清单

| 脚本 | 课程位置 | 内容 | 复跑 |
|---|---|---|---|
| [tour.py](../codeview/experiments/tour.py.html) | M03 | 全管线七步巡览 | ⚠️ 采样文本 |
| [m06_bench_kv.py](../codeview/experiments/m06_bench_kv.py.html) | M06 · 对照实验 | 无缓存全量重算 vs KV cache（耗时/前向 token 数/逐位一致） | ⚠️ 时延基准 |
| [m06_advance_pit.py](../codeview/experiments/m06_advance_pit.py.html) | M06 · 坑卡 1 | advance 空操作复现：喂入 token 9+…+32=492、输出仍一致（回填重建） | ✅ |
| [m05_exp_bf16.py](../codeview/experiments/m05_exp_bf16.py.html) | M05 | bf16/fp16/fp32 数值对比 | ✅ |
| [m05_exp_rope.py](../codeview/experiments/m05_exp_rope.py.html) | M05 | RoPE 旋转矩阵性质 | ✅ |
| [m07_equiv.py](../codeview/experiments/m07_equiv.py.html) | M07 | 手写 attention 与 SDPA 等价 | ✅ |
| [m07_fill.py](../codeview/experiments/m07_fill.py.html) | M07 · 填空 | 路径③ 三空模板（回填重建，填空后可跑） | ⚠️ 模板 |
| [m07_fill_answer.py](../codeview/experiments/m07_fill_answer.py.html) | M07 · 填空答案 | 三空的答案版 | ✅ |
| [m08_exp_route.py](../codeview/experiments/m08_exp_route.py.html) | M08 · 路由小节 | 路由分布 / aux loss train·eval / 偏心 router / 直通验证 | ✅ |
| [m08_exp_ddp.py](../codeview/experiments/m08_exp_ddp.py.html) | M08 · 挂图小节 | DDP 未使用参数真报错 vs 0·Σp 挂图（gloo 单进程） | ✅ |
| [m08_exp_params.py](../codeview/experiments/m08_exp_params.py.html) | M08 · 参数账本 | dense vs MoE 总参数 / 每 token 激活（hook 数专家） | ✅ |
| [m08_exp_k2.py](../codeview/experiments/m08_exp_k2.py.html) | M08 · 变式 1 | num_experts_per_tok=2 的路由与激活 | ✅ |
| [m09_tiny_loss.py](../codeview/experiments/m09_tiny_loss.py.html) | M09 | tiny 模型移位交叉熵手算 | ✅ |
| [m09_ledger.py](../codeview/experiments/m09_ledger.py.html) | M09 | 63.91M 参数账本逐项 | ✅ |
| [m09_meta_repro.py](../codeview/experiments/m09_meta_repro.py.html) | M09 | meta 设备不匹配的 nan→修复复现 | ⚠️ 随机数 |
| [m09_strict_load.py](../codeview/experiments/m09_strict_load.py.html) | M09 · 验收 | strict 加载官方权重 | ✅ |
| [m10_observe.py](../codeview/experiments/m10_observe.py.html) | M10 | tokenizer 观察三件套 | ✅ |
| [m10_bpe.py](../codeview/experiments/m10_bpe.py.html) | M10 | BPE 合并规则手办 | ✅ |
| [m10_render.py](../codeview/experiments/m10_render.py.html) | M10 | chat template 渲染 | ✅ |
| [m10_pit.py](../codeview/experiments/m10_pit.py.html) | M10 · 坑卡 | 特殊 token 落坑与修复 | ✅ |
| [m10_accept.py](../codeview/experiments/m10_accept.py.html) | M10 · 验收 | 五往返模板断言 | ✅ |
| [m11_bench.py](../codeview/experiments/m11_bench.py.html) | M11 | 三种取样姿势对比（argparse --mode） | ⚠️ 重数据 |
| [m11_registry.py](../codeview/experiments/m11_registry.py.html) | M11 | 注册表解析 + 错误路径 + 缓存对比 | ⚠️ 内存读数 |
| [m11_pit.py](../codeview/experiments/m11_pit.py.html) | M11 · 坑卡 | 数据集 schema 落坑与修复 | ⚠️ 内存读数 |
| [m11_ex.py](../codeview/experiments/m11_ex.py.html) | M11 · 变式 | slice 直取 vs 扫全量（含"换口径重取"段） | ⚠️ API 演进 |
| [m11_accept.py](../codeview/experiments/m11_accept.py.html) | M11 · 验收 | 注册表对账 | ⚠️ 时延读数 |
| [m12_fill.py](../codeview/experiments/m12_fill.py.html) | M12 · 填空 | 数据扫描主循环三空 + 验收 | ✅ |
| [m12_first.py](../codeview/experiments/m12_first.py.html) | M12 | 首样本逐字段解剖 | ✅ |
| [m12_pre.py](../codeview/experiments/m12_pre.py.html) | M12 | pretrain 样本构造 | ✅ |
| [m12_variant.py](../codeview/experiments/m12_variant.py.html) | M12 · 变式 1 | 比例扫描完整版 | ✅ |
| [m13_primer_gd.py](../codeview/experiments/m13_primer_gd.py.html) | M13 · 入门 | 手写 3 步梯度下降（回填重建） | ✅ |
| [m13_primer_clip.py](../codeview/experiments/m13_primer_clip.py.html) | M13 · 入门 | clip_grad_norm_ 手算等价（回填重建） | ✅ |
| [m13_lr_curve.py](../codeview/experiments/m13_lr_curve.py.html) | M13 | 学习率曲线（页面以表格呈现） | ✅ |
| [m13_underflow.py](../codeview/experiments/m13_underflow.py.html) | M13 | fp16 下溢 vs bf16（回填重建，需 CUDA） | ✅ |
| [m13_bench_dtype.py](../codeview/experiments/m13_bench_dtype.py.html) | M13 | 跨 dtype 训练计时 | ⚠️ 时延基准 |
| [m13_warmup.py](../codeview/experiments/m13_warmup.py.html) | M13 | warmup 曲线 | ✅ |
| [m14_inspect.py](../codeview/experiments/m14_inspect.py.html) | M14 | checkpoint 六件套解剖 | ✅ |
| [m14_del_optim.py](../codeview/experiments/m14_del_optim.py.html) | M14 | 挖掉 optimizer 件的 KeyError | ✅ |
| [m14_kill9.sh](../codeview/experiments/m14_kill9.sh.html) | M14 · 坑卡 | kill -9 中断与 .tmp 残留 | ⚠️ 进程管理 |
| [m15_nosync_off.py](../codeview/experiments/m15_nosync_off.py.html) | M15 · 坑卡 | no_sync 包装对照（torchrun） | ⚠️ 多卡 |
| [m15_equiv_batch.py](../codeview/experiments/m15_equiv_batch.py.html) | M15 | 梯度累积与大批量等价 | ✅ |
| [m16_verify.py](../codeview/experiments/m16_verify.py.html) | M16 · 里程碑 B | 训练产物核验 | ✅ |
| [m17_mem.py](../codeview/experiments/m17_mem.py.html) | M17 | DPO 显存对照（页面以表格呈现） | ✅ |
| [m17_margin.py](../codeview/experiments/m17_margin.py.html) | M17 | margin 抽样统计（60 行） | ⚠️ 统计漂移 |
| [m17_pair.py](../codeview/experiments/m17_pair.py.html) | M17 | 成对样本对照 | ✅ |
| [m18_exp_lora.py](../codeview/experiments/m18_exp_lora.py.html) | M18 | LoRA 参数占比 + merge 一致性 | ✅ |
| [m18_ratio_assert.py](../codeview/experiments/m18_ratio_assert.py.html) | M18 · 验收 | 可训练占比断言 | ✅ |
| [m19_rollout.py](../codeview/experiments/m19_rollout.py.html) | M19 | GRPO rollout 采样 | ✅ |
| [m20_demo.py](../codeview/experiments/m20_demo.py.html) | M20 | 采样五连数值例子 | ✅ |
| [m20_double_t.py](../codeview/experiments/m20_double_t.py.html) | M20 · 坑卡 | 双重温度插桩（修复前） | ✅ |
| [m20_double_t2.py](../codeview/experiments/m20_double_t2.py.html) | M20 · 坑卡 | 双重温度修复版 | ✅ |
| [m20_chat_fix.py](../codeview/experiments/m20_chat_fix.py.html) | M20 · 坑卡 | ChatSession.respond 运行时补丁 | ✅ |
| [m20_rep_variant.py](../codeview/experiments/m20_rep_variant.py.html) | M20 · 变式 | 重复率测量 | ✅ |
| [m21_verify_qwen.py](../codeview/experiments/m21_verify_qwen.py.html) | M21 | Qwen3 对照验证（13 行） | ⚠️ 采样 |
| [m21_rep_rate.py](../codeview/experiments/m21_rep_rate.py.html) | M21 · 变式 | 重复率口径对比 | ✅ |
| [m22_temp_ab3.py](../codeview/experiments/m22_temp_ab3.py.html) | M22 · 第三步 | 温度 A/B 终版（8 题 + tok/s） | ⚠️ 全量评测 |

> 收录原则：页面引用的脚本一律入库、保持原名原样；原始副本已丢失者以页面所载全文回填重建并在表中注明
> （m07_fill、m13_primer_gd、m13_primer_clip、m13_underflow、m06_advance_pit、m06_bench_kv）。
> `tour_greedy.py`（M03 变式"复制 tour.py 改贪心"）为读者练习动作，不单独收录。
