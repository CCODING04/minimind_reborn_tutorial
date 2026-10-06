# 课程设计：minimind_reborn_tutorial

> 课程：《跟着 minimind_reborn 造一个大模型》
> 状态：v1（2026-09-29）。本文约束每一页的写法与验收；模块清单的**单一事实源是 `assets/course.js` 的 `COURSE` 数组**，本文表格如与之冲突以 COURSE 为准。
> 上游文档：[ANALYSIS.md](ANALYSIS.md)（参照教程 Part7_minimind 的优劣分析与继承决定）。

---

## 1. 定位与受众

- **实现目标**：[`Reproduce/minimind_reborn`](../../../Reproduce/minimind_reborn)——64M 中文小模型的工程化全流程训练项目（pretrain→SFT→DPO→蒸馏/LoRA→RL→推理服务，85 个测试（写作时实测），8 个冒烟配方）。
- **受众**：会 Python 与基本 PyTorch（张量/autograd/nn.Module），LLM 零经验或初学；对真实工程项目的配置/测试/分布式经验不限。
- **教学主线（混合式）**：主线**跟读 reborn 真实代码** + **动手跑冒烟与实验** + **复用其 85 个测试做绿灯验收**；每模块配"变式练习"（先预测再观察）；关键模块（attention、loss_mask）增加页内填空环节。reborn 本身就是"对照答案"，不另造教学专用代码。
- **硬件分层**：主线全部 CPU/tiny 配方可完成；64M 真训与双卡 DDP 标注 🎮 选做；无 GPU 时走 `[存档]` 读档分析路线（读真实日志是正式学习路径，不是降级）。

## 2. 课程结构（6 篇 22 模块）

```
P1 开箱·用     M01 ─ M02 ─ M03                      现象与地图
P2 模型·造     M04 ─ M05 ─ M06 ─ M07 ─ M08 ─ M09 🏁A   自底向上建模型
P3 数据        M10 ─ M11 ─ M12                       语料到样本
P4 训练        M13 ─ M14 ─ M15 ─ M16 🏁B ─ M17 ─ M18 ─ M19   基座到对齐
P5 推理·评测   M20 ─ M21                             白箱推理与打分
P6 毕业        M22 🏁C                                全链路与毕业实验
```

| # | 文件 | 标题 | 核心内容 | 关键 reborn 资产 | 里程碑/填空 |
|---|---|---|---|---|---|
| M01 | `m01_get_started.html` | 课程导览与第一个绿灯 | 学习方法、uv 环境、verify_env、make smoke | `Makefile`、`tools/verify_env.py`、`configs/smoke/` | |
| M02 | `m02_chat.html` | 与 64M 模型对话 | 权重加载、引擎门面、温度/top_p 旋钮、真实 bug 猎场 | `tools/chat.py`、`inference/engine.py`、`out/*.pth` | 🔍 chat 入口签名 bug（实测发现） |
| M03 | `m03_pipeline.html` | 全链路地图：一次请求的一生 | 七站巡游 tokenizer→engine→generator→model→kv→sampling→detok、架构总览 | `inference/` 全目录、`models/` 总览 | 概念地图（后续每课解剖一站） |
| M04 | `m04_config.html` | 配置即实验 | 分域 dataclass、YAML recipe 只写差异、CLI --set、守门人校验 | `configuration/{schemas,loader}.py`、`configs/*.yaml` | 🔍 bool CLI 覆盖失效（cf46125） |
| M05 | `m05_layers.html` | 地基件：RMSNorm 与 RoPE | fp32 归一化纪律、旋转位置编码推导、YaRN 折叠深读 | `models/layers.py` | |
| M06 | `m06_kv_cache.html` | KV Cache 预分配 | 一次性预分配 vs torch.cat、写读游标、5 个测试 | `models/kv_cache.py`、`test_kv_cache.py` | |
| M07 | `m07_attention.html` | Attention：GQA、QK-Norm 与三条路径 | 手动路径/SDPA 等价性、GQA 头复用 | `models/attention.py` | ✏️ 填空（因果 mask 分支）；🔍 等价性先行 |
| M08 | `m08_ffn_moe.html` | FFN：SwiGLU 与 MoE | SwiGLU 结构、top-1 直通、负载均衡辅助损失 | `models/feedforward.py` | |
| M09 | `m09_model.html` | 组装 ForCausalLM → 🏁A | weight tying、移位交叉熵、参数账本 63.91M、官方权重 strict 互通 | `models/{model,config,weights}.py`、`test_model.py` | 🏁A；🔍 RoPE buffer meta 丢失 |
| M10 | `m10_tokenizer.html` | tokenizer 与 chat 模板 | 6400 词表 BPE、特殊 token、模板即代码 | `assets/tokenizer/`、`constants.py`、`tools/train_tokenizer.py` | |
| M11 | `m11_data_access.html` | 数据访问层 | registry 声明式注册、O(N) 字节偏移索引 O(1) 取样 | `data/{registry,datasets,manifest}.py`、`datasets.json` | |
| M12 | `m12_datasets.html` | Dataset 家族与 loss_mask | Pretrain/SFT/DPO/RLAIF/Agent 五数据集、C1–C5 修复、脆耦合 | `data/{datasets,loss_mask}.py`、`test_loss_mask.py` | ✏️ 填空（response span 匹配） |
| M13 | `m13_trainer.html` | 训练基座 Trainer | 生命周期、LR 纯函数、AMP、decay 分组、prefetch、MFU/ETA | `training/common/`、`test_common.py` | 🔍 ETA 恒 0（9b9a105） |
| M14 | `m14_checkpoint.html` | checkpoint 六件套与续训 | 六件套、RNG 状态、原子写、kill -9 实验 | `training/common/checkpoint.py`、`docs/pretrain_readiness.md` | 🔍 续训记账漂移（fd9fdc8）、残留 resume（500513b） |
| M15 | `m15_ddp.html` | DDP 与 no_sync | 容错 dist 封装、rank0 门控、no_sync 梯度累积 | `utils/dist.py` | 🔍 缺 no_sync 12 倍通信（67.6%→88.5%） |
| M16 | `m16_pretrain_sft.html` | pretrain/SFT 实战 → 🏁B | 37 行母版 workflow、同构之美、loss 曲线读法 | `training/{pretrain,sft}/`、`runs/` 存档 | 🏁B（tiny 基座 + SFT 对比） |
| M17 | `m17_dpo.html` | DPO 偏好对齐 | Bradley-Terry→DPO 推导、冻结参考模型、拼批前向 | `training/dpo/` | |
| M18 | `m18_distill_lora.html` | 蒸馏与 LoRA | 温度软化 KL、低秩适配、只存适配器 | `training/distill/`、`models/lora.py`、`training/lora/` | |
| M19 | `m19_rl.html` | RL 一览：rollout 的世界 | RolloutEngine 抽象、GRPO 组相对优势、PPO/Agent 概览 | `training/{rollout,grpo,ppo,agent}/` | 🔍 mypy 揪出 PPO all_reduce（f8af44d） |
| M20 | `m20_inference.html` | 推理栈深潜 | 采样纯函数链契约、三重停止、KV 预分配推理、engine 门面、serve | `inference/{sampling,generator,chat,engine}.py`、`tools/serve.py` | 🔍 chat 签名 bug 深修（M02 伏笔回收） |
| M21 | `m21_eval.html` | 评测与官方对照 | 固定题评测、多轮/长上下文套件、官方权重对照、HF 导出 | `tools/eval_*.py`、`out/official_compare.md`、`convert_model.py` | |
| M22 | `m22_graduation.html` | 全链路与毕业实验 → 🏁C | run_full.sh、八范式冒烟、改进假设小型对照实验 | `tools/run_full.sh`、`configs/smoke/` | 🏁C（毕业） |

**里程碑定义**：
- 🏁A（M09）：模型建成——`test_model.py` 主体绿 + 官方 `pretrain_768.pth` 以 strict=True 载入（模型与官方逐键等价的最硬证据）。
- 🏁B（M16）：自己的基座——tiny pretrain 收敛权重可被自己的推理链路加载生成；读档分析 reborn 64M run。
- 🏁C（M22）：毕业——85 测全绿 + 八范式 smoke 全通过 + 一份有数据支撑的毕业小实验。

**写作顺序与状态**：22 模块已全部成文（2026-10-01 发布，COURSE 数组全部 status=done）。

## 3. 教学约定

### 3.1 每模块固定骨架

```
模块头（徽章+标题+导语+元数据条：前置/时长/新概念/对照文件）
→ 学完你能（3~5 条，动词开头）
→ 正文（概念卡/步骤/代码精读/坑卡/工业对照/猎场，即需即讲）
→ 折叠测验（≥3 题，每题附"答错回看"指回具体小节）
→ 变式练习（先预测再观察，答案折叠）
→ 验收块（具体命令 + 绿灯标准 + 失败排查折叠）
→ 本课词汇（页尾 glossary）
→ 页间导航（← 上一课 / 标记完成 / 下一课 →）
```

### 3.2 元素体系（继承 Part7，视觉换 GitHub Alerts 语言）

| 元素 | class | 视觉 | 用途与规则 |
|---|---|---|---|
| 直觉卡 | `.callout.tip` | 绿 · 💡 | 先直觉后数学；每模块 0~2 张 |
| 严谨卡 | `.callout.note` | 蓝 · 📐 | 真推导/精确定义，带 arxiv 链接 |
| 坑卡 | `.callout.warn` | 黄 · ⚠️ | 症状→原因→解法 三段式；每模块 ≥1（含"本课刻意没有坑，因为…"） |
| 红线 | `.callout.danger` | 红 · 🛑 | 违反即 bug 的硬约束；须可追溯到真实 bug |
| 工业对照 | `.callout.industry` | 紫 · 🏭 | 教学实现 vs 工业实践分野 |
| 折叠测验 | `details.quiz` | — | ≥3 题；答案含"答错回看 §x" |
| 变式练习 | `.exercise` | — | "先写下预测"留白 → 动手 → 折叠答案（含机制解释）；每模块 ≥1 |
| 填空 | `.fillblank pre code` | — | 仅 M07/M12 等约定模块；`____` 占位 + 折叠分步提示 |
| 验收块 | `.accept` | 绿框 ✅ | 命令 + 可判定绿灯（"3 passed"而非"正常"）+ 排查折叠；每模块 ≥1 |
| 里程碑 | `.milestone` | 🏁 徽章 | 仅 M09/M16/M22；清单项绑定具体绿灯 |
| 概念图 | `pre.ascii` | — | ASCII 依赖图/数据流图，模块头后或"地图"小节 |
| 溯源标注 | `.prov` | 等宽小字 | `[存档 路径 · 环境]` 或 `[作者实测 · 环境指纹]` |

### 3.3 代码与命令规范

- 代码块：`.code`（浅底 + 文件路径标头 + 语言标签）承载源码；`.term`（GitHub dark 终端底）承载命令与真实输出，命令以 `$ ` 前缀。**语法高亮由 `assets/course.js` 内置高亮器自动完成**（python/bash/yaml/json，按 `.code-head .lang` 识别）——写课者直接放纯文本代码即可，无需手工标注着色 span；含 `.fb-blank` 填空或 `data-nohighlight` 的块自动跳过。**token 配色为极简三色**（源自 outlierDet 报告的代码框字体设计）：灰注释 `#7a8089`、绿字符串 `#0f7b4f`、加粗蓝关键字 `var(--accent)`；数字/函数名保持正文色。字体 `13px/1.75` 等宽。所有代码块（含终端块）右上角有**悬停浮现的"复制"按钮**（course.js 注入）。
- 张量操作必须带 shape 注释 `# (B, T, C)`；关键行注释写"为什么"。
- 单块 ≤40 行，超长拆步；文字叙述中引用 reborn 文件写 `Reproduce/minimind_reborn/...`，`<a href>` 链接则用 `../../Reproduce/minimind_reborn/...`（课程页位于 course/minimind_reborn_tutorial/，两级回到仓库根）。
- **数字溯源**：页内每个数字（loss/显存/吞吐/参数量）须有 `.prov` 溯源；作者实测数字来自写作时真实运行（环境指纹：`2×RTX 4090 · torch 2.14.0+cu130 · Python 3.12`）。

### 3.4 bug 猎场（真实缺陷教学）

素材全部来自 reborn git 史与实测（索引见 §2 表 🔍 标记；完整清单借自旧蓝图 syllabus §7）。呈现格式：症状（学习者会看到什么）→ 定位过程（读日志/git show）→ 修复 commit → 回归锁定 → 通用教训。**已实测发现的新猎场**：`tools/chat.py` 与 `inference/engine.py` 的 `generate_tokens` 签名不匹配（M02 呈现、M20 深修）——测试桩与真实入口之间的缝隙。

### 3.5 语言与排版

中文正文、中英之间空格、术语首现标英文；正文 16.5px/1.8；emoji 只在法定标记处出现。

## 4. 视觉规范（tokens v2：blueprint 移植 × Coursera 布局）

### 4.1 Design Tokens v2（blueprint 明色基底；暗色全套见 §4.6）

```css
--canvas: #ffffff;        /* 主内容底 */
--subtle: #f3f5f8;        /* 侧栏/代码块底/表头 */
--border: #d6dae1;        /* 边线 */
--border-muted: #8b929e;
--fg: #16181d;            /* 正文 */
--fg-muted: #59636e;      /* 次要文字 */
--accent: #0969da;        /* 链接/进行中/交互 */
--accent-subtle: #ddf4ff;
--success: #1a7f37; --success-subtle: #dafbe1;   /* 完成/验收/绿灯 */
--attention: #9a6700; --attention-subtle: #fff8c5; /* 坑卡/警告 */
--danger: #cf222e;   --danger-subtle: #ffebe9;     /* 红线/失败输出 */
--done: #8250df;     --done-subtle: #fbefff;       /* 工业对照/里程碑 */
--term-bg: #0d1117; --term-fg: #e6edf3;            /* 终端块（GitHub dark） */
字体：-apple-system,"Segoe UI","Noto Sans","PingFang SC","Microsoft YaHei",sans-serif
等宽：ui-monospace,"SF Mono","Cascadia Code",Consolas,monospace
```

### 4.2 布局（Coursera 式）

```
┌ topbar（sticky 56px）──────────────────────────────────────┐
│ ☰ │ 课程名 · 当前篇 │ 进度 3/22 ▓▓░░ │ 🏠课程地图 │ GitHub ↗ │
├─ sidebar(268px) ─┬─ main（860px，白底）────┬─ toc(220px) ────┤
│ 第一篇 开箱·用    │  模块头 + 元数据 chips  │ (≥1440px 显示)  │
│  ● M01 ✓        │  正文…                 │ 页内 h2/h3 锚点 │
│  ● M02 ✓        │                        │                 │
│  ○ M03 ← 当前    │                        │                 │
│ 第二篇 模型·造   │                        │                 │
│  ○ M04 …        │                        │                 │
│  …（篇可折叠）   │                        │                 │
└──────────────────┴────────────────────────┴─────────────────┘
```

- 侧栏：`--subtle` 底；模块项 = 完成圈（✓ 绿 / ○ 空心，点击切换，localStorage 持久化）+ 编号+标题；当前项左侧 3px accent 竖条 + 加粗；篇标题可折叠（默认展开当前篇）。
- 待成文模块：显示但置灰 + "待成文"徽章，不可点击。
- 移动端（<1024px）：侧栏收为抽屉（topbar ☰ 触发）；toc 隐藏。
- index.html：Coursera 课程落地页式——hero（课程名/一句话/时长/进度）+ 快速开始 + 六篇模块网格卡（含完成态）+ FAQ。

### 4.3 共享资源

- `assets/course.css`：全部样式（**页面禁止内联样式**，QA 检查 `style=` 属性）。
- `assets/course.js`：`COURSE` 数组（单一事实源：id/file/part/title/minutes/status/milestone/hasFillblank）+ 侧栏渲染 + 进度 + toc + 抽屉。
- MathJax 3：仅含公式页面加载（`assets/mathjax.html` 片段约定），断网时显示 LaTeX 原文不塌版（页脚明示）。

### 4.6 设计 tokens v2（blueprint 移植）与暗色模式

v2 起，`course.css` 的 tokens 基底移植自 answer-me-with-html 的 blueprint 主题（工程蓝图风），并新增明/暗/自动三态：

- **三态机制**：`<html data-mode="dark">` 显式暗色；`data-mode="light"` 显式明亮；未设置（默认）= 自动，由 CSS `prefers-color-scheme` 原生生效（JS 加载前零闪烁）。topbar 的 ◐/☀/☾ 按钮三态循环，`localStorage["mrt-theme-v1"]` 持久化。两个暗色 token 块（显式 + auto 媒体查询）**必须同步维护**（CSS 无变量 include）。
- **移植映射**（明色）：中性色与主色取 blueprint——`--subtle #f3f5f8 / --border #d6dae1 / --fg #16181d / --fg-muted #4b5260 / --accent #1d5fbf`；暗色全套取 blueprint dark（`--canvas #0d1c31 / --accent #6ea8ff` 等，以 course.css 为准）。
- **有意偏离 blueprint 两处**：① 四语义色保留课程 hue（绿=验收/完成、黄=坑、红=红线、紫=工业/里程碑——教学语义优先，blueprint 的 ok=accent 会压平五卡语义），暗色下按对比度重调；② 圆角保留 6px 体系（blueprint 为直角），维持教程"教科书"质感。
- **新增变量**：`--code-c/--code-s`（代码三色随态切换）、`--topbar-bg/--acc-head-bg/--overlay/--drawer-shadow/--card-hover-shadow/--on-accent/--on-success`（暗色覆盖的派生表面色与"色上文字"色）。终端块恒为 GitHub dark，两态共用。
- **写作约束不变**：页面禁止内联样式；改 tokens 后全量递增资源版本号并重跑 build_codeview.py（同步 codeview 页引用）。

## 5. 工作流与 QA

### 5.1 写作 SOP

1. 从 COURSE 数组与本文 §2 表取模块规格；
2. **先跑通**：页面上每条命令作者真实执行，输出留档（写入页面即溯源）；
3. 按 §3.1 骨架成文，元素符合 §3.2/3.3 规范；**仓库文件链接一律写原始形式 `../../Reproduce/minimind_reborn/...`**；
4. 过 §5.2 自查清单；
5. 跑 `python3 tools/build_codeview.py`（把仓库链接转为站内代码查看页并生成对应 view 页）→ `python3 tools/check_course.py` 全绿后入库。
   - codeview 页特性：面包屑 + 行号列 + 每行 `#L42` 锚点 + 三色高亮 + 复制按钮；>200KB 文件截断提示；
   - 改动 course.css/js 后：全量递增页面资源版本号 `?v=N`，并重跑 build_codeview.py（它会同步 view 页的版本引用）。

### 5.2 自查清单（每模块）

- [ ] 元数据条完整（前置/时长/新概念/对照文件）
- [ ] 学完你能 3~5 条动词开头
- [ ] 坑卡 ≥1（或声明"本课刻意没有坑"）；溯源标注覆盖全部数字
- [ ] 折叠测验 ≥3 且有"答错回看"；变式练习含"先预测"
- [ ] 验收块绿灯可判定；命令为作者实测
- [ ] **术语首现逐词自查**：每个术语要么当场解释（A），要么显式桥接"XX 模块讲"（B），零裸用（零基础受众）
- [ ] 导语第一句承接上一课；结尾一句话钩住下一课
- [ ] 词汇表：本课新术语 ≥3 条
- [ ] 上/下一课链接正确；无内联样式；标题层级 h1→h2→h3

### 5.3 `tools/check_course.py` 契约

- **断链**：所有 `<a href>` 本地目标存在；页内锚点有对应 id。
- **nav 一致**：每页声明的 module id 与 COURSE 数组一致；prev/next 链与数组顺序一致。
- **正文路径引用**：扫描 `Reproduce/minimind_reborn/...` 形态的代码引用并核对文件存在（豁免：学习者产出路径、通配符）。
- **codeview 链接**：页面不得残留未转换的 `../../Reproduce` href；每个 codeview 链接目标存在；codeview 生成页结构（cv-topbar/data-lines）与资源版本一致性。
- **结构必备**：每模块页含 `.unit-head`/`.accept`/`details.quiz`/`.exercise`/`.glossary`/`.pager`；无 `style=` 内联属性；`<pre>` 内不嵌 `<div>`。
- **索引同步**：index.html 的模块卡数量/状态与 COURSE 一致。

退出码非 0 即失败，CI 可直接用。
