# 跟着 minimind_reborn 造一个大模型 · 交互式课程

以工程化重构的 [minimind_reborn](https://github.com/CCODING04/minimind-reborn)（64M 中文小模型：pretrain → SFT → DPO → 蒸馏/LoRA → RL → 推理服务）为唯一实现目标的 LLM 全栈实训课——**跟读真实工程代码、跑通训练全链路、用 93 个单元测试做绿灯验收**。

- **23 个模块 / 7 篇**（Week-0 准备 → 开箱 → 模型 → 数据 → 训练 → 推理评测 → 毕业），三座里程碑（🏁A 官方权重 strict 互通 / 🏁B 训出自己的基座 / 🏁C 八范式冒烟+毕业实验）+ **2 个附录**（B 轨·自写挑战总纲，终点 🏁D / PyTorch 特殊用法指南）
- **102 个站内代码查看页**：正文提到的每个源文件都能点开在线阅读（行号、语法高亮、逐行锚点、一键复制）
- **零基础友好**：术语首现必有解释或桥接；每课折叠测验 + 先预测再观察的变式练习 + 可判定的验收块
- **完全离线自足**：公式渲染（MathJax）、样式、脚本全部本地化，无外部资源依赖

## 本地查看

```bash
cd minimind_reborn_tutorial
python3 -m http.server 6006
# 浏览器打开 http://127.0.0.1:6006
```

任意静态托管（nginx / GitHub Pages / 对象存储）直接指向本目录即可。

## 目录结构

```
├── index.html            课程主页（Coursera 式模块地图，进度存浏览器 localStorage）
├── m00_week0.html        Week-0 准备页（三档受众自测 + 最小工具箱 + 版本绿灯）
├── m01..m22 *.html       22 个主线模块页（GitHub light 风格，PC/平板/手机自适应）
├── btrack.html           附录 · B 轨自写挑战总纲（🏁D 自己的模型说话）
├── pytorch_guide.html    附录 · PyTorch 特殊用法指南
├── assets/               设计系统 + 课程引擎（CSS/JS）+ 本地 MathJax + favicon
├── codeview/             102 个源码查看页（镜像仓库目录结构，#L42 逐行锚点）
├── ANALYSIS.md           参照教程（Part7_minimind）的优劣分析与继承决定
├── DESIGN.md             课程设计：模块结构 / 教学约定 / 写作 SOP / QA 契约
└── tools/                构建与检查管线（见下）
```

## 构建与检查管线

| 工具 | 作用 |
|---|---|
| `tools/build_codeview.py` | 扫描课程页引用的源文件 → 生成 `codeview/` 查看页，并把仓库链接改写为站内链接 |
| `tools/check_course.py` | QA：断链 / 锚点 / nav 一致性 / 正文路径存在性 / 结构必备 / 禁内联样式 |
| `tools/panelize.py` | 把页面正文区块幂等包进 `<section class="panel">`（新页/新章节接入时用） |
| `tools/build_package.py` | 生成 `packaged_website/` 单页自包含部署包（CSS/JS 内联 + 源码快照，`file://` 可开） |
| `tools/interact_check.mjs` | 54 项交互/链接/五档视口自动检查（26 页全覆盖，playwright + 本机 chrome） |

改动 CSS/JS 后：全量递增页面资源版本号 `?v=N` → 依次重跑上述脚本。

## 内容与致谢

课程内容与全部实测数据基于 minimind_reborn 项目（官方 minimind 的工程化重构）。上游项目：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)。2026-10-06 修复周期：reborn 侧修复 chat 签名 / 双重温度 / manifest 缓存键 / DPO·蒸馏多卡设备 / 贪心对照 / RL 续训 / 上下文守卫 / 偶数批等 12 项（测试 85→93），课程 bug 猎场章节同步改为"发现→修复→回归锁定"口径。
