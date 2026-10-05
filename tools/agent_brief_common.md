# 模块写作公共规范（agent 必读）

> 本文件是课程模块页写作的完整规范。写任何模块前：通读本文件 → 读 DESIGN.md → 打开 m02_chat.html 与 m04_config.html 作为**骨架与文风的唯一参照**（结构完全同构）。写完必须自查并报告。

## 页面骨架（逐字照抄参照页的结构）

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{M}XX · {标题} | 跟着 minimind_reborn 造一个大模型</title>
<link rel="stylesheet" href="assets/course.css?v=2">
（含公式则加 MathJax 配置 + CDN script，照抄 m01/m02 的写法）
</head>
<body data-module="mXX">
<header class="topbar">…（照抄，改 crumb 篇名）</header>
<div class="layout">
  <aside class="sidebar" id="sidebar"></aside>
  <main class="main"><div class="content">
    <header class="unit-head">
      <div class="kicker"><span class="badge-m">MXX</span><span>第 N 篇 · 篇名</span>（里程碑模块加 <span class="badge-ms">🏁X</span>）</div>
      <h1>标题</h1>
      <p class="lede">30~60 字导语：本课在旅程中的位置 + 承接上一课</p>
      <div class="meta">
        <span class="chip">前置：<b>MXX</b></span>
        <span class="chip">时长：<b>约 NN 分钟</b></span>
        <span class="chip">新概念：<b>≤2 个</b></span>
        <span class="chip">对照：<code>文件列表</code></span>
      </div>
    </header>
    <h2 id="objectives">学完你能</h2>（4~5 条动词开头）
    …正文…
    <h2 id="quiz">检验一下</h2>（details.quiz ×≥3）
    <h2 id="exercise">变式练习</h2>（.exercise ×≥1）
    <div class="accept" id="accept">…</div>
    <section class="glossary"><h2>本课词汇</h2><dl>…≥6 条</dl></section>
    <nav class="pager" id="pager"></nav>
  </div></main>
  <aside class="toc" id="toc"></aside>
</div>
<script src="assets/course.js?v=2"></script>
</body>
</html>
```

## 硬规则（QA 会拦，违反即返工）

1. **零内联样式**：表格列宽用 `u-w120/u-w130/u-w200/u-w44` 类；其他样式一律用 assets/course.css 现有类。
2. **代码块结构**：`<div class="codeblock"><div class="code-head"><span>标头</span><span class="lang">python|bash|yaml|json</span></div><pre class="code"><code>纯文本代码</code></pre></div>`。code-head 不得嵌进 pre。**代码放纯文本即可**——语法高亮由 course.js 自动完成（灰注释/绿字符串/加粗蓝关键字），**不要手工加着色 span**。
3. **填空块**（仅 m07/m12 等约定模块）：在 code 内用 `<span class="fb-blank">____</span>` 标空位；此类块自动跳过高亮。
4. **终端块**：`<pre class="term"><code><span class="p">$ </span><span class="cmd">命令</span>\n<span class="out">输出</span></code></pre>`；报错行 `.err`、成功行 `.ok`、弱化行 `.dim`。终端块的 span 是**必须手工写的**（高亮器不管 .term）。
5. **仓库文件链接**：href 写原始形式 `<a href="../../Reproduce/minimind_reborn/相对路径">`（构建脚本会转为站内代码查看页）；文字叙述仍写 `Reproduce/minimind_reborn/...` 或文件名。
6. **HTML 转义**：代码/终端里的 `<` `>` `&` 必须转义。
7. **张量操作带 shape 注释** `# (B, T, C)`；单块代码 ≤40 行；关键行注释写"为什么"。
8. h2/h3 全部带 id；quiz 答案含 `答错回看 <a href="#节id">`；exercise 含 `<div class="ex-predict"></div>` 先预测再观察、答案在 details 折叠且含机制解释。
9. **数字溯源**：每个数字附 `<p class="prov"><b>[作者实测 · 2026-09-30 · 2×RTX 4090 · torch 2.14.0+cu130]</b> 说明…</p>` 或 `[存档 runs/… · 环境]`（引用 reborn 历史产物时）。**页面上每条命令你都必须真实跑过**，输出贴页面；跑不了的（如双卡实测）用存档并标注。
10. 长度 25~35KB；中文中英间空格；文风直接具体、每句话有信息量；emoji 只在法定标记（💡📐⚠️🛑🏭🪜🏁🔍🎮✏️）处出现。

## 连贯性纪律（零基础受众，本轮强化）

- **每个术语首现**必须二选一：A=当场给一句话解释/括号定义；B=显式桥接"XX 模块详讲，本课只需知道…"。**不允许裸用**。自查时逐词过一遍（包括你以为人尽皆知的词：logits、严格加载、上/下文、梯度、分词……受众只保证会 Python 基础 PyTorch）。
- **开头承接**：导语第一句回指上一课的产出/结论；**结尾预告**：验收块后或 glossary 前一句话钩住下一课（ Pager 的下一课链接之外）。
- 新概念 ≤2 个/模块；重推导放 `<details>` 折叠深读卡，开头标注"可跳过，不影响主线"。
- 前面模块已讲过的词不要重复开课，直接用（可在词汇表标注"见 MXX"）。

## 教学元素用法（与参照页一致）

- 💡 tip callout（直觉）、📐 note callout（严谨/推导）、⚠️ warn callout（坑卡：`<ul class="pit-stages">` 里 `sym/cause/fix` 三 li）、🛑 danger callout（红线）、🏭 industry callout（工业对照）。
- 每模块 ≥1 坑卡（没有真实坑就写"本课刻意没有坑，因为…"）、≥1 验收块（绿灯可判定，如 `5 passed`）、≥3 测验、≥1 变式练习。
- 里程碑模块（m09/m16/m22）：`.milestone` 块（flag + 清单），清单项对应具体绿灯。
- 🔍 bug 猎场：症状→定位（git show/日志）→修复→回归锁定→通用教训；素材全部真实（brief 会给）。
- 🎮 选做标注：需要 64M/GPU 的实验标 🎮，同时给无 GPU 的存档分析路线。

## 工作流程

1. 读本文件 + DESIGN.md + 参照页（m02/m04）；读 brief 指定的 reborn 源文件（只读，不得修改仓库；临时脚本放 /tmp）。
2. **先跑通再写**：brief 列的命令逐条真跑（在 Reproduce/minimind_reborn 下 `uv run …`），采集输出。
3. 按骨架成文 → 过"连贯性纪律"逐词自查 → 写完跑 `cd /home/admin02/Code/WorkSpace/PythonFormat/course/minimind_reborn_tutorial && python3 tools/check_course.py`（你的页面 status 可能还是 planned——QA 会告警"文件已存在请改 done"，**你要自己把 assets/course.js 里你负责的模块 status 改为 "done"**，只改你自己那一行）。
4. 最终报告：文件字节数 / 每条必跑命令的真实输出摘要 / 术语首现自查清单（词→A/B 类处理方式）/ QA 结果。
