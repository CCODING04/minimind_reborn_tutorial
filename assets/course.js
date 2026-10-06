/* ============================================================
   minimind_reborn_tutorial · 课程引擎
   COURSE 数组是模块清单的单一事实源（DESIGN.md §2 与之冲突时以本数组为准）。
   渲染：侧栏课程树 / 顶栏进度 / 页内目录 / 页间导航 / 完成状态(localStorage)。
   ============================================================ */

"use strict";

/* —— 模块清单（唯一事实源）——
   status: done=已成文 | planned=待成文
   milestone: 'A'|'B'|'C'（可选）  minutes: 预计学习分钟数 */
const PARTS = [
  { id: 0, name: "第 0 篇 · 准备" },
  { id: 1, name: "第一篇 · 开箱：用" },
  { id: 2, name: "第二篇 · 模型：造" },
  { id: 3, name: "第三篇 · 数据" },
  { id: 4, name: "第四篇 · 训练" },
  { id: 5, name: "第五篇 · 推理与评测" },
  { id: 6, name: "第六篇 · 毕业" },
];

const COURSE = [
  { id: "m00", file: "m00_week0.html", part: 0, title: "Week-0 准备：环境与前置自测", minutes: 40, status: "done" },
  { id: "m01", file: "m01_get_started.html", part: 1, title: "课程导览与第一个绿灯", minutes: 45, status: "done" },
  { id: "m02", file: "m02_chat.html", part: 1, title: "与 64M 模型对话", minutes: 45, status: "done" },
  { id: "m03", file: "m03_pipeline.html", part: 1, title: "全链路地图：一次请求的一生", minutes: 50, status: "done" },
  { id: "m04", file: "m04_config.html", part: 2, title: "配置即实验：分域 schema 与守门人", minutes: 60, status: "done" },
  { id: "m05", file: "m05_layers.html", part: 2, title: "地基件：RMSNorm 与 RoPE", minutes: 75, status: "done" },
  { id: "m06", file: "m06_kv_cache.html", part: 2, title: "KV Cache：预分配设计", minutes: 60, status: "done" },
  { id: "m07", file: "m07_attention.html", part: 2, title: "Attention：GQA、QK-Norm 与三条路径", minutes: 80, status: "done", fill: true },
  { id: "m08", file: "m08_ffn_moe.html", part: 2, title: "FFN：SwiGLU 与 MoE", minutes: 70, status: "done" },
  { id: "m09", file: "m09_model.html", part: 2, title: "组装 ForCausalLM：参数账本与官方权重互通", minutes: 85, status: "done", milestone: "A" },
  { id: "m10", file: "m10_tokenizer.html", part: 3, title: "tokenizer 与 chat 模板", minutes: 55, status: "done" },
  { id: "m11", file: "m11_data_access.html", part: 3, title: "数据访问层：registry 与字节偏移索引", minutes: 60, status: "done" },
  { id: "m12", file: "m12_datasets.html", part: 3, title: "Dataset 家族与 loss_mask", minutes: 80, status: "done", fill: true },
  { id: "m13", file: "m13_trainer.html", part: 4, title: "训练基座：Trainer 生命周期与可观测", minutes: 80, status: "done" },
  { id: "m14", file: "m14_checkpoint.html", part: 4, title: "checkpoint 六件套与续训", minutes: 70, status: "done" },
  { id: "m15", file: "m15_ddp.html", part: 4, title: "DDP 与 no_sync", minutes: 70, status: "done" },
  { id: "m16", file: "m16_pretrain_sft.html", part: 4, title: "pretrain / SFT 实战", minutes: 90, status: "done", milestone: "B" },
  { id: "m17", file: "m17_dpo.html", part: 4, title: "DPO：偏好对齐", minutes: 110, status: "done" },
  { id: "m18", file: "m18_distill_lora.html", part: 4, title: "蒸馏与 LoRA：小成本改造", minutes: 95, status: "done" },
  { id: "m19", file: "m19_rl.html", part: 4, title: "RL 一览：rollout 的世界", minutes: 105, status: "done" },
  { id: "m20", file: "m20_inference.html", part: 5, title: "推理栈深潜：采样、停止与引擎", minutes: 85, status: "done" },
  { id: "m21", file: "m21_eval.html", part: 5, title: "评测与官方对照", minutes: 65, status: "done" },
  { id: "m22", file: "m22_graduation.html", part: 6, title: "全链路与毕业实验", minutes: 150, status: "done", milestone: "C" },
];

/* 附录页：不进 COURSE 序列（无 pager、不计入 23 模块进度），侧栏与主页地图单独渲染 */
const APPENDIX = [
  { id: "btrack", file: "btrack.html", title: "B 轨 · 自写挑战总纲（🏁D）" },
  { id: "pytorch-guide", file: "pytorch_guide.html", title: "PyTorch 特殊用法指南" },
];

/* —— 进度（localStorage）—— */
const PROGRESS_KEY = "mrt-progress-v1";
function loadProgress() {
  try { return new Set(JSON.parse(localStorage.getItem(PROGRESS_KEY) || "[]")); }
  catch { return new Set(); }
}
function saveProgress(set) {
  try { localStorage.setItem(PROGRESS_KEY, JSON.stringify([...set])); } catch { /* 隐私模式忽略 */ }
}
let progress = loadProgress();

/* —— 主题（明/暗/自动，localStorage）——
   未设置=自动跟随系统（CSS prefers-color-scheme 原生生效，JS 加载前无闪烁）；
   显式 light/dark 写 <html data-mode>，与进度同库持久化。 */
const THEME_KEY = "mrt-theme-v1";
const THEME_META = { auto: ["◐", "自动（跟随系统）"], light: ["☀", "明亮"], dark: ["☾", "暗黑"] };
function currentTheme() { return localStorage.getItem(THEME_KEY) || "auto"; }
function applyTheme() {
  const mode = currentTheme();
  if (mode === "auto") delete document.documentElement.dataset.mode;
  else document.documentElement.dataset.mode = mode;
}
/* —— 排版风格（经典 / 样章，localStorage）—— */
const STYLE_KEY = "mrt-style-v1";
function applyStyle() {
  if (localStorage.getItem(STYLE_KEY) === "pilot") document.documentElement.dataset.style = "pilot";
  else delete document.documentElement.dataset.style;
}
function mountStyleToggle() {
  const bar = document.querySelector(".topbar, .cv-topbar");
  if (!bar || bar.querySelector(".style-toggle")) return;
  const btn = document.createElement("button");
  btn.className = "style-toggle";
  btn.type = "button";
  btn.setAttribute("aria-label", "切换排版风格");
  const refresh = () => {
    const pilot = localStorage.getItem(STYLE_KEY) === "pilot";
    btn.textContent = pilot ? "样章" : "经典";
    btn.title = `排版风格：${pilot ? "样章" : "经典"}（点击切换）`;
  };
  btn.addEventListener("click", () => {
    const pilot = localStorage.getItem(STYLE_KEY) === "pilot";
    if (pilot) localStorage.removeItem(STYLE_KEY);
    else localStorage.setItem(STYLE_KEY, "pilot");
    applyStyle();
    refresh();
  });
  refresh();
  const anchor = bar.querySelector(".theme-toggle");
  if (anchor) anchor.before(btn); else bar.appendChild(btn);
}

function mountThemeToggle() {
  const bar = document.querySelector(".topbar, .cv-topbar");
  if (!bar || bar.querySelector(".theme-toggle")) return;
  const btn = document.createElement("button");
  btn.className = "theme-toggle";
  btn.type = "button";
  btn.setAttribute("aria-label", "切换明暗主题");
  const refresh = () => {
    const [icon, name] = THEME_META[currentTheme()];
    btn.textContent = icon;
    btn.title = `主题：${name}（点击切换）`;
  };
  btn.addEventListener("click", () => {
    const order = ["auto", "light", "dark"];
    const next = order[(order.indexOf(currentTheme()) + 1) % order.length];
    if (next === "auto") localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, next);
    applyTheme();
    refresh();
  });
  refresh();
  const anchor = bar.querySelector(".progress-wrap, .cv-meta");
  if (anchor) anchor.before(btn); else bar.appendChild(btn);
}

const byId = Object.fromEntries(COURSE.map((m) => [m.id, m]));
const currentId = document.body.dataset.module || "";
const currentPage = document.body.dataset.page || "";

/* —— 侧栏课程树 —— */
function renderSidebar() {
  const mount = document.getElementById("sidebar");
  if (!mount) return;
  const frag = document.createDocumentFragment();
  for (const part of PARTS) {
    const mods = COURSE.filter((m) => m.part === part.id);
    const doneCount = mods.filter((m) => progress.has(m.id)).length;
    const wrap = document.createElement("div");
    wrap.className = "part" + (currentId && byId[currentId] && byId[currentId].part !== part.id ? " collapsed" : "");
    const collapsedPref = localStorage.getItem("mrt-part-" + part.id);
    if (collapsedPref === "1") wrap.classList.add("collapsed");
    if (collapsedPref === "0") wrap.classList.remove("collapsed");

    const head = document.createElement("button");
    head.className = "part-head";
    head.innerHTML = `<span class="caret">▼</span><span>${part.name}</span>` +
      `<span class="count">${doneCount}/${mods.length}</span>`;
    head.addEventListener("click", () => {
      wrap.classList.toggle("collapsed");
      localStorage.setItem("mrt-part-" + part.id, wrap.classList.contains("collapsed") ? "1" : "0");
    });
    wrap.appendChild(head);

    const list = document.createElement("div");
    list.className = "part-modules";
    for (const m of mods) {
      const a = document.createElement("a");
      a.className = "mod" + (m.id === currentId ? " current" : "") + (progress.has(m.id) ? " done" : "") + (m.status === "planned" ? " planned" : "");
      if (m.status === "planned" && m.id !== currentId) {
        a.href = "javascript:void(0)";
        a.title = "待成文（首批 M01–M04 已发布，其余模块批量制作中）";
      } else {
        a.href = m.file;
      }
      const num = m.id.replace("m", "M").toUpperCase();
      let tags = "";
      if (m.milestone) tags += `<span class="tag milestone">🏁${m.milestone}</span>`;
      if (m.status === "planned") tags += `<span class="tag">待成文</span>`;
      a.innerHTML = `<span class="done-dot" role="button" tabindex="0" aria-label="标记${num}完成" title="点击切换完成状态"></span>` +
        `<span class="num">${num}</span><span class="label">${m.title}</span>${tags}`;
      const dot = a.querySelector(".done-dot");
      const toggle = (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        if (m.status === "planned") return;
        if (progress.has(m.id)) progress.delete(m.id); else progress.add(m.id);
        saveProgress(progress); syncProgressUI();
      };
      dot.addEventListener("click", toggle);
      dot.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") toggle(ev); });
      list.appendChild(a);
    }
    wrap.appendChild(list);
    frag.appendChild(wrap);
  }
  /* 附录区：纯链接，不进 23 模块进度口径、无 done-dot */
  if (APPENDIX.length) {
    const awrap = document.createElement("div");
    awrap.className = "part appendix";
    const ahead = document.createElement("div");
    ahead.className = "part-head";
    ahead.innerHTML = `<span class="caret">▼</span><span>附录</span>`;
    awrap.appendChild(ahead);
    const alist = document.createElement("div");
    alist.className = "part-modules";
    for (const m of APPENDIX) {
      const a = document.createElement("a");
      a.className = "mod" + (m.id === currentPage ? " current" : "");
      a.href = m.file;
      a.innerHTML = `<span class="num">附</span><span class="label">${m.title}</span>`;
      alist.appendChild(a);
    }
    awrap.appendChild(alist);
    frag.appendChild(awrap);
  }
  mount.innerHTML = "";
  mount.appendChild(frag);
}

/* —— 顶栏进度 —— */
function syncProgressUI() {
  const done = COURSE.filter((m) => progress.has(m.id)).length;
  const el = document.getElementById("topbar-progress");
  if (el) el.textContent = `${done}/${COURSE.length}`;
  const bar = document.querySelector(".topbar .progress-bar i");
  if (bar) bar.style.width = (COURSE.length ? (100 * done) / COURSE.length : 0) + "%";
  const btn = document.getElementById("complete-btn");
  if (btn && currentId) {
    const on = progress.has(currentId);
    btn.classList.toggle("on", on);
    btn.textContent = on ? "✓ 已完成" : "标记本课完成";
  }
  renderSidebar();
  renderIndexGrid();
}

/* —— 页内目录（≥1440px 显示）—— */
function renderToc() {
  const mount = document.getElementById("toc");
  if (!mount) return;
  const heads = [...document.querySelectorAll("main .content h2[id], main .content h3[id]")];
  if (!heads.length) { mount.remove(); return; }
  const box = document.createElement("div");
  box.className = "toc-title";
  box.textContent = "本页目录";
  mount.appendChild(box);
  for (const h of heads) {
    const a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = h.textContent.replace(/\s*#$/, "");
    a.className = h.tagName === "H3" ? "l3" : "";
    a.setAttribute("data-target", h.id);
    mount.appendChild(a);
  }
  const links = [...mount.querySelectorAll("a[data-target]")];
  const spy = () => {
    let current = null;
    for (const h of heads) {
      if (h.getBoundingClientRect().top < 100) current = h.id;
    }
    for (const a of links) a.classList.toggle("current", a.dataset.target === current);
  };
  document.addEventListener("scroll", spy, { passive: true });
  spy();
}

/* —— 页间导航 —— */
function renderPager() {
  const mount = document.getElementById("pager");
  if (!mount || !currentId || !byId[currentId]) return;
  const idx = COURSE.findIndex((m) => m.id === currentId);
  const prev = COURSE[idx - 1], next = COURSE[idx + 1];
  const mk = (m, dir, cls) => {
    if (!m || m.status === "planned") {
      const d = document.createElement("div");
      d.className = "pg disabled " + (cls || "");
      d.innerHTML = `<span class="pg-dir">${dir}</span><span class="pg-title">${m ? m.title + "（待成文）" : "已是首课/末课"}</span>`;
      return d;
    }
    const a = document.createElement("a");
    a.className = "pg " + (cls || "");
    a.href = m.file;
    a.innerHTML = `<span class="pg-dir">${dir}</span><span class="pg-title">${m.title}</span>`;
    return a;
  };
  mount.appendChild(mk(prev, "← 上一课"));
  const btn = document.createElement("button");
  btn.id = "complete-btn";
  btn.className = "complete-btn";
  btn.textContent = progress.has(currentId) ? "✓ 已完成" : "标记本课完成";
  btn.addEventListener("click", () => {
    if (progress.has(currentId)) progress.delete(currentId); else progress.add(currentId);
    saveProgress(progress); syncProgressUI();
    btn.scrollIntoView({ block: "nearest" });
  });
  mount.appendChild(btn);
  mount.appendChild(mk(next, "下一课 →", "next"));
}

/* —— 移动端抽屉（Esc 可关、aria 状态同步、开时锁背景滚动） —— */
function initDrawer() {
  const toggle = document.querySelector(".topbar .nav-toggle");
  if (!toggle) return;
  const overlay = document.createElement("div");
  overlay.className = "nav-overlay";
  overlay.addEventListener("click", () => setOpen(false));
  document.body.appendChild(overlay);
  const setOpen = (open) => {
    document.body.classList.toggle("nav-open", open);
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
  };
  toggle.setAttribute("aria-expanded", "false");
  toggle.addEventListener("click", () => setOpen(!document.body.classList.contains("nav-open")));
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && document.body.classList.contains("nav-open")) setOpen(false);
  });
}

/* —— 标题锚点 —— */
function initHeadingAnchors() {
  for (const h of document.querySelectorAll("main .content h2[id], main .content h3[id]")) {
    const a = document.createElement("a");
    a.className = "h-anchor";
    a.href = "#" + h.id;
    a.textContent = "#";
    h.appendChild(a);
  }
}

/* —— index 课程主页：模块网格（由 COURSE 生成，保证与侧栏一致）—— */
function renderIndexGrid() {
  const mount = document.getElementById("parts-grid");
  if (!mount) return;
  const frag = document.createDocumentFragment();
  for (const part of PARTS) {
    const mods = COURSE.filter((m) => m.part === part.id);
    const doneCount = mods.filter((m) => progress.has(m.id)).length;
      const card = document.createElement("div");
      card.className = "part-card";
      const desc = {
        0: "正式开学前的 Week-0：三档受众自测、最小工具箱清点与三盏版本绿灯——零门槛入口，40 分钟决定你从哪一档进入正课。",
        1: "先把 64M 模型跑起来、玩起来，画出全链路黑箱地图——后面每一课都在解剖其中一站。",
        2: "自底向上读模型源码：配置 → 地基件 → KV Cache → 注意力 → FFN/MoE → 整机组装，终点是官方权重 strict 互通。",
        3: "语料如何变成样本：tokenizer 与 chat 模板、字节偏移索引、五类数据集与全系统最脆的 loss_mask。",
        4: "从最小训练循环到八种范式：Trainer 基座、断点续训、DDP，然后 pretrain→SFT→DPO→蒸馏/LoRA→RL 一路走完。",
        5: "白箱拆开推理栈：采样纯函数链、三重停止、引擎门面；再用评测给模型打分、与官方对照。",
        6: "一条命令跑通全链路，做一个有数据支撑的小实验，拿走你自己训出的模型——硬件档位决定它是 tiny 还是 64M，见下表。",
      }[part.id] || "";
      const items = mods.map((m) => {
        const num = m.id.replace("m", "M").toUpperCase();
        const done = progress.has(m.id);
        const cls = done ? "done" : m.status === "planned" ? "planned" : "";
        const label = m.status === "planned"
          ? `<span>${m.title} <span class="tag">待成文</span></span>`
          : `<a href="${m.file}">${m.title}</a>`;
        const ms = m.milestone ? ` <span class="tag milestone">🏁 ${m.milestone}</span>` : "";
        return `<li class="${cls}" data-mid="${m.id}"><span class="done-dot" title="点击切换完成状态"></span><span><b class="mod-num">${num}</b> ${label}${ms}</span></li>`;
      }).join("");
      card.innerHTML = `<div class="pc-head"><span class="pc-num">P${part.id}</span>` +
        `<span class="pc-name">${part.name}</span>` +
        `<span class="pc-count">${doneCount}/${mods.length}</span></div>` +
        `<p class="pc-desc">${desc}</p><ul class="pc-mods">${items}</ul>`;
      // 圆点与侧栏同款：点击/回车切换完成态
      card.querySelectorAll("li .done-dot").forEach((dot) => {
        const mid = dot.closest("li").dataset.mid;
        dot.setAttribute("role", "button");
        dot.setAttribute("tabindex", "0");
        const flip = (ev) => {
          ev.preventDefault(); ev.stopPropagation();
          if (!mid || !byId[mid] || byId[mid].status === "planned") return;
          if (progress.has(mid)) progress.delete(mid); else progress.add(mid);
          saveProgress(progress); syncProgressUI();
        };
        dot.addEventListener("click", flip);
        dot.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") flip(ev); });
      });
      frag.appendChild(card);
  }
  /* 主页地图的附录卡（与侧栏同源 APPENDIX） */
  if (APPENDIX.length) {
    const card = document.createElement("div");
    card.className = "part-card appendix";
    const items = APPENDIX.map((m) =>
      `<li><span><b class="mod-num">附</b> <a href="${m.file}">${m.title}</a></span></li>`
    ).join("");
    card.innerHTML = `<div class="pc-head"><span class="pc-num">附</span>` +
      `<span class="pc-name">附录</span></div>` +
      `<p class="pc-desc">主线之外的两份常备参考：八站自写挑战轨道（终点 🏁D 自己的模型说话）与课程用到的 PyTorch 特殊用法即查手册。</p>` +
      `<ul class="pc-mods">${items}</ul>`;
    frag.appendChild(card);
  }
  mount.innerHTML = "";
  mount.appendChild(frag);
}

/* —— 内置语法高亮（离线可用；对 pre.code > code 文本着色，统一重建标记）——
   颜色由 course.css 的 .k/.s/.n/.f/.c 提供。含 .fb-blank 填空或 data-nohighlight
   的代码块跳过（保住手工标记）。语言取 .code-head .lang 文本，缺省 python。 */
const HL_KEYWORDS = {
  python: "def class return if elif else for while in not and or is None True False import from as with try except finally raise yield lambda pass break continue global nonlocal assert async await self match case",
  bash: "if then else elif fi for in do done while until case esac function export source return local echo cd set unset shift read printf ls mkdir rm cp mv cat grep sed awk curl pip python python3 uv make git torchrun",
  yaml: "true false null on off yes no",
  json: "true false null",
};
const HL_TOKEN = {
  python: /(?<s>"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(?<c>#[^\n]*)|(?<dec>@[\w.]+)|(?<n>\b(?:0[xXbBoO][\da-fA-F_]+|\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?j?)\b)|(?<id>[A-Za-z_]\w*)/g,
  bash: /(?<s>"(?:\\.|[^"\\\n])*"|'(?:[^'\\\n])*')|(?<c>#[^\n]*)|(?<v>\$\{?\w+\}?)|(?<n>\b\d+\b)|(?<id>[A-Za-z_][\w.-]*)/g,
  yaml: /(?<s>"(?:\\.|[^"\\\n])*"|'(?:[^'\n])*')|(?<c>#[^\n]*)|(?<key>^[ \t]*-?[ \t]*[\w."-]+(?=:))|(?<n>\b\d+(?:\.\d+)?\b)|(?<id>[A-Za-z_]\w*)/gm,
  json: /(?<s>"(?:\\.|[^"\\\n])*")|(?<c>\/\/[^\n]*)|(?<n>\b-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|(?<id>[A-Za-z_]\w*)/g,
};
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
/* 把源码切成 [{cls, text}] 片段（cls=null 为正文） */
function hlSegments(text, lang) {
  const rx = HL_TOKEN[lang];
  rx.lastIndex = 0;
  const kws = new Set(HL_KEYWORDS[lang].split(" "));
  const segs = [];
  let last = 0, m;
  const push = (cls, s) => { if (s) segs.push({ cls, text: s }); };
  while ((m = rx.exec(text)) !== null) {
    const g = m.groups;
    push(null, text.slice(last, m.index));
    if (g.s) push("s", m[0]);
    else if (g.c) push("c", m[0]);
    else if (g.dec || g.v || g.key) push("f", m[0]);
    else if (g.n) push("n", m[0]);
    else if (g.id) {
      const after = text[m.index + m[0].length];
      if (kws.has(m[0])) push("k", m[0]);
      else if (after === "(") push("f", m[0]);
      else push(null, m[0]);
    } else push(null, m[0]);
    last = m.index + m[0].length;
  }
  push(null, text.slice(last));
  return segs;
}
function emitInline(segs) {
  return segs.map((s) => (s.cls ? `<span class="${s.cls}">${esc(s.text)}</span>` : esc(s.text))).join("");
}
/* 逐行模式：每行包成 <span class="cl" id="Ln">，供 codeview 行号与 #Ln 锚点 */
function emitLines(segs) {
  let out = "", line = 1;
  const openLine = () => { out += `<span class="cl" id="L${line}">`; };
  const closeLine = () => { out += "</span>"; line++; };
  openLine();
  for (const seg of segs) {
    const parts = seg.text.split("\n");
    parts.forEach((p, i) => {
      if (i > 0) { closeLine(); openLine(); }
      if (!p) return;
      out += seg.cls ? `<span class="${seg.cls}">${esc(p)}</span>` : esc(p);
    });
  }
  closeLine();
  return out;
}
function highlightCode() {
  for (const pre of document.querySelectorAll("pre.code")) {
    const code = pre.querySelector("code");
    if (!code || code.querySelector(".fb-blank") || pre.hasAttribute("data-nohighlight")) continue;
    const block = pre.closest(".codeblock");
    const langRaw = block ? (block.querySelector(".code-head .lang") || {}).textContent : "";
    const langRawTrim = (langRaw || "").toLowerCase().trim();
    if (langRawTrim === "text") continue;   // 纯文本：不着色
    const lang = { python: "python", py: "python", bash: "bash", sh: "bash", shell: "bash", yaml: "yaml", yml: "yaml", json: "json" }[langRawTrim] || "python";
    const text = code.textContent;
    const segs = hlSegments(text, lang);
    code.innerHTML = code.dataset.lines === "1"
      ? emitLines(segs)
      : emitInline(segs);
  }
}

/* —— 代码块复制按钮（悬停浮现；复制成功显示"已复制"1.5s）—— */
function copyTextFallback(text, done) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand("copy"); done(); } catch { /* 忽略 */ }
  document.body.removeChild(ta);
}
function initCopyButtons() {
  for (const pre of document.querySelectorAll("pre.code, pre.term, pre.ascii")) {
    if (pre.querySelector(".copy-btn")) continue;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.textContent = "复制";
    btn.addEventListener("click", () => {
      const codeEl = pre.querySelector("code");
      // $ 提示符标记了 user-select:none，但 innerText/textContent 仍会包含它——
      // 克隆后剔除 .p 再取文本，保证"所见（可选中）即所复制"
      let text;
      if (codeEl) {
        const clone = codeEl.cloneNode(true);
        clone.querySelectorAll(".p").forEach((n) => n.remove());
        text = clone.textContent;
      } else {
        text = pre.innerText;
      }
      text = text.trimEnd();
      const done = () => {
        btn.textContent = "已复制";
        setTimeout(() => { btn.textContent = "复制"; }, 1500);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, () => copyTextFallback(text, done));
      } else {
        copyTextFallback(text, done);
      }
    });
    pre.appendChild(btn);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  applyTheme();
  applyStyle();
  mountThemeToggle();
  mountStyleToggle();
  initDrawer();
  renderSidebar();
  renderPager();        // 先创建按钮，再同步状态（syncProgressUI 会设置按钮文字）
  syncProgressUI();
  initHeadingAnchors();
  highlightCode();
  initCopyButtons();
  renderToc();
  renderIndexGrid();
});
