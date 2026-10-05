#!/usr/bin/env python3
"""把课程 + minimind_reborn 源码快照打包成可独立部署的静态网站。

输出：packaged_website/
├── index.html / m01..m04.html     课程页（reborn 链接重写为站内 source/ 页面）
├── assets/course.css|js           框架资源（原样，含语法高亮与复制按钮）
├── source.css                     源码浏览页样式（行号/面包屑）
├── source/index.html              源码文件树索引
├── source/<仓库相对路径>.html      每个源文件一页（服务端高亮 + 行号 + 复制）
└── raw/...                        不宜 HTML 化的大文件原样（如 tokenizer.json）

规则：
- 源码页服务端高亮（三色：注释/字符串/关键字，与课程 CSS 现状一致），逐行近似、
  跨行三引号串仅首尾行着色（可接受）；
- 课程页行内 <code>路径</code> 自动链接化：仅精确匹配已注册文件；裸 basename
  仅在唯一时注册（workflow.py/loss.py/chat.py/__init__.py 等重名不注册）；
- 构建完成后自检：包内全部本地 href 目标存在，锚点存在，失败退出码 1。
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

COURSE = Path(__file__).resolve().parent.parent
REBORN = COURSE.parent.parent / "Reproduce" / "minimind_reborn"
OUT = COURSE / "packaged_website"

TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".toml", ".json", ".sh", ".txt", ".cfg"}
TEXT_NAMES = {"Makefile"}
EXCLUDE_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache", ".ruff_cache",
                ".mypy_cache", "node_modules", "out", "checkpoints", "runs", ".zcode"}
RAW_THRESHOLD = 200_000  # 超过 200KB 的文本文件不做 HTML 页，进 raw/

LANG_MAP = {".py": "python", ".yaml": "yaml", ".yml": "yaml", ".json": "json",
            ".sh": "bash", ".toml": "toml", ".md": "md", ".cfg": "toml", ".txt": "md"}
LANG_LABEL = {"python": "python", "yaml": "yaml", "json": "json", "bash": "bash",
              "toml": "toml", "md": "markdown"}
HL_LANGS = {"python", "yaml", "json", "bash"}

SNAPSHOT_DATE = "2026-09-30"

# ── 服务端高亮（与 assets/course.js 的三色规则一致） ──────────────────
PY_KW = ("def class return if elif else for while in not and or is None True False import from as "
         "with try except finally raise yield lambda pass break continue global nonlocal assert "
         "async await self match case").split()
BASH_KW = ("if then else elif fi for in do done while until case esac function export source return "
           "local echo cd set unset shift read printf ls mkdir rm cp mv cat grep sed awk curl pip "
           "python python3 uv make git torchrun").split()
YAML_KW = {"true", "false", "null", "on", "off", "yes", "no"}

TOKEN_RE = {
    "python": re.compile(
        r'(?P<s>[rbf]?"(?:\\.|[^"\\])*"?|[rbf]?\'(?:\\.|[^\'\\])*\'?)|(?P<c>\#[^\n]*)|'
        r'(?P<id>[A-Za-z_]\w*)'),
    "bash": re.compile(
        r"""(?P<s>"(?:\\.|[^"\\])*"?|'[^']*'?)|(?P<c>\#[^\n]*)|(?P<id>[A-Za-z_][\w./-]*)"""),
    "yaml": re.compile(r'(?P<s>"(?:\\.|[^"\\])*"?|\'[^\']*\')|(?P<c>\#[^\n]*)|(?P<id>[A-Za-z_][\w.-]*)'),
    "json": re.compile(r'(?P<s>"(?:\\.|[^"\\])*")|(?P<c>//[^\n]*)|(?P<id>[A-Za-z_]\w*)'),
}

def hl_line(line: str, lang: str) -> str:
    """单行三色高亮：注释灰 / 字符串绿 / 关键字加粗蓝，其余正文色。"""
    if lang not in HL_LANGS:
        return html.escape(line)
    out, pos = [], 0
    for m in TOKEN_RE[lang].finditer(line):
        out.append(html.escape(line[pos:m.start()]))
        g = m.lastgroup
        text = html.escape(m.group(0))
        if g == "s":
            out.append(f'<span class="s">{text}</span>')
        elif g == "c":
            out.append(f'<span class="c">{text}</span>')
        elif g == "id":
            word = m.group(0)
            kws = PY_KW if lang == "python" else BASH_KW if lang == "bash" else YAML_KW
            cls = "k" if word in kws else None
            out.append(f'<span class="k">{text}</span>' if cls else text)
        else:
            out.append(text)
        pos = m.end()
    out.append(html.escape(line[pos:]))
    return "".join(out)

# ── 收集 reborn 文本文件 ──────────────────────────────────────────────
def collect_files() -> list[Path]:
    files = []
    for p in sorted(REBORN.rglob("*")):
        parts = p.relative_to(REBORN).parts
        if not p.is_file() or any(part in EXCLUDE_DIRS for part in parts):
            # 例外：out/ 下的评测报告 .md 是课程引用的存档文档，收进来
            if not ("out" in parts and p.suffix == ".md"):
                continue
        if p.suffix in TEXT_SUFFIXES or p.name in TEXT_NAMES:
            files.append(p)
    return files

# ── 源码浏览页 ────────────────────────────────────────────────────────
def crumb(rel: str, prefix: str) -> str:
    return f'<div class="crumbs"><a href="{prefix}index.html">← 课程首页</a><span class="sep">·</span>' \
           f'<a href="{prefix}source/index.html">源码索引</a><span class="sep">·</span>' \
           f'<span class="path">{rel}</span><span class="snap">reborn 快照 @{SNAPSHOT_DATE}</span></div>'

PAGE_SHELL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} · minimind_reborn 源码</title>
<style>{css}</style>
<style>{source_css}</style>
</head>
<body data-page="source">
<header class="topbar">
  <a class="brand" href="{prefix}index.html"><span class="logo">MR</span>
  <span>minimind_reborn 教程<span class="crumb"> · 源码浏览</span></span></a>
  <div class="progress-wrap"><a class="map-link" href="{prefix}index.html#map">课程地图</a></div>
</header>
<div class="src-page">
{body}
<footer class="course-footer">minimind_reborn 源码快照 · HEAD 06db33c (2026-09-28) + 工作区修改 · 打包于 {date} · 仅用于课程对照阅读</footer>
</div>
<script>{js}</script>
</body>
</html>
"""

def build_source_page(rel: str, src: Path, css: str, js: str, source_css: str) -> str:
    lang = LANG_MAP.get(src.suffix, "md")
    prefix = "../" * (rel.count("/") + 1)   # source/src/minimind_reborn/… 需要 5 级
    text = src.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    body_parts = [crumb(rel, prefix)]
    body_parts.append(
        f'<div class="codeblock"><div class="code-head"><span>{html.escape("Reproduce/minimind_reborn/" + rel)}</span>'
        f'<span class="lang">{LANG_LABEL.get(lang, "text")} · {len(lines)} 行</span></div>'
        # 服务端已高亮；data-nohighlight 阻止客户端高亮器重写 innerHTML（会抹掉行号 span）
        f'<pre class="code src" data-nohighlight><code>')
    for line in lines:
        body_parts.append(f'<span class="srcline">{hl_line(line, lang)}</span>')
    body_parts.append("</code></pre></div>")
    return PAGE_SHELL.format(title=html.escape(rel), prefix=prefix, css=css, js=js,
                             source_css=source_css, body="".join(body_parts), date=SNAPSHOT_DATE)

def build_source_index(files: list[tuple[str, Path]], css: str, js: str, source_css: str) -> str:
    groups: dict[str, list[tuple[str, int]]] = {}
    for rel, src in files:
        parts = Path(rel).parts
        key = "/".join(parts[:3]) if parts[0] == "src" else parts[0]
        groups.setdefault(key, []).append((rel, len(src.read_text(errors="replace").splitlines())))
    rows = []
    for key in sorted(groups):
        rows.append(f'<tr><th colspan="2">{html.escape(key)}/</th></tr>')
        for rel, n in groups[key]:
            href = rel + ".html"      # 索引页已在 source/ 内，相对自身目录
            rows.append(f'<tr><td class="u-w44"><a href="{href}"><code>{html.escape(rel)}</code></a></td>'
                        f'<td>{n} 行</td></tr>')
    body = ('<div class="content content-wide">'
            '<h2 class="section-title" id="top">minimind_reborn 源码索引</h2>'
            '<p class="qs-note">打包快照 <code>HEAD 06db33c</code>（2026-09-28）+ 工作区修改，与课程页面引用的行号一致。'
            '点击任意文件在线查看源码（带行号、语法高亮、复制按钮）。大文件（tokenizer 词表）在 <code>raw/</code> 下原样提供。</p>'
            '<div class="table-scroll"><table>' + "".join(rows) + "</table></div>"
            '<p class="qs-note"><a href="../index.html">← 返回课程首页</a></p></div>')
    return PAGE_SHELL.format(title="源码索引", prefix="../", css=css, js=js,
                             source_css=source_css, body=body, date=SNAPSHOT_DATE)

# ── 课程页重写 ────────────────────────────────────────────────────────
CODE_RE = re.compile(r'(<a [^>]*>)?\s*<code>([^<]+)</code>')
HREF_RE = re.compile(r'href="\.\./\.\./Reproduce/minimind_reborn/([^"]+)"')
# 课程页里已由 build_codeview.py 转成站内 codeview/ 的链接 → 包内统一指向 source/
HREF_CV_RE = re.compile(r'href="codeview/([^"#?]+)\.html"')

def build_link_keys(files: list[str]) -> dict[str, str]:
    """精确匹配键 → 仓库相对路径。裸 basename 仅唯一时注册。"""
    keys: dict[str, str] = {}
    for rel in files:
        keys[rel] = rel                                   # src/minimind_reborn/models/model.py
        if rel.startswith("src/minimind_reborn/"):
            keys[rel.removeprefix("src/minimind_reborn/")] = rel   # models/model.py
        keys["Reproduce/minimind_reborn/" + rel] = rel
    base: dict[str, list[str]] = {}
    for rel in files:
        base.setdefault(Path(rel).name, []).append(rel)
    for name, rels in base.items():
        if len(rels) == 1 and not name.startswith("__init__"):
            keys.setdefault(name, rels[0])                # engine.py / trainer.py 等唯一裸名
    return keys

def rewrite_course_page(text: str, keys: dict[str, str], raw_files: dict[str, str],
                        css: str, js: str) -> str:
    # 资源内联：file:// 协议下 ?v= 查询串会被当作文件名导致 404；
    # 内联后单文件自包含，任何打开方式（http/静态托管/双击 file://）都一致
    text = re.sub(r'<link rel="stylesheet" href="assets/course\.css\?v=\d+">',
                  lambda m: "<style>" + css + "</style>", text)
    text = re.sub(r'<script src="assets/course\.js\?v=\d+"></script>',
                  lambda m: "<script>" + js + "</script>", text)

    def href_sub(m: re.Match) -> str:
        rel = m.group(1)
        if rel in raw_files:
            return f'href="{raw_files[rel]}"'
        return f'href="source/{rel}.html"'
    text = HREF_RE.sub(href_sub, text)
    text = HREF_CV_RE.sub(
        lambda m: f'href="{raw_files.get(m.group(1), "source/" + m.group(1) + ".html")}"', text)

    def code_sub(m: re.Match) -> str:
        lead, content = m.group(1), m.group(2).strip()
        if lead:  # 已在链接内
            return m.group(0)
        rel = keys.get(content)
        if not rel:
            return m.group(0)
        href = raw_files.get(rel, f"source/{rel}.html")
        return f'<a href="{href}" title="在线查看源码"><code>{m.group(2)}</code></a>'
    return CODE_RE.sub(code_sub, text)

# ── 自检 ──────────────────────────────────────────────────────────────
def check_links(out_dir: Path) -> list[str]:
    errs = []
    for page in sorted(out_dir.rglob("*.html")):
        raw = page.read_text(encoding="utf-8")
        # 剥离 script/style 块：内联 JS 的模板字符串（如 href="${m.file}"）不是 HTML 链接
        body = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", "", raw)
        ids = set(re.findall(r'id="([^"]+)"', body))
        for m in re.finditer(r'href="([^"]+)"', body):
            href = m.group(1)
            if href.startswith(("http", "javascript:", "mailto:", "data:")):
                continue
            if href.startswith("#"):
                if href[1:] and href[1:] not in ids:
                    errs.append(f"{page.relative_to(out_dir)}: 缺锚点 {href}")
                continue
            target, _, anchor = href.partition("#")
            path = (page.parent / target.split("?")[0]).resolve()
            if not path.exists():
                errs.append(f"{page.relative_to(out_dir)}: 断链 {href}")
            elif anchor and path.suffix == ".html" and path.is_file():
                if f'id="{anchor}"' not in path.read_text(encoding="utf-8", errors="ignore"):
                    errs.append(f"{page.relative_to(out_dir)}: 目标缺锚点 {href}")
    return errs

# ── 主流程 ────────────────────────────────────────────────────────────
def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "source").mkdir(parents=True)

    # 1. 课程页与资源（assets 读取后内联进每页，不再外部依赖）
    pages = list(COURSE.glob("*.html"))
    css = (COURSE / "assets" / "course.css").read_text(encoding="utf-8")
    js = (COURSE / "assets" / "course.js").read_text(encoding="utf-8")
    for page in pages:
        shutil.copy(page, OUT / page.name)
    # favicon 与本地 MathJax 保持文件形态（vendor 体积大，不适合逐页内联）
    for rel in ["favicon.svg", "vendor/tex-svg.js"]:
        src = COURSE / "assets" / rel
        if src.is_file():
            (OUT / "assets" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, OUT / "assets" / rel)

    # 2. reborn 源码
    files = collect_files()
    rendered: list[tuple[str, Path]] = []
    raw_files: dict[str, str] = {}
    for src in files:
        rel = src.relative_to(REBORN).as_posix()
        if src.stat().st_size > RAW_THRESHOLD:
            dest = OUT / "raw" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest)
            raw_files[rel] = "raw/" + rel
        else:
            rendered.append((rel, src))
    for rel, src in rendered:
        page = build_source_page(rel, src, css, js, SOURCE_CSS)
        dest = OUT / "source" / (rel + ".html")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(page, encoding="utf-8")
    (OUT / "source" / "index.html").write_text(
        build_source_index(rendered, css, js, SOURCE_CSS), encoding="utf-8")

    # 3. 课程页重写（链接 + 资源内联）
    keys = build_link_keys([rel for rel, _ in rendered] + list(raw_files))
    for page in OUT.glob("*.html"):
        if page.parent != OUT:
            continue
        text = page.read_text(encoding="utf-8")
        page.write_text(rewrite_course_page(text, keys, raw_files, css, js), encoding="utf-8")

    # 4. 自检
    errs = check_links(OUT)
    n_pages = len(list(OUT.glob("*.html"))) + len(list((OUT / "source").rglob("*.html")))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"打包完成：{n_pages} 个页面（源码页 {len(rendered)}），raw 文件 {len(raw_files)}，"
          f"总大小 {size/1e6:.1f} MB")
    for e in errs:
        print("ERROR", e)
    return 1 if errs else 0

SOURCE_CSS = """
/* 源码浏览页专属：面包屑 + 行号（复用 course.css 的全部 tokens 与代码块样式） */
.src-page{max-width:1080px;margin:0 auto;padding:26px 40px 80px}
@media(max-width:640px){.src-page{padding:20px 14px 60px}}
.crumbs{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;margin:6px 0 18px;
  font-size:13px;color:var(--fg-muted)}
.crumbs .sep{color:var(--fg-faint)}
.crumbs .path{font-family:var(--mono);color:var(--fg);font-size:12.5px;overflow-wrap:anywhere;min-width:0}
.crumbs .snap{margin-left:auto;font:600 11px/1.6 var(--sans);color:var(--done);
  background:var(--done-subtle);border:1px solid var(--done-border);border-radius:10px;padding:2px 10px}
pre.code.src{font-size:12.8px;line-height:1.78;max-height:78vh;overflow:auto}
pre.code.src code{counter-reset:ln}
pre.code.src .srcline{display:block;counter-increment:ln;min-height:1.78em}
pre.code.src .srcline::before{
  content:counter(ln);display:inline-block;width:3em;margin-right:1.4em;
  text-align:right;color:var(--fg-faint);opacity:.65;user-select:none;font-size:.92em}
@media print{pre.code.src{max-height:none}.crumbs .snap{display:none}}
"""

if __name__ == "__main__":
    sys.exit(main())
