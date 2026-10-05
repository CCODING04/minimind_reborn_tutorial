#!/usr/bin/env python3
"""生成 GitHub 式代码查看页（codeview/），并重写课程页里的仓库文件链接。

行为：
  1. 扫描课程目录顶层 *.html，收集对 Reproduce/minimind_reborn 文件的引用：
     - <a href="../../Reproduce/minimind_reborn/X"> 链接
     - .code-head 标头里以 Reproduce/minimind_reborn/X 开头的路径标签
  2. 为每个被引用文件生成 codeview/X.html（镜像仓库目录结构）：
     面包屑顶栏 + 行号列 + 每行 #Ln 锚点 + 三色语法高亮（course.js 客户端完成）+ 复制按钮
  3. 重写课程页链接：
     href="../../Reproduce/minimind_reborn/X" → href="codeview/X.html"
     code-head 路径标签 → 指向 codeview 页的 <a>
  4. 清理 codeview/ 下不再被引用的孤儿页（--keep-orphans 关闭）。

约定：写课时照常写 ../../Reproduce/... 原始路径（单一事实源），本脚本负责统一转换；
改完跑 python3 tools/build_codeview.py，再跑 tools/check_course.py。
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = COURSE_DIR.parent.parent
REBORN = REPO_ROOT / "Reproduce" / "minimind_reborn"
CODEVIEW = COURSE_DIR / "codeview"

MAX_BYTES = 200 * 1024   # 超过则截断
MAX_LINES = 500

LANG_BY_EXT = {
    ".py": "python", ".yaml": "yaml", ".yml": "yaml", ".json": "json",
    ".sh": "bash", ".bash": "bash", ".toml": "text", ".md": "text", ".txt": "text",
    ".cfg": "text", ".ini": "text", ".jsonl": "text", ".lock": "text",
}

HREF_REPO = re.compile(r'href="\.\./\.\./Reproduce/minimind_reborn/([^"#?]+)"')
HREF_CV = re.compile(r'href="codeview/([^"#?]+\.html)"')
HEAD_LABEL = re.compile(
    r'(<div class="code-head">)<span>Reproduce/minimind_reborn/([^ <·"]+)([^<]*)</span>'
)
# 正文（<code> 内或 prov 标注）中提及的仓库文件：全前缀形式与裸相对路径形式
PROSE_FULL = re.compile(r'Reproduce/minimind_reborn/([\w./-]+\.[A-Za-z0-9]+)')
# 裸相对路径：正文任意位置的 dir-anchored 路径（含终端命令行内），存在性校验自动解析
PROSE_BARE = re.compile(r'(?<![\w./-])((?:src|configs|tools|tests|docs|out|assets)/[\w./-]+\.(?:py|ya?ml|json|sh|toml|md))\b')
# 正文里未被链接化的全前缀 <code> 路径 → 自动挂上 codeview 链接（(?<!>) 跳过已在 <a> 内的）
PROSE_CODE_FULL = re.compile(r'(?<!>)<code>Reproduce/minimind_reborn/([\w./-]+\.[A-Za-z0-9]+)</code>')
# 不做代码查看页的二进制/产物后缀
SKIP_SUFFIXES = {".pth", ".pt", ".bin", ".npz", ".png", ".jpg", ".jpeg", ".webp",
                 ".ico", ".woff", ".woff2", ".ttf", ".lock", ".gif", ".pdf", ".tgz"}


def page_version() -> str:
    idx = (COURSE_DIR / "index.html").read_text(encoding="utf-8")
    m = re.search(r"course\.css\?v=(\d+)", idx)
    return m.group(1) if m else "1"


def collect_refs() -> dict[str, set[str]]:
    """返回 {仓库相对路径: 引用它的页面集合}（href + 标头 + 正文提及）。"""
    refs: dict[str, set[str]] = {}

    def add(path: str, page: str) -> None:
        path = path.strip().strip("/")
        if not path:
            return
        refs.setdefault(path, set()).add(page)

    for page in sorted(COURSE_DIR.glob("*.html")):
        body = page.read_text(encoding="utf-8")
        for m in HREF_REPO.finditer(body):
            add(m.group(1), page.name)
        for m in HREF_CV.finditer(body):
            add(m.group(1)[: -len(".html")], page.name)
        for m in HEAD_LABEL.finditer(body):
            add(m.group(2), page.name)
        for m in PROSE_FULL.finditer(body):
            add(m.group(1), page.name)
        for m in PROSE_BARE.finditer(body):
            p = m.group(1)
            if p in refs or (REBORN / p).is_file():
                add(p, page.name)
            elif (REBORN / "src" / "minimind_reborn" / p).is_file():
                add("src/minimind_reborn/" + p, page.name)
    return refs


def gen_page(rel: str) -> str:
    src = REBORN / rel
    raw = src.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    truncated = False
    if len(raw) > MAX_BYTES:
        text = text[:MAX_BYTES]
        truncated = True
    lines = text.split("\n")
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES]
        truncated = True
    # 逐行 span 由客户端高亮器生成（data-lines=1）；这里只放转义后的原文
    body = html.escape("\n".join(lines))
    if body.endswith("\n"):
        body = body[:-1]

    ext = src.suffix.lower()
    lang = LANG_BY_EXT.get(ext, "text")
    n_lines = len(lines)
    kb = max(1, len(raw) // 1024)
    ver = page_version()
    title = src.name
    # codeview 页在 codeview/<多级目录>/ 下，assets 前缀按深度计算
    prefix = "../" * (len(Path(rel).parts) - 1 + 1)

    notice = (
        f'<div class="cv-notice">⚠️ 文件较大（{kb}KB），仅显示前 {n_lines} 行——'
        f'完整内容见仓库 Reproduce/minimind_reborn/{html.escape(rel)}</div>'
        if truncated else ""
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)} · minimind_reborn 代码查看 | 跟着 minimind_reborn 造一个大模型</title>
<link rel="icon" type="image/svg+xml" href="{prefix}assets/favicon.svg">
<link rel="stylesheet" href="{prefix}assets/course.css?v={ver}">
</head>
<body data-page="codeview">
<header class="cv-topbar">
  <a class="cv-back" href="{prefix}index.html">← 课程地图</a>
  <span class="cv-crumb">Reproduce/minimind_reborn/{html.escape(rel)}</span>
  <span class="cv-meta">{n_lines} 行 · {kb}KB · minimind_reborn</span>
</header>
<main class="cv-main">
  {notice}<div class="codeblock"><div class="code-head"><span>{html.escape(rel)}</span><span class="lang">{lang}</span></div><pre class="code"><code data-lines="1">{body}</code></pre></div>
</main>
<script src="{prefix}assets/course.js?v={ver}"></script>
</body>
</html>
"""


def rewrite_pages(refs: dict[str, set[str]]) -> list[str]:
    changed = []
    for page in sorted(COURSE_DIR.glob("*.html")):
        body = page.read_text(encoding="utf-8")
        orig = body
        body = HREF_REPO.sub(lambda m: f'href="codeview/{m.group(1)}.html"', body)
        body = HEAD_LABEL.sub(
            lambda m: f'{m.group(1)}<span><a href="codeview/{m.group(2)}.html">{m.group(2)}</a>{m.group(3)}</span>',
            body,
        )

        def prose_link(m: re.Match) -> str:
            rel = m.group(1)
            if rel in refs:
                return f'<a href="codeview/{rel}.html"><code>Reproduce/minimind_reborn/{rel}</code></a>'
            return m.group(0)
        body = PROSE_CODE_FULL.sub(prose_link, body)
        if body != orig:
            page.write_text(body, encoding="utf-8")
            changed.append(page.name)
    return changed


def main() -> int:
    if not REBORN.is_dir():
        print(f"ERROR: 仓库不存在：{REBORN}")
        return 1

    refs = collect_refs()
    errors = 0
    written = []
    for rel in sorted(refs):
        src = REBORN / rel
        if src.suffix.lower() in SKIP_SUFFIXES:
            continue
        if not src.is_file():
            print(f"ERROR: 引用的文件不存在：Reproduce/minimind_reborn/{rel}")
            errors += 1
            continue
        out = CODEVIEW / f"{rel}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(gen_page(rel), encoding="utf-8")
        written.append(rel)

    # 清理孤儿页
    removed = []
    keep = {f"{rel}.html" for rel in refs}
    if CODEVIEW.is_dir():
        for f in CODEVIEW.rglob("*.html"):
            relname = str(f.relative_to(CODEVIEW))
            if "--keep-orphans" in sys.argv:
                continue
            if relname not in keep:
                f.unlink()
                removed.append(relname)
        # 清掉空目录
        for d in sorted([p for p in CODEVIEW.rglob("*") if p.is_dir()], reverse=True):
            if not any(d.iterdir()):
                d.rmdir()

    changed = rewrite_pages(refs)

    print(f"codeview：生成 {len(written)} 页（{len(removed)} 个孤儿已清理）")
    for rel in written:
        src = REBORN / rel
        print(f"  {rel}  ← 引用页：{', '.join(sorted(refs[rel]))}")
    if removed:
        print(f"清理：{', '.join(removed)}")
    if changed:
        print(f"链接重写：{', '.join(changed)}")
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
