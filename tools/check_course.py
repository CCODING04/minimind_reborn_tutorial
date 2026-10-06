#!/usr/bin/env python3
"""课程结构 QA 检查（DESIGN.md §5.3 契约）。

检查项：
  1. 断链：<a href> 本地目标存在；页内锚点 (#id) 有对应元素 id
  2. nav 一致：每页 <body data-module> 与 course.js COURSE 数组一致；
     status=done 的模块文件必须存在；status=planned 的模块文件不得存在
  3. 正文路径引用：文本/代码中出现 Reproduce/minimind_reborn/... 形态的路径，
     核对仓库内存在（豁免：通配符 *、学习者产出、明显占位）
  4. 结构必备：模块页含 .unit-head/.accept/details.quiz/.exercise/.glossary/.pager；
     全部页面无 style= 内联属性；title/charset/lang/css/js 齐备
  5. quiz 回看链接：details.quiz 内 <a href="#..."> 锚点存在

用法：python3 tools/check_course.py [--reborn <minimind-reborn 仓库路径>]
REBORN 定位优先级：--reborn 参数 > REBORN_REPO 环境变量 > 旧 monorepo 布局
（COURSE_DIR.parent.parent/Reproduce/minimind_reborn）。
退出码：0 全绿；1 存在错误；2 REBORN 路径无效。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parent.parent

# codeview 下这些目录是「存档快照」：源为运行时产物（out/、runs/ 不随仓库分发），
# 已有页保留展示但无法与源核对——QA 对其降级为警告而非错误。
ARCHIVE_TOP_DIRS = ("out", "runs")


def resolve_reborn() -> Path:
    raw = ""
    if "--reborn" in sys.argv:
        i = sys.argv.index("--reborn")
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
            raw = sys.argv[i + 1]
    raw = raw or os.environ.get("REBORN_REPO", "")
    if raw:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            return p
        print(f"ERROR: --reborn / REBORN_REPO 指定的仓库不存在：{p}")
        sys.exit(2)
    return COURSE_DIR.parent.parent / "Reproduce" / "minimind_reborn"


REBORN = resolve_reborn()
CODEVIEW = COURSE_DIR / "codeview"

errors: list[str] = []
warns: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warns.append(msg)


# ── 解析 course.js 的 COURSE 数组（单一事实源） ─────────────────────────
def parse_course_js(js: str) -> list[dict[str, str]]:
    m = re.search(r"const COURSE = \[(.*?)\];", js, re.S)
    if not m:
        err("course.js: 找不到 COURSE 数组")
        return []
    mods = []
    for entry in re.finditer(r"\{([^{}]+)\}", m.group(1)):
        d = dict(re.findall(r'(\w+)\s*:\s*"([^"]*)"', entry.group(1)))
        if "id" in d and "file" in d:
            mods.append(d)
    return mods


def check_html(page: Path, body: str, module_ids: set[str]) -> None:
    rel = page.relative_to(COURSE_DIR)

    # ── 基础头检查 ──
    if '<meta charset="UTF-8">' not in body:
        err(f"{rel}: 缺少 charset 声明")
    if 'lang="zh-CN"' not in body:
        err(f"{rel}: 缺少 lang=zh-CN")
    if 'assets/course.css' not in body:
        err(f"{rel}: 未引用 assets/course.css")
    if 'assets/course.js' not in body:
        err(f"{rel}: 未引用 assets/course.js")
    # 资源版本化：浏览器启发式缓存会钉死旧 CSS/JS（http.server 不发 Cache-Control），
    # 引用必须带 ?v=N；改动 CSS/JS 后全量递增版本号
    if "assets/course.css?v=" not in body or "assets/course.js?v=" not in body:
        err(f"{rel}: 资源引用缺少版本参数（应为 assets/course.css?v=N / course.js?v=N）")
    if "<title>" not in body:
        err(f"{rel}: 缺少 <title>")

    # ── 内联样式禁令 ──
    for m in re.finditer(r'style="[^"]*"', body):
        err(f"{rel}: 发现内联样式 {m.group(0)[:60]}")

    # ── pre 内非法块级元素（code-head 必须包在 .codeblock 里、置于 pre 之外） ──
    for m in re.finditer(r"<pre[^>]*>\s*<div", body):
        err(f"{rel}: <pre> 内嵌套了 <div>——标头应使用 <div class=\"codeblock\"><div class=\"code-head\">…</div><pre>…</pre></div> 结构")

    # ── 断链：本地 href ──
    ids = set(re.findall(r'id="([^"]+)"', body))
    for m in re.finditer(r'href="([^"]+)"', body):
        href = m.group(1)
        if href.startswith(("http://", "https://", "javascript:", "mailto:")):
            continue
        if href.startswith("#"):
            if href[1:] and href[1:] not in ids:
                err(f"{rel}: 页内锚点不存在 {href}")
            continue
        target = href.split("#")[0].split("?")[0]
        if not target:
            continue
        anchor = href.split("#")[1] if "#" in href else None
        path = (page.parent / target).resolve()
        if not path.exists():
            err(f"{rel}: 断链 {href}")
            continue
        if anchor and path.suffix == ".html":
            txt = path.read_text(encoding="utf-8", errors="ignore")
            if f'id="{anchor}"' not in txt:
                err(f"{rel}: 跨页锚点不存在 {href}")

    # ── 正文路径引用（豁免通配符与学习者产出） ──
    for m in re.finditer(r"Reproduce/minimind_reborn/([\w./*-]+)", body):
        p = m.group(1).rstrip(".")
        if "*" in p or "<" in p or p.endswith("/"):
            continue
        if not (REBORN / p).exists():
            # 允许省略 src/minimind_reborn 前缀的简写已在文档声明，仅告警
            if (REBORN / "src" / "minimind_reborn" / p).exists():
                continue
            err(f"{rel}: 正文引用的仓库路径不存在 Reproduce/minimind_reborn/{p}")

    # ── codeview 链接规则：仓库 href 必须已改写为 codeview/，且目标存在 ──
    if re.search(r'href="\.\./\.\./Reproduce', body):
        err(f"{rel}: 存在未改写的仓库直链（应运行 tools/build_codeview.py 转为 codeview/ 链接）")
    for m in re.finditer(r'href="(codeview/[^"#?]+\.html)"', body):
        if not (COURSE_DIR / m.group(1)).exists():
            err(f"{rel}: codeview 链接目标不存在 {m.group(1)}")

    # ── 模块页结构必备 ──
    dm = re.search(r'<body data-module="(m\d+)"', body)
    is_module = bool(dm)
    if is_module:
        mid = dm.group(1)
        if mid not in module_ids:
            err(f"{rel}: data-module={mid} 不在 COURSE 数组中")
        for cls, name in [
            ("unit-head", "模块头"), ("accept", "验收块"), ("glossary", "词汇表"),
        ]:
            if f'class="{cls}' not in body and f'"{cls}"' not in body:
                err(f"{rel}: 模块页缺少 .{cls}（{name}）")
        n_quiz = body.count('class="quiz"') or body.count("details class=\"quiz\"")
        if n_quiz < 3:
            err(f"{rel}: 折叠测验不足 3 个（当前 {n_quiz}）")
        if 'class="exercise"' not in body:
            err(f"{rel}: 缺少 .exercise 变式练习")
        if 'id="pager"' not in body:
            err(f"{rel}: 缺少 #pager 页间导航挂载点")
        if 'class="ex-predict"' not in body:
            err(f"{rel}: 变式练习缺少 .ex-predict 预测留白")
        if not re.search(r"\[作者实测|\[存档", body):
            err(f"{rel}: 页面没有任何 .prov 溯源标注（作者实测/存档）")
        title = re.search(r"<title>(.*?)</title>", body)
        if title and not title.group(1).startswith(mid.upper()):
            err(f"{rel}: title 应以 {mid.upper()} 开头，实际 {title.group(1)[:30]}")
        # quiz 答错回看链接
        for qm in re.finditer(r"<details class=\"quiz\">(.*?)</details>", body, re.S):
            if '<a href="#' not in qm.group(1):
                warn(f"{rel}: 某测验缺少『答错回看』锚点链接")


def main() -> int:
    js = (COURSE_DIR / "assets" / "course.js").read_text(encoding="utf-8")
    course = parse_course_js(js)
    if not course:
        return 1
    done_ids = {m["id"] for m in course if m.get("status") == "done"}
    planned_ids = {m["id"] for m in course if m.get("status") == "planned"}
    module_ids = {m["id"] for m in course}
    files = {m["file"]: m for m in course}

    # ── COURSE 与文件系统一致 ──
    for m in course:
        f = COURSE_DIR / m["file"]
        if m.get("status") == "done" and not f.exists():
            err(f"course.js: {m['id']} status=done 但文件不存在 {m['file']}")
        if m.get("status") == "planned" and f.exists():
            warn(f"course.js: {m['id']} status=planned 但文件已存在——请改为 done")

    # ── 页面检查 ──
    pages = sorted(COURSE_DIR.glob("*.html"))
    for page in pages:
        body = page.read_text(encoding="utf-8")
        if page.name in files:
            dm = re.search(r'<body data-module="(m\d+)"', body)
            expect = files[page.name]["id"]
            if not dm:
                err(f"{page.name}: 缺少 data-module 声明（应为 {expect}）")
            elif dm.group(1) != expect:
                err(f"{page.name}: data-module={dm.group(1)} 与 COURSE 期望 {expect} 不一致")
        check_html(page, body, module_ids)

    # ── index 与侧栏（JS 生成）基本盘 ──
    if not (COURSE_DIR / "index.html").exists():
        err("缺少 index.html")
    if done_ids & planned_ids:
        err("course.js: 同一模块同时 done 与 planned")

    # ── codeview 生成页检查：结构 + 版本 + 完整断链（返回链接/行锚/assets） ──
    idx_css = re.search(r"course\.css\?v=(\d+)", (COURSE_DIR / "index.html").read_text(encoding="utf-8"))
    idx_ver = idx_css.group(1) if idx_css else None
    cv_pages = sorted(CODEVIEW.glob("**/*.html")) if CODEVIEW.is_dir() else []
    for page in cv_pages:
        rel = page.relative_to(COURSE_DIR)
        body = page.read_text(encoding="utf-8", errors="ignore")
        if 'class="cv-topbar"' not in body:
            err(f"{rel}: codeview 页缺少 cv-topbar 面包屑")
        if 'data-lines="1"' not in body:
            err(f"{rel}: codeview 页缺少 data-lines 逐行渲染标记")
        if "../assets/course.js" not in body or "../assets/course.css" not in body:
            err(f"{rel}: codeview 页未引用共享 assets")
        v = re.search(r"course\.css\?v=(\d+)", body)
        if idx_ver and v and v.group(1) != idx_ver:
            err(f"{rel}: 资源版本 {v.group(1)} 与首页 {idx_ver} 不一致（重跑 build_codeview.py）")
        src_rel = page.relative_to(CODEVIEW).as_posix()[: -len(".html")]
        if not (REBORN / src_rel).exists():
            if src_rel.split("/")[0] in ARCHIVE_TOP_DIRS:
                warn(f"{rel}: 存档快照（源为 {src_rel.split('/')[0]}/ 运行时产物，不随仓库分发，无法与源核对）")
            else:
                err(f"{rel}: 对应源文件已不存在")
        # 断链：本地 href 必须存在（返回链接、assets、无外链）
        ids = set(re.findall(r'id="([^"]+)"', body))
        for m in re.finditer(r'(?:href|src)="([^"]+)"', re.sub(r"<script[\s\S]*?</script>", "", body)):
            href = m.group(1)
            if href.startswith(("http://", "https://", "data:", "#")):
                continue
            target = href.split("#")[0].split("?")[0]
            if not target:
                continue
            if not (page.parent / target).resolve().exists():
                err(f"{rel}: 断链 {href}")

    # ── 汇总 ──
    for w in warns:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print(f"\n检查完成：{len(pages)} 个页面，{len(course)} 个模块登记，"
          f"{len(errors)} 错误，{len(warns)} 警告")
    print(f"REBORN = {REBORN}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
