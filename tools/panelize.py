#!/usr/bin/env python3
"""把课程页 .content 内的小节包进 <section class="panel">（blueprint 面板化，DESIGN §4.6）。

规则：
  1. <div class="content"> 内：首个 <h2 之前的内容（unit-head 等）包为 panel-head；
     每个 <h2（含 id）到下一个边界（<h2 / .glossary / #pager）包为一个 panel；
     .glossary 整体包为 panel（其内部 h2 不作为边界）。
  2. 幂等：content 内已存在 class="panel" 则跳过该页。
  3. 不改任何文字内容；h2 的 id 原样保留（TOC / 答错回看锚点不受影响）。

用法：python3 tools/panelize.py [页面...]   # 缺省处理全部 m*.html + btrack.html + pytorch_guide.html
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GLOB = ["m*.html", "btrack.html", "pytorch_guide.html"]

CONTENT_OPEN = re.compile(r'<div class="content(?:\s[^"]*)?">')
BOUNDARY = re.compile(r'(<h2(?![^>]*class="[^"]*glossary)|<div class="glossary"|<nav class="pager"|<footer)', re.S)


def find_content_span(body: str) -> tuple[int, int] | None:
    m = CONTENT_OPEN.search(body)
    if not m:
        return None
    start = m.end()
    # 找与 content div 配对的闭合 </div>：从 start 起做深度计数
    depth = 1
    i = start
    for tok in re.finditer(r"<div\b|</div>", body[start:]):
        if tok.group(0) == "<div":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                end = start + tok.start()
                return start, end
    return None


def panelize_page(page: Path) -> str:
    body = page.read_text(encoding="utf-8")
    span = find_content_span(body)
    if not span:
        return "skip（无 .content）"
    c0, c1 = span
    content = body[c0:c1]
    if 'class="panel' in content:
        return "skip（已 panel 化）"

    # 边界位置：h2（非 glossary 内部——glossary 整体作为边界，其后不再有 h2 需要处理）
    marks = [m.start() for m in BOUNDARY.finditer(content)]
    if not marks:
        return "skip（无小节边界）"

    pieces: list[str] = []
    # 头部（首个边界前）→ panel-head
    pieces.append(f'<section class="panel panel-head">\n{content[:marks[0]].strip()}\n</section>')
    # 各小节
    for k, s in enumerate(marks):
        e = marks[k + 1] if k + 1 < len(marks) else len(content)
        seg = content[s:e].strip()
        if not seg:
            continue
        cls = "panel glossary-panel" if '<div class="glossary"' in seg[:60] else "panel"
        pieces.append(f'<section class="{cls}">\n{seg}\n</section>')
    new_content = "\n\n".join(pieces) + "\n      "
    body = body[:c0] + new_content + body[c1:]
    page.write_text(body, encoding="utf-8")
    return f"ok（{len(marks)} 个 panel）"


def main() -> int:
    targets: list[Path] = []
    if len(sys.argv) > 1:
        targets = [Path(a) for a in sys.argv[1:]]
    else:
        for g in DEFAULT_GLOB:
            targets.extend(sorted(COURSE_DIR.glob(g)))
    for page in targets:
        if not page.is_absolute():
            page = COURSE_DIR / page
        print(f"{page.name}: {panelize_page(page)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
