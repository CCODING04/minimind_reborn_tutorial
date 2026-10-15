#!/usr/bin/env python3
"""复跑 experiments/ 实验脚本，并与 expected/ 存档逐行比对（结果审查）。

用法：
  python3 tools/verify_experiments.py                 # 复跑全部可自动验证的脚本
  python3 tools/verify_experiments.py --only m08_exp_route
  python3 tools/verify_experiments.py --update        # 重录 expected/ 存档

约定：
  - experiments/ 下每个脚本默认都应可自动复跑；不能自动跑的（重数据/GPU/多卡依赖）
    加入下方 MANUAL 集合并注明原因，页面输出以"作者实测"存档为准；
  - 复跑环境：minimind_reborn 仓库的 uv 环境（定位优先级同 build_codeview.py：
    --reborn 参数 > REBORN_REPO 环境变量 > 默认 ../.. 兄弟布局）；
  - 比对口径：stdout+stderr 合并、按行 rstrip 容忍行尾空白，其余逐字一致。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = COURSE_DIR / "experiments"
EXPECTED_DIR = EXP_DIR / "expected"
DEFAULT_TIMEOUT = 300  # 秒；单脚本超时按失败计

# 不自动复跑的脚本：重数据/GPU/多卡依赖，或输出本质不稳定（时延/采样/内存读数逐次漂移）。
# 这些脚本的页面终端输出以"作者实测"存档为准；expected/ 不存档。
MANUAL: dict[str, str] = {
    "tour": "采样解码文本逐次不同（温度采样）",
    "m06_bench_kv": "时延基准，数字随机器态漂移",
    "m07_fill": "填空模板（三处 ____ 待读者填写后才可运行）",
    "m09_meta_repro": "随机初始化 logits 偏差逐次不同",
    "m11_bench": "需 1.24GB 数据集 + 每姿势独立进程 /usr/bin/time 三连跑",
    "m11_ex": "现版仓库 _fingerprint 接口已演进（作者实测存档）",
    "m11_pit": "峰值内存读数逐次漂移（739/740MB）",
    "m11_registry": "峰值内存读数逐次漂移（739/740MB）",
    "m11_accept": "输出含时延/内存读数，逐次漂移",
    "m13_bench_dtype": "时延基准，数字随机器态漂移",
    "m14_kill9": "进程管理实验（kill -9），人工运行",
    "m15_nosync_off": "torchrun 多卡训练，人工运行",
    "m17_margin": "数据依赖 + 统计数字漂移",
    "m21_verify_qwen": "Qwen3 采样输出逐次不同",
    "m22_temp_ab3": "温度 A/B 全量评测（采样+吞吐，7 分钟级）",
}

# 复跑较慢的稳定脚本：放宽单脚本超时（秒）
TIMEOUT_OVERRIDES: dict[str, int] = {
    "m17_mem": 480, "m18_exp_lora": 480, "m19_rollout": 480,
}


def resolve_reborn(argv: list[str]) -> Path:
    raw = ""
    if "--reborn" in argv:
        i = argv.index("--reborn")
        if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            raw = argv[i + 1]
    raw = raw or os.environ.get("REBORN_REPO", "")
    p = Path(raw).expanduser().resolve() if raw else COURSE_DIR.parent.parent / "Reproduce" / "minimind_reborn"
    if not p.is_dir():
        print(f"ERROR: minimind_reborn 仓库不存在：{p}（--reborn / REBORN_REPO 可指定）")
        sys.exit(2)
    return p


def run_one(reborn: Path, script: Path) -> tuple[int, str]:
    timeout = TIMEOUT_OVERRIDES.get(script.stem, DEFAULT_TIMEOUT)
    proc = subprocess.run(
        ["uv", "run", "python", str(script)],
        cwd=reborn, capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def norm(text: str) -> list[str]:
    return [ln.rstrip() for ln in text.splitlines()]


def main() -> int:
    argv = sys.argv[1:]
    reborn = resolve_reborn(argv)
    update = "--update" in argv
    only = argv[argv.index("--only") + 1] if "--only" in argv and argv.index("--only") + 1 < len(argv) else ""

    scripts = sorted(EXP_DIR.glob("*.py"))
    if only:
        scripts = [s for s in scripts if s.stem == only]
        if not scripts:
            print(f"ERROR: experiments/ 下没有 {only}.py")
            return 2

    passed, failed, updated, skipped = [], [], [], []
    for script in scripts:
        name = script.stem
        if name in MANUAL and not only:
            skipped.append(name)
            continue
        expected_file = EXPECTED_DIR / f"{name}.txt"
        try:
            code, out = run_one(reborn, script)
        except subprocess.TimeoutExpired:
            failed.append(name)
            print(f"FAIL {name}: 复跑超时（>{TIMEOUT_OVERRIDES.get(script.stem, DEFAULT_TIMEOUT)}s）")
            continue
        if code != 0:
            failed.append(name)
            print(f"FAIL {name}: 退出码 {code}\n{out[-1500:]}")
            continue
        if update or not expected_file.is_file():
            expected_file.write_text(out, encoding="utf-8")
            updated.append(name)
            print(f"RECD {name}: expected/ 存档已写入（{len(out.splitlines())} 行）")
            continue
        expected = norm(expected_file.read_text(encoding="utf-8"))
        actual = norm(out)
        if expected == actual:
            passed.append(name)
            print(f"PASS {name}（{len(actual)} 行一致）")
        else:
            failed.append(name)
            print(f"FAIL {name}: 与 expected/ 存档不一致")
            for i, (e, a) in enumerate(zip(expected + [""] * 8, actual + [""] * 8), 1):
                if e != a:
                    print(f"  第 {i} 行\n    存档: {e!r}\n    复跑: {a!r}")
                    break
            if len(expected) != len(actual):
                print(f"  行数：存档 {len(expected)} vs 复跑 {len(actual)}")

    print(f"\n合计：{len(passed)} 通过 / {len(failed)} 失败 / {len(updated)} 重录 / {len(skipped)} manual 跳过")
    if skipped:
        print(f"manual（不自动复跑）：{', '.join(skipped)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
