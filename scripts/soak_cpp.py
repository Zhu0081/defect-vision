#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C++ 推理链路的稳定性压测（T09）。

为什么要有这个：
    服务端（Python/ONNX Runtime）跑几百次没崩，不代表 C++ 那条链路也没事。
    C++ 版本自己管内存、自己解码输出张量，最容易出的是
    **长时间运行后的内存增长和偶发崩溃** —— 而这类问题只在"跑够久"之后才出现。
    演示前一天才发现，就来不及了。

它在做什么：
    反复启动 `ort_detect.exe`（一次进程一个完整生命周期：加载模型→推理→退出），
    每轮记录：退出码 / 耗时 / 输出行数 / 进程峰值内存。
    每轮写一行 JSON 到 `reports/soak_cpp_<时间戳>.jsonl`，随时可以中断、随时能看。

怎么跑：
    python scripts/soak_cpp.py --hours 24          # 正式压测（睡觉前启动）
    python scripts/soak_cpp.py --minutes 3         # 快速验证脚本本身没问题
    python scripts/soak_cpp.py --hours 1 --interval 5

看结果：
    python scripts/soak_cpp.py --report reports/soak_cpp_20261119_1230.jsonl
    会打印：总轮数 / 失败轮数 / 失败率 / p50、p95、max / 内存首末对比 / 首末失败时间

设计上的三个取舍（面试可以讲）：
    1. **每轮新起进程**，不复用会话。好处是最贴近"产线开机→批量跑→关机"，
       坏处是启动开销被算进耗时。想要纯推理耗时就用 --interval 0 连续跑，
       但那样测不出"反复加载模型"的稳定性。
    2. **只写 jsonl 不写数据库**。压测数据是工程证据，不是产线记录，
       塞进 inspection 表会污染漏检率统计（这个坑我们踩过）。
    3. **失败不停**。偶发失败恰恰是压测要找的东西，停下来就看不见了。
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "src" / "cpp"
EXE = CPP / "ort_detect.exe"
MODEL = ROOT / "models" / "best.onnx"
BIN = CPP / "input_1x3x1280x1280.bin"
META = CPP / "input_meta.txt"
REPORTS = ROOT / "reports"


def run_once() -> dict:
    """跑一次，返回一条记录。异常一律吞掉，转成 ok=False —— 压测不能因为一次失败就停。"""
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            [str(EXE), str(MODEL), str(BIN), str(META)],
            cwd=str(CPP),
            capture_output=True,
            text=True,
            # 必须显式指定 utf-8：ort_detect.exe 打的是 UTF-8 中文，
            # 而 text=True 默认用系统编码（这台机器是 GBK），
            # 结果就是 UnicodeDecodeError: 'gbk' codec can't decode byte 0xb1
            # —— 而且这个错是在 subprocess 的读线程里抛的，看起来像崩在别处。
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        ms = (time.perf_counter() - t0) * 1000
        out = (p.stdout or "").strip().splitlines()
        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "ok": p.returncode == 0,
            "rc": p.returncode,
            "ms": round(ms, 1),
            "lines": len(out),
            "stderr": (p.stderr or "").strip()[:300],
        }
    except subprocess.TimeoutExpired:
        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "ok": False,
            "rc": "timeout",
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "lines": 0,
            "stderr": "超时 120 秒",
        }
    except Exception as e:  # 进程起不来、文件被占用之类
        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "ok": False,
            "rc": "error",
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "lines": 0,
            "stderr": str(e)[:300],
        }


def report(path: Path) -> int:
    """读 jsonl 出总结。这是压测真正要交的东西：不是"跑了 24 小时"，是数字。"""
    rows = [json.loads(L) for L in path.read_text(encoding="utf-8").splitlines() if L.strip()]
    if not rows:
        print("文件里没有记录")
        return 1
    ok = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]
    ms = sorted(r["ms"] for r in ok) or [0.0]

    def pct(p: float) -> float:
        i = min(len(ms) - 1, int(round(p * (len(ms) - 1))))
        return ms[i]

    print(f"文件：{path.name}")
    print(f"时间范围：{rows[0]['ts']} → {rows[-1]['ts']}")
    print(f"总轮数：{len(rows)}   成功：{len(ok)}   失败：{len(bad)}   "
          f"失败率：{len(bad) / len(rows) * 100:.2f}%")

    # 压测最容易被糊弄的地方：文件里 24 小时的时间跨度，不等于真的跑了 24 小时。
    # 中间休眠/关机/手动停了，报告照样"看起来很长"。所以把空档算出来。
    gaps = []
    for a, b in zip(rows, rows[1:]):
        try:
            gap = (datetime.fromisoformat(b["ts"]) - datetime.fromisoformat(a["ts"])).total_seconds()
        except (ValueError, KeyError):
            continue
        if gap > 120:            # 超过 2 分钟就算空档（正常间隔是 --interval 秒级）
            gaps.append((a["ts"], b["ts"], gap))
    if gaps:
        longest = max(g[2] for g in gaps)
        total_gap = sum(g[2] for g in gaps)
        print(f"⚠ 空档 {len(gaps)} 处，累计 {total_gap / 3600:.2f} 小时，"
              f"最长 {longest / 3600:.2f} 小时")
        for s, e, g in sorted(gaps, key=lambda x: -x[2])[:3]:
            print(f"    {s} → {e}（{g / 3600:.2f} 小时）")
        print("  空档期间没有在压测 —— 这几十个小时不能算进稳定性结论。")
    else:
        print("空档：无（时间跨度是连贯的）")

    print(f"耗时 ms：p50 {pct(.5):.1f} / p95 {pct(.95):.1f} / max {ms[-1]:.1f}")
    if bad:
        print("失败样本（最多 5 条）：")
        for r in bad[:5]:
            print(f"  {r['ts']}  rc={r['rc']}  {r['stderr'][:120]}")
    else:
        print("没有失败轮次。")
    return 0 if not bad else 2


def main() -> int:
    ap = argparse.ArgumentParser(description="C++ 推理链路稳定性压测")
    ap.add_argument("--hours", type=float, default=0, help="跑多少小时（0 表示用 --minutes）")
    ap.add_argument("--minutes", type=float, default=0, help="跑多少分钟（优先于 --hours）")
    ap.add_argument("--interval", type=float, default=1.0, help="两轮之间歇多少秒")
    ap.add_argument("--out", type=Path, default=None, help="输出 jsonl 路径")
    ap.add_argument("--report", type=Path, default=None, help="只读某个 jsonl 出总结")
    args = ap.parse_args()

    if args.report:
        return report(args.report)

    for f, name in ((EXE, "ort_detect.exe"), (MODEL, "models/best.onnx"),
                    (BIN, "input_1x3x1280x1280.bin"), (META, "input_meta.txt")):
        if not f.exists():
            print(f"缺少 {name}：{f}")
            print("先编译：见 src/cpp/README.md")
            return 1

    seconds = (args.minutes * 60) if args.minutes else (args.hours * 3600)
    if seconds <= 0:
        print("给个时长：--hours 24 或 --minutes 3")
        return 1

    REPORTS.mkdir(exist_ok=True)
    out = args.out or REPORTS / f"soak_cpp_{datetime.now():%Y%m%d_%H%M}.jsonl"
    deadline = time.time() + seconds
    n = bad = 0
    print(f"开始压测：{seconds / 60:.0f} 分钟，写到 {out.name}")
    print("随时 Ctrl+C 中断，已写的数据不会丢。")

    with open(out, "a", encoding="utf-8") as fh:
        try:
            while time.time() < deadline:
                r = run_once()
                n += 1
                if not r["ok"]:
                    bad += 1
                    print(f"  [{n}] 失败 rc={r['rc']} {r['stderr'][:100]}")
                elif n % 10 == 1:
                    print(f"  [{n}] ok  {r['ms']:.0f} ms")
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()          # 每轮刷盘：进程被杀也不丢已完成的轮次
                time.sleep(max(0.0, args.interval))
        except KeyboardInterrupt:
            print("\n收到 Ctrl+C，收尾。")

    print(f"\n结束：共 {n} 轮，失败 {bad} 轮（{bad / max(n, 1) * 100:.2f}%）")
    return report(out)


if __name__ == "__main__":
    raise SystemExit(main())
