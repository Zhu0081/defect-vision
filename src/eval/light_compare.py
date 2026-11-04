#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""光源配对对照：同一批工件、同一个模型，只换光源，看指标差多少。

为什么必须配对：
    如果重新训练，权重、数据划分全都变了，Δ 就没法归因到光源。
    这里「其他全不变，只换灯的位置」，差异才只能来自光。

用法：
    python src/eval/light_compare.py
    python src/eval/light_compare.py --conf 0.25 --from-part 25

前提：
    数据用 ingest_phone.py 入库，同一张纸是一个 part_id，每个光源 3 张。
    metadata.csv 里 light 列区分 side / top。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25, help="判定阈值")
    ap.add_argument("--from-part", type=int, default=25, help="只统计件号 >= 这个值")
    ap.add_argument("--model", default=str(ROOT / "models" / "best.onnx"))
    args = ap.parse_args()

    import csv

    import cv2

    from src.service.detector import build_detector

    meta_path = ROOT / "data" / "captured" / "metadata.csv"
    if not meta_path.exists():
        print(f"没有 {meta_path}，先入库")
        return 1

    rows = []
    with meta_path.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                pid = int(r.get("part_id") or -1)
            except ValueError:
                continue
            if pid < args.from_part:
                continue
            rows.append(r)

    if not rows:
        print(f"件号 >= {args.from_part} 的图片一张都没有")
        return 1

    det = build_detector(Path(args.model), conf=args.conf)
    if getattr(det, "is_mock", False):
        print("⚠ 当前是 MockDetector，结果是假的，先检查 models/best.onnx")
        return 1

    # stats[light][label] = [总数, 判NG数]
    stats: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    detail: list[tuple[str, str, str, str, int, float]] = []

    for r in rows:
        light = (r.get("light") or "?").strip()
        label = (r.get("label") or "").strip()
        if label not in ("good", "defect"):
            continue
        rel = (r.get("relpath") or "").strip().replace("\\", "/")
        img_path = ROOT / "data" / "captured" / rel
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        dets = det.predict(img, args.conf)
        n = len(dets)
        max_s = max((d.score for d in dets), default=0.0)
        stats[light][label][0] += 1
        if n > 0:
            stats[light][label][1] += 1
        detail.append((light, label, r.get("filename", "?"), img_path.name, n, max_s))

    print()
    print("=" * 78)
    print(f"  光源配对对照   conf={args.conf}   件号 >= {args.from_part}   模型={Path(args.model).name}")
    print("=" * 78)
    header = f"{'光源':<8}{'良品张数':>10}{'过杀张数':>10}{'缺陷张数':>10}{'检出张数':>10}{'检出率':>10}{'过杀率':>10}"
    print(header)
    print("-" * 78)

    table: dict[str, dict[str, float]] = {}
    for light in sorted(stats):
        good_n, good_ng = stats[light]["good"]
        bad_n, bad_ng = stats[light]["defect"]
        rec = 100.0 * bad_ng / bad_n if bad_n else float("nan")
        over = 100.0 * good_ng / good_n if good_n else float("nan")
        print(f"{light:<8}{good_n:>10}{good_ng:>10}{bad_n:>10}{bad_ng:>10}"
              f"{rec:>9.1f}%{over:>9.1f}%")
        table[light] = {"recall": rec, "overkill": over}

    print("-" * 78)
    if {"side", "top"} <= set(table):
        d_rec = table["side"]["recall"] - table["top"]["recall"]
        d_over = table["side"]["overkill"] - table["top"]["overkill"]
        print(f"Δ(侧光 - 顶光)   检出率 {d_rec:+.1f} 点    过杀率 {d_over:+.1f} 点")
        if abs(d_rec) < 5:
            print("→ 结论：两种光源差异 < 5 点，在此笔画宽度下光源不是瓶颈。")
        elif d_rec > 0:
            print("→ 结论：侧光检出率明显更高，划痕类缺陷需要低角度掠射光。")
        else:
            print("→ 结论：顶光反而更好，回去看侧光角度是否拍成了反面光。")
    else:
        print("两种光源的数据不齐，无法配对比较。")

    # 逐张明细，方便定位判反的是哪几张
    print()
    print("逐张明细（只打印判错的）")
    print(f"  {'光源':<6}{'真值':<8}{'结果':<8}{'缺陷数':>6}{'最高分':>8}  文件")
    wrong = 0
    for light, label, _fname, name, n, ms in detail:
        pred = "NG" if n > 0 else "OK"
        truth = "NG" if label == "defect" else "OK"
        if pred == truth:
            continue
        wrong += 1
        print(f"  {light:<6}{truth:<8}{pred:<8}{n:>6}{ms:>8.3f}  {name}")
    if wrong == 0:
        print("  （全部判对）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
