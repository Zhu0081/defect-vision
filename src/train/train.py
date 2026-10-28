# -*- coding: utf-8 -*-
"""
train.py —— 训练 YOLO 分割模型，并在 test 集上评估

用法（第一次跑，冒烟）：
    python src\\train\\train.py --epochs 30 --imgsz 640 --batch 4

用法（跑够，正式基线）：
    python src\\train\\train.py --epochs 100 --imgsz 640 --batch 4 --name baseline_v1

只评估已有权重（不训练）：
    python src\\train\\train.py --eval-only --weights runs\\segment\\baseline_v1\\weights\\best.pt

Windows 上默认 workers=0：多进程 dataloader 在 Windows 下很容易崩（spawn 机制 + 中文路径）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = ROOT / "data" / "yolo" / "data.yaml"
RUNS = ROOT / "runs" / "segment"
REPORTS = ROOT / "reports"
N_CLASSES = 4


def fmt(x) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def evaluate(weights: Path, imgsz: int, split: str, device: str) -> dict:
    """在指定 split 上跑一遍验证，返回关键指标。"""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(data=str(DATA_YAML), imgsz=imgsz, split=split, device=device, verbose=False)
    out = {
        "split": split,
        "seg_map50": float(m.seg.map50) if m.seg else None,
        "seg_map50_95": float(m.seg.map) if m.seg else None,
        "box_map50": float(m.box.map50) if m.box else None,
        "box_map50_95": float(m.box.map) if m.box else None,
    }
    # 每类指标：maps 是按类别 id 索引的数组，ap_class_index 只包含「本 split 里真的有数据」的类
    try:
        seg_maps = getattr(m, "seg", None)
        if seg_maps is not None:
            maps = getattr(seg_maps, "maps", None)
            idx = getattr(seg_maps, "ap_class_index", None)
            if maps is not None and idx is not None:
                out["per_class"] = {m.names[int(i)]: float(maps[int(i)]) for i in idx}
    except Exception as e:
        out["per_class_error"] = str(e)
    return out


def print_metrics(tag: str, r: dict) -> None:
    print(f"\n{'='*58}")
    print(f"  {tag}   ({r['split']} 集)")
    print(f"{'='*58}")
    print(f"  seg  mAP50      {fmt(r.get('seg_map50'))}")
    print(f"  seg  mAP50-95   {fmt(r.get('seg_map50_95'))}")
    print(f"  box  mAP50      {fmt(r.get('box_map50'))}")
    print(f"  box  mAP50-95   {fmt(r.get('box_map50_95'))}")
    if r.get("per_class"):
        print(f"{'-'*58}")
        for k, v in r["per_class"].items():
            print(f"  {k:<15} {fmt(v)}")
    print(f"{'='*58}")


LOCAL_PRETRAINED = ROOT / "models" / "yolo11n-seg.pt"


def default_model() -> str:
    """优先用本地已下好的权重；没有才让 ultralytics 去网上拉（国内经常拉不动）。"""
    return str(LOCAL_PRETRAINED) if LOCAL_PRETRAINED.exists() else "yolo11n-seg.pt"


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO 分割训练 + 评估")
    ap.add_argument("--model", default=default_model(), help="预训练权重或 yaml")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0, help="Windows 建议 0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default=None)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--weights", type=Path, default=None)
    args = ap.parse_args()

    if not DATA_YAML.exists():
        sys.exit(f"[错误] 找不到 {DATA_YAML}\n       先跑 python src\\data\\prepare.py")
    REPORTS.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        if not args.weights or not args.weights.exists():
            sys.exit(f"[错误] --weights 指向的文件不存在：{args.weights}")
        r = evaluate(args.weights, args.imgsz, "test", args.device)
        print_metrics("评估", r)
        return 0

    name = args.name or f"run_{datetime.now():%m%d_%H%M}"
    print(f"[开始] {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"       权重 {args.model}  epochs {args.epochs}  imgsz {args.imgsz}  batch {args.batch}")
    print(f"       数据 {DATA_YAML}")
    print(f"       输出 {RUNS / name}")
    print(f"       ⏳ CPU 训练会比较慢，前 1 分钟是在加载数据，别急着按 Ctrl+C\n")

    from ultralytics import YOLO

    t0 = time.time()
    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        patience=args.patience,
        plots=True,
        val=True,
        seed=0,
        deterministic=True,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n[训练完成] 用时 {elapsed/60:.1f} 分钟")

    best = RUNS / name / "weights" / "best.pt"
    if not best.exists():
        sys.exit(f"[错误] 没找到 {best}")

    summary = {"name": name, "elapsed_min": round(elapsed / 60, 2),
               "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
               "n_train_images": None, "timestamp": datetime.now().isoformat(timespec="seconds")}

    for split in ("val", "test"):
        r = evaluate(best, args.imgsz, split, args.device)
        print_metrics(f"{name} / {split}", r)
        summary[f"{split}_seg_map50"] = r.get("seg_map50")
        summary[f"{split}_seg_map50_95"] = r.get("seg_map50_95")
        summary[f"{split}_per_class"] = r.get("per_class")

    out = REPORTS / f"train_{name}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n指标已存：{out}")

    print("\n⚠️ 提醒：42 张图训出来的指标**不可信**，它的作用是证明链路打通。")
    print("   真正的数字要等数据补到 200 张以上、并且按工件分组重训。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
