"""置信度阈值扫描：一张图定出该把阈值调到多少。

工业视觉里，同一个模型、不同阈值，得到的是两个完全不同的产品：
阈值高了漏检多（缺陷流到客户手上），阈值低了过杀多（良品被当废品扔）。
所以"模型 mAP 多少"不是产线语言，"阈值 0.45 时漏检 0%、过杀 13%"才是。

用法：
    python src/eval/threshold_sweep.py                     # val+test 两个集合
    python src/eval/threshold_sweep.py --splits val        # 只看 val
    python src/eval/threshold_sweep.py --max-miss 5        # 漏检率上限 5%

输出：
    reports/threshold_sweep.csv      每个阈值一行
    reports/figures/threshold_curve.png
    reports/threshold_sweep.md       结论（可直接贴进 README）

核心技巧：**推理只跑一次**。
先按 conf=0.01 把所有框都取出来，记下每张图的最大置信度；
之后每个阈值只是拿这个分数比大小，不需要重新推理。
42 张图 × 19 个阈值 = 798 次推理 vs 42 次推理，差一个数量级。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "data" / "yolo" / "images"
LABELS = ROOT / "data" / "yolo" / "labels"

# 阈值网格：0.05 到 0.95，步长 0.05
GRID = [round(0.05 * i, 2) for i in range(1, 20)]

# 推理尺寸必须和训练尺寸一致。train.py 用的是多少，这里就得是多少：
# 1280 训练 + 640 推理 = 把模型的「视力」直接砍一半，指标会假性暴跌。
INFER_SIZE = 1280
_INFER_SIZE = INFER_SIZE


def collect(splits: list[str], conf_floor: float) -> list[dict]:
    """把所有图的「最大置信度」和真值一次取出来。"""
    from ultralytics import YOLO

    model = load_model(_FORCE_MODEL)
    items: list[dict] = []

    for sp in splits:
        img_dir = IMAGES / sp
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*.jpg")):
            lab = LABELS / sp / (img.stem + ".txt")
            gt = bool(lab.exists() and lab.read_text(encoding="utf-8", errors="ignore").strip())

            # conf 放到地板值，尽量把低分框也留下来，之后靠阈值筛
            res = model.predict(str(img), conf=conf_floor, iou=0.5,
                                imgsz=_INFER_SIZE, verbose=False)[0]
            confs = res.boxes.conf.cpu().numpy() if res.boxes is not None and len(res.boxes) else np.array([])
            items.append(
                {
                    "split": sp,
                    "name": img.name,
                    "gt": gt,
                    "max_conf": float(confs.max()) if confs.size else 0.0,
                    "n_boxes": int(confs.size),
                }
            )
            print(f"  {sp:5s} {img.name:38s} 真值={'NG' if gt else 'OK'}  最高分={items[-1]['max_conf']:.3f}")
    return items


_MODEL = None
_MODEL_PATH = None
_FORCE_MODEL: Path | None = None


def load_model(path: Path | None = None):
    global _MODEL, _MODEL_PATH
    path = path or (ROOT / "runs" / "segment" / "baseline_v1" / "weights" / "best.pt")
    if _MODEL is None or _MODEL_PATH != path:
        from ultralytics import YOLO

        if not path.exists():
            sys.exit(f"找不到模型 {path}\n先把 src/train/train.py 跑完")
        _MODEL = YOLO(str(path))
        _MODEL_PATH = path
    return _MODEL


def sweep(items: list[dict], t: float) -> dict:
    """给定阈值，统计四个数。"""
    tp = fp = fn = tn = 0
    for it in items:
        pred = it["max_conf"] >= t
        if it["gt"] and pred:
            tp += 1
        elif it["gt"] and not pred:
            fn += 1
        elif not it["gt"] and pred:
            fp += 1
        else:
            tn += 1
    miss = fn / (fn + tp) * 100 if (fn + tp) else float("nan")
    over = fp / (fp + tn) * 100 if (fp + tn) else float("nan")
    acc = (tp + tn) / max(1, tp + tn + fp + fn) * 100
    return {"t": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "miss": miss, "over": over, "acc": acc}


def pick(rows: list[dict], max_miss: float) -> dict:
    """在「漏检率不超过上限」的约束下，挑过杀率最低的阈值。"""
    ok = [r for r in rows if not np.isnan(r["miss"]) and r["miss"] <= max_miss]
    if ok:
        return min(ok, key=lambda r: (r["over"], r["t"]))
    # 一条都达不到，那就退而求其次：漏检优先
    ok = [r for r in rows if not np.isnan(r["miss"])]
    return min(ok, key=lambda r: (r["miss"], r["over"])) if ok else rows[0]


def plot(rows: list[dict], items: list[dict], best: dict, out: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ts = [r["t"] for r in rows]

    ax = axes[0]
    ax.plot(ts, [r["miss"] for r in rows], "o-", color="#d62728", label="漏检率 (NG 判成 OK)")
    ax.plot(ts, [r["over"] for r in rows], "s-", color="#1f77b4", label="过杀率 (OK 判成 NG)")
    ax.axvline(best["t"], ls="--", color="gray", lw=1)
    ax.annotate(
        f"推荐阈值 {best['t']:.2f}\n漏检 {best['miss']:.0f}%  过杀 {best['over']:.0f}%",
        xy=(best["t"], best["over"]),
        xytext=(best["t"] + 0.06, max(best["over"], 40) + 10),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    ax.set_xlabel("置信度阈值")
    ax.set_ylabel("百分比 (%)")
    ax.set_title("阈值越高越保守：漏检下降，过杀上升")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    ng = [it["max_conf"] for it in items if it["gt"]]
    ok = [it["max_conf"] for it in items if not it["gt"]]
    bins = np.arange(0, 1.05, 0.05)
    ax.hist(ng, bins=bins, color="#d62728", alpha=0.7, label=f"缺陷图 {len(ng)} 张")
    ax.hist(ok, bins=bins, color="#1f77b4", alpha=0.7, label=f"良品图 {len(ok)} 张")
    ax.axvline(best["t"], ls="--", color="gray", lw=1)
    ax.set_xlabel("该图检出的最高置信度")
    ax.set_ylabel("图片数")
    ax.set_title("两个分布重叠得越少，模型越好用")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\n曲线已存 {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--max-miss", type=float, default=5.0, help="漏检率上限 %%，默认 5")
    ap.add_argument("--conf-floor", type=float, default=0.01)
    ap.add_argument("--imgsz", type=int, default=INFER_SIZE,
                    help=f"推理尺寸，必须和训练一致，默认 {INFER_SIZE}")
    ap.add_argument("--tag", default="", help="输出文件名后缀，比如 val 只跑验证集时用 --tag _val")
    ap.add_argument("--model", default="",
                    help="权重路径，默认 runs/segment/baseline_v1/weights/best.pt")
    args = ap.parse_args()

    global _FORCE_MODEL, _INFER_SIZE
    if args.model:
        _FORCE_MODEL = ROOT / args.model
    _INFER_SIZE = args.imgsz
    print(f"推理尺寸 {_INFER_SIZE}（必须等于训练尺寸）")

    tag = args.tag
    out_csv = ROOT / "reports" / f"threshold_sweep{tag}.csv"
    out_md = ROOT / "reports" / f"threshold_sweep{tag}.md"
    out_png = ROOT / "reports" / "figures" / f"threshold_curve{tag}.png"

    print(f"推理一次，扫描 {len(GRID)} 个阈值\n")
    items = collect(args.splits, args.conf_floor)
    rows = [sweep(items, t) for t in GRID]

    n_ng = sum(1 for it in items if it["gt"])
    n_ok = len(items) - n_ng
    print(f"\n样本：{len(items)} 张（缺陷 {n_ng}，良品 {n_ok}），集合={','.join(args.splits)}\n")

    print(f"{'阈值':>6} {'漏检%':>7} {'过杀%':>7} {'准确率%':>8}   TP  FP  FN  TN")
    print("-" * 58)
    for r in rows:
        mark = ""
        print(
            f"{r['t']:>6.2f} {r['miss']:>7.0f} {r['over']:>7.0f} {r['acc']:>8.0f}  "
            f"{r['tp']:>3} {r['fp']:>3} {r['fn']:>3} {r['tn']:>3}{mark}"
        )

    best = pick(rows, args.max_miss)
    print("-" * 58)
    print(
        f"推荐阈值 {best['t']:.2f}：漏检 {best['miss']:.0f}%，过杀 {best['over']:.0f}%，"
        f"准确率 {best['acc']:.0f}%（约束：漏检 ≤ {args.max_miss:.0f}%）"
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"明细已存 {out_csv}")

    plot(rows, items, best, out_png)

    # 给一段可以直接贴进 README / 简历的话
    out_md.write_text(
        f"""# 阈值扫描结论

- 样本：{len(items)} 张（缺陷 {n_ng} 张、良品 {n_ok} 张），集合 {', '.join(args.splits)}
- 模型：`runs/segment/baseline_v1/weights/best.pt`
- 推荐阈值：**{best['t']:.2f}**
- 该阈值下：漏检 **{best['miss']:.0f}%**，过杀 **{best['over']:.0f}%**，准确率 {best['acc']:.0f}%

选阈值的规则：漏检率不超过 {args.max_miss:.0f}% 的前提下，取过杀率最低。
理由：漏检的代价是缺陷流到客户手上，过杀的代价是多看两眼，两者不对等。

![阈值曲线](figures/threshold_curve{tag}.png)
""",
        encoding="utf-8",
    )
    print(f"结论已存 {out_md}")


if __name__ == "__main__":
    main()
