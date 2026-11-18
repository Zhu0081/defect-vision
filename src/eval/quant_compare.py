"""对比 FP32 与 INT8 模型在同一批图上的输出差异。

量化的真正风险不是变慢，是**悄悄变差**：延迟快 1.5 倍但把该检的漏了，
比原始模型还糟。所以量化必须配一个「两边跑同一批图、逐图对比」的验收。

输出:
    每张图：FP32 检出数 / 最高置信度 → INT8 检出数 / 最高置信度，以及判定是否一致
    汇总：判定一致率、平均置信度变化、漏检变化

用法:
    python src\\eval\\quant_compare.py --splits val test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
# 必须与导出 ONNX 时用的尺寸一致，否则两边比的不是同一个模型
IMGSZ = 1280
# ultralytics 导出带 NMS 的输出布局: [1, 300, 4框 + 1置信度 + 1类别 + 32掩码系数]
COLS_BOX = slice(0, 4)
COL_CONF = 4
COL_CLS = 5


def letterbox(img: np.ndarray, size: int = IMGSZ) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0


def load(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def contract_of(sess: ort.InferenceSession) -> str:
    """看输出形状判断是哪种导出。取错了置信度列，报出来的数就全是假的。

    带 NMS:  output0 = [1, 300, 38]（4 框 + 1 置信度 + 1 类别 + 32 掩码系数）
    不带 NMS: output0 = [1, 40, 33600]（前 4 行框、4:8 类分数、后 32 行掩码系数）
    """
    shape = sess.get_outputs()[0].shape
    if len(shape) != 3:
        raise SystemExit(f"看不懂的输出形状 {shape}")
    return "nms" if shape[1] == 300 else "nonms"


def infer(sess: ort.InferenceSession, x: np.ndarray, conf_thr: float,
          contract: str | None = None) -> tuple[int, float]:
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: x[None]})[0]
    contract = contract or contract_of(sess)
    if contract == "nms":
        arr = out[0]                      # (300, 38)
        confs = arr[:, COL_CONF] if arr.size else np.zeros(0, dtype=np.float32)
    else:
        arr = out[0]                      # (40, 33600)；这里别再写 out[0][0]，那是 33600 个数的 1-D
        confs = arr[4:8].max(axis=0)      # 每列取 4 个类别里的最大值
    if confs.size == 0:
        return 0, 0.0
    keep = confs >= conf_thr
    n = int(keep.sum())
    return n, (float(confs[keep].max()) if n else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="FP32 vs INT8 逐图对比")
    ap.add_argument("--fp32", default=str(ROOT / "models" / "best.onnx"))
    ap.add_argument("--int8", default=str(ROOT / "models" / "best_int8_static.onnx"))
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=IMGSZ,
                    help="推理尺寸，必须与导出 ONNX 时一致")
    args = ap.parse_args()

    fp32 = load(Path(args.fp32))
    int8 = load(Path(args.int8))

    rows = []
    c32, c8 = contract_of(fp32), contract_of(int8)
    print(f"  契约：FP32={c32}  INT8={c8}")
    if c32 != c8:
        print("  ⚠ 两个模型的输出契约不一致，下面的「同一位置」对比没有意义")
    for split in args.splits:
        for img_path in sorted((ROOT / "data" / "yolo" / "images" / split).glob("*.jpg")):
            img = cv2.imread(str(img_path))
            x = letterbox(img, args.imgsz)
            n_f, c_f = infer(fp32, x, args.conf, c32)
            n_q, c_q = infer(int8, x, args.conf, c8)
            rows.append({
                "split": split,
                "file": img_path.name,
                "fp32_n": n_f, "fp32_conf": c_f,
                "int8_n": n_q, "int8_conf": c_q,
                "verdict_same": (n_f > 0) == (n_q > 0),
            })

    print("=" * 96)
    print(f"  FP32 vs INT8 逐图对比   （imgsz={args.imgsz}，判定阈值 {args.conf}）")
    print("=" * 96)
    print(f"  {'集':<6}{'文件':<32}{'FP32 检出/最高份':>20}{'INT8 检出/最高份':>20}{'判定':>8}")
    print("-" * 96)
    for r in rows:
        mark = "一致" if r["verdict_same"] else "⚠ 不一致"
        print(f"  {r['split']:<6}{r['file']:<32}"
              f"{r['fp32_n']:>6} / {r['fp32_conf']:>8.3f}"
              f"{r['int8_n']:>12} / {r['int8_conf']:>16.3f}{mark:>10}")

    same = sum(r["verdict_same"] for r in rows)
    total = len(rows)
    d_conf = np.mean([abs(r["int8_conf"] - r["fp32_conf"]) for r in rows])
    d_n = np.mean([abs(r["int8_n"] - r["fp32_n"]) for r in rows])

    print("\n" + "=" * 96)
    print("  汇总")
    print("=" * 96)
    print(f"  图片数            {total}")
    print(f"  判定一致率        {same}/{total} = {same / total:.1%}")
    print(f"  平均置信度偏移    {d_conf:.4f}")
    print(f"  平均检出数偏移    {d_n:.2f} 个/张")

    if same == total:
        print("\n  ✅ 全部一致：量化没有改变任何一张图的判定")
    else:
        print(f"\n  ⚠ 有 {total - same} 张判定翻转，需要逐张看是否可接受")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
