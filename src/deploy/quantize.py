"""INT8 量化：FP32 vs 动态量化 vs 静态量化，比延迟也比精度。

用法:
    python src\\deploy\\quantize.py --mode dynamic
    python src\\deploy\\quantize.py --mode static --calib 60
    python src\\deploy\\quantize.py --mode bench-only

为什么量化的动态版不校准、静态版要校准：
    动态量化 —— 权重预先转 INT8，激活值在推理时**临时**量化（带 min/max 统计）。
                不用校准数据，对 MatMul/Gemm 多的模型友好（如 Transformer）。
    静态量化 —— 权重和激活**都**预先转 INT8，需要一批真实图跑一遍统计激活范围，
                否则范围猜错，精度崩。CNN（Conv 多）用这个才真正快。

⚠ 必须排除的节点:
    NonMaxSuppression / TopK / Resize / GatherND / ScatterND —— 这些算子的 INT8 实现
    要么不存在、要么精度不可接受。不排除会报错或结果错乱。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
OUT_DIR = MODELS
FRUIT_COLORS = ("green", "red")  # 不参与逻辑，只为可读性


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #
def load_sess(path: Path, threads: int = 0) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def bench(path: Path, imgsz: int = 640, n: int = 30, warmup: int = 5) -> dict:
    sess = load_sess(path)
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {name: x})

    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1000)

    ts_arr = np.array(ts)
    size_mb = path.stat().st_size / 1024 / 1024
    return {
        "file": path.name,
        "size_mb": round(size_mb, 2),
        "p50_ms": round(float(np.percentile(ts_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(ts_arr, 95)), 2),
        "mean_ms": round(float(ts_arr.mean()), 2),
        "runs": n,
    }


# --------------------------------------------------------------------------- #
# 量化
# --------------------------------------------------------------------------- #
def exclude_nodes(path: Path) -> list[str]:
    """找出不该参与量化的节点名。"""
    import onnx

    m = onnx.load(str(path))
    bad_ops = {"NonMaxSuppression", "TopK", "Resize", "GatherND", "ScatterND", "Shape", "Constant"}
    names = [n.name for n in m.graph.node if n.op_type in bad_ops and n.name]
    return names


def ensure_opset13(src: Path) -> Path:
    """静态量化会生成带 `axis` 属性的 DequantizeLinear，而该属性从 opset 13 才有。

    本项目的 ONNX 是 opset 12（当初为兼容 RKNN 选的），直接量化会得到
    `Unrecognized attribute: axis for operator DequantizeLinear` 的非法模型。
    这里先升到 13 再量化。
    """
    import onnx
    from onnx import version_converter

    m = onnx.load(str(src))
    cur = max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))
    if cur >= 13:
        return src

    dst = src.with_name(src.stem + "_op13.onnx")
    if not dst.exists():
        conv = version_converter.convert_version(m, 13)
        onnx.save(conv, str(dst))
    print(f"  opset {cur} → 13：{dst.name}")
    return dst


def quantize_dynamic_mode(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=exclude_nodes(src),
        per_channel=True,
    )


def build_calib_reader(src: Path, imgsz: int, limit: int):
    """从训练集取真实图片做校准（LetterBox 到 640，BGR->RGB，归一化）。"""
    import cv2
    from onnxruntime.quantization import CalibrationDataReader

    img_dir = ROOT / "data" / "yolo" / "images" / "train"
    files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    files = files[:limit]
    if not files:
        raise SystemExit(f"校准集为空: {img_dir}")

    print(f"  校准图 {len(files)} 张（来自 train，绝不能用 val/test，否则泄漏）")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = []
            for f in files:
                im = cv2.imread(str(f))
                h, w = im.shape[:2]
                r = min(imgsz / h, imgsz / w)
                nh, nw = int(round(h * r)), int(round(w * r))
                im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
                canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
                top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
                canvas[top:top + nh, left:left + nw] = im
                x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                self.items.append({name: x[None]})
            self.i = 0

        def get_next(self):
            if self.i >= len(self.items):
                return None
            item = self.items[self.i]
            self.i += 1
            return item

        def rewind(self):
            self.i = 0

    global name
    sess = load_sess(src)
    name = sess.get_inputs()[0].name
    return Reader()


def quantize_static_mode(src: Path, dst: Path, imgsz: int, calib: int,
                         op_types: list[str] | None = None) -> None:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    reader = build_calib_reader(src, imgsz, calib)
    quantize_static(
        model_input=str(ensure_opset13(src)),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=exclude_nodes(src),
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=op_types,
    )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="INT8 量化与延迟对比")
    ap.add_argument("--model", default=str(MODELS / "best.onnx"))
    ap.add_argument("--mode", choices=["dynamic", "static", "bench-only", "all"], default="dynamic")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calib", type=int, default=60, help="静态量化的校准图数量")
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()

    src = Path(args.model)
    if not src.exists():
        print(f"❌ 找不到模型: {src}")
        return 1

    results = []

    print("=" * 58)
    print("  FP32 基线")
    print("=" * 58)
    fp32 = bench(src, args.imgsz, args.runs)
    results.append({**fp32, "kind": "FP32"})
    print(f"  {fp32['file']}  {fp32['size_mb']} MB  p50 {fp32['p50_ms']} ms")

    if args.mode in ("dynamic", "all"):
        print("\n" + "=" * 58)
        print("  动态量化 (Weight-only INT8)")
        print("=" * 58)
        dst = OUT_DIR / "best_int8_dynamic.onnx"
        t0 = time.perf_counter()
        quantize_dynamic_mode(src, dst)
        print(f"  生成 {dst.name}（{time.perf_counter() - t0:.1f} 秒）")
        r = bench(dst, args.imgsz, args.runs)
        results.append({**r, "kind": "INT8 动态"})
        print(f"  {r['size_mb']} MB  p50 {r['p50_ms']} ms")

    if args.mode in ("static", "all"):
        print("\n" + "=" * 58)
        print("  静态量化 (QDQ, 需校准)")
        print("=" * 58)
        dst = OUT_DIR / "best_int8_static.onnx"
        t0 = time.perf_counter()
        quantize_static_mode(src, dst, args.imgsz, args.calib)
        print(f"  生成 {dst.name}（{time.perf_counter() - t0:.1f} 秒）")
        r = bench(dst, args.imgsz, args.runs)
        results.append({**r, "kind": "INT8 静态"})
        print(f"  {r['size_mb']} MB  p50 {r['p50_ms']} ms")

    if len(results) > 1:
        print("\n" + "=" * 58)
        print("  对比表")
        print("=" * 58)
        base = results[0]["p50_ms"]
        print(f"  {'方案':<14}{'体积MB':>9}{'p50 ms':>10}{'加速比':>9}")
        for r in results:
            print(f"  {r['kind']:<14}{r['size_mb']:>9}{r['p50_ms']:>10}{base / r['p50_ms']:>8.2f}x")

        out = ROOT / "reports" / "quantization.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  已存 {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
