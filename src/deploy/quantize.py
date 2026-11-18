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
import os
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


def model_meta(path: Path) -> dict:
    """读模型的真实输入尺寸与输出契约。

    踩过的坑：`--imgsz` 默认写过 640，而部署模型是 1280 —— 用 640 的随机张量
    去喂 1280 的模型，要么直接报错，要么（在旧版本里）悄悄拿错产物做对比。
    所以尺寸一律从模型里读，不许由人传。

    输出契约也要记：带 NMS 的导出是 `[1,300,38]`，不带 NMS 的是 `[1,40,33600]`。
    两种契约的延迟不可比 —— 一个把 NMS 算在模型里，一个不算。
    """
    import hashlib

    import onnx

    m = onnx.load(str(path), load_external_data=False)
    opset = max((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), default=0)
    in_dims = [d.dim_value or d.dim_param for d in m.graph.input[0].type.tensor_type.shape.dim]
    out_shapes = [
        [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim] for o in m.graph.output
    ]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    h, w = int(in_dims[2]), int(in_dims[3])
    contract = "nms" if any(s[-1] == 38 for s in out_shapes if len(s) == 3) else "nonms"
    return {
        "file": path.name,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256_12": digest,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "opset": opset,
        "imgsz": [h, w],
        "output_shapes": out_shapes,
        "contract": contract,
    }


def bench(path: Path, imgsz: int | None = None, n: int = 30, warmup: int = 5,
          threads: int = 0) -> dict:
    """测延迟。imgsz 不给就从模型里读；给了但和模型不一致就报错退出。"""
    meta = model_meta(path)
    real = meta["imgsz"][0]
    if imgsz is not None and imgsz != real:
        raise SystemExit(
            f"❌ --imgsz {imgsz} 与模型实际输入 {real} 不一致：{path.name}\n"
            f"   尺寸必须与导出时一致，否则比的不是同一个模型。"
        )

    sess = load_sess(path, threads)
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, real, real).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {name: x})

    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1000)

    ts_arr = np.array(ts)
    return {
        **meta,
        "p50_ms": round(float(np.percentile(ts_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(ts_arr, 95)), 2),
        "mean_ms": round(float(ts_arr.mean()), 2),
        "min_ms": round(float(ts_arr.min()), 2),
        "runs": n,
        "warmup": warmup,
        "threads": threads,
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


def exclude_near_outputs(path: Path, hops: int = 2) -> list[str]:
    """排除「离模型输出 hops 跳以内」的节点。

    为什么必须这么做（这个坑花了一整晚）：
        `output0` 是 `[1, 40, 33600]` 的一张**混合**张量 ——
        第 0–3 行是框坐标（0～1300 像素），第 4–7 行是类别置信度（0～1）。
        量化是**整个张量共用一个 scale**：max≈1282 → scale≈1282/255≈5。
        于是 0.51 的置信度被四舍五入成 **0**，模型对所有图片都返回「没有缺陷」。

    实测：不排除 → 类别分数一律 0.000，17 张缺陷图全部漏检；
          排除输出前 2 跳（13 个节点）→ 0.527 / 0.555，与 FP32 的 0.511 / 0.563 基本一致。
    """
    import onnx

    g = onnx.load(str(path)).graph
    producer = {o: n for n in g.node for o in n.output}
    frontier = {o.name for o in g.output}
    near: set[str] = set(frontier)
    for _ in range(hops + 1):
        nxt: set[str] = set()
        for t in frontier:
            n = producer.get(t)
            if n is not None:
                near.update(n.output)
                nxt.update(n.input)
        frontier = nxt
    return [n.name for n in g.node if n.name and any(o in near for o in n.output)]


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
    # 不能因为"文件已存在"就跳过：源模型换了（比如 640 换成 1280），
    # 旧转换产物会静默留下来，calibration 时才会以
    # "Got invalid dimensions ... Expected: 640" 的形式暴雷。
    conv = version_converter.convert_version(m, 13)
    onnx.save(conv, str(dst))
    print(f"  opset {cur} → 13：{dst.name}（输入 {conv.graph.input[0].type.tensor_type.shape.dim[2].dim_value}）")
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
    """从训练集取真实图片做校准（LetterBox 到**模型实际尺寸**，BGR->RGB，归一化）。

    ⚠ 踩过的坑：这里曾经用 `--imgsz` 的默认值 640，而模型是 1280。
    校准图被压到 640 塞进 1280 的模型，统计出来的激活范围完全是错的 ——
    量化后的模型对所有图片都返回 0 个框（置信度恒为 0），
    但脚本一路打印"成功"，没人发现。现在 imgsz 必须由调用方从模型里读出来传进来。
    """
    import cv2
    from onnxruntime.quantization import CalibrationDataReader

    img_dir = ROOT / "data" / "yolo" / "images" / "train"
    files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    files = files[:limit]
    if not files:
        raise SystemExit(f"校准集为空: {img_dir}")

    print(f"  校准图 {len(files)} 张（来自 train，绝不能用 val/test，否则泄漏）")
    print(f"  校准尺寸 {imgsz}（与模型输入一致才算对）")

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


def quantize_static_mode(src: Path, dst: Path, imgsz: int | None, calib: int,
                         op_types: list[str] | None = None,
                         method: str = "minmax",
                         head_hops: int = 2,
                         fmt: str = "qoperator") -> None:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    if imgsz is None:
        imgsz = model_meta(src)["imgsz"][0]

    cmap = {"minmax": CalibrationMethod.MinMax,
            "percentile": CalibrationMethod.Percentile,
            "entropy": CalibrationMethod.Entropy}
    fmap = {"qdq": QuantFormat.QDQ, "qoperator": QuantFormat.QOperator}
    op13 = ensure_opset13(src)
    excl = exclude_nodes(op13) + exclude_near_outputs(op13, head_hops)
    print(f"  量化格式 {fmt} · 排除 {len(excl)} 个节点（含输出前 {head_hops} 跳）")
    reader = build_calib_reader(src, imgsz, calib)
    quantize_static(
        model_input=str(op13),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=fmap[fmt],
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=excl,
        per_channel=True,
        calibrate_method=cmap[method],
        op_types_to_quantize=op_types,
    )


# --------------------------------------------------------------------------- #
# 量化后必须做的一致性检查
# --------------------------------------------------------------------------- #
def _letterbox_file(path: Path, imgsz: int) -> np.ndarray:
    import cv2

    im = cv2.imread(str(path))
    h, w = im.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top:top + nh, left:left + nw] = im
    return canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0


def _detect_count(outs: list, contract: str, thr: float) -> tuple[int, float]:
    """两种输出契约的置信度取法不同，取错了就永远统计成 0。"""
    if contract == "nms":
        conf = outs[0][0][:, 4]          # [300, 38]：第 5 列是置信度
    else:
        conf = outs[0][0][4:8].max(axis=0)  # [40, 33600]：第 5–8 行是 4 类分数
    n = int((conf >= thr).sum())
    return n, (float(conf.max()) if conf.size else 0.0)


def sanity(fp32_path: Path, int8_path: Path, thr: float = 0.05, limit: int = 20) -> dict | None:
    """拿真实图逐张对比 FP32 与 INT8 的检出数。

    只测延迟不测精度 = 把一个「什么都不输出」的模型当成加速 1.7 倍卖出去。
    实测就发生过：静态量化后所有缺陷图返回 0 个框，判定一致率 57.5%（那 57.5% 全是良品）。
    """
    m32 = model_meta(fp32_path)
    m8 = model_meta(int8_path)
    if m32["contract"] != m8["contract"]:
        print("  ⚠ 两个模型输出契约不同，无法逐图对比（先确认是同一版权重导出的）")
        return None

    imgsz = m32["imgsz"][0]
    s32 = load_sess(fp32_path)
    s8 = load_sess(int8_path)
    name = s32.get_inputs()[0].name

    files = sorted((ROOT / "data" / "yolo" / "images" / "val").glob("*.jpg"))[:limit]
    rows = []
    for p in files:
        x = _letterbox_file(p, imgsz)
        n32, c32 = _detect_count(s32.run(None, {name: x[None]}), m32["contract"], thr)
        n8, c8 = _detect_count(s8.run(None, {name: x[None]}), m8["contract"], thr)
        rows.append({"file": p.name, "fp32_n": n32, "fp32_conf": c32,
                     "int8_n": n8, "int8_conf": c8,
                     "same": (n32 > 0) == (n8 > 0)})

    same = sum(r["same"] for r in rows)
    zero8 = sum(1 for r in rows if r["int8_n"] == 0)
    real_defect = sum(1 for r in rows if r["fp32_n"] > 0)
    print(f"\n  {'文件':<34}{'FP32 检出/置信':>20}{'INT8 检出/置信':>20}{'判定':>8}")
    print("  " + "-" * 82)
    for r in rows:
        print(f"  {r['file']:<34}{r['fp32_n']:>6} / {r['fp32_conf']:>8.3f}"
              f"{r['int8_n']:>14} / {r['int8_conf']:>10.3f}{'  一致' if r['same'] else '  ⚠ 翻转'}")
    print(f"\n  判定一致率 {same}/{len(rows)} = {same / len(rows):.1%}")
    if real_defect and zero8 == len(rows):
        print("  ❌ INT8 对所有图片都返回 0 个框 —— 校准尺寸或校准方法有问题，这个模型不能用")
    elif real_defect and all(r["int8_n"] == 0 for r in rows if r["fp32_n"] > 0):
        print("  ❌ 所有缺陷图都被 INT8 漏检 —— 这个模型不能用")
    return {"same": same, "total": len(rows), "rows": rows}


# --------------------------------------------------------------------------- #
# 报告落盘（json 和 md 必须来自同一次运行，否则迟早对不上）
# --------------------------------------------------------------------------- #
def environment() -> dict:
    import platform

    import onnxruntime

    return {
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "onnxruntime": onnxruntime.__version__,
        "python": platform.python_version(),
        "cpu_logical": os.cpu_count(),
    }


def write_reports(results: list[dict], env: dict) -> tuple[Path, Path]:
    """一次运行同时写 JSON 和 Markdown —— 两份文件不可能再打架。"""
    payload = {"env": env, "results": results}
    js = ROOT / "reports" / "quantization.json"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    groups: dict[tuple, list[dict]] = {}
    for r in results:
        groups.setdefault((r["imgsz"][0], r["contract"]), []).append(r)

    lines = [
        "# 量化与延迟对比（自动生成，勿手改）",
        "",
        "> 本文件由 `src/deploy/quantize.py` 生成。数字与 `quantization.json` 来自**同一次运行**。",
        "> 复现命令、模型 sha、线程数、运行次数全部记在下面 —— 缺任何一项，这份数字就不可引用。",
        "",
        "## 运行环境",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 生成时间 | {env['run_at']} |",
        f"| onnxruntime | {env['onnxruntime']} |",
        f"| Python | {env['python']} |",
        f"| 逻辑 CPU | {env['cpu_logical']} |",
        "",
    ]

    for (size, contract), rows in sorted(groups.items()):
        name = "带 NMS" if contract == "nms" else "不带 NMS"
        fp32 = next((r for r in rows if "int8" not in r["file"].lower()), None)
        base = fp32["p50_ms"] if fp32 else min(r["p50_ms"] for r in rows)
        lines += [
            f"## {size}×{size} · {name}",
            "",
            f"输出契约：`{contract}` · 同组内对比有效，**跨组不可比**（NMS 算不算在模型里不一样）。",
            "",
            f"「相对 FP32」一列：数字 = FP32 p50 ÷ 本模型 p50。**大于 1 是变快，小于 1 是变慢。**"
            f"（基准：`{fp32['file'] if fp32 else '最快者'}` = 1.00×）",
            "",
            "| 模型 | 体积 MB | p50 ms | p95 ms | 相对 FP32 | sha256 |",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| `{r['file']}` | {r['size_mb']} | {r['p50_ms']} | {r['p95_ms']} | "
                f"{base / r['p50_ms']:.2f}× | `{r['sha256_12']}` |"
            )
        lines += ["", f"每组 {rows[0]['runs']} 次计时，预热 {rows[0]['warmup']} 次，"
                      f"线程 {rows[0]['threads'] or 'ORT 默认'}。", ""]

    lines += [
        "## 这份报告**不能**说明什么",
        "",
        "1. **只测了延迟，没测精度。** 量化会让模型变差还是不变，这个脚本不回答。",
        "   要回答得跑 `src/eval/quant_compare.py`，而且它目前只支持带 NMS 的 `[1,300,38]` 契约，",
        "   部署用的不带 NMS 模型它还比不了 —— 这是下一个要补的洞。",
        "2. **简历/汇报里不许写「量化无损」。** 没有精度对比就没有这个结论。",
        "3. 随机张量测的是算力，不是数据分布。真实图片的耗时可能略高（后处理另算）。",
        "",
        "## 复现命令",
        "",
        "```powershell",
        "python src\\deploy\\quantize.py --models models\\best.onnx models\\best_nonms_int8.onnx",
        "```",
        "",
        "> ⚠ 历史教训：此前的 `量化对比.md` 记 FP32 p50 **32.07 ms**（640、20 线程、baseline_v1），",
        "> 而 `quantization.json` 记 FP32 p50 **216.81 ms**（1280、默认线程、NMS 契约）。",
        "> 两个数都不是假的，但**它们不是同一个东西**，却被当成同一个指标引用了半个月。",
    ]

    md = ROOT / "reports" / "量化对比.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return js, md


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="INT8 量化与延迟对比")
    ap.add_argument("--model", default=str(MODELS / "best.onnx"))
    ap.add_argument("--models", nargs="*", default=None,
                    help="只跑基准：给一组模型路径，逐个体测并出报告（不量化）")
    ap.add_argument("--mode", choices=["dynamic", "static", "convonly", "bench-only", "all"],
                    default="dynamic")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="默认从模型里读；给了就必须与模型一致，否则报错")
    ap.add_argument("--calib", type=int, default=60, help="静态量化的校准图数量")
    ap.add_argument("--calib-method", choices=["minmax", "percentile", "entropy"],
                    default="minmax", help="校准方法；输出全零时先换 percentile 试试")
    ap.add_argument("--head-hops", type=int, default=2,
                    help="排除离输出多少跳以内的节点，默认 2（输出 Concat 混了框坐标和置信度）")
    ap.add_argument("--format", choices=["qoperator", "qdq"], default="qoperator",
                    help="qoperator 在 CPU 上更快，qdq 留给 NPU/GPU")
    ap.add_argument("--out", default=None, help="量化产物路径（不写会覆盖默认文件名）")
    ap.add_argument("--conf", type=float, default=0.05, help="一致性检查用的置信度阈值")
    ap.add_argument("--threads", type=int, default=0, help="intra_op 线程数，0=ORT 默认")
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()

    env = environment()

    # ---- 只跑基准：多模型横向对比 ----
    if args.models:
        results = []
        for raw in args.models:
            p = Path(raw)
            if not p.is_absolute():
                p = ROOT / raw
            if not p.exists():
                print(f"❌ 找不到模型: {p}")
                return 1
            r = bench(p, args.imgsz, args.runs, threads=args.threads)
            results.append(r)
            print(f"  {r['file']:<28} {r['imgsz'][0]:>5}  {r['size_mb']:>6} MB  "
                  f"p50 {r['p50_ms']:>7} ms  [{r['contract']}]")

        js, md = write_reports(results, env)
        print(f"\n  已写 {js.relative_to(ROOT)}")
        print(f"  已写 {md.relative_to(ROOT)}")
        return 0

    src = Path(args.model)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        print(f"❌ 找不到模型: {src}")
        return 1

    results = []

    print("=" * 58)
    print("  FP32 基线")
    print("=" * 58)
    fp32 = bench(src, args.imgsz, args.runs, threads=args.threads)
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
        r = bench(dst, args.imgsz, args.runs, threads=args.threads)
        results.append({**r, "kind": "INT8 动态"})
        print(f"  {r['size_mb']} MB  p50 {r['p50_ms']} ms")

    if args.mode in ("static", "convonly"):
        only_conv = args.mode == "convonly"
        title = "静态量化 · 只量化 Conv" if only_conv else "静态量化 (QDQ, 需校准)"
        print("\n" + "=" * 58)
        print(f"  {title}")
        print("=" * 58)
        dst = Path(args.out) if args.out else OUT_DIR / (
            "best_int8_convonly.onnx" if only_conv else "best_int8_static.onnx")
        if not dst.is_absolute():
            dst = ROOT / dst
        t0 = time.perf_counter()
        quantize_static_mode(src, dst, args.imgsz, args.calib,
                             op_types=["Conv"] if only_conv else None,
                             method=args.calib_method,
                             head_hops=args.head_hops,
                             fmt=args.format)
        print(f"  生成 {dst.name}（{time.perf_counter() - t0:.1f} 秒）")

        print("\n  --- 量化后一致性检查（不通过就别用） ---")
        report = sanity(src, dst, thr=args.conf)
        r = bench(dst, args.imgsz, args.runs, threads=args.threads)
        results.append({**r, "kind": "INT8 仅Conv" if only_conv else "INT8 静态",
                        "sanity": None if report is None else
                        f"{report['same']}/{report['total']}"})
        print(f"\n  {r['size_mb']} MB  p50 {r['p50_ms']} ms")

    if len(results) > 1:
        print("\n" + "=" * 58)
        print("  对比表")
        print("=" * 58)
        base = results[0]["p50_ms"]
        print(f"  {'方案':<14}{'体积MB':>9}{'p50 ms':>10}{'加速比':>9}")
        for r in results:
            print(f"  {r['kind']:<14}{r['size_mb']:>9}{r['p50_ms']:>10}{base / r['p50_ms']:>8.2f}x")

        js, md = write_reports(results, env)
        print(f"\n  已写 {js.relative_to(ROOT)}")
        print(f"  已写 {md.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
