#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检测器：接口 + 两个实现。

为什么要抽象成一层：
    现在没有模型，必须有东西让 API 先跑起来（MockDetector）；
    等 W3 训出 best.onnx，换一行配置就切过去，上层接口代码一个字不改。
    这就是「可插拔」——面试时能讲成一个架构决策，不是凑代码。

⚠ 重要：is_mock=True 时返回的是**假数据**。
   接口响应里专门带了这个字段，就是为了任何时候都不把假结果当真。
"""

from __future__ import annotations

import os

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LABELS = ["scratch", "stain", "missing", "deform"]


def _sha256_prefix(path: Path, n: int = 12) -> str:
    """模型文件的 sha256 前 n 位。

    为什么记这个而不是只记文件名：每次训练导出的都叫 best.onnx，
    名字一样、内容完全不同。要看「这条记录是哪版权重跑的」只能看内容摘要。
    按 1 MB 分块读，避免把 11.6 MB 一次性进内存（模型以后会更小，习惯先立好）。
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    # 归一化到原图 [0,1] 的 xyxy
    box: tuple[float, float, float, float]
    # 归一化到原图 [0,1] 的多边形轮廓（分割模型才有；检测框模型为 None）
    polygon: tuple[tuple[float, float], ...] | None = None
    # 同一块框里的全部轮廓（断续划痕会拆成多段）。polygon 恒等于 polygons[0]。
    polygons: tuple[tuple[tuple[float, float], ...], ...] | None = None


class BaseDetector:
    name: str = "base"
    is_mock: bool = False
    # 可追溯字段：老版本没有，统一按 None 处理（查询端显示「未知」，不做推断）
    model_sha: str | None = None
    imgsz: int | None = None

    def __init__(self, conf: float = 0.25) -> None:
        self.conf = conf

    def predict(self, img: np.ndarray, conf: float | None = None) -> list[Detection]:  # pragma: no cover
        """conf 为 None 时用构造时传入的默认值；否则按调用方给的阈值过滤。"""
        raise NotImplementedError


class MockDetector(BaseDetector):
    """假检测器：结果由图片内容哈希决定，同一张图每次都一样（可复现）。

    存在的唯一意义：让 /detect 的整条链路（上传 → 解码 → 推理接口 → 落库 → 返回）
    今晚就能跑通并被测试，而不是等模型训完才能验证工程。
    """

    name = "mock-v0"
    is_mock = True

    def predict(self, img: np.ndarray, conf: float | None = None) -> list[Detection]:
        h, w = img.shape[:2]
        digest = hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()
        seed = int(digest[:12], 16)
        rng = random.Random(seed)

        # 约 1/3 的图判为良品，其余 1~2 个缺陷
        if seed % 3 == 0:
            return []

        out: list[Detection] = []
        for _ in range(1 + seed % 2):
            bw = rng.uniform(0.05, 0.22)
            bh = rng.uniform(0.05, 0.22)
            x1 = rng.uniform(0.02, 0.98 - bw)
            y1 = rng.uniform(0.02, 0.98 - bh)
            score = round(rng.uniform(0.35, 0.93), 3)
            if score < (self.conf if conf is None else conf):
                continue
            out.append(
                Detection(
                    label=rng.choice(LABELS),
                    score=score,
                    box=(round(x1, 4), round(y1, 4), round(x1 + bw, 4), round(y1 + bh, 4)),
                )
            )
        return out


class OnnxDetector(BaseDetector):
    """ONNX Runtime CPU 推理（自己解码 + 自己 NMS + 自己还原掩膜）。

    输出格式（ultralytics 标准导出，**不加 nms=True**）：

        output0: [1, 4 + nc + 32, 33600]   cx,cy,w,h + nc 个类别分数 + 32 个 mask 系数
        output1: [1, 32, 320, 320]         32 张 mask 原型

    ⚠ 为什么不用 nms=True 导出（踩过一次，值得记）：
      加 nms=True 后输出变成 [1, 300, 38]，框是好用的，但**同一行里的 32 个 mask
      系数和这张图的掩膜已经对不上了**。实测拿它去和 PyTorch 的真掩膜算空间相关系数
      只有 0.083（≈无关），解出来的"多边形"是满屏噪点。
      所以要做掩膜，就必须拿不带 NMS 的导出、在后处理里自己算。

    顺带解决另一个老问题：阈值写进模型后，运行时只能往上调、不能往下调；
    现在 NMS 在后处理里，`conf` 参数任意调。

    导出命令（项目根目录）：
        python -c "from ultralytics import YOLO; YOLO('runs/segment/baseline_v4/weights/best.pt')\
            .export(format='onnx', imgsz=1280, opset=12, simplify=True, nms=False, device='cpu')"

    没装 onnxruntime、或模型文件不存在时，build_detector() 会回退到 MockDetector。
    """

    name = "onnx-cpu"

    def __init__(self, model_path: Path, conf: float = 0.25,
                 imgsz: int = int(os.getenv("DV_IMGSZ", "1280"))) -> None:
        import onnxruntime as ort  # 延迟导入：没装也不影响其它功能

        self.conf = conf
        self.imgsz = imgsz
        self.model_path = Path(model_path)
        # 线程数是这台机器上最便宜的优化：默认用满 20 线程反而更慢。
        # 实测 1280 输入 p50：1 线程 1062ms / 4 线程 402ms / 8 线程 265ms / 20 线程 385ms。
        # 20 核里 8 个 P 核 + 12 个 E 核，线程开满后调度器把活分给慢核，同步等待成了瓶颈。
        n_threads = int(os.getenv("DV_THREADS", "8"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = n_threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), opts, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.name = f"onnx:{self.model_path.name}"
        self.threads = n_threads
        # 模型内容摘要（sha256 前 12 位）。
        # 为什么不用文件名：每次训练导出的都叫 best.onnx，文件名一样内容完全不同。
        # 11.6 MB 读一遍几十毫秒，只在启动时算一次，换来的能力是
        # 「这条检测记录到底是哪版权重跑出来的」以后能查。
        self.model_sha = _sha256_prefix(self.model_path)

    # -- 预处理 ------------------------------------------------------
    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """等比缩放 + 灰边填充到 imgsz×imgsz。

        千万不能直接 cv2.resize 到 640×640：那会把缺陷的宽高比拉变形，
        模型学到的是变形的划痕，现场换个分辨率就崩。
        """
        import cv2

        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
        return canvas, r, left, top

    def predict(self, img: np.ndarray, conf: float | None = None) -> list[Detection]:
        thr = self.conf if conf is None else conf
        h, w = img.shape[:2]
        canvas, r, left, top = self._letterbox(img)

        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, 归一化
        blob = np.transpose(blob, (2, 0, 1))[None]            # HWC -> NCHW

        outs = self.session.run(None, {self.input_name: blob})
        pred = np.asarray(outs[0])[0].T           # (4+nc+32, N) -> (N, ...)
        protos = np.asarray(outs[1])[0] if len(outs) > 1 else None

        nc = len(LABELS)
        if pred.shape[1] < 4 + nc:
            raise RuntimeError(f"模型输出只有 {pred.shape[1]} 列，装不下 {nc} 个类别")

        cls_scores = pred[:, 4 : 4 + nc]
        cls = cls_scores.argmax(axis=1)
        score = cls_scores[np.arange(len(pred)), cls]

        keep = score >= thr
        if not keep.any():
            return []

        # cx,cy,w,h（画布像素）→ xyxy（画布像素）
        cx, cy, bw, bh = pred[keep, 0], pred[keep, 1], pred[keep, 2], pred[keep, 3]
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2

        keep_idx = np.flatnonzero(keep)
        kept = self._nms(x1, y1, x2, y2, score[keep], iou_thr=0.5)

        coefs = pred[keep, 4 + nc :] if protos is not None else None

        out: list[Detection] = []
        for j in kept:
            k = keep_idx[j]
            ox1 = max(0.0, (float(x1[j]) - left) / r) / w
            oy1 = max(0.0, (float(y1[j]) - top) / r) / h
            ox2 = min(float(w), (float(x2[j]) - left) / r) / w
            oy2 = min(float(h), (float(y2[j]) - top) / r) / h
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            idx = int(cls[k])
            rings = (
                self._decode_mask(coefs[j], protos, (x1[j], y1[j], x2[j], y2[j]),
                                  left, top, r, w, h)
                if coefs is not None else None
            )
            out.append(
                Detection(
                    label=LABELS[idx] if 0 <= idx < len(LABELS) else f"cls{idx}",
                    score=round(float(score[k]), 3),
                    box=(round(ox1, 4), round(oy1, 4), round(ox2, 4), round(oy2, 4)),
                    polygon=rings[0] if rings else None,
                    polygons=tuple(rings) if rings else None,
                )
            )
        return out

    @staticmethod
    def _nms(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
             scores: np.ndarray, iou_thr: float = 0.5) -> list[int]:
        """按分数从高到低贪心抑制：和已保留框的 IoU 超过阈值就丢掉。"""
        order = scores.argsort()[::-1]
        areas = (x2 - x1) * (y2 - y1)
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            iw = np.maximum(0.0, xx2 - xx1)
            ih = np.maximum(0.0, yy2 - yy1)
            inter = iw * ih
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_thr]
        return keep

    def _decode_mask(self, coef: np.ndarray, proto: np.ndarray,
                     box_canvas: tuple[float, float, float, float],
                     left: int, top: int, r: float, w: int, h: int,
                     thr: float = 0.25, min_area: int = 12
                     ) -> list[tuple[tuple[float, float], ...]]:
        """32 个系数 × 32 张 mask 原型 → 一张轮廓图 → 简化成多边形。

        返回**多段**（按面积从大到小）。断续的划痕会被膨胀切成好几段，
        只取最大那段会把真实范围画短；第一段仍然当 `polygon` 用，兼容老数据。

        两个必须做的裁剪（少一步都会画歪）：
        ① 先把内容区（去掉 letterbox 灰边）裁出来；
        ② **再把检测框外面全部清零** —— 不裁的话，纸的边缘、台灯的亮斑
           也会被这套系数点亮，面积比缺陷大得多，"最大连通块"就会选到纸边上去。
           实测：不裁框时多边形框住半张纸，裁了才贴着划痕走。
        """
        import cv2

        ph, pw = proto.shape[1], proto.shape[2]
        flat = proto.reshape(proto.shape[0], -1).astype(np.float32)
        logits = np.clip(coef.astype(np.float32) @ flat, -30.0, 30.0)
        mask = (1.0 / (1.0 + np.exp(-logits))).reshape(ph, pw)

        # 原型网格 → 画布像素的缩放倍率
        scale = float(self.imgsz) / pw
        # 内容区在原型网格上的范围（去掉上下左右灰边）
        mx1 = int(round(left / scale))
        my1 = int(round(top / scale))
        mx2 = int(round((left + round(w * r)) / scale))
        my2 = int(round((top + round(h * r)) / scale))
        mx2 = max(mx1 + 2, min(mx2, pw))
        my2 = max(my1 + 2, min(my2, ph))
        crop = mask[my1:my2, mx1:mx2].copy()

        # ② 检测框以外清零（坐标都是原型网格坐标）
        bx1 = max(0, int(round(box_canvas[0] / scale)) - mx1)
        by1 = max(0, int(round(box_canvas[1] / scale)) - my1)
        bx2 = min(crop.shape[1], int(round(box_canvas[2] / scale)) - mx1)
        by2 = min(crop.shape[0], int(round(box_canvas[3] / scale)) - my1)
        if bx2 - bx1 < 2 or by2 - by1 < 2:
            return []
        inside = np.zeros_like(crop)
        inside[by1:by2, bx1:bx2] = crop[by1:by2, bx1:bx2]
        crop = inside

        # ③ 上采样再做阈值和轮廓
        #    原型只有 320×320，一个原型像素 = 原图约 12.6 px。
        #    20 px 宽的划痕在原型网格上只剩 1.6 px，直接阈值会把两端吃掉，
        #    轮廓点也会卡在 12 px 的格子上。先放大 8 倍再找轮廓，边缘才连续。
        UP = 8
        big = cv2.resize(crop, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
        binary = (big > thr).astype(np.uint8)
        # 再膨胀两个上采样像素（约 3 个原图像素），把断开的笔画粘起来
        binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
        if int(binary.sum()) < min_area * UP * UP // 8:
            return []

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        # 断续的划痕会被膨胀切成一串小段。只取最大那段 → 低估真实范围；
        # 所以这里按面积从大到小保留所有「够大」的段（≥ 最大段的 30%），
        # 第一段仍然作为 polygon 返回（老字段、老数据都能读），全部段放进 polygons。
        areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)
        keep_min = max(areas[0] * 0.30, float(min_area) * UP * UP / 8.0)
        rings: list[tuple[tuple[float, float], ...]] = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(c) < keep_min:
                continue
            eps = 0.002 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
            if approx.shape[0] < 3:
                continue

            # 轮廓点（上采样网格）→ 原型网格 → 画布像素 → 原图像素 → 归一化
            approx[:, 0] = (approx[:, 0] / UP + mx1) * scale
            approx[:, 1] = (approx[:, 1] / UP + my1) * scale
            approx[:, 0] = (approx[:, 0] - left) / r / w
            approx[:, 1] = (approx[:, 1] - top) / r / h
            approx[:, 0] = np.clip(approx[:, 0], 0.0, 1.0)
            approx[:, 1] = np.clip(approx[:, 1], 0.0, 1.0)
            rings.append(tuple((round(float(px), 4), round(float(py), 4))
                               for px, py in approx))
        return rings

    def _decode_mask_unused(self, coef: np.ndarray, proto: np.ndarray,
                            left: int, top: int, r: float, w: int, h: int,
                            thr: float = 0.5, min_area: int = 20
                            ) -> tuple[tuple[float, float], ...] | None:
        """（旧版，已废弃）原型按整幅 imgsz 画布映射，实测会偏。"""
        import cv2

        ch, cw = proto.shape[1], proto.shape[2]
        flat = proto.reshape(proto.shape[0], -1).astype(np.float32)
        logits = coef.astype(np.float32) @ flat
        mask = (1.0 / (1.0 + np.exp(-logits))).reshape(ch, cw)
        binary = (mask > thr).astype(np.uint8)
        if int(binary.sum()) < min_area:
            return None

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        biggest = max(contours, key=cv2.contourArea)
        eps = 0.002 * cv2.arcLength(biggest, True)
        approx = cv2.approxPolyDP(biggest, eps, True).reshape(-1, 2).astype(np.float32)
        if approx.shape[0] < 3:
            return None

        scale = float(self.imgsz) / cw          # 160 → 1280
        approx = approx * scale
        approx[:, 0] = (approx[:, 0] - left) / r
        approx[:, 1] = (approx[:, 1] - top) / r
        approx[:, 0] = np.clip(approx[:, 0], 0.0, float(w)) / w
        approx[:, 1] = np.clip(approx[:, 1], 0.0, float(h)) / h
        return tuple((round(float(px), 4), round(float(py), 4)) for px, py in approx)


def _mock_allowed() -> bool:
    """是否允许在模型加载失败时退回假模型。

    默认 **不允许**。原因：假模型返回的是编造的结果，界面上只多一行小字，
    产线上没人会注意到「今天所有件都判成 NG」其实是模型没加载上。
    想看演示 / 跑测试，显式设 `DV_ALLOW_MOCK=1`；生产环境不设这个变量。

    口径统一在 config.ALLOW_MOCK，这里只做一层兜底（detector 也允许脱离
    config 单独用，比如脚本里直接 import）。
    """
    try:
        from . import config
        return bool(config.ALLOW_MOCK)
    except Exception:
        return os.getenv("DV_ALLOW_MOCK", "0").strip().lower() in {"1", "true", "yes", "on"}


def build_detector(model_path: Path, conf: float = 0.25,
                   imgsz: int = int(os.getenv("DV_IMGSZ", "1280"))) -> BaseDetector:
    """加载真模型。加载不上时：默认**直接抛错退出**，只有显式允许才退回 Mock。"""

    def _fallback(reason: str) -> BaseDetector:
        if _mock_allowed():
            print(f"[detector] {reason} → 已按 DV_ALLOW_MOCK=1 退回 MockDetector（结果是假的）")
            return MockDetector(conf=conf)
        raise RuntimeError(
            f"[detector] 模型不可用：{reason}\n"
            f"  生产模式不允许静默降级成假模型，服务直接退出。\n"
            f"  只做演示/测试：设环境变量 DV_ALLOW_MOCK=1\n"
            f"  真要检测：确认 {model_path} 存在，且 pip install onnxruntime"
        )

    if not Path(model_path).exists():
        return _fallback(f"找不到模型文件 {model_path}")
    try:
        det = OnnxDetector(model_path, conf=conf)
        print(f"[detector] 已加载 {model_path.name}（sha {det.model_sha}）")
        return det
    except ImportError:
        return _fallback("没装 onnxruntime（pip install onnxruntime）")
    except Exception as exc:  # 模型损坏、opset 不兼容等
        return _fallback(f"加载失败：{exc}")
