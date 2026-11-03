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


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    # 归一化到原图 [0,1] 的 xyxy
    box: tuple[float, float, float, float]


class BaseDetector:
    name: str = "base"
    is_mock: bool = False

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
    """ONNX Runtime CPU 推理。

    对模型输出格式的要求（**这是最容易踩的坑**）：
        ultralytics YOLO 默认导出的是 [1, 84, 8400]，里面是未解码的 DFL 分布，
        直接用会得到一堆乱框。导出时加 nms=True，输出才变成带 NMS 的 [1, N, 6+]：

            detect 模型：[1, 300, 6]      x1,y1,x2,y2,score,class_id
            seg 模型：   [1, 300, 38]     x1,y1,x2,y2,score,class_id + 32 个 mask 系数
                         （本项目是 seg 模型，只取前 6 列；mask 系数暂时丢掉）

        ⚠ 导出的框坐标是在 **letterbox 后的 imgsz×imgsz 画布上**（已实测），
          不是原图像素 —— 所以下面的反 letterbox 是必需的，不能省。

    导出命令（Windows PowerShell，在项目根目录）：
        python -c "from ultralytics import YOLO; YOLO('runs/segment/baseline_v1/weights/best.pt')\
            .export(format='onnx', imgsz=1280, opset=12, simplify=True, nms=True, device='cpu')"

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

        preds = self.session.run(None, {self.input_name: blob})[0]
        preds = np.asarray(preds)
        if preds.ndim == 3:
            preds = preds[0]

        out: list[Detection] = []
        for row in preds:
            if row.shape[0] < 6:
                continue
            x1, y1, x2, y2, score, cls = row[:6]
            if float(score) < thr:
                continue
            # 从 letterbox 坐标还原到原图，再归一化
            ox1 = max(0.0, (float(x1) - left) / r) / w
            oy1 = max(0.0, (float(y1) - top) / r) / h
            ox2 = min(float(w), (float(x2) - left) / r) / w
            oy2 = min(float(h), (float(y2) - top) / r) / h
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            idx = int(cls)
            out.append(
                Detection(
                    label=LABELS[idx] if 0 <= idx < len(LABELS) else f"cls{idx}",
                    score=round(float(score), 3),
                    box=(round(ox1, 4), round(oy1, 4), round(ox2, 4), round(oy2, 4)),
                )
            )
        return out


def build_detector(model_path: Path, conf: float = 0.25,
                   imgsz: int = int(os.getenv("DV_IMGSZ", "1280"))) -> BaseDetector:
    """优先用真模型；条件不满足就退回 Mock，并在控制台说清楚为什么。"""
    if not Path(model_path).exists():
        print(f"[detector] 没找到模型 {model_path}，先用 MockDetector（结果是假的）")
        return MockDetector(conf=conf)
    try:
        det = OnnxDetector(model_path, conf=conf)
        print(f"[detector] 已加载 {model_path.name}")
        return det
    except ImportError:
        print("[detector] 没装 onnxruntime，先用 MockDetector。装：pip install onnxruntime")
        return MockDetector(conf=conf)
    except Exception as exc:  # 模型损坏、opset 不兼容等
        print(f"[detector] 加载失败（{exc}），回退 MockDetector")
        return MockDetector(conf=conf)
