#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 C++ 推理程序准备一份输入张量（.bin + .txt）。

为什么需要这一步：
    ONNX Runtime 的 C++ 接口只吃「已经是 float32、已经是 NCHW、已经是 1x3xHxW」的裸数组，
    它不管图片解码、不管缩放、不管归一化。Python 版是在 detector.py 里用 numpy 做的，
    C++ 版要么自己写 JPEG 解码（很重），要么先把这一步用 Python 做完。

    本项目选择：**先用 Python 生成输入，C++ 只负责推理**。
    这样 C++ 那一侧专注在真正要学的东西上 —— 会话创建、张量形状、输出解析。
    等这步跑通，再考虑用 stb_image 把解码也搬进 C++。

输出（默认写到 src/cpp/ 下）：
    input_1x3x1280x1280.bin   float32 小端，NCHW，RGB，/255 归一化
    input_meta.txt            四行：r / left / top / 原图宽高

用法：
    python src/cpp/make_input.py data\\captured\\light_side\\card\\card_p022_scratch_002.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

IMGSZ = 1280


def letterbox(img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """等比缩放 + 灰边(114)填充到 IMGSZ×IMGSZ。和 detector.py 里的实现保持一致。"""
    h, w = img.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python src/cpp/make_input.py <image.jpg>")
        return 2

    src = Path(sys.argv[1])
    img = cv2.imread(str(src))
    if img is None:
        print(f"读不到图片: {src}")
        return 1

    h, w = img.shape[:2]
    canvas, r, left, top = letterbox(img)

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0   # BGR->RGB, 归一化
    blob = np.transpose(blob, (2, 0, 1))[None]              # HWC -> NCHW
    blob = np.ascontiguousarray(blob, dtype=np.float32)

    out_dir = Path(__file__).resolve().parent
    bin_path = out_dir / f"input_1x3x{IMGSZ}x{IMGSZ}.bin"
    meta_path = out_dir / "input_meta.txt"

    blob.tofile(bin_path)                                   # 裸 float32，无文件头
    meta_path.write_text(
        f"{r}\n{left}\n{top}\n{w} {h}\n", encoding="utf-8"
    )

    print(f"图片   : {src}")
    print(f"原图   : {w} x {h}")
    print(f"缩放   : r={r:.6f}  left={left}  top={top}")
    print(f"张量   : {blob.shape}  {blob.dtype}  {blob.nbytes:,} 字节")
    print(f"写入   : {bin_path}")
    print(f"写入   : {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
