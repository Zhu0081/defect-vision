#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冒烟测试：一条命令验证服务是否真的能用。

为什么要有这个文件：
    面试官问「你怎么确认改了代码没把接口改坏」，答「我跑冒烟测试」比「我点了点」强一档。
    以后每次改完跑一遍，2 秒出结果。

用法（服务已启动）：
    python src/service/smoke_test.py
    python src/service/smoke_test.py --base http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import io
import sys

import cv2
import numpy as np
import requests

# Windows 控制台默认 GBK，打印中文/emoji 会直接抛 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PASS, FAIL = "[PASS]", "[FAIL]"
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _failed
    if cond:
        print(f"{PASS} {name}  {detail}")
    else:
        _failed += 1
        print(f"{FAIL} {name}  {detail}")


def make_image(w: int = 800, h: int = 600) -> bytes:
    """造一张「像工件」的图：灰底 + 斜纹 + 两个暗斑。"""
    img = np.full((h, w, 3), 140, dtype=np.uint8)
    for x in range(0, w, 40):
        cv2.line(img, (x, 0), (x + 120, h), (120, 120, 120), 1)
    cv2.rectangle(img, (180, 160), (250, 200), (40, 40, 40), -1)
    cv2.circle(img, (560, 400), 26, (30, 30, 60), -1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buf.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    # 1) 健康检查
    try:
        r = requests.get(f"{base}/health", timeout=10)
    except requests.RequestException as exc:
        print(f"{FAIL} 连不上服务：{exc}")
        print("        先把服务起起来：python -m uvicorn src.service.main:app --port 8000")
        return 1
    check("GET /health", r.status_code == 200, f"{r.json()}")
    health = r.json()
    if health.get("is_mock"):
        print("      注意：当前是 Mock 模型，检测结果是假数据")

    # 2) 正常检测
    img_bytes = make_image()
    files = {"file": ("smoke.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    data = {"part_id": "T001", "light": "top", "batch_id": "SMOKE", "conf": "0.25"}
    r = requests.post(f"{base}/detect", files=files, data=data, timeout=30)
    check("POST /detect", r.status_code == 200, f"HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text[:800])
        return 1

    body = r.json()
    iid = body["inspection_id"]
    check("返回字段完整", all(k in body for k in
                              ("inspection_id", "verdict", "items", "latency_ms", "image_url")))
    check("verdict 合法", body["verdict"] in ("ok", "ng"), body["verdict"])
    check("n_defects 与 items 一致", body["n_defects"] == len(body["items"]))
    check("耗时是正数", body["latency_ms"] > 0, f"{body['latency_ms']} ms")

    # 3) 同一张图再传一次：Mock 是确定性的，结果应当一致
    files2 = {"file": ("smoke.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    r2 = requests.post(f"{base}/detect", files=files2, data=data, timeout=30)
    check("重复检测可复现", r2.status_code == 200 and r2.json()["n_defects"] == body["n_defects"],
          f"第一次 {body['n_defects']} 个 / 第二次 {r2.json().get('n_defects')} 个")

    # 4) 明细查询
    r = requests.get(f"{base}/records/{iid}", timeout=10)
    check("GET /records/{id}", r.status_code == 200 and r.json()["id"] == iid)

    # 5) 列表
    r = requests.get(f"{base}/records", params={"limit": 5}, timeout=10)
    check("GET /records", r.status_code == 200 and isinstance(r.json(), list),
          f"{len(r.json())} 条")

    # 6) 统计（GROUP BY 那条 SQL）
    r = requests.get(f"{base}/stats", timeout=10)
    check("GET /stats", r.status_code == 200, f"{r.json()}")

    # 7) 原图回取
    r = requests.get(f"{base}{body['image_url']}", timeout=10)
    check("GET /image/{id}", r.status_code == 200 and r.content[:2] == b"\xff\xd8",
          f"{len(r.content)} 字节")

    # 8) 错误处理：非图片应当 400，而不是 500
    files3 = {"file": ("bad.txt", io.BytesIO(b"not an image at all"), "text/plain")}
    r = requests.post(f"{base}/detect", files=files3, timeout=10)
    check("非图片返回 400", r.status_code == 400, f"HTTP {r.status_code}")

    # 9) 不存在的记录应当 404
    r = requests.get(f"{base}/records/99999999", timeout=10)
    check("不存在的 id 返回 404", r.status_code == 404, f"HTTP {r.status_code}")

    print()
    if _failed:
        print(f"{_failed} 项失败")
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
