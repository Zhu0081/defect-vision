# -*- coding: utf-8 -*-
"""接口回归测试。

跑：
    python -m pytest tests -q

覆盖的是「不依赖模型精度」的行为契约：
    - 健康检查、页面能打开
    - 空文件 / 不是图片 / 太大 → 分别是 400 / 400 / 413
    - 正常图 → 200 且记录真的落库了
    - 不存在的 id → 404

为什么不测 mAP：那是模型问题，数据一换就变；这里测的是**服务**不许坏。
"""

import cv2
import numpy as np


def _png_bytes(w: int = 64, h: int = 64) -> bytes:
    """现造一张纯色 PNG，不依赖任何测试图片文件。"""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok, "cv2.imencode 失败"
    return buf.tobytes()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # 走 mock 说明测试环境没有真模型，这是设计好的
    assert body["is_mock"] is True


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_empty_file(client):
    r = client.post("/detect", files={"file": ("a.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_not_an_image(client):
    r = client.post("/detect", files={"file": ("a.jpg", b"hello world", "image/jpeg")})
    assert r.status_code == 400


def test_too_large(client):
    # conftest 把上限调成了 1 MB，这里造 2 MB
    blob = b"\x00" * (2 * 1024 * 1024)
    r = client.post("/detect", files={"file": ("big.jpg", blob, "image/jpeg")})
    assert r.status_code == 413
    assert "MB" in r.json()["detail"]


def test_detect_and_persist(client):
    r = client.post(
        "/detect",
        files={"file": ("ok.png", _png_bytes(), "image/png")},
        data={"light": "side", "part_id": "001", "conf": "0.5"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["inspection_id"] > 0
    assert body["verdict"] in ("ok", "ng")
    assert body["image_url"] == f"/image/{body['inspection_id']}"

    # 关键断言：接口说保存了，那就必须真能查到
    got = client.get(f"/records/{body['inspection_id']}")
    assert got.status_code == 200
    assert got.json()["part_id"] == "001"


def test_record_not_found(client):
    r = client.get("/records/999999")
    assert r.status_code == 404


def test_records_limit_is_clamped(client):
    assert client.get("/records?limit=0").status_code == 200
    assert client.get("/records?limit=9999").status_code == 200


def test_stats_and_report(client):
    assert client.get("/stats").status_code == 200
    r = client.get("/report")
    assert r.status_code == 200
    assert "confusion" in r.json()
