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


def test_conf_threshold_takes_effect(client):
    """回归测试：conf 必须真的影响结果。

    2026-11-03 的 bug —— 接口声明了 conf、校验了 conf、网页上也画了输入框，
    但 main.py 调 det.predict() 时压根没把它传进去，阈值是个装饰品。
    这里把阈值调到 0.99（任何预测分数都到不了），必须一个框都不剩。
    """
    png = _png_bytes()

    r = client.post(
        "/detect",
        files={"file": ("a.png", png, "image/png")},
        data={"conf": "0.99"},
    )
    assert r.status_code == 200
    assert r.json()["n_defects"] == 0, "conf=0.99 还有框，说明阈值没生效"
    assert r.json()["verdict"] == "ok"

    # 同一张图、低阈值，应该（至少有可能）有框 —— 证明差异来自 conf 而不是图片
    r2 = client.post(
        "/detect",
        files={"file": ("b.png", png, "image/png")},
        data={"conf": "0.01"},
    )
    assert r2.status_code == 200
    assert r2.json()["n_defects"] >= r.json()["n_defects"]


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


def test_report_excludes_mock_records(client):
    """回归测试：Mock 记录绝不能进统计。

    2026-11-17 的审计发现 —— 库里有 is_mock 列、写入也确实写了，
    但 /report 和 /stats 的 SQL 压根没过滤它。后果是：
    在没放模型的机器上跑一次演示，漏检率/过杀率/节拍全被假数据污染，
    而且看不出被污染过。

    测试环境本身就是 MockDetector（conftest 把模型路径指到不存在的文件），
    所以这里每做一次检测，就是在往库里塞一条 is_mock=true 的记录。
    正确的行为是：这些记录全部被排除，只计入 excluded_mock。
    """
    png = _png_bytes()
    before = client.get("/report").json()

    r = client.post("/detect", files={"file": ("m.png", png, "image/png")})
    assert r.status_code == 200

    after = client.get("/report").json()
    # 真模型记录数没变
    assert after["total_inspections"] == before["total_inspections"]
    # Mock 记录数 +1（且没有进 labeled）
    assert after["excluded_mock"] == before["excluded_mock"] + 1
    assert after["labeled"] == 0 or after["labeled"] == before["labeled"]

    # /stats 也不许出现 Mock 造成的行数变化
    stats_before = client.get("/stats").json()
    assert sum(row["n"] for row in stats_before) == after["total_inspections"]


def test_ai_report_falls_back_to_template(client):
    """没配 API Key 时，AI 报告必须降级成模板，而不是 500。

    这是这个功能最重要的测试：报告是锦上添花，绝不能因为它把主流程搞挂。
    conftest 已经把 DV_LLM_API_KEY 清空，所以这里走的一定是降级通道。
    """
    r = client.post("/detect", files={"file": ("r.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    iid = r.json()["inspection_id"]

    rep = client.post(f"/records/{iid}/report")
    assert rep.status_code == 200
    body = rep.json()
    assert body["source"] == "template"
    assert body["cached"] is False
    # 四个固定小标题，前端靠它渲染分节
    for sec in ("判定结论", "缺陷情况", "风险提示", "建议动作"):
        assert sec in body["text"], f"报告缺少小节：{sec}"

    # 说存了就必须真能查回来
    got = client.get(f"/records/{iid}").json()
    assert got["ai_report"] == body["text"]
    assert got["ai_report_source"] == "template"


def test_ai_report_is_cached(client):
    """同一条记录点第二次，不能重复生成（重复生成 = 重复花钱）。"""
    r = client.post("/detect", files={"file": ("c.png", _png_bytes(), "image/png")})
    iid = r.json()["inspection_id"]

    first = client.post(f"/records/{iid}/report").json()
    assert first["cached"] is False

    second = client.post(f"/records/{iid}/report").json()
    assert second["cached"] is True
    assert second["text"] == first["text"]

    # force=true 强制重新生成
    third = client.post(f"/records/{iid}/report?force=true").json()
    assert third["cached"] is False


def test_ai_report_unknown_id(client):
    assert client.post("/records/999999/report").status_code == 404


def test_defect_polygon_roundtrip(client):
    """轮廓必须能存进去再原样查回来。

    为什么单独测：MockDetector 不产出 polygon，走 /detect 测不到这条路径。
    而真模型产出的轮廓如果不落库，历史记录就只能重算一遍（P1-4 的问题）。
    """
    from src.service import db

    poly = [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]]
    iid = db.save_inspection(
        batch_id="B_POLY_TEST",
        filename="card_p001_scratch_001.jpg",
        image_path="x.jpg",
        part_id="001",
        light="side",
        model_name="test",
        is_mock=False,
        verdict="ng",
        n_defects=1,
        max_score=0.9,
        latency_ms=12.0,
        items=[
            {
                "label": "scratch", "score": 0.9,
                "x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4,
                "polygon": poly,
            },
            {   # 没有轮廓的缺陷：库里存 NULL，接口统一成 []
                "label": "stain", "score": 0.5,
                "x1": 0.5, "y1": 0.5, "x2": 0.6, "y2": 0.6,
                "polygon": [],
            },
        ],
    )

    got = client.get(f"/records/{iid}").json()
    by_label = {it["label"]: it for it in got["items"]}
    assert by_label["scratch"]["polygon"] == poly
    assert by_label["stain"]["polygon"] == []

    # 存成 JSON 文本（能被 json 解析回来），不是 Python repr 那种单引号串
    from sqlalchemy import text as _sql

    conn = db.get_engine().connect()
    raw = conn.execute(
        _sql("SELECT polygon FROM defect_item WHERE inspection_id = :i"), {"i": iid}
    ).all()
    assert any(r[0] and r[0].lstrip().startswith("[[") for r in raw)


def test_report_carries_its_scope(client):
    """统计必须把口径一起返回。

    为什么值得一条测试：同一份库，全历史是「漏检 19.39%」，
    只看当前模型上线后是「漏检 0%」。数字本身不会说话，口径才会。
    """
    body = client.get("/report").json()
    assert body["filters"] == {"since": None, "model": None, "model_sha": None}

    # 一个远在未来的 since：什么都筛不出来，而且必须是干干净净的 0，不是 500
    empty = client.get("/report?since=2999-01-01").json()
    assert empty["total_inspections"] == 0
    assert empty["labeled"] == 0
    assert empty["miss_rate_pct"] is None
    assert empty["filters"]["since"] is not None

    # 写错的日期要 400，不能静默当成「不过滤」
    assert client.get("/report?since=abc").status_code == 400

def test_report_returns_denominators(client):
    """比率必须连分母一起返回。

    为什么：`过杀率 0%` 可能是「真的零过杀」，也可能是「一条良品都没有」。
    两个 0% 在调用方看来长得一模一样 —— 这正是当初误读「漏检 0%」的病根。
    """
    body = client.get("/report").json()
    assert set(body["samples"]) == {"labeled", "positive", "negative"}
    s = body["samples"]
    assert s["positive"] + s["negative"] == s["labeled"] == body["labeled"]

    # 一个什么都没有的区间：分母是 0，两个比率都必须是 None 而不是 0
    empty = client.get("/report?since=2999-01-01").json()
    assert empty["samples"] == {"labeled": 0, "positive": 0, "negative": 0}
    assert empty["miss_rate_pct"] is None
    assert empty["overkill_rate_pct"] is None


def test_record_carries_fingerprint(client):
    """检测记录必须能回答「哪份权重 + 多大输入 + 什么阈值」。

    111 条历史里 model_name 只有一个值（onnx:best.onnx），却横跨 20 天、换过四版权重
    —— 只看文件名永远分不清。所以必须记内容摘要 + 推理参数。
    测试环境是 MockDetector（没有文件、没有 imgsz），此时三者应当是「如实为空」，
    不是编一个假值。
    """
    import cv2 as _cv2
    import numpy as _np

    ok, buf = _cv2.imencode(".png", _np.full((64, 64, 3), 180, dtype=_np.uint8))
    r = client.post("/detect", files={"file": ("fp.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200
    iid = r.json()["inspection_id"]

    rec = client.get(f"/records/{iid}").json()
    assert "model_sha" in rec and "imgsz" in rec and "conf_thr" in rec
    assert rec["model_sha"] is None      # Mock 没有模型文件，就不许有摘要
    assert rec["imgsz"] is None          # Mock 不做 letterbox，也不许有尺寸
    assert rec["conf_thr"] == 0.25       # 阈值是真实生效的值，必须记下来


def test_report_filters_by_model_sha(client):
    """按内容摘要筛选：老记录没有摘要，用任何 sha 去筛都应该是 0 条，而不是报错。"""
    body = client.get("/report?model_sha=deadbeef0000").json()
    assert body["filters"]["model_sha"] == "deadbeef0000"
    assert body["total_inspections"] == 0
    assert body["samples"]["labeled"] == 0
