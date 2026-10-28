#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缺陷检测服务（FastAPI）。

启动（在 F:\\defect-vision 目录下）：
    python -m uvicorn src.service.main:app --reload --port 8000

打开：
    http://127.0.0.1:8000/          ← 上传页面（手点就能测）
    http://127.0.0.1:8000/docs      ← 自动生成的接口文档（面试演示用这个）

这个文件里值得对着 JD 讲的东西：
    - 启动时建表（lifespan）而不是每次请求都建
    - 上传大小/格式双重校验，错误码分明 400/413
    - CPU 推理扔进线程池，不阻塞事件循环
    - 一次事务写「主表 + 明细表」
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import config, db, schemas
from .detector import build_detector

# 前端是纯静态单页（一个 html + 内联 css/js），不需要 node/打包工具。
# 放成独立文件而不是 python 字符串，改样式不用碰后端代码。
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时做的事：建目录、建表、加载模型。只做一次，不是每个请求做一次。"""
    db_name = db.init_db()
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app.state.detector = build_detector(config.MODEL_PATH, conf=config.DEFAULT_CONF)
    app.state.db_name = db_name
    print(f"[startup] 数据库={db_name}  模型={app.state.detector.name}  mock={app.state.detector.is_mock}")
    yield
    print("[shutdown] bye")


app = FastAPI(
    title="工业缺陷视觉检测服务",
    version="0.1.0",
    description="上传一张工件图 → 返回缺陷框与判定结果，并落库。",
    lifespan=lifespan,
)

# /static/... 留给以后放 js/css/图片；现在页面是单文件，先占位
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> FileResponse:
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail=f"前端文件缺失：{INDEX_HTML}")
    return FileResponse(str(INDEX_HTML), media_type="text/html; charset=utf-8")


@app.get("/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    det = app.state.detector
    return schemas.HealthResponse(
        status="ok",
        db=app.state.db_name,
        model_name=det.name,
        is_mock=det.is_mock,
        upload_dir=str(config.UPLOAD_DIR),
    )


@app.post("/detect", response_model=schemas.DetectResponse)
async def detect(
    file: UploadFile = File(..., description="待检测图像（jpg/png）"),
    batch_id: str | None = Form(None, description="批次号，不传自动生成"),
    part_id: str | None = Form(None, description="工件件号"),
    light: str | None = Form(None, description="光源：top/side/back"),
    conf: float = Form(config.DEFAULT_CONF, ge=0.0, le=1.0, description="置信度阈值"),
) -> schemas.DetectResponse:
    t0 = time.perf_counter()

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空文件")
    if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {config.MAX_UPLOAD_MB} MB")

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="不是可解码的图像（jpg/png？）")

    # 推理是 CPU 密集的，必须丢线程池，否则会卡住整个事件循环
    det = app.state.detector
    detections = await run_in_threadpool(det.predict, img)

    # 落盘：按日期分目录，避免一个目录几万张图
    day = datetime.now().strftime("%Y%m%d")
    sub = config.UPLOAD_DIR / day
    sub.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex[:16]}.jpg"
    path = sub / name
    ok = await run_in_threadpool(
        cv2.imwrite, str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    if not ok:
        raise HTTPException(status_code=500, detail="图像写入失败")

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    items = [
        {"label": d.label, "score": d.score, "x1": d.box[0], "y1": d.box[1],
         "x2": d.box[2], "y2": d.box[3]}
        for d in detections
    ]
    verdict = "ng" if items else "ok"
    max_score = max((d.score for d in detections), default=0.0)

    # 只生成一次，落库和返回必须用同一个值（否则返回的批次号对不上库里的）
    bid = batch_id or f"B{datetime.now():%Y%m%d%H%M%S}"

    iid = await run_in_threadpool(
        db.save_inspection,
        batch_id=bid,
        filename=file.filename or name,
        image_path=str(path.relative_to(config.BASE_DIR)).replace("\\", "/"),
        part_id=part_id,
        light=light,
        model_name=det.name,
        is_mock=det.is_mock,
        verdict=verdict,
        n_defects=len(items),
        max_score=float(max_score),
        latency_ms=latency_ms,
        items=items,
    )

    return schemas.DetectResponse(
        inspection_id=iid,
        batch_id=bid,
        verdict=verdict,
        n_defects=len(items),
        max_score=float(max_score),
        latency_ms=latency_ms,
        model_name=det.name,
        is_mock=det.is_mock,
        items=[
            schemas.DefectItem(label=i["label"], score=i["score"],
                               box=[i["x1"], i["y1"], i["x2"], i["y2"]])
            for i in items
        ],
        image_url=f"/image/{iid}",
    )


@app.get("/records", response_model=list[schemas.RecordOut])
def list_records(limit: int = 20) -> list[dict]:
    limit = max(1, min(limit, 200))
    return db.list_inspections(limit=limit)


@app.get("/records/{iid}")
def get_record(iid: int) -> dict:
    row = db.get_inspection(iid)
    if row is None:
        raise HTTPException(status_code=404, detail=f"没有 id={iid} 的记录")
    return row


@app.get("/image/{iid}", include_in_schema=False)
def get_image(iid: int) -> FileResponse:
    row = db.get_inspection(iid)
    if row is None:
        raise HTTPException(status_code=404, detail=f"没有 id={iid} 的记录")
    p = Path(row["image_path"])
    if not p.is_absolute():
        p = config.BASE_DIR / p
    if not p.exists():
        raise HTTPException(status_code=404, detail="图片文件已不存在")
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/stats", response_model=list[schemas.StatsRow])
def stats() -> list[dict]:
    return db.stats_by_light()


@app.post("/maintenance/backfill-gt", tags=["维护"])
def backfill_gt() -> dict:
    """按文件名把真值补进 gt_verdict。幂等：重复调用不会覆盖已经填过的行。"""
    return db.backfill_gt()


@app.get("/report", tags=["统计"])
def report() -> dict:
    """混淆矩阵 + 漏检率/过杀率 + 节拍分位数 —— 产线要看的就是这几个数。"""
    return db.report_summary()
