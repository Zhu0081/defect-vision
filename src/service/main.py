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

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import config, db, schemas
from .detector import build_detector

# 结构化日志：时间、级别、来源、消息。
# 为什么不用 print：容器里排障时，print 没有时间戳也没有级别，
# 出问题只能靠猜；logging 还能通过 DV_LOG_LEVEL 一行改成 DEBUG。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
log = logging.getLogger("defect")

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
    log.info(
        "[startup] 数据库=%s 模型=%s mock=%s 上传目录=%s",
        db_name, app.state.detector.name, app.state.detector.is_mock, config.UPLOAD_DIR,
    )
    yield
    log.info("[shutdown] bye")


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
    request: Request,
    file: UploadFile = File(..., description="待检测图像（jpg/png）"),
    batch_id: str | None = Form(None, description="批次号，不传自动生成"),
    part_id: str | None = Form(None, description="工件件号"),
    light: str | None = Form(None, description="光源：top/side/back"),
    conf: float = Form(config.DEFAULT_CONF, ge=0.0, le=1.0, description="置信度阈值"),
) -> schemas.DetectResponse:
    t0 = time.perf_counter()

    # 上传大小必须「边读边拦」，不能先 file.read() 再判断。
    # 因为 read() 会把整个文件一次性读进内存：别人传 2GB，进程先吃掉 2GB 才回一个 413。
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    too_big = f"文件超过 {config.MAX_UPLOAD_MB} MB"

    # 第一道闸：看请求头声明的大小（便宜，但客户端可以撒谎）
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail=too_big)

    # 第二道闸：分块读，边读边累加（不信任请求头）
    # 注意：Starlette 的 UploadFile 没有 .stream()，只有 await read(size)，
    # 所以这里手动循环读 64KB 一块，读到空字节为止。
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail=too_big)
        chunks.append(chunk)
    raw = b"".join(chunks)

    if not raw:
        raise HTTPException(status_code=400, detail="空文件")

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="不是可解码的图像（jpg/png？）")

    # 推理是 CPU 密集的，必须丢线程池，否则会卡住整个事件循环
    det = app.state.detector
    # conf 必须传进去：之前这里漏了这个参数，导致页面上的「阈值」输入框
    # 收得到、校验得到，但完全不影响结果（一个纯粹的装饰品）。
    detections = await run_in_threadpool(det.predict, img, conf)

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

    # 落盘路径：能相对项目根就存相对路径（仓库搬家不失效），
    # 上传目录被指到项目外面（比如挂了个外部卷）就存绝对路径。
    # 以前这里直接 relative_to()，一旦 DV_UPLOAD_DIR 在项目外就抛 ValueError → 500。
    try:
        stored_path = str(path.relative_to(config.BASE_DIR)).replace("\\", "/")
    except ValueError:
        stored_path = str(path).replace("\\", "/")

    try:
        iid = await run_in_threadpool(
            db.save_inspection,
            batch_id=bid,
            filename=file.filename or name,
            image_path=stored_path,
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
    except Exception:
        # 图已经写进磁盘、库却没写成功 → 这会留下「孤儿文件」：
        # 磁盘上有一张没人认领的图，越积越多。所以失败时把它删掉。
        path.unlink(missing_ok=True)
        log.exception("[detect] 落库失败，已删除刚写入的图片 %s", path)
        raise HTTPException(status_code=500, detail="保存检测记录失败") from None

    log.info(
        "[detect] id=%s verdict=%s 缺陷=%d 耗时=%.1fms 模型=%s",
        iid, verdict, len(items), latency_ms, det.name,
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


@app.get("/records/{iid}", response_model=schemas.RecordOut)
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
