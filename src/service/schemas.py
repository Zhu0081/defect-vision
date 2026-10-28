#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""响应模型（Pydantic）。

它同时干三件事：
    1. 校验输入输出（FastAPI 自动拿它做请求/响应校验）
    2. 自动生成 OpenAPI 文档（打开 /docs 就能看到，面试演示非常好用）
    3. 当接口契约 —— 前端、测试、别的人都照这个写
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DefectItem(BaseModel):
    label: str = Field(..., description="缺陷类别：scratch / stain / missing / deform")
    score: float = Field(..., ge=0, le=1, description="置信度 0~1")
    box: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="归一化坐标 [x1, y1, x2, y2]，取值 0~1",
    )


class DetectResponse(BaseModel):
    inspection_id: int
    batch_id: str
    verdict: str = Field(..., description="ok=良品 / ng=有缺陷")
    n_defects: int
    max_score: float
    latency_ms: float = Field(..., description="端到端耗时（含解码与推理，不含网络）")
    model_name: str
    is_mock: bool = Field(..., description="true 表示模型还没接上，结果是假数据")
    items: list[DefectItem]
    image_url: str


class HealthResponse(BaseModel):
    status: str
    db: str
    model_name: str
    is_mock: bool
    upload_dir: str


class RecordOut(BaseModel):
    id: int
    batch_id: str
    filename: str
    part_id: str | None = None
    light: str | None = None
    verdict: str
    n_defects: int
    max_score: float
    latency_ms: float
    created_at: str | None = None


class StatsRow(BaseModel):
    light: str | None = None
    n: int
    n_ng: int
    avg_defects: float | None = None
    avg_latency_ms: float | None = None
