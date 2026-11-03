#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""响应模型（Pydantic）。

它同时干三件事：
    1. 校验输入输出（FastAPI 自动拿它做请求/响应校验）
    2. 自动生成 OpenAPI 文档（打开 /docs 就能看到，面试演示非常好用）
    3. 当接口契约 —— 前端、测试、别的人都照这个写
"""

from __future__ import annotations

from datetime import datetime

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
    # 必须是 datetime 而不是 str：PostgreSQL 的 TIMESTAMP 列返回的是 datetime 对象，
    # 声明成 str 会让 FastAPI 在响应校验时抛 ResponseValidationError → 500。
    # 而 SQLite 是动态类型，同样的列返回的是字符串，pydantic 也能解析，
    # 所以 datetime 这个声明两种数据库都吃得下。
    created_at: datetime | None = None
    # 报告字段可空：还没生成过就是 None，前端据此决定「显示生成按钮」还是「显示报告」
    ai_report: str | None = None
    ai_report_source: str | None = None
    ai_report_at: str | None = None


class AiReportOut(BaseModel):
    inspection_id: int
    text: str = Field(..., description="Markdown 报告正文，四个固定小标题")
    source: str = Field(..., description="llm=真调了大模型 / template=降级到规则模板")
    model: str | None = Field(None, description="实际使用的模型名，降级时为 null")
    latency_ms: float = Field(..., description="生成耗时（含网络）")
    cached: bool = Field(False, description="true 表示直接返回库里已存的报告，没有重新生成")


class StatsRow(BaseModel):
    light: str | None = None
    n: int
    n_ng: int
    avg_defects: float | None = None
    avg_latency_ms: float | None = None
