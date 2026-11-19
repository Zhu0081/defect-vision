#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""响应模型（Pydantic）。

它同时干三件事：
    1. 校验输入输出（FastAPI 自动拿它做请求/响应校验）
    2. 自动生成 OpenAPI 文档（打开 /docs 就能看到，面试演示非常好用）
    3. 当接口契约 —— 前端、测试、别的人都照这个写
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class DefectItem(BaseModel):
    label: str = Field(..., description="缺陷类别：scratch / stain / missing / deform")
    score: float = Field(..., ge=0, le=1, description="置信度 0~1")
    box: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="归一化坐标 [x1, y1, x2, y2]，取值 0~1",
    )
    polygon: list[list[float]] = Field(
        default_factory=list,
        description="归一化的缺陷轮廓 [[x, y], ...]；检测框模型返回空数组",
    )
    polygons: list[list[list[float]]] = Field(
        default_factory=list,
        description="同一块框里的全部轮廓（断续划痕拆成多段时用）；polygon 恒等于第一段",
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


class RecordItem(BaseModel):
    """单条缺陷明细（历史记录详情里的每一行）。

    之前 RecordOut 里没有这个字段，于是 /records/{id} 把 defect_item 查出来又被
    response_model 过滤掉了 —— 前端拿不到框，历史记录只能显示原图。
    """

    label: str
    score: float
    box: list[float] = Field(..., min_length=4, max_length=4,
                             description="归一化坐标 [x1, y1, x2, y2]")
    # 老数据没有轮廓，统一返回 []（库里是 NULL）
    polygon: list[list[float]] = Field(default_factory=list,
                                       description="归一化多边形顶点，没有就是 []")
    polygons: list[list[list[float]]] = Field(
        default_factory=list,
        description="同一块框里的全部轮廓；没有就是 []。polygon 恒等于第一段")


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

    # 这两列 get_inspection() 用 SELECT * 本来就查出来了，只是以前没在 response_model 里声明，
    # 被 FastAPI 静默过滤掉了 —— 结果网页回放历史时不知道自己看的是哪个模型跑出来的。
    # 注意 list_inspections() 没查这两列，所以默认值必须是可空/可省，否则列表接口会 500。
    model_name: str | None = None
    is_mock: bool | None = None

    # 可追溯三件套。老记录是 NULL —— 前端显示「未知」，绝不按日期猜。
    # 为什么要有：111 条历史里 model_name 只有 1 个值，但那 111 条横跨 20 天、
    # 换过四版权重。没有 sha，就说不清「这条记录到底是谁跑出来的」。
    model_sha: str | None = None
    imgsz: int | None = None
    conf_thr: float | None = None

    @field_validator("created_at")
    @classmethod
    def _created_at_is_utc(cls, v: datetime | None) -> datetime | None:
        """把数据库里的「裸 UTC 墙钟时间」标上 UTC 时区。

        inspections.created_at 是 `TIMESTAMP`（不带时区）列，默认值 `CURRENT_TIMESTAMP`
        在容器里按 UTC 落库，所以存进去的是 UTC 的墙钟时间，但**没有时区标记**。
        原样返回的话 FastAPI 会序列化成 "2026-11-17T05:59:02"，
        浏览器和 C# 都会按「本地时间」去解析 → 显示比实际慢 8 小时。

        这里补上 UTC，序列化结果变成 "2026-11-17T05:59:02+00:00"，
        前端才能换算成本地时间显示。已经带时区的值原样返回，不覆盖。
        """
        if v is None:
            return None
        return v if v.tzinfo is not None else v.replace(tzinfo=timezone.utc)
    # 报告字段可空：还没生成过就是 None，前端据此决定「显示生成按钮」还是「显示报告」
    ai_report: str | None = None
    ai_report_source: str | None = None
    ai_report_at: str | None = None
    # 缺陷明细：历史记录要能回放出当时的框和轮廓，不能每次重新推理一遍
    items: list[RecordItem] = Field(default_factory=list)


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
