#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 检测报告生成（LLM + 规则模板，双通道）。

为什么要这一层：
    产线班长不懂 mAP 和置信度，他要的是一句人话 ——「22 号件的划痕要复检」。
    把结构化检测结果翻译成人话，是大模型最擅长、也最便宜的用法。

三条设计约束（面试可以直接讲）：
    1. **不喂图片，只喂结构化数据**
       视觉大模型又贵又慢还不稳定；而检测结果本来就已经是结构化数据了，
       模型只需要「把数字翻译成句子」，不需要「看图」。这一步省掉 90% 的成本。
    2. **必须能降级**
       没配 key、超时、限流、返回垃圾 —— 任何一种都不能让接口 500。
       降级走规则模板，响应里标 source=template，演示永远不会白屏。
    3. **只准引用给定数字**
       prompt 里明确禁止推测，temperature 设 0.2。质检报告编数字是要出事故的。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from . import config

log = logging.getLogger("defect.llm")

# 缺陷类别的中文名（和标注规范、前端保持同一套编号）
LABEL_CN = {
    "scratch": "划痕",
    "stain": "脏污",
    "missing": "缺件",
    "deform": "变形破损",
}

SYSTEM_PROMPT = """你是一名工业视觉质检工程师的助手，读者是产线班长。

规则（违反任何一条都算失败）：
1. 只使用用户给出的数字。禁止推测、禁止编造、禁止描述你看不到的图片内容。
2. 语言直白，不要出现 mAP、IoU 这类指标名，也不要解释算法原理。
3. 输出必须且只能包含下面四个小标题，每个标题下不超过两句话：
### 判定结论
### 缺陷情况
### 风险提示
### 建议动作
4. 全文不超过 200 字，不要用列表符号，不要加开场白。"""


def _cn(label: str) -> str:
    return LABEL_CN.get(label, label)


@dataclass
class ReportResult:
    """一次报告生成的结果。

    source 取值：
        "llm"      —— 真的调了大模型
        "template" —— 降级到规则模板（没 key / 调用失败 / 返回为空）
    为什么要暴露这个字段：前端要显示来源，面试官要能一眼看出「哪部分是模型写的」。
    把降级悄悄藏起来，就是在骗人。
    """

    text: str
    source: str
    model: str | None = None
    latency_ms: float = 0.0


# --------------------------------------------------------------------------
# 事实整理：把数据库的一行记录，压成模型能直接吃的一小段文本
# --------------------------------------------------------------------------
def build_facts(record: dict[str, Any], summary: dict[str, Any] | None = None) -> dict[str, Any]:
    items = record.get("items") or []
    defects = [
        {"label": _cn(it.get("label", "")), "score": round(float(it.get("score") or 0), 3)}
        for it in items
    ]
    defects.sort(key=lambda d: d["score"], reverse=True)

    facts: dict[str, Any] = {
        "part_id": record.get("part_id") or "未标注",
        "light": record.get("light") or "未标注",
        "verdict": record.get("verdict"),
        "n_defects": int(record.get("n_defects") or 0),
        "max_score": round(float(record.get("max_score") or 0), 3),
        "latency_ms": round(float(record.get("latency_ms") or 0), 1),
        "model_name": record.get("model_name") or "unknown",
        "is_mock": bool(record.get("is_mock")),
        "defects": defects,
        "global": None,
    }

    if summary:
        conf = summary.get("confusion") or {}
        lat = summary.get("latency_ms") or {}
        facts["global"] = {
            "total": summary.get("total_inspections"),
            "fp": conf.get("fp"),
            "fn": conf.get("fn"),
            "miss_pct": summary.get("miss_rate_pct"),
            "overkill_pct": summary.get("overkill_rate_pct"),
            "p50": lat.get("p50"),
            "p95": lat.get("p95"),
        }
    return facts


def render_facts_text(facts: dict[str, Any]) -> str:
    """把事实渲染成「给模型看」的纯文本。

    为什么不用 JSON：JSON 的括号和引号会占掉不少 token，而且模型偶尔会去
    「续写」JSON 结构。一行一个字段的键值对更稳，也更省。
    """
    lines = [
        f"工件号：{facts['part_id']}",
        f"光源：{facts['light']}",
        f"检测模型：{facts['model_name']}" + ("（注意：当前是 Mock 假模型，数据不代表真实检测）" if facts["is_mock"] else ""),
        f"判定：{'NG 不良' if facts['verdict'] == 'ng' else 'OK 良品'}",
        f"缺陷数：{facts['n_defects']}",
        f"最高置信度：{facts['max_score']:.3f}",
        f"单张耗时：{facts['latency_ms']:.1f} ms",
    ]

    if facts["defects"]:
        detail = "、".join(f"{d['label']} {d['score']:.3f}" for d in facts["defects"][:10])
        lines.append(f"缺陷明细：{detail}")
    else:
        lines.append("缺陷明细：无")

    g = facts.get("global")
    if g:
        parts = [f"累计检测 {g['total']} 次"]
        if g.get("miss_pct") is not None:
            parts.append(f"漏检率 {g['miss_pct']}%")
        if g.get("overkill_pct") is not None:
            parts.append(f"过杀率 {g['overkill_pct']}%")
        if g.get("p50") is not None:
            parts.append(f"节拍 p50 {g['p50']} ms / p95 {g['p95']} ms")
        lines.append("整体统计：" + "，".join(parts))

    return "\n".join(lines)


# --------------------------------------------------------------------------
# 降级通道：规则模板
# --------------------------------------------------------------------------
def _risk_level(max_score: float, n: int) -> tuple[str, str]:
    """按最高置信度分档给出风险与建议。

    这三档不是拍脑袋定的：置信度 0.8 以上在同一名工件上极少误报，
    0.4~0.8 是最需要人工复检的灰区，0.4 以下多半是噪声或打光问题。
    """
    if n == 0:
        return (
            "本次未检出缺陷。若该批次历史漏检率偏高，仍建议按抽样比例抽检。",
            "按正常流程放行，并记录本次判定以便后续回溯。",
        )
    if max_score >= 0.8:
        return (
            f"最高置信度 {max_score:.2f}，属于强特征缺陷，误报概率低。",
            "直接判为不良，转人工复检确认后隔离，不要流入下一道工序。",
        )
    if max_score >= 0.4:
        return (
            f"最高置信度 {max_score:.2f}，落在灰区 —— 可能是真实缺陷，也可能是反光或污点。",
            "保留原图，安排人工复检；同一工件连拍 2 张再判一次可降低误判。",
        )
    return (
        f"最高置信度仅 {max_score:.2f}，接近判定阈值，可信度不足。",
        "先检查打光与对焦（纸面亮度建议 110~150），必要时换侧光重拍，不要凭这一张判废。",
    )


def template_report(facts: dict[str, Any]) -> str:
    """规则模板报告：和大模型输出同结构，保证前端渲染一致。"""
    n = facts["n_defects"]
    verdict_cn = "NG 不良" if facts["verdict"] == "ng" else "OK 良品"
    risk, action = _risk_level(float(facts["max_score"]), n)

    if n:
        counts: dict[str, int] = {}
        for d in facts["defects"]:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        detail = "，".join(f"{k} {v} 处" for k, v in counts.items())
        top = facts["defects"][0]
        body = f"共检出 {n} 处缺陷：{detail}。最高为{top['label']}，置信度 {top['score']:.3f}。"
    else:
        body = "未检出缺陷框，判定为良品。"

    return (
        f"### 判定结论\n"
        f"{facts['part_id']} 号件（{facts['light']} 光源）判定为 **{verdict_cn}**，"
        f"单张检测耗时 {facts['latency_ms']:.0f} ms。\n\n"
        f"### 缺陷情况\n{body}\n\n"
        f"### 风险提示\n{risk}\n\n"
        f"### 建议动作\n{action}"
    )


# --------------------------------------------------------------------------
# 主通道：调大模型
# --------------------------------------------------------------------------
async def _call_llm(facts: dict[str, Any]) -> str:
    """调 OpenAI 兼容的 /chat/completions，返回纯文本。异常一律往上抛，由上层降级。"""
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_facts_text(facts)},
        ],
        # 质检报告要的是稳定复现，不是文采：温度压低到 0.2
        "temperature": 0.2,
        "max_tokens": config.LLM_MAX_TOKENS,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    url = config.LLM_BASE_URL.rstrip("/") + "/chat/completions"

    async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_S) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"模型没有返回 choices：{str(data)[:200]}")
    text = (choices[0].get("message") or {}).get("content") or ""
    text = text.strip()
    if not text:
        raise ValueError("模型返回了空内容")
    return text


async def generate_report(
    record: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> ReportResult:
    """生成一份报告。**永远不抛异常** —— 最差的情况也是返回模板报告。"""
    facts = build_facts(record, summary)
    t0 = time.perf_counter()

    if not config.LLM_API_KEY:
        log.info("[llm] 未配置 DV_LLM_API_KEY，使用规则模板")
        return ReportResult(
            text=template_report(facts),
            source="template",
            model=None,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    try:
        text = await _call_llm(facts)
        log.info("[llm] 模型=%s 字数=%d", config.LLM_MODEL, len(text))
        return ReportResult(
            text=text,
            source="llm",
            model=config.LLM_MODEL,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
    except Exception as e:  # noqa: BLE001 —— 就是要兜住所有异常
        # 这里不能 raise：报告是锦上添花，不能因为它把整个检测流程搞挂
        log.warning("[llm] 调用失败，降级到模板：%s: %s", type(e).__name__, e)
        return ReportResult(
            text=template_report(facts),
            source="template",
            model=None,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
