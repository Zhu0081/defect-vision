#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据库层：建表 + 查询，全部用原生 SQL。

为什么用原生 SQL 而不是 ORM：
    这 11 个岗位里有 3 个明确写了「掌握 SQL / 能完成表结构设计、查询与读写」。
    ORM 会把这些藏起来，面试一问 JOIN、索引、事务就露馅。
    这里每一条 SQL 都是能直接抄进面试答对的。

表结构（两张，一对多）：
    inspection   一次检测（一张图）一条
    defect_item  该次检测里的每一个缺陷框
"""

from __future__ import annotations

from pathlib import PurePath
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import DATABASE_URL

_engine: Engine | None = None

# SQLite 和 PostgreSQL 的自增/时间类型写法不同，各留一份。
# 改数据类型只改这里，业务代码不动。
DDL_SQLITE = [
    """
    CREATE TABLE IF NOT EXISTS inspection (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id    VARCHAR(64)  NOT NULL,
        filename    VARCHAR(255) NOT NULL,
        image_path  VARCHAR(512) NOT NULL,
        part_id     VARCHAR(64),
        light       VARCHAR(16),
        gt_verdict  VARCHAR(16),
        model_name  VARCHAR(64)  NOT NULL,
        is_mock     BOOLEAN      NOT NULL DEFAULT 0,
        verdict     VARCHAR(16)  NOT NULL,
        n_defects   INTEGER      NOT NULL DEFAULT 0,
        max_score   REAL         NOT NULL DEFAULT 0,
        latency_ms  REAL         NOT NULL,
        created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS defect_item (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        inspection_id INTEGER NOT NULL REFERENCES inspection(id) ON DELETE CASCADE,
        label         VARCHAR(32) NOT NULL,
        score         REAL    NOT NULL,
        x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_inspection_batch ON inspection(batch_id)",
    "CREATE INDEX IF NOT EXISTS idx_inspection_light ON inspection(light)",
    "CREATE INDEX IF NOT EXISTS idx_item_inspection ON defect_item(inspection_id)",
]

DDL_PG = [
    """
    CREATE TABLE IF NOT EXISTS inspection (
        id          SERIAL PRIMARY KEY,
        batch_id    VARCHAR(64)  NOT NULL,
        filename    VARCHAR(255) NOT NULL,
        image_path  VARCHAR(512) NOT NULL,
        part_id     VARCHAR(64),
        light       VARCHAR(16),
        gt_verdict  VARCHAR(16),
        model_name  VARCHAR(64)  NOT NULL,
        is_mock     BOOLEAN      NOT NULL DEFAULT FALSE,
        verdict     VARCHAR(16)  NOT NULL,
        n_defects   INTEGER      NOT NULL DEFAULT 0,
        max_score   REAL         NOT NULL DEFAULT 0,
        latency_ms  REAL         NOT NULL,
        created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS defect_item (
        id            SERIAL PRIMARY KEY,
        inspection_id INTEGER NOT NULL REFERENCES inspection(id) ON DELETE CASCADE,
        label         VARCHAR(32) NOT NULL,
        score         REAL    NOT NULL,
        x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_inspection_batch ON inspection(batch_id)",
    "CREATE INDEX IF NOT EXISTS idx_inspection_light ON inspection(light)",
    "CREATE INDEX IF NOT EXISTS idx_item_inspection ON defect_item(inspection_id)",
]


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        # pool_pre_ping：连接被数据库踢掉后自动重连，熬夜调试时能少一半玄学报错
        _engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
    return _engine


# 加列清单：(表名, 列名, 建列语句)
# 为什么需要它：表已经存在时 CREATE TABLE IF NOT EXISTS 什么也不做，
# 新字段必须单独 ALTER。生产上这一步叫 schema migration。
MIGRATIONS: list[tuple[str, str, str]] = [
    ("inspection", "gt_verdict", "gt_verdict VARCHAR(16)"),
    # AI 报告的三个字段：正文、来源（llm/template）、生成时间。
    # 分开存而不是只存正文，是因为「这段字是谁写的」必须能被验证 ——
    # 面试官问「这句是模型写的还是你写的」，你得能查。
    ("inspection", "ai_report", "ai_report TEXT"),
    ("inspection", "ai_report_source", "ai_report_source VARCHAR(16)"),
    ("inspection", "ai_report_at", "ai_report_at VARCHAR(32)"),
]


def _has_column(conn, table: str, column: str) -> bool:
    if conn.dialect.name == "postgresql":
        row = conn.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = :t AND column_name = :c"
            ),
            {"t": table, "c": column},
        ).first()
        return row is not None
    rows = conn.execute(text(f"PRAGMA table_info({table})")).all()
    return any(r[1] == column for r in rows)


def init_db() -> str:
    eng = get_engine()
    is_pg = eng.dialect.name == "postgresql"
    stmts = DDL_PG if is_pg else DDL_SQLITE
    with eng.begin() as conn:
        for s in stmts:
            conn.execute(text(s))
        for table, column, ddl in MIGRATIONS:
            if not _has_column(conn, table, column):
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))
                print(f"[migrate] {table} 增加列 {column}")
    return eng.dialect.name


def save_inspection(
    *,
    batch_id: str,
    filename: str,
    image_path: str,
    part_id: str | None,
    light: str | None,
    model_name: str,
    is_mock: bool,
    verdict: str,
    n_defects: int,
    max_score: float,
    latency_ms: float,
    items: list[dict[str, Any]],
) -> int:
    """一次事务写入「检测记录 + 它的所有缺陷框」，返回 inspection.id。"""
    eng = get_engine()
    pg = eng.dialect.name == "postgresql"
    insert_sql = """
        INSERT INTO inspection
            (batch_id, filename, image_path, part_id, light, model_name,
             is_mock, verdict, n_defects, max_score, latency_ms)
        VALUES
            (:batch_id, :filename, :image_path, :part_id, :light, :model_name,
             :is_mock, :verdict, :n_defects, :max_score, :latency_ms)
    """ + (" RETURNING id" if pg else "")

    with eng.begin() as conn:
        res = conn.execute(
            text(insert_sql),
            {
                "batch_id": batch_id,
                "filename": filename,
                "image_path": image_path,
                "part_id": part_id,
                "light": light,
                "model_name": model_name,
                "is_mock": is_mock,
                "verdict": verdict,
                "n_defects": n_defects,
                "max_score": max_score,
                "latency_ms": latency_ms,
            },
        )
        iid = res.scalar_one() if pg else res.lastrowid

        if items:
            conn.execute(
                text(
                    """
                    INSERT INTO defect_item (inspection_id, label, score, x1, y1, x2, y2)
                    VALUES (:iid, :label, :score, :x1, :y1, :x2, :y2)
                    """
                ),
                [
                    {
                        "iid": iid,
                        "label": it["label"],
                        "score": it["score"],
                        "x1": it["x1"],
                        "y1": it["y1"],
                        "x2": it["x2"],
                        "y2": it["y2"],
                    }
                    for it in items
                ],
            )
    return int(iid)


def list_inspections(limit: int = 20) -> list[dict]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, batch_id, filename, part_id, light, verdict,
                       n_defects, max_score, latency_ms, created_at
                FROM inspection
                ORDER BY id DESC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        ).mappings().all()
    return [dict(r) for r in rows]


def get_inspection(iid: int) -> dict | None:
    """JOIN 查询：一次检测 + 它的缺陷明细。"""
    with get_engine().connect() as conn:
        head = conn.execute(
            text("SELECT * FROM inspection WHERE id = :iid"), {"iid": iid}
        ).mappings().first()
        if head is None:
            return None
        items = conn.execute(
            text(
                """
                SELECT label, score, x1, y1, x2, y2
                FROM defect_item
                WHERE inspection_id = :iid
                ORDER BY score DESC
                """
            ),
            {"iid": iid},
        ).mappings().all()
    out = dict(head)
    out["items"] = [dict(r) for r in items]
    return out


def save_ai_report(iid: int, report_text: str, source: str) -> None:
    """把生成的报告写回记录。

    为什么不是 INSERT 一张新表：一份检测对应一份报告，1:1 关系。
    另开表会让 get_inspection 多一次 JOIN，而这里省下的复杂度不值那个钱。

    ⚠ 参数名不能叫 text —— db.py 顶部 `from sqlalchemy import text` 导入了同名函数，
    参数会把它遮蔽掉，于是 `text("UPDATE ...")` 变成「调用一个字符串」→
    TypeError: 'str' object is not callable。
    """
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                UPDATE inspection
                   SET ai_report = :t,
                       ai_report_source = :s,
                       ai_report_at = :at
                 WHERE id = :iid
                """
            ),
            {
                "t": report_text,
                "s": source,
                "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "iid": iid,
            },
        )


def stats_by_light() -> list[dict]:
    """按光源聚合 —— 这就是那张「光源 × 模型」对比表的数据来源。"""
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT light,
                       COUNT(*)                                   AS n,
                       SUM(CASE WHEN verdict = 'ng' THEN 1 ELSE 0 END) AS n_ng,
                       ROUND(AVG(n_defects), 2)                   AS avg_defects,
                       ROUND(AVG(latency_ms), 1)                  AS avg_latency_ms
                FROM inspection
                GROUP BY light
                ORDER BY n DESC
                """
            )
        ).mappings().all()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# 真值（ground truth）：算漏检率 / 过杀率必须有它
#
# 好消息是采集规范已经把答案写在文件名里了：
#     pcb_p001_good_001.jpg      -> ok
#     pcb_p001_scratch_003.jpg   -> ng
# 所以不用手工标注一遍，解析文件名就能回填。
# --------------------------------------------------------------------------

DEFECT_TOKENS = ("scratch", "stain", "missing", "deform")


def gt_from_filename(filename: str) -> str | None:
    """从文件名猜真值。认不出来就返回 None（统计时会被忽略，不会污染指标）。"""
    stem = PurePath(filename or "").stem.lower()
    tokens = set(stem.split("_"))
    if "good" in tokens:
        return "ok"
    if tokens & set(DEFECT_TOKENS):
        return "ng"
    return None


def backfill_gt() -> dict[str, int]:
    """给 gt_verdict 还是空的行按文件名补上。已经填过的绝不覆盖。"""
    eng = get_engine()
    with eng.begin() as conn:
        rows = conn.execute(
            text("SELECT id, filename FROM inspection WHERE gt_verdict IS NULL")
        ).all()
        filled = 0
        unknown = 0
        for iid, fname in rows:
            gt = gt_from_filename(fname)
            if gt is None:
                unknown += 1
                continue
            conn.execute(
                text("UPDATE inspection SET gt_verdict = :g WHERE id = :i"),
                {"g": gt, "i": iid},
            )
            filled += 1
    return {"scanned": len(rows), "filled": filled, "unknown": unknown}


def _percentile(values: list[float], p: float) -> float:
    """线性插值分位数。PG 有 percentile_cont，SQLite 没有，所以在 Python 里算。"""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)


def report_summary() -> dict[str, Any]:
    """混淆矩阵 + 漏检率/过杀率 + 节拍分位数。

    约定：gt_verdict = 'ng' 是正样本（真有缺陷）。
        TP = 真有缺陷 && 判 ng
        FN = 真有缺陷 && 判 ok   -> 漏检（最贵，流到客户手里）
        FP = 本来好的 && 判 ng   -> 过杀（产线自己承担成本）
        TN = 本来好的 && 判 ok
    """
    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT verdict, gt_verdict, COUNT(*) AS n
                FROM inspection
                WHERE gt_verdict IS NOT NULL
                GROUP BY verdict, gt_verdict
                """
            )
        ).all()
        lat_rows = conn.execute(text("SELECT latency_ms FROM inspection")).all()
        total = conn.execute(text("SELECT COUNT(*) FROM inspection")).scalar_one()

    tp = fn = fp = tn = 0
    for verdict, gt, n in rows:
        n = int(n)
        if gt == "ng":
            if verdict == "ng":
                tp += n
            else:
                fn += n
        else:
            if verdict == "ng":
                fp += n
            else:
                tn += n

    labeled = tp + fn + fp + tn
    pos = tp + fn
    neg = fp + tn
    lat = [float(r[0]) for r in lat_rows]

    return {
        "total_inspections": int(total),
        "labeled": labeled,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "miss_rate_pct": round(fn / pos * 100, 2) if pos else None,
        "overkill_rate_pct": round(fp / neg * 100, 2) if neg else None,
        "latency_ms": {
            "p50": _percentile(lat, 0.50),
            "p95": _percentile(lat, 0.95),
            "max": round(max(lat), 2) if lat else 0.0,
        },
    }
