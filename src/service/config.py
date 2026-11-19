#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置：全部从环境变量读，代码里不写死任何路径或密码。

为什么这么做：
    岗位 JD 里反复出现「部署、环境配置、密钥管理」。把配置外置是最基本的一条，
    面试官问「你怎么上生产」时，你说「改环境变量就行，代码不用动」——这就够了。

用法：
    # 本地 SQLite（默认，零安装）
    uvicorn src.service.main:app --reload

    # 换成 PostgreSQL（装好 PG 之后）
    $env:DV_DATABASE_URL = "postgresql+psycopg://dv:dv@localhost:5432/defect"
"""

from __future__ import annotations

import os
from pathlib import Path

# src/service/config.py -> parents[0]=service, [1]=src, [2]=项目根 F:\defect-vision
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("DV_DATA_DIR", str(BASE_DIR / "data")))
UPLOAD_DIR = Path(os.getenv("DV_UPLOAD_DIR", str(DATA_DIR / "uploads")))
MODEL_PATH = Path(os.getenv("DV_MODEL_PATH", str(BASE_DIR / "models" / "best.onnx")))

# 默认 SQLite：今晚不装数据库也能跑通；换 PG 只改这一个环境变量
DATABASE_URL = os.getenv("DV_DATABASE_URL", f"sqlite:///{(DATA_DIR / 'defect.db').as_posix()}")

MAX_UPLOAD_MB = int(os.getenv("DV_MAX_UPLOAD_MB", "16"))

# 阈值：低于它的框直接丢掉（后续按产线要求再调）
DEFAULT_CONF = float(os.getenv("DV_DEFAULT_CONF", "0.25"))

# ---------- 安全开关 ----------
# DV_ALLOW_MOCK=1 时，模型加载失败才允许退回假模型；不设（默认）就直接报错退出。
# 理由：假模型返回的是编造结果，界面上只多一行小字，产线上没人会注意到
# 「今天所有件都判成 NG」其实是模型没加载上。
ALLOW_MOCK = os.getenv("DV_ALLOW_MOCK", "0").strip() in ("1", "true", "True", "yes")

# DV_API_TOKEN 为空（默认）= 不做认证，本机/局域网演示直接用；
# 设了值以后，所有**会改数据**的接口（上传检测、生成报告、维护脚本）都要求
# 请求头 `X-API-Token` 与它一致，否则 401。查询类接口不拦，方便演示时直接看页面。
API_TOKEN = os.getenv("DV_API_TOKEN", "").strip()

# ---------- 大模型（AI 检测报告）----------
# 走 OpenAI 兼容协议：DeepSeek / 通义 / 智谱 / 月之暗面 / 本地 vLLM / Ollama 都是这个格式。
# 换服务商只改 base_url 和 model 两个环境变量，代码一行不动 —— 这就是「配置外置」的意义。
#
# 没配 key 时服务照样能用：会退回到「规则模板」生成报告，并在响应里标明 source=template。
# 宁可降级，也不要让一个按钮点下去白屏 —— 演示现场不能靠运气。
LLM_BASE_URL = os.getenv("DV_LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("DV_LLM_API_KEY", "")
LLM_MODEL = os.getenv("DV_LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT_S = float(os.getenv("DV_LLM_TIMEOUT_S", "20"))
LLM_MAX_TOKENS = int(os.getenv("DV_LLM_MAX_TOKENS", "700"))
