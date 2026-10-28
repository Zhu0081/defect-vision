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
