# -*- coding: utf-8 -*-
"""pytest 公共配置。

三件事在**导入 app 之前**做完，顺序不能反：
    1. 把数据目录/上传目录/数据库都指到临时目录 —— 测试绝不能写脏你的真数据
    2. 把模型路径指到一个不存在的文件，**并显式允许 Mock**——
       生产模式下模型加载失败会直接抛错退出（这是刻意的，见 detector._fallback），
       测试要的是「没有模型也能跑」，所以必须设 DV_ALLOW_MOCK=1。
       不设这一行，19 个用例会在 lifespan 里全部报 RuntimeError。
    3. 把上传上限调到 1 MB —— 测 413 时不用真造一个 16 MB 的文件

注意：config.py 在 import 时就读取这些环境变量，
所以必须先设 os.environ，再 import src.service.main。
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="dv-test-"))
(_TMP / "data" / "uploads").mkdir(parents=True, exist_ok=True)

os.environ["DV_DATA_DIR"] = str(_TMP / "data")
os.environ["DV_UPLOAD_DIR"] = str(_TMP / "data" / "uploads")
os.environ["DV_DATABASE_URL"] = f"sqlite:///{(_TMP / 'data' / 'test.db').as_posix()}"
os.environ["DV_MODEL_PATH"] = str(_TMP / "models" / "does-not-exist.onnx")
# 测试环境必须允许 Mock，否则 build_detector 会按生产模式直接抛错（见 conftest 顶部说明）
os.environ["DV_ALLOW_MOCK"] = "1"
os.environ["DV_MAX_UPLOAD_MB"] = "1"
# 强制清空大模型 key：测试绝不能真的去调外部接口（慢、要钱、还不稳定）。
# 清空后 AI 报告会自动走规则模板，正好测「降级通道」。
os.environ["DV_LLM_API_KEY"] = ""

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.service.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    """整个测试会话共用一个 client。

    必须用 with：Starlette 的 TestClient 只有在进入上下文时才执行 lifespan，
    而建表、加载模型都在 lifespan 里 —— 不用 with 的话 /detect 会报 500。
    """
    with TestClient(app) as c:
        yield c
