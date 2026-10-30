# -*- coding: utf-8 -*-
"""pytest 公共配置。

三件事在**导入 app 之前**做完，顺序不能反：
    1. 把数据目录/上传目录/数据库都指到临时目录 —— 测试绝不能写脏你的真数据
    2. 把模型路径指到一个不存在的文件 —— 没有模型时服务会自动退回 MockDetector，
       测试因此不依赖 models/best.onnx（别人 clone 下来也能直接跑）
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
os.environ["DV_MAX_UPLOAD_MB"] = "1"

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
