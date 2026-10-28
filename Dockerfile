# 工业缺陷检测服务 · 镜像
#
# 构建：docker build -t defect-vision:0.1 .
# 运行：docker compose up
#
# 三个值得对着 JD 讲的点：
#   1. slim 基础镜像 + 只装运行必需的系统库，最终镜像 ~700MB 而不是 2GB+
#   2. 依赖先 COPY 再 COPY 代码 —— 改代码不会让 pip 层失效，重建只要几秒
#   3. HEALTHCHECK 用 python 自己探活，不额外装 curl

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# onnxruntime 需要 libgomp1（OpenMP 运行时），opencv-headless 需要 libglib2.0-0
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# —— 先装依赖，再拷代码。这一层能被缓存，改代码不用重新 pip install ——
COPY requirements-docker.txt .
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements-docker.txt

# —— 代码 ——
COPY src ./src
COPY models/best.onnx ./models/best.onnx

RUN mkdir -p data/uploads

EXPOSE 8000

# 用 python 探活，镜像里不用装 curl
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# --host 0.0.0.0 是必须的：容器内 127.0.0.1 只有容器自己能看到
CMD ["python", "-m", "uvicorn", "src.service.main:app", "--host", "0.0.0.0", "--port", "8000"]
