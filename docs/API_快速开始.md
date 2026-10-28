# 缺陷检测服务 · 快速开始

> 状态：**已跑通并通过 13 项冒烟测试**（2026-10-26）。
> 模型仍然是 `mock-v0`（假数据），但**接口、数据库、错误处理全是真的**。
> 等 W3 训出 `models/best.onnx`，只改一个环境变量就切成真模型，上层代码不动。

---

## 1. 启动（30 秒）

```powershell
cd F:\defect-vision
python -m uvicorn src.service.main:app --reload --port 8000
```

看到这行就成功了：

```
[startup] 数据库=sqlite  模型=mock-v0  mock=True
INFO:     Uvicorn running on http://127.0.0.1:8000
```

然后浏览器打开：

| 地址 | 用途 |
|---|---|
| http://127.0.0.1:8000/ | **上传页面**，选图点按钮就能测 |
| http://127.0.0.1:8000/docs | **自动生成的接口文档**，能直接点试（面试演示用这个） |

`--reload` 表示改代码自动重启，开发时开着；演示时去掉。

---

## 2. 五个接口

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/health` | 健康检查：数据库类型、模型名、是否 mock |
| POST | `/detect` | 上传一张图 → 检测 → 落库 → 返回结果 |
| GET | `/records?limit=20` | 最近的检测记录列表 |
| GET | `/records/{id}` | 单条记录 + 缺陷明细（**JOIN 查询**） |
| GET | `/stats` | 按光源聚合的统计（**GROUP BY**），光源对比表的数据源 |

### 2.1 健康检查

```powershell
curl.exe http://127.0.0.1:8000/health
```

```json
{"status":"ok","db":"sqlite","model_name":"mock-v0","is_mock":true,"upload_dir":"F:\\defect-vision\\data\\uploads"}
```

### 2.2 检测一张图

```powershell
curl.exe -X POST http://127.0.0.1:8000/detect `
  -F "file=@F:\defect-vision\data\captured\a.jpg" `
  -F "light=side" `
  -F "part_id=001" `
  -F "conf=0.25"
```

返回：

```json
{
  "inspection_id": 5,
  "batch_id": "B20261026163012",
  "verdict": "ng",
  "n_defects": 2,
  "max_score": 0.871,
  "latency_ms": 9.44,
  "model_name": "mock-v0",
  "is_mock": true,
  "items": [{"label": "scratch", "score": 0.871, "box": [0.31, 0.22, 0.48, 0.39]}],
  "image_url": "/image/5"
}
```

⚠ 路径**不要带中文和空格**，`curl.exe -F` 对非 ASCII 路径容易出错。图先拷到 `F:\tmp\a.jpg`。

### 2.3 查记录 / 统计

```powershell
curl.exe "http://127.0.0.1:8000/records?limit=5"
curl.exe http://127.0.0.1:8000/records/5
curl.exe http://127.0.0.1:8000/stats
```

---

## 3. 数据库

默认用 **SQLite**，文件在 `F:\defect-vision\data\defect.db`，**不用装任何数据库**。

### 两张表（一对多）

```
inspection     一次检测（一张图）一条
  id, batch_id, filename, image_path, part_id, light,
  model_name, is_mock, verdict, n_defects, max_score, latency_ms, created_at

defect_item    该次检测里的每一个缺陷框
  id, inspection_id(外键), label, score, x1, y1, x2, y2
```

为什么拆两张：一张图可能有多个缺陷，塞进一列存 JSON 之后就没法「按缺陷类型统计」了。
拆开加外键 + 索引，才能做 `GROUP BY label` 这种查询——这正是 JD 里「表结构设计」考的东西。

### 想直接看库里的数据

```powershell
python -c "import sqlite3;c=sqlite3.connect(r'F:\defect-vision\data\defect.db');print(c.execute('select id,light,verdict,n_defects from inspection order by id desc limit 5').fetchall())"
```

或者装免费的 **DB Browser for SQLite**，图形界面点着看更直观。

### 换成 PostgreSQL（等要写进简历时再做）

```powershell
# 1. 装 PostgreSQL 后建库建用户
# 2. 设环境变量，注意驱动是 psycopg
$env:DV_DATABASE_URL = "postgresql+psycopg://dv:dv@localhost:5432/defect"
python -m pip install "psycopg[binary]"
# 3. 重启服务，自动建表（DDL_PG 那份）
```

代码一行不用改 —— 这是把配置外置的好处。

---

## 4. 冒烟测试

```powershell
$env:PYTHONIOENCODING="utf-8"
python src\service\smoke_test.py
```

输出 `全部通过 ✅` 说明 13 项全过：健康检查、检测、字段完整性、可复现性、明细查询、列表、统计、原图回取、非图片 400、不存在 id 返回 404。

**以后每次改代码都跑一遍。** 面试被问「你怎么保证没改坏」，这就是答案。

---

## 5. 排错

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | 不在 `F:\defect-vision` 目录下启动。先 `cd F:\defect-vision` |
| `[Errno 10048] 端口已被占用` | 换端口 `--port 8001`，或 `netstat -ano \| findstr :8000` 找进程 |
| 413 文件太大 | 默认 16MB，改环境变量 `DV_MAX_UPLOAD_MB` |
| 400 不是可解码的图像 | 传了非 jpg/png，或者文件在路上坏了 |
| 8000 打不开但服务在跑 | 确认用 `127.0.0.1` 不是 `0.0.0.0`；防火墙弹窗要点允许 |
| 改了代码没生效 | 启服务时加了 `--reload` 吗 |

---

## 6. 这些代码对应哪条 JD

| JD 原文 | 这个项目里的落点 | 面试怎么讲 |
|---|---|---|
| 熟练使用 Python，独立编码调试 | 整个 `src/service/` | 讲你怎么定位 413/400 边界 |
| 熟悉 FastAPI 至少一种后端框架 | `main.py` 全部接口 | 讲 Pydantic 校验 + 自动 OpenAPI |
| 掌握 SQL，表结构设计、查询读写 | `db.py` 两张表 + 外键 + 索引 + JOIN + GROUP BY | **把 SQL 背下来**，别只说「用过 ORM」 |
| 独立部署到服务器、Linux、Docker | 本地已跑通；Docker 待做 | 讲环境变量外置、启动时建表 |
| 大模型 API / Prompt 设计 | 下一步接（检测结果自动写报告） | 见 W4 |
| 熟悉 Git、密钥管理 | 待做：`.gitignore` 排除 `.env` 和 `data/` | 讲「为什么数据库文件不进仓库」 |
| AI 应用实践、数据处理 | 检测结果落库 → 统计 → 报表 | 讲数据闭环 |

---

## 7. 下一步（按顺序做，别跳）

1. **今晚**：把 3~5 张真图从 `/` 上传页刷一遍，看返回的 json 结构
2. **明天**：`git init` + 第一次提交（`data/`、`models/`、`__pycache__` 进 `.gitignore`）
3. **W3**：训练出第一个 `best.onnx`，`DV_MODEL_PATH` 指过去，看 `is_mock` 变 `false`
4. **W4**：检测结果 → 调大模型 API 自动生成中文报告（`/report/{id}`）
5. **W5**：写 `Dockerfile`（本机还没装 Docker，装完先 `docker build` 一次，别等最后）
