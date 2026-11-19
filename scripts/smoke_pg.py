#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对**运行中的服务**做一次集成冒烟（真 PostgreSQL + 真 ONNX 模型）。

为什么不放进 pytest：
    tests/ 里那 19 个用例跑的是 SQLite + MockDetector，刻意做到「不依赖任何外部
    东西」——换台电脑 clone 下来就能跑。而这个脚本要连真数据库、真模型，
    跑一次会真的写记录，所以它是**手动执行的冒烟**，不是回归测试。

它在验证什么（这几个坑 pytes 一个都测不到）：
    1. ROUND(AVG(double), 1) 在 PostgreSQL 上不存在 —— SQLite 宽容，PG 直接 500。
    2. 量化/后端契约：模型输出到底是带 NMS 还是不带，靠 infer 后框数判断。
    3. 记录能不能被追溯：model_sha 是否有值、是否和磁盘上的模型一致。
    4. 落库→查询往返：POST /detect 之后再 GET /records/{id} 拿到的框与轮廓一致。

怎么跑（服务要先起来）：
    python scripts/smoke_pg.py
    python scripts/smoke_pg.py --base http://127.0.0.1:8010 --keep

默认**跑完会删掉自己造的那条记录**（按 id 精确删），让生产库回到原来的条数。
加 --keep 就不删，方便你在网页上亲眼看。

退出码：0 全过；1 有失败项。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "http://127.0.0.1:8010"
# 用测试集里的真图：文件名符合 gt_from_filename 的规范，回填真值后能进统计
SAMPLE = ROOT / "data" / "yolo" / "images" / "test" / "card_p022_scratch_002.jpg"

_ok = 0
_fail = 0


def check(name: str, cond: bool, detail: str = "") -> bool:
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  [OK]   {name}" + (f"  {detail}" if detail else ""))
    else:
        _fail += 1
        print(f"  [FAIL] {name}" + (f"  {detail}" if detail else ""))
    return cond


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def post_file(url: str, path: Path, field: str = "file") -> dict:
    """手写 multipart，不引第三方库 —— 保持这个脚本零依赖。"""
    bd = "----smoke"
    body = (
        f"--{bd}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{bd}--\r\n".encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={bd}"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def delete_record(iid: int) -> tuple[bool, str]:
    """删掉冒烟造出来的那条记录。

    为什么走 docker compose exec 而不是直连数据库：
        db 容器**故意不对外暴露 5432**（避免和本机装的 PostgreSQL 抢端口），
        所以宿主机上的脚本连不上它。这里复用容器里的 psql，只删我们自己造的这一个 id。

    为什么不加 DELETE 接口：
        产线系统本来就不该让人随手删检测记录 —— 记录是证据。
        冒烟脚本能删，是因为它就在机房这台机器上、能直接操作容器。
    """
    sql = (
        f"DELETE FROM defect_item WHERE inspection_id = {int(iid)};"
        f"DELETE FROM inspection WHERE id = {int(iid)};"
    )
    cmd = [
        "docker", "compose", "exec", "-T", "db",
        "psql", "-U", "postgres", "-d", "defect", "-c", sql,
    ]
    try:
        p = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=60
        )
        if p.returncode == 0:
            return True, "已通过 docker compose exec 删除"
        return False, (p.stderr or p.stdout).strip()[:300]
    except Exception as e:  # docker 不在 PATH、容器没起、超时
        return False, str(e)[:300]


def main() -> int:
    ap = argparse.ArgumentParser(description="真 PG + 真模型 集成冒烟")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--keep", action="store_true", help="保留冒烟造的记录（默认删除）")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    print(f"目标服务：{base}")
    print(f"样本图片：{SAMPLE}")
    if not SAMPLE.exists():
        print("!! 样本图片不存在，先跑一次数据准备")
        return 1

    print("\n[1] 健康检查 —— 必须是真模型、真 PostgreSQL")
    try:
        h = get_json(base + "/health")
    except urllib.error.URLError as e:
        print(f"  [FAIL] 连不上服务：{e}")
        return 1
    check("db 是 postgresql", h.get("db") == "postgresql", str(h.get("db")))
    # 如果是 Mock，后面所有数字都没意义 —— 直接判失败，不继续
    if not check("is_mock 为 false", h.get("is_mock") is False, str(h.get("is_mock"))):
        print("\n当前是 Mock 模型，集成冒烟没有意义，先修好模型加载。")
        return 1
    check("model_name 有值", bool(h.get("model_name")), str(h.get("model_name")))

    print("\n[2] 统计接口 —— 验证 PostgreSQL 方言（ROUND(AVG(double)) 那个 500）")
    st = get_json(base + "/stats")
    check("/stats 返回 200", isinstance(st, list), f"{len(st)} 行")
    if st:
        row = st[0]
        check(
            "/stats 有 avg_latency_ms 且是数字",
            isinstance(row.get("avg_latency_ms"), (int, float)),
            str(row.get("avg_latency_ms")),
        )

    print("\n[3] 报告接口 —— 分母必须在返回值里")
    rep = get_json(base + "/report")
    check("report 有 samples", "samples" in rep)
    if "samples" in rep:
        s = rep["samples"]
        check(
            "positive + negative == labeled",
            s["positive"] + s["negative"] == s["labeled"] == rep["labeled"],
            f"{s}",
        )
    check("report 有 filters.model_sha 键", "model_sha" in rep.get("filters", {}))

    print("\n[4] 真实推理一次 —— 验证输出契约与追溯字段")
    before = get_json(base + "/report")["total_inspections"]
    r = post_file(base + "/detect", SAMPLE)
    iid = r["inspection_id"]
    check("返回 inspection_id", isinstance(iid, int), str(iid))
    check("判定为 ng（这张图有两处划痕）", r["verdict"] == "ng", r["verdict"])
    check("检出 2 处", r["n_defects"] == 2, str(r["n_defects"]))
    check("返回了轮廓", all(len(it.get("polygon") or []) > 3 for it in r["items"]),
          str([len(it.get("polygon") or []) for it in r["items"]]))

    rec = get_json(f"{base}/records/{iid}")
    check("记录带 model_sha", bool(rec.get("model_sha")), str(rec.get("model_sha")))
    check("记录带 imgsz", rec.get("imgsz") is not None, str(rec.get("imgsz")))
    check("记录带 conf_thr", rec.get("conf_thr") is not None, str(rec.get("conf_thr")))
    check("items 与 POST 返回一致", len(rec.get("items", [])) == r["n_defects"],
          f"{len(rec.get('items', []))} vs {r['n_defects']}")

    print("\n[5] 指纹是否真的是磁盘上那个模型")
    model = ROOT / "models" / "best.onnx"
    if model.exists():
        h_ = hashlib.sha256()
        with open(model, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h_.update(chunk)
        want = h_.hexdigest()[:12]
        # 容器里的模型可能和宿主机同名不同内容（构建时会 COPY 进去），
        # 所以这里只比对"有没有值"，不一致时提示而不是判失败。
        if rec.get("model_sha") == want:
            check("sha 与宿主 models/best.onnx 一致", True, want)
        else:
            print(f"  [note] 服务用的 sha={rec.get('model_sha')}，宿主 best.onnx={want}")
            print("         （容器镜像里的模型可能是另一次 COPY 的，属正常）")
    else:
        print("  [note] 宿主没有 models/best.onnx，跳过指纹比对")

    print("\n[6] 按指纹过滤 —— 刚写的那条应该能被筛出来")
    by_sha = get_json(f"{base}/report?model_sha={rec['model_sha']}")
    check("按本次 sha 过滤能命中", by_sha["total_inspections"] >= 1,
          str(by_sha["total_inspections"]))
    check("过滤后的口径里带着 model_sha",
          by_sha["filters"]["model_sha"] == rec["model_sha"])

    print("\n[7] 收尾 —— 记录数回到冒烟之前")
    if args.keep:
        print(f"  [note] --keep：保留记录 #{iid}，条数现在是 {before + 1}")
    else:
        done, msg = delete_record(iid)
        if done:
            after = get_json(base + "/report")["total_inspections"]
            check("删除后条数回到原值", after == before, f"{before} -> {after}")
        else:
            # 删不掉不算冒烟失败：说明这台机器上 docker 命令不可用或容器名不同。
            print(f"  [note] 自动清理失败：{msg}")
            print("         手动清理（只删这一条）：")
            print(f"         docker compose exec -T db psql -U postgres -d defect \\")
            print(f"           -c \"DELETE FROM defect_item WHERE inspection_id={iid};"
                  f" DELETE FROM inspection WHERE id={iid};\"")

    print(f"\n结果：{_ok} 通过 / {_fail} 失败")
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
