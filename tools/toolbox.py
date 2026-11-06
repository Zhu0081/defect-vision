#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缺陷检测 · 工具箱（图形菜单版，不用敲命令）

用法：双击 F:\\defect-vision\\工具箱.bat

设计原则：你只需要「选数字」，其余交互全部由这里代做。
所有命令都会先打印出来再执行，方便你出错时截给我看。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

APP_URL = "http://127.0.0.1:8010"
DOCKER_EXE = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
PY = sys.executable


# ---------------------------------------------------------------- 小工具

def rule(ch: str = "=", n: int = 62) -> None:
    print(ch * n)


def title(text: str) -> None:
    print()
    rule()
    print(f"  {text}")
    rule()


def run(cmd: list[str], *, quiet: bool = False) -> int:
    """执行命令，返回退出码。命令本身永远先打印出来。"""
    print()
    print("$ " + " ".join(cmd))
    print("-" * 62)
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT))
    except FileNotFoundError:
        print(f"✗ 找不到程序：{cmd[0]}")
        return 127
    code = proc.returncode
    if code != 0 and not quiet:
        print(f"\n✗ 退出码 {code}（把上面这段截图发我）")
    return code


def ask(prompt: str, default: str = "") -> str:
    tip = f"[{default}]" if default else ""
    try:
        raw = input(f"{prompt}{tip}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)
    return raw or default


def pause() -> None:
    try:
        input("\n回车继续…")
    except (EOFError, KeyboardInterrupt):
        raise SystemExit(0)


# ---------------------------------------------------------------- 环境探测

def capture(cmd: list[str], timeout: float | None = None) -> tuple[int, str]:
    """抓命令输出。强制按 UTF-8 解码——git/docker 输出的是 UTF-8，
    而 Windows 默认按 GBK 解，中文提交信息会直接炸 UnicodeDecodeError。"""
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""


def docker_alive() -> bool:
    code, _ = capture(["docker", "ps"], timeout=15)
    return code == 0


def ensure_docker(*, start_ui: bool = True) -> bool:
    """确保 Docker 引擎可用，必要时拉起 Docker Desktop。"""
    if docker_alive():
        return True

    print("\nDocker 引擎没在跑，正在启动 Docker Desktop…")
    if not Path(DOCKER_EXE).exists():
        print(f"✗ 找不到 {DOCKER_EXE}，请手动启动 Docker Desktop")
        return False
    subprocess.Popen([DOCKER_EXE])
    if not start_ui:
        return False

    for i in range(1, 31):
        time.sleep(4)
        print(f"  等待引擎就绪… {i * 4}s", end="\r", flush=True)
        if docker_alive():
            print("\n✓ Docker 引擎就绪" + " " * 20)
            return True
    print("\n✗ 等了 120 秒引擎还没起来，点开右下角鲸鱼图标看一眼")
    return False


def health() -> dict | None:
    try:
        with urllib.request.urlopen(f"{APP_URL}/health", timeout=5) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


# ---------------------------------------------------------------- 各个功能

def todo_start() -> None:
    title("启动网站")
    if not ensure_docker():
        return
    if run(["docker", "compose", "up", "-d", "--wait", "--build"]) != 0:
        return
    print("\n等接口热起来…")
    for _ in range(20):
        time.sleep(2)
        if health():
            break
    h = health()
    if h:
        print(f"\n✓ 服务已就绪：{h.get('db')} / {h.get('model_name')} / mock={h.get('is_mock')}")
        print(f"  正在打开 {APP_URL}")
        webbrowser.open(APP_URL)
    else:
        print("\n✗ 服务起来了但接口没响应，用「8. 一键体检」看看")


def todo_stop() -> None:
    title("关掉网站")
    if not ensure_docker(start_ui=False):
        return
    run(["docker", "compose", "down"])
    print("\n✓ 已关闭（数据还在卷里，下次启动还在）")


def todo_ingest() -> None:
    title("导入手机照片")
    part = ask("工件号（如 025）")
    if not part.isdigit():
        print("✗ 工件号要是数字，比如 025")
        return
    part = f"{int(part):03d}"

    light = ask("光源 side / top", "side").lower()
    if light not in ("side", "top"):
        print("✗ 光源只能是 side 或 top")
        return

    is_defect = ask("这批是 良品 还是 缺陷？(g/d)", "g").lower().startswith("d")
    label = "defect" if is_defect else "good"
    defect = ""
    if is_defect:
        defect = ask("缺陷类型 scratch/stain/missing/deform", "scratch").lower()
        if defect not in ("scratch", "stain", "missing", "deform"):
            print("✗ 类型不对")
            return

    src = ROOT / "data" / "phone_inbox" / f"drop_{light}" / f"p{part}"
    if not src.exists():
        print(f"\n✗ 文件夹不存在：{src}")
        print("  先在资源管理器里把照片放进去，再回来跑这个")
        return
    n = len([p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"\n文件夹里有 {n} 张图：{src}")
    if n == 0:
        print("✗ 一张都没有")
        return

    cmd = [PY, r"src\data\ingest_phone.py", "--src", str(src),
           "--category", "card", "--part", part, "--light", light, "--label", label]
    if defect:
        cmd += ["--defect", defect]

    if ask("先预演一遍（不真入库）？(Y/n)", "y").lower().startswith("y"):
        run(cmd + ["--dry-run"])
        print()
        if not ask("看着没问题，真入库？(y/N)", "n").lower().startswith("y"):
            print("已取消")
            return
    run(cmd)
    print("\n✓ 入库完成。记得把这一批照片从 drop 里挪到 done，别下次重复导")


def todo_light_compare() -> None:
    title("光源实验对比")
    conf = ask("判定阈值", "0.25")
    run([PY, r"src\eval\light_compare.py", "--conf", conf, "--from-part", "25"])


def todo_train() -> None:
    title("训练新模型")
    epochs = ask("训练轮数", "30")
    name = ask("这次的名字", "baseline_" + datetime.now().strftime("%m%d_%H%M"))
    run([PY, r"src\train\train.py", "--epochs", epochs, "--imgsz", "1280",
         "--batch", "4", "--name", name])
    print(f"\n结果在 runs\\segment\\{name}\\，权重在 weights\\best.pt")


def todo_test() -> None:
    title("跑测试")
    run([PY, "-m", "pytest", "tests", "-q"])


def todo_push() -> None:
    title("推送代码")
    run(["git", "status", "--short"])
    msg = ask("这次改了什么（一句话）", "update")
    run(["git", "add", "-A"])
    run(["git", "commit", "-m", msg])
    run(["git", "push", "gitee", "main"])
    run(["git", "push", "github", "main"])
    print("\n✓ 两个平台都推了（网页有 1 分钟延迟）")


def todo_checkup() -> None:
    title("一键体检")
    ok, bad = "✓", "✗"

    print(f"  {ok if Path('models/best.onnx').exists() else bad} 模型文件 models/best.onnx")

    d_alive = docker_alive()
    print(f"  {ok if d_alive else bad} Docker 引擎")
    if d_alive:
        _, out = capture(["docker", "compose", "ps", "--format", "{{.Name}}  {{.State}}"])
        for line in out.strip().splitlines():
            print(f"      {line}")

    h = health()
    if h:
        print(f"  {ok} 网页接口 {APP_URL}")
        print(f"      数据库 {h.get('db')} / 模型 {h.get('model_name')} / mock={h.get('is_mock')}")
    else:
        print(f"  {bad} 网页接口没响应（用 1 号菜单启动）")

    meta = ROOT / "data" / "captured" / "metadata.csv"
    if meta.exists():
        import csv
        with meta.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        parts = {r.get("part_id") for r in rows}
        print(f"  {ok} 数据 {len(rows)} 张 / {len(parts)} 个工件")
    else:
        print(f"  {bad} 没有 metadata.csv")

    _, out = capture(["git", "log", "--oneline", "-3"])
    print("  最近提交：")
    for line in out.strip().splitlines():
        print(f"      {line}")


MENU = [
    ("启动网站（自动开浏览器）", todo_start),
    ("关掉网站", todo_stop),
    ("导入手机照片", todo_ingest),
    ("光源实验对比表", todo_light_compare),
    ("训练新模型", todo_train),
    ("跑测试", todo_test),
    ("推送代码到 gitee + github", todo_push),
    ("一键体检", todo_checkup),
]


def main() -> int:
    while True:
        print()
        rule("=")
        print("        工业表面缺陷检测 · 工具箱")
        rule("=")
        for i, (name, _) in enumerate(MENU, 1):
            print(f"   {i}. {name}")
        print("   0. 退出")
        rule("-")

        try:
            raw = input("请选择编号，回车执行: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if raw in ("0", "q", ""):
            return 0
        if not raw.isdigit() or not (1 <= int(raw) <= len(MENU)):
            print("没有这个编号")
            continue

        name, fn = MENU[int(raw) - 1]
        try:
            fn()
        except KeyboardInterrupt:
            print("\n（已中断）")
        except Exception as exc:  # noqa: BLE001
            print(f"\n✗ 出错了：{type(exc).__name__}: {exc}")
            print("  把这段截图发我")
        pause()


if __name__ == "__main__":
    raise SystemExit(main())
