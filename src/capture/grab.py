#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grab.py —— 缺陷检测项目：多光源采图工具（在 Windows 本机跑，不要在 WSL 里跑）

为什么在 Windows 跑：
    WSL2 访问 USB 摄像头要装 usbipd，折腾且不稳；而采图只要求「稳定、可重复」，
    不做推理，所以放在 Windows 侧用 OpenCV + DirectShow 最省事。

这个脚本存在唯一目的：**保证同一批数据的曝光、白平衡、对焦是锁死的**。
    否则后面「顶光 / 侧光 / 背光」的对比实验全是伪结论——
    你以为是光源变了，其实是相机的自动曝光偷偷变了。

用法：
    # 1) 先看有哪些摄像头
    python src/capture/grab.py --list-cameras

    # 2) 采 PCB 1 号件，顶光，良品
    python src/capture/grab.py --source 0 --category pcb --part 001 --light top

    # 3) 采 PCB 1 号件，顶光，划痕缺陷（进窗口后按 1）
    python src/capture/grab.py --source 0 --category pcb --part 001 --light top --defect scratch

    # 4) 手机 IP Webcam（推荐，画质比笔记本摄像头好得多）
    python src/capture/grab.py --source http://192.168.1.5:8080/video --category metal --part 002 --light back

    # 5) 只调曝光不采图（要按键盘）
    python src/capture/grab.py --source 0 --calibrate

    # 6) ★ 不用按任何键：自动扫一组曝光，拍成图让你挑（推荐新手先用这个）
    python src/capture/grab.py --source 0 --sweep

    # 7) ★★ 本机摄像头手动曝光无效时的正解：全自动模式，直接拍
    python src/capture/grab.py --source 0 --auto --category pcb --part 001 --light top

窗口内快捷键：
    SPACE   拍摄
    1-4     选择缺陷类型（scratch / stain / missing / deform），自动切到「缺陷」
    0       切回「良品」
    [ ]     曝光 -1 / +1（不同相机单位不同，靠 luma 读数找到合适的值）
    f       重新自动对焦一次（对好之后会锁死）
    n       换下一件 / 换光源 / 换类别（会提示输入）
    u       撤销上一张（删文件 + 删 CSV 行）
    q       退出

输出：
    data/captured/light_<光源>/<类别>/<类别>_p<件号>_<良品或缺陷>_<序号>.jpg
    data/captured/metadata.csv      ← 每张图的完整参数，后面做光源消融实验要用

依赖：
    pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ★ 必须在 import cv2 之前设置。不设的话，手机 IP Webcam / RTSP 的预览会延迟 2~3 秒，
#   你按空格拍下来的其实是 3 秒前的画面（手都挪开了，对焦点错位）。
#     rtsp_transport;tcp  —— 走 TCP，不乱序不丢帧
#     fflags;nobuffer     —— 不让 FFmpeg 攒缓冲
#     flags;low_delay     —— 低延迟解码
#     max_delay;500000    —— 最多容忍 0.5 秒（单位：微秒）
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000",
)

import cv2          # noqa: E402  （必须放在上面的环境变量之后）
import numpy as np  # noqa: E402

# Windows 控制台默认 GBK，打印中文会炸，这里强制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


CATEGORIES = ["pcb", "metal", "plastic", "card"]
LIGHTS = ["top", "side", "back"]
DEFECT_TYPES = {1: "scratch", 2: "stain", 3: "missing", 4: "deform"}
DEFECT_TO_KEY = {v: k for k, v in DEFECT_TYPES.items()}

CSV_FIELDS = [
    "filename", "relpath", "category", "part_id", "light", "label", "defect",
    "width", "height", "exposure", "focus", "wb", "luma_mean", "sharpness",
    "luma_ok", "captured_at",
]

LUMA_TOL = 0.10          # 亮度偏离基准超过 10% 就报警
MIN_SHARP_DEFAULT = 30.0  # Laplacian 方差低于这个值说明糊了


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------
def parse_res(s: str) -> tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise SystemExit(f"--res 格式应为 1920x1080，收到：{s}")


def mean_luma(frame: np.ndarray) -> float:
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def ascii_safe(s: str) -> str:
    """窗口上不能显示中文（OpenCV 不认 CJK），转成安全字符串。"""
    return "".join(ch if ord(ch) < 128 else "?" for ch in s)


def draw_overlay(frame: np.ndarray, lines: list[str], warns: list[str]) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    scale = max(0.6, w / 1280.0)
    fs = 0.7 * scale
    th = max(1, int(round(1.6 * scale)))
    dy = int(round(28 * scale))
    y = int(round(26 * scale))

    def put(text, color, yy):
        cv2.putText(img, text, (12, yy), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(img, text, (12, yy), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)

    for t in lines:
        put(ascii_safe(t), (0, 255, 0), y)
        y += dy
    y += int(dy * 0.4)
    for t in warns:
        put(ascii_safe(t), (0, 0, 255), y)
        y += dy
    return img


# --------------------------------------------------------------------------
# 相机
# --------------------------------------------------------------------------
def list_cameras() -> None:
    print("正在探测摄像头（0-5）...\n")
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
    for i in range(6):
        cap = cv2.VideoCapture(i, backend)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  [{i}] OK    {w}x{h}")
        elif cap.isOpened():
            print(f"  [{i}] 打开但读不到帧（被其他程序占用？）")
        cap.release()
    print("\n用法：--source 0")


def open_camera(source: str, res: tuple[int, int], use_mjpg: bool = True) -> cv2.VideoCapture:
    """打开相机并尽量按「工业相机」的方式配置：固定分辨率、MJPG、先关自动。"""
    if source.isdigit():
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
        cap = cv2.VideoCapture(int(source), backend)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise SystemExit(f"打不开视频源：{source}\n  摄像头序号先跑 --list-cameras 确认。")

    # ★ 顺序很重要：先设 FOURCC 再设分辨率。
    # 很多 USB / 笔记本摄像头只有 MJPG 才能跑到 1080p，用默认 YUY2 会卡在 640x480@30。
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def lock_camera(cap: cv2.VideoCapture, exposure: float | None,
                focus: float | None, wb: float | None) -> None:
    """关掉一切「自动」，这是数据一致性的前提。"""
    # 注意：CAP_PROP_AUTO_EXPOSURE 在不同后端语义相反。
    # DirectShow：0.25 = 手动；V4L2：1 = 手动。这里按 Windows 处理。
    if exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    if focus is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, focus)
    if wb is not None:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)


def enable_auto(cap: cv2.VideoCapture, settle: float = 2.5) -> tuple[float, float]:
    """退回「全自动」：曝光 / 对焦 / 白平衡都交给相机。

    为什么需要它：本机笔记本摄像头在手动曝光模式下，把 CAP_PROP_EXPOSURE 设成
    任何值（实测 -13 ~ +1）画面都会掉到接近全黑（luma 1~17），而自动模式是 87。
    也就是说这个驱动的手动曝光等于不可用。与其硬锁，不如固定光源 + 全自动，
    再把 luma 记进 CSV，后面写报告时如实说明这个限制。
    """
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # DirectShow：0.75 = 自动
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    t0 = time.time()
    frame = None
    while time.time() - t0 < settle:
        ok, fr = cap.read()
        if ok:
            frame = fr
        time.sleep(0.02)

    if frame is None:
        print("  [警告] 自动模式下读不到帧")
        return 0.0, 0.0

    lum = mean_luma(frame)
    shp = sharpness(frame)
    print(f"已切到全自动模式：luma={lum:.1f}  sharp={shp:.1f}")
    print(f"  相机自报读数：exposure={cap.get(cv2.CAP_PROP_EXPOSURE):.2f}  "
          f"focus={cap.get(cv2.CAP_PROP_FOCUS):.2f}  "
          f"wb={cap.get(cv2.CAP_PROP_WB_TEMPERATURE):.0f}")
    print("  注意：自动模式会随光源强弱自己调整，所以【灯的位置和亮度全程不许动】。\n")
    return lum, shp


# --------------------------------------------------------------------------
# 交互
# --------------------------------------------------------------------------
def ask(prompt: str, choices: list[str], default: str | None = None) -> str:
    while True:
        hint = f"[{'/'.join(choices)}]"
        if default:
            hint += f" (默认 {default})"
        raw = input(f"{prompt} {hint}: ").strip().lower()
        if not raw and default:
            return default
        if raw in choices:
            return raw
        print("  输入不对，重来。")


def ask_new_workpiece(st: dict) -> None:
    print("\n——— 新建采集对象 ———")
    st["category"] = ask("工件类别", CATEGORIES, st.get("category"))
    st["part"] = input(f"件号（如 001，当前 {st.get('part') or '无'}）: ").strip() or st.get("part")
    if not st.get("part"):
        st["part"] = "000"
    st["light"] = ask("光源", LIGHTS, st.get("light"))


def run_calibrate(cap: cv2.VideoCapture, exposure: float | None) -> None:
    """只调曝光和对焦，不采图。采图前先跑一次这个，把参数记下来。"""
    win = "calibrate - [ ] exposure  f focus  q quit"
    set_nice_window(win)
    print("目标：让 luma 稳定在 110~150，且 sharpness 越高越好（>100 基本够用）。")
    print("调好后把 exp 的值记下来，下次用 --exposure 传进去。\n")
    print("⚠ 按键没反应？先鼠标点一下视频窗口（让它获得焦点），再按键。")
    print("  还是不行就直接用 --sweep，那个全程不用按键。\n")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("读帧失败")
            break
        luma = mean_luma(frame)
        sharp = sharpness(frame)
        lines = [
            ">>> CLICK THIS WINDOW FIRST, then press keys <<<",
            f"luma={luma:6.1f}   (target 110-150)",
            f"sharp={sharp:7.1f}  (>100 good)",
            f"exposure={exposure}",
            "  [ -    ] +    f autofocus    q quit",
        ]
        warns = []
        if luma < 110:
            warns.append("too dark -> press ]")
        elif luma > 150:
            warns.append("too bright -> press [")
        if sharp < 100:
            warns.append("soft image -> focus / move closer")
        cv2.imshow(win, draw_overlay(frame, lines, warns))

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in (ord("["), ord("-"), ord("]"), ord("+"), ord("=")):
            cur = cap.get(cv2.CAP_PROP_EXPOSURE) if exposure is None else exposure
            down = k in (ord("["), ord("-"))
            exposure = cur - 1 if down else cur + 1
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            time.sleep(0.15)
            print(f"  exposure = {exposure}")
        if k == ord("f"):
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            time.sleep(1.2)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            print(f"  focus = {cap.get(cv2.CAP_PROP_FOCUS):.1f}")

    cv2.destroyAllWindows()
    print(f"\n最终参数：--exposure {exposure}")
    print("（曝光锁定后，采图全程不要再动台灯亮度）")


def set_nice_window(win: str, w: int = 1280, h: int = 720) -> None:
    """建窗口并置顶——否则窗口开在终端后面，用户以为没反应。"""
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w, h)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass


def run_sweep(cap: cv2.VideoCapture, lo: float, hi: float, step: float,
              out_dir: Path) -> None:
    """不用按任何键：自动扫一组曝光值各拍一张，用数字告诉你该选哪个。

    这是给「按了键却没反应」的情况准备的兜底方案——
    它完全不依赖 OpenCV 窗口的键盘焦点。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print("先自动对焦一次并锁定……")
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    time.sleep(1.5)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    for _ in range(15):
        cap.read()

    print(f"开始扫描曝光 {lo} → {hi}（步长 {step}），每个值拍一张\n")
    results: list[tuple[float, float, float, Path]] = []
    v = round(float(lo), 3)
    while v <= hi + 1e-9:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, v)
        time.sleep(0.30)
        ok, frame = False, None
        for _ in range(6):
            ok, frame = cap.read()
            if ok:
                break
            time.sleep(0.10)
        if not ok or frame is None:
            print(f"  曝光 {v:6.1f}   读帧失败，跳过")
            v += step
            continue

        lum = mean_luma(frame)
        shp = sharpness(frame)
        path = out_dir / f"exp_{v:+.1f}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        results.append((v, lum, shp, path))
        flag = "   <== 落在 110~150" if 110 <= lum <= 150 else ""
        print(f"  曝光 {v:6.1f}   luma {lum:6.1f}   sharp {shp:7.1f}{flag}")
        v += step

    if not results:
        print("\n全部失败。先跑 --list-cameras 确认摄像头能被读到帧。")
        return

    lums = [r[1] for r in results]
    print()
    print(f"luma 跨度：{min(lums):.0f} → {max(lums):.0f}（变化 {max(lums) - min(lums):.0f}）")

    if max(lums) - min(lums) < 8:
        print("⚠ 曝光从头扫到尾，亮度几乎没变 —— 这台相机多半不支持手动曝光。")
        print("  对策（按性价比）：")
        print("  1. 用手机 IP Webcam，在 App 里把曝光/白平衡/对焦锁死，画质还更好")
        print("  2. 靠摆灯的距离控制亮度：灯近一点亮，远一点暗，然后全程不动灯")
        print("  3. 换 USB 摄像头（几十块的 UVC 一般支持）")

    good = [r for r in results if 110 <= r[1] <= 150]
    if good:
        best = min(good, key=lambda r: abs(r[1] - 130))
        print(f"\n✔ 推荐用：--exposure {best[0]}")
        print(f"  该张样图：{best[3]}")
    else:
        best = min(results, key=lambda r: abs(r[1] - 130))
        print(f"\n⚠ 没有一个曝光落在 110~150，最接近的是 {best[0]:.1f}（luma {best[1]:.0f}）")
        print("  先改灯（移近/移远/加柔光纸），再扫一次。别硬用。")

    print(f"\n全部样张在：{out_dir}")
    print("双击打开，挑清楚的那张，然后按它上面的曝光值去采图。")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="多光源缺陷采图工具")
    ap.add_argument("--source", default="0", help="摄像头序号或 RTSP/HTTP 地址")
    ap.add_argument("--out", default="data/captured", help="输出根目录")
    ap.add_argument("--res", default="1920x1080", help="采集分辨率，如 1920x1080")
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--focus", type=float, default=None)
    ap.add_argument("--wb", type=float, default=None)
    ap.add_argument("--category", default=None, choices=CATEGORIES)
    ap.add_argument("--part", default=None, help="件号")
    ap.add_argument("--light", default=None, choices=LIGHTS)
    ap.add_argument("--defect", default=None, choices=list(DEFECT_TYPES.values()))
    ap.add_argument("--min-sharp", type=float, default=MIN_SHARP_DEFAULT)
    ap.add_argument("--warmup", type=int, default=30, help="丢弃前 N 帧等自动曝光稳定")
    ap.add_argument("--no-mjpg", action="store_true", help="不强制 MJPG")
    ap.add_argument("--auto", action="store_true",
                    help="全自动曝光/对焦/白平衡（本机摄像头手动曝光不可靠时用这个）")
    ap.add_argument("--calibrate", action="store_true", help="只调曝光/对焦，不采图")
    ap.add_argument("--sweep", action="store_true",
                    help="不用按键：自动扫一组曝光各拍一张，挑完再采图")
    ap.add_argument("--sweep-from", type=float, default=-13)
    ap.add_argument("--sweep-to", type=float, default=-1)
    ap.add_argument("--sweep-step", type=float, default=1)
    ap.add_argument("--list-cameras", action="store_true")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "metadata.csv"
    res = parse_res(args.res)

    # 载入已有记录（支持断点续采、计号连续）
    rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            print(f"已有 {len(rows)} 条记录，继续采。")

    def flush_csv() -> None:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)

    cap = open_camera(args.source, res, use_mjpg=not args.no_mjpg)
    auto_mode = bool(args.auto)
    if auto_mode:
        enable_auto(cap)
        if not args.source.isdigit():
            print("提示：手机 / 网络流的曝光、对焦脚本这边设了也不生效，")
            print("     必须去手机 App 里锁：对焦 → 固定，白平衡 → 固定，然后全程别动手机。\n")
    else:
        lock_camera(cap, args.exposure, args.focus, args.wb)

    if args.sweep:
        try:
            run_sweep(cap, args.sweep_from, args.sweep_to, args.sweep_step,
                      out_root / "_calib")
        finally:
            cap.release()
        return

    if args.calibrate:
        try:
            run_calibrate(cap, args.exposure)
        finally:
            cap.release()
        return

    st = {
        "category": args.category,
        "part": args.part,
        "light": args.light,
        "defect": args.defect,       # None 表示良品
        "exposure": "auto" if auto_mode else args.exposure,
        "focus": args.focus,
    }
    if not all([st["category"], st["part"], st["light"]]):
        ask_new_workpiece(st)
    print(f"\n当前：类别={st['category']} 件号={st['part']} 光源={st['light']} "
          f"标签={st['defect'] or '良品'}")
    if auto_mode:
        print("提示：全自动模式。把工件放好、灯摆好，直接按空格拍。\n")
    else:
        print("提示：先把工件放好，对着按 f 自动对焦，再按 ] / [ 把 luma 调到 110~150 之间。\n")

    for _ in range(max(0, args.warmup)):
        cap.read()

    print("=" * 58)
    print("会弹出两个窗口：")
    print("  1) 视频窗口  ← 按键前必须先用鼠标点一下它（给它焦点）")
    print("  2) 本终端    ← 这里只显示文字，在这儿按键没用")
    print("拍照 = 点一下视频窗口，然后按【空格】")
    print("=" * 58)
    win = "grab - defect capture"
    set_nice_window(win)

    base_luma: float | None = None
    saved: list[dict] = []
    last_sharp = 0.0
    last_luma = 0.0
    fails = 0
    shots = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails % 30 == 1:
                    print(f"  [警告] 连续读帧失败 {fails} 次，尝试重连...")
                if fails >= 150:
                    cap.release()
                    time.sleep(2)
                    cap = open_camera(args.source, res, use_mjpg=not args.no_mjpg)
                    if auto_mode:
                        enable_auto(cap, settle=1.0)
                    else:
                        lock_camera(cap, st["exposure"], st["focus"], args.wb)
                    fails = 0
                continue
            fails = 0

            last_luma = mean_luma(frame)
            last_sharp = sharpness(frame)

            if base_luma is None:
                drift = 0.0
            else:
                drift = (last_luma - base_luma) / base_luma

            label = st["defect"] or "GOOD"
            lines = [
                f"category={st['category']}  part={st['part']}  light={st['light']}  label={label}",
                f"luma={last_luma:6.1f}  base={base_luma if base_luma is not None else 0:6.1f}  drift={drift:+.1%}",
                f"sharp={last_sharp:7.1f}  min={args.min_sharp:.0f}  exp={st['exposure']}  focus={st['focus']}",
                f"saved this session: {shots}",
                "CLICK WINDOW FIRST | SPACE=shot 0-4=type []=exposure f=focus n=new u=undo q=quit",
            ]
            warns = []
            if base_luma is not None and abs(drift) > LUMA_TOL:
                warns.append(f"!! LUMA DRIFT {drift:+.1%}  -> exposure/lighting changed!")
            if last_sharp < args.min_sharp:
                warns.append(f"!! BLURRY (sharp {last_sharp:.0f} < {args.min_sharp:.0f})")

            cv2.imshow(win, draw_overlay(frame, lines, warns))
            k = cv2.waitKey(1) & 0xFF

            # ---------------- 拍摄 ----------------
            if k in (ord(" "), 13, ord("p")):
                if last_sharp < args.min_sharp:
                    print(f"  糊了（sharp={last_sharp:.1f}），没保存。挪近点或按 f 重新对焦。")
                    continue

                if base_luma is None:
                    base_luma = last_luma
                    print(f"  基准亮度已锁定：{base_luma:.1f}")

                sub = out_root / f"light_{st['light']}" / st["category"]
                sub.mkdir(parents=True, exist_ok=True)
                lbl = st["defect"] or "good"
                prefix = f"{st['category']}_p{st['part']}_{lbl}_"
                seq = len(list(sub.glob(prefix + "*.jpg"))) + 1
                name = f"{prefix}{seq:03d}.jpg"
                path = sub / name

                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    print(f"  写文件失败：{path}")
                    continue

                h, w = frame.shape[:2]
                row = {
                    "filename": name,
                    "relpath": str(path.relative_to(out_root.parent)).replace("\\", "/"),
                    "category": st["category"],
                    "part_id": st["part"],
                    "light": st["light"],
                    "label": "defect" if st["defect"] else "good",
                    "defect": st["defect"] or "",
                    "width": w,
                    "height": h,
                    "exposure": "" if st["exposure"] is None else st["exposure"],
                    "focus": "" if st["focus"] is None else st["focus"],
                    "wb": "" if args.wb is None else args.wb,
                    "luma_mean": f"{last_luma:.2f}",
                    "sharpness": f"{last_sharp:.2f}",
                    "luma_ok": "1" if (base_luma is None or abs(drift) <= LUMA_TOL) else "0",
                    "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                rows.append(row)
                saved.append(row)
                flush_csv()
                shots += 1
                print(f"  [{shots:3d}] {name}  luma={last_luma:.0f} sharp={last_sharp:.0f}")

            # ---------------- 缺陷类型 ----------------
            elif ord("0") <= k <= ord("4"):
                n = k - ord("0")
                if n == 0:
                    st["defect"] = None
                    print("  -> 良品")
                elif n in DEFECT_TYPES:
                    st["defect"] = DEFECT_TYPES[n]
                    print(f"  -> 缺陷：{st['defect']}")

            # ---------------- 曝光微调 ----------------
            elif k in (ord("["), ord("-"), ord("]"), ord("+"), ord("=")):
                if auto_mode:
                    print("  --auto 模式下曝光由相机自己管，[ ] 无效。要手动锁请去掉 --auto。")
                else:
                    down = k in (ord("["), ord("-"))
                    if st["exposure"] is None:
                        st["exposure"] = cap.get(cv2.CAP_PROP_EXPOSURE)
                    st["exposure"] = st["exposure"] + (-1 if down else 1)
                    lock_camera(cap, st["exposure"], st["focus"], args.wb)
                    time.sleep(0.15)
                    print(f"  exposure = {st['exposure']}   luma={last_luma:.0f}")

            # ---------------- 对焦 ----------------
            elif k == ord("f"):
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                time.sleep(1.2)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                st["focus"] = round(cap.get(cv2.CAP_PROP_FOCUS), 1)
                print(f"  已对焦并锁定，focus = {st['focus']}")

            # ---------------- 换对象 ----------------
            elif k == ord("n"):
                cv2.destroyWindow(win)
                ask_new_workpiece(st)
                base_luma = None   # 换光源必须重新建立亮度基准
                print(f"  -> 类别={st['category']} 件号={st['part']} 光源={st['light']}")
                set_nice_window(win)
                print("  （窗口会重新弹出，记得用鼠标点一下它再按键）")

            # ---------------- 撤销 ----------------
            elif k == ord("u"):
                if not saved:
                    print("  这一轮还没拍，没得撤。")
                else:
                    row = saved.pop()
                    rows.remove(row)
                    p = out_root.parent / row["relpath"]
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
                    flush_csv()
                    shots = max(0, shots - 1)
                    print(f"  已撤销：{row['filename']}")

            # ---------------- 退出 ----------------
            elif k == ord("q") or k == 27:
                break

    except KeyboardInterrupt:
        print("\n中断。")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        flush_csv()

    print(f"\n本轮拍摄 {shots} 张，元数据：{csv_path}")
    print(f"累计 {len(rows)} 张。")
    if rows:
        from collections import Counter
        c = Counter((r["light"], r["label"]) for r in rows)
        print("\n当前数据分布：")
        for (light, label), n in sorted(c.items()):
            print(f"  light_{light:<5} {label:<7} {n:4d}")


if __name__ == "__main__":
    main()
