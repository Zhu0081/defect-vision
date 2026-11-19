#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理上传目录里"没有数据库记录"的图。

为什么需要这个脚本：
    `data/uploads` 只增不减。按 5000 件/天、每张 2 MB 算，一天 10 GB、一个月 300 GB，
    硬盘两周就满。但直接按天数删又很危险 —— 数据库里有点检记录的图**不能删**，
    删了历史记录就只剩一行字、没有图。

所以规则只有一条：
    文件在 `uploads` 里，但 `inspection.image_path` 里查不到它 → 才能删。

两种用法：
    # 先看会删什么（默认，什么都不删）
    python scripts/purge_uploads.py

    # 真删（同时清掉超过 N 天的空目录）
    python scripts/purge_uploads.py --apply

    # 只清 30 天以前的孤儿文件（有记录的永远不动）
    python scripts/purge_uploads.py --days 30 --apply
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from src.service import config, db  # noqa: E402


def referenced_names() -> set[str]:
    """数据库里所有被引用到的文件名（只取 basename，兼容老数据存绝对路径）。"""
    cols = ("image_path", "filename")
    names: set[str] = set()
    with db.get_engine().connect() as conn:
        for col in cols:
            try:
                rows = conn.execute(text(f"SELECT {col} FROM inspection")).all()
            except Exception as exc:  # 列不存在/表还没建
                print(f"[skip] 读 inspection.{col} 失败：{exc}")
                continue
            for (v,) in rows:
                if v:
                    names.add(Path(str(v)).name)
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description="删除没有数据库记录的上传图（孤儿文件）")
    ap.add_argument("--apply", action="store_true",
                    help="真的删；不加这个参数只打印将要删什么")
    ap.add_argument("--days", type=float, default=0,
                    help="只处理这么多天以前的文件（默认 0 = 所有孤儿文件都算）")
    ap.add_argument("--dir", type=Path, default=config.UPLOAD_DIR,
                    help=f"上传目录（默认 {config.UPLOAD_DIR}）")
    args = ap.parse_args()

    upload_dir: Path = args.dir
    if not upload_dir.exists():
        print(f"目录不存在：{upload_dir}")
        return 1

    keep = referenced_names()
    cutoff = time.time() - args.days * 86400 if args.days > 0 else None

    orphans: list[tuple[Path, int]] = []
    total_bytes = 0
    for p in sorted(upload_dir.rglob("*")):
        if not p.is_file():
            continue
        total_bytes += p.stat().st_size
        if p.name in keep:
            continue                     # 数据库里有记录 → 永远不动
        if cutoff is not None and p.stat().st_mtime > cutoff:
            continue                     # 太新，先留着
        orphans.append((p, p.stat().st_size))

    total_mb = total_bytes / 1024 / 1024
    free_mb = sum(sz for _, sz in orphans) / 1024 / 1024

    print(f"上传目录      {upload_dir}")
    print(f"文件总数      {len(list(x for x in upload_dir.rglob('*') if x.is_file()))}")
    print(f"目录体积      {total_mb:.1f} MB")
    print(f"数据库引用    {len(keep)} 个文件名")
    print(f"孤儿文件      {len(orphans)} 个 / {free_mb:.1f} MB")

    if not orphans:
        print("没有可清理的文件。")
        return 0

    for p, sz in orphans[:10]:
        print(f"  - {p.relative_to(upload_dir)}  ({sz/1024:.0f} KB)")
    if len(orphans) > 10:
        print(f"  …… 还有 {len(orphans) - 10} 个")

    if not args.apply:
        print("\n这是预演（没删任何东西）。确认无误后加 --apply 再跑一次。")
        return 0

    removed, failed = 0, 0
    for p, _ in orphans:
        try:
            p.unlink()
            removed += 1
        except OSError as exc:
            failed += 1
            print(f"  [fail] {p}: {exc}")

    # 顺手清掉空目录（只删空的，非空会抛 OSError，忽略即可）
    for d in sorted((x for x in upload_dir.rglob("*") if x.is_dir()),
                    key=lambda x: len(x.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass

    print(f"\n已删除 {removed} 个文件、{free_mb:.1f} MB；失败 {failed} 个。")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
