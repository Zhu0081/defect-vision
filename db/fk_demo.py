"""外键演示 —— 只在 test.db 的副本上跑，不会动你的真数据。

用法：python db\\fk_demo.py
"""
import os
import shutil
import sqlite3
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(HERE, "test.db")
tmp = os.path.join(tempfile.gettempdir(), "fk_demo.db")

if not os.path.exists(src):
    raise SystemExit("找不到 db/test.db，先把建表和插入跑完再来")

shutil.copyfile(src, tmp)


def try_insert(con, label):
    try:
        con.execute("INSERT INTO defect (inspection_id, cls) VALUES (999, 'ghost')")
        con.commit()
        print(f"  {label} -> 插入成功（数据库没管）")
    except sqlite3.IntegrityError as e:
        print(f"  {label} -> 被拒绝：{e}")


print("【1】默认状态")
con = sqlite3.connect(tmp)
print("  PRAGMA foreign_keys =", con.execute("PRAGMA foreign_keys").fetchone()[0], "（0 = 关）")
try_insert(con, "往 defect 插 inspection_id=999")
con.close()

print("\n【2】执行 PRAGMA foreign_keys = ON 之后")
con = sqlite3.connect(tmp)
con.execute("PRAGMA foreign_keys = ON")
print("  PRAGMA foreign_keys =", con.execute("PRAGMA foreign_keys").fetchone()[0], "（1 = 开）")
try_insert(con, "往 defect 插 inspection_id=999")
print("\n  inspection 表里真的有 999 这条记录吗？")
print("  ", con.execute("SELECT COUNT(*) FROM inspection WHERE id = 999").fetchone()[0], "条")
con.close()

os.remove(tmp)
print("\n（演示完毕，真实数据库没有被改动）")
