"""极简查询器 —— 只为练 SQL 用，正式服务里不用它。

用法：
    python db\\q.py "select * from inspection"
    python db\\q.py "select * from defect where score > 0.8"

换一个库（比如服务真正的数据库）就加第二个参数：
    python db\\q.py "select * from inspection" data\\defect.db

不写 SQL 参数时，默认查整张表。
"""
import sys
import sqlite3
from pathlib import Path

USAGE_DB = Path(__file__).with_name("test.db")
DB = Path(sys.argv[2]) if len(sys.argv) > 2 else USAGE_DB
if not DB.exists():
    print(f"找不到数据库文件：{DB.resolve()}")
    print("提示：第二个参数要写数据库文件的路径，例如 data\\defect.db")
    sys.exit(1)
sql = sys.argv[1] if len(sys.argv) > 1 else "select * from inspection"

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row

try:
    rows = con.execute(sql).fetchall()
except sqlite3.Error as e:
    print(f"SQL error: {e}")
    sys.exit(1)

if not rows:
    print("0 row(s)")
    sys.exit(0)


def cell(v) -> str:
    return "NULL" if v is None else str(v)


cols = rows[0].keys()
widths = [max(len(c), *(len(cell(r[c])) for r in rows)) for c in cols]

header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
print(header)
print("-" * len(header))
for r in rows:
    print("  ".join(cell(r[c]).ljust(w) for c, w in zip(cols, widths)))
print(f"\n{len(rows)} row(s)")
