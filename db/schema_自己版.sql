-- 删表：子表在前，父表在后
DROP TABLE IF EXISTS defect;
DROP TABLE IF EXISTS inspection;

-- 表一：一次检测一行
CREATE TABLE inspection (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    light TEXT,
    verdict TEXT,
    n_defects INTEGER,
    latency_ms REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 表二：一个缺陷一行
CREATE TABLE defect (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL,
    cls TEXT NOT NULL,
    score REAL,
    x INTEGER,
    y INTEGER,
    w INTEGER,
    h INTEGER,
    FOREIGN KEY (inspection_id) REFERENCES inspection(id)
);
