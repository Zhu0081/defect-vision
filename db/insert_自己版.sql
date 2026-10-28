-- 第 1 次检测：侧光，不合格，检出 3 个缺陷
INSERT INTO inspection (light, verdict, n_defects, latency_ms)
VALUES ('side', 'ng', 3, 9.8);

INSERT INTO defect (inspection_id, cls, score, x, y, w, h) VALUES (1, 'scratch', 0.87, 120, 340, 60, 18);
INSERT INTO defect (inspection_id, cls, score, x, y, w, h) VALUES (1, 'stain',   0.62, 450, 210, 90, 70);
INSERT INTO defect (inspection_id, cls, score, x, y, w, h) VALUES (1, 'scratch', 0.91, 700, 500, 40, 15);

-- 第 2 次检测：顶光，合格，0 个缺陷（所以 defect 表里没有对应行）
INSERT INTO inspection (light, verdict, n_defects, latency_ms)
VALUES ('top', 'ok', 0, 12.4);
