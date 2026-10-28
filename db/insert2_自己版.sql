-- 第 3 次检测：背光，不合格，检出 2 个缺件
INSERT INTO inspection (light, verdict, n_defects, latency_ms)
VALUES ('back', 'ng', 2, 11.2);

INSERT INTO defect (inspection_id, cls, score, x, y, w, h) VALUES (3, 'missing', 0.95, 210, 180, 34, 34);
INSERT INTO defect (inspection_id, cls, score, x, y, w, h) VALUES (3, 'missing', 0.71, 660, 420, 28, 30);
