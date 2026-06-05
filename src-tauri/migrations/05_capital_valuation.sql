-- 给股本快照表补充估值字段（来自 zhitu hs/real/ssjy：pe 市盈率、sjl 市净率）。
-- 估值是非技术预测维度，随刷新更新。SQLite 无 ADD COLUMN IF NOT EXISTS，迁移运行器对
-- "duplicate column name" 容错。
ALTER TABLE stock_capital ADD COLUMN pe REAL NOT NULL DEFAULT 0;
ALTER TABLE stock_capital ADD COLUMN pb REAL NOT NULL DEFAULT 0;
