-- 给股票详情表补充「人工板块分类」字段（科技/能源/矿业/电力/能源金属/消费/半导体/军工汽车/医药生物）。
-- 该分类由 examples/fetch_sector_leaders.rs 的龙头清单写入，用于股票列表页按板块分组展示。
-- 与 zhitu 细分 industry 区分：category 为粗板块、人工策划。
-- SQLite 无 ADD COLUMN IF NOT EXISTS，迁移运行器对 "duplicate column name" 容错。
ALTER TABLE stock ADD COLUMN category TEXT NOT NULL DEFAULT '';
