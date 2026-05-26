-- 量比 / 换手率 数据支撑
-- 1. 独立的股本表（流通股本用于换手率计算）
CREATE TABLE IF NOT EXISTS stock_capital (
    symbol TEXT PRIMARY KEY,
    circulating_shares REAL NOT NULL DEFAULT 0,
    total_shares REAL NOT NULL DEFAULT 0,
    circulating_market_cap REAL NOT NULL DEFAULT 0,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2. 历史数据增加量比列（换手率 turnover_rate 列已存在，仅回填）
-- 注意：SQLite 不支持 ADD COLUMN IF NOT EXISTS，迁移执行器会忽略 "duplicate column" 错误
ALTER TABLE historical_data ADD COLUMN volume_ratio REAL NOT NULL DEFAULT 0;

-- 3. 实时数据同步增加量比列
ALTER TABLE realtime_data ADD COLUMN volume_ratio REAL NOT NULL DEFAULT 0;
