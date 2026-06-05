-- 基本面财务指标（来自 zhitu hs/gs/cwzb），按季度报告期存储，供点对点截面因子使用。
-- 多个报告期/股票 → 主键 (symbol, report_date)。值缺失（接口返回 "--"）存 NULL。
CREATE TABLE IF NOT EXISTS stock_fundamentals (
    symbol TEXT NOT NULL,
    report_date TEXT NOT NULL,      -- 报告期(季度末)，如 2026-03-31
    eps REAL,                       -- 每股收益(mgsy)，按报告期口径(YTD累计)
    bps REAL,                       -- 每股净资产(mgjz)
    roe REAL,                       -- 净资产收益率(jzsy, %)
    profit_growth REAL,             -- 净利润增长率(jlzz, %)
    revenue_growth REAL,            -- 主营收入增长率(zysr, %)
    debt_ratio REAL,                -- 资产负债率(zcfzl, %)，缺失为 NULL
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, report_date)
);

CREATE INDEX IF NOT EXISTS idx_stock_fundamentals_symbol ON stock_fundamentals(symbol);
