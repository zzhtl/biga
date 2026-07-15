-- 股票代码统一为纯 6 位，并按各表业务主键合并历史重复数据。
-- 兼容 002466、002466.SZ、sz002466 三种格式，可重复执行。

-- 股票信息优先保留有效名称。同为有效名称时，带交易所的 API 数据优先于旧的纯代码数据。
DELETE FROM stock_info
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_stock_info_canonical_symbol')
  AND rowid IN (
    SELECT row_id FROM (
        SELECT rowid AS row_id,
               ROW_NUMBER() OVER (
                   PARTITION BY CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
                   ORDER BY CASE
                                WHEN TRIM(name) <> ''
                                 AND TRIM(name) <> TRIM(symbol)
                                 AND TRIM(name) <> CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
                                THEN 0 ELSE 1
                            END,
                            LENGTH(TRIM(symbol)) DESC,
                            rowid DESC
               ) AS duplicate_rank
        FROM stock_info
    ) AS ranked
    WHERE duplicate_rank > 1
);

UPDATE stock_info
SET symbol = CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END,
    exchange = LOWER(TRIM(exchange))
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_stock_info_canonical_symbol')
  AND (symbol <> CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
       OR exchange <> LOWER(TRIM(exchange)));

-- 股票详情按代码去重，优先保留已经规范化的行。
DELETE FROM stock
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_stock_info_canonical_symbol')
  AND rowid IN (
    SELECT row_id FROM (
        SELECT rowid AS row_id,
               ROW_NUMBER() OVER (
                   PARTITION BY CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
                   ORDER BY LENGTH(TRIM(symbol)), rowid
               ) AS duplicate_rank
        FROM stock
    ) AS ranked
    WHERE duplicate_rank > 1
);

UPDATE stock
SET symbol = CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END,
    exchange = LOWER(TRIM(exchange))
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_stock_info_canonical_symbol')
  AND (symbol <> CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
       OR exchange <> LOWER(TRIM(exchange)));

-- K 线按代码和日期合并，重叠日期优先保留规范代码行，非重叠日期全部保留。
DELETE FROM historical_data
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_stock_info_canonical_symbol')
  AND rowid IN (
    SELECT row_id FROM (
        SELECT rowid AS row_id,
               ROW_NUMBER() OVER (
                   PARTITION BY CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END, date
                   ORDER BY LENGTH(TRIM(symbol)), rowid
               ) AS duplicate_rank
        FROM historical_data
    ) AS ranked
    WHERE duplicate_rank > 1
);

UPDATE historical_data
SET symbol = CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_stock_info_canonical_symbol')
  AND symbol <> CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END;

-- 实时行情与股本快照保留更新时间最新的行。
DELETE FROM realtime_data
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_stock_info_canonical_symbol')
  AND rowid IN (
    SELECT row_id FROM (
        SELECT rowid AS row_id,
               ROW_NUMBER() OVER (
                   PARTITION BY CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
                   ORDER BY date DESC, LENGTH(TRIM(symbol)), rowid DESC
               ) AS duplicate_rank
        FROM realtime_data
    ) AS ranked
    WHERE duplicate_rank > 1
);

UPDATE realtime_data
SET symbol = CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_stock_info_canonical_symbol')
  AND symbol <> CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END;

DELETE FROM stock_capital
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_stock_info_canonical_symbol')
  AND rowid IN (
    SELECT row_id FROM (
        SELECT rowid AS row_id,
               ROW_NUMBER() OVER (
                   PARTITION BY CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
                   ORDER BY updated_at DESC, LENGTH(TRIM(symbol)), rowid DESC
               ) AS duplicate_rank
        FROM stock_capital
    ) AS ranked
    WHERE duplicate_rank > 1
);

UPDATE stock_capital
SET symbol = CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_stock_info_canonical_symbol')
  AND symbol <> CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END;

-- 基本面按代码和报告期合并，保留更新时间最新的行。
DELETE FROM stock_fundamentals
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_stock_info_canonical_symbol')
  AND rowid IN (
    SELECT row_id FROM (
        SELECT rowid AS row_id,
               ROW_NUMBER() OVER (
                   PARTITION BY CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END, report_date
                   ORDER BY updated_at DESC, LENGTH(TRIM(symbol)), rowid DESC
               ) AS duplicate_rank
        FROM stock_fundamentals
    ) AS ranked
    WHERE duplicate_rank > 1
);

UPDATE stock_fundamentals
SET symbol = CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_stock_info_canonical_symbol')
  AND symbol <> CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END;

-- 收藏重复时保留排序更靠前、添加时间更早的行。
DELETE FROM watchlist
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_stock_info_canonical_symbol')
  AND rowid IN (
    SELECT row_id FROM (
        SELECT rowid AS row_id,
               ROW_NUMBER() OVER (
                   PARTITION BY CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
                   ORDER BY sort_order, added_at, LENGTH(TRIM(symbol)), rowid
               ) AS duplicate_rank
        FROM watchlist
    ) AS ranked
    WHERE duplicate_rank > 1
);

UPDATE watchlist
SET symbol = CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_stock_info_canonical_symbol')
  AND symbol <> CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END;

-- 即使绕过 Rust 写入层，纯代码、后缀代码和前缀代码也不能再次形成逻辑重复。
CREATE UNIQUE INDEX IF NOT EXISTS uq_stock_info_canonical_symbol ON stock_info (
    CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_stock_canonical_symbol ON stock (
    CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_historical_data_canonical_symbol_date ON historical_data (
    CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END,
    date
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_realtime_data_canonical_symbol ON realtime_data (
    CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_stock_capital_canonical_symbol ON stock_capital (
    CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_stock_fundamentals_canonical_symbol_report_date ON stock_fundamentals (
    CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END,
    report_date
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_watchlist_canonical_symbol ON watchlist (
    CASE WHEN LOWER(SUBSTR(TRIM(symbol), 1, 2)) IN ('sh', 'sz', 'bj') THEN SUBSTR(TRIM(symbol), 3, 6) ELSE SUBSTR(TRIM(symbol), 1, 6) END
);
