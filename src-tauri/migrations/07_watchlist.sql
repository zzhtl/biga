-- 用户收藏池（选票池）。symbol 统一存纯 6 位代码（与 fetch_universe 入库约定一致）；
-- 历史/股本/基本面查询时经 resolve_historical_symbol 解析库内变体（纯6位 / .SZ / .SH 混格）。
-- 指标不落库：全部展示指标（多周期涨跌幅/52周位置/PE/PB/基本面）每次由本地
-- historical_data / stock_capital / stock_fundamentals 现算——收藏池 ≤ 几十只，
-- 一条窗口查询毫秒级；避免"底表已刷新但缓存未失效"一类一致性 bug，
-- 且数据新鲜度天然等于最新 K 线日期。
CREATE TABLE IF NOT EXISTS watchlist (
    symbol     TEXT PRIMARY KEY,
    added_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    sort_order INTEGER NOT NULL DEFAULT 0
);
