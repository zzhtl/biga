-- 交易纪律：账户资金 / 持仓 / 成交流水 / 纪律事件。
-- 与 watchlist 同一思路「指标不落库」：浮盈、总资产、裁决全部由 positions + 最新 K 线现算，
-- 只有跨会话必须保留的事实（成本价、成交、棘轮止损价、违纪留痕）才入库，
-- 避免"底表已刷新但缓存未失效"一类一致性 bug。
-- symbol 统一存纯 6 位代码（写入前过 canonical_stock_symbol）。
--
-- 不写 REFERENCES：create_pool() 未开 PRAGMA foreign_keys，写外键等于"看起来有约束实际没有"，
-- 关联完整性由命令层事务保证。position_id / event_id 的取值约束见 commands/discipline.rs。

-- 账户资金与规则配置。单行表，id 固定为 1。
-- 只存 cash：总资产 = cash + Σ(持仓数量 × 最新收盘价)，每次现算。
CREATE TABLE IF NOT EXISTS discipline_account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash REAL NOT NULL DEFAULT 0,               -- 可用现金（元），由成交流水自动增减
    rules_json TEXT NOT NULL DEFAULT '',        -- DisciplineRules 序列化；空串 = 用出厂默认值
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO discipline_account (id, cash, rules_json) VALUES (1, 0, '');

-- 持仓。一只股票同时最多一条 open 记录（加仓合并进同一条并重算加权成本），
-- 由下方的部分唯一索引强制。
CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY,                        -- uuid v4
    symbol TEXT NOT NULL,                       -- 纯 6 位
    status TEXT NOT NULL DEFAULT 'open',        -- open / closed
    open_date TEXT NOT NULL,                    -- 建仓日 YYYY-MM-DD
    close_date TEXT,                            -- 清仓日，未清仓为 NULL
    cost_price REAL NOT NULL,                   -- 加权平均成本；部分卖出不变，加仓时重算
    quantity INTEGER NOT NULL,                  -- 当前持股数
    initial_quantity INTEGER NOT NULL,          -- 建仓股数，复盘对照用
    initial_stop REAL NOT NULL,                 -- 建仓登记的止损价，永不修改，只作复盘基准
    stop_price REAL NOT NULL,                   -- 当前生效止损价，棘轮：只上移不下移
    stop_basis TEXT NOT NULL DEFAULT 'fixed',   -- 当前止损由哪个候选胜出：fixed / atr / support / trailing
    target_price REAL,                          -- 建仓登记的目标价
    highest_price REAL,                         -- 持仓期最高价（用日线 high，每次全量重算）；窗口不足为 NULL
    highest_price_date TEXT,                    -- 最高价出现日，供复盘核对
    scale_out_done INTEGER NOT NULL DEFAULT 0,  -- 已执行的分批止盈档数
    realized_pnl REAL NOT NULL DEFAULT 0,       -- 已实现盈亏（已扣手续费）
    basis_suspect INTEGER NOT NULL DEFAULT 0,   -- 1 = 检测到价格基准变化（除权/除息），卖出规则挂起
    note TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_positions_open_unique ON positions(symbol) WHERE status = 'open';

-- 成交流水。每笔买卖一行，是 positions 各字段的事实来源。
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,                        -- uuid v4
    position_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,                         -- buy / sell
    price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    trade_date TEXT NOT NULL,                   -- YYYY-MM-DD
    fee REAL NOT NULL DEFAULT 0,                -- 手续费 + 印花税，人工填
    rule_code TEXT,                             -- 触发本次成交的纪律码；主动操作为 NULL
    event_id TEXT,                              -- 关联 discipline_events.id：这笔成交是在执行哪条裁决
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_position ON trades(position_id, trade_date);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_date ON trades(symbol, trade_date);

-- 纪律事件。裁决命中即入库，用户必须处理：执行(complied) 或 不执行并填理由(violated)。
-- 违纪留痕是本模块存在的意义——reason 在 violated 时由命令层强制校验非空。
CREATE TABLE IF NOT EXISTS discipline_events (
    id TEXT PRIMARY KEY,
    position_id TEXT,                           -- 买入否决类事件无持仓，为 NULL
    symbol TEXT NOT NULL,
    event_date TEXT NOT NULL,                   -- 触发日（对应 K 线日期）YYYY-MM-DD
    rule_code TEXT NOT NULL,
    severity TEXT NOT NULL,                     -- low / medium / high
    action_required TEXT NOT NULL,              -- exit_all / reduce_half / blocked_entry
    resolution TEXT NOT NULL DEFAULT 'pending', -- pending / complied / violated
    reason TEXT,                                -- resolution = violated 时必填
    trigger_close REAL NOT NULL,                -- 触发时收盘价，供复盘核对
    evidence_json TEXT NOT NULL DEFAULT '[]',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_discipline_events_pending ON discipline_events(resolution, event_date);
CREATE UNIQUE INDEX IF NOT EXISTS idx_discipline_events_dedup ON discipline_events(position_id, rule_code, event_date);
