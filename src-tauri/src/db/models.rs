//! 数据模型定义

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

// =============================================================================
// 股票基本信息
// =============================================================================

/// 股票基本信息
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct StockInfo {
    #[sqlx(rename = "symbol")]
    pub symbol: String,
    pub name: String,
    pub exchange: String,
}

/// 股票基本信息（API响应格式）
#[derive(Debug, Serialize, Deserialize)]
pub struct StockInfoItem {
    #[serde(rename = "dm")]
    pub symbol: String,
    #[serde(rename = "mc")]
    pub name: String,
    #[serde(rename = "jys")]
    pub exchange: String,
}

impl From<StockInfoItem> for StockInfo {
    fn from(item: StockInfoItem) -> Self {
        Self {
            symbol: item.symbol,
            name: item.name,
            exchange: item.exchange,
        }
    }
}

/// 股票详细信息
#[derive(Default, Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Stock {
    #[sqlx(rename = "symbol")]
    pub symbol: String,
    pub name: String,
    pub area: String,
    pub industry: String,
    pub market: String,
    #[serde(rename = "ts_code")]
    #[sqlx(rename = "exchange")]
    pub exchange: String,
    pub list_date: String,
    pub act_name: String,
    pub act_ent_type: String,
    /// 人工板块分类（科技/能源/矿业/电力/能源金属/消费/半导体/军工汽车/医药生物），
    /// 由龙头清单写入，用于股票列表页分组。CSV 导入路径无此列，故 serde 默认空串。
    #[serde(default)]
    #[sqlx(default)]
    pub category: String,
}

// =============================================================================
// 历史数据
// =============================================================================

/// 股票历史数据
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct HistoricalData {
    #[sqlx(rename = "symbol")]
    pub symbol: String,
    #[sqlx(rename = "date")]
    pub date: NaiveDate,
    pub open: f64,
    pub close: f64,
    pub high: f64,
    pub low: f64,
    pub volume: i64,
    pub amount: f64,
    pub amplitude: f64,
    /// 换手率（%）= 成交额 / 流通市值 × 100，回填得到
    pub turnover_rate: f64,
    /// 量比 = 当日成交量 / 过去 N 日平均成交量，回填得到
    #[sqlx(default)]
    pub volume_ratio: f64,
    pub change_percent: f64,
    pub change: f64,
}

/// 历史数据（API响应格式）
#[derive(Debug, Deserialize)]
pub struct HistoricalDataItem {
    #[serde(rename = "t")]
    pub date: String,
    #[serde(rename = "o")]
    pub open: f64,
    #[serde(rename = "h")]
    pub high: f64,
    #[serde(rename = "l")]
    pub low: f64,
    #[serde(rename = "c")]
    pub close: f64,
    #[serde(rename = "v")]
    pub volume: f64,
    #[serde(rename = "a")]
    pub amount: f64,
    #[serde(rename = "pc")]
    pub pre_close: f64,
}

impl HistoricalDataItem {
    /// 转换为历史数据模型
    pub fn to_historical_data(&self, symbol: &str) -> Option<HistoricalData> {
        let date = chrono::NaiveDate::parse_from_str(&self.date, "%Y-%m-%d").ok()?;
        let change = self.close - self.pre_close;
        let change_percent = if self.pre_close > 0.0 {
            change / self.pre_close * 100.0
        } else {
            0.0
        };
        let amplitude = if self.pre_close > 0.0 {
            (self.high - self.low) / self.pre_close * 100.0
        } else {
            0.0
        };

        Some(HistoricalData {
            symbol: symbol.to_string(),
            date,
            open: self.open,
            close: self.close,
            high: self.high,
            low: self.low,
            volume: (self.volume * 100.0) as i64, // 手转股
            amount: self.amount,
            amplitude,
            turnover_rate: 0.0, // 由 backfill_volume_metrics 回填
            volume_ratio: 0.0,  // 由 backfill_volume_metrics 回填
            change_percent,
            change,
        })
    }
}

// =============================================================================
// 实时数据
// =============================================================================

/// 实时行情数据
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct RealtimeData {
    #[sqlx(rename = "symbol")]
    pub symbol: String,
    #[sqlx(rename = "name")]
    pub name: String,
    #[sqlx(rename = "date")]
    pub date: NaiveDate,
    pub close: f64,
    pub volume: i64,
    pub amount: f64,
    pub amplitude: f64,
    pub turnover_rate: f64,
    #[sqlx(default)]
    pub volume_ratio: f64,
    pub change_percent: f64,
    pub change: f64,
}

// =============================================================================
// 股本信息（量比/换手率计算所需）
// =============================================================================

/// 股本数据（来自实时接口 hs/real/ssjy）
#[derive(Debug, Clone, Default, Serialize, Deserialize, FromRow)]
pub struct StockCapital {
    pub symbol: String,
    /// 流通股本（股）= 流通市值 / 最新价
    pub circulating_shares: f64,
    /// 总股本（股）= 总市值 / 最新价
    pub total_shares: f64,
    /// 流通市值（元）
    pub circulating_market_cap: f64,
    /// 市盈率（来自 ssjy pe），非技术估值因子
    #[sqlx(default)]
    pub pe: f64,
    /// 市净率（来自 ssjy sjl），非技术估值因子
    #[sqlx(default)]
    pub pb: f64,
}

/// 实时行情接口（hs/real/ssjy）响应中与股本/量比/换手率相关的字段
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RealtimeQuoteItem {
    /// 流通市值（元）
    #[serde(rename = "lt", default)]
    pub circulating_market_cap: f64,
    /// 总市值（元）
    #[serde(rename = "sz", default)]
    pub total_market_cap: f64,
    /// 换手率（%）
    #[serde(rename = "hs", default)]
    pub turnover_rate: f64,
    /// 量比（%）
    #[serde(rename = "lb", default)]
    pub volume_ratio: f64,
    /// 市盈率
    #[serde(rename = "pe", default)]
    pub pe: f64,
    /// 市净率
    #[serde(rename = "sjl", default)]
    pub pb: f64,
}

// =============================================================================
// 预测模型相关
// =============================================================================

// =============================================================================
// 基本面财务指标（非技术数据，来自 zhitu hs/gs/cwzb）
// =============================================================================

/// 单个报告期的基本面财务指标。值缺失（接口返回 "--"）为 None。
#[derive(Debug, Clone, Default, Serialize, Deserialize, FromRow)]
pub struct StockFundamental {
    pub symbol: String,
    /// 报告期(季度末)，如 2026-03-31
    pub report_date: String,
    /// 每股收益(YTD累计口径)
    pub eps: Option<f64>,
    /// 每股净资产
    pub bps: Option<f64>,
    /// 净资产收益率(%)
    pub roe: Option<f64>,
    /// 净利润增长率(%)
    pub profit_growth: Option<f64>,
    /// 主营收入增长率(%)
    pub revenue_growth: Option<f64>,
    /// 资产负债率(%)
    pub debt_ratio: Option<f64>,
}

/// 预测模型信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionModelInfo {
    pub id: String,
    pub name: String,
    pub stock_code: String,
    pub created_at: u64,
    pub model_type: String,
    pub features: Vec<String>,
    pub target: String,
    pub prediction_days: usize,
    pub accuracy: f64,
}

// =============================================================================
// 交易纪律
// =============================================================================

/// 账户资金与规则配置（`discipline_account` 单行表，id 固定为 1）。
///
/// 只存 `cash`：总资产 = cash + Σ 持仓市值，每次现算不落库
/// （与 `07_watchlist.sql` 的「指标不落库」同一思路）。
#[derive(Debug, Clone, Default, Serialize, Deserialize, FromRow)]
pub struct DisciplineAccount {
    /// 可用现金（元），由成交流水自动增减
    pub cash: f64,
    /// `DisciplineRules` 序列化；空串表示用出厂默认值
    pub rules_json: String,
    pub updated_at: String,
}

/// 持仓。一只股票同时最多一条 open 记录（由部分唯一索引强制）。
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Position {
    pub id: String,
    pub symbol: String,
    /// open / closed
    pub status: String,
    pub open_date: String,
    pub close_date: Option<String>,
    /// 加权平均成本。**部分卖出不摊薄**，理由见 `commands/discipline.rs::apply_trade`
    pub cost_price: f64,
    pub quantity: i64,
    pub initial_quantity: i64,
    /// 建仓登记的止损价，永不修改，只作复盘基准
    pub initial_stop: f64,
    /// 当前生效止损价，棘轮：只上移不下移
    pub stop_price: f64,
    pub stop_basis: String,
    pub target_price: Option<f64>,
    /// 持仓期最高价（日线 high，每次扫描全量重算）
    pub highest_price: Option<f64>,
    pub highest_price_date: Option<String>,
    pub scale_out_done: i64,
    pub realized_pnl: f64,
    /// 1 = 检测到价格基准变化（除权/除息），卖出规则挂起
    pub basis_suspect: i64,
    pub note: Option<String>,
}

/// 成交流水（append-only），是 `Position` 各字段的事实来源
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Trade {
    pub id: String,
    pub position_id: String,
    pub symbol: String,
    /// buy / sell
    pub side: String,
    pub price: f64,
    pub quantity: i64,
    pub trade_date: String,
    /// 手续费 + 印花税，人工填
    pub fee: f64,
    /// 触发本次成交的纪律码；主动操作为 NULL
    pub rule_code: Option<String>,
    /// 关联 `DisciplineEvent::id`：这笔成交在执行哪条裁决
    pub event_id: Option<String>,
}

/// 纪律事件 + 处置留痕。`resolution = 'violated'` 时 `reason` 由命令层强制校验非空。
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct DisciplineEvent {
    pub id: String,
    pub position_id: Option<String>,
    pub symbol: String,
    /// 触发所依据的 K 线日期，不是系统时间 —— 事件可重放
    pub event_date: String,
    pub rule_code: String,
    pub severity: String,
    /// exit_all / reduce_half / blocked_entry
    pub action_required: String,
    /// pending / complied / violated
    pub resolution: String,
    pub reason: Option<String>,
    pub trigger_close: f64,
    pub evidence_json: String,
    pub created_at: String,
    pub resolved_at: Option<String>,
}
