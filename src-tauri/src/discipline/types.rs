//! 纪律引擎的输入 / 输出类型。
//!
//! 裁决单元 [`DisciplineItem`] 刻意复刻 `prediction/analysis/risk_warning.rs` 的
//! `code / category / severity / evidence` 骨架：稳定 code 供前端去重筛选，
//! evidence 装可核验的数值。**照抄结构而不复用代码**，避免让 prediction 层为
//! discipline 改签名（依赖必须单向）。

use serde::{Deserialize, Serialize};

use crate::prediction::types::RiskLevel;

// =============================================================================
// 裁决单元
// =============================================================================

/// 规则所属维度
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisciplineCategory {
    /// 卖出纪律
    Exit,
    /// 买入准入
    Entry,
    /// 头寸规模
    Sizing,
    /// 组合层约束（仓位上限 / 冷静期 / 熔断）
    Portfolio,
    /// 数据可用性
    Data,
}

/// 裁决要求的动作。语义强度递减：MustSell > MustReduce > Blocked > Warn。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisciplineAction {
    /// 只提示，不要求动作
    Warn,
    /// 拒绝买入
    Blocked,
    /// 必须减仓
    MustReduce,
    /// 必须清仓
    MustSell,
}

impl DisciplineAction {
    pub fn label(self) -> &'static str {
        match self {
            Self::Warn => "提示",
            Self::Blocked => "拒绝买入",
            Self::MustReduce => "必须减仓",
            Self::MustSell => "必须清仓",
        }
    }

    /// 是否需要用户处理（进 discipline_events 表并触发置顶横幅）
    pub fn requires_resolution(self) -> bool {
        matches!(self, Self::MustSell | Self::MustReduce)
    }
}

/// 单条纪律命中
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisciplineItem {
    /// 稳定码，供前端去重 / 筛选 / 排序，以及违纪统计归类
    pub code: String,
    pub category: DisciplineCategory,
    pub severity: RiskLevel,
    pub action: DisciplineAction,
    pub title: String,
    pub detail: String,
    /// 可核验的数值证据
    pub evidence: Vec<String>,
    /// 触发价（通常是收盘价）
    pub trigger_price: Option<f64>,
    /// 阈值价（止损价 / 支撑位 / 分批止盈线）
    pub threshold_price: Option<f64>,
    /// 分批止盈建议减仓股数
    pub suggested_quantity: Option<i64>,
}

pub(crate) fn item(
    code: &str,
    category: DisciplineCategory,
    severity: RiskLevel,
    action: DisciplineAction,
    title: &str,
    detail: &str,
    evidence: Vec<String>,
) -> DisciplineItem {
    DisciplineItem {
        code: code.to_string(),
        category,
        severity,
        action,
        title: title.to_string(),
        detail: detail.to_string(),
        evidence,
        trigger_price: None,
        threshold_price: None,
        suggested_quantity: None,
    }
}

impl DisciplineItem {
    pub(crate) fn at(mut self, trigger: f64, threshold: f64) -> Self {
        self.trigger_price = Some(trigger);
        self.threshold_price = Some(threshold);
        self
    }

    pub(crate) fn with_quantity(mut self, quantity: i64) -> Self {
        self.suggested_quantity = Some(quantity);
        self
    }
}

/// 按 code 去重后追加（复刻 `risk_warning::push_warning`）
pub(crate) fn push_item(items: &mut Vec<DisciplineItem>, incoming: DisciplineItem) {
    if !items.iter().any(|existing| existing.code == incoming.code) {
        items.push(incoming);
    }
}

/// 按 severity 降序、code 升序排序，并返回整体等级（复刻 `risk_warning::finalize_summary`）
pub(crate) fn finalize(items: &mut [DisciplineItem]) -> RiskLevel {
    items.sort_by(|a, b| b.severity.cmp(&a.severity).then_with(|| a.code.cmp(&b.code)));
    items
        .iter()
        .map(|entry| entry.severity)
        .max()
        .unwrap_or(RiskLevel::Low)
}

/// 全部命中中最强的动作要求
pub(crate) fn strongest_action(items: &[DisciplineItem]) -> DisciplineAction {
    items
        .iter()
        .map(|entry| entry.action)
        .max()
        .unwrap_or(DisciplineAction::Warn)
}

// =============================================================================
// 止损
// =============================================================================

/// 当前生效止损由哪个候选胜出
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopBasis {
    /// 成本 × (1 − 固定百分比)
    Fixed,
    /// 成本 − N × ATR
    Atr,
    /// 最近支撑位下方
    Support,
    /// 持仓最高价回撤（移动止盈）
    Trailing,
}

impl StopBasis {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fixed => "固定百分比",
            Self::Atr => "ATR 波动",
            Self::Support => "支撑位",
            Self::Trailing => "移动止盈",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Atr => "atr",
            Self::Support => "support",
            Self::Trailing => "trailing",
        }
    }

    pub fn parse(raw: &str) -> Self {
        match raw {
            "atr" => Self::Atr,
            "support" => Self::Support,
            "trailing" => Self::Trailing,
            _ => Self::Fixed,
        }
    }
}

/// 止损计算的全部结果，四个候选都保留以便前端展示"为什么是这个价"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopComputation {
    pub stop_price: f64,
    pub basis: StopBasis,
    pub candidate_fixed: f64,
    pub candidate_atr: Option<f64>,
    pub candidate_support: Option<f64>,
    pub candidate_trailing: Option<f64>,
    /// true = 本次算出的候选低于现有止损，被棘轮挡住
    pub ratcheted: bool,
    /// 相对上次生效止损的上移量，`None` = 未上移
    pub raised_to: Option<f64>,
    pub evidence: Vec<String>,
}

// =============================================================================
// 引擎输入
// =============================================================================

/// 持仓快照。全部字段来自 positions 表，无任何预测量。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSnapshot {
    pub position_id: String,
    pub symbol: String,
    pub open_date: String,
    /// 当前持股数
    pub quantity: i64,
    /// 加权平均成本。**卖出不摊薄**，理由见 `commands/discipline.rs` 的 `apply_trade`
    pub cost_price: f64,
    /// 建仓登记的止损价，永不修改，只作复盘基准
    pub initial_stop: f64,
    /// 当前生效止损价（棘轮后）
    pub stop_price: f64,
    pub stop_basis: StopBasis,
    pub target_price: Option<f64>,
    /// 持仓期最高价（日线 high）。窗口不足为 None，此时移动止盈不参与。
    pub highest_price: Option<f64>,
    pub highest_price_date: Option<String>,
    /// 已执行的分批止盈档数
    pub scale_out_done: i32,
    /// true = 检测到价格基准变化（除权/除息），卖出规则降级
    pub basis_suspect: bool,
}

/// 市场事实。全部从 historical_data 现算，逐字段可核验。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketFacts {
    pub symbol: String,
    /// 最新已收盘 K 线日期
    pub latest_date: String,
    /// 距今自然日
    pub staleness_days: i64,
    /// 触发口径：一律用收盘价，不用盘中价、不用 realtime 快照
    pub close: f64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub prev_close: f64,
    /// 0.0 = 未回填
    pub volume_ratio: f64,
    pub change_percent: f64,
    /// `calculate_atr` 返回 0.0（长度不足）时映射为 None
    pub atr: Option<f64>,
    pub atr_percent: Option<f64>,
    /// 收盘价下方最近的支撑位（用于计算止损候选）
    pub nearest_support: Option<f64>,
    /// 收盘价上方最近的支撑位 = 刚刚被跌破的那一条（用于破位判定）
    pub broken_support: Option<f64>,
    /// MA20，长度不足为 None
    pub ma20: Option<f64>,
    /// 持有交易日数 = 建仓日起的 K 线根数，不是自然日
    pub holding_trading_days: i64,
    /// 持仓期最高价（日线 high，每次全量重算）。窗口不足或无建仓日为 None。
    pub holding_high: Option<f64>,
    pub holding_high_date: Option<String>,
    pub limit_up_percent: f64,
    pub limit_down_percent: f64,
    /// 一字跌停：最高 == 最低 且贴跌停
    pub is_limit_down_locked: bool,
    pub is_limit_up_locked: bool,
    pub bars_used: usize,
    /// 疑似发生除权/除息的日期
    pub suspected_corporate_action: Option<String>,
}

/// 组合状态：账户级约束的输入
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PortfolioState {
    /// 总资产 = 现金 + Σ 持仓市值，现算不落库
    pub total_equity: f64,
    pub cash: f64,
    /// 全部 open 持仓按最新收盘计的市值
    pub total_market_value: f64,
    pub open_position_count: usize,
    /// 目标票已持市值，0 = 新开仓
    pub symbol_market_value: f64,
    /// Some = 加仓场景
    pub symbol_cost_price: Option<f64>,
    /// 熔断窗口内的连续止损次数
    pub consecutive_losses: i64,
    /// 最近一次止损离场的交易日距今多少个交易日；None = 从未止损
    pub bars_since_last_stop_out: Option<i64>,
    /// 最近一次亏损平仓距今多少个交易日
    pub bars_since_last_loss: Option<i64>,
}

/// 拟买入候选：用户自己填的决定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryCandidate {
    pub symbol: String,
    pub entry_price: f64,
    /// None = 让引擎给最大可买
    pub intended_quantity: Option<i64>,
    /// 必填，缺失即 Blocked —— 强制"先想好怎么输"
    pub stop_price: Option<f64>,
    /// 必填，缺失即 Blocked
    pub target_price: Option<f64>,
    /// 透传 `risk_summary.level`，不重算。只用于否决，不用于放行。
    pub risk_level: Option<RiskLevel>,
}

// =============================================================================
// 引擎输出
// =============================================================================

/// 持仓的可核验指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionMetrics {
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub unrealized_pnl_percent: f64,
    /// 距生效止损还有多少（%），负数 = 已跌破
    pub distance_to_stop_percent: f64,
    /// 从持仓最高价的回撤（%）
    pub drawdown_from_high_percent: Option<f64>,
    /// 按生效止损计的剩余风险敞口（元）
    pub risk_exposure: f64,
    pub holding_trading_days: i64,
}

/// 卖出侧裁决
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitVerdict {
    pub position_id: String,
    pub symbol: String,
    /// = `MarketFacts::latest_date`，不是系统时间 —— 事件可重放
    pub event_date: String,
    pub level: RiskLevel,
    pub level_label: String,
    pub action: DisciplineAction,
    pub effective_stop: f64,
    pub stop_basis: StopBasis,
    /// Some = 本次扫描止损应上移到该价
    pub stop_raised_to: Option<f64>,
    /// 持仓最高价的重算结果，命令层据此回写
    pub highest_price: Option<f64>,
    pub highest_price_date: Option<String>,
    pub metrics: PositionMetrics,
    /// 建仓当日：T+1 不可卖
    pub t1_locked: bool,
    /// false = 停牌 / 一字跌停，全部强制动作降级为提示
    pub tradable: bool,
    pub items: Vec<DisciplineItem>,
}

/// 头寸规模的计算过程，四个上限都保留以便说明"为什么只能买这么多"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSizing {
    pub risk_per_share: f64,
    /// 单笔风险预算 = 总资产 × max_risk_pct
    pub risk_budget: f64,
    pub cap_by_risk: i64,
    pub cap_by_single_position: i64,
    pub cap_by_total_position: i64,
    pub cap_by_cash: i64,
    /// 哪一条卡住了
    pub binding_constraint: SizingConstraint,
    /// 已按手数取整
    pub max_shares: i64,
    pub max_amount: f64,
    /// 最小交易单位（主板 100，科创板 200）
    pub min_lot: i64,
    /// 递增步长（主板 100，科创板 1）
    pub lot_step: i64,
    /// true = 一手都买不起，诚实拒绝，不凑合
    pub below_one_lot: bool,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SizingConstraint {
    Risk,
    SinglePosition,
    TotalPosition,
    Cash,
}

impl SizingConstraint {
    pub fn label(self) -> &'static str {
        match self {
            Self::Risk => "单笔风险预算",
            Self::SinglePosition => "单票仓位上限",
            Self::TotalPosition => "总仓位上限",
            Self::Cash => "可用现金",
        }
    }
}

/// 买入准入裁决
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryVerdict {
    pub symbol: String,
    /// 全部 Blocked 项为空 → true。语义是「没有违反纪律」，不是「建议买入」。
    pub allowed: bool,
    pub level: RiskLevel,
    pub level_label: String,
    pub sizing: PositionSizing,
    /// 真实盈亏比 =（目标价 − 买入价）/（买入价 − 止损价）
    pub reward_risk_ratio: Option<f64>,
    pub items: Vec<DisciplineItem>,
    /// 后端固定下发，前端必须展示
    pub disclaimer: String,
}

/// 前端必须原样展示的口径声明
pub const ENTRY_DISCLAIMER: &str =
    "「未发现纪律冲突」只表示这笔买入没有违反你自己设定的资金管理规则，不构成任何买入建议，\
     也不代表方向判断。方向不可预测，纪律只负责让你亏得起。";
