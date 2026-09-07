//! 交易纪律命令层。
//!
//! 职责边界：**取数 → 调纯函数 → 落库**。所有阈值判定都在 `crate::discipline` 的纯函数里，
//! 本模块不做任何"是否触发"的判断，否则规则会有两套实现，六个月后必然漂移——
//! 而且漂移的方向一定是让人心里舒服的那个方向。
//!
//! ⚠️ 与 `watchlist.rs` 一致的诚实原则：本模块**不新增任何方向规则**。纪律裁决的输入
//! 全是客观事实（成本价、收盘价、持仓最高价、ATR、支撑位、持有交易日数、账户资金）。
//! 买入准入里的 `risk_level` 是原样透传 `analyze_prediction_risk` 的既有输出，
//! 且只用于否决，不用于放行。
//!
//! # 「强制填理由」是后端硬约束
//!
//! 把纪律做成前端 `if` 判断等于没做。三处校验都在这里：
//! 1. [`resolve_discipline_event`] 违纪时理由须 ≥ [`MIN_REASON_CHARS`] 字
//! 2. [`open_discipline_position`] 内部复跑准入，被拦时无理由直接拒绝写库
//! 3. 违纪理由入库后不可撤销，复盘页会把它和最终差额并排展示

use chrono::{Local, NaiveDate};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tauri::State;
use uuid::Uuid;

use crate::db::models::{DisciplineEvent, HistoricalData, Position, Trade};
use crate::db::repository;
use crate::discipline::entry::screen_entry;
use crate::discipline::exit::evaluate_exit;
use crate::discipline::facts::build_market_facts;
use crate::discipline::replay::{replay_disciplined_exit, ReplayEntry, ReplayOutcome};
use crate::discipline::rules::DisciplineRules;
use crate::discipline::stop::{compute_stop, StopInput};
use crate::discipline::types::{
    DisciplineAction, EntryCandidate, EntryVerdict, ExitVerdict, MarketFacts, PortfolioState,
    PositionSnapshot, StopBasis, ENTRY_DISCLAIMER,
};
use crate::error::AppError;
use crate::prediction::types::RiskLevel;
use crate::utils::canonical_stock_symbol;

/// 违纪理由的最小字数。设成 10 是为了让人没法用「.」或「忘了」敷衍过去——
/// 写满 10 个字的过程本身就是一次复盘。
pub const MIN_REASON_CHARS: usize = 10;

/// 取数窗口：覆盖一年多的日线，够算 ATR、支撑位、MA20 与持仓最高价
const BARS_WINDOW: usize = 300;

/// 复盘回放时在实际清仓日之后多取的根数，供 T+1 成交与跌停顺延使用
const REPLAY_TAIL_BARS: i64 = 15;

const REVIEW_DISCLAIMER: &str = "「严格守纪」是同一笔建仓下的反事实估算，不是回测收益：\
     只回答「纪律会不会让你更早离场、离场价差多少」，不回答「如果拿得更久会怎样」——\
     后者需要先决定拿多久，而那本身就是一个策略。成交价一律取触发次日开盘价，\
     一字跌停与停牌按顺延处理，顺延超限的笔已从统计中剔除并单独计数。";

// =============================================================================
// 下发给前端的视图类型
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountView {
    pub cash: f64,
    /// 总资产 = 现金 + Σ 持仓市值，现算不落库
    pub total_equity: f64,
    pub total_market_value: f64,
    pub open_position_count: usize,
    pub rules: DisciplineRules,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionView {
    #[serde(flatten)]
    pub position: Position,
    pub name: String,
    pub last_close: Option<f64>,
    pub verdict: Option<ExitVerdict>,
    /// 数据不足时的说明，此时 verdict 为 None
    pub unavailable_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisciplineBoard {
    pub account: AccountView,
    pub positions: Vec<PositionView>,
    /// 待处理裁决，前端置顶红色横幅用
    pub pending_events: Vec<DisciplineEvent>,
    pub disclaimer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewRow {
    pub position_id: String,
    pub symbol: String,
    pub name: String,
    pub open_date: String,
    pub close_date: String,
    pub cost_price: f64,
    pub quantity: i64,
    pub outcome: ReplayOutcome,
    /// 这笔上发生过的违纪记录（含当时填的理由）
    pub violations: Vec<DisciplineEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleStat {
    pub rule_code: String,
    pub violated_count: usize,
    /// 该规则违纪笔的累计差额（守纪 − 实际）
    pub difference_total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisciplineReview {
    pub closed_count: usize,
    pub replayed_count: usize,
    /// 因跌停/停牌无法回放、已从统计剔除
    pub unresolved_count: usize,
    pub complied_count: usize,
    pub violated_count: usize,
    /// 守纪率 = 已执行 /（已执行 + 违纪）；分母为 0 时 None
    pub compliance_rate: Option<f64>,
    pub actual_pnl_total: f64,
    pub disciplined_pnl_total: f64,
    /// 守纪 − 实际。正数 = 严格守纪本可以少亏或多赚这么多
    pub difference_total: f64,
    /// false = 用户从未填过手续费，两边都按 0 计
    pub fee_recorded: bool,
    pub by_rule: Vec<RuleStat>,
    pub rows: Vec<ReviewRow>,
    pub disclaimer: String,
}

// =============================================================================
// 内部工具
// =============================================================================

fn severity_str(level: RiskLevel) -> &'static str {
    match level {
        RiskLevel::Low => "low",
        RiskLevel::Medium => "medium",
        RiskLevel::High => "high",
    }
}

fn action_str(action: DisciplineAction) -> &'static str {
    match action {
        DisciplineAction::MustSell => "exit_all",
        DisciplineAction::MustReduce => "reduce_half",
        DisciplineAction::Blocked => "blocked_entry",
        DisciplineAction::Warn => "warn",
    }
}

fn today() -> NaiveDate {
    Local::now().date_naive()
}

fn parse_date(raw: &str, field: &str) -> Result<NaiveDate, AppError> {
    NaiveDate::parse_from_str(raw.trim(), "%Y-%m-%d")
        .map_err(|_| AppError::InvalidInput(format!("{field}格式应为 YYYY-MM-DD，收到「{raw}」")))
}

async fn load_rules(pool: &SqlitePool) -> Result<DisciplineRules, AppError> {
    let account = repository::get_discipline_account(pool).await?;
    Ok(DisciplineRules::from_json(&account.rules_json).sanitized())
}

async fn stock_name(symbol: &str, pool: &SqlitePool) -> String {
    repository::get_stock_info(symbol, pool)
        .await
        .map(|info| info.name)
        .unwrap_or_else(|_| symbol.to_string())
}

async fn load_bars(symbol: &str, pool: &SqlitePool) -> Result<Vec<HistoricalData>, AppError> {
    repository::get_recent_historical_data(symbol, BARS_WINDOW, pool).await
}

fn snapshot_of(position: &Position) -> PositionSnapshot {
    PositionSnapshot {
        position_id: position.id.clone(),
        symbol: position.symbol.clone(),
        open_date: position.open_date.clone(),
        quantity: position.quantity,
        cost_price: position.cost_price,
        initial_stop: position.initial_stop,
        stop_price: position.stop_price,
        stop_basis: StopBasis::parse(&position.stop_basis),
        target_price: position.target_price,
        highest_price: position.highest_price,
        highest_price_date: position.highest_price_date.clone(),
        scale_out_done: position.scale_out_done as i32,
        basis_suspect: position.basis_suspect != 0,
    }
}

/// 组合状态：总资产、各类上限的占用、冷静期与熔断的输入。
///
/// `target_symbol` 是本次拟买入的标的（用于取"已持该票市值"与冷静期）；
/// 只看盘时传 None。
async fn build_portfolio_state(
    target_symbol: Option<&str>,
    pool: &SqlitePool,
) -> Result<PortfolioState, AppError> {
    let account = repository::get_discipline_account(pool).await?;
    let open_positions = repository::list_positions(Some("open"), pool).await?;

    let mut total_market_value = 0.0;
    let mut symbol_market_value = 0.0;
    let mut symbol_cost_price = None;
    let target = target_symbol.map(canonical_stock_symbol);

    for position in &open_positions {
        // 停牌当天没有新 K 线时退回最后一根收盘价；再取不到就用成本价，
        // 宁可高估已用仓位（更严），也不要把它当成 0
        let last_close = repository::get_latest_close_price(&position.symbol, pool)
            .await?
            .unwrap_or(position.cost_price);
        let value = last_close * position.quantity as f64;
        total_market_value += value;
        if target.as_deref() == Some(position.symbol.as_str()) {
            symbol_market_value = value;
            symbol_cost_price = Some(position.cost_price);
        }
    }

    // 连续亏损：按平仓日倒序数，遇到第一笔盈利即停
    let mut closed = repository::list_positions(Some("closed"), pool).await?;
    closed.sort_by(|a, b| b.close_date.cmp(&a.close_date));
    let mut consecutive_losses = 0i64;
    for position in &closed {
        if position.realized_pnl < 0.0 {
            consecutive_losses += 1;
        } else {
            break;
        }
    }

    let now = today().format("%Y-%m-%d").to_string();
    let bars_since_last_loss = match closed.iter().find(|p| p.realized_pnl < 0.0) {
        Some(position) => match &position.close_date {
            Some(date) => Some(repository::count_trading_days_between(date, &now, pool).await?),
            None => None,
        },
        None => None,
    };

    // 冷静期只看目标股：最近一次以亏损平仓的那笔
    let bars_since_last_stop_out = match &target {
        Some(symbol) => {
            let last_loss = closed
                .iter()
                .find(|p| p.symbol == *symbol && p.realized_pnl < 0.0)
                .and_then(|p| p.close_date.clone());
            match last_loss {
                Some(date) => {
                    Some(repository::count_trading_days_between(&date, &now, pool).await?)
                }
                None => None,
            }
        }
        None => None,
    };

    Ok(PortfolioState {
        total_equity: account.cash + total_market_value,
        cash: account.cash,
        total_market_value,
        open_position_count: open_positions.len(),
        symbol_market_value,
        symbol_cost_price,
        consecutive_losses,
        bars_since_last_stop_out,
        bars_since_last_loss,
    })
}

async fn build_facts_for(
    symbol: &str,
    open_date: Option<NaiveDate>,
    rules: &DisciplineRules,
    pool: &SqlitePool,
) -> Result<Option<MarketFacts>, AppError> {
    let bars = load_bars(symbol, pool).await?;
    if bars.is_empty() {
        return Ok(None);
    }
    let name = stock_name(symbol, pool).await;
    Ok(build_market_facts(symbol, &name, &bars, open_date, today(), rules))
}

/// 把一条裁决落成待处理事件。同持仓 + 同规则 + 同日靠唯一索引幂等去重。
async fn persist_verdict_events(
    verdict: &ExitVerdict,
    pool: &SqlitePool,
) -> Result<(), AppError> {
    for hit in verdict.items.iter().filter(|hit| hit.action.requires_resolution()) {
        let event = DisciplineEvent {
            id: Uuid::new_v4().to_string(),
            position_id: Some(verdict.position_id.clone()),
            symbol: verdict.symbol.clone(),
            event_date: verdict.event_date.clone(),
            rule_code: hit.code.clone(),
            severity: severity_str(hit.severity).to_string(),
            action_required: action_str(hit.action).to_string(),
            resolution: "pending".to_string(),
            reason: None,
            trigger_close: hit.trigger_price.unwrap_or(verdict.metrics.market_value),
            evidence_json: serde_json::to_string(&hit.evidence).unwrap_or_else(|_| "[]".to_string()),
            created_at: String::new(),
            resolved_at: None,
        };
        repository::insert_discipline_event(pool, &event).await?;
    }
    Ok(())
}

// =============================================================================
// 账户
// =============================================================================

pub(crate) async fn discipline_account_view(pool: &SqlitePool) -> Result<AccountView, AppError> {
    let account = repository::get_discipline_account(pool).await?;
    let portfolio = build_portfolio_state(None, pool).await?;
    Ok(AccountView {
        cash: account.cash,
        total_equity: portfolio.total_equity,
        total_market_value: portfolio.total_market_value,
        open_position_count: portfolio.open_position_count,
        rules: DisciplineRules::from_json(&account.rules_json).sanitized(),
        updated_at: account.updated_at,
    })
}

#[tauri::command]
pub async fn get_discipline_account(pool: State<'_, SqlitePool>) -> Result<AccountView, AppError> {
    discipline_account_view(&pool).await
}

#[tauri::command]
pub async fn save_discipline_account(
    cash: f64,
    rules: Option<DisciplineRules>,
    pool: State<'_, SqlitePool>,
) -> Result<AccountView, AppError> {
    if !cash.is_finite() || cash < 0.0 {
        return Err(AppError::InvalidInput("可用现金不能为负".to_string()));
    }
    let rules = match rules {
        Some(rules) => rules.sanitized(),
        None => load_rules(&pool).await?,
    };
    repository::save_discipline_account(&pool, cash, &rules.to_json()).await?;
    discipline_account_view(&pool).await
}

// =============================================================================
// 持仓看板（扫描 + 落事件 + 回写止损）
// =============================================================================

pub(crate) async fn scan_discipline_board(pool: &SqlitePool) -> Result<DisciplineBoard, AppError> {
    let rules = load_rules(pool).await?;
    let positions = repository::list_positions(Some("open"), pool).await?;
    let mut views = Vec::with_capacity(positions.len());

    for mut position in positions {
        let name = stock_name(&position.symbol, pool).await;
        let open_date = parse_date(&position.open_date, "建仓日")?;
        let bars = load_bars(&position.symbol, pool).await?;
        let last_close = bars.last().map(|bar| bar.close);

        let facts = if bars.is_empty() {
            None
        } else {
            build_market_facts(&position.symbol, &name, &bars, Some(open_date), today(), &rules)
        };

        let Some(facts) = facts else {
            views.push(PositionView {
                position,
                name,
                last_close,
                verdict: None,
                unavailable_reason: Some(
                    "本地没有该股历史行情，请先在历史数据页刷新后再评估纪律".to_string(),
                ),
            });
            continue;
        };

        let verdict = evaluate_exit(&snapshot_of(&position), &facts, &rules);

        // 回写：止损棘轮 + 全量重算的最高价 + 除权挂起标记
        position.stop_price = verdict.effective_stop;
        position.stop_basis = verdict.stop_basis.as_str().to_string();
        position.highest_price = verdict.highest_price;
        position.highest_price_date = verdict.highest_price_date.clone();
        position.basis_suspect = i64::from(facts.suspected_corporate_action.is_some());
        repository::update_position(pool, &position).await?;

        persist_verdict_events(&verdict, pool).await?;

        views.push(PositionView {
            position,
            name,
            last_close,
            verdict: Some(verdict),
            unavailable_reason: None,
        });
    }

    // 按严重度降序：要处理的排最前面
    views.sort_by(|a, b| {
        let rank = |view: &PositionView| {
            view.verdict
                .as_ref()
                .map(|v| (v.action, v.level))
                .unwrap_or((DisciplineAction::Warn, RiskLevel::Low))
        };
        rank(b).cmp(&rank(a))
    });

    Ok(DisciplineBoard {
        account: discipline_account_view(pool).await?,
        positions: views,
        pending_events: repository::list_discipline_events(Some("pending"), pool).await?,
        disclaimer: ENTRY_DISCLAIMER.to_string(),
    })
}

#[tauri::command]
pub async fn get_discipline_positions(
    pool: State<'_, SqlitePool>,
) -> Result<DisciplineBoard, AppError> {
    scan_discipline_board(&pool).await
}

// =============================================================================
// 买入准入
// =============================================================================

pub(crate) async fn check_buy_inner(
    symbol: &str,
    buy_price: f64,
    stop_price: Option<f64>,
    target_price: Option<f64>,
    quantity: Option<i64>,
    risk_level: Option<RiskLevel>,
    pool: &SqlitePool,
) -> Result<EntryVerdict, AppError> {
    let symbol = canonical_stock_symbol(symbol);
    if symbol.is_empty() {
        return Err(AppError::InvalidInput("股票代码不能为空".to_string()));
    }
    if !buy_price.is_finite() || buy_price <= 0.0 {
        return Err(AppError::InvalidInput("买入价必须大于 0".to_string()));
    }

    let rules = load_rules(pool).await?;
    let portfolio = build_portfolio_state(Some(&symbol), pool).await?;
    let facts = build_facts_for(&symbol, None, &rules, pool).await?.ok_or_else(|| {
        AppError::InvalidInput(
            "本地没有该股历史行情，无法核算支撑位与波动率，请先刷新历史数据".to_string(),
        )
    })?;

    let candidate = EntryCandidate {
        symbol,
        entry_price: buy_price,
        intended_quantity: quantity,
        stop_price,
        target_price,
        risk_level,
    };
    Ok(screen_entry(&candidate, &portfolio, &facts, &rules))
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn check_buy_discipline(
    symbol: String,
    buy_price: f64,
    stop_price: Option<f64>,
    target_price: Option<f64>,
    quantity: Option<i64>,
    risk_level: Option<RiskLevel>,
    pool: State<'_, SqlitePool>,
) -> Result<EntryVerdict, AppError> {
    check_buy_inner(&symbol, buy_price, stop_price, target_price, quantity, risk_level, &pool).await
}

// =============================================================================
// 建仓 / 成交
// =============================================================================

fn validate_reason(reason: Option<&str>, context: &str) -> Result<String, AppError> {
    let reason = reason.unwrap_or("").trim();
    if reason.chars().count() < MIN_REASON_CHARS {
        return Err(AppError::InvalidInput(format!(
            "{context}，必须填写至少 {MIN_REASON_CHARS} 个字的理由。写下来这件事本身就是复盘——\
             半年后你会需要看到当时的想法。"
        )));
    }
    Ok(reason.to_string())
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn open_position_inner(
    symbol: &str,
    price: f64,
    quantity: i64,
    stop_price: f64,
    target_price: f64,
    trade_date: &str,
    fee: f64,
    override_reason: Option<&str>,
    pool: &SqlitePool,
) -> Result<Position, AppError> {
    let symbol = canonical_stock_symbol(symbol);
    if quantity <= 0 {
        return Err(AppError::InvalidInput("买入股数必须大于 0".to_string()));
    }
    let trade_date_parsed = parse_date(trade_date, "成交日")?;
    if repository::get_open_position(&symbol, pool).await?.is_some() {
        return Err(AppError::InvalidInput(
            "该股已有未平仓持仓，请用「记录成交」加仓，而不是新建一笔".to_string(),
        ));
    }

    // 复跑准入。前端已经跑过一次，这里再跑是因为前端的判断不算数。
    let verdict = check_buy_inner(
        &symbol,
        price,
        Some(stop_price),
        Some(target_price),
        Some(quantity),
        None,
        pool,
    )
    .await?;

    let blocked: Vec<_> = verdict
        .items
        .iter()
        .filter(|hit| hit.action == DisciplineAction::Blocked)
        .collect();

    let mut violation_reason = None;
    if !blocked.is_empty() {
        let codes: Vec<&str> = blocked.iter().map(|hit| hit.code.as_str()).collect();
        let reason = validate_reason(
            override_reason,
            &format!("这笔买入触发了 {} 条纪律（{}）", blocked.len(), codes.join("、")),
        )?;
        violation_reason = Some(reason);
    }

    let position_id = Uuid::new_v4().to_string();
    let cost_price = (price * quantity as f64 + fee) / quantity as f64;

    // 建仓止损：用户填的价与引擎算的取更严者（更高者）。
    // 用户可以比引擎更严格，但不能更松——那正是纪律要拦的。
    let rules = load_rules(pool).await?;
    let facts = build_facts_for(&symbol, Some(trade_date_parsed), &rules, pool).await?;
    let computed = compute_stop(
        &StopInput {
            cost_price,
            close: price,
            previous_stop: None,
            previous_basis: None,
            atr: facts.as_ref().and_then(|f| f.atr),
            nearest_support: facts.as_ref().and_then(|f| f.nearest_support),
            highest_price: None,
        },
        &rules,
    );
    let effective_stop = stop_price.max(computed.stop_price);
    let basis = if effective_stop > computed.stop_price {
        StopBasis::Fixed
    } else {
        computed.basis
    };

    let position = Position {
        id: position_id.clone(),
        symbol: symbol.clone(),
        status: "open".to_string(),
        open_date: trade_date.trim().to_string(),
        close_date: None,
        cost_price,
        quantity,
        initial_quantity: quantity,
        initial_stop: effective_stop,
        stop_price: effective_stop,
        stop_basis: basis.as_str().to_string(),
        target_price: Some(target_price),
        highest_price: None,
        highest_price_date: None,
        scale_out_done: 0,
        realized_pnl: 0.0,
        basis_suspect: 0,
        note: None,
    };
    repository::insert_position(pool, &position).await?;

    repository::insert_trade(
        pool,
        &Trade {
            id: Uuid::new_v4().to_string(),
            position_id: position_id.clone(),
            symbol: symbol.clone(),
            side: "buy".to_string(),
            price,
            quantity,
            trade_date: trade_date.trim().to_string(),
            fee,
            rule_code: None,
            event_id: None,
        },
    )
    .await?;
    repository::adjust_discipline_cash(pool, -(price * quantity as f64 + fee)).await?;

    // 越权留痕：每条被无视的准入规则各记一条，理由存档
    if let Some(reason) = violation_reason {
        for hit in blocked {
            repository::insert_discipline_event(
                pool,
                &DisciplineEvent {
                    id: Uuid::new_v4().to_string(),
                    position_id: Some(position_id.clone()),
                    symbol: symbol.clone(),
                    event_date: trade_date.trim().to_string(),
                    rule_code: hit.code.clone(),
                    severity: severity_str(hit.severity).to_string(),
                    action_required: "blocked_entry".to_string(),
                    resolution: "violated".to_string(),
                    reason: Some(reason.clone()),
                    trigger_close: price,
                    evidence_json: serde_json::to_string(&hit.evidence)
                        .unwrap_or_else(|_| "[]".to_string()),
                    created_at: String::new(),
                    resolved_at: None,
                },
            )
            .await?;
        }
    }

    Ok(position)
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn open_discipline_position(
    symbol: String,
    price: f64,
    quantity: i64,
    stop_price: f64,
    target_price: f64,
    trade_date: String,
    fee: Option<f64>,
    override_reason: Option<String>,
    pool: State<'_, SqlitePool>,
) -> Result<Position, AppError> {
    open_position_inner(
        &symbol,
        price,
        quantity,
        stop_price,
        target_price,
        &trade_date,
        fee.unwrap_or(0.0),
        override_reason.as_deref(),
        &pool,
    )
    .await
}

/// 记录一笔成交并推进持仓状态机。
///
/// # 加权成本在部分卖出后不摊薄
///
/// 券商的「摊薄成本」口径会在盈利卖出后把剩余股份成本压低，于是
/// `成本 × (1 − 8%)` 跟着下移——正好违反「止损只上移」，还给你一个心理台阶：
/// 「这单已经赚了，可以再扛一点」。这是把盈利单扛成亏损单的标准路径。
/// 所以这里：**卖出只减股数、只累加已实现盈亏，成本价不动；只有买入才重算加权成本。**
#[allow(clippy::too_many_arguments)]
pub(crate) async fn record_trade_inner(
    position_id: &str,
    side: &str,
    price: f64,
    quantity: i64,
    trade_date: &str,
    fee: f64,
    rule_code: Option<&str>,
    event_id: Option<&str>,
    pool: &SqlitePool,
) -> Result<Position, AppError> {
    let mut position = repository::get_position(position_id, pool)
        .await?
        .ok_or_else(|| AppError::InvalidInput("找不到该持仓".to_string()))?;
    if position.status != "open" {
        return Err(AppError::InvalidInput("该持仓已平仓，无法再记录成交".to_string()));
    }
    if quantity <= 0 {
        return Err(AppError::InvalidInput("成交股数必须大于 0".to_string()));
    }
    if !price.is_finite() || price <= 0.0 {
        return Err(AppError::InvalidInput("成交价必须大于 0".to_string()));
    }
    parse_date(trade_date, "成交日")?;

    let cash_delta = match side {
        "buy" => {
            let new_quantity = position.quantity + quantity;
            position.cost_price = (position.cost_price * position.quantity as f64
                + price * quantity as f64
                + fee)
                / new_quantity as f64;
            position.quantity = new_quantity;
            position.initial_quantity += quantity;
            -(price * quantity as f64 + fee)
        }
        "sell" => {
            if quantity > position.quantity {
                return Err(AppError::InvalidInput(format!(
                    "卖出 {quantity} 股超过持仓 {} 股",
                    position.quantity
                )));
            }
            position.realized_pnl +=
                (price - position.cost_price) * quantity as f64 - fee;
            position.quantity -= quantity;
            if matches!(rule_code, Some("EXIT_SCALE_OUT_1") | Some("EXIT_SCALE_OUT_2")) {
                position.scale_out_done += 1;
            }
            if position.quantity == 0 {
                position.status = "closed".to_string();
                position.close_date = Some(trade_date.trim().to_string());
            }
            price * quantity as f64 - fee
        }
        other => {
            return Err(AppError::InvalidInput(format!(
                "成交方向只能是 buy 或 sell，收到「{other}」"
            )))
        }
    };

    // 加仓后重算止损：棘轮保证补跌摊薄不会把止损带下去
    if side == "buy" && position.status == "open" {
        let rules = load_rules(pool).await?;
        let facts = build_facts_for(
            &position.symbol,
            Some(parse_date(&position.open_date, "建仓日")?),
            &rules,
            pool,
        )
        .await?;
        let computed = compute_stop(
            &StopInput {
                cost_price: position.cost_price,
                close: facts.as_ref().map(|f| f.close).unwrap_or(price),
                previous_stop: Some(position.stop_price),
                previous_basis: Some(StopBasis::parse(&position.stop_basis)),
                atr: facts.as_ref().and_then(|f| f.atr),
                nearest_support: facts.as_ref().and_then(|f| f.nearest_support),
                highest_price: facts.as_ref().and_then(|f| f.holding_high),
            },
            &rules,
        );
        position.stop_price = computed.stop_price;
        position.stop_basis = computed.basis.as_str().to_string();
    }

    repository::update_position(pool, &position).await?;
    repository::insert_trade(
        pool,
        &Trade {
            id: Uuid::new_v4().to_string(),
            position_id: position_id.to_string(),
            symbol: position.symbol.clone(),
            side: side.to_string(),
            price,
            quantity,
            trade_date: trade_date.trim().to_string(),
            fee,
            rule_code: rule_code.map(str::to_string),
            event_id: event_id.map(str::to_string),
        },
    )
    .await?;
    repository::adjust_discipline_cash(pool, cash_delta).await?;

    // 这笔成交在执行某条裁决 → 把事件标记为已执行
    if let Some(event_id) = event_id {
        repository::resolve_discipline_event(pool, event_id, "complied", "").await?;
    }

    Ok(position)
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn record_discipline_trade(
    position_id: String,
    side: String,
    price: f64,
    quantity: i64,
    trade_date: String,
    fee: Option<f64>,
    rule_code: Option<String>,
    event_id: Option<String>,
    pool: State<'_, SqlitePool>,
) -> Result<Position, AppError> {
    record_trade_inner(
        &position_id,
        &side,
        price,
        quantity,
        &trade_date,
        fee.unwrap_or(0.0),
        rule_code.as_deref(),
        event_id.as_deref(),
        &pool,
    )
    .await
}

// =============================================================================
// 事件处置
// =============================================================================

pub(crate) async fn resolve_event_inner(
    event_id: &str,
    resolution: &str,
    reason: Option<&str>,
    pool: &SqlitePool,
) -> Result<(), AppError> {
    let event = repository::get_discipline_event(event_id, pool)
        .await?
        .ok_or_else(|| AppError::InvalidInput("找不到该纪律事件".to_string()))?;
    if event.resolution != "pending" {
        return Err(AppError::InvalidInput(format!(
            "该事件已处置为「{}」，不能重复处置",
            event.resolution
        )));
    }

    match resolution {
        "violated" => {
            let reason = validate_reason(
                reason,
                &format!("选择不执行 {}", event.rule_code),
            )?;
            repository::resolve_discipline_event(pool, event_id, "violated", &reason).await
        }
        "complied" => {
            // 「已执行」不能空口说白话：必须有一笔关联到本事件的卖出成交
            let has_trade = repository::list_trades(event.position_id.as_deref(), pool)
                .await?
                .iter()
                .any(|trade| trade.event_id.as_deref() == Some(event_id));
            if !has_trade {
                return Err(AppError::InvalidInput(
                    "标记「已执行」前必须先录入对应的卖出成交，否则纪律记录会失真".to_string(),
                ));
            }
            repository::resolve_discipline_event(pool, event_id, "complied", "").await
        }
        other => Err(AppError::InvalidInput(format!(
            "处置结果只能是 complied 或 violated，收到「{other}」"
        ))),
    }
}

#[tauri::command]
pub async fn resolve_discipline_event(
    event_id: String,
    resolution: String,
    reason: Option<String>,
    pool: State<'_, SqlitePool>,
) -> Result<(), AppError> {
    resolve_event_inner(&event_id, &resolution, reason.as_deref(), &pool).await
}

// =============================================================================
// 复盘
// =============================================================================

pub(crate) async fn discipline_review_inner(
    pool: &SqlitePool,
) -> Result<DisciplineReview, AppError> {
    let rules = load_rules(pool).await?;
    let closed = repository::list_positions(Some("closed"), pool).await?;
    let events = repository::list_discipline_events(None, pool).await?;

    let complied_count = events.iter().filter(|e| e.resolution == "complied").count();
    let violated_count = events.iter().filter(|e| e.resolution == "violated").count();
    let resolved_total = complied_count + violated_count;

    let mut rows: Vec<ReviewRow> = Vec::new();
    let mut fee_recorded = false;
    let mut unresolved_count = 0usize;

    for position in &closed {
        let Some(close_date_raw) = position.close_date.clone() else {
            continue;
        };
        let open_date = parse_date(&position.open_date, "建仓日")?;
        let close_date = parse_date(&close_date_raw, "清仓日")?;

        let trades = repository::list_trades(Some(&position.id), pool).await?;
        let turnover: f64 = trades
            .iter()
            .map(|trade| trade.price * trade.quantity as f64)
            .sum();
        let fees: f64 = trades.iter().map(|trade| trade.fee).sum();
        if fees > 0.0 {
            fee_recorded = true;
        }
        // 两边同口径：守纪侧按该笔实际发生的加权费率反算
        let fee_rate = if turnover > 0.0 { fees / turnover } else { 0.0 };

        let bars = load_bars(&position.symbol, pool).await?;
        let name = stock_name(&position.symbol, pool).await;
        let window: Vec<HistoricalData> = bars
            .iter()
            .filter(|bar| {
                bar.date >= open_date
                    && bar.date <= close_date + chrono::Duration::days(REPLAY_TAIL_BARS * 2)
            })
            .cloned()
            .collect();
        if window.is_empty() {
            continue;
        }

        let outcome = replay_disciplined_exit(
            &ReplayEntry {
                position_id: position.id.clone(),
                symbol: position.symbol.clone(),
                stock_name: name.clone(),
                open_date,
                close_date,
                cost_price: position.cost_price,
                quantity: position.initial_quantity,
                initial_stop: position.initial_stop,
                target_price: position.target_price,
                fee_rate,
                actual_pnl: position.realized_pnl,
            },
            &window,
            &rules,
        );
        if outcome.unresolved {
            unresolved_count += 1;
        }

        rows.push(ReviewRow {
            position_id: position.id.clone(),
            symbol: position.symbol.clone(),
            name,
            open_date: position.open_date.clone(),
            close_date: close_date_raw,
            cost_price: position.cost_price,
            quantity: position.initial_quantity,
            outcome,
            violations: events
                .iter()
                .filter(|e| {
                    e.position_id.as_deref() == Some(position.id.as_str())
                        && e.resolution == "violated"
                })
                .cloned()
                .collect(),
        });
    }

    // 先定序再统计：scored 借用 rows，排序必须发生在借用之前
    rows.sort_by(|a, b| b.close_date.cmp(&a.close_date));
    let scored: Vec<&ReviewRow> = rows.iter().filter(|row| !row.outcome.unresolved).collect();
    let actual_pnl_total: f64 = scored.iter().map(|row| row.outcome.actual_pnl).sum();
    let disciplined_pnl_total: f64 = scored.iter().map(|row| row.outcome.disciplined_pnl).sum();

    // 违纪规则排行：把违纪 code 与该笔的差额挂钩
    let mut by_rule: Vec<RuleStat> = Vec::new();
    for row in &scored {
        for violation in &row.violations {
            match by_rule
                .iter_mut()
                .find(|stat| stat.rule_code == violation.rule_code)
            {
                Some(stat) => {
                    stat.violated_count += 1;
                    stat.difference_total += row.outcome.difference;
                }
                None => by_rule.push(RuleStat {
                    rule_code: violation.rule_code.clone(),
                    violated_count: 1,
                    difference_total: row.outcome.difference,
                }),
            }
        }
    }
    by_rule.sort_by(|a, b| a.difference_total.total_cmp(&b.difference_total));

    Ok(DisciplineReview {
        closed_count: closed.len(),
        replayed_count: scored.len(),
        unresolved_count,
        complied_count,
        violated_count,
        compliance_rate: (resolved_total > 0)
            .then(|| complied_count as f64 / resolved_total as f64),
        actual_pnl_total,
        disciplined_pnl_total,
        difference_total: disciplined_pnl_total - actual_pnl_total,
        fee_recorded,
        by_rule,
        rows,
        disclaimer: REVIEW_DISCLAIMER.to_string(),
    })
}

#[tauri::command]
pub async fn get_discipline_review(
    pool: State<'_, SqlitePool>,
) -> Result<DisciplineReview, AppError> {
    discipline_review_inner(&pool).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn run_migration(pool: &SqlitePool, sql: &str) {
        for statement in sql.split(';') {
            let statement = statement.trim();
            if statement.is_empty() {
                continue;
            }
            sqlx::query(statement)
                .execute(pool)
                .await
                .unwrap_or_else(|e| panic!("迁移语句应执行成功: {e}\n{statement}"));
        }
    }

    /// 内存库 + 一只有 120 根平稳日线的测试股
    async fn pool_with_data(closes: &[f64]) -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("应创建内存 SQLite");
        for sql in [
            include_str!("../../migrations/01_create_tables.sql"),
            include_str!("../../migrations/03_volume_metrics.sql"),
            include_str!("../../migrations/09_trading_discipline.sql"),
        ] {
            run_migration(&pool, sql).await;
        }
        sqlx::query("INSERT INTO stock_info (symbol, name, exchange) VALUES ('600519', '贵州茅台', 'sh')")
            .execute(&pool)
            .await
            .expect("应插入股票信息");

        let base = NaiveDate::from_ymd_opt(2026, 1, 5).unwrap();
        for (index, close) in closes.iter().enumerate() {
            let prev = if index == 0 { *close } else { closes[index - 1] };
            let date = (base + chrono::Duration::days(index as i64))
                .format("%Y-%m-%d")
                .to_string();
            sqlx::query(
                "INSERT INTO historical_data (symbol, date, open, close, high, low, volume, amount, \
                 amplitude, turnover_rate, change_percent, change, volume_ratio) \
                 VALUES ('600519', ?, ?, ?, ?, ?, 1000000, 1000000, 1, 1, ?, ?, 1)",
            )
            .bind(&date)
            .bind(prev)
            .bind(close)
            .bind(close.max(prev) + 0.2)
            .bind(close.min(prev) - 0.2)
            .bind((close - prev) / prev * 100.0)
            .bind(close - prev)
            .execute(&pool)
            .await
            .expect("应插入日线");
        }
        repository::save_discipline_account(&pool, 200_000.0, "")
            .await
            .expect("应初始化账户");
        pool
    }

    fn flat(n: usize, price: f64) -> Vec<f64> {
        vec![price; n]
    }

    /// 建一笔完全合规的持仓：买 10 元、止损 9.5、目标 11.5
    async fn open_compliant(pool: &SqlitePool) -> Position {
        open_position_inner("600519", 10.0, 1000, 9.5, 11.5, "2026-03-05", 0.0, None, pool)
            .await
            .expect("合规买入不应被拒")
    }

    #[tokio::test]
    async fn selling_part_of_a_position_never_dilutes_the_cost_basis() {
        let pool = pool_with_data(&flat(120, 10.0)).await;
        let position = open_compliant(&pool).await;

        // 涨到 12 卖掉一半
        let after = record_trade_inner(
            &position.id, "sell", 12.0, 500, "2026-03-20", 0.0, None, None, &pool,
        )
        .await
        .expect("卖出应成功");

        assert_eq!(after.quantity, 500);
        assert!(
            (after.cost_price - 10.0).abs() < 1e-9,
            "部分卖出后成本价必须不变——摊薄口径会让止损跟着下移，正好违反棘轮"
        );
        assert!((after.realized_pnl - 1000.0).abs() < 1e-9, "(12−10)×500 = 1000");
        assert_eq!(after.status, "open");

        let account = repository::get_discipline_account(&pool).await.unwrap();
        assert!(
            (account.cash - (200_000.0 - 10_000.0 + 6_000.0)).abs() < 1e-6,
            "现金应随成交自动增减"
        );
    }

    #[tokio::test]
    async fn selling_everything_closes_the_position() {
        let pool = pool_with_data(&flat(120, 10.0)).await;
        let position = open_compliant(&pool).await;
        let after = record_trade_inner(
            &position.id, "sell", 9.0, 1000, "2026-03-20", 5.0, None, None, &pool,
        )
        .await
        .unwrap();
        assert_eq!(after.status, "closed");
        assert_eq!(after.close_date.as_deref(), Some("2026-03-20"));
        assert!((after.realized_pnl - (-1005.0)).abs() < 1e-9, "(9−10)×1000 − 5 手续费");
    }

    #[tokio::test]
    async fn adding_to_a_position_recomputes_cost_but_cannot_lower_the_stop() {
        let pool = pool_with_data(&flat(120, 10.0)).await;
        let position = open_compliant(&pool).await;
        let stop_before = position.stop_price;

        // 8 元加仓 1000 股，加权成本降到 9
        let after = record_trade_inner(
            &position.id, "buy", 8.0, 1000, "2026-03-20", 0.0, None, None, &pool,
        )
        .await
        .unwrap();

        assert!((after.cost_price - 9.0).abs() < 1e-9, "加权成本 (10000+8000)/2000 = 9");
        assert!(
            after.stop_price >= stop_before,
            "补跌摊薄不能把止损带下去：{} → {}",
            stop_before,
            after.stop_price
        );
    }

    #[tokio::test]
    async fn a_second_open_position_on_the_same_symbol_is_rejected() {
        let pool = pool_with_data(&flat(120, 10.0)).await;
        open_compliant(&pool).await;
        let err = open_position_inner(
            "600519", 10.0, 100, 9.5, 11.5, "2026-03-06", 0.0, None, &pool,
        )
        .await
        .expect_err("同股票不应有两条未平仓持仓");
        assert!(err.to_string().contains("已有未平仓持仓"));
    }

    #[tokio::test]
    async fn a_blocked_entry_without_a_reason_never_reaches_the_database() {
        let pool = pool_with_data(&flat(120, 10.0)).await;
        // 盈亏比只有 1:1，必被 ENTRY_RR_TOO_LOW 拦
        let err = open_position_inner(
            "600519", 10.0, 1000, 9.5, 10.5, "2026-03-05", 0.0, None, &pool,
        )
        .await
        .expect_err("触发准入纪律且无理由时必须拒绝写库");
        assert!(err.to_string().contains("ENTRY_RR_TOO_LOW"));
        assert!(
            repository::list_positions(None, &pool).await.unwrap().is_empty(),
            "被拒的买入不能留下任何持仓记录"
        );
    }

    #[tokio::test]
    async fn a_short_override_reason_is_rejected() {
        let pool = pool_with_data(&flat(120, 10.0)).await;
        let err = open_position_inner(
            "600519", 10.0, 1000, 9.5, 10.5, "2026-03-05", 0.0, Some("想买"), &pool,
        )
        .await
        .expect_err("理由太短应被拒");
        assert!(err.to_string().contains(&MIN_REASON_CHARS.to_string()));
    }

    #[tokio::test]
    async fn overriding_an_entry_rule_leaves_a_permanent_record() {
        let pool = pool_with_data(&flat(120, 10.0)).await;
        let position = open_position_inner(
            "600519", 10.0, 1000, 9.5, 10.5, "2026-03-05", 0.0,
            Some("这只票我跟踪很久了，愿意为它破一次盈亏比的例"),
            &pool,
        )
        .await
        .expect("填了足够长的理由应放行");

        let events = repository::list_discipline_events(Some("violated"), &pool).await.unwrap();
        assert_eq!(events.len(), 1, "每条被无视的准入规则各记一条");
        assert_eq!(events[0].rule_code, "ENTRY_RR_TOO_LOW");
        assert_eq!(events[0].position_id.as_deref(), Some(position.id.as_str()));
        assert!(events[0].reason.as_deref().unwrap().contains("跟踪很久"));
    }

    #[tokio::test]
    async fn scanning_twice_on_the_same_day_does_not_duplicate_events() {
        // 建仓后价格腰斩到止损下方
        let mut closes = flat(115, 10.0);
        closes.extend(flat(5, 8.0));
        let pool = pool_with_data(&closes).await;
        open_position_inner("600519", 10.0, 1000, 9.5, 11.5, "2026-01-05", 0.0, None, &pool)
            .await
            .unwrap();

        let first = scan_discipline_board(&pool).await.unwrap();
        assert!(!first.pending_events.is_empty(), "跌破止损应产生待处理事件");
        let count_after_first = first.pending_events.len();

        let second = scan_discipline_board(&pool).await.unwrap();
        assert_eq!(
            second.pending_events.len(),
            count_after_first,
            "同持仓+同规则+同日重复扫描必须幂等，否则横幅会被刷屏"
        );
    }

    #[tokio::test]
    async fn marking_complied_requires_an_actual_trade() {
        let mut closes = flat(115, 10.0);
        closes.extend(flat(5, 8.0));
        let pool = pool_with_data(&closes).await;
        let position =
            open_position_inner("600519", 10.0, 1000, 9.5, 11.5, "2026-01-05", 0.0, None, &pool)
                .await
                .unwrap();
        let board = scan_discipline_board(&pool).await.unwrap();
        let event = board.pending_events.first().expect("应有待处理事件").clone();

        let err = resolve_event_inner(&event.id, "complied", None, &pool)
            .await
            .expect_err("没有成交记录就说「已执行」应被拒");
        assert!(err.to_string().contains("必须先录入对应的卖出成交"));

        // 真的卖了再标记
        record_trade_inner(
            &position.id, "sell", 8.0, 1000, "2026-05-04", 0.0,
            Some(&event.rule_code), Some(&event.id), &pool,
        )
        .await
        .unwrap();
        let refreshed = repository::get_discipline_event(&event.id, &pool).await.unwrap().unwrap();
        assert_eq!(refreshed.resolution, "complied", "录成交时应自动标记为已执行");
    }

    #[tokio::test]
    async fn refusing_to_act_requires_a_written_reason() {
        let mut closes = flat(115, 10.0);
        closes.extend(flat(5, 8.0));
        let pool = pool_with_data(&closes).await;
        open_position_inner("600519", 10.0, 1000, 9.5, 11.5, "2026-01-05", 0.0, None, &pool)
            .await
            .unwrap();
        let board = scan_discipline_board(&pool).await.unwrap();
        let event = board.pending_events.first().unwrap().clone();

        assert!(resolve_event_inner(&event.id, "violated", Some("不卖"), &pool)
            .await
            .is_err());

        resolve_event_inner(
            &event.id,
            "violated",
            Some("我认为这是恐慌盘，明天会回来，决定不执行止损"),
            &pool,
        )
        .await
        .expect("理由够长应放行");

        let refreshed = repository::get_discipline_event(&event.id, &pool).await.unwrap().unwrap();
        assert_eq!(refreshed.resolution, "violated");
        assert!(refreshed.reason.as_deref().unwrap().contains("恐慌盘"));

        assert!(
            resolve_event_inner(&event.id, "complied", None, &pool).await.is_err(),
            "已处置的事件不能重复处置——违纪记录必须不可撤销"
        );
    }

    #[tokio::test]
    async fn review_compares_disciplined_and_actual_on_closed_positions_only() {
        // 前 115 天 10 元，之后跌到 8 元
        let mut closes = flat(115, 10.0);
        closes.extend(flat(10, 8.0));
        let pool = pool_with_data(&closes).await;
        let position =
            open_position_inner("600519", 10.0, 1000, 9.5, 11.5, "2026-01-05", 0.0, None, &pool)
                .await
                .unwrap();

        // 违纪：跌破止损没走，一路扛到最后才割
        record_trade_inner(
            &position.id, "sell", 8.0, 1000, "2026-05-09", 0.0, None, None, &pool,
        )
        .await
        .unwrap();

        let review = discipline_review_inner(&pool).await.unwrap();
        assert_eq!(review.closed_count, 1);
        assert_eq!(review.replayed_count, 1);
        let row = &review.rows[0];
        assert!(!row.outcome.same_as_actual, "纪律本应更早离场");
        assert!(
            review.disciplined_pnl_total > review.actual_pnl_total,
            "严格守纪的亏损应小于实际：守纪 {} vs 实际 {}",
            review.disciplined_pnl_total,
            review.actual_pnl_total
        );
        assert!(review.difference_total > 0.0, "差额为正 = 违纪的代价");
        assert!(review.disclaimer.contains("反事实估算"));
    }

    #[tokio::test]
    async fn open_positions_are_excluded_from_the_review() {
        let pool = pool_with_data(&flat(120, 10.0)).await;
        open_compliant(&pool).await;
        let review = discipline_review_inner(&pool).await.unwrap();
        assert_eq!(review.closed_count, 0);
        assert!(
            review.rows.is_empty(),
            "未平仓的赌局不能用今天的价格记分"
        );
    }

    #[tokio::test]
    async fn total_equity_is_computed_from_cash_plus_market_value() {
        let pool = pool_with_data(&flat(120, 10.0)).await;
        open_compliant(&pool).await;
        let view = discipline_account_view(&pool).await.unwrap();
        assert!((view.cash - 190_000.0).abs() < 1e-6, "买入 1 万后现金应减少");
        assert!((view.total_market_value - 10_000.0).abs() < 1e-6);
        assert!(
            (view.total_equity - 200_000.0).abs() < 1e-6,
            "总资产 = 现金 + 持仓市值，买入当下不应变化"
        );
    }

    /// 走一遍完整的用户旅程，等价于手动点一遍界面：
    /// 设资金 → 无止损被拦 → 盈亏比不够被拦 → 合规建仓 → 价格跌破 → 扫描出必卖 →
    /// 选择不执行并留痕 → 重复扫描不刷屏 → 一路扛到更低价才割 → 复盘算出违纪的代价。
    ///
    /// 本机没有图形环境，`bun run tauri dev` 跑不起来，这个测试是它的等价替身：
    /// 走的是命令层真实入口与真实建表 SQL，只是没有渲染。
    #[tokio::test]
    async fn the_full_journey_from_blocked_entry_to_review() {
        // 前 110 根 10 元，随后一路下跌到 6 元
        let mut closes = flat(110, 10.0);
        closes.extend([8.0, 7.5, 7.0, 6.5]);
        closes.extend(flat(11, 6.0));
        let pool = pool_with_data(&closes).await;
        let base = NaiveDate::from_ymd_opt(2026, 1, 5).unwrap();
        let buy_date = (base + chrono::Duration::days(105)).format("%Y-%m-%d").to_string();
        let sell_date = (base + chrono::Duration::days(124)).format("%Y-%m-%d").to_string();

        // 1. 没填止损：连算都算不了仓位，直接拦
        let no_stop = check_buy_inner("600519", 10.0, None, Some(11.5), None, None, &pool)
            .await
            .unwrap();
        assert!(!no_stop.allowed);
        assert!(no_stop.items.iter().any(|hit| hit.code == "ENTRY_STOP_REQUIRED"));
        assert!(no_stop.disclaimer.contains("不构成任何买入建议"));

        // 2. 盈亏比 1.5:1 不够
        let thin = check_buy_inner("600519", 10.0, Some(9.5), Some(10.75), None, None, &pool)
            .await
            .unwrap();
        assert!(!thin.allowed);
        assert!(thin.items.iter().any(|hit| hit.code == "ENTRY_RR_TOO_LOW"));

        // 3. 合规：止损 9.5、目标 11.5 → 盈亏比 3:1
        let ok = check_buy_inner("600519", 10.0, Some(9.5), Some(11.5), None, None, &pool)
            .await
            .unwrap();
        assert!(ok.allowed, "合规参数不该被拦：{:?}", ok.items);
        assert_eq!(ok.sizing.max_shares, 5000, "25% 单票上限 / 10 元 = 5000 股");

        let position =
            open_position_inner("600519", 10.0, 1000, 9.5, 11.5, &buy_date, 0.0, None, &pool)
                .await
                .expect("合规买入应成功");
        assert!(
            position.stop_price >= 9.5,
            "生效止损不得低于用户自己填的那条：{}",
            position.stop_price
        );

        // 4. 扫描：价格已跌到 6 元，远在止损之下
        let board = scan_discipline_board(&pool).await.unwrap();
        let verdict = board.positions[0].verdict.as_ref().expect("应有裁决");
        assert_eq!(verdict.action, DisciplineAction::MustSell);
        assert!(verdict.items.iter().any(|hit| hit.code == "EXIT_HARD_STOP"));
        let event = board
            .pending_events
            .iter()
            .find(|e| e.rule_code == "EXIT_HARD_STOP")
            .expect("硬止损应落成待处理事件")
            .clone();

        // 5. 选择不执行，理由入库
        resolve_event_inner(
            &event.id,
            "violated",
            Some("跌得太急了，我觉得是错杀，想再等一等看看能不能反弹"),
            &pool,
        )
        .await
        .expect("填了足够长的理由应放行");

        // 6. 再扫一次不应重复产生同一条
        let again = scan_discipline_board(&pool).await.unwrap();
        assert!(
            !again.pending_events.iter().any(|e| e.id == event.id),
            "已处置的事件不应再出现在待办里"
        );

        // 7. 最终还是割在更低的 6 元
        record_trade_inner(&position.id, "sell", 6.0, 1000, &sell_date, 0.0, None, None, &pool)
            .await
            .unwrap();

        // 8. 复盘：守纪本应在跌破次日开盘就走
        let review = discipline_review_inner(&pool).await.unwrap();
        assert_eq!(review.violated_count, 1);
        assert_eq!(review.compliance_rate, Some(0.0), "唯一一条纪律被违反，守纪率 0");
        let row = &review.rows[0];
        let fill = row.outcome.fills.first().expect("守纪路径应有一笔成交");
        assert_eq!(fill.rule_code, "EXIT_HARD_STOP");
        assert!(
            (fill.fill_price - 8.0).abs() < 1e-9,
            "成交价应是触发次日的开盘价 8.0，不是止损价 9.5"
        );
        assert!((row.outcome.actual_pnl - (-4000.0)).abs() < 1e-6, "实际割在 6 元 → 亏 4000");
        assert!(
            (row.outcome.disciplined_pnl - (-2000.0)).abs() < 1e-6,
            "守纪走在 8 元 → 亏 2000"
        );
        assert!(
            (review.difference_total - 2000.0).abs() < 1e-6,
            "差额 2000 元，就是这次不执行止损的代价"
        );
        assert_eq!(review.by_rule[0].rule_code, "EXIT_HARD_STOP");
        assert!(!row.violations.is_empty(), "复盘要能看到当时填的理由");
        assert!(row.violations[0].reason.as_deref().unwrap().contains("错杀"));
    }

    /// 前端的 TS 类型是手写的（项目无代码生成），所以下发形状必须钉住。
    /// `PositionView` 用 `serde(flatten)` 把 Position 各列摊平在顶层——
    /// 这个行为一旦变了，前端会静默拿到 undefined 而不是报错。
    #[tokio::test]
    async fn the_board_payload_matches_the_hand_written_frontend_types() {
        let pool = pool_with_data(&flat(120, 10.0)).await;
        open_compliant(&pool).await;
        let board = scan_discipline_board(&pool).await.unwrap();
        let json = serde_json::to_value(&board).unwrap();

        let position = &json["positions"][0];
        for key in [
            "id", "symbol", "status", "open_date", "cost_price", "quantity", "initial_quantity",
            "initial_stop", "stop_price", "stop_basis", "target_price", "highest_price",
            "scale_out_done", "realized_pnl", "basis_suspect",
        ] {
            assert!(!position[key].is_null() || key == "target_price" || key == "highest_price",
                "Position 的 {key} 应被 flatten 到 PositionRow 顶层");
            assert!(position.get(key).is_some(), "PositionRow 缺少字段 {key}");
        }
        for key in ["name", "last_close", "verdict", "unavailable_reason"] {
            assert!(position.get(key).is_some(), "PositionRow 缺少视图字段 {key}");
        }
        assert!(
            position.get("position").is_none(),
            "flatten 后不应再有嵌套的 position 对象，否则前端类型全错"
        );

        // 枚举的 serde 表示必须是 snake_case 字符串，和 TS 的字面量联合类型对齐
        let verdict = &position["verdict"];
        assert!(verdict["stop_basis"].as_str().is_some());
        assert!(
            ["warn", "blocked", "must_reduce", "must_sell"]
                .contains(&verdict["action"].as_str().unwrap()),
            "DisciplineAction 必须序列化成 snake_case"
        );
        assert!(["low", "medium", "high"].contains(&verdict["level"].as_str().unwrap()));

        for key in ["cash", "total_equity", "total_market_value", "open_position_count", "rules"] {
            assert!(json["account"].get(key).is_some(), "AccountView 缺少 {key}");
        }
        assert!(json["account"]["rules"].get("scale_out_tiers").unwrap().is_array());
        assert!(json.get("pending_events").is_some());
        assert!(json.get("disclaimer").is_some());
    }

    #[tokio::test]
    async fn saving_rules_round_trips_through_the_account_row() {
        let pool = pool_with_data(&flat(120, 10.0)).await;
        let custom = DisciplineRules { fixed_stop_pct: 5.0, max_risk_pct: 1.0, ..Default::default() };
        // #[tauri::command] 薄壳需要 State，无法在单测里构造，直接测其转调的仓储 + 视图闭环
        repository::save_discipline_account(&pool, 120_000.0, &custom.to_json())
            .await
            .unwrap();
        let view = discipline_account_view(&pool).await.unwrap();
        assert_eq!(view.rules.fixed_stop_pct, 5.0);
        assert_eq!(view.rules.max_risk_pct, 1.0);
        assert!((view.cash - 120_000.0).abs() < 1e-6);
    }
}
