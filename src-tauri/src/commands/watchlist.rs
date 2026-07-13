//! 收藏池命令模块
//!
//! 提供收藏股票池（选票池）的增删查、指标概览与一键综合预测。
//!
//! ⚠️ 诚实预测原则（.claude/CLAUDE.md 结论 1/8/11）：本模块**不新增任何方向规则**——
//! 综合预测的方向/信号强度/买卖点/建议全部原样透传专业引擎既有输出（点预测=历史
//! 无条件漂移锚，真实不确定性=80% 校准区间带）；历史基准率仅作"无技能对照"展示；
//! 动量/52 周位置/多周期涨跌幅均为描述性指标，非收益预测。

use crate::commands::stock_prediction::{
    get_valuation_context, predict_with_professional_strategy_inner, ValuationContext,
};
use crate::db::connection::create_temp_pool;
use crate::db::models::{HistoricalData, StockCapital, StockFundamental};
use crate::db::repository::{
    get_recent_historical_data, get_recent_historical_data_for_symbols, get_stock_capital,
    get_stock_fundamentals, resolve_historical_symbol,
};
use crate::error::AppError;
use crate::prediction::types::{
    PredictionInterval, PredictionRequest, ProfessionalPredictionResponse, RiskSummary,
};
use chrono::{Datelike, Duration, Local, NaiveDate};
use sqlx::SqlitePool;
use std::collections::HashMap;
use tauri::State;

/// 概览/综合预测的取数窗口（根）：覆盖一年约 244 个交易日 + YTD 最坏回溯与假期余量
const OVERVIEW_BARS: usize = 270;
/// 综合预测的分析历史窗口（与预测页"纯技术分析"同路径，inner 内部会 clamp 到 [120, 3000]）
const COMPREHENSIVE_HISTORY_DAYS: usize = 1500;

// =============================================================================
// 指标纯函数（全部诚实缺省：数据不足返回 None，不用不完整窗口凑数）
// =============================================================================

/// 归一为纯 6 位代码（收藏池存库规范）；提不出恰好 6 位数字时原样返回 trim 结果
fn canonical_symbol(symbol: &str) -> String {
    let digits: String = symbol.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() == 6 {
        digits
    } else {
        symbol.trim().to_string()
    }
}

/// N 个交易日涨跌幅（%）：末收盘 / 倒数第 N+1 根收盘 - 1。
/// bars 需时间正序；长度不足 N+1 根时返回 None
fn period_change(bars: &[HistoricalData], n: usize) -> Option<f64> {
    if n == 0 || bars.len() < n + 1 {
        return None;
    }
    let last = bars.last()?.close;
    let base = bars[bars.len() - 1 - n].close;
    (base > 0.0).then(|| (last / base - 1.0) * 100.0)
}

/// 年初至今涨跌幅（%）：基准 = 最后一根「日期早于最新K线年份 1 月 1 日」的收盘（上年收官价）；
/// 窗口内找不到（次新股/窗口不够）返回 None
fn ytd_change(bars: &[HistoricalData]) -> Option<f64> {
    let last = bars.last()?;
    let jan1 = NaiveDate::from_ymd_opt(last.date.year(), 1, 1)?;
    let base = bars.iter().rev().find(|b| b.date < jan1)?.close;
    (base > 0.0).then(|| (last.close / base - 1.0) * 100.0)
}

/// 一年涨跌幅（%）：基准 = 最后一根「日期 ≤ 最新日期 - 365 自然日」的收盘；窗口内无则 None
fn one_year_change(bars: &[HistoricalData]) -> Option<f64> {
    let last = bars.last()?;
    let cutoff = last.date - Duration::days(365);
    let base = bars.iter().rev().find(|b| b.date <= cutoff)?.close;
    (base > 0.0).then(|| (last.close / base - 1.0) * 100.0)
}

/// 52 周高/低与当前位置：近 365 自然日窗口的 max(high)/min(low)，
/// position = (close - 低) / (高 - 低) × 100（0=贴近年内低点，100=贴近年内高点）；
/// 高低重合时 position 为 None
fn week52_stats(bars: &[HistoricalData]) -> Option<(f64, f64, Option<f64>)> {
    let last = bars.last()?;
    let cutoff = last.date - Duration::days(365);
    let mut high = f64::MIN;
    let mut low = f64::MAX;
    for b in bars.iter().filter(|b| b.date > cutoff) {
        high = high.max(b.high);
        low = low.min(b.low);
    }
    if high < low {
        return None;
    }
    let position = (high > low).then(|| (last.close - low) / (high - low) * 100.0);
    Some((high, low, position))
}

/// 近 n 个交易日「收盘较前收上涨」占比（0-1）与日均涨跌幅（%）。
/// 用收盘价现算而非依赖 change_percent 字段（后者可能未回填）；不足 n+1 根返回 None
fn daily_up_ratio(bars: &[HistoricalData], n: usize) -> Option<(f64, f64)> {
    if n == 0 || bars.len() < n + 1 {
        return None;
    }
    let tail = &bars[bars.len() - n - 1..];
    let mut ups = 0usize;
    let mut sum = 0.0;
    for w in tail.windows(2) {
        let prev = w[0].close;
        if prev <= 0.0 {
            return None;
        }
        let chg = (w[1].close / prev - 1.0) * 100.0;
        if chg > 0.0 {
            ups += 1;
        }
        sum += chg;
    }
    Some((ups as f64 / n as f64, sum / n as f64))
}

/// 0 / 非有限值视为"未刷新/无数据"，返回 None（同 get_valuation_context 口径）
fn nonzero(v: f64) -> Option<f64> {
    (v.is_finite() && v != 0.0).then_some(v)
}

// =============================================================================
// 数据查找辅助（symbol 混格兼容：先按收藏的纯 6 位查，miss 再按历史表解析出的变体查）
// =============================================================================

/// stock_info 权威名称：按 6 位前缀匹配、排除"代码当名称"的脏行（同 stock_realtime 的回退策略）
async fn lookup_stock_name(symbol: &str, pool: &SqlitePool) -> String {
    let row: Option<(String,)> = sqlx::query_as(
        "SELECT name FROM stock_info
         WHERE substr(symbol, 1, 6) = substr(?, 1, 6) AND name <> '' AND name <> symbol
         LIMIT 1",
    )
    .bind(symbol)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    row.map(|(name,)| name).unwrap_or_default()
}

async fn lookup_capital(
    canonical: &str,
    resolved: Option<&str>,
    pool: &SqlitePool,
) -> Option<StockCapital> {
    if let Ok(Some(cap)) = get_stock_capital(canonical, pool).await {
        return Some(cap);
    }
    match resolved {
        Some(r) if r != canonical => get_stock_capital(r, pool).await.ok().flatten(),
        _ => None,
    }
}

/// 最新报告期基本面（get_stock_fundamentals 按报告期升序，取最后一个）
async fn lookup_fundamental_latest(
    canonical: &str,
    resolved: Option<&str>,
    pool: &SqlitePool,
) -> Option<StockFundamental> {
    if let Ok(funds) = get_stock_fundamentals(canonical, pool).await {
        if let Some(latest) = funds.into_iter().last() {
            return Some(latest);
        }
    }
    match resolved {
        Some(r) if r != canonical => get_stock_fundamentals(r, pool)
            .await
            .ok()
            .and_then(|funds| funds.into_iter().last()),
        _ => None,
    }
}

// =============================================================================
// 收藏池 CRUD + 指标概览
// =============================================================================

/// 收藏概览行——全部指标现算；Option 字段 = 数据缺失时前端显示占位符而非 0
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WatchlistItem {
    pub symbol: String,
    pub name: String,
    pub added_at: String,
    pub sort_order: i64,
    // 最新有效 K 线
    pub latest_date: Option<String>,
    /// 最新 K 线距今自然日数（数据新鲜度；周末/假期 1-2 天属正常）
    pub staleness_days: Option<i64>,
    pub close: Option<f64>,
    pub change_percent: Option<f64>,
    pub turnover_rate: Option<f64>,
    pub volume_ratio: Option<f64>,
    // 多周期涨跌幅（%）
    pub change_5d: Option<f64>,
    pub change_10d: Option<f64>,
    pub change_20d: Option<f64>,
    pub change_ytd: Option<f64>,
    pub change_1y: Option<f64>,
    // 52 周
    pub week52_high: Option<f64>,
    pub week52_low: Option<f64>,
    pub week52_position: Option<f64>,
    // 估值（stock_capital，0→None 同 get_valuation_context 口径）
    pub pe: Option<f64>,
    pub pb: Option<f64>,
    pub circulating_market_cap_yi: Option<f64>,
    // 最新基本面（stock_fundamentals 最新报告期）
    pub roe: Option<f64>,
    pub revenue_growth: Option<f64>,
    pub report_date: Option<String>,
}

/// 获取收藏池全部股票的指标概览（全部现算，不依赖缓存）
#[tauri::command]
pub async fn get_watchlist_overview(
    pool: State<'_, SqlitePool>,
) -> Result<Vec<WatchlistItem>, AppError> {
    watchlist_overview(&pool).await
}

/// 概览实现（与命令解耦，便于冒烟测试直接调用）
pub(crate) async fn watchlist_overview(pool: &SqlitePool) -> Result<Vec<WatchlistItem>, AppError> {
    let rows: Vec<(String, String, i64)> = sqlx::query_as(
        "SELECT symbol, added_at, sort_order FROM watchlist ORDER BY sort_order, added_at",
    )
    .fetch_all(pool)
    .await?;

    // 库内历史 symbol 存在纯6位 / .SZ / .SH 混格，而批量查询不解析变体——先逐票解析
    let mut resolved_rows: Vec<(String, String, i64, Option<String>)> =
        Vec::with_capacity(rows.len());
    let mut resolved_symbols: Vec<String> = Vec::new();
    for (symbol, added_at, sort_order) in rows {
        let resolved = resolve_historical_symbol(&symbol, pool).await?;
        if let Some(r) = &resolved {
            if !resolved_symbols.contains(r) {
                resolved_symbols.push(r.clone());
            }
        }
        resolved_rows.push((symbol, added_at, sort_order, resolved));
    }

    // 一次批量取窗口，避免逐票 N 次查询
    let bars_by_symbol: HashMap<String, Vec<HistoricalData>> =
        get_recent_historical_data_for_symbols(&resolved_symbols, OVERVIEW_BARS, pool)
            .await?
            .into_iter()
            .collect();

    let today = Local::now().date_naive();
    let mut items = Vec::with_capacity(resolved_rows.len());
    for (symbol, added_at, sort_order, resolved) in resolved_rows {
        let name = lookup_stock_name(&symbol, pool).await;
        let capital = lookup_capital(&symbol, resolved.as_deref(), pool).await;
        let fundamental = lookup_fundamental_latest(&symbol, resolved.as_deref(), pool).await;
        let bars = resolved
            .as_ref()
            .and_then(|r| bars_by_symbol.get(r))
            .map(|b| b.as_slice())
            .unwrap_or(&[]);
        let last = bars.last();
        let w52 = week52_stats(bars);

        items.push(WatchlistItem {
            symbol,
            name,
            added_at,
            sort_order,
            latest_date: last.map(|b| b.date.format("%Y-%m-%d").to_string()),
            staleness_days: last.map(|b| (today - b.date).num_days()),
            close: last.map(|b| b.close),
            change_percent: last.map(|b| b.change_percent),
            turnover_rate: last.and_then(|b| nonzero(b.turnover_rate)),
            volume_ratio: last.and_then(|b| nonzero(b.volume_ratio)),
            change_5d: period_change(bars, 5),
            change_10d: period_change(bars, 10),
            change_20d: period_change(bars, 20),
            change_ytd: ytd_change(bars),
            change_1y: one_year_change(bars),
            week52_high: w52.map(|(h, _, _)| h),
            week52_low: w52.map(|(_, l, _)| l),
            week52_position: w52.and_then(|(_, _, p)| p),
            pe: capital.as_ref().and_then(|c| nonzero(c.pe)),
            pb: capital.as_ref().and_then(|c| nonzero(c.pb)),
            circulating_market_cap_yi: capital
                .as_ref()
                .and_then(|c| nonzero(c.circulating_market_cap).map(|v| v / 1.0e8)),
            roe: fundamental.as_ref().and_then(|f| f.roe),
            revenue_growth: fundamental.as_ref().and_then(|f| f.revenue_growth),
            report_date: fundamental.as_ref().map(|f| f.report_date.clone()),
        });
    }
    Ok(items)
}

/// 添加收藏（symbol 归一为纯 6 位入库；重复添加幂等忽略）
#[tauri::command]
pub async fn add_to_watchlist(
    symbol: String,
    pool: State<'_, SqlitePool>,
) -> Result<(), AppError> {
    let canonical = canonical_symbol(&symbol);
    if canonical.is_empty() {
        return Err(AppError::InvalidInput("股票代码不能为空".to_string()));
    }
    sqlx::query(
        "INSERT OR IGNORE INTO watchlist (symbol, sort_order)
         VALUES (?, (SELECT COALESCE(MAX(sort_order), 0) + 1 FROM watchlist))",
    )
    .bind(&canonical)
    .execute(&*pool)
    .await?;
    Ok(())
}

/// 移除收藏
#[tauri::command]
pub async fn remove_from_watchlist(
    symbol: String,
    pool: State<'_, SqlitePool>,
) -> Result<(), AppError> {
    let canonical = canonical_symbol(&symbol);
    sqlx::query("DELETE FROM watchlist WHERE symbol = ?")
        .bind(&canonical)
        .execute(&*pool)
        .await?;
    Ok(())
}

/// 收藏代码列表（纯 6 位），供列表/实时行情/预测页做星标判断
#[tauri::command]
pub async fn get_watchlist_symbols(pool: State<'_, SqlitePool>) -> Result<Vec<String>, AppError> {
    let rows: Vec<(String,)> =
        sqlx::query_as("SELECT symbol FROM watchlist ORDER BY sort_order, added_at")
            .fetch_all(&*pool)
            .await?;
    Ok(rows.into_iter().map(|(s,)| s).collect())
}

// =============================================================================
// 一键综合预测
// =============================================================================

/// 免责标注：后端固定下发，前端必须展示
const COMPREHENSIVE_DISCLAIMER: &str = "本报告为技术面+估值的描述性综合，非投资建议。经实证单股方向不可预测（引擎方向准确率不高于朴素基准），「信号强度」非方向命中概率；点预测为历史无条件漂移锚，真实不确定性以 80% 校准区间带为准；历史基准率为无技能对照；排序仅为相对强弱描述。市场有风险，决策需独立判断。";

/// 一键综合预测报告——决策摘要字段全部**原样透传**专业引擎既有输出，零新增方向规则
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComprehensiveReport {
    pub symbol: String,
    pub name: String,
    /// 报告生成时刻（本地时间）
    pub generated_at: String,
    /// 所依据最新 K 线日期（数据新鲜度）
    pub latest_date: String,
    pub staleness_days: i64,
    pub current_price: f64,
    pub prediction_days: usize,

    // —— 决策摘要（透传引擎输出）——
    /// 末日交易信号文本（引擎既有输出，缺省"中性"）
    pub direction: String,
    /// 信号强度 0.25-0.92（非方向命中概率）
    pub signal_strength: f64,
    /// 预期累计涨跌幅（%）= 末日预测价 / 最新真实价 - 1（漂移锚，仅供参考）
    pub expected_change_percent: f64,
    /// 末日 80% 校准区间带（真实不确定性的诚实表达）
    pub interval: Option<PredictionInterval>,
    pub current_advice: String,
    pub risk_level: String,
    /// 结构化风险事实，供收藏页筛选、排序和展开证据。
    #[serde(default)]
    pub risk_summary: RiskSummary,
    /// 自适应多因子得分（0-100，引擎既有输出）
    pub adaptive_score: f64,
    pub buy_point_count: usize,
    pub sell_point_count: usize,
    pub nearest_support: Option<f64>,
    pub nearest_resistance: Option<f64>,
    pub key_factors: Vec<String>,

    // —— 描述性动量/位置 ——
    pub momentum_5d: Option<f64>,
    pub momentum_20d: Option<f64>,
    pub momentum_60d: Option<f64>,
    pub week52_position: Option<f64>,

    // —— 历史基准率（无技能对照）——
    /// 近 20 个交易日日线上涨占比（0-1）
    pub up_ratio_20d: Option<f64>,
    pub up_ratio_60d: Option<f64>,
    pub up_ratio_250d: Option<f64>,
    /// 近 250 个交易日日均涨跌幅（%）
    pub avg_daily_change_250d: Option<f64>,

    // —— 完整明细（预测页直接复用现有渲染管线，引擎只跑一遍）——
    pub prediction: ProfessionalPredictionResponse,
    pub valuation: ValuationContext,

    /// 免责标注（前端必须展示）
    pub disclaimer: String,
}

/// 一键综合预测：专业策略预测（纯技术路径，与预测页"纯技术分析"同口径）
/// + 估值上下文 + 描述性动量/52周位置 + 历史基准率，一次调用聚合成报告。
/// 纯本地计算，不消耗 API 额度。
#[tauri::command]
pub async fn comprehensive_predict(
    symbol: String,
    days: Option<usize>,
) -> Result<ComprehensiveReport, String> {
    let prediction_days = days.unwrap_or(5).clamp(1, 30);
    let canonical = canonical_symbol(&symbol);
    if canonical.is_empty() {
        return Err("股票代码不能为空".to_string());
    }

    // 1) 专业策略预测：无模型依赖、确定性（use_candle=false 刻意与"纯技术分析"tab 同口径）
    let request = PredictionRequest {
        stock_code: canonical.clone(),
        model_name: None,
        prediction_days,
        use_candle: false,
    };
    let prediction =
        predict_with_professional_strategy_inner(request, Some(COMPREHENSIVE_HISTORY_DAYS)).await?;

    // 2) 估值上下文（失败降级为全 None，不阻断报告主体）
    let valuation = get_valuation_context(canonical.clone())
        .await
        .unwrap_or(ValuationContext {
            symbol: canonical.clone(),
            pe: None,
            pb: None,
            circulating_market_cap_yi: None,
            report_date: None,
            roe: None,
            eps: None,
            bps: None,
            revenue_growth: None,
            profit_growth: None,
        });

    // 3) 描述性动量/52周位置 + 历史基准率（单票查询内部自带 symbol 变体解析）
    let pool = create_temp_pool().await?;
    let bars = get_recent_historical_data(&canonical, OVERVIEW_BARS, &pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))?;
    let last_bar = bars.last().ok_or("无有效历史数据")?;
    let name = lookup_stock_name(&canonical, &pool).await;

    // 决策摘要：从引擎响应原样抽取
    let last_pred = prediction
        .predictions
        .predictions
        .last()
        .ok_or("预测结果为空")?;
    let base_price = prediction
        .predictions
        .last_real_data
        .as_ref()
        .map(|d| d.price)
        .filter(|p| *p > 0.0)
        .unwrap_or(last_bar.close);
    let expected_change_percent = if base_price > 0.0 {
        (last_pred.predicted_price / base_price - 1.0) * 100.0
    } else {
        0.0
    };
    let pa = &prediction.professional_analysis;
    let risk_summary = prediction
        .predictions
        .diagnostics
        .as_ref()
        .map(|diagnostics| diagnostics.risk_summary.clone())
        .unwrap_or_default();
    let up20 = daily_up_ratio(&bars, 20);
    let up60 = daily_up_ratio(&bars, 60);
    let up250 = daily_up_ratio(&bars, 250);

    Ok(ComprehensiveReport {
        symbol: canonical,
        name,
        generated_at: Local::now().format("%Y-%m-%d %H:%M").to_string(),
        latest_date: last_bar.date.format("%Y-%m-%d").to_string(),
        staleness_days: (Local::now().date_naive() - last_bar.date).num_days(),
        current_price: last_bar.close,
        prediction_days,
        direction: last_pred
            .trading_signal
            .clone()
            .unwrap_or_else(|| "中性".to_string()),
        signal_strength: last_pred.confidence,
        expected_change_percent,
        interval: last_pred.interval.clone(),
        current_advice: pa.current_advice.clone(),
        risk_level: pa.risk_level.clone(),
        risk_summary,
        adaptive_score: pa.multi_factor_score.adaptive_score,
        buy_point_count: pa.buy_points.len(),
        sell_point_count: pa.sell_points.len(),
        nearest_support: pa.support_resistance.support_levels.first().copied(),
        nearest_resistance: pa.support_resistance.resistance_levels.first().copied(),
        key_factors: last_pred.key_factors.clone().unwrap_or_default(),
        momentum_5d: period_change(&bars, 5),
        momentum_20d: period_change(&bars, 20),
        momentum_60d: period_change(&bars, 60),
        week52_position: week52_stats(&bars).and_then(|(_, _, p)| p),
        up_ratio_20d: up20.map(|(r, _)| r),
        up_ratio_60d: up60.map(|(r, _)| r),
        up_ratio_250d: up250.map(|(r, _)| r),
        avg_daily_change_250d: up250.map(|(_, avg)| avg),
        prediction,
        valuation,
        disclaimer: COMPREHENSIVE_DISCLAIMER.to_string(),
    })
}

// =============================================================================
// 单元测试
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(date: &str, close: f64) -> HistoricalData {
        HistoricalData {
            symbol: "600519".to_string(),
            date: NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap(),
            open: close,
            close,
            high: close * 1.01,
            low: close * 0.99,
            volume: 1000,
            amount: 1.0,
            amplitude: 0.0,
            turnover_rate: 0.0,
            volume_ratio: 0.0,
            change_percent: 0.0,
            change: 0.0,
        }
    }

    /// 生成从 start 起連续工作日的收盘序列（简化：直接逐日 +1 天，不跳周末，不影响口径断言）
    fn bars_from(start: &str, closes: &[f64]) -> Vec<HistoricalData> {
        let start = NaiveDate::parse_from_str(start, "%Y-%m-%d").unwrap();
        closes
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let mut b = bar("2026-01-01", c);
                b.date = start + Duration::days(i as i64);
                b
            })
            .collect()
    }

    #[test]
    fn canonical_symbol_normalizes_variants() {
        assert_eq!(canonical_symbol("600519"), "600519");
        assert_eq!(canonical_symbol("600519.SH"), "600519");
        assert_eq!(canonical_symbol("sh600519"), "600519");
        assert_eq!(canonical_symbol(" 000001.SZ "), "000001");
        // 提不出 6 位数字：原样 trim 返回
        assert_eq!(canonical_symbol(" abc "), "abc");
        assert_eq!(canonical_symbol("12345"), "12345");
    }

    #[test]
    fn period_change_uses_n_plus_1_bars() {
        // 6 根：5 日涨跌 = 105/100 - 1 = 5%
        let bars = bars_from("2026-06-01", &[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);
        let chg = period_change(&bars, 5).unwrap();
        assert!((chg - 5.0).abs() < 1e-9);
        // 只有 5 根不足以算 5 日（需要 6 根），诚实缺省
        assert!(period_change(&bars[1..], 5).is_none());
        assert!(period_change(&bars, 0).is_none());
    }

    #[test]
    fn ytd_change_uses_last_close_of_previous_year() {
        // 跨年：上年收官 120，最新 132 → +10%
        let mut bars = bars_from("2025-12-29", &[118.0, 120.0]);
        bars.extend(bars_from("2026-01-05", &[125.0, 130.0, 132.0]));
        let chg = ytd_change(&bars).unwrap();
        assert!((chg - 10.0).abs() < 1e-9);
        // 窗口内全是今年（次新股）→ None
        let this_year_only = bars_from("2026-02-01", &[10.0, 11.0, 12.0]);
        assert!(ytd_change(&this_year_only).is_none());
    }

    #[test]
    fn one_year_change_uses_bar_at_or_before_cutoff() {
        // 两根相隔 365 天 + 最新一根：基准取 2025-06-30 的 80 → 100/80 - 1 = 25%
        let mut bars = vec![bar("2025-06-30", 80.0)];
        bars.push(bar("2026-06-29", 98.0));
        bars.push(bar("2026-06-30", 100.0));
        let chg = one_year_change(&bars).unwrap();
        assert!((chg - 25.0).abs() < 1e-9);
        // 窗口不满一年 → None
        let short = bars_from("2026-05-01", &[10.0, 11.0]);
        assert!(one_year_change(&short).is_none());
    }

    #[test]
    fn week52_stats_position_between_high_low() {
        // 收盘 90..110，最新 110：高 110*1.01、低 90*0.99
        let closes: Vec<f64> = (90..=110).map(|v| v as f64).collect();
        let bars = bars_from("2026-01-05", &closes);
        let (high, low, position) = week52_stats(&bars).unwrap();
        assert!((high - 110.0 * 1.01).abs() < 1e-9);
        assert!((low - 90.0 * 0.99).abs() < 1e-9);
        let p = position.unwrap();
        assert!(p > 90.0 && p <= 100.0, "最新价贴近年内高点，位置应接近 100，实际 {p}");
        // 高低重合（横盘一字）→ position None
        let mut flat = bar("2026-06-30", 50.0);
        flat.high = 50.0;
        flat.low = 50.0;
        let (_, _, pos) = week52_stats(&[flat]).unwrap();
        assert!(pos.is_none());
    }

    #[test]
    fn daily_up_ratio_counts_close_over_close() {
        // 100→101→99→102→103：4 个变化里 3 涨 1 跌
        let bars = bars_from("2026-06-01", &[100.0, 101.0, 99.0, 102.0, 103.0]);
        let (ratio, avg) = daily_up_ratio(&bars, 4).unwrap();
        assert!((ratio - 0.75).abs() < 1e-9);
        // 平均日涨跌为正（整体上涨 3%）
        assert!(avg > 0.0);
        // 根数不足 → None
        assert!(daily_up_ratio(&bars, 10).is_none());
    }

    #[test]
    fn nonzero_filters_unrefreshed_values() {
        assert_eq!(nonzero(0.0), None);
        assert_eq!(nonzero(f64::NAN), None);
        assert_eq!(nonzero(12.5), Some(12.5));
        assert_eq!(nonzero(-3.0), Some(-3.0));
    }

    /// 真库冒烟测试（默认忽略；`cargo test -- --ignored` 显式运行）：
    /// 依赖本地 db/stock_data.db 中已有 600519 历史数据。
    /// 覆盖：迁移 SQL 幂等执行 → 收藏 CRUD SQL → 概览全算法 → 一键综合预测全管线。
    #[tokio::test]
    #[ignore]
    async fn smoke_watchlist_and_comprehensive_on_real_db() {
        let pool = create_temp_pool().await.expect("打开本地数据库失败");

        // 1) 迁移 SQL 幂等执行（与 lib.rs 启动逻辑同源）
        let sql = std::fs::read_to_string("migrations/07_watchlist.sql")
            .expect("读取迁移文件失败");
        for statement in sql.split(';') {
            let statement = statement.trim();
            if !statement.is_empty() {
                sqlx::query(statement).execute(&pool).await.expect("迁移执行失败");
            }
        }

        // 2) 收藏 CRUD（与命令同 SQL；用带后缀变体验证 canonical 归一）
        let canonical = canonical_symbol("600519.SH");
        assert_eq!(canonical, "600519");
        sqlx::query(
            "INSERT OR IGNORE INTO watchlist (symbol, sort_order)
             VALUES (?, (SELECT COALESCE(MAX(sort_order), 0) + 1 FROM watchlist))",
        )
        .bind(&canonical)
        .execute(&pool)
        .await
        .expect("添加收藏失败");

        // 3) 概览：600519 行应存在且核心指标非空
        let items = watchlist_overview(&pool).await.expect("概览查询失败");
        let item = items
            .iter()
            .find(|i| i.symbol == "600519")
            .expect("概览中未找到 600519");
        assert!(item.close.is_some(), "现价缺失");
        assert!(item.latest_date.is_some(), "最新日期缺失");
        assert!(item.change_5d.is_some(), "5日涨跌缺失");
        assert!(item.change_1y.is_some(), "一年涨跌缺失");
        assert!(item.week52_position.is_some(), "52周位置缺失");
        println!(
            "概览 OK: {} {} close={:?} 5d={:?} ytd={:?} 1y={:?} pe={:?} 52w={:?} stale={:?}",
            item.symbol, item.name, item.close, item.change_5d, item.change_ytd,
            item.change_1y, item.pe, item.week52_position, item.staleness_days
        );

        // 4) 一键综合预测全管线
        let report = comprehensive_predict("600519".to_string(), Some(5))
            .await
            .expect("综合预测失败");
        assert_eq!(report.symbol, "600519");
        assert!(report.current_price > 0.0);
        assert!(
            (0.0..=1.0).contains(&report.signal_strength),
            "信号强度超出范围: {}",
            report.signal_strength
        );
        assert!(!report.direction.is_empty(), "方向文本为空");
        assert!(!report.current_advice.is_empty(), "建议文本为空");
        assert!(report.interval.is_some(), "缺少校准区间带");
        assert!(!report.disclaimer.is_empty(), "缺少免责标注");
        for r in [report.up_ratio_20d, report.up_ratio_60d, report.up_ratio_250d]
            .into_iter()
            .flatten()
        {
            assert!((0.0..=1.0).contains(&r), "基准率超出 [0,1]: {r}");
        }
        assert_eq!(
            report.prediction.predictions.predictions.len(),
            5,
            "预测天数与请求不符"
        );
        println!(
            "综合预测 OK: {} 方向={} 强度={:.2} 预期={:.2}% 区间={:?} 建议={} 风险={} 基准率20d={:?}",
            report.symbol, report.direction, report.signal_strength,
            report.expected_change_percent,
            report.interval.as_ref().map(|i| (i.lower_change_percent, i.upper_change_percent)),
            report.current_advice, report.risk_level, report.up_ratio_20d
        );

        // 5) 清理：移除测试收藏，数据库净零变化
        sqlx::query("DELETE FROM watchlist WHERE symbol = ?")
            .bind(&canonical)
            .execute(&pool)
            .await
            .expect("清理失败");
    }
}
