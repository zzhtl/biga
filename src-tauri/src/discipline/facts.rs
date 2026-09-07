//! 从日线序列提取 [`MarketFacts`]。
//!
//! # 三条刻意不对称的口径（改动前先读完）
//!
//! 1. **触发判定用 `close`，不用盘中价、不用 realtime 快照。**
//!    盘中价会让同一天不同时刻结论不同，复盘时无法重放；盘中最低价会被插针大量误触发。
//!    代价是最快收盘后才报警 —— 但 T+1 市场本来就只能明天卖，这个延迟是真实的。
//!
//! 2. **持仓最高价用日线 `high`，不用 `close`。**
//!    移动止盈的语义是"从最高点回撤 X%"，最高点就是盘中最高。用 `close` 记峰值会
//!    系统性低估回撤、让移动止盈晚触发 —— 那正是自欺的方向。
//!
//! 3. 两条合起来是有意的不对称：记峰值往"回撤不被低估"偏，判触发往"不用噪音制造
//!    无法复现的信号"偏。**两条都朝保守方向**。请不要"统一"掉它们。
//!
//! # 最高价全量重算而非增量 max
//!
//! `highest = max(highest, today_high)` 一旦因漏刷新、跳日、用户改建仓日而写错，
//! 就永久污染且无法自愈。全量重算是幂等的、可被单测钉死、能被"重新扫描"修复。
//! 持仓最多几十只 × 几百根 K 线，成本可忽略。

use chrono::NaiveDate;

use crate::db::models::HistoricalData;
use crate::discipline::rules::DisciplineRules;
use crate::discipline::types::MarketFacts;
use crate::prediction::analysis::support_resistance::calculate_support_resistance;
use crate::prediction::indicators::atr::calculate_atr;
use crate::utils::math::calculate_ma;

/// MA20 至少需要 20 根；`calculate_ma` 在长度不足时会静默返回最后一个值
/// （等于收盘价），必须自己拦住，否则"跌破 MA20"永远不会触发。
const MA20_PERIOD: usize = 20;

/// 判定价格基准变化的相对阈值。除权除息会让 K 线跳空，
/// 但接口给的 `pre_close` 是除权后基准，两者对不上即可识别。
const BASIS_CHANGE_TOLERANCE: f64 = 0.02;

/// A 股涨跌停限幅（百分比，返回 `(跌停, 涨停)`）。
///
/// 与 `professional_engine::get_stock_price_limits` 的区别：那个函数为预测留了裕度
/// （主板返回 ±9.5），这里需要**真实**限幅来判断"一字板是否可成交"，所以单独实现。
/// 刻意不改动预测引擎（外科手术式改动）。
pub fn price_limit_percent(symbol: &str, name: &str) -> (f64, f64) {
    let code = symbol.trim_start_matches(|c: char| !c.is_ascii_digit());
    // 创业板 / 科创板即使被 ST 也是 ±20%，所以先判板块
    if code.starts_with("688") || code.starts_with("300") || code.starts_with("301") {
        return (-20.0, 20.0);
    }
    if name.to_uppercase().contains("ST") {
        return (-5.0, 5.0);
    }
    // 北交所 ±30%。库内目前无此类标的，保留分支避免误判为主板（known gap）
    if code.starts_with('4') || code.starts_with('8') {
        return (-30.0, 30.0);
    }
    (-10.0, 10.0)
}

/// 检测价格基准变化（除权 / 除息 / 复权口径切换）。
///
/// 数据源是 `hs/history/{symbol}/d/n`（**不复权**），10 送 10 后价格腰斩而成本价不变，
/// 会立刻假触发硬止损。检测不需要额外数据源：接口的 `pre_close` 就是 `close - change`，
/// 正常情况下它应等于上一根 K 线的 `close`，对不上即为基准变化。
///
/// 返回最近一次疑似发生的日期。
pub fn detect_basis_change(bars: &[HistoricalData]) -> Option<String> {
    let mut latest = None;
    for pair in bars.windows(2) {
        let (prev, current) = (&pair[0], &pair[1]);
        if prev.close <= 0.0 {
            continue;
        }
        let implied_prev_close = current.close - current.change;
        if implied_prev_close <= 0.0 {
            continue;
        }
        let drift = (implied_prev_close - prev.close).abs() / prev.close;
        if drift > BASIS_CHANGE_TOLERANCE {
            latest = Some(current.date.format("%Y-%m-%d").to_string());
        }
    }
    latest
}

/// 构造市场事实。数据不足返回 `None` —— 沿用 `watchlist.rs` 的诚实缺省惯例，
/// 绝不用不完整窗口凑数。
///
/// `bars` 必须时间正序。`open_date` 为 `None` 时（买入准入场景，尚无持仓）
/// 不计算持有天数与持仓最高价。
pub fn build_market_facts(
    symbol: &str,
    stock_name: &str,
    bars: &[HistoricalData],
    open_date: Option<NaiveDate>,
    today: NaiveDate,
    rules: &DisciplineRules,
) -> Option<MarketFacts> {
    let last = bars.last()?;
    let closes: Vec<f64> = bars.iter().map(|bar| bar.close).collect();
    let highs: Vec<f64> = bars.iter().map(|bar| bar.high).collect();
    let lows: Vec<f64> = bars.iter().map(|bar| bar.low).collect();

    // calculate_atr 长度不足时返回 0.0，必须映射成 None——当成 `cost - 0` 用会让止损等于成本价
    let atr = Some(calculate_atr(&highs, &lows, &closes, rules.atr_period)).filter(|v| *v > 0.0);
    let atr_percent = atr.filter(|_| last.close > 0.0).map(|v| v / last.close * 100.0);

    let support_resistance = calculate_support_resistance(&closes, &highs, &lows, last.close);
    let nearest_support = support_resistance
        .support_levels
        .iter()
        .copied()
        .filter(|level| *level > 0.0 && *level < last.close)
        .max_by(f64::total_cmp);
    // 现价上方最近的支撑 = 刚刚被跌破的那一条。破位判定要用它，不是下方那条。
    // 取法与 professional_engine/signals.rs:275-282 的 broken_support 一致。
    let broken_support = support_resistance
        .support_levels
        .iter()
        .copied()
        .filter(|level| *level > last.close)
        .min_by(f64::total_cmp);

    // calculate_ma 在长度不足时返回最后一个值（= 收盘价），必须自己拦
    let ma20 = (closes.len() >= MA20_PERIOD).then(|| calculate_ma(&closes, MA20_PERIOD));

    let suspected_corporate_action = detect_basis_change(bars);

    // 持仓最高价：从建仓日（或基准变化日，取更晚者）起全量重算
    let high_window_start = match (&open_date, &suspected_corporate_action) {
        (Some(opened), Some(changed)) => NaiveDate::parse_from_str(changed, "%Y-%m-%d")
            .ok()
            .map(|changed| changed.max(*opened))
            .or(Some(*opened)),
        (Some(opened), None) => Some(*opened),
        _ => None,
    };
    let (holding_high, holding_high_date) = match high_window_start {
        Some(start) => bars
            .iter()
            .filter(|bar| bar.date >= start)
            .max_by(|a, b| a.high.total_cmp(&b.high))
            .map(|bar| (Some(bar.high), Some(bar.date.format("%Y-%m-%d").to_string())))
            .unwrap_or((None, None)),
        None => (None, None),
    };

    let holding_trading_days = open_date
        .map(|start| bars.iter().filter(|bar| bar.date >= start).count() as i64)
        .unwrap_or(0);

    let (limit_down_percent, limit_up_percent) = price_limit_percent(symbol, stock_name);
    // 一字板：最高 == 最低（全天只有一个价），且贴在限幅上。留 0.3pct 容差应对四舍五入。
    let one_price_day = (last.high - last.low).abs() < f64::EPSILON;
    let is_limit_down_locked = one_price_day && last.change_percent <= limit_down_percent + 0.3;
    let is_limit_up_locked = one_price_day && last.change_percent >= limit_up_percent - 0.3;

    Some(MarketFacts {
        symbol: symbol.to_string(),
        latest_date: last.date.format("%Y-%m-%d").to_string(),
        staleness_days: (today - last.date).num_days().max(0),
        close: last.close,
        open: last.open,
        high: last.high,
        low: last.low,
        prev_close: last.close - last.change,
        volume_ratio: last.volume_ratio,
        change_percent: last.change_percent,
        atr,
        atr_percent,
        nearest_support,
        broken_support,
        ma20,
        holding_trading_days,
        holding_high,
        holding_high_date,
        limit_up_percent,
        limit_down_percent,
        is_limit_down_locked,
        is_limit_up_locked,
        bars_used: bars.len(),
        suspected_corporate_action,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(date: &str, open: f64, high: f64, low: f64, close: f64, change: f64) -> HistoricalData {
        HistoricalData {
            symbol: "600519".to_string(),
            date: NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap(),
            open,
            close,
            high,
            low,
            volume: 1_000_000,
            amount: close * 1_000_000.0,
            amplitude: 0.0,
            turnover_rate: 1.0,
            volume_ratio: 1.0,
            change_percent: if close - change > 0.0 {
                change / (close - change) * 100.0
            } else {
                0.0
            },
            change,
        }
    }

    /// 连续 n 根，收盘从 start 起每日 +step，pre_close 连续（无除权）
    fn series(n: usize, start: f64, step: f64) -> Vec<HistoricalData> {
        let base = NaiveDate::from_ymd_opt(2026, 1, 5).unwrap();
        (0..n)
            .map(|i| {
                let close = start + step * i as f64;
                let prev = if i == 0 { close } else { start + step * (i - 1) as f64 };
                bar(
                    &(base + chrono::Duration::days(i as i64))
                        .format("%Y-%m-%d")
                        .to_string(),
                    prev,
                    close.max(prev) + 0.5,
                    close.min(prev) - 0.5,
                    close,
                    close - prev,
                )
            })
            .collect()
    }

    #[test]
    fn star_and_st_boards_get_their_real_limits() {
        assert_eq!(price_limit_percent("688981", "中芯国际"), (-20.0, 20.0));
        assert_eq!(price_limit_percent("300750", "宁德时代"), (-20.0, 20.0));
        assert_eq!(price_limit_percent("600519", "贵州茅台"), (-10.0, 10.0));
        assert_eq!(price_limit_percent("600666", "*ST奥瑞"), (-5.0, 5.0));
        assert_eq!(
            price_limit_percent("300123", "ST太空"),
            (-20.0, 20.0),
            "创业板即使 ST 也是 ±20%"
        );
    }

    #[test]
    fn basis_change_is_detected_from_pre_close_mismatch() {
        let mut bars = series(30, 20.0, 0.1);
        // 制造 10 送 10：收盘腰斩，但接口的 pre_close（= close − change）也是除权后基准，
        // 所以 change_percent 看起来正常，只有与上一根 close 对比才能发现
        let last = bars.last_mut().unwrap();
        last.close = 11.5;
        last.change = 0.1; // 隐含 pre_close = 11.4，而上一根 close ≈ 22.8
        assert!(
            detect_basis_change(&bars).is_some(),
            "除权导致的基准跳变必须被检出，否则会假触发硬止损"
        );
    }

    #[test]
    fn continuous_series_has_no_false_basis_change() {
        assert!(
            detect_basis_change(&series(60, 20.0, 0.15)).is_none(),
            "正常连续行情不应误报除权"
        );
    }

    #[test]
    fn holding_high_uses_daily_high_and_is_recomputed_in_full() {
        let bars = series(40, 20.0, 0.1);
        let opened = bars[10].date;
        let facts = build_market_facts(
            "600519",
            "贵州茅台",
            &bars,
            Some(opened),
            bars.last().unwrap().date,
            &DisciplineRules::default(),
        )
        .unwrap();
        let expected = bars
            .iter()
            .filter(|b| b.date >= opened)
            .map(|b| b.high)
            .fold(f64::MIN, f64::max);
        assert_eq!(facts.holding_high, Some(expected), "最高价必须取日线 high 的全量最大值");
        assert_eq!(facts.holding_trading_days, 30, "持有交易日数 = 建仓日起的 K 线根数");
    }

    #[test]
    fn ma20_is_none_when_window_too_short() {
        let bars = series(10, 20.0, 0.1);
        let facts = build_market_facts(
            "600519",
            "贵州茅台",
            &bars,
            None,
            bars.last().unwrap().date,
            &DisciplineRules::default(),
        )
        .unwrap();
        assert!(
            facts.ma20.is_none(),
            "calculate_ma 长度不足会静默返回收盘价，必须拦住否则跌破 MA20 永不触发"
        );
        assert!(facts.atr.is_none(), "K 线不足 14 根时 ATR 必须是 None 而非 0.0");
    }

    #[test]
    fn limit_down_lock_requires_a_single_price_day() {
        let mut bars = series(30, 20.0, -0.1);
        let last = bars.last_mut().unwrap();
        let prev_close = last.close - last.change;
        last.close = prev_close * 0.9;
        last.open = last.close;
        last.high = last.close;
        last.low = last.close;
        last.change = last.close - prev_close;
        last.change_percent = -10.0;
        let facts = build_market_facts(
            "600519",
            "贵州茅台",
            &bars,
            None,
            bars.last().unwrap().date,
            &DisciplineRules::default(),
        )
        .unwrap();
        assert!(facts.is_limit_down_locked, "一字跌停应被识别为不可成交");
    }

    #[test]
    fn empty_bars_yield_none_instead_of_zeros() {
        assert!(build_market_facts(
            "600519",
            "贵州茅台",
            &[],
            None,
            NaiveDate::from_ymd_opt(2026, 1, 5).unwrap(),
            &DisciplineRules::default(),
        )
        .is_none());
    }
}
