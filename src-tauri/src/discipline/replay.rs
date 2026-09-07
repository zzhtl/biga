//! 严格守纪回放：如果每条「必须卖」都执行了，这笔会是什么结果。
//!
//! # 十条口径（这是全项目最容易做出骗自己的数字的地方）
//!
//! 1. **触发判定在收盘后**。第 t 日 `close ≤ stop` → 事件产生于 t 日收盘，最早 t+1 成交。
//!    绝不能"第 t 日按止损价成交"——那是把止损价当成交价，假设零滑点零跳空。
//! 2. **成交价 = 次日开盘价**。不是止损价、不是次日收盘、不是均价。开盘价是唯一一个
//!    "决策已定、执行无选择"的价格；任何盘中价都隐含"我挑了个好时候卖"的后视偏差。
//! 3. **跳空低开低于止损价 → 就按开盘价成交，认下更大的亏损**。这是止损的真实成本。
//!    对称地，跳空高开也按开盘价成交，不做"我会再等等"的择时。
//! 4. **一字跌停不可成交 → 顺延到下一个可成交日**。不实现这条会给出"跌停也能跑掉"的
//!    假象，把守纪收益系统性高估——而高估的正好是最惨的那几笔。
//! 5. **停牌（日期不连续）→ 顺延**到下一根存在的 K 线。
//! 6. **顺延上限 [`MAX_DEFER_BARS`] 根**，超过则标记 `unresolved`，**从统计中剔除并单独计数**，
//!    绝不用最后一根价格硬凑。
//! 7. **回放窗口 = 建仓日 .. 实际清仓日**。窗口内从未触发，或触发后的可成交日已晚于
//!    实际清仓日 → 守纪结果与实际相同，差额为 0。
//!    这意味着复盘只回答**「纪律会不会让你更早离场、离场价差多少」**，
//!    不回答「如果你拿得更久会怎样」——后者需要先决定"拿多久"，而那本身就是一个策略。
//! 8. **只回放已平仓的持仓**。未平仓的只在看板展示浮动，不进对照——
//!    否则等于用今天的价格给一场没打完的赌记分。
//! 9. **费用两边同口径**。守纪路径按该笔实际发生的加权费率反算；用户没填 fee 时两边都按 0，
//!    并在界面标注「未计费用」。不能一边算费用一边不算。
//! 10. **不合成组合净值曲线、不算年化/夏普/胜率提升**。样本量个位数到几十笔，
//!     任何比率统计都是噪音；而合成净值需要假设止损释放的资金去了哪里，
//!     那是一条新的方向规则，违反项目基调。只给逐笔差额和明细。

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::db::models::HistoricalData;
use crate::discipline::exit::evaluate_exit;
use crate::discipline::facts::{build_market_facts, price_limit_percent};
use crate::discipline::rules::DisciplineRules;
use crate::discipline::types::{PositionSnapshot, StopBasis};

/// 因跌停/停牌最多顺延多少根 K 线，超过即认定无法回放
pub const MAX_DEFER_BARS: usize = 10;

/// 回放输入：一笔已平仓持仓的建仓事实
#[derive(Debug, Clone)]
pub struct ReplayEntry {
    pub position_id: String,
    pub symbol: String,
    pub stock_name: String,
    pub open_date: NaiveDate,
    /// 实际清仓日，回放窗口的右端
    pub close_date: NaiveDate,
    pub cost_price: f64,
    pub quantity: i64,
    pub initial_stop: f64,
    pub target_price: Option<f64>,
    /// 该笔实际发生的加权费率（Σfee / Σ成交额）。两边同口径。
    pub fee_rate: f64,
    /// 实际已实现盈亏，用于计算差额
    pub actual_pnl: f64,
}

/// 守纪路径上的一笔成交
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayFill {
    pub rule_code: String,
    /// 收盘判定日
    pub trigger_date: String,
    /// 顺延后的实际成交日
    pub fill_date: String,
    /// = 成交日开盘价
    pub fill_price: f64,
    pub quantity: i64,
    /// 因跌停/停牌顺延了几根
    pub deferred_bars: usize,
    pub defer_reason: Option<String>,
}

/// 止损棘轮轨迹，让复盘明细可核验："止损从 10.00 → 11.20 → 13.50 是怎么抬上去的"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopStep {
    pub date: String,
    pub stop_price: f64,
    pub basis: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayOutcome {
    pub position_id: String,
    pub symbol: String,
    /// 守纪路径的成交序列（可能含分批止盈的多笔）
    pub fills: Vec<ReplayFill>,
    /// true = 顺延超限，无法回放，必须从统计中剔除
    pub unresolved: bool,
    pub unresolved_reason: Option<String>,
    /// true = 窗口内纪律从未让你更早离场，结果与实际相同
    pub same_as_actual: bool,
    pub disciplined_pnl: f64,
    pub actual_pnl: f64,
    /// 守纪 − 实际。正数 = 守纪本可以少亏/多赚这么多
    pub difference: f64,
    pub stop_path: Vec<StopStep>,
}

/// 该日能否卖出：一字跌停无法成交（一字涨停可以卖，所以只判跌停）
fn sellable(bar: &HistoricalData, limit_down_percent: f64) -> bool {
    let one_price_day = (bar.high - bar.low).abs() < f64::EPSILON;
    !(one_price_day && bar.change_percent <= limit_down_percent + 0.3)
}

/// 回放一笔已平仓持仓的严格守纪路径。纯函数。
///
/// `bars` 需时间正序，且应覆盖 `open_date ..= close_date` 之后再多几根（供 T+1 成交与顺延使用）。
pub fn replay_disciplined_exit(
    entry: &ReplayEntry,
    bars: &[HistoricalData],
    rules: &DisciplineRules,
) -> ReplayOutcome {
    let (limit_down_percent, _) = price_limit_percent(&entry.symbol, &entry.stock_name);

    let mut fills: Vec<ReplayFill> = Vec::new();
    let mut stop_path: Vec<StopStep> = Vec::new();
    let mut remaining = entry.quantity;
    let mut stop_price = entry.initial_stop;
    let mut stop_basis = StopBasis::Fixed;
    let mut scale_out_done = 0i32;
    let mut unresolved_reason: Option<String> = None;

    // 扫描窗口：建仓日 ..= 实际清仓日
    let scan_indices: Vec<usize> = bars
        .iter()
        .enumerate()
        .filter(|(_, bar)| bar.date >= entry.open_date && bar.date <= entry.close_date)
        .map(|(index, _)| index)
        .collect();

    for &index in &scan_indices {
        if remaining <= 0 {
            break;
        }
        let snapshot = PositionSnapshot {
            position_id: entry.position_id.clone(),
            symbol: entry.symbol.clone(),
            open_date: entry.open_date.format("%Y-%m-%d").to_string(),
            quantity: remaining,
            cost_price: entry.cost_price,
            initial_stop: entry.initial_stop,
            stop_price,
            stop_basis,
            target_price: entry.target_price,
            // 回放中由 facts 全量重算，这里的值不参与判定
            highest_price: None,
            highest_price_date: None,
            scale_out_done,
            basis_suspect: false,
        };

        let Some(facts) = build_market_facts(
            &entry.symbol,
            &entry.stock_name,
            &bars[..=index],
            Some(entry.open_date),
            bars[index].date,
            rules,
        ) else {
            continue;
        };

        let verdict = evaluate_exit(&snapshot, &facts, rules);

        // 棘轮：把止损状态带到下一根
        if verdict.effective_stop > stop_price {
            stop_price = verdict.effective_stop;
            stop_basis = verdict.stop_basis;
            stop_path.push(StopStep {
                date: facts.latest_date.clone(),
                stop_price,
                basis: stop_basis.as_str().to_string(),
            });
        }

        // 回放里不看降级后的 action（降级只影响当日能否成交，顺延逻辑在下面独立处理），
        // 直接取原始规则命中，避免"一字跌停当天被降级成 Warn 就永远不卖了"。
        let Some(hit) = verdict
            .items
            .iter()
            .find(|hit| {
                matches!(
                    hit.code.as_str(),
                    "EXIT_HARD_STOP"
                        | "EXIT_TRAILING_STOP"
                        | "EXIT_SUPPORT_BREAKDOWN"
                        | "EXIT_TIME_STOP"
                        | "EXIT_SCALE_OUT_1"
                        | "EXIT_SCALE_OUT_2"
                )
            })
            .cloned()
        else {
            continue;
        };

        let is_full_exit = !matches!(hit.code.as_str(), "EXIT_SCALE_OUT_1" | "EXIT_SCALE_OUT_2");
        let want_quantity = if is_full_exit {
            remaining
        } else {
            hit.suggested_quantity.unwrap_or(0).min(remaining)
        };
        if want_quantity <= 0 {
            continue;
        }

        // T+1：从下一根开始找第一个可成交日
        let mut fill_index = None;
        let mut deferred = 0usize;
        let mut defer_reason = None;
        for (offset, bar) in bars[index + 1..].iter().enumerate() {
            if deferred > MAX_DEFER_BARS {
                break;
            }
            if sellable(bar, limit_down_percent) {
                fill_index = Some(index + 1 + offset);
                break;
            }
            deferred += 1;
            defer_reason = Some("一字跌停无法成交".to_string());
        }

        let Some(fill_index) = fill_index else {
            unresolved_reason = Some(format!(
                "{} 触发 {} 后连续 {} 根无法成交（跌停/停牌/数据不足）",
                facts.latest_date, hit.code, deferred
            ));
            break;
        };
        if deferred > MAX_DEFER_BARS {
            unresolved_reason = Some(format!(
                "{} 触发 {} 后顺延超过 {} 根",
                facts.latest_date, hit.code, MAX_DEFER_BARS
            ));
            break;
        }
        // 成交日晚于实际清仓日 → 纪律没能让你更早离场
        if bars[fill_index].date > entry.close_date {
            break;
        }

        fills.push(ReplayFill {
            rule_code: hit.code.clone(),
            trigger_date: facts.latest_date.clone(),
            fill_date: bars[fill_index].date.format("%Y-%m-%d").to_string(),
            fill_price: bars[fill_index].open,
            quantity: want_quantity,
            deferred_bars: deferred,
            defer_reason: defer_reason.clone(),
        });
        remaining -= want_quantity;
        if !is_full_exit {
            scale_out_done += 1;
        }
    }

    let unresolved = unresolved_reason.is_some();
    let same_as_actual = fills.is_empty() && !unresolved;

    // 未在守纪路径卖出的剩余股份，按实际结果计价——窗口内纪律没让你更早走，
    // 这部分的盈亏与实际完全相同，差额只来自被提前卖掉的那些股份。
    let sold_quantity: i64 = fills.iter().map(|fill| fill.quantity).sum();
    let residual_quantity = (entry.quantity - sold_quantity).max(0);
    let actual_pnl_per_share = if entry.quantity > 0 {
        entry.actual_pnl / entry.quantity as f64
    } else {
        0.0
    };

    let disciplined_pnl = if unresolved {
        0.0
    } else {
        let from_fills: f64 = fills
            .iter()
            .map(|fill| {
                let gross = (fill.fill_price - entry.cost_price) * fill.quantity as f64;
                let fee = fill.fill_price * fill.quantity as f64 * entry.fee_rate;
                gross - fee
            })
            .sum();
        from_fills + actual_pnl_per_share * residual_quantity as f64
    };

    ReplayOutcome {
        position_id: entry.position_id.clone(),
        symbol: entry.symbol.clone(),
        fills,
        unresolved,
        unresolved_reason,
        same_as_actual,
        disciplined_pnl,
        actual_pnl: entry.actual_pnl,
        difference: if unresolved { 0.0 } else { disciplined_pnl - entry.actual_pnl },
        stop_path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(date: &str, open: f64, high: f64, low: f64, close: f64, prev_close: f64) -> HistoricalData {
        let change = close - prev_close;
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
            change_percent: if prev_close > 0.0 { change / prev_close * 100.0 } else { 0.0 },
            change,
        }
    }

    /// n 根平稳上行 K 线，收盘 = start + step*i
    fn ramp(n: usize, start: f64, step: f64) -> Vec<HistoricalData> {
        let base = NaiveDate::from_ymd_opt(2026, 1, 5).unwrap();
        (0..n)
            .map(|i| {
                let close = start + step * i as f64;
                let prev = if i == 0 { close } else { start + step * (i - 1) as f64 };
                bar(
                    &(base + chrono::Duration::days(i as i64)).format("%Y-%m-%d").to_string(),
                    prev,
                    close.max(prev) + 0.2,
                    close.min(prev) - 0.2,
                    close,
                    prev,
                )
            })
            .collect()
    }

    fn entry_at(bars: &[HistoricalData], open_index: usize, close_index: usize, cost: f64) -> ReplayEntry {
        ReplayEntry {
            position_id: "p1".to_string(),
            symbol: "600519".to_string(),
            stock_name: "贵州茅台".to_string(),
            open_date: bars[open_index].date,
            close_date: bars[close_index].date,
            cost_price: cost,
            quantity: 1000,
            initial_stop: cost * 0.92,
            target_price: Some(cost * 1.2),
            fee_rate: 0.0,
            actual_pnl: 0.0,
        }
    }

    #[test]
    fn no_trigger_means_identical_to_actual() {
        let bars = ramp(60, 20.0, 0.05);
        let entry = entry_at(&bars, 30, 50, 21.5);
        let out = replay_disciplined_exit(&entry, &bars, &DisciplineRules::default());
        assert!(out.same_as_actual, "平稳上行且未达任何阈值时，纪律不应产生更早的离场");
        assert_eq!(out.difference, 0.0);
    }

    #[test]
    fn hard_stop_fills_at_next_open_not_at_the_stop_price() {
        let mut bars = ramp(40, 20.0, 0.0);
        // 第 30 根暴跌到止损下方，第 31 根低开
        bars[30].close = 17.0;
        bars[30].change = -3.0;
        bars[30].change_percent = -15.0;
        bars[30].low = 17.0;
        bars[31].open = 16.5;

        let mut entry = entry_at(&bars, 20, 39, 20.0);
        entry.initial_stop = 18.4; // 20 × 0.92
        let out = replay_disciplined_exit(&entry, &bars, &DisciplineRules::default());

        let fill = out.fills.first().expect("跌破止损必须产生一笔守纪成交");
        assert_eq!(fill.rule_code, "EXIT_HARD_STOP");
        assert_eq!(fill.trigger_date, "2026-02-04", "触发日应是收盘跌破的那一天");
        assert!(
            (fill.fill_price - 16.5).abs() < 1e-9,
            "成交价必须是次日开盘价 16.5，不是止损价 18.4——跳空的代价要如实计入"
        );
    }

    #[test]
    fn limit_down_lock_defers_the_fill() {
        let mut bars = ramp(40, 20.0, 0.0);
        bars[30].close = 17.0;
        bars[30].change = -3.0;
        bars[30].change_percent = -15.0;
        // 次日与第三日一字跌停，第四日才能成交
        for index in [31usize, 32] {
            let prev_close = bars[index - 1].close;
            let price = prev_close * 0.9;
            bars[index].open = price;
            bars[index].high = price;
            bars[index].low = price;
            bars[index].close = price;
            bars[index].change = price - prev_close;
            bars[index].change_percent = -10.0;
        }
        bars[33].open = 13.0;
        bars[33].high = 13.5;
        bars[33].low = 12.8;

        let mut entry = entry_at(&bars, 20, 39, 20.0);
        entry.initial_stop = 18.4;
        let out = replay_disciplined_exit(&entry, &bars, &DisciplineRules::default());

        let fill = out.fills.first().expect("应有成交");
        assert_eq!(fill.deferred_bars, 2, "两个一字跌停日必须顺延，不能假装能卖掉");
        assert!((fill.fill_price - 13.0).abs() < 1e-9, "成交价应是第一个可成交日的开盘价");
    }

    #[test]
    fn endless_limit_down_marks_unresolved_instead_of_faking_a_price() {
        let mut bars = ramp(60, 20.0, 0.0);
        bars[30].close = 17.0;
        bars[30].change = -3.0;
        bars[30].change_percent = -15.0;
        for index in 31..=59 {
            let prev_close = bars[index - 1].close;
            let price = prev_close * 0.9;
            bars[index].open = price;
            bars[index].high = price;
            bars[index].low = price;
            bars[index].close = price;
            bars[index].change = price - prev_close;
            bars[index].change_percent = -10.0;
        }
        let mut entry = entry_at(&bars, 20, 59, 20.0);
        entry.initial_stop = 18.4;
        let out = replay_disciplined_exit(&entry, &bars, &DisciplineRules::default());

        assert!(out.unresolved, "连续跌停超过顺延上限必须标记为无法回放");
        assert_eq!(out.difference, 0.0, "无法回放的笔不能贡献差额，必须从统计剔除");
        assert!(out.unresolved_reason.is_some());
    }

    #[test]
    fn stop_path_records_the_ratchet() {
        // 持续上涨触发移动止盈，止损应被逐步抬高
        let bars = ramp(80, 20.0, 0.3);
        let entry = entry_at(&bars, 5, 79, 21.0);
        let out = replay_disciplined_exit(&entry, &bars, &DisciplineRules::default());
        assert!(
            out.stop_path.len() >= 2,
            "上涨过程中止损应被多次上移，轨迹要可核验"
        );
        let prices: Vec<f64> = out.stop_path.iter().map(|step| step.stop_price).collect();
        assert!(
            prices.windows(2).all(|pair| pair[1] > pair[0]),
            "棘轮轨迹必须单调递增，出现下移说明止损被放宽了"
        );
    }

    #[test]
    fn fee_rate_is_applied_to_the_disciplined_side() {
        let mut bars = ramp(40, 20.0, 0.0);
        bars[30].close = 17.0;
        bars[30].change = -3.0;
        bars[30].change_percent = -15.0;
        bars[31].open = 17.0;

        let mut entry = entry_at(&bars, 20, 39, 20.0);
        entry.initial_stop = 18.4;
        entry.fee_rate = 0.001;
        let out = replay_disciplined_exit(&entry, &bars, &DisciplineRules::default());
        let expected = (17.0 - 20.0) * 1000.0 - 17.0 * 1000.0 * 0.001;
        assert!(
            (out.disciplined_pnl - expected).abs() < 1e-6,
            "守纪侧必须扣与实际同口径的费用"
        );
    }
}
