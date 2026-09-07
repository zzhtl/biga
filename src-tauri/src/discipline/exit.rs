//! 卖出纪律：什么条件下不该继续持有。
//!
//! # 「必须卖，卖错也要卖」
//!
//! 本模块输出的 [`DisciplineAction::MustSell`] 是无条件的。它不问"会不会反弹"——
//! 那是方向判断，本项目已有结论：单股方向不可预测。它只问一件事：
//! **这笔持仓是否还符合你建仓时给自己定的条件**。不符合就走人，对错事后再说。
//!
//! 卖出规则的误判代价是"少赚"，买入规则的误判代价是"亏本金"。这个不对称是有意的：
//! 卖出侧宁可多触发，买入侧宁可多拦截。
//!
//! # 降级链（顺序固定，改动前先读）
//!
//! 1. **价格基准变化（除权/除息）→ 全部卖出规则挂起**。
//!    数据源不复权，10 送 10 后收盘价腰斩而成本价不变，`close <= stop` 会立刻成立——
//!    硬止损恰恰是**最容易**被假触发的那一条，不是"不受影响"的那一条。
//!    宁可不判，也不能拿错基准逼人割肉。
//! 2. **一字跌停 → 强制动作降级为提示**（今天根本卖不掉，报"必须卖"只是制造焦虑）。
//! 3. **T+1 锁定 → 保持必须卖，但文案改成"明日开盘执行"**。

use crate::discipline::rules::DisciplineRules;
use crate::discipline::stop::{compute_stop, StopInput};
use crate::discipline::types::{
    finalize, item, push_item, strongest_action, DisciplineAction, DisciplineCategory,
    DisciplineItem, ExitVerdict, MarketFacts, PositionMetrics, PositionSnapshot, StopBasis,
};
use crate::prediction::analysis::support_resistance::is_breakdown;
use crate::prediction::types::RiskLevel;

/// 卖出侧唯一入口。纯函数：同样的输入永远同样的输出，可在回放里逐日重跑。
pub fn evaluate_exit(
    position: &PositionSnapshot,
    facts: &MarketFacts,
    rules: &DisciplineRules,
) -> ExitVerdict {
    let cost = position.cost_price;
    let close = facts.close;

    // 止损用**本次重算**的持仓最高价，不用库里可能过期的那个
    let stop = compute_stop(
        &StopInput {
            cost_price: cost,
            close,
            previous_stop: Some(position.stop_price),
            previous_basis: Some(position.stop_basis),
            atr: facts.atr,
            nearest_support: facts.nearest_support,
            highest_price: facts.holding_high,
        },
        rules,
    );

    let gain_pct = if cost > 0.0 { (close / cost - 1.0) * 100.0 } else { 0.0 };
    let unrealized_pnl = (close - cost) * position.quantity as f64;
    let drawdown_from_high_percent = facts
        .holding_high
        .filter(|high| *high > 0.0)
        .map(|high| (high - close) / high * 100.0);

    let metrics = PositionMetrics {
        market_value: close * position.quantity as f64,
        unrealized_pnl,
        unrealized_pnl_percent: gain_pct,
        distance_to_stop_percent: if close > 0.0 {
            (close - stop.stop_price) / close * 100.0
        } else {
            0.0
        },
        drawdown_from_high_percent,
        risk_exposure: ((close - stop.stop_price) * position.quantity as f64).max(0.0),
        holding_trading_days: facts.holding_trading_days,
    };

    let t1_locked = position.open_date == facts.latest_date;
    let tradable = !facts.is_limit_down_locked;
    let basis_changed = position.basis_suspect || facts.suspected_corporate_action.is_some();

    let mut items: Vec<DisciplineItem> = Vec::new();

    // —— 降级链第 1 层：基准变化时只报数据问题，不产出任何卖出指令 ——
    if basis_changed {
        let when = facts
            .suspected_corporate_action
            .clone()
            .unwrap_or_else(|| "此前".to_string());
        push_item(
            &mut items,
            item(
                "DATA_PRICE_BASIS_CHANGED",
                DisciplineCategory::Data,
                RiskLevel::High,
                DisciplineAction::Warn,
                "价格基准已变化，卖出纪律已挂起",
                "行情源为不复权日线，检测到除权/除息导致的价格跳变。此时收盘价与你的成本价\
                 不在同一基准上，任何止损判定都会失真。请手工修正持仓成本价与止损价后恢复裁决。",
                vec![
                    format!("疑似发生日 {when}"),
                    format!("当前收盘 {close:.3}，登记成本 {cost:.3}"),
                    "修正入口：持仓行 → 除权修正".to_string(),
                ],
            ),
        );
        let level = finalize(&mut items);
        return ExitVerdict {
            position_id: position.position_id.clone(),
            symbol: position.symbol.clone(),
            event_date: facts.latest_date.clone(),
            level,
            level_label: level.label().to_string(),
            action: DisciplineAction::Warn,
            effective_stop: position.stop_price,
            stop_basis: position.stop_basis,
            stop_raised_to: None,
            highest_price: facts.holding_high,
            highest_price_date: facts.holding_high_date.clone(),
            metrics,
            t1_locked,
            tradable,
            items,
        };
    }

    // —— 硬止损 / 移动止盈：同一个比较，按胜出的 basis 分成两条 code ——
    if close <= stop.stop_price {
        let is_trailing = stop.basis == StopBasis::Trailing;
        let mut evidence = vec![
            format!("收盘 {close:.3} ≤ 止损 {:.3}", stop.stop_price),
            format!("止损依据：{}", stop.basis.label()),
            format!("成本 {cost:.3}，浮动盈亏 {gain_pct:+.2}%"),
        ];
        evidence.extend(stop.evidence.iter().cloned());
        if is_trailing {
            if let (Some(high), Some(drawdown)) = (facts.holding_high, drawdown_from_high_percent) {
                evidence.insert(
                    1,
                    format!(
                        "持仓最高 {high:.3}（{}），已回撤 {drawdown:.2}%，阈值 {:.0}%",
                        facts.holding_high_date.clone().unwrap_or_default(),
                        rules.trail_pct
                    ),
                );
            }
        }
        push_item(
            &mut items,
            item(
                if is_trailing { "EXIT_TRAILING_STOP" } else { "EXIT_HARD_STOP" },
                DisciplineCategory::Exit,
                RiskLevel::High,
                DisciplineAction::MustSell,
                if is_trailing { "移动止盈触发，必须清仓" } else { "跌破止损，必须清仓" },
                if is_trailing {
                    "从持仓最高点的回撤已达阈值。利润是账面数字，落袋才算数——现在走，\
                     把这一段赚到的留住。"
                } else {
                    "收盘已跌破生效止损价。这条线是你建仓时自己定的，现在执行它。\
                     卖错的代价是少赚，不卖错的代价是本金。"
                },
                evidence,
            )
            .at(close, stop.stop_price),
        );
    }

    // —— 破位止损：建仓依据已被事实证伪 ——
    // 用现价上方刚被跌破的那条支撑（与 professional_engine/signals.rs 的取法一致），
    // 并要求量比确认，避免缩量假摔。
    if let Some(support) = facts.broken_support {
        if close < cost && is_breakdown(close, support, facts.volume_ratio) {
            push_item(
                &mut items,
                item(
                    "EXIT_SUPPORT_BREAKDOWN",
                    DisciplineCategory::Exit,
                    RiskLevel::High,
                    DisciplineAction::MustSell,
                    "放量跌破关键支撑，入场依据已失效",
                    "这不是对后市的判断，而是对前提的核对：你建仓时依赖的技术位已被放量击穿，\
                     继续持有等于在一个已经不成立的理由上加注。",
                    vec![
                        format!("支撑位 {support:.3}，收盘 {close:.3}"),
                        format!("跌破幅度 {:.2}%", (support - close) / support * 100.0),
                        format!("量比 {:.2}（放量确认阈值 1.2）", facts.volume_ratio),
                        format!("当前浮亏 {gain_pct:+.2}%"),
                    ],
                )
                .at(close, support),
            );
        }
    }

    // —— 时间止损：机会成本也是成本 ——
    if facts.holding_trading_days >= rules.time_stop_bars && gain_pct < rules.time_stop_min_gain_pct
    {
        push_item(
            &mut items,
            item(
                "EXIT_TIME_STOP",
                DisciplineCategory::Exit,
                RiskLevel::Medium,
                DisciplineAction::MustSell,
                "时间止损：持有超期且未达预期",
                "钱被套在一只不涨的票上，本身就是亏损——它没能去做别的事。\
                 不是它跌了才叫错，是它没按你预期的节奏走就该换。",
                vec![
                    format!(
                        "已持有 {} 个交易日（阈值 {}）",
                        facts.holding_trading_days, rules.time_stop_bars
                    ),
                    format!(
                        "浮动盈亏 {gain_pct:+.2}%，未达 {:+.1}%",
                        rules.time_stop_min_gain_pct
                    ),
                ],
            )
            .at(close, cost * (1.0 + rules.time_stop_min_gain_pct / 100.0)),
        );
    }

    // —— 分批止盈 ——
    // 卖出可以是零股，但统一按 100 股向下取整，避免留下无法再拆的碎股。
    for (tier_index, tier_gain) in rules.scale_out_tiers.iter().copied().enumerate() {
        if gain_pct < tier_gain || position.scale_out_done as usize > tier_index {
            continue;
        }
        let suggested = ((position.quantity as f64 * rules.scale_out_fraction) / 100.0).floor()
            as i64
            * 100;
        if suggested <= 0 {
            continue;
        }
        push_item(
            &mut items,
            item(
                if tier_index == 0 { "EXIT_SCALE_OUT_1" } else { "EXIT_SCALE_OUT_2" },
                DisciplineCategory::Exit,
                RiskLevel::Medium,
                DisciplineAction::MustReduce,
                &format!("浮盈达 {tier_gain:.0}%，必须减仓"),
                "分批兑现。留下的仓位继续跑，卖掉的部分锁死收益——这样既不会在回调里\
                 把利润全吐回去，也不会因为全卖了而错过后面的主升段。",
                vec![
                    format!("浮盈 {gain_pct:+.2}%（档位 {tier_gain:.0}%）"),
                    format!("当前 {} 股，建议减 {suggested} 股", position.quantity),
                    format!("减仓比例 {:.0}%", rules.scale_out_fraction * 100.0),
                ],
            )
            .at(close, cost * (1.0 + tier_gain / 100.0))
            .with_quantity(suggested),
        );
        break; // 一次只推进一档
    }

    // —— 提示类 ——
    if let Some(ma20) = facts.ma20 {
        if close < ma20 && close > cost {
            push_item(
                &mut items,
                item(
                    "EXIT_MA20_BREAK",
                    DisciplineCategory::Exit,
                    RiskLevel::Medium,
                    DisciplineAction::Warn,
                    "盈利仓位跌破 MA20",
                    "中期均线失守。不强制离场，但这通常是移动止盈将要触发的前兆，\
                     可以开始留意。",
                    vec![
                        format!("收盘 {close:.3} < MA20 {ma20:.3}"),
                        format!("当前仍浮盈 {gain_pct:+.2}%"),
                    ],
                )
                .at(close, ma20),
            );
        }
    }

    if let Some(raised) = stop.raised_to {
        push_item(
            &mut items,
            item(
                "EXIT_STOP_SHOULD_RISE",
                DisciplineCategory::Exit,
                RiskLevel::Low,
                DisciplineAction::Warn,
                "止损位已上移",
                "棘轮生效：止损只上移不下移。上移后本笔的最大亏损随之收窄。",
                vec![
                    format!("{:.3} → {raised:.3}", position.stop_price),
                    format!("新依据：{}", stop.basis.label()),
                ],
            ),
        );
    }

    if position.target_price.is_none() {
        push_item(
            &mut items,
            item(
                "DATA_TARGET_MISSING",
                DisciplineCategory::Data,
                RiskLevel::Medium,
                DisciplineAction::Warn,
                "未登记目标价",
                "没有目标价就算不出盈亏比，这笔持仓无法进入复盘对照。请补录。",
                vec!["复盘的逐笔差额需要目标价作为出场参照".to_string()],
            ),
        );
    }

    if facts.staleness_days > rules.stale_days {
        let severity = if facts.staleness_days > 10 { RiskLevel::High } else { RiskLevel::Medium };
        push_item(
            &mut items,
            item(
                "DATA_STALE_QUOTE",
                DisciplineCategory::Data,
                severity,
                DisciplineAction::Warn,
                "行情数据陈旧",
                "裁决基于较旧的 K 线，真实价格可能已明显偏离。请先刷新历史数据，\
                 或确认该股是否停牌。",
                vec![
                    format!("最新 K 线 {}", facts.latest_date),
                    format!("距今 {} 个自然日", facts.staleness_days),
                ],
            ),
        );
    }

    if facts.atr.is_none() || facts.holding_high.is_none() {
        push_item(
            &mut items,
            item(
                "DATA_INSUFFICIENT_WINDOW",
                DisciplineCategory::Data,
                RiskLevel::Low,
                DisciplineAction::Warn,
                "K 线窗口不足，部分止损候选未参与",
                "数据不足时该候选直接跳过，不用不完整窗口凑数。止损仍由固定百分比兜底。",
                vec![
                    format!("可用 K 线 {} 根", facts.bars_used),
                    format!("ATR 可用：{}", if facts.atr.is_some() { "是" } else { "否" }),
                    format!(
                        "持仓最高价可用：{}",
                        if facts.holding_high.is_some() { "是" } else { "否" }
                    ),
                ],
            ),
        );
    }

    if facts.is_limit_down_locked {
        push_item(
            &mut items,
            item(
                "EXIT_LIMIT_DOWN_UNFILLABLE",
                DisciplineCategory::Data,
                RiskLevel::Medium,
                DisciplineAction::Warn,
                "一字跌停，今日无法成交",
                "强制动作已降级为提示。明日开盘继续挂单，不要因为「卖不掉」就放弃这条纪律。",
                vec![
                    format!("收盘 {close:.3}，涨跌幅 {:.2}%", facts.change_percent),
                    format!("跌停限幅 {:.1}%", facts.limit_down_percent),
                ],
            ),
        );
    }

    if t1_locked {
        push_item(
            &mut items,
            item(
                "EXIT_T1_LOCKED",
                DisciplineCategory::Data,
                RiskLevel::Low,
                DisciplineAction::Warn,
                "建仓当日，T+1 不可卖出",
                "A 股 T+1：今日买入的股份最早明日才能卖。若已触发必卖，明日开盘执行。",
                vec![format!("建仓日 {}", position.open_date)],
            ),
        );
    }

    // —— 降级链第 2 层：不可成交时把强制动作降为提示 ——
    if !tradable {
        for entry in items.iter_mut() {
            if entry.action.requires_resolution() {
                entry.action = DisciplineAction::Warn;
                entry.detail = format!("{}（今日一字跌停无法成交，明日开盘继续执行）", entry.detail);
            }
        }
    }

    let level = finalize(&mut items);
    let action = strongest_action(&items);

    ExitVerdict {
        position_id: position.position_id.clone(),
        symbol: position.symbol.clone(),
        event_date: facts.latest_date.clone(),
        level,
        level_label: level.label().to_string(),
        action,
        effective_stop: stop.stop_price,
        stop_basis: stop.basis,
        stop_raised_to: stop.raised_to,
        highest_price: facts.holding_high,
        highest_price_date: facts.holding_high_date.clone(),
        metrics,
        t1_locked,
        tradable,
        items,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(close: f64) -> MarketFacts {
        MarketFacts {
            symbol: "600519".to_string(),
            latest_date: "2026-03-02".to_string(),
            staleness_days: 0,
            close,
            open: close,
            high: close,
            low: close,
            prev_close: close,
            volume_ratio: 1.0,
            change_percent: 0.0,
            atr: Some(1.0),
            atr_percent: Some(1.0),
            nearest_support: None,
            broken_support: None,
            ma20: Some(close),
            holding_trading_days: 5,
            holding_high: Some(close),
            holding_high_date: Some("2026-03-02".to_string()),
            limit_up_percent: 10.0,
            limit_down_percent: -10.0,
            is_limit_down_locked: false,
            is_limit_up_locked: false,
            bars_used: 250,
            suspected_corporate_action: None,
        }
    }

    fn position(cost: f64, stop: f64) -> PositionSnapshot {
        PositionSnapshot {
            position_id: "p1".to_string(),
            symbol: "600519".to_string(),
            open_date: "2026-02-24".to_string(),
            quantity: 1000,
            cost_price: cost,
            initial_stop: stop,
            stop_price: stop,
            stop_basis: StopBasis::Fixed,
            target_price: Some(cost * 1.2),
            highest_price: None,
            highest_price_date: None,
            scale_out_done: 0,
            basis_suspect: false,
        }
    }

    fn codes(verdict: &ExitVerdict) -> Vec<&str> {
        verdict.items.iter().map(|hit| hit.code.as_str()).collect()
    }

    #[test]
    fn hard_stop_fires_when_close_breaks_the_stop() {
        let verdict = evaluate_exit(&position(100.0, 92.0), &facts(91.0), &DisciplineRules::default());
        assert!(codes(&verdict).contains(&"EXIT_HARD_STOP"));
        assert_eq!(verdict.action, DisciplineAction::MustSell);
        assert_eq!(verdict.level, RiskLevel::High);
    }

    #[test]
    fn hard_stop_does_not_fire_one_tick_above_the_stop() {
        // atr 置 None 让固定百分比成为唯一候选，否则生效止损会被更严的 ATR 候选顶上去
        let mut market = facts(92.01);
        market.atr = None;
        let verdict = evaluate_exit(&position(100.0, 92.0), &market, &DisciplineRules::default());
        assert!(!codes(&verdict).contains(&"EXIT_HARD_STOP"), "阈值上方不应触发");
    }

    #[test]
    fn normal_a_share_volatility_leaves_the_fixed_line_in_charge() {
        // atr_mult = 3.5 的交叉点是 ATR% = 8/3.5 = 2.29%。A 股日均 ATR% 多在 2–4%，
        // 所以绝大多数票由固定 8% 说了算——这正是把 atr_mult 从 2.0 提到 3.5 的目的。
        // 这里 ATR = 2.5（ATR% = 2.5%）：3.5 × 2.5 = 8.75% > 8%，ATR 线让位。
        let mut market = facts(93.0);
        market.atr = Some(2.5);
        let verdict = evaluate_exit(&position(100.0, 92.0), &market, &DisciplineRules::default());
        assert_eq!(verdict.stop_basis, StopBasis::Fixed, "常见波动率下必须是固定线当家");
        assert!((verdict.effective_stop - 92.0).abs() < 1e-9, "止损就落在成本 −8%");
        assert!(!codes(&verdict).contains(&"EXIT_HARD_STOP"), "收盘 93 在 92 之上，不该触发");
    }

    #[test]
    fn very_low_volatility_still_lets_atr_tighten_the_stop() {
        // ATR% = 1% 的安静票：3.5 × 1 = 3.5% < 8%，ATR 线仍接管，止损收到 96.5。
        // 这是有意保留的——止损越紧，sizing 允许买越多，风险敞口不变；
        // 而 3.5 倍已经把它从「日常止损线」压回「只在真正安静的票上稍作收紧」。
        let rules = DisciplineRules::default();
        let verdict = evaluate_exit(&position(100.0, 92.0), &facts(96.0), &rules);
        assert_eq!(verdict.stop_basis, StopBasis::Atr);
        assert!((verdict.effective_stop - 96.5).abs() < 1e-9, "100 − 3.5×1.0");
        assert!(codes(&verdict).contains(&"EXIT_HARD_STOP"));

        // 同一只票收盘 97.0：atr_mult 还是 2.0 时止损在 98.0，这天就被扫出去了；
        // 提到 3.5 后止损退到 96.5，这档正常回撤不再触发。这就是本次调参的实际差别。
        let survives = evaluate_exit(&position(100.0, 92.0), &facts(97.0), &rules);
        assert!(
            !codes(&survives).contains(&"EXIT_HARD_STOP"),
            "3.5 倍下 97.0 不应触发——低波动股被正常震荡扫出正是要解决的问题"
        );
    }

    #[test]
    fn trailing_stop_is_reported_separately_from_hard_stop() {
        // 浮盈 15% 已过启动线，最高 130 → 移动止损 119.6，收盘 115 跌破
        let mut market = facts(115.0);
        market.holding_high = Some(130.0);
        market.atr = None;
        let verdict = evaluate_exit(&position(100.0, 92.0), &market, &DisciplineRules::default());
        assert!(codes(&verdict).contains(&"EXIT_TRAILING_STOP"));
        assert!(!codes(&verdict).contains(&"EXIT_HARD_STOP"), "移动止盈应有独立 code 便于复盘归因");
        assert_eq!(verdict.action, DisciplineAction::MustSell);
    }

    #[test]
    fn time_stop_fires_only_when_both_conditions_hold() {
        let rules = DisciplineRules::default();
        let mut market = facts(101.0); // 浮盈 1% < 3%
        market.holding_trading_days = 20;
        assert!(codes(&evaluate_exit(&position(100.0, 92.0), &market, &rules)).contains(&"EXIT_TIME_STOP"));

        market.holding_trading_days = 19;
        assert!(
            !codes(&evaluate_exit(&position(100.0, 92.0), &market, &rules)).contains(&"EXIT_TIME_STOP"),
            "未满 20 个交易日不应触发"
        );

        market.holding_trading_days = 30;
        let profitable = facts(105.0);
        let mut profitable = profitable;
        profitable.holding_trading_days = 30;
        assert!(
            !codes(&evaluate_exit(&position(100.0, 92.0), &profitable, &rules)).contains(&"EXIT_TIME_STOP"),
            "浮盈已达标就不该因超期离场"
        );
    }

    #[test]
    fn scale_out_advances_one_tier_at_a_time() {
        let rules = DisciplineRules::default();
        // 浮盈 32% 同时越过两档，但一次只推进一档
        let mut market = facts(132.0);
        market.holding_high = Some(132.0);
        market.atr = None;
        let verdict = evaluate_exit(&position(100.0, 92.0), &market, &rules);
        let hits = codes(&verdict);
        assert!(hits.contains(&"EXIT_SCALE_OUT_1"));
        assert!(!hits.contains(&"EXIT_SCALE_OUT_2"), "一次扫描只推进一档");

        let mut done_one = position(100.0, 92.0);
        done_one.scale_out_done = 1;
        let verdict = evaluate_exit(&done_one, &market, &rules);
        assert!(codes(&verdict).contains(&"EXIT_SCALE_OUT_2"), "已减一档后应推进第二档");
    }

    #[test]
    fn scale_out_suggests_a_tradable_share_count() {
        let mut market = facts(120.0);
        market.holding_high = Some(120.0);
        market.atr = None;
        let verdict = evaluate_exit(&position(100.0, 92.0), &market, &DisciplineRules::default());
        let hit = verdict.items.iter().find(|h| h.code == "EXIT_SCALE_OUT_1").unwrap();
        assert_eq!(hit.suggested_quantity, Some(500), "1000 股减半 = 500 股");
    }

    #[test]
    fn scale_out_is_skipped_when_the_slice_is_below_one_lot() {
        let mut small = position(100.0, 92.0);
        small.quantity = 100; // 减半 = 50 股，取整后为 0
        let mut market = facts(120.0);
        market.holding_high = Some(120.0);
        market.atr = None;
        assert!(
            !codes(&evaluate_exit(&small, &market, &DisciplineRules::default()))
                .contains(&"EXIT_SCALE_OUT_1"),
            "拆不出整手时不应产出无法执行的减仓指令"
        );
    }

    #[test]
    fn support_breakdown_requires_volume_confirmation() {
        let rules = DisciplineRules::default();
        let mut market = facts(95.0);
        market.broken_support = Some(98.0);
        market.volume_ratio = 1.0; // 未放量
        assert!(
            !codes(&evaluate_exit(&position(100.0, 80.0), &market, &rules))
                .contains(&"EXIT_SUPPORT_BREAKDOWN"),
            "缩量假摔不应触发破位止损"
        );

        market.volume_ratio = 1.5;
        assert!(codes(&evaluate_exit(&position(100.0, 80.0), &market, &rules))
            .contains(&"EXIT_SUPPORT_BREAKDOWN"));
    }

    #[test]
    fn basis_change_suspends_every_exit_rule() {
        // 除权导致收盘腰斩：若不挂起，硬止损会被假触发
        let mut market = facts(50.0);
        market.suspected_corporate_action = Some("2026-03-02".to_string());
        let verdict = evaluate_exit(&position(100.0, 92.0), &market, &DisciplineRules::default());
        let hits = codes(&verdict);
        assert_eq!(hits, vec!["DATA_PRICE_BASIS_CHANGED"], "除权时只应报数据问题");
        assert!(
            !hits.contains(&"EXIT_HARD_STOP"),
            "硬止损恰恰是最容易被除权假触发的那条，必须一并挂起"
        );
        assert_eq!(verdict.action, DisciplineAction::Warn);
    }

    #[test]
    fn position_flagged_by_db_is_also_suspended() {
        let mut flagged = position(100.0, 92.0);
        flagged.basis_suspect = true;
        let verdict = evaluate_exit(&flagged, &facts(50.0), &DisciplineRules::default());
        assert_eq!(codes(&verdict), vec!["DATA_PRICE_BASIS_CHANGED"]);
    }

    #[test]
    fn limit_down_lock_downgrades_forced_actions_to_warnings() {
        let mut market = facts(91.0);
        market.is_limit_down_locked = true;
        let verdict = evaluate_exit(&position(100.0, 92.0), &market, &DisciplineRules::default());
        assert!(codes(&verdict).contains(&"EXIT_HARD_STOP"), "规则仍应命中");
        assert_eq!(
            verdict.action,
            DisciplineAction::Warn,
            "今天根本卖不掉，报「必须卖」只是制造焦虑"
        );
        assert!(!verdict.tradable);
        let hit = verdict.items.iter().find(|h| h.code == "EXIT_HARD_STOP").unwrap();
        assert!(hit.detail.contains("明日开盘继续执行"));
    }

    #[test]
    fn t1_lock_is_flagged_on_the_entry_day() {
        let mut same_day = position(100.0, 92.0);
        same_day.open_date = "2026-03-02".to_string();
        let verdict = evaluate_exit(&same_day, &facts(91.0), &DisciplineRules::default());
        assert!(verdict.t1_locked);
        assert!(codes(&verdict).contains(&"EXIT_T1_LOCKED"));
        assert_eq!(verdict.action, DisciplineAction::MustSell, "T+1 不改变必卖结论，只改执行时点");
    }

    #[test]
    fn ma20_break_only_warns_and_only_when_profitable() {
        let rules = DisciplineRules::default();
        let mut market = facts(105.0);
        market.ma20 = Some(110.0);
        market.atr = None;
        let verdict = evaluate_exit(&position(100.0, 92.0), &market, &rules);
        assert!(codes(&verdict).contains(&"EXIT_MA20_BREAK"));
        assert_eq!(verdict.action, DisciplineAction::Warn);

        let mut losing = facts(95.0);
        losing.ma20 = Some(110.0);
        assert!(
            !codes(&evaluate_exit(&position(100.0, 92.0), &losing, &rules)).contains(&"EXIT_MA20_BREAK"),
            "亏损仓位由止损接管，不需要这条提示"
        );
    }

    #[test]
    fn missing_target_price_is_reported_for_review_completeness() {
        let mut no_target = position(100.0, 92.0);
        no_target.target_price = None;
        assert!(codes(&evaluate_exit(&no_target, &facts(100.0), &DisciplineRules::default()))
            .contains(&"DATA_TARGET_MISSING"));
    }

    #[test]
    fn metrics_expose_distance_to_stop_and_drawdown() {
        let mut market = facts(110.0);
        market.holding_high = Some(120.0);
        let verdict = evaluate_exit(&position(100.0, 92.0), &market, &DisciplineRules::default());
        assert!((verdict.metrics.unrealized_pnl_percent - 10.0).abs() < 1e-9);
        assert!((verdict.metrics.unrealized_pnl - 10_000.0).abs() < 1e-9);
        let drawdown = verdict.metrics.drawdown_from_high_percent.unwrap();
        assert!((drawdown - 8.333_333).abs() < 1e-3, "从 120 回撤到 110 应约 8.33%");
    }
}
