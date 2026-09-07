//! 买入准入：这笔买入是否符合你自己的资金管理规则。
//!
//! # 本模块只做否决，不做推荐
//!
//! [`screen_entry`] 返回 `allowed = true` 的语义是「**没有违反**你设定的规则」，
//! 不是「建议买入」。它不看方向、不看信号、不打分。前端文案必须是「未发现纪律冲突」，
//! 展示 [`ENTRY_DISCLAIMER`](crate::discipline::types::ENTRY_DISCLAIMER)。
//!
//! 唯一透传的预测层产物是 `risk_level`，且**只用于否决，不用于放行**：
//! 风险等级高会拦住买入，但风险等级低不构成任何买入理由。
//!
//! # 为什么强制先填止损价和目标价
//!
//! 不填止损就算不出每股风险，算不出风险就定不了仓位，定不了仓位就没有"亏小钱"。
//! 更重要的是：**买之前就把"我错了会怎样"想清楚**，是这套系统全部约束力的起点。
//! 事后再定止损，人一定会把它定在当前价下方一点点，或者干脆不定。

use crate::discipline::rules::DisciplineRules;
use crate::discipline::sizing::size_position;
use crate::discipline::types::{
    finalize, item, push_item, DisciplineAction, DisciplineCategory, DisciplineItem,
    EntryCandidate, EntryVerdict, MarketFacts, PortfolioState, ENTRY_DISCLAIMER,
};
#[cfg(test)]
use crate::discipline::types::SizingConstraint;
use crate::prediction::types::RiskLevel;

/// 买入准入唯一入口。纯函数，不落库。
pub fn screen_entry(
    candidate: &EntryCandidate,
    portfolio: &PortfolioState,
    facts: &MarketFacts,
    rules: &DisciplineRules,
) -> EntryVerdict {
    let entry_price = candidate.entry_price;
    let stop_price = candidate.stop_price;
    let target_price = candidate.target_price;
    let is_add_on = portfolio.symbol_cost_price.is_some();

    let mut items: Vec<DisciplineItem> = Vec::new();

    // —— 账户未配置：一切免谈 ——
    if portfolio.total_equity <= 0.0 {
        push_item(
            &mut items,
            item(
                "ENTRY_SETTINGS_MISSING",
                DisciplineCategory::Portfolio,
                RiskLevel::High,
                DisciplineAction::Blocked,
                "未设置账户资金",
                "所有仓位规则的分母都是账户总资产。请先在纪律设置里填写可用现金。",
                vec!["总资产 = 可用现金 + 持仓市值（现算）".to_string()],
            ),
        );
    }

    // —— 强制先想好怎么输 ——
    match stop_price {
        None => push_item(
            &mut items,
            item(
                "ENTRY_STOP_REQUIRED",
                DisciplineCategory::Entry,
                RiskLevel::High,
                DisciplineAction::Blocked,
                "未填写止损价",
                "买之前先回答「我错了在哪里认」。没有止损价就算不出每股风险，\
                 也就定不了仓位——那意味着这笔交易的最大亏损是不受控的。",
                vec!["止损价必须低于买入价".to_string()],
            ),
        ),
        Some(stop) if stop >= entry_price => push_item(
            &mut items,
            item(
                "ENTRY_STOP_REQUIRED",
                DisciplineCategory::Entry,
                RiskLevel::High,
                DisciplineAction::Blocked,
                "止损价不低于买入价",
                "止损价必须严格低于买入价，否则每股风险为零或为负，仓位无法计算。",
                vec![format!("买入价 {entry_price:.3}，止损价 {stop:.3}")],
            ),
        ),
        _ => {}
    }

    match target_price {
        None => push_item(
            &mut items,
            item(
                "ENTRY_TARGET_REQUIRED",
                DisciplineCategory::Entry,
                RiskLevel::High,
                DisciplineAction::Blocked,
                "未填写目标价",
                "没有目标价就算不出盈亏比，无法判断这笔交易值不值得冒这个险。",
                vec!["目标价必须高于买入价".to_string()],
            ),
        ),
        Some(target) if target <= entry_price => push_item(
            &mut items,
            item(
                "ENTRY_TARGET_REQUIRED",
                DisciplineCategory::Entry,
                RiskLevel::High,
                DisciplineAction::Blocked,
                "目标价不高于买入价",
                "目标价必须严格高于买入价。",
                vec![format!("买入价 {entry_price:.3}，目标价 {target:.3}")],
            ),
        ),
        _ => {}
    }

    // —— 盈亏比与止损宽度 ——
    let reward_risk_ratio = match (stop_price, target_price) {
        (Some(stop), Some(target)) if stop < entry_price && target > entry_price => {
            Some((target - entry_price) / (entry_price - stop))
        }
        _ => None,
    };

    if let Some(ratio) = reward_risk_ratio {
        if ratio < rules.min_rr {
            push_item(
                &mut items,
                item(
                    "ENTRY_RR_TOO_LOW",
                    DisciplineCategory::Entry,
                    RiskLevel::High,
                    DisciplineAction::Blocked,
                    "盈亏比不达标",
                    "赚的空间不够抵消亏的风险。盈亏比 2:1 时胜率 40% 就是正期望；\
                     低于这条线，你需要极高的胜率才能不亏——而胜率是你控制不了的。",
                    vec![
                        format!("盈亏比 {ratio:.2}:1，门槛 {:.1}:1", rules.min_rr),
                        format!(
                            "上行空间 {:.3}，下行风险 {:.3}",
                            target_price.unwrap_or(0.0) - entry_price,
                            entry_price - stop_price.unwrap_or(0.0)
                        ),
                    ],
                ),
            );
        }
    }

    if let Some(stop) = stop_price {
        if stop < entry_price && entry_price > 0.0 {
            let width = (entry_price - stop) / entry_price * 100.0;
            if width > rules.max_stop_pct {
                push_item(
                    &mut items,
                    item(
                        "ENTRY_STOP_TOO_WIDE",
                        DisciplineCategory::Entry,
                        RiskLevel::Medium,
                        DisciplineAction::Blocked,
                        "止损过宽",
                        "止损离买入价太远，说明你依赖的技术位不足以界定「我错了」。\
                         风险敞口虽然仍被仓位控住，但这种交易的胜负更接近掷硬币。",
                        vec![format!("止损宽度 {width:.2}%，上限 {:.1}%", rules.max_stop_pct)],
                    ),
                );
            }
        }
    }

    // —— 风险等级：透传，只用于否决 ——
    if let Some(level) = candidate.risk_level {
        if level > rules.max_entry_risk_level {
            push_item(
                &mut items,
                item(
                    "ENTRY_RISK_LEVEL",
                    DisciplineCategory::Entry,
                    RiskLevel::High,
                    DisciplineAction::Blocked,
                    "风险等级超过准入门槛",
                    "该股当前的事实告警等级超出你设定的准入上限。此项只用于否决——\
                     等级低不构成任何买入理由。",
                    vec![
                        format!("当前 {}", level.label()),
                        format!("准入上限 {}", rules.max_entry_risk_level.label()),
                    ],
                ),
            );
        }
    }

    // —— 补跌加仓：散户最常见的爆仓路径 ——
    if let Some(existing_cost) = portfolio.symbol_cost_price {
        if facts.close < existing_cost {
            push_item(
                &mut items,
                item(
                    "ENTRY_AVERAGING_DOWN",
                    DisciplineCategory::Entry,
                    RiskLevel::High,
                    DisciplineAction::Blocked,
                    "禁止对浮亏持仓加仓",
                    "摊薄成本会让你在一个已经证明判断错误的方向上加大赌注，\
                     并且把原本 2% 的单笔风险放大成不可控的敞口。想加仓，先等它回到成本之上。",
                    vec![
                        format!("已持成本 {existing_cost:.3}，现价 {:.3}", facts.close),
                        format!("当前浮亏 {:.2}%", (facts.close / existing_cost - 1.0) * 100.0),
                    ],
                ),
            );
        }
    }

    // —— 冷静期：防报复性交易 ——
    if let Some(bars) = portfolio.bars_since_last_stop_out {
        if bars < rules.cooldown_bars {
            push_item(
                &mut items,
                item(
                    "ENTRY_COOLDOWN",
                    DisciplineCategory::Portfolio,
                    RiskLevel::High,
                    DisciplineAction::Blocked,
                    "该股处于止损冷静期",
                    "刚在这只票上止损离场就想买回来，多半是想把亏的赚回来，而不是发现了新机会。\
                     冷静期就是用来隔开这两件事的。",
                    vec![format!(
                        "距上次止损 {bars} 个交易日，冷静期 {} 个交易日",
                        rules.cooldown_bars
                    )],
                ),
            );
        }
    }

    // —— 熔断：连续亏损后强制停手 ——
    if portfolio.consecutive_losses >= rules.breaker_losses {
        let within_window = portfolio
            .bars_since_last_loss
            .is_none_or(|bars| bars < rules.breaker_bars);
        if within_window {
            push_item(
                &mut items,
                item(
                    "ENTRY_CIRCUIT_BREAKER",
                    DisciplineCategory::Portfolio,
                    RiskLevel::High,
                    DisciplineAction::Blocked,
                    "连续亏损熔断中",
                    "连续亏损通常说明当前市场环境与你的方法不匹配，而不是运气问题。\
                     此时加大频率只会加快亏损。停手几天，让统计噪音过去。",
                    vec![
                        format!(
                            "连续亏损 {} 笔（阈值 {}）",
                            portfolio.consecutive_losses, rules.breaker_losses
                        ),
                        match portfolio.bars_since_last_loss {
                            Some(bars) => format!(
                                "距最后一笔 {bars} 个交易日，需满 {}",
                                rules.breaker_bars
                            ),
                            None => "尚未产生新的交易日".to_string(),
                        },
                    ],
                ),
            );
        }
    }

    // —— 持股只数上限 ——
    if !is_add_on && portfolio.open_position_count >= rules.max_holdings {
        push_item(
            &mut items,
            item(
                "ENTRY_MAX_HOLDINGS",
                DisciplineCategory::Portfolio,
                RiskLevel::High,
                DisciplineAction::Blocked,
                "持仓只数已达上限",
                "盯得过来的票才是你的能力圈。想买新的，先卖掉一只——这个取舍本身\
                 就是在逼你比较哪一笔更值得留。",
                vec![format!(
                    "当前 {} 只，上限 {} 只",
                    portfolio.open_position_count, rules.max_holdings
                )],
            ),
        );
    }

    // —— 头寸规模 ——
    let sizing = size_position(
        &candidate.symbol,
        entry_price,
        stop_price.unwrap_or(entry_price),
        portfolio,
        rules,
    );

    if sizing.below_one_lot && portfolio.total_equity > 0.0 && stop_price.is_some_and(|s| s < entry_price)
    {
        push_item(
            &mut items,
            item(
                "ENTRY_SIZE_BELOW_LOT",
                DisciplineCategory::Sizing,
                RiskLevel::Medium,
                DisciplineAction::Blocked,
                "按风险预算买不到一手",
                "这只票对当前账户太贵，或者止损设得太宽。**不要为了能买而放宽止损**——\
                 那等于用放大亏损来换一次入场机会。换一只便宜的，或者等一个更近的止损位。",
                sizing.evidence.clone(),
            ),
        );
    }

    // —— 用户指定股数时，逐个上限比对，命中哪个报哪个 ——
    if let Some(intended) = candidate.intended_quantity.filter(|q| *q > 0) {
        for (cap, code, title, detail) in [
            (
                sizing.cap_by_risk,
                "ENTRY_RISK_BUDGET",
                "超出单笔风险预算",
                "这笔的最大亏损会超过总资产的既定比例。单笔风险恒定是「亏小钱」的全部机制，\
                 一次破例就足以让后面九次守纪白做。",
            ),
            (
                sizing.cap_by_single_position,
                "ENTRY_POSITION_LIMIT",
                "超出单票仓位上限",
                "单票占比过高会让一次意外把整个账户带走。分散不是为了多赚，是为了活着。",
            ),
            (
                sizing.cap_by_total_position,
                "ENTRY_TOTAL_EXPOSURE",
                "超出总仓位上限",
                "留一部分现金不是浪费。满仓时你没有任何应对意外的余地，只能被动承受。",
            ),
            (
                sizing.cap_by_cash,
                "ENTRY_INSUFFICIENT_CASH",
                "可用现金不足",
                "现金不够买这么多股。",
            ),
        ] {
            if intended > cap {
                push_item(
                    &mut items,
                    item(
                        code,
                        DisciplineCategory::Sizing,
                        RiskLevel::High,
                        DisciplineAction::Blocked,
                        title,
                        detail,
                        vec![
                            format!("拟买 {intended} 股，该项上限 {cap} 股"),
                            format!("综合最大可买 {} 股", sizing.max_shares),
                        ],
                    ),
                );
            }
        }
    }

    // —— 提示类 ——
    if facts.is_limit_up_locked {
        push_item(
            &mut items,
            item(
                "ENTRY_LIMIT_UP_UNFILLABLE",
                DisciplineCategory::Data,
                RiskLevel::Medium,
                DisciplineAction::Warn,
                "一字涨停，今日难以成交",
                "按当前价格挂单大概率买不到。若明日高开，请重新核算止损与仓位——\
                 买入价变了，能买的股数也就变了。",
                vec![format!("涨跌幅 {:.2}%，涨停限幅 {:.1}%", facts.change_percent, facts.limit_up_percent)],
            ),
        );
    }

    if facts.staleness_days > rules.stale_days {
        push_item(
            &mut items,
            item(
                "ENTRY_DATA_STALE",
                DisciplineCategory::Data,
                RiskLevel::Medium,
                DisciplineAction::Warn,
                "行情数据陈旧",
                "支撑位与 ATR 基于较旧的 K 线，据此算出的止损可能已不适用。建议先刷新数据。",
                vec![
                    format!("最新 K 线 {}", facts.latest_date),
                    format!("距今 {} 个自然日", facts.staleness_days),
                ],
            ),
        );
    }

    let level = finalize(&mut items);
    let allowed = !items
        .iter()
        .any(|entry| entry.action == DisciplineAction::Blocked);

    EntryVerdict {
        symbol: candidate.symbol.clone(),
        allowed,
        level,
        level_label: level.label().to_string(),
        sizing,
        reward_risk_ratio,
        items,
        disclaimer: ENTRY_DISCLAIMER.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> MarketFacts {
        MarketFacts {
            symbol: "600519".to_string(),
            latest_date: "2026-03-02".to_string(),
            staleness_days: 0,
            close: 10.0,
            open: 10.0,
            high: 10.0,
            low: 10.0,
            prev_close: 10.0,
            volume_ratio: 1.0,
            change_percent: 0.0,
            atr: Some(0.3),
            atr_percent: Some(3.0),
            nearest_support: Some(9.5),
            broken_support: None,
            ma20: Some(10.0),
            holding_trading_days: 0,
            holding_high: None,
            holding_high_date: None,
            limit_up_percent: 10.0,
            limit_down_percent: -10.0,
            is_limit_down_locked: false,
            is_limit_up_locked: false,
            bars_used: 250,
            suspected_corporate_action: None,
        }
    }

    fn portfolio() -> PortfolioState {
        PortfolioState {
            total_equity: 200_000.0,
            cash: 200_000.0,
            ..Default::default()
        }
    }

    /// 一笔完全合规的买入：买 10、止损 9.5（5%）、目标 11.5（盈亏比 3:1）
    fn candidate() -> EntryCandidate {
        EntryCandidate {
            symbol: "600519".to_string(),
            entry_price: 10.0,
            intended_quantity: None,
            stop_price: Some(9.5),
            target_price: Some(11.5),
            risk_level: None,
        }
    }

    fn codes(verdict: &EntryVerdict) -> Vec<&str> {
        verdict.items.iter().map(|hit| hit.code.as_str()).collect()
    }

    #[test]
    fn a_compliant_entry_is_allowed_but_never_recommended() {
        let verdict = screen_entry(&candidate(), &portfolio(), &facts(), &DisciplineRules::default());
        assert!(verdict.allowed);
        assert!(verdict.items.is_empty());
        assert!(
            verdict.disclaimer.contains("不构成任何买入建议"),
            "allowed 的语义是「没违反纪律」，必须原样下发免责口径"
        );
        // 风险预算 4000 / 每股 0.5 = 8000 股，但 8000×10 = 8 万 = 总资产 40%，
        // 被 25% 单票上限截到 5000 股。两道闸门都生效才是对的。
        assert_eq!(verdict.sizing.cap_by_risk, 8000);
        assert_eq!(verdict.sizing.max_shares, 5000);
        assert_eq!(verdict.sizing.binding_constraint, SizingConstraint::SinglePosition);
    }

    #[test]
    fn missing_stop_or_target_blocks_the_entry() {
        let rules = DisciplineRules::default();
        let mut no_stop = candidate();
        no_stop.stop_price = None;
        let verdict = screen_entry(&no_stop, &portfolio(), &facts(), &rules);
        assert!(!verdict.allowed);
        assert!(codes(&verdict).contains(&"ENTRY_STOP_REQUIRED"));

        let mut no_target = candidate();
        no_target.target_price = None;
        assert!(codes(&screen_entry(&no_target, &portfolio(), &facts(), &rules))
            .contains(&"ENTRY_TARGET_REQUIRED"));
    }

    #[test]
    fn stop_above_entry_price_is_rejected() {
        let mut inverted = candidate();
        inverted.stop_price = Some(10.5);
        let verdict = screen_entry(&inverted, &portfolio(), &facts(), &DisciplineRules::default());
        assert!(codes(&verdict).contains(&"ENTRY_STOP_REQUIRED"));
        assert!(!verdict.allowed);
    }

    #[test]
    fn reward_risk_below_two_to_one_is_blocked() {
        let mut thin = candidate();
        thin.target_price = Some(10.75); // 上行 0.75 / 下行 0.5 = 1.5:1
        let verdict = screen_entry(&thin, &portfolio(), &facts(), &DisciplineRules::default());
        assert!(codes(&verdict).contains(&"ENTRY_RR_TOO_LOW"));
        assert!((verdict.reward_risk_ratio.unwrap() - 1.5).abs() < 1e-9);
        assert!(!verdict.allowed);
    }

    #[test]
    fn exactly_two_to_one_passes_the_threshold() {
        let mut exact = candidate();
        exact.target_price = Some(11.0); // 1.0 / 0.5 = 2.0
        assert!(screen_entry(&exact, &portfolio(), &facts(), &DisciplineRules::default()).allowed);
    }

    #[test]
    fn stop_wider_than_the_cap_is_blocked() {
        let mut wide = candidate();
        wide.stop_price = Some(8.5); // 15% > 10%
        wide.target_price = Some(14.0);
        let verdict = screen_entry(&wide, &portfolio(), &facts(), &DisciplineRules::default());
        assert!(codes(&verdict).contains(&"ENTRY_STOP_TOO_WIDE"));
        assert!(!verdict.allowed);
    }

    #[test]
    fn averaging_down_on_a_losing_position_is_blocked() {
        let losing = PortfolioState {
            symbol_cost_price: Some(12.0), // 已持成本 12，现价 10
            symbol_market_value: 12_000.0,
            open_position_count: 1,
            ..portfolio()
        };
        let verdict = screen_entry(&candidate(), &losing, &facts(), &DisciplineRules::default());
        assert!(codes(&verdict).contains(&"ENTRY_AVERAGING_DOWN"));
        assert!(!verdict.allowed);
    }

    #[test]
    fn adding_to_a_winning_position_is_not_blocked_by_that_rule() {
        let winning = PortfolioState {
            symbol_cost_price: Some(8.0), // 已持成本 8，现价 10，浮盈
            symbol_market_value: 8_000.0,
            open_position_count: 1,
            ..portfolio()
        };
        assert!(!codes(&screen_entry(&candidate(), &winning, &facts(), &DisciplineRules::default()))
            .contains(&"ENTRY_AVERAGING_DOWN"));
    }

    #[test]
    fn cooldown_boundary_is_exclusive() {
        let rules = DisciplineRules::default(); // cooldown_bars = 10
        let inside = PortfolioState { bars_since_last_stop_out: Some(9), ..portfolio() };
        assert!(codes(&screen_entry(&candidate(), &inside, &facts(), &rules)).contains(&"ENTRY_COOLDOWN"));

        let outside = PortfolioState { bars_since_last_stop_out: Some(10), ..portfolio() };
        assert!(
            !codes(&screen_entry(&candidate(), &outside, &facts(), &rules)).contains(&"ENTRY_COOLDOWN"),
            "满 10 个交易日即解除"
        );
    }

    #[test]
    fn circuit_breaker_blocks_only_inside_the_cooling_window() {
        let rules = DisciplineRules::default(); // 连败 3 笔 → 停手 5 日
        let hot = PortfolioState {
            consecutive_losses: 3,
            bars_since_last_loss: Some(2),
            ..portfolio()
        };
        assert!(codes(&screen_entry(&candidate(), &hot, &facts(), &rules))
            .contains(&"ENTRY_CIRCUIT_BREAKER"));

        let cooled = PortfolioState {
            consecutive_losses: 3,
            bars_since_last_loss: Some(5),
            ..portfolio()
        };
        assert!(
            !codes(&screen_entry(&candidate(), &cooled, &facts(), &rules))
                .contains(&"ENTRY_CIRCUIT_BREAKER"),
            "停手期满后应放行"
        );

        let two_losses = PortfolioState {
            consecutive_losses: 2,
            bars_since_last_loss: Some(1),
            ..portfolio()
        };
        assert!(!codes(&screen_entry(&candidate(), &two_losses, &facts(), &rules))
            .contains(&"ENTRY_CIRCUIT_BREAKER"));
    }

    #[test]
    fn holdings_cap_blocks_new_names_but_not_add_ons() {
        let rules = DisciplineRules::default(); // 上限 5 只
        let full = PortfolioState { open_position_count: 5, ..portfolio() };
        assert!(codes(&screen_entry(&candidate(), &full, &facts(), &rules))
            .contains(&"ENTRY_MAX_HOLDINGS"));

        let full_but_add_on = PortfolioState {
            open_position_count: 5,
            symbol_cost_price: Some(8.0),
            symbol_market_value: 8_000.0,
            ..portfolio()
        };
        assert!(
            !codes(&screen_entry(&candidate(), &full_but_add_on, &facts(), &rules))
                .contains(&"ENTRY_MAX_HOLDINGS"),
            "加仓已有持仓不增加持股只数"
        );
    }

    #[test]
    fn risk_level_can_only_veto_never_endorse() {
        let rules = DisciplineRules::default(); // 准入上限 Medium
        let mut high_risk = candidate();
        high_risk.risk_level = Some(RiskLevel::High);
        let verdict = screen_entry(&high_risk, &portfolio(), &facts(), &rules);
        assert!(codes(&verdict).contains(&"ENTRY_RISK_LEVEL"));
        assert!(!verdict.allowed);

        let mut low_risk = candidate();
        low_risk.risk_level = Some(RiskLevel::Low);
        let verdict = screen_entry(&low_risk, &portfolio(), &facts(), &rules);
        assert!(verdict.allowed);
        assert!(
            verdict.items.is_empty(),
            "风险等级低只是不否决，绝不产出任何鼓励买入的条目"
        );
    }

    #[test]
    fn account_without_capital_blocks_everything() {
        let empty = PortfolioState::default();
        let verdict = screen_entry(&candidate(), &empty, &facts(), &DisciplineRules::default());
        assert!(codes(&verdict).contains(&"ENTRY_SETTINGS_MISSING"));
        assert!(!verdict.allowed);
    }

    #[test]
    fn too_expensive_for_one_lot_blocks_instead_of_rounding_up() {
        let tiny = PortfolioState { total_equity: 3_000.0, cash: 3_000.0, ..Default::default() };
        let mut pricey = candidate();
        pricey.entry_price = 100.0;
        pricey.stop_price = Some(95.0);
        pricey.target_price = Some(115.0);
        let verdict = screen_entry(&pricey, &tiny, &facts(), &DisciplineRules::default());
        assert!(codes(&verdict).contains(&"ENTRY_SIZE_BELOW_LOT"));
        assert!(!verdict.allowed);
        assert!(
            verdict.items.iter().any(|h| h.detail.contains("不要为了能买而放宽止损")),
            "拒绝时要说清楚为什么，否则人会去改止损"
        );
    }

    #[test]
    fn intended_quantity_over_the_risk_budget_is_named_precisely() {
        let mut greedy = candidate();
        greedy.intended_quantity = Some(9_000); // 上限 8000
        let verdict = screen_entry(&greedy, &portfolio(), &facts(), &DisciplineRules::default());
        assert!(codes(&verdict).contains(&"ENTRY_RISK_BUDGET"));
        assert!(!verdict.allowed);
    }

    #[test]
    fn intended_quantity_over_the_single_name_cap_is_named_precisely() {
        // 每股风险 0.05 → 风险预算允许 80000 股，但单票 25% = 50000 元 / 10 = 5000 股
        let mut thin_stop = candidate();
        thin_stop.stop_price = Some(9.95);
        thin_stop.target_price = Some(10.2);
        thin_stop.intended_quantity = Some(6_000);
        let verdict = screen_entry(&thin_stop, &portfolio(), &facts(), &DisciplineRules::default());
        assert!(codes(&verdict).contains(&"ENTRY_POSITION_LIMIT"));
    }

    #[test]
    fn limit_up_lock_only_warns() {
        let mut market = facts();
        market.is_limit_up_locked = true;
        let verdict = screen_entry(&candidate(), &portfolio(), &market, &DisciplineRules::default());
        assert!(codes(&verdict).contains(&"ENTRY_LIMIT_UP_UNFILLABLE"));
        assert!(verdict.allowed, "买不到是执行问题，不是纪律冲突");
    }

    #[test]
    fn stale_data_only_warns() {
        let mut market = facts();
        market.staleness_days = 9;
        let verdict = screen_entry(&candidate(), &portfolio(), &market, &DisciplineRules::default());
        assert!(codes(&verdict).contains(&"ENTRY_DATA_STALE"));
        assert!(verdict.allowed);
    }
}
