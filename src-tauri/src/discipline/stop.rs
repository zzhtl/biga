//! 止损线：三者取最严 + 移动止盈 + 棘轮。
//!
//! # 「取最严」= 取 max
//!
//! 止损价**越高**，能容忍的亏损**越小**，纪律**越严**。所以取最严是 `max` 不是 `min`。
//! 这一行极易写反，写反的后果是止损永远挂在最松的那条线上，且不会有任何报错——
//! 只会在某个下跌里多亏 5 个点。`strictest_wins` 测试就是钉这一行的。
//!
//! # 棘轮：只上移，永不下移
//!
//! `final = max(candidate, previous_stop)`。这是整套纪律的地基：
//! 禁止"再等等"、禁止靠补跌摊薄成本来放宽止损。
//!
//! # 为什么 ATR 锚在成本而不是现价
//!
//! 用户口径是「成本 − N×ATR」。浮盈很大时这一项会明显宽于其它候选而被 max 淘汰，无害。
//! 浮盈保护由 `trailing` 项独立负责 —— 本函数只管「本金保护线」，两条职责分离、各自可测。

use crate::discipline::rules::DisciplineRules;
use crate::discipline::types::{StopBasis, StopComputation};

/// 止损计算的输入
#[derive(Debug, Clone, Copy)]
pub struct StopInput {
    pub cost_price: f64,
    pub close: f64,
    /// None = 建仓时首次计算，无棘轮约束
    pub previous_stop: Option<f64>,
    pub previous_basis: Option<StopBasis>,
    /// None = K 线长度不足，ATR 不可用（`calculate_atr` 返回 0.0 时必须映射为 None）
    pub atr: Option<f64>,
    /// 收盘价下方最近的支撑位
    pub nearest_support: Option<f64>,
    /// 持仓期最高价（日线 high）。None = 窗口不足，移动止盈不参与
    pub highest_price: Option<f64>,
}

/// 计算生效止损价。纯函数：同样的输入永远同样的输出。
pub fn compute_stop(input: &StopInput, rules: &DisciplineRules) -> StopComputation {
    let cost = input.cost_price;

    // 候选 1：固定百分比。永远可用，是兜底。
    let candidate_fixed = cost * (1.0 - rules.fixed_stop_pct / 100.0);

    // 候选 2：ATR。atr 为 None（K 线不足）时**跳过**，绝不能当成 cost - 0 用——
    // 那会让止损直接等于成本价，任何微小回撤都触发。
    let candidate_atr = input
        .atr
        .filter(|atr| *atr > 0.0)
        .map(|atr| cost - rules.atr_mult * atr);

    // 候选 3：支撑位下方。支撑位本身来自 calculate_support_resistance，已保证在现价下方。
    let candidate_support = input
        .nearest_support
        .filter(|support| *support > 0.0)
        .map(|support| support * (1.0 - rules.support_buffer_pct / 100.0));

    // 候选 4：移动止盈。仅当浮盈达到启动线后参与——没赚到钱就没有利润可保护。
    let gain_pct = if cost > 0.0 {
        (input.close / cost - 1.0) * 100.0
    } else {
        0.0
    };
    let candidate_trailing = input
        .highest_price
        .filter(|high| *high > 0.0 && gain_pct >= rules.trail_arm_pct)
        .map(|high| high * (1.0 - rules.trail_pct / 100.0));

    // 取最严 = 取 max
    let mut raw = candidate_fixed;
    let mut basis = StopBasis::Fixed;
    for (value, candidate_basis) in [
        (candidate_atr, StopBasis::Atr),
        (candidate_support, StopBasis::Support),
        (candidate_trailing, StopBasis::Trailing),
    ] {
        if let Some(value) = value {
            if value > raw {
                raw = value;
                basis = candidate_basis;
            }
        }
    }

    // 棘轮
    let previous = input.previous_stop.unwrap_or(f64::NEG_INFINITY);
    let ratcheted = raw < previous;
    let stop_price = if ratcheted { previous } else { raw };
    if ratcheted {
        // 被挡住时沿用上次的依据，否则前端会显示一个并未生效的 basis
        basis = input.previous_basis.unwrap_or(basis);
    }
    let raised_to = input
        .previous_stop
        .filter(|prev| stop_price > *prev + f64::EPSILON)
        .map(|_| stop_price);

    let mut evidence = vec![
        format!("成本价 {cost:.3}"),
        format!("固定 {:.0}% → {candidate_fixed:.3}", rules.fixed_stop_pct),
    ];
    match candidate_atr {
        Some(value) => evidence.push(format!(
            "ATR({}) × {:.1} → {value:.3}",
            rules.atr_period, rules.atr_mult
        )),
        None => evidence.push("ATR 不可用（K 线长度不足），该候选跳过".to_string()),
    }
    match (input.nearest_support, candidate_support) {
        (Some(support), Some(value)) => evidence.push(format!(
            "支撑 {support:.3} − {:.0}% → {value:.3}",
            rules.support_buffer_pct
        )),
        _ => evidence.push("现价下方无有效支撑位，该候选跳过".to_string()),
    }
    match (input.highest_price, candidate_trailing) {
        (Some(high), Some(value)) => evidence.push(format!(
            "持仓最高 {high:.3} − {:.0}% → {value:.3}",
            rules.trail_pct
        )),
        (Some(high), None) => evidence.push(format!(
            "持仓最高 {high:.3}，浮盈 {gain_pct:+.2}% 未达移动止盈启动线 {:.0}%",
            rules.trail_arm_pct
        )),
        _ => evidence.push("持仓最高价窗口不足，移动止盈不参与".to_string()),
    }
    evidence.push(format!("取最严 → {raw:.3}（{}）", basis.label()));
    if ratcheted {
        evidence.push(format!(
            "棘轮生效：候选 {raw:.3} 低于现有止损 {previous:.3}，止损不下移"
        ));
    }

    StopComputation {
        stop_price,
        basis,
        candidate_fixed,
        candidate_atr,
        candidate_support,
        candidate_trailing,
        ratcheted,
        raised_to,
        evidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> StopInput {
        StopInput {
            cost_price: 100.0,
            close: 100.0,
            previous_stop: None,
            previous_basis: None,
            atr: None,
            nearest_support: None,
            highest_price: None,
        }
    }

    #[test]
    fn fixed_is_the_fallback_when_nothing_else_available() {
        let out = compute_stop(&base(), &DisciplineRules::default());
        assert_eq!(out.basis, StopBasis::Fixed);
        assert!((out.stop_price - 92.0).abs() < 1e-9, "成本 100 × (1−8%) 应为 92");
    }

    #[test]
    fn strictest_wins_means_highest_stop_wins() {
        // ATR 很小（ATR% = 0.5%）→ cost − 3.5×0.5 = 98.25，高于固定的 92，应当胜出。
        // 期望值从 rules 现算而不是写死，免得调 atr_mult 时这里又变成一个待修的魔数。
        let rules = DisciplineRules::default();
        let input = StopInput { atr: Some(0.5), ..base() };
        let out = compute_stop(&input, &rules);
        assert_eq!(out.basis, StopBasis::Atr, "取最严 = 取 max，止损价越高越严");
        assert!((out.stop_price - (100.0 - rules.atr_mult * 0.5)).abs() < 1e-9);
        assert!(out.stop_price > out.candidate_fixed, "胜出的候选必须严于固定线");
    }

    #[test]
    fn wide_atr_loses_to_fixed() {
        // ATR 很大 → cost − 2×10 = 80，低于固定的 92，应被淘汰
        let input = StopInput { atr: Some(10.0), ..base() };
        let out = compute_stop(&input, &DisciplineRules::default());
        assert_eq!(out.basis, StopBasis::Fixed);
        assert!((out.stop_price - 92.0).abs() < 1e-9);
    }

    #[test]
    fn zero_atr_is_skipped_not_treated_as_cost_minus_zero() {
        // calculate_atr 长度不足时返回 0.0；若不过滤，止损会等于成本价 100
        let input = StopInput { atr: Some(0.0), ..base() };
        let out = compute_stop(&input, &DisciplineRules::default());
        assert!(out.candidate_atr.is_none(), "ATR=0 必须视为不可用");
        assert!((out.stop_price - 92.0).abs() < 1e-9, "不能把止损算成成本价");
    }

    #[test]
    fn support_candidate_uses_buffer_below_the_level() {
        let input = StopInput { nearest_support: Some(96.0), ..base() };
        let out = compute_stop(&input, &DisciplineRules::default());
        assert_eq!(out.basis, StopBasis::Support);
        assert!((out.stop_price - 95.04).abs() < 1e-9, "96 × (1−1%) = 95.04");
    }

    #[test]
    fn trailing_does_not_participate_before_the_arm_threshold() {
        // 浮盈 5% < 启动线 10%
        let input = StopInput { close: 105.0, highest_price: Some(108.0), ..base() };
        let out = compute_stop(&input, &DisciplineRules::default());
        assert!(out.candidate_trailing.is_none(), "未达启动线时移动止盈不参与");
        assert_eq!(out.basis, StopBasis::Fixed);
    }

    #[test]
    fn trailing_locks_profit_once_armed() {
        // 浮盈 20% ≥ 启动线，最高 125 → 125 × (1−8%) = 115
        let input = StopInput { close: 120.0, highest_price: Some(125.0), ..base() };
        let out = compute_stop(&input, &DisciplineRules::default());
        assert_eq!(out.basis, StopBasis::Trailing);
        assert!((out.stop_price - 115.0).abs() < 1e-9);
        assert!(out.stop_price > input.cost_price, "移动止盈生效后止损应高于成本，利润被锁住");
    }

    #[test]
    fn ratchet_blocks_a_lower_candidate() {
        // 现有止损 115（此前移动止盈抬上去的），今日回落使候选跌回 92
        let input = StopInput {
            previous_stop: Some(115.0),
            previous_basis: Some(StopBasis::Trailing),
            ..base()
        };
        let out = compute_stop(&input, &DisciplineRules::default());
        assert!(out.ratcheted, "候选低于现有止损时棘轮必须生效");
        assert!((out.stop_price - 115.0).abs() < 1e-9, "止损永不下移");
        assert_eq!(out.basis, StopBasis::Trailing, "被挡住时应沿用上次的依据");
        assert!(out.raised_to.is_none());
    }

    #[test]
    fn averaging_down_cannot_loosen_the_stop() {
        // 补跌摊薄：成本从 100 降到 80，固定候选降到 73.6，但现有止损是 92
        let input = StopInput {
            cost_price: 80.0,
            close: 78.0,
            previous_stop: Some(92.0),
            previous_basis: Some(StopBasis::Fixed),
            ..base()
        };
        let out = compute_stop(&input, &DisciplineRules::default());
        assert!((out.stop_price - 92.0).abs() < 1e-9, "摊薄成本不能成为放宽止损的借口");
    }

    #[test]
    fn raised_to_reports_the_upward_move() {
        let input = StopInput {
            close: 120.0,
            highest_price: Some(125.0),
            previous_stop: Some(92.0),
            previous_basis: Some(StopBasis::Fixed),
            ..base()
        };
        let out = compute_stop(&input, &DisciplineRules::default());
        assert_eq!(out.raised_to, Some(115.0), "止损上移时应报告新价位供命令层回写");
        assert!(!out.ratcheted);
    }
}
