//! 头寸规模：由「单笔最大亏损」反推最大可买股数。
//!
//! # 这是「亏小钱」的全部数学
//!
//! ```text
//! 每股风险 = 买入价 − 止损价
//! 风险预算 = 总资产 × max_risk_pct
//! 可买股数 = 风险预算 / 每股风险
//! ```
//!
//! 止损设得越宽，能买的越少 —— 风险敞口恒定。这条恒等式意味着：
//! **想多买就得把止损收紧，而不是把风险放大**。
//!
//! # 一手都买不起时诚实拒绝
//!
//! `max_shares == 0` 时返回 `below_one_lot = true`，**不给 100 股凑合**。
//! 这个拒绝传递了真实信息：这只票对当前账户太贵，或止损设得太宽。
//! 为了"能买"而放宽止损，正是这套系统要拦的行为。
//!
//! # 明确不做
//!
//! 不把佣金/印花税/过户费算进事前规模计算。费率因券商而异，猜一个会造成假精度。
//! 费用只在录成交时手填，进已实现盈亏。

use crate::discipline::rules::DisciplineRules;
use crate::discipline::types::{PortfolioState, PositionSizing, SizingConstraint};

/// A 股最小交易单位与递增步长。
///
/// - 科创板（688）：200 股起，1 股递增。按 100 取整会被券商拒单。
/// - 主板 / 创业板：100 股起，100 股递增。
/// - 北交所（8/4 开头）：100 股起、1 股递增。库内目前无此类标的，暂按主板处理（known gap）。
pub fn lot_rule(symbol: &str) -> (i64, i64) {
    let code = symbol.trim_start_matches(|c: char| !c.is_ascii_digit());
    if code.starts_with("688") {
        (200, 1)
    } else {
        (100, 100)
    }
}

/// 向下取整到合法下单股数
fn round_down_to_lot(raw: f64, min_lot: i64, step: i64) -> i64 {
    if !raw.is_finite() || raw < min_lot as f64 {
        return 0;
    }
    let extra = ((raw - min_lot as f64) / step as f64).floor() as i64;
    min_lot + extra * step
}

/// 计算最大可买股数。纯函数。
pub fn size_position(
    symbol: &str,
    entry_price: f64,
    stop_price: f64,
    portfolio: &PortfolioState,
    rules: &DisciplineRules,
) -> PositionSizing {
    let (min_lot, lot_step) = lot_rule(symbol);
    let risk_per_share = entry_price - stop_price;

    // 入参非法时不猜：返回全 0，由 entry.rs 的 ENTRY_STOP_REQUIRED 负责给出人话解释
    if entry_price <= 0.0 || risk_per_share <= 0.0 || portfolio.total_equity <= 0.0 {
        return PositionSizing {
            risk_per_share,
            risk_budget: 0.0,
            cap_by_risk: 0,
            cap_by_single_position: 0,
            cap_by_total_position: 0,
            cap_by_cash: 0,
            binding_constraint: SizingConstraint::Risk,
            max_shares: 0,
            max_amount: 0.0,
            min_lot,
            lot_step,
            below_one_lot: true,
            evidence: vec![if portfolio.total_equity <= 0.0 {
                "账户总资产为 0，请先在纪律设置里填写可用现金".to_string()
            } else {
                format!("止损价 {stop_price:.3} 不低于买入价 {entry_price:.3}，无法计算每股风险")
            }],
        };
    }

    let risk_budget = portfolio.total_equity * rules.max_risk_pct / 100.0;
    let single_headroom =
        portfolio.total_equity * rules.max_single_pct / 100.0 - portfolio.symbol_market_value;
    let total_headroom =
        portfolio.total_equity * rules.max_total_pct / 100.0 - portfolio.total_market_value;

    let raw_by_risk = risk_budget / risk_per_share;
    let raw_by_single = (single_headroom / entry_price).max(0.0);
    let raw_by_total = (total_headroom / entry_price).max(0.0);
    let raw_by_cash = (portfolio.cash / entry_price).max(0.0);

    let cap_by_risk = round_down_to_lot(raw_by_risk, min_lot, lot_step);
    let cap_by_single_position = round_down_to_lot(raw_by_single, min_lot, lot_step);
    let cap_by_total_position = round_down_to_lot(raw_by_total, min_lot, lot_step);
    let cap_by_cash = round_down_to_lot(raw_by_cash, min_lot, lot_step);

    // 取最小的那个上限，并记录是谁卡住的（用未取整的原始值比较，避免取整后并列时归错因）
    let mut binding = SizingConstraint::Risk;
    let mut smallest = raw_by_risk;
    for (raw, constraint) in [
        (raw_by_single, SizingConstraint::SinglePosition),
        (raw_by_total, SizingConstraint::TotalPosition),
        (raw_by_cash, SizingConstraint::Cash),
    ] {
        if raw < smallest {
            smallest = raw;
            binding = constraint;
        }
    }

    let max_shares = cap_by_risk
        .min(cap_by_single_position)
        .min(cap_by_total_position)
        .min(cap_by_cash);
    let max_amount = max_shares as f64 * entry_price;

    let evidence = vec![
        format!(
            "总资产 {:.0} 元 × 单笔风险 {:.1}% = 风险预算 {risk_budget:.0} 元",
            portfolio.total_equity, rules.max_risk_pct
        ),
        format!(
            "每股风险 {risk_per_share:.3} 元（买入 {entry_price:.3} − 止损 {stop_price:.3}）"
        ),
        format!("按风险预算最多 {cap_by_risk} 股"),
        format!(
            "按单票上限 {:.0}% 最多 {cap_by_single_position} 股",
            rules.max_single_pct
        ),
        format!(
            "按总仓上限 {:.0}% 最多 {cap_by_total_position} 股",
            rules.max_total_pct
        ),
        format!("按可用现金 {:.0} 元最多 {cap_by_cash} 股", portfolio.cash),
        format!(
            "取最小 → {max_shares} 股（受限于{}），最小交易单位 {min_lot} 股 / 步长 {lot_step} 股",
            binding.label()
        ),
    ];

    PositionSizing {
        risk_per_share,
        risk_budget,
        cap_by_risk,
        cap_by_single_position,
        cap_by_total_position,
        cap_by_cash,
        binding_constraint: binding,
        max_shares,
        max_amount,
        min_lot,
        lot_step,
        below_one_lot: max_shares == 0,
        evidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn portfolio(equity: f64) -> PortfolioState {
        PortfolioState {
            total_equity: equity,
            cash: equity,
            ..Default::default()
        }
    }

    #[test]
    fn star_market_uses_200_share_minimum_with_single_share_steps() {
        assert_eq!(lot_rule("688981"), (200, 1), "科创板 200 股起、1 股递增");
        assert_eq!(lot_rule("600519"), (100, 100), "主板 100 股起、100 股递增");
        assert_eq!(lot_rule("300750"), (100, 100), "创业板同主板");
        assert_eq!(lot_rule("sh600519"), (100, 100), "带前缀也要能识别");
    }

    #[test]
    fn round_down_respects_min_lot_and_step() {
        assert_eq!(round_down_to_lot(250.0, 100, 100), 200);
        assert_eq!(round_down_to_lot(99.0, 100, 100), 0, "不足一手返回 0，不凑合");
        assert_eq!(round_down_to_lot(250.0, 200, 1), 250, "科创板允许 1 股递增");
        assert_eq!(round_down_to_lot(199.0, 200, 1), 0, "科创板不足 200 股返回 0");
    }

    #[test]
    fn risk_budget_drives_the_share_count() {
        // 总资产 100000 × 2% = 2000 风险预算；每股风险 1 元 → 2000 股
        let out = size_position("600519", 10.0, 9.0, &portfolio(100_000.0), &DisciplineRules::default());
        assert_eq!(out.cap_by_risk, 2000);
        // 但单票上限 25% = 25000 元 / 10 元 = 2500 股，总仓 80% = 8000 股，现金 10000 股
        assert_eq!(out.max_shares, 2000);
        assert_eq!(out.binding_constraint, SizingConstraint::Risk);
    }

    #[test]
    fn wider_stop_means_fewer_shares_at_identical_risk() {
        let rules = DisciplineRules::default();
        let tight = size_position("600519", 10.0, 9.5, &portfolio(100_000.0), &rules);
        let wide = size_position("600519", 10.0, 8.0, &portfolio(100_000.0), &rules);
        assert!(
            wide.max_shares < tight.max_shares,
            "止损放宽必须让可买股数变少，风险敞口才恒定"
        );
        // 风险预算这一路的敞口恒等于 2000 元（max_shares 还会再被单票上限截断）
        assert!((tight.cap_by_risk as f64 * 0.5 - 2000.0).abs() < 1e-9);
        assert!((wide.cap_by_risk as f64 * 2.0 - 2000.0).abs() < 1e-9);
    }

    #[test]
    fn single_position_cap_can_bind() {
        // 每股风险很小 → 风险预算允许买很多，但单票 25% 会先卡住
        let out = size_position("600519", 10.0, 9.9, &portfolio(100_000.0), &DisciplineRules::default());
        assert_eq!(out.binding_constraint, SizingConstraint::SinglePosition);
        assert_eq!(out.max_shares, 2500, "25% × 100000 / 10 = 2500 股");
    }

    #[test]
    fn existing_holding_reduces_single_position_headroom() {
        let state = PortfolioState {
            total_equity: 100_000.0,
            cash: 100_000.0,
            symbol_market_value: 20_000.0, // 已持 2 万，25% 上限只剩 5000 元
            ..Default::default()
        };
        let out = size_position("600519", 10.0, 9.9, &state, &DisciplineRules::default());
        assert_eq!(out.cap_by_single_position, 500, "剩余额度 5000 / 10 = 500 股");
    }

    #[test]
    fn cash_can_bind_even_when_equity_is_large() {
        let state = PortfolioState {
            total_equity: 100_000.0,
            cash: 3_000.0,
            total_market_value: 97_000.0,
            ..Default::default()
        };
        let out = size_position("600519", 10.0, 9.0, &state, &DisciplineRules::default());
        assert_eq!(out.binding_constraint, SizingConstraint::TotalPosition);
        assert_eq!(out.max_shares, 0, "总仓已超 80%，不允许再买");
    }

    #[test]
    fn too_expensive_for_one_lot_is_an_honest_rejection() {
        // 总资产 5000，风险预算 100 元；每股风险 5 元 → 20 股 < 100 股一手
        let out = size_position("600519", 100.0, 95.0, &portfolio(5_000.0), &DisciplineRules::default());
        assert_eq!(out.max_shares, 0);
        assert!(out.below_one_lot, "一手都买不起时必须诚实拒绝，不能给 100 股凑合");
    }

    #[test]
    fn invalid_stop_yields_zero_instead_of_guessing() {
        let out = size_position("600519", 10.0, 10.0, &portfolio(100_000.0), &DisciplineRules::default());
        assert_eq!(out.max_shares, 0);
        assert!(out.below_one_lot);
    }

    #[test]
    fn zero_equity_blocks_everything() {
        let out = size_position("600519", 10.0, 9.0, &portfolio(0.0), &DisciplineRules::default());
        assert_eq!(out.max_shares, 0);
        assert!(out.evidence[0].contains("总资产"), "应提示先设置资金而不是给 0 股了事");
    }
}
