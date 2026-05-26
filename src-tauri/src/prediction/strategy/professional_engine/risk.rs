//! 风险评估

use super::{PredictionContext, RiskAssessment};
use crate::prediction::analysis::market_regime::VolatilityLevel;

/// 风险评估
pub(super) fn assess_risk(ctx: &PredictionContext, expected_change: f64) -> RiskAssessment {
    // 波动率风险
    let volatility_risk = match ctx.market_regime.volatility_level {
        VolatilityLevel::VeryHigh => 0.9,
        VolatilityLevel::High => 0.7,
        VolatilityLevel::Normal => 0.5,
        VolatilityLevel::Low => 0.3,
        VolatilityLevel::VeryLow => 0.2,
    };

    // 计算支撑阻力距离
    let support_distance = if ctx.support_resistance.support_levels.is_empty() {
        5.0
    } else {
        let nearest_support = ctx
            .support_resistance
            .support_levels
            .iter()
            .filter(|&&s| s < ctx.current_price)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(ctx.current_price * 0.95);
        ((ctx.current_price - nearest_support) / ctx.current_price * 100.0).abs()
    };

    let resistance_distance = if ctx.support_resistance.resistance_levels.is_empty() {
        5.0
    } else {
        let nearest_resistance = ctx
            .support_resistance
            .resistance_levels
            .iter()
            .filter(|&&r| r > ctx.current_price)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(ctx.current_price * 1.05);
        ((nearest_resistance - ctx.current_price) / ctx.current_price * 100.0).abs()
    };

    // 建议止损（基于ATR和波动率）
    let base_stop = ctx.volatility * 100.0 * 2.0;
    let suggested_stop_loss = base_stop.max(2.0).min(8.0);

    // 建议止盈（基于风险收益比目标2:1）
    let suggested_take_profit = if expected_change.abs() > 0.5 {
        expected_change.abs() * 2.0
    } else {
        suggested_stop_loss * 2.0
    };

    // 风险收益比
    let risk_reward_ratio = if suggested_stop_loss > 0.0 {
        suggested_take_profit / suggested_stop_loss
    } else {
        2.0
    };

    // 风险等级
    let risk_level = if volatility_risk > 0.7 && support_distance > 4.0 {
        "高风险"
    } else if volatility_risk > 0.5 || support_distance > 3.0 {
        "中等风险"
    } else {
        "低风险"
    };

    RiskAssessment {
        risk_level: risk_level.to_string(),
        volatility_risk,
        support_distance,
        resistance_distance,
        suggested_stop_loss,
        suggested_take_profit,
        risk_reward_ratio,
    }
}
