//! 预期涨跌幅计算与 A 股涨跌停限制

use super::{get_stock_price_limits, PredictionContext, PredictionDirection, SignalConfirmation};
use crate::prediction::analysis::market_regime::{MarketRegime, StrategyType};

/// 根据A股规则限制预测幅度
fn apply_a_share_limits(change: f64, stock_code: Option<&str>) -> f64 {
    let (limit_down, limit_up) = get_stock_price_limits(stock_code);
    change.clamp(limit_down, limit_up)
}

/// 计算预期变化和预测区间
pub(super) fn calculate_expected_change(
    ctx: &PredictionContext,
    direction: &PredictionDirection,
    confirmation: &SignalConfirmation,
) -> (f64, (f64, f64)) {
    let strategy = ctx.market_regime.regime.recommended_strategy();

    // 基础预期变化
    let base_change = match strategy {
        StrategyType::TrendFollowing => calculate_trend_following_change(ctx, direction),
        StrategyType::MeanReversion => calculate_mean_reversion_change(ctx, direction),
        StrategyType::Reversal => calculate_reversal_change(ctx, direction),
    };

    // 根据信号确认级别调整
    let confirmation_multiplier = confirmation.to_weight();
    let adjusted_change = base_change * confirmation_multiplier;

    // 应用A股涨跌停限制
    let limited_change = apply_a_share_limits(adjusted_change, ctx.stock_code.as_deref());

    // 根据波动率计算预测区间
    let volatility_multiplier = ctx.market_regime.volatility_level.adjustment_factor();
    let range_width = ctx.volatility * 100.0 * volatility_multiplier * 1.5;

    // 预测区间也要遵守涨跌停限制
    let lower = apply_a_share_limits(limited_change - range_width, ctx.stock_code.as_deref());
    let upper = apply_a_share_limits(limited_change + range_width, ctx.stock_code.as_deref());

    (limited_change, (lower, upper))
}

/// 趋势跟踪策略计算
fn calculate_trend_following_change(
    ctx: &PredictionContext,
    direction: &PredictionDirection,
) -> f64 {
    let trend_strength = ctx.trend_analysis.trend_strength;
    let regime_strength = ctx.market_regime.trend_strength;

    // 基础变化基于趋势强度
    let base = (trend_strength + regime_strength) / 2.0 * 2.0; // 百分比

    // 技术指标调整
    let tech_adjustment = if ctx.indicators.macd_histogram > 0.0 {
        0.3
    } else if ctx.indicators.macd_histogram < 0.0 {
        -0.3
    } else {
        0.0
    };

    // 量价配合调整
    let volume_adjustment = match ctx.volume_signal.direction.as_str() {
        "上涨" if ctx.volume_signal.confidence > 0.7 => 0.4,
        "下跌" if ctx.volume_signal.confidence > 0.7 => -0.4,
        _ => 0.0,
    };

    let change = base + tech_adjustment + volume_adjustment;

    // 根据方向限制变化范围
    match direction {
        PredictionDirection::StrongBullish => change.max(0.5).min(5.0),
        PredictionDirection::Bullish => change.max(0.2).min(3.0),
        PredictionDirection::StrongBearish => change.min(-0.5).max(-5.0),
        PredictionDirection::Bearish => change.min(-0.2).max(-3.0),
        PredictionDirection::Neutral => change.clamp(-1.0, 1.0),
    }
}

/// 均值回归策略计算
fn calculate_mean_reversion_change(
    ctx: &PredictionContext,
    _direction: &PredictionDirection,
) -> f64 {
    // 使用布林带位置和RSI计算均值回归
    let rsi = ctx.indicators.rsi;

    // RSI偏离度
    let rsi_deviation = (50.0 - rsi) / 50.0;

    // 基于偏离度的回归预期
    let base = rsi_deviation * 1.5;

    // CCI调整
    let cci_adjustment = if ctx.indicators.cci > 100.0 {
        -0.3
    } else if ctx.indicators.cci < -100.0 {
        0.3
    } else {
        0.0
    };

    let change = base + cci_adjustment;

    // 震荡市变化幅度相对较小
    change.clamp(-2.5, 2.5)
}

/// 反转策略计算
fn calculate_reversal_change(ctx: &PredictionContext, direction: &PredictionDirection) -> f64 {
    // 基于背离和转折点信号
    let divergence_component = ctx.divergence.composite_score * 2.0;

    // 基于RSI极端值
    let rsi_component = if ctx.indicators.rsi > 80.0 {
        -1.5
    } else if ctx.indicators.rsi < 20.0 {
        1.5
    } else {
        0.0
    };

    // 基于市场状态
    let regime_component = match ctx.market_regime.regime {
        MarketRegime::PotentialTop => -1.0,
        MarketRegime::PotentialBottom => 1.0,
        _ => 0.0,
    };

    let change = divergence_component + rsi_component + regime_component;

    // 反转信号通常预期较大变化；最终方向由综合信号决定，避免方向与涨跌幅相互矛盾。
    directional_reversal_change(change, direction)
}

fn directional_reversal_change(change: f64, direction: &PredictionDirection) -> f64 {
    let magnitude = change.abs();
    match direction {
        PredictionDirection::StrongBullish | PredictionDirection::StrongBearish => {
            let magnitude = magnitude.max(0.5).min(4.0);
            if matches!(direction, PredictionDirection::StrongBullish) {
                magnitude
            } else {
                -magnitude
            }
        }
        PredictionDirection::Bullish | PredictionDirection::Bearish => {
            let magnitude = magnitude.max(0.2).min(2.5);
            if matches!(direction, PredictionDirection::Bullish) {
                magnitude
            } else {
                -magnitude
            }
        }
        PredictionDirection::Neutral => change.clamp(-1.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reversal_change_aligns_with_bearish_direction() {
        let change = directional_reversal_change(1.5, &PredictionDirection::Bearish);

        assert!(change < 0.0);
    }

    #[test]
    fn test_reversal_change_aligns_with_bullish_direction() {
        let change = directional_reversal_change(-1.5, &PredictionDirection::Bullish);

        assert!(change > 0.0);
    }
}
