//! 信号确认级别、预测方向与综合置信度

use super::{
    PredictionContext, PredictionDirection, SignalConfirmation, SignalSummary,
};
use crate::prediction::analysis::divergence::DivergenceAnalysis;
use crate::prediction::analysis::market_regime::{MarketRegime, MarketRegimeAnalysis, VolatilityLevel};

/// 计算信号确认级别
pub(super) fn calculate_signal_confirmation(summary: &SignalSummary) -> SignalConfirmation {
    let net_signals = (summary.bullish_signals as i32 - summary.bearish_signals as i32).abs();
    let dominant_signals = summary.bullish_signals.max(summary.bearish_signals);

    if dominant_signals >= 4 && net_signals >= 3 {
        SignalConfirmation::StrongConfirm
    } else if dominant_signals >= 3 && net_signals >= 2 {
        SignalConfirmation::ModerateConfirm
    } else if dominant_signals >= 2 && net_signals >= 1 {
        SignalConfirmation::WeakConfirm
    } else {
        SignalConfirmation::NoConfirm
    }
}

/// 确定预测方向
pub(super) fn determine_prediction_direction(
    summary: &SignalSummary,
    regime: &MarketRegimeAnalysis,
    divergence: &DivergenceAnalysis,
) -> PredictionDirection {
    let net_score = summary.net_signal_score;

    // 考虑背离信号的特殊权重（背离往往预示反转）
    let divergence_adjustment = if divergence.has_divergence {
        divergence.composite_score * 0.2
    } else {
        0.0
    };

    // 考虑市场状态
    let regime_adjustment = match regime.regime {
        MarketRegime::PotentialTop => -0.15,
        MarketRegime::PotentialBottom => 0.15,
        _ => 0.0,
    };

    let adjusted_score = net_score + divergence_adjustment + regime_adjustment;

    if adjusted_score > 0.6 {
        PredictionDirection::StrongBullish
    } else if adjusted_score > 0.25 {
        PredictionDirection::Bullish
    } else if adjusted_score < -0.6 {
        PredictionDirection::StrongBearish
    } else if adjusted_score < -0.25 {
        PredictionDirection::Bearish
    } else {
        PredictionDirection::Neutral
    }
}

/// 计算综合置信度
pub(super) fn calculate_comprehensive_confidence(
    ctx: &PredictionContext,
    confirmation: &SignalConfirmation,
    direction: &PredictionDirection,
) -> f64 {
    // 基础置信度来自信号确认
    let base_confidence = confirmation.to_weight() * 0.5;

    // 市场状态置信度
    let regime_confidence = ctx.market_regime.confidence * 0.15;

    // 趋势一致性加成
    let trend_consistency = if ctx.market_regime.regime.is_trending() {
        ctx.trend_analysis.trend_confidence * 0.15
    } else {
        0.1
    };

    // 背离信号加成/减成
    let divergence_factor = if ctx.divergence.has_divergence {
        // 背离与预测方向一致时加成
        let divergence_bullish = ctx.divergence.composite_score > 0.0;
        let prediction_bullish = matches!(
            direction,
            PredictionDirection::StrongBullish | PredictionDirection::Bullish
        );

        if divergence_bullish == prediction_bullish {
            ctx.divergence.overall_confidence * 0.10
        } else {
            -0.05 // 背离与预测方向冲突时减少置信度
        }
    } else {
        0.0
    };

    // 波动率影响（高波动降低置信度）
    let volatility_penalty = match ctx.market_regime.volatility_level {
        VolatilityLevel::VeryHigh => -0.10,
        VolatilityLevel::High => -0.05,
        VolatilityLevel::Normal => 0.0,
        VolatilityLevel::Low => 0.03,
        VolatilityLevel::VeryLow => 0.05,
    };

    let confidence = base_confidence
        + regime_confidence
        + trend_consistency
        + divergence_factor
        + volatility_penalty;

    confidence.clamp(0.25, 0.92)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_confirmation() {
        let summary = SignalSummary {
            bullish_signals: 5,
            bearish_signals: 1,
            signal_details: vec![],
            net_signal_score: 0.7,
        };
        let confirmation = calculate_signal_confirmation(&summary);
        assert_eq!(confirmation, SignalConfirmation::StrongConfirm);
    }
}
