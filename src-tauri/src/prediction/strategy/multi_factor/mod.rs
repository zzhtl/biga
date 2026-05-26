//! 多因子评分策略
//!
//! 基于华尔街量化策略优化，特点：
//! - 非线性组合：使用sigmoid和指数函数平滑极端值
//! - 动态权重：根据市场状态自适应调整因子权重
//! - 信号确认：多重条件验证，提高信号可靠性
//!
//! 子模块拆分：
//! - [`factors`]：各因子（趋势/量价/动量/形态/支撑阻力/情绪/波动率）评分
//! - [`weights`]：市场状态自适应权重
//! - [`transform`]：非线性变换、信号确认与信号生成

use crate::config::weights::*;
use crate::prediction::analysis::market_regime::{MarketRegime, VolatilityLevel};
use crate::prediction::analysis::{
    PatternRecognition, SupportResistance, TrendState, VolumePriceSignal,
};
use crate::prediction::indicators::TechnicalIndicatorValues;
use serde::{Deserialize, Serialize};

mod factors;
mod transform;
mod weights;

use factors::{
    calculate_momentum_score_enhanced, calculate_pattern_score_enhanced,
    calculate_sentiment_score_enhanced, calculate_sr_score_enhanced,
    calculate_trend_score_enhanced, calculate_volatility_score_enhanced,
    calculate_volume_price_score_enhanced,
};
use transform::{
    apply_confirmation_adjustment, count_signal_confirmations, generate_enhanced_signal,
    sigmoid_transform,
};
use weights::get_adaptive_weights;

/// 多因子评分结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiFactorScore {
    pub total_score: f64,
    pub trend_score: f64,
    pub volume_price_score: f64,
    pub momentum_score: f64,
    pub pattern_score: f64,
    pub support_resistance_score: f64,
    pub sentiment_score: f64,
    pub volatility_score: f64,
    pub signal: String,
    pub signal_strength: f64,
    /// 市场自适应调整后的得分
    pub adaptive_score: f64,
    /// 信号确认数量
    pub confirmation_count: i32,
}

impl Default for MultiFactorScore {
    fn default() -> Self {
        Self {
            total_score: 50.0,
            trend_score: 50.0,
            volume_price_score: 50.0,
            momentum_score: 50.0,
            pattern_score: 50.0,
            support_resistance_score: 50.0,
            sentiment_score: 50.0,
            volatility_score: 50.0,
            signal: "中性".to_string(),
            signal_strength: 0.5,
            adaptive_score: 50.0,
            confirmation_count: 0,
        }
    }
}

/// 计算多因子综合评分（基础版本）
pub fn calculate_multi_factor_score(
    trend_state: &TrendState,
    volume_signal: &VolumePriceSignal,
    indicators: &TechnicalIndicatorValues,
    patterns: &[PatternRecognition],
    support_resistance: &SupportResistance,
    volatility: f64,
) -> MultiFactorScore {
    calculate_adaptive_multi_factor_score(
        trend_state,
        volume_signal,
        indicators,
        patterns,
        support_resistance,
        volatility,
        None, // 无市场状态时使用默认权重
        None,
    )
}

/// 计算自适应多因子综合评分（专业版本）
pub fn calculate_adaptive_multi_factor_score(
    trend_state: &TrendState,
    volume_signal: &VolumePriceSignal,
    indicators: &TechnicalIndicatorValues,
    patterns: &[PatternRecognition],
    support_resistance: &SupportResistance,
    volatility: f64,
    market_regime: Option<&MarketRegime>,
    volatility_level: Option<&VolatilityLevel>,
) -> MultiFactorScore {
    // 获取动态权重
    let weights = get_adaptive_weights(market_regime, volatility_level);

    // 计算各因子评分（使用非线性变换）
    let trend_score = calculate_trend_score_enhanced(trend_state, indicators);
    let volume_price_score = calculate_volume_price_score_enhanced(volume_signal, indicators);
    let momentum_score = calculate_momentum_score_enhanced(indicators);
    let pattern_score = calculate_pattern_score_enhanced(patterns);
    let support_resistance_score = calculate_sr_score_enhanced(support_resistance);
    let sentiment_score = calculate_sentiment_score_enhanced(indicators);
    let volatility_score = calculate_volatility_score_enhanced(volatility, volatility_level);

    // 计算信号确认数量
    let confirmation_count = count_signal_confirmations(
        trend_score,
        volume_price_score,
        momentum_score,
        pattern_score,
        sentiment_score,
    );

    // 使用非线性组合（避免极端值主导）
    let weighted_scores = [
        (sigmoid_transform(trend_score), weights.trend),
        (sigmoid_transform(volume_price_score), weights.volume_price),
        (sigmoid_transform(momentum_score), weights.momentum),
        (sigmoid_transform(pattern_score), weights.pattern),
        (sigmoid_transform(support_resistance_score), weights.support_resistance),
        (sigmoid_transform(sentiment_score), weights.sentiment),
        (sigmoid_transform(volatility_score), weights.volatility),
    ];

    // 加权平均
    let total_weight: f64 = weighted_scores.iter().map(|(_, w)| w).sum();
    let total_score: f64 = weighted_scores
        .iter()
        .map(|(score, weight)| score * weight)
        .sum::<f64>()
        / total_weight;

    // 归一化到 0-100
    let total_score = (total_score * 100.0).clamp(0.0, 100.0);

    // 应用信号确认加成/惩罚
    let confirmation_adjusted = apply_confirmation_adjustment(total_score, confirmation_count);

    // 生成增强信号
    let (signal, signal_strength) =
        generate_enhanced_signal(confirmation_adjusted, confirmation_count, market_regime);

    MultiFactorScore {
        total_score,
        trend_score: trend_score * 100.0,
        volume_price_score: volume_price_score * 100.0,
        momentum_score: momentum_score * 100.0,
        pattern_score: pattern_score * 100.0,
        support_resistance_score: support_resistance_score * 100.0,
        sentiment_score: sentiment_score * 100.0,
        volatility_score: volatility_score * 100.0,
        signal,
        signal_strength,
        adaptive_score: confirmation_adjusted,
        confirmation_count,
    }
}

/// 根据多因子评分计算涨跌幅调整
pub fn calculate_multi_factor_adjustment(score: &MultiFactorScore) -> f64 {
    let total = score.total_score;

    if total > 75.0 {
        0.005 + (total - 75.0) * MULTI_FACTOR_STRONG_BULLISH_IMPACT
    } else if total > 60.0 {
        (total - 60.0) * MULTI_FACTOR_BULLISH_IMPACT
    } else if total < 25.0 {
        -0.005 - (25.0 - total) * MULTI_FACTOR_STRONG_BEARISH_IMPACT
    } else if total < 40.0 {
        -(40.0 - total) * MULTI_FACTOR_BEARISH_IMPACT
    } else {
        MULTI_FACTOR_NEUTRAL_BIAS
    }
}
