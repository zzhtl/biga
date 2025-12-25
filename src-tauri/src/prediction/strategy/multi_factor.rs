//! 多因子评分策略
//! 
//! 基于华尔街量化策略优化，特点：
//! - 非线性组合：使用sigmoid和指数函数平滑极端值
//! - 动态权重：根据市场状态自适应调整因子权重
//! - 信号确认：多重条件验证，提高信号可靠性

use crate::config::weights::*;
use crate::prediction::analysis::{TrendState, VolumePriceSignal, PatternRecognition, SupportResistance};
use crate::prediction::analysis::market_regime::{MarketRegime, VolatilityLevel};
use crate::prediction::indicators::TechnicalIndicatorValues;
use serde::{Deserialize, Serialize};

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
        trend_score, volume_price_score, momentum_score, 
        pattern_score, sentiment_score
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
    let total_score: f64 = weighted_scores.iter()
        .map(|(score, weight)| score * weight)
        .sum::<f64>() / total_weight;
    
    // 归一化到 0-100
    let total_score = (total_score * 100.0).clamp(0.0, 100.0);
    
    // 应用信号确认加成/惩罚
    let confirmation_adjusted = apply_confirmation_adjustment(total_score, confirmation_count);
    
    // 生成增强信号
    let (signal, signal_strength) = generate_enhanced_signal(
        confirmation_adjusted, 
        confirmation_count,
        market_regime
    );
    
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

/// 动态权重结构
struct AdaptiveWeights {
    trend: f64,
    volume_price: f64,
    momentum: f64,
    pattern: f64,
    support_resistance: f64,
    sentiment: f64,
    volatility: f64,
}

/// 根据市场状态获取自适应权重
fn get_adaptive_weights(
    regime: Option<&MarketRegime>,
    volatility: Option<&VolatilityLevel>,
) -> AdaptiveWeights {
    let base = AdaptiveWeights {
        trend: TREND_FACTOR_WEIGHT,
        volume_price: VOLUME_PRICE_FACTOR_WEIGHT,
        momentum: MOMENTUM_FACTOR_WEIGHT,
        pattern: PATTERN_FACTOR_WEIGHT,
        support_resistance: SUPPORT_RESISTANCE_FACTOR_WEIGHT,
        sentiment: SENTIMENT_FACTOR_WEIGHT,
        volatility: VOLATILITY_FACTOR_WEIGHT,
    };
    
    // 根据市场状态调整
    match regime {
        Some(MarketRegime::StrongUptrend) | Some(MarketRegime::StrongDowntrend) => {
            // 强趋势市场：增加趋势和动量权重
            AdaptiveWeights {
                trend: base.trend * 1.4,
                volume_price: base.volume_price * 1.2,
                momentum: base.momentum * 1.3,
                pattern: base.pattern * 0.8,
                support_resistance: base.support_resistance * 0.7,
                sentiment: base.sentiment * 0.9,
                volatility: base.volatility * 0.8,
            }
        }
        Some(MarketRegime::Ranging) => {
            // 震荡市场：增加支撑阻力和形态权重
            AdaptiveWeights {
                trend: base.trend * 0.7,
                volume_price: base.volume_price * 1.1,
                momentum: base.momentum * 0.8,
                pattern: base.pattern * 1.4,
                support_resistance: base.support_resistance * 1.5,
                sentiment: base.sentiment * 1.2,
                volatility: base.volatility * 1.3,
            }
        }
        Some(MarketRegime::PotentialTop) | Some(MarketRegime::PotentialBottom) => {
            // 转折点：增加情绪和形态权重
            AdaptiveWeights {
                trend: base.trend * 0.8,
                volume_price: base.volume_price * 1.3,
                momentum: base.momentum * 1.1,
                pattern: base.pattern * 1.5,
                support_resistance: base.support_resistance * 1.3,
                sentiment: base.sentiment * 1.4,
                volatility: base.volatility * 1.2,
            }
        }
        _ => {
            // 根据波动率微调
            match volatility {
                Some(VolatilityLevel::VeryHigh) | Some(VolatilityLevel::High) => {
                    AdaptiveWeights {
                        volatility: base.volatility * 1.5,
                        support_resistance: base.support_resistance * 1.3,
                        ..base
                    }
                }
                Some(VolatilityLevel::VeryLow) | Some(VolatilityLevel::Low) => {
                    AdaptiveWeights {
                        trend: base.trend * 1.2,
                        momentum: base.momentum * 1.2,
                        ..base
                    }
                }
                _ => base,
            }
        }
    }
}

/// Sigmoid变换（平滑极端值）
fn sigmoid_transform(x: f64) -> f64 {
    // 将0-1范围的值通过sigmoid平滑
    let centered = (x - 0.5) * 4.0; // 中心化并放大
    1.0 / (1.0 + (-centered).exp())
}

/// 计算信号确认数量
fn count_signal_confirmations(
    trend: f64,
    volume: f64,
    momentum: f64,
    pattern: f64,
    sentiment: f64,
) -> i32 {
    let bullish_threshold = 0.6;
    let bearish_threshold = 0.4;
    
    let scores = [trend, volume, momentum, pattern, sentiment];
    
    let bullish_count = scores.iter().filter(|&&s| s > bullish_threshold).count();
    let bearish_count = scores.iter().filter(|&&s| s < bearish_threshold).count();
    
    // 返回优势方向的确认数量
    if bullish_count > bearish_count {
        bullish_count as i32
    } else if bearish_count > bullish_count {
        -(bearish_count as i32)
    } else {
        0
    }
}

/// 应用确认加成/惩罚
fn apply_confirmation_adjustment(score: f64, confirmations: i32) -> f64 {
    let abs_conf = confirmations.abs();
    
    let adjustment = match abs_conf {
        4..=5 => 8.0,  // 强确认：+8分
        3 => 5.0,      // 中等确认：+5分
        2 => 2.0,      // 弱确认：+2分
        _ => -3.0,     // 无确认/冲突：-3分
    };
    
    // 根据方向应用调整
    let direction_factor = if confirmations > 0 && score > 50.0 {
        1.0
    } else if confirmations < 0 && score < 50.0 {
        1.0
    } else if confirmations == 0 {
        1.0
    } else {
        0.5 // 方向冲突时减半调整
    };
    
    (score + adjustment * direction_factor).clamp(0.0, 100.0)
}

// =============================================================================
// 增强版因子评分函数
// =============================================================================

/// 增强版趋势评分（考虑多个趋势指标）
fn calculate_trend_score_enhanced(trend: &TrendState, indicators: &TechnicalIndicatorValues) -> f64 {
    let base_score: f64 = match trend {
        TrendState::StrongBullish => 0.90,
        TrendState::Bullish => 0.70,
        TrendState::Neutral => 0.50,
        TrendState::Bearish => 0.30,
        TrendState::StrongBearish => 0.10,
    };
    
    // MACD趋势确认
    let macd_confirmation: f64 = if indicators.macd_dif > indicators.macd_dea {
        0.05
    } else if indicators.macd_dif < indicators.macd_dea {
        -0.05
    } else {
        0.0
    };
    
    // MACD柱状图方向
    let hist_direction: f64 = if indicators.macd_histogram > 0.0 {
        0.03
    } else if indicators.macd_histogram < 0.0 {
        -0.03
    } else {
        0.0
    };
    
    (base_score + macd_confirmation + hist_direction).clamp(0.0, 1.0)
}

/// 增强版量价评分
fn calculate_volume_price_score_enhanced(signal: &VolumePriceSignal, indicators: &TechnicalIndicatorValues) -> f64 {
    let base_score: f64 = match signal.direction.as_str() {
        "上涨" => 0.5 + signal.confidence * 0.4,
        "下跌" => 0.5 - signal.confidence * 0.4,
        _ => 0.5,
    };
    
    // OBV趋势确认
    let obv_confirmation: f64 = if indicators.obv_trend > 0.05 {
        0.08
    } else if indicators.obv_trend < -0.05 {
        -0.08
    } else {
        0.0
    };
    
    (base_score + obv_confirmation).clamp(0.0, 1.0)
}

/// 增强版动量评分（多指标综合）
fn calculate_momentum_score_enhanced(indicators: &TechnicalIndicatorValues) -> f64 {
    let mut score = 0.5;
    let mut weight_sum = 0.0;
    let mut weighted_score = 0.0;
    
    // RSI 评分 (权重0.35)
    let rsi_weight = 0.35;
    let rsi_score = if indicators.rsi > 80.0 {
        0.1  // 极度超买
    } else if indicators.rsi > 70.0 {
        0.25 // 超买
    } else if indicators.rsi > 55.0 {
        0.6  // 偏强
    } else if indicators.rsi > 45.0 {
        0.5  // 中性
    } else if indicators.rsi > 30.0 {
        0.4  // 偏弱
    } else if indicators.rsi > 20.0 {
        0.75 // 超卖（反转机会）
    } else {
        0.9  // 极度超卖（强反转机会）
    };
    weighted_score += rsi_score * rsi_weight;
    weight_sum += rsi_weight;
    
    // MACD 评分 (权重0.35)
    let macd_weight = 0.35;
    let mut macd_score = 0.5;
    
    if indicators.macd_golden_cross {
        macd_score += 0.25;
    } else if indicators.macd_death_cross {
        macd_score -= 0.25;
    }
    
    if indicators.macd_histogram > 0.0 {
        macd_score += 0.1 + (indicators.macd_histogram * 50.0).min(0.15);
    } else if indicators.macd_histogram < 0.0 {
        macd_score -= 0.1 + (indicators.macd_histogram.abs() * 50.0).min(0.15);
    }
    
    macd_score = macd_score.clamp(0.0, 1.0);
    weighted_score += macd_score * macd_weight;
    weight_sum += macd_weight;
    
    // KDJ 评分 (权重0.30)
    let kdj_weight = 0.30;
    let mut kdj_score: f64 = 0.5;
    
    if indicators.kdj_golden_cross {
        kdj_score += 0.2;
    } else if indicators.kdj_death_cross {
        kdj_score -= 0.2;
    }
    
    // J值极端情况
    if indicators.kdj_j > 100.0 {
        kdj_score -= 0.15;
    } else if indicators.kdj_j < 0.0 {
        kdj_score += 0.15;
    }
    
    kdj_score = kdj_score.clamp(0.0_f64, 1.0_f64);
    weighted_score += kdj_score * kdj_weight;
    weight_sum += kdj_weight;
    
    if weight_sum > 0.0 {
        score = weighted_score / weight_sum;
    }
    
    score.clamp(0.0, 1.0)
}

/// 增强版形态评分
fn calculate_pattern_score_enhanced(patterns: &[PatternRecognition]) -> f64 {
    if patterns.is_empty() {
        return 0.5;
    }
    
    let mut bullish_weight: f64 = 0.0;
    let mut bearish_weight: f64 = 0.0;
    
    for pattern in patterns {
        let weight = pattern.reliability;
        if pattern.is_bullish {
            bullish_weight += weight;
        } else {
            bearish_weight += weight;
        }
    }
    
    let total_weight = bullish_weight + bearish_weight;
    if total_weight == 0.0 {
        return 0.5;
    }
    
    // 计算净方向
    let net_direction = (bullish_weight - bearish_weight) / total_weight;
    
    // 映射到0-1范围，使用平滑函数
    (0.5 + net_direction * 0.4).clamp(0.0_f64, 1.0_f64)
}

/// 增强版支撑阻力评分
fn calculate_sr_score_enhanced(sr: &SupportResistance) -> f64 {
    let mut score: f64 = 0.5;
    
    // 基于位置描述
    if sr.current_position.contains("强支撑") {
        score += 0.2;
    } else if sr.current_position.contains("支撑") {
        score += 0.1;
    } else if sr.current_position.contains("强压力") || sr.current_position.contains("强阻力") {
        score -= 0.2;
    } else if sr.current_position.contains("压力") || sr.current_position.contains("阻力") {
        score -= 0.1;
    }
    
    // 考虑距离支撑/阻力的远近
    if !sr.support_levels.is_empty() && !sr.resistance_levels.is_empty() {
        // 如果更接近支撑，稍微加分
        let support_count = sr.support_levels.len();
        let resistance_count = sr.resistance_levels.len();
        
        if support_count > resistance_count {
            score += 0.05;
        } else if resistance_count > support_count {
            score -= 0.05;
        }
    }
    
    score.clamp(0.0_f64, 1.0_f64)
}

/// 增强版情绪评分
fn calculate_sentiment_score_enhanced(indicators: &TechnicalIndicatorValues) -> f64 {
    let mut score: f64 = 0.5;
    
    // KDJ超买超卖（反向操作逻辑）
    if indicators.kdj_oversold {
        score += 0.15; // 超卖是买入机会
    }
    if indicators.kdj_overbought {
        score -= 0.15; // 超买是卖出信号
    }
    
    // CCI极端值
    if indicators.cci < -200.0 {
        score += 0.12; // 极度超卖
    } else if indicators.cci < -100.0 {
        score += 0.06;
    } else if indicators.cci > 200.0 {
        score -= 0.12; // 极度超买
    } else if indicators.cci > 100.0 {
        score -= 0.06;
    }
    
    // RSI极端值加成
    if indicators.rsi < 25.0 {
        score += 0.1;
    } else if indicators.rsi > 75.0 {
        score -= 0.1;
    }
    
    score.clamp(0.0_f64, 1.0_f64)
}

/// 增强版波动率评分
fn calculate_volatility_score_enhanced(volatility: f64, level: Option<&VolatilityLevel>) -> f64 {
    // 基础评分
    let base: f64 = if volatility < 0.015 {
        0.75  // 极低波动
    } else if volatility < 0.025 {
        0.65  // 低波动
    } else if volatility < 0.035 {
        0.55  // 正常波动
    } else if volatility < 0.05 {
        0.40  // 高波动
    } else {
        0.25  // 极高波动
    };
    
    // 根据波动率级别微调
    let adjustment: f64 = match level {
        Some(VolatilityLevel::VeryLow) => 0.05,
        Some(VolatilityLevel::Low) => 0.02,
        Some(VolatilityLevel::Normal) => 0.0,
        Some(VolatilityLevel::High) => -0.03,
        Some(VolatilityLevel::VeryHigh) => -0.08,
        None => 0.0,
    };
    
    (base + adjustment).clamp(0.0_f64, 1.0_f64)
}

/// 生成增强信号
fn generate_enhanced_signal(
    score: f64,
    confirmations: i32,
    regime: Option<&MarketRegime>,
) -> (String, f64) {
    // 基础信号阈值
    let (signal, base_strength) = if score >= 78.0 {
        ("强烈看涨", 0.92)
    } else if score >= 65.0 {
        ("看涨", 0.75)
    } else if score >= 55.0 {
        ("温和看涨", 0.60)
    } else if score >= 45.0 {
        ("中性", 0.50)
    } else if score >= 35.0 {
        ("温和看跌", 0.60)
    } else if score >= 22.0 {
        ("看跌", 0.75)
    } else {
        ("强烈看跌", 0.92)
    };
    
    // 根据确认数量调整强度
    let confirmation_factor: f64 = match confirmations.abs() {
        4..=5 => 1.1,
        3 => 1.05,
        2 => 1.0,
        1 => 0.95,
        _ => 0.85,
    };
    
    // 根据市场状态微调
    let regime_factor: f64 = match regime {
        Some(MarketRegime::StrongUptrend) | Some(MarketRegime::StrongDowntrend) => 1.05,
        Some(MarketRegime::PotentialTop) | Some(MarketRegime::PotentialBottom) => 1.08,
        _ => 1.0,
    };
    
    let final_strength = (base_strength * confirmation_factor * regime_factor).clamp(0.3_f64, 0.95_f64);
    
    (signal.to_string(), final_strength)
}

// =============================================================================
// 保留原始函数供向后兼容（允许未使用）
// =============================================================================

#[allow(dead_code)]
fn calculate_trend_score(trend: &TrendState) -> f64 {
    match trend {
        TrendState::StrongBullish => 0.9,
        TrendState::Bullish => 0.7,
        TrendState::Neutral => 0.5,
        TrendState::Bearish => 0.3,
        TrendState::StrongBearish => 0.1,
    }
}

#[allow(dead_code)]
fn calculate_volume_price_score(signal: &VolumePriceSignal) -> f64 {
    match signal.direction.as_str() {
        "上涨" => 0.5 + signal.confidence * 0.4,
        "下跌" => 0.5 - signal.confidence * 0.4,
        _ => 0.5,
    }
}

#[allow(dead_code)]
fn calculate_momentum_score(indicators: &TechnicalIndicatorValues) -> f64 {
    calculate_momentum_score_enhanced(indicators)
}

#[allow(dead_code)]
fn calculate_pattern_score(patterns: &[PatternRecognition]) -> f64 {
    calculate_pattern_score_enhanced(patterns)
}

#[allow(dead_code)]
fn calculate_sr_score(sr: &SupportResistance) -> f64 {
    calculate_sr_score_enhanced(sr)
}

#[allow(dead_code)]
fn calculate_sentiment_score(indicators: &TechnicalIndicatorValues) -> f64 {
    calculate_sentiment_score_enhanced(indicators)
}

#[allow(dead_code)]
fn calculate_volatility_score(volatility: f64) -> f64 {
    calculate_volatility_score_enhanced(volatility, None)
}

#[allow(dead_code)]
fn generate_signal(total_score: f64) -> (String, f64) {
    generate_enhanced_signal(total_score, 0, None)
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


