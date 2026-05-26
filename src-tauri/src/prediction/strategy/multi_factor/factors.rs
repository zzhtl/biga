//! 各因子评分（趋势/量价/动量/形态/支撑阻力/情绪/波动率）

use crate::prediction::analysis::market_regime::VolatilityLevel;
use crate::prediction::analysis::{
    PatternRecognition, SupportResistance, TrendState, VolumePriceSignal,
};
use crate::prediction::indicators::TechnicalIndicatorValues;

// =============================================================================
// 增强版因子评分函数
// =============================================================================

/// 增强版趋势评分（考虑多个趋势指标）
pub(super) fn calculate_trend_score_enhanced(
    trend: &TrendState,
    indicators: &TechnicalIndicatorValues,
) -> f64 {
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
pub(super) fn calculate_volume_price_score_enhanced(
    signal: &VolumePriceSignal,
    indicators: &TechnicalIndicatorValues,
) -> f64 {
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

    // 量比确认：放量配合方向加分，缩量背离减分
    let vr = indicators.volume_ratio;
    let volume_ratio_adjustment: f64 = match signal.direction.as_str() {
        "上涨" => {
            if vr > 1.5 {
                0.10 // 放量上涨，量价齐升
            } else if vr < 0.7 {
                -0.06 // 缩量上涨，动能存疑
            } else {
                0.0
            }
        }
        "下跌" => {
            if vr > 1.5 {
                -0.10 // 放量下跌，抛压加剧
            } else if vr < 0.7 {
                0.05 // 缩量下跌，抛压减轻
            } else {
                0.0
            }
        }
        _ => 0.0,
    };

    // 换手率：适度换手(2%-15%)流动性健康、信号更可信；极高换手(>15%)警惕过热
    let turnover = indicators.turnover_rate;
    let turnover_adjustment: f64 = if turnover > 15.0 {
        -0.04
    } else if turnover >= 2.0 {
        0.03
    } else {
        0.0
    };

    (base_score + obv_confirmation + volume_ratio_adjustment + turnover_adjustment).clamp(0.0, 1.0)
}

/// 增强版动量评分（多指标综合）
pub(super) fn calculate_momentum_score_enhanced(indicators: &TechnicalIndicatorValues) -> f64 {
    let mut score = 0.5;
    let mut weight_sum = 0.0;
    let mut weighted_score = 0.0;

    // RSI 评分 (权重0.35)
    let rsi_weight = 0.35;
    let rsi_score = if indicators.rsi > 80.0 {
        0.1 // 极度超买
    } else if indicators.rsi > 70.0 {
        0.25 // 超买
    } else if indicators.rsi > 55.0 {
        0.6 // 偏强
    } else if indicators.rsi > 45.0 {
        0.5 // 中性
    } else if indicators.rsi > 30.0 {
        0.4 // 偏弱
    } else if indicators.rsi > 20.0 {
        0.75 // 超卖（反转机会）
    } else {
        0.9 // 极度超卖（强反转机会）
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
pub(super) fn calculate_pattern_score_enhanced(patterns: &[PatternRecognition]) -> f64 {
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
pub(super) fn calculate_sr_score_enhanced(sr: &SupportResistance) -> f64 {
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
pub(super) fn calculate_sentiment_score_enhanced(indicators: &TechnicalIndicatorValues) -> f64 {
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
pub(super) fn calculate_volatility_score_enhanced(
    volatility: f64,
    level: Option<&VolatilityLevel>,
) -> f64 {
    // 基础评分
    let base: f64 = if volatility < 0.015 {
        0.75 // 极低波动
    } else if volatility < 0.025 {
        0.65 // 低波动
    } else if volatility < 0.035 {
        0.55 // 正常波动
    } else if volatility < 0.05 {
        0.40 // 高波动
    } else {
        0.25 // 极高波动
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

// =============================================================================
// 保留原始函数供向后兼容（允许未使用）
// =============================================================================

#[allow(dead_code)]
pub(super) fn calculate_trend_score(trend: &TrendState) -> f64 {
    match trend {
        TrendState::StrongBullish => 0.9,
        TrendState::Bullish => 0.7,
        TrendState::Neutral => 0.5,
        TrendState::Bearish => 0.3,
        TrendState::StrongBearish => 0.1,
    }
}

#[allow(dead_code)]
pub(super) fn calculate_volume_price_score(signal: &VolumePriceSignal) -> f64 {
    match signal.direction.as_str() {
        "上涨" => 0.5 + signal.confidence * 0.4,
        "下跌" => 0.5 - signal.confidence * 0.4,
        _ => 0.5,
    }
}

#[allow(dead_code)]
pub(super) fn calculate_momentum_score(indicators: &TechnicalIndicatorValues) -> f64 {
    calculate_momentum_score_enhanced(indicators)
}

#[allow(dead_code)]
pub(super) fn calculate_pattern_score(patterns: &[PatternRecognition]) -> f64 {
    calculate_pattern_score_enhanced(patterns)
}

#[allow(dead_code)]
pub(super) fn calculate_sr_score(sr: &SupportResistance) -> f64 {
    calculate_sr_score_enhanced(sr)
}

#[allow(dead_code)]
pub(super) fn calculate_sentiment_score(indicators: &TechnicalIndicatorValues) -> f64 {
    calculate_sentiment_score_enhanced(indicators)
}

#[allow(dead_code)]
pub(super) fn calculate_volatility_score(volatility: f64) -> f64 {
    calculate_volatility_score_enhanced(volatility, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prediction::analysis::volume::VolumePriceSignal;

    fn up_signal() -> VolumePriceSignal {
        VolumePriceSignal {
            direction: "上涨".to_string(),
            confidence: 0.7,
            change_range: (0.5, 3.0),
            signal: "买入".to_string(),
            price_trend: "温和上涨".to_string(),
            volume_trend: "温和放量".to_string(),
            volume_ratio: 1.0,
            key_factors: vec![],
        }
    }

    #[test]
    fn test_volume_ratio_influences_score() {
        let signal = up_signal();
        // 放量上涨（量比2.0）应高于缩量上涨（量比0.5）
        let high_vr = TechnicalIndicatorValues {
            volume_ratio: 2.0,
            ..Default::default()
        };
        let low_vr = TechnicalIndicatorValues {
            volume_ratio: 0.5,
            ..Default::default()
        };
        let s_high = calculate_volume_price_score_enhanced(&signal, &high_vr);
        let s_low = calculate_volume_price_score_enhanced(&signal, &low_vr);
        assert!(s_high > s_low, "放量上涨评分应高于缩量上涨");
    }

    #[test]
    fn test_turnover_healthy_bonus() {
        let signal = up_signal();
        // 健康换手（5%）应不低于无换手数据
        let healthy = TechnicalIndicatorValues {
            turnover_rate: 5.0,
            ..Default::default()
        };
        let none = TechnicalIndicatorValues::default();
        let s_healthy = calculate_volume_price_score_enhanced(&signal, &healthy);
        let s_none = calculate_volume_price_score_enhanced(&signal, &none);
        assert!(s_healthy >= s_none);
    }
}
