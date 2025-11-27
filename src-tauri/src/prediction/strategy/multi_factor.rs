//! 多因子评分策略

use crate::config::weights::*;
use crate::prediction::analysis::{TrendState, VolumePriceSignal, PatternRecognition, SupportResistance};
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
        }
    }
}

/// 计算多因子综合评分
pub fn calculate_multi_factor_score(
    trend_state: &TrendState,
    volume_signal: &VolumePriceSignal,
    indicators: &TechnicalIndicatorValues,
    patterns: &[PatternRecognition],
    support_resistance: &SupportResistance,
    volatility: f64,
) -> MultiFactorScore {
    // 趋势因子评分
    let trend_score = calculate_trend_score(trend_state);
    
    // 量价因子评分
    let volume_price_score = calculate_volume_price_score(volume_signal);
    
    // 动量因子评分
    let momentum_score = calculate_momentum_score(indicators);
    
    // K线形态因子评分
    let pattern_score = calculate_pattern_score(patterns);
    
    // 支撑阻力因子评分
    let support_resistance_score = calculate_sr_score(support_resistance);
    
    // 情绪因子评分
    let sentiment_score = calculate_sentiment_score(indicators);
    
    // 波动率因子评分
    let volatility_score = calculate_volatility_score(volatility);
    
    // 加权综合评分
    let total_score = 
        trend_score * TREND_FACTOR_WEIGHT +
        volume_price_score * VOLUME_PRICE_FACTOR_WEIGHT +
        momentum_score * MOMENTUM_FACTOR_WEIGHT +
        pattern_score * PATTERN_FACTOR_WEIGHT +
        support_resistance_score * SUPPORT_RESISTANCE_FACTOR_WEIGHT +
        sentiment_score * SENTIMENT_FACTOR_WEIGHT +
        volatility_score * VOLATILITY_FACTOR_WEIGHT;
    
    // 归一化到 0-100
    let total_score = (total_score * 100.0).clamp(0.0, 100.0);
    
    // 生成信号
    let (signal, signal_strength) = generate_signal(total_score);
    
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
    }
}

fn calculate_trend_score(trend: &TrendState) -> f64 {
    match trend {
        TrendState::StrongBullish => 0.9,
        TrendState::Bullish => 0.7,
        TrendState::Neutral => 0.5,
        TrendState::Bearish => 0.3,
        TrendState::StrongBearish => 0.1,
    }
}

fn calculate_volume_price_score(signal: &VolumePriceSignal) -> f64 {
    match signal.direction.as_str() {
        "上涨" => 0.5 + signal.confidence * 0.4,
        "下跌" => 0.5 - signal.confidence * 0.4,
        _ => 0.5,
    }
}

fn calculate_momentum_score(indicators: &TechnicalIndicatorValues) -> f64 {
    let mut score = 0.5;
    
    // RSI 评分
    if indicators.rsi > 70.0 {
        score -= (indicators.rsi - 70.0) / 60.0;
    } else if indicators.rsi < 30.0 {
        score += (30.0 - indicators.rsi) / 60.0;
    }
    
    // MACD 评分
    if indicators.macd_histogram > 0.0 {
        score += 0.1;
    } else if indicators.macd_histogram < 0.0 {
        score -= 0.1;
    }
    
    // 金叉死叉
    if indicators.macd_golden_cross {
        score += 0.15;
    }
    if indicators.macd_death_cross {
        score -= 0.15;
    }
    
    score.clamp(0.0_f64, 1.0_f64)
}

fn calculate_pattern_score(patterns: &[PatternRecognition]) -> f64 {
    let mut score = 0.5;
    
    for pattern in patterns {
        if pattern.is_bullish {
            score += pattern.reliability * 0.2;
        } else {
            score -= pattern.reliability * 0.2;
        }
    }
    
    score.clamp(0.0_f64, 1.0_f64)
}

fn calculate_sr_score(sr: &SupportResistance) -> f64 {
    if sr.current_position.contains("支撑") {
        0.6
    } else if sr.current_position.contains("压力") {
        0.4
    } else {
        0.5
    }
}

fn calculate_sentiment_score(indicators: &TechnicalIndicatorValues) -> f64 {
    let mut score: f64 = 0.5;
    
    // KDJ 评分
    if indicators.kdj_oversold {
        score += 0.15;
    }
    if indicators.kdj_overbought {
        score -= 0.15;
    }
    
    score.clamp(0.0, 1.0)
}

fn calculate_volatility_score(volatility: f64) -> f64 {
    // 波动率越低，得分越高（更稳定）
    if volatility < 0.02 {
        0.7
    } else if volatility < 0.03 {
        0.6
    } else if volatility < 0.05 {
        0.5
    } else {
        0.3
    }
}

fn generate_signal(total_score: f64) -> (String, f64) {
    if total_score >= 75.0 {
        ("强烈看涨".to_string(), 0.9)
    } else if total_score >= 60.0 {
        ("看涨".to_string(), 0.7)
    } else if total_score >= 45.0 {
        ("中性".to_string(), 0.5)
    } else if total_score >= 30.0 {
        ("看跌".to_string(), 0.7)
    } else {
        ("强烈看跌".to_string(), 0.9)
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

