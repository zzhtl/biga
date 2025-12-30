//! 增强信号确认系统
//! 
//! 核心功能：
//! 1. 多信号冲突检测与调和
//! 2. 动态信号权重调整
//! 3. 信号一致性评估
//! 4. 假信号过滤

use serde::{Deserialize, Serialize};
use crate::prediction::indicators::TechnicalIndicatorValues;
use crate::prediction::analysis::{TrendState, VolumePriceSignal};
use crate::prediction::analysis::market_regime::{MarketRegime, VolatilityLevel};

// =============================================================================
// 核心类型定义
// =============================================================================

/// 单个信号来源
#[derive(Debug, Clone)]
pub struct SignalSource {
    /// 信号名称
    pub name: String,
    /// 信号方向 (-1.0 到 1.0)
    pub direction: f64,
    /// 信号强度 (0 到 1.0)
    pub strength: f64,
    /// 信号可靠度 (0 到 1.0)
    pub reliability: f64,
    /// 信号类型
    pub signal_type: SignalType,
    /// 是否与市场状态一致
    pub regime_aligned: bool,
}

impl SignalSource {
    /// 计算加权得分
    pub fn weighted_score(&self) -> f64 {
        self.direction * self.strength * self.reliability
    }
}

/// 信号类型分类
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalType {
    Trend,        // 趋势信号
    Momentum,     // 动量信号
    Oscillator,   // 震荡指标信号
    Volume,       // 量能信号
    Pattern,      // 形态信号
    Sentiment,    // 情绪信号
}

impl SignalType {
    /// 获取信号类型在不同市场状态下的基础权重
    pub fn base_weight(&self, regime: &MarketRegime) -> f64 {
        match regime {
            MarketRegime::StrongUptrend | MarketRegime::StrongDowntrend => {
                match self {
                    Self::Trend => 0.30,
                    Self::Momentum => 0.25,
                    Self::Oscillator => 0.15,
                    Self::Volume => 0.20,
                    Self::Pattern => 0.05,
                    Self::Sentiment => 0.05,
                }
            }
            MarketRegime::ModerateUptrend | MarketRegime::ModerateDowntrend => {
                match self {
                    Self::Trend => 0.25,
                    Self::Momentum => 0.22,
                    Self::Oscillator => 0.18,
                    Self::Volume => 0.20,
                    Self::Pattern => 0.08,
                    Self::Sentiment => 0.07,
                }
            }
            MarketRegime::Ranging => {
                match self {
                    Self::Trend => 0.10,
                    Self::Momentum => 0.15,
                    Self::Oscillator => 0.28,
                    Self::Volume => 0.17,
                    Self::Pattern => 0.18,
                    Self::Sentiment => 0.12,
                }
            }
            MarketRegime::PotentialTop | MarketRegime::PotentialBottom => {
                match self {
                    Self::Trend => 0.12,
                    Self::Momentum => 0.18,
                    Self::Oscillator => 0.22,
                    Self::Volume => 0.20,
                    Self::Pattern => 0.15,
                    Self::Sentiment => 0.13,
                }
            }
        }
    }
}

/// 信号冲突类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConflictType {
    NoConflict,           // 无冲突
    MinorConflict,        // 轻微冲突
    ModerateConflict,     // 中等冲突
    SevereConflict,       // 严重冲突
    CompleteConflict,     // 完全对立
}

impl ConflictType {
    /// 冲突导致的置信度折扣
    pub fn confidence_discount(&self) -> f64 {
        match self {
            Self::NoConflict => 1.0,
            Self::MinorConflict => 0.90,
            Self::ModerateConflict => 0.75,
            Self::SevereConflict => 0.55,
            Self::CompleteConflict => 0.35,
        }
    }
}

/// 信号确认结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfirmationResult {
    /// 综合信号方向 (-1.0 到 1.0)
    pub direction: f64,
    /// 综合信号强度 (0 到 1.0)
    pub strength: f64,
    /// 确认级别
    pub confirmation_level: ConfirmationLevel,
    /// 冲突程度
    pub conflict_level: String,
    /// 置信度调整因子
    pub confidence_factor: f64,
    /// 主导信号来源
    pub dominant_signal: String,
    /// 信号一致性得分
    pub consistency_score: f64,
    /// 是否为假信号（高冲突低一致性）
    pub is_potential_false_signal: bool,
    /// 建议采取的行动
    pub suggested_action: String,
}

/// 确认级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmationLevel {
    /// 强确认（4+信号同向，高一致性）
    Strong,
    /// 中等确认（3信号同向）
    Moderate,
    /// 弱确认（2信号同向）
    Weak,
    /// 无效信号（冲突过多）
    Invalid,
}

impl ConfirmationLevel {
    pub fn to_string(&self) -> String {
        match self {
            Self::Strong => "强确认".to_string(),
            Self::Moderate => "中等确认".to_string(),
            Self::Weak => "弱确认".to_string(),
            Self::Invalid => "信号无效".to_string(),
        }
    }
    
    pub fn to_weight(&self) -> f64 {
        match self {
            Self::Strong => 1.0,
            Self::Moderate => 0.75,
            Self::Weak => 0.50,
            Self::Invalid => 0.25,
        }
    }
}

// =============================================================================
// 核心分析函数
// =============================================================================

/// 综合信号确认分析
pub fn analyze_signal_confirmation(
    indicators: &TechnicalIndicatorValues,
    trend: &TrendState,
    volume_signal: &VolumePriceSignal,
    regime: &MarketRegime,
    volatility: &VolatilityLevel,
) -> SignalConfirmationResult {
    // 1. 收集所有信号源
    let signals = collect_all_signals(indicators, trend, volume_signal, regime);
    
    // 2. 检测冲突
    let conflict_type = detect_conflicts(&signals);
    
    // 3. 计算动态权重
    let weighted_signals = apply_dynamic_weights(&signals, regime, volatility);
    
    // 4. 计算综合方向和强度
    let (direction, strength) = calculate_combined_signal(&weighted_signals);
    
    // 5. 评估一致性
    let consistency_score = calculate_consistency(&signals, direction);
    
    // 6. 确定确认级别
    let confirmation_level = determine_confirmation_level(&signals, consistency_score, &conflict_type);
    
    // 7. 检测假信号
    let is_potential_false_signal = check_false_signal(&signals, consistency_score, &conflict_type);
    
    // 8. 找出主导信号
    let dominant_signal = find_dominant_signal(&signals);
    
    // 9. 计算置信度因子
    let confidence_factor = conflict_type.confidence_discount() * 
        consistency_score * 
        confirmation_level.to_weight();
    
    // 10. 生成建议
    let suggested_action = generate_action_suggestion(
        direction,
        strength,
        &confirmation_level,
        is_potential_false_signal,
    );
    
    SignalConfirmationResult {
        direction,
        strength,
        confirmation_level,
        conflict_level: format!("{:?}", conflict_type),
        confidence_factor,
        dominant_signal,
        consistency_score,
        is_potential_false_signal,
        suggested_action,
    }
}

/// 收集所有技术信号
fn collect_all_signals(
    ind: &TechnicalIndicatorValues,
    trend: &TrendState,
    volume: &VolumePriceSignal,
    regime: &MarketRegime,
) -> Vec<SignalSource> {
    let mut signals = Vec::new();
    
    // 趋势信号
    let trend_direction = match trend {
        TrendState::StrongBullish => 1.0,
        TrendState::Bullish => 0.6,
        TrendState::Neutral => 0.0,
        TrendState::Bearish => -0.6,
        TrendState::StrongBearish => -1.0,
    };
    signals.push(SignalSource {
        name: "趋势".to_string(),
        direction: trend_direction,
        strength: trend_direction.abs(),
        reliability: 0.85,
        signal_type: SignalType::Trend,
        regime_aligned: regime.is_trending() == trend.is_bullish() || regime.is_trending() == trend.is_bearish(),
    });
    
    // MACD 信号
    let macd_direction = if ind.macd_golden_cross {
        0.8
    } else if ind.macd_death_cross {
        -0.8
    } else if ind.macd_histogram > 0.0 {
        0.4
    } else if ind.macd_histogram < 0.0 {
        -0.4
    } else {
        0.0
    };
    signals.push(SignalSource {
        name: "MACD".to_string(),
        direction: macd_direction,
        strength: macd_direction.abs(),
        reliability: 0.80,
        signal_type: SignalType::Momentum,
        regime_aligned: true,
    });
    
    // RSI 信号（逆向指标在极端区域）
    let rsi_direction = if ind.rsi < 25.0 {
        0.7  // 极度超卖，看涨
    } else if ind.rsi < 35.0 {
        0.4
    } else if ind.rsi > 75.0 {
        -0.7  // 极度超买，看跌
    } else if ind.rsi > 65.0 {
        -0.4
    } else {
        0.0
    };
    signals.push(SignalSource {
        name: "RSI".to_string(),
        direction: rsi_direction,
        strength: rsi_direction.abs(),
        reliability: 0.75,
        signal_type: SignalType::Oscillator,
        regime_aligned: !regime.is_trending() || rsi_direction.abs() < 0.5,
    });
    
    // KDJ 信号
    let kdj_direction = if ind.kdj_golden_cross && ind.kdj_oversold {
        0.85
    } else if ind.kdj_golden_cross {
        0.5
    } else if ind.kdj_death_cross && ind.kdj_overbought {
        -0.85
    } else if ind.kdj_death_cross {
        -0.5
    } else if ind.kdj_j < 20.0 {
        0.6
    } else if ind.kdj_j > 80.0 {
        -0.6
    } else {
        0.0
    };
    signals.push(SignalSource {
        name: "KDJ".to_string(),
        direction: kdj_direction,
        strength: kdj_direction.abs(),
        reliability: 0.70,
        signal_type: SignalType::Oscillator,
        regime_aligned: true,
    });
    
    // Williams %R 信号
    let wr_direction = if ind.williams_oversold {
        0.65  // 超卖看涨
    } else if ind.williams_overbought {
        -0.65  // 超买看跌
    } else if ind.williams_r < -70.0 {
        0.35
    } else if ind.williams_r > -30.0 {
        -0.35
    } else {
        0.0
    };
    signals.push(SignalSource {
        name: "Williams%R".to_string(),
        direction: wr_direction,
        strength: wr_direction.abs(),
        reliability: 0.68,
        signal_type: SignalType::Oscillator,
        regime_aligned: true,
    });
    
    // ROC 动量信号
    let roc_direction = if ind.roc > 8.0 {
        0.7
    } else if ind.roc > 3.0 {
        0.4
    } else if ind.roc < -8.0 {
        -0.7
    } else if ind.roc < -3.0 {
        -0.4
    } else {
        0.0
    };
    signals.push(SignalSource {
        name: "ROC".to_string(),
        direction: roc_direction,
        strength: roc_direction.abs(),
        reliability: 0.72,
        signal_type: SignalType::Momentum,
        regime_aligned: regime.is_trending(),
    });
    
    // CCI 信号
    let cci_direction = if ind.cci < -200.0 {
        0.75
    } else if ind.cci < -100.0 {
        0.4
    } else if ind.cci > 200.0 {
        -0.75
    } else if ind.cci > 100.0 {
        -0.4
    } else {
        0.0
    };
    signals.push(SignalSource {
        name: "CCI".to_string(),
        direction: cci_direction,
        strength: cci_direction.abs(),
        reliability: 0.65,
        signal_type: SignalType::Oscillator,
        regime_aligned: true,
    });
    
    // 量能信号
    let vol_direction = match volume.direction.as_str() {
        "上涨" => volume.confidence * 0.8,
        "下跌" => -volume.confidence * 0.8,
        _ => 0.0,
    };
    signals.push(SignalSource {
        name: "量价".to_string(),
        direction: vol_direction,
        strength: vol_direction.abs(),
        reliability: 0.78,
        signal_type: SignalType::Volume,
        regime_aligned: true,
    });
    
    // BRAR 人气信号
    let brar_direction = if ind.br > 150.0 && ind.ar > 130.0 {
        -0.5  // 过热
    } else if ind.br < 50.0 || ind.ar < 70.0 {
        0.5   // 超卖
    } else if ind.br > ind.ar && ind.br > 100.0 {
        0.3
    } else if ind.br < ind.ar && ind.br < 100.0 {
        -0.3
    } else {
        0.0
    };
    signals.push(SignalSource {
        name: "BRAR".to_string(),
        direction: brar_direction,
        strength: brar_direction.abs(),
        reliability: 0.60,
        signal_type: SignalType::Sentiment,
        regime_aligned: true,
    });
    
    signals
}

/// 检测信号冲突
fn detect_conflicts(signals: &[SignalSource]) -> ConflictType {
    let mut bullish_count = 0;
    let mut bearish_count = 0;
    let mut bullish_strength = 0.0;
    let mut bearish_strength = 0.0;
    
    for signal in signals {
        if signal.direction > 0.2 {
            bullish_count += 1;
            bullish_strength += signal.direction * signal.strength;
        } else if signal.direction < -0.2 {
            bearish_count += 1;
            bearish_strength += signal.direction.abs() * signal.strength;
        }
    }
    
    let total_signals = bullish_count + bearish_count;
    if total_signals == 0 {
        return ConflictType::NoConflict;
    }
    
    let balance = (bullish_strength - bearish_strength).abs() / (bullish_strength + bearish_strength).max(0.01);
    let direction_diff = if bullish_count > bearish_count { 
        bullish_count - bearish_count 
    } else { 
        bearish_count - bullish_count 
    };
    let direction_ratio = direction_diff as f64 / total_signals as f64;
    
    if direction_ratio > 0.7 && balance > 0.6 {
        ConflictType::NoConflict
    } else if direction_ratio > 0.5 && balance > 0.4 {
        ConflictType::MinorConflict
    } else if direction_ratio > 0.3 && balance > 0.2 {
        ConflictType::ModerateConflict
    } else if direction_ratio > 0.1 {
        ConflictType::SevereConflict
    } else {
        ConflictType::CompleteConflict
    }
}

/// 应用动态权重
fn apply_dynamic_weights(
    signals: &[SignalSource],
    regime: &MarketRegime,
    volatility: &VolatilityLevel,
) -> Vec<(f64, f64)> {
    let volatility_factor = match volatility {
        VolatilityLevel::VeryHigh => 0.70,
        VolatilityLevel::High => 0.85,
        VolatilityLevel::Normal => 1.0,
        VolatilityLevel::Low => 1.05,
        VolatilityLevel::VeryLow => 0.95,
    };
    
    signals.iter().map(|s| {
        let base_weight = s.signal_type.base_weight(regime);
        let regime_bonus = if s.regime_aligned { 1.1 } else { 0.9 };
        let weight = base_weight * s.reliability * regime_bonus * volatility_factor;
        (s.direction * s.strength, weight)
    }).collect()
}

/// 计算综合信号
fn calculate_combined_signal(weighted: &[(f64, f64)]) -> (f64, f64) {
    let total_weight: f64 = weighted.iter().map(|(_, w)| w).sum();
    if total_weight == 0.0 {
        return (0.0, 0.0);
    }
    
    let weighted_sum: f64 = weighted.iter().map(|(s, w)| s * w).sum();
    let direction = (weighted_sum / total_weight).clamp(-1.0, 1.0);
    let strength = direction.abs();
    
    (direction, strength)
}

/// 计算信号一致性
fn calculate_consistency(signals: &[SignalSource], overall_direction: f64) -> f64 {
    if signals.is_empty() {
        return 0.5;
    }
    
    let consistent_signals: f64 = signals.iter()
        .filter(|s| (s.direction > 0.0) == (overall_direction > 0.0) || s.direction.abs() < 0.2)
        .map(|s| s.reliability)
        .sum();
    
    let total_reliability: f64 = signals.iter().map(|s| s.reliability).sum();
    
    if total_reliability == 0.0 {
        return 0.5;
    }
    
    (consistent_signals / total_reliability).clamp(0.3, 1.0)
}

/// 确定确认级别
fn determine_confirmation_level(
    signals: &[SignalSource],
    consistency: f64,
    conflict: &ConflictType,
) -> ConfirmationLevel {
    let strong_signals = signals.iter()
        .filter(|s| s.strength > 0.5 && s.reliability > 0.7)
        .count();
    
    match conflict {
        ConflictType::CompleteConflict => ConfirmationLevel::Invalid,
        ConflictType::SevereConflict => {
            if consistency > 0.7 { ConfirmationLevel::Weak } else { ConfirmationLevel::Invalid }
        }
        _ => {
            if strong_signals >= 4 && consistency > 0.75 {
                ConfirmationLevel::Strong
            } else if strong_signals >= 3 && consistency > 0.6 {
                ConfirmationLevel::Moderate
            } else if strong_signals >= 2 && consistency > 0.5 {
                ConfirmationLevel::Weak
            } else {
                ConfirmationLevel::Invalid
            }
        }
    }
}

/// 检测假信号
fn check_false_signal(
    signals: &[SignalSource],
    consistency: f64,
    conflict: &ConflictType,
) -> bool {
    // 假信号特征：
    // 1. 高冲突 + 低一致性
    // 2. 震荡指标和趋势指标严重分歧
    // 3. 量能不配合
    
    if consistency < 0.45 && matches!(conflict, ConflictType::SevereConflict | ConflictType::CompleteConflict) {
        return true;
    }
    
    // 检查震荡指标和趋势指标分歧
    let trend_signals: Vec<_> = signals.iter()
        .filter(|s| matches!(s.signal_type, SignalType::Trend | SignalType::Momentum))
        .collect();
    let oscillator_signals: Vec<_> = signals.iter()
        .filter(|s| matches!(s.signal_type, SignalType::Oscillator))
        .collect();
    
    if !trend_signals.is_empty() && !oscillator_signals.is_empty() {
        let trend_avg: f64 = trend_signals.iter().map(|s| s.direction).sum::<f64>() / trend_signals.len() as f64;
        let osc_avg: f64 = oscillator_signals.iter().map(|s| s.direction).sum::<f64>() / oscillator_signals.len() as f64;
        
        // 严重分歧
        if (trend_avg > 0.3 && osc_avg < -0.3) || (trend_avg < -0.3 && osc_avg > 0.3) {
            return true;
        }
    }
    
    false
}

/// 找出主导信号
fn find_dominant_signal(signals: &[SignalSource]) -> String {
    signals.iter()
        .max_by(|a, b| a.weighted_score().abs().partial_cmp(&b.weighted_score().abs()).unwrap())
        .map(|s| s.name.clone())
        .unwrap_or_else(|| "无".to_string())
}

/// 生成行动建议
fn generate_action_suggestion(
    direction: f64,
    strength: f64,
    level: &ConfirmationLevel,
    is_false: bool,
) -> String {
    if is_false {
        return "信号冲突严重，建议观望等待信号清晰".to_string();
    }
    
    match level {
        ConfirmationLevel::Invalid => {
            "信号无效，不宜操作".to_string()
        }
        ConfirmationLevel::Weak => {
            if direction > 0.3 {
                format!("弱看涨信号(强度{:.0}%)，可小仓位试探", strength * 100.0)
            } else if direction < -0.3 {
                format!("弱看跌信号(强度{:.0}%)，注意风险", strength * 100.0)
            } else {
                "方向不明，建议观望".to_string()
            }
        }
        ConfirmationLevel::Moderate => {
            if direction > 0.3 {
                format!("中等看涨信号(强度{:.0}%)，可适量建仓", strength * 100.0)
            } else if direction < -0.3 {
                format!("中等看跌信号(强度{:.0}%)，考虑减仓", strength * 100.0)
            } else {
                "信号中性，保持观望".to_string()
            }
        }
        ConfirmationLevel::Strong => {
            if direction > 0.5 {
                format!("强烈看涨信号(强度{:.0}%)，积极做多", strength * 100.0)
            } else if direction < -0.5 {
                format!("强烈看跌信号(强度{:.0}%)，建议规避", strength * 100.0)
            } else {
                "多空均衡，持币观望".to_string()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_conflict_detection() {
        let signals = vec![
            SignalSource {
                name: "test1".to_string(),
                direction: 0.8,
                strength: 0.9,
                reliability: 0.85,
                signal_type: SignalType::Trend,
                regime_aligned: true,
            },
            SignalSource {
                name: "test2".to_string(),
                direction: -0.7,
                strength: 0.8,
                reliability: 0.80,
                signal_type: SignalType::Oscillator,
                regime_aligned: true,
            },
        ];
        
        let conflict = detect_conflicts(&signals);
        assert!(matches!(conflict, ConflictType::SevereConflict | ConflictType::CompleteConflict));
    }
}

