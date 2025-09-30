/// 多因子综合评分模型
/// 整合技术指标、K线形态、量价分析等多个维度进行综合评分

use serde::{Deserialize, Serialize};
use super::candlestick_patterns::{PatternRecognition, Direction};
use super::core_weights::*; // 核心评分权重
use super::constants::*;    // 技术参数配置

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiFactorScore {
    pub total_score: f64,              // 总分 0-100
    pub factors: Vec<FactorScore>,     // 各因子得分
    pub signal_quality: SignalQuality, // 信号质量等级
    pub operation_suggestion: String,   // 操作建议
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorScore {
    pub name: String,           // 因子名称
    pub score: f64,             // 得分 0-100
    pub weight: f64,            // 权重
    pub description: String,    // 描述
    pub status: FactorStatus,   // 状态
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactorStatus {
    VeryBullish,    // 非常看涨
    Bullish,        // 看涨
    Neutral,        // 中性
    Bearish,        // 看跌
    VeryBearish,    // 非常看跌
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalQuality {
    Excellent,      // 优秀 (85-100分)
    Good,           // 良好 (70-85分)
    Fair,           // 一般 (55-70分)
    Poor,           // 较差 (40-55分)
    VeryPoor,       // 很差 (<40分)
}

impl SignalQuality {
    pub fn from_score(score: f64) -> Self {
        use crate::stock_prediction::constants::*;
        
        if score >= EXCELLENT_SIGNAL_THRESHOLD {
            SignalQuality::Excellent
        } else if score >= GOOD_SIGNAL_THRESHOLD {
            SignalQuality::Good
        } else if score >= FAIR_SIGNAL_THRESHOLD {
            SignalQuality::Fair
        } else if score >= POOR_SIGNAL_THRESHOLD {
            SignalQuality::Poor
        } else {
            SignalQuality::VeryPoor
        }
    }
    
    pub fn to_string(&self) -> &str {
        match self {
            SignalQuality::Excellent => "优秀",
            SignalQuality::Good => "良好",
            SignalQuality::Fair => "一般",
            SignalQuality::Poor => "较差",
            SignalQuality::VeryPoor => "很差",
        }
    }
}

/// 趋势因子评分
pub fn score_trend_factor(
    ma5: f64,
    ma10: f64,
    ma20: f64,
    ma60: f64,
    current_price: f64,
) -> FactorScore {
    let mut score: f64 = 50.0; // 基准分
    let mut reasons: Vec<String> = Vec::new();
    
    // 均线多头排列：使用配置常量
    if ma5 > ma10 && ma10 > ma20 && ma20 > ma60 {
        score += BULLISH_MA_ALIGNMENT_SCORE;
        reasons.push("均线多头排列".to_string());
    }
    // 均线空头排列：使用配置常量
    else if ma5 < ma10 && ma10 < ma20 && ma20 < ma60 {
        score -= BEARISH_MA_ALIGNMENT_SCORE;
        reasons.push("均线空头排列".to_string());
    }
    
    // 价格位于均线之上：使用配置常量
    if current_price > ma5 && current_price > ma20 {
        score += PRICE_ABOVE_MA_SCORE;
        reasons.push("价格位于主要均线之上".to_string());
    }
    // 价格位于均线之下：使用配置常量
    else if current_price < ma5 && current_price < ma20 {
        score -= PRICE_BELOW_MA_SCORE;
        reasons.push("价格位于主要均线之下".to_string());
    }
    
    // 短期均线向上发散：使用配置常量
    let ma5_slope = (ma5 - ma10) / ma10;
    if ma5_slope > MA_DIVERGENCE_THRESHOLD {
        score += MA_UPWARD_DIVERGENCE_SCORE;
        reasons.push("短期均线向上发散".to_string());
    } else if ma5_slope < -MA_DIVERGENCE_THRESHOLD {
        score -= MA_UPWARD_DIVERGENCE_SCORE;
        reasons.push("短期均线向下发散".to_string());
    }
    
    // 严格限制在5-95分范围
    score = score.clamp(5.0, 95.0);
    
    let status = if score >= 75.0 {
        FactorStatus::VeryBullish
    } else if score >= 60.0 {
        FactorStatus::Bullish
    } else if score >= 40.0 {
        FactorStatus::Neutral
    } else if score >= 25.0 {
        FactorStatus::Bearish
    } else {
        FactorStatus::VeryBearish
    };
    
    FactorScore {
        name: "趋势因子".to_string(),
        score,
        weight: TREND_FACTOR_WEIGHT,
        description: reasons.join("; "),
        status,
    }
}

/// 量价因子评分
pub fn score_volume_factor(
    volume_trend: &str,
    volume_price_sync: bool,
    accumulation_signal: f64,
    obv_trend: &str,
) -> FactorScore {
    let mut score: f64 = 50.0;
    let mut reasons: Vec<String> = Vec::new();
    
    // 量价配合：使用配置常量
    if volume_price_sync {
        score += VOLUME_PRICE_SYNC_SCORE;
        reasons.push("量价配合良好".to_string());
    } else {
        score -= VOLUME_PRICE_SYNC_SCORE;
        reasons.push("量价背离".to_string());
    }
    
    // OBV趋势：使用配置常量
    if obv_trend.contains("上升") {
        score += OBV_UPTREND_SCORE;
        reasons.push("OBV上升趋势".to_string());
    } else if obv_trend.contains("下降") {
        score -= OBV_UPTREND_SCORE;
        reasons.push("OBV下降趋势".to_string());
    }
    
    // 吸筹信号：使用配置常量
    if accumulation_signal > STRONG_ACCUMULATION_THRESHOLD {
        score += STRONG_ACCUMULATION_SCORE;
        reasons.push("强烈吸筹信号".to_string());
    } else if accumulation_signal > NORMAL_ACCUMULATION_THRESHOLD {
        score += NORMAL_ACCUMULATION_SCORE;
        reasons.push("检测到吸筹".to_string());
    }
    
    // 量能趋势：使用配置常量
    if volume_trend == "放量" {
        score += VOLUME_SURGE_SCORE;
        reasons.push("成交量放大".to_string());
    } else if volume_trend == "缩量" {
        score -= VOLUME_SHRINK_SCORE;
        reasons.push("成交量萎缩".to_string());
    }
    
    // 严格限制在5-95分范围
    score = score.clamp(5.0, 95.0);
    
    let status = if score >= 75.0 {
        FactorStatus::VeryBullish
    } else if score >= 60.0 {
        FactorStatus::Bullish
    } else if score >= 40.0 {
        FactorStatus::Neutral
    } else if score >= 25.0 {
        FactorStatus::Bearish
    } else {
        FactorStatus::VeryBearish
    };
    
    FactorScore {
        name: "量价因子".to_string(),
        score,
        weight: VOLUME_PRICE_FACTOR_WEIGHT,
        description: reasons.join("; "),
        status,
    }
}

/// K线形态因子评分
pub fn score_pattern_factor(patterns: &[PatternRecognition]) -> FactorScore {
    let mut score: f64 = 50.0;
    let mut reasons: Vec<String> = Vec::new();
    
    if patterns.is_empty() {
        return FactorScore {
            name: "形态因子".to_string(),
            score: 50.0,
            weight: PATTERN_FACTOR_WEIGHT,
            description: "未检测到明显K线形态".to_string(),
            status: FactorStatus::Neutral,
        };
    }
    
    // 按可靠性加权评分
    for pattern in patterns {
        let pattern_score = pattern.strength * pattern.reliability * 100.0;
        
        match pattern.direction {
            Direction::Bullish => {
                score += pattern_score * PATTERN_IMPACT_FACTOR;
                reasons.push(format!("看涨形态: {}", pattern.description));
            }
            Direction::Bearish => {
                score -= pattern_score * PATTERN_IMPACT_FACTOR;
                reasons.push(format!("看跌形态: {}", pattern.description));
            }
            Direction::Neutral => {
                // 中性形态，略微降分（不确定性）
                score -= NEUTRAL_PATTERN_PENALTY;
            }
        }
    }
    
    score = score.max(0.0).min(100.0);
    
    let status = if score >= 75.0 {
        FactorStatus::VeryBullish
    } else if score >= 60.0 {
        FactorStatus::Bullish
    } else if score >= 40.0 {
        FactorStatus::Neutral
    } else if score >= 25.0 {
        FactorStatus::Bearish
    } else {
        FactorStatus::VeryBearish
    };
    
    FactorScore {
        name: "形态因子".to_string(),
        score,
        weight: PATTERN_FACTOR_WEIGHT,
        description: if reasons.is_empty() { 
            "无明显形态".to_string() 
        } else { 
            reasons.join("; ") 
        },
        status,
    }
}

/// 动量因子评分
pub fn score_momentum_factor(
    rsi: f64,
    macd_dif: f64,
    macd_dea: f64,
) -> FactorScore {
    let mut score: f64 = 50.0;
    let mut reasons: Vec<String> = Vec::new();
    
    // RSI评分：使用配置常量
    if rsi < RSI_OVERSOLD_THRESHOLD {
        score += RSI_OVERSOLD_SCORE;
        reasons.push("RSI超卖".to_string());
    } else if rsi > RSI_OVERBOUGHT_THRESHOLD {
        score -= RSI_OVERBOUGHT_SCORE;
        reasons.push("RSI超买".to_string());
    } else if rsi > 45.0 && rsi < 55.0 {
        score += 5.0;
        reasons.push("RSI中性偏强".to_string());
    }
    
    // MACD评分：使用配置常量
    if macd_dif > macd_dea {
        if macd_dif > 0.0 && macd_dea > 0.0 {
            score += MACD_BULLISH_ABOVE_ZERO_SCORE;
            reasons.push("MACD双线上穿零轴".to_string());
        } else {
            score += MACD_GOLDEN_CROSS_SCORE;
            reasons.push("MACD金叉".to_string());
        }
    } else {
        if macd_dif < 0.0 && macd_dea < 0.0 {
            score -= MACD_BEARISH_BELOW_ZERO_SCORE;
            reasons.push("MACD双线下穿零轴".to_string());
        } else {
            score -= MACD_DEATH_CROSS_SCORE;
            reasons.push("MACD死叉".to_string());
        }
    }
    
    // MACD柱状图变化：使用配置常量
    let macd_bar = macd_dif - macd_dea;
    if macd_bar > 0.0 {
        score += MACD_POSITIVE_BAR_SCORE;
        reasons.push("MACD红柱".to_string());
    } else {
        score -= MACD_NEGATIVE_BAR_SCORE;
        reasons.push("MACD绿柱".to_string());
    }
    
    score = score.max(0.0).min(100.0);
    
    let status = if score >= 75.0 {
        FactorStatus::VeryBullish
    } else if score >= 60.0 {
        FactorStatus::Bullish
    } else if score >= 40.0 {
        FactorStatus::Neutral
    } else if score >= 25.0 {
        FactorStatus::Bearish
    } else {
        FactorStatus::VeryBearish
    };
    
    FactorScore {
        name: "动量因子".to_string(),
        score,
        weight: MOMENTUM_FACTOR_WEIGHT,
        description: reasons.join("; "),
        status,
    }
}

/// 支撑压力因子评分
pub fn score_support_resistance_factor(
    current_price: f64,
    support_levels: &[f64],
    resistance_levels: &[f64],
) -> FactorScore {
    let mut score: f64 = 50.0;
    let mut reasons: Vec<String> = Vec::new();
    
    // 计算距离最近支撑位的距离：使用配置常量
    if let Some(&nearest_support) = support_levels.first() {
        let distance_pct = (current_price - nearest_support) / current_price;
        
        if distance_pct < STRONG_LEVEL_DISTANCE_THRESHOLD {
            score += NEAR_STRONG_SUPPORT_SCORE;
            reasons.push(format!("接近强支撑({:.2}元)", nearest_support));
        } else if distance_pct < NORMAL_LEVEL_DISTANCE_THRESHOLD {
            score += NEAR_SUPPORT_SCORE;
            reasons.push("靠近支撑区域".to_string());
        }
    }
    
    // 计算距离最近压力位的距离：使用配置常量
    if let Some(&nearest_resistance) = resistance_levels.first() {
        let distance_pct = (nearest_resistance - current_price) / current_price;
        
        if distance_pct < STRONG_LEVEL_DISTANCE_THRESHOLD {
            score -= NEAR_STRONG_RESISTANCE_SCORE;
            reasons.push(format!("接近强压力({:.2}元)", nearest_resistance));
        } else if distance_pct < NORMAL_LEVEL_DISTANCE_THRESHOLD {
            score -= NEAR_RESISTANCE_SCORE;
            reasons.push("靠近压力区域".to_string());
        } else if distance_pct > UPSIDE_SUFFICIENT_THRESHOLD {
            score += SUFFICIENT_UPSIDE_SCORE;
            reasons.push("上方空间充足".to_string());
        }
    }
    
    // 位置评估：使用配置常量
    if !support_levels.is_empty() && !resistance_levels.is_empty() {
        let support = support_levels.first().unwrap();
        let resistance = resistance_levels.first().unwrap();
        let range = resistance - support;
        
        if range > 0.0 {
            let position = (current_price - support) / range;
            if position < BOTTOM_ZONE_THRESHOLD {
                score += BOTTOM_POSITION_SCORE;
                reasons.push("位于区间底部".to_string());
            } else if position > TOP_ZONE_THRESHOLD {
                score -= TOP_POSITION_SCORE;
                reasons.push("位于区间顶部".to_string());
            }
        }
    }
    
    score = score.max(0.0).min(100.0);
    
    let status = if score >= 75.0 {
        FactorStatus::VeryBullish
    } else if score >= 60.0 {
        FactorStatus::Bullish
    } else if score >= 40.0 {
        FactorStatus::Neutral
    } else if score >= 25.0 {
        FactorStatus::Bearish
    } else {
        FactorStatus::VeryBearish
    };
    
    FactorScore {
        name: "支撑压力".to_string(),
        score,
        weight: SUPPORT_RESISTANCE_FACTOR_WEIGHT,
        description: if reasons.is_empty() { 
            "位置中性".to_string() 
        } else { 
            reasons.join("; ") 
        },
        status,
    }
}

/// 多周期共振因子评分
pub fn score_multi_timeframe_factor(
    resonance_level: i32,
    resonance_direction: &str,
    signal_quality: f64,
) -> FactorScore {
    let mut score: f64 = 50.0;
    let mut reasons: Vec<String> = Vec::new();
    
    // 共振级别评分（0-3级）：使用配置常量
    score += resonance_level as f64 * RESONANCE_LEVEL_SCORE;
    
    if resonance_level >= 2 {
        reasons.push(format!("{}级共振", resonance_level));
    }
    
    // 共振方向：使用配置常量
    if resonance_direction.contains("多头") {
        score += MULTI_BULLISH_RESONANCE_SCORE;
        reasons.push("多周期多头共振".to_string());
    } else if resonance_direction.contains("空头") {
        score -= MULTI_BEARISH_RESONANCE_SCORE;
        reasons.push("多周期空头共振".to_string());
    }
    
    // 信号质量：使用配置常量
    score += (signal_quality - 50.0) * SIGNAL_QUALITY_IMPACT;
    
    score = score.max(0.0).min(100.0);
    
    let status = if score >= 75.0 {
        FactorStatus::VeryBullish
    } else if score >= 60.0 {
        FactorStatus::Bullish
    } else if score >= 40.0 {
        FactorStatus::Neutral
    } else if score >= 25.0 {
        FactorStatus::Bearish
    } else {
        FactorStatus::VeryBearish
    };
    
    FactorScore {
        name: "多周期共振".to_string(),
        score,
        weight: MULTI_TIMEFRAME_FACTOR_WEIGHT,
        description: if reasons.is_empty() { 
            "无明显共振".to_string() 
        } else { 
            reasons.join("; ") 
        },
        status,
    }
}

/// 市场情绪因子评分
pub fn score_sentiment_factor(
    sentiment_score: f64,
    fear_greed_index: f64,
    market_phase: &str,
) -> FactorScore {
    let mut score = sentiment_score; // 直接使用情绪得分
    let mut reasons = Vec::new();
    
    reasons.push(format!("市场阶段: {}", market_phase));
    
    // 根据恐惧贪婪指数调整
    if fear_greed_index > 75.0 {
        reasons.push("极度贪婪,警惕回调风险".to_string());
        score = (score * 0.8).min(70.0); // 限制最高得分,避免追高
    } else if fear_greed_index < 25.0 {
        reasons.push("极度恐惧,可能是机会".to_string());
        score = (score + 10.0).min(75.0); // 恐慌时反而加分(逆向思维)
    }
    
    let status = if score >= 75.0 {
        FactorStatus::VeryBullish
    } else if score >= 60.0 {
        FactorStatus::Bullish
    } else if score >= 40.0 {
        FactorStatus::Neutral
    } else if score >= 25.0 {
        FactorStatus::Bearish
    } else {
        FactorStatus::VeryBearish
    };
    
    FactorScore {
        name: "市场情绪".to_string(),
        score,
        weight: SENTIMENT_FACTOR_WEIGHT,
        description: reasons.join("; "),
        status,
    }
}

/// 波动率因子评分
pub fn score_volatility_factor(
    atr: f64,
    current_price: f64,
) -> FactorScore {
    let mut score: f64 = 50.0;
    let mut reasons: Vec<String> = Vec::new();
    
    // 计算波动率百分比
    let volatility_pct = atr / current_price;
    
    // 金融逻辑：使用配置常量
    // - 低波动: 市场稳定,适合持仓
    // - 中等波动: 正常波动
    // - 高波动: 风险增加
    // - 极高波动: 极端风险
    
    if volatility_pct < VERY_LOW_VOLATILITY_THRESHOLD {
        score += VERY_LOW_VOLATILITY_SCORE;
        reasons.push(format!("极低波动({:.2}%),市场平稳", volatility_pct * 100.0));
    } else if volatility_pct < LOW_VOLATILITY_THRESHOLD {
        score += LOW_VOLATILITY_SCORE;
        reasons.push(format!("低波动({:.2}%),适合持仓", volatility_pct * 100.0));
    } else if volatility_pct < MEDIUM_VOLATILITY_THRESHOLD {
        reasons.push(format!("中等波动({:.2}%),正常范围", volatility_pct * 100.0));
    } else if volatility_pct < HIGH_VOLATILITY_THRESHOLD {
        score -= HIGH_VOLATILITY_SCORE;
        reasons.push(format!("高波动({:.2}%),注意风险", volatility_pct * 100.0));
    } else {
        score -= VERY_HIGH_VOLATILITY_SCORE;
        reasons.push(format!("极高波动({:.2}%),极端风险", volatility_pct * 100.0));
    }
    
    score = score.clamp(0.0, 100.0);
    
    let status = if score >= 70.0 {
        FactorStatus::Bullish
    } else if score >= 50.0 {
        FactorStatus::Neutral
    } else if score >= 30.0 {
        FactorStatus::Bearish
    } else {
        FactorStatus::VeryBearish
    };
    
    FactorScore {
        name: "波动率".to_string(),
        score,
        weight: VOLATILITY_FACTOR_WEIGHT,
        description: reasons.join("; "),
        status,
    }
}

/// 智能权重调整 - 根据市场环境动态调整各因子权重
/// 金融逻辑: 
/// - 强趋势市场: 趋势因子和多周期共振权重增加
/// - 震荡市场: 支撑压力和形态因子权重增加
/// - 高波动市场: 降低所有因子权重,提高风险意识
/// 
/// v2修正: 降低调整幅度,避免权重失衡
pub fn adjust_factor_weights(
    factors: &mut [FactorScore],
    market_phase: &str,
    volatility_pct: f64,
    adx: f64, // ADX趋势强度指标
) {
    // 根据ADX调整权重：使用配置常量
    let trend_multiplier = if adx > STRONG_TREND_ADX_THRESHOLD {
        STRONG_TREND_MULTIPLIER // 强趋势,趋势因子权重增加
    } else if adx > MEDIUM_TREND_ADX_THRESHOLD {
        MEDIUM_TREND_MULTIPLIER // 中等趋势,权重适度增加
    } else {
        WEAK_TREND_MULTIPLIER // 弱趋势/震荡,权重降低
    };
    
    // 根据市场阶段调整 (降低调整幅度)
    let phase_adjustments: std::collections::HashMap<&str, Vec<(&str, f64)>> = [
        ("过热期-注意风险", vec![("情绪", 1.1), ("波动率", 1.15), ("趋势", 0.95)]),
        ("上升期", vec![("趋势", 1.12), ("多周期", 1.12), ("动量", 1.08)]),
        ("震荡期", vec![("支撑压力", 1.15), ("形态", 1.1), ("趋势", 0.90)]),
        ("下跌期", vec![("波动率", 1.1), ("情绪", 1.08), ("支撑压力", 1.12)]),
        ("恐慌期-机会期", vec![("情绪", 1.15), ("支撑压力", 1.15), ("波动率", 1.1)]),
    ].iter().cloned().collect();
    
    // 应用调整
    for factor in factors.iter_mut() {
        let factor_name = factor.name.as_str();
        
        // ADX趋势调整
        if factor_name == "趋势因子" || factor_name == "多周期共振" {
            factor.weight *= trend_multiplier;
        } else if factor_name == "支撑压力" || factor_name == "形态因子" {
            factor.weight *= 2.0 - trend_multiplier; // 反向调整
        }
        
        // 市场阶段调整
        if let Some(adjustments) = phase_adjustments.get(market_phase) {
            for (name_keyword, multiplier) in adjustments {
                if factor_name.contains(name_keyword) {
                    factor.weight *= multiplier;
                }
            }
        }
        
        // 高波动惩罚：使用配置常量
        if volatility_pct > HIGH_VOLATILITY_THRESHOLD {
            factor.weight *= HIGH_VOLATILITY_PENALTY; // 极高波动,权重降低
        } else if volatility_pct > MEDIUM_VOLATILITY_THRESHOLD {
            factor.weight *= MEDIUM_HIGH_VOLATILITY_PENALTY; // 高波动,权重小幅降低
        }
    }
    
    // 权重归一化
    let total_weight: f64 = factors.iter().map(|f| f.weight).sum();
    if total_weight > 0.0 {
        for factor in factors.iter_mut() {
            factor.weight /= total_weight;
            
            // ⭐ 关键修正: 设置权重上下限,避免极端情况：使用配置常量
            factor.weight = factor.weight.clamp(MIN_SINGLE_FACTOR_WEIGHT, MAX_SINGLE_FACTOR_WEIGHT);
        }
    }
    
    // 再次归一化,确保总和为1.0
    let final_total: f64 = factors.iter().map(|f| f.weight).sum();
    if final_total > 0.0 {
        for factor in factors.iter_mut() {
            factor.weight /= final_total;
        }
    }
}

/// 计算综合评分
pub fn calculate_multi_factor_score(factors: Vec<FactorScore>) -> MultiFactorScore {
    // 加权平均计算总分
    let total_score: f64 = factors.iter()
        .map(|f| f.score * f.weight)
        .sum();
    
    let signal_quality = SignalQuality::from_score(total_score);
    
    // 生成操作建议：使用配置常量
    let operation_suggestion = if total_score >= STRONG_BUY_THRESHOLD {
        "强烈建议买入，设置止损位后建仓".to_string()
    } else if total_score >= BUY_THRESHOLD {
        "可以考虑买入，注意控制仓位".to_string()
    } else if total_score >= LIGHT_BUY_THRESHOLD {
        "可以轻仓试探，严格止损".to_string()
    } else if total_score >= HOLD_THRESHOLD {
        "观望为主，等待更好时机".to_string()
    } else if total_score >= CONSIDER_SELL_THRESHOLD {
        "不建议买入，考虑减仓".to_string()
    } else {
        "建议卖出或空仓，规避风险".to_string()
    };
    
    MultiFactorScore {
        total_score,
        factors,
        signal_quality,
        operation_suggestion,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trend_factor_bullish() {
        let score = score_trend_factor(15.0, 14.0, 13.0, 12.0, 15.5);
        assert!(score.score > 70.0);
        assert!(matches!(score.status, FactorStatus::VeryBullish | FactorStatus::Bullish));
    }
    
    #[test]
    fn test_signal_quality() {
        assert_eq!(SignalQuality::from_score(90.0), SignalQuality::Excellent);
        assert_eq!(SignalQuality::from_score(75.0), SignalQuality::Good);
        assert_eq!(SignalQuality::from_score(60.0), SignalQuality::Fair);
    }
} 