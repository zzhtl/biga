/// 多因子综合评分模型
/// 整合技术指标、K线形态、量价分析等多个维度进行综合评分

use serde::{Deserialize, Serialize};
use super::candlestick_patterns::{PatternRecognition, Direction};

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
        if score >= 85.0 {
            SignalQuality::Excellent
        } else if score >= 70.0 {
            SignalQuality::Good
        } else if score >= 55.0 {
            SignalQuality::Fair
        } else if score >= 40.0 {
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
    
    // 均线多头排列：+30分
    if ma5 > ma10 && ma10 > ma20 && ma20 > ma60 {
        score += 30.0;
        reasons.push("均线多头排列".to_string());
    }
    // 均线空头排列：-30分
    else if ma5 < ma10 && ma10 < ma20 && ma20 < ma60 {
        score -= 30.0;
        reasons.push("均线空头排列".to_string());
    }
    
    // 价格位于均线之上：+15分
    if current_price > ma5 && current_price > ma20 {
        score += 15.0;
        reasons.push("价格位于主要均线之上".to_string());
    }
    // 价格位于均线之下：-15分
    else if current_price < ma5 && current_price < ma20 {
        score -= 15.0;
        reasons.push("价格位于主要均线之下".to_string());
    }
    
    // 短期均线向上发散：+10分
    let ma5_slope = (ma5 - ma10) / ma10;
    if ma5_slope > 0.02 {
        score += 10.0;
        reasons.push("短期均线向上发散".to_string());
    } else if ma5_slope < -0.02 {
        score -= 10.0;
        reasons.push("短期均线向下发散".to_string());
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
        name: "趋势因子".to_string(),
        score,
        weight: 0.25,
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
    
    // 量价配合：+20分
    if volume_price_sync {
        score += 20.0;
        reasons.push("量价配合良好".to_string());
    } else {
        score -= 15.0;
        reasons.push("量价背离".to_string());
    }
    
    // OBV趋势：±15分
    if obv_trend.contains("上升") {
        score += 15.0;
        reasons.push("OBV上升趋势".to_string());
    } else if obv_trend.contains("下降") {
        score -= 15.0;
        reasons.push("OBV下降趋势".to_string());
    }
    
    // 吸筹信号：最高+20分
    if accumulation_signal > 70.0 {
        score += 20.0;
        reasons.push("强烈吸筹信号".to_string());
    } else if accumulation_signal > 50.0 {
        score += 10.0;
        reasons.push("检测到吸筹".to_string());
    }
    
    // 量能趋势：±10分
    if volume_trend == "放量" {
        score += 10.0;
        reasons.push("成交量放大".to_string());
    } else if volume_trend == "缩量" {
        score -= 5.0;
        reasons.push("成交量萎缩".to_string());
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
        name: "量价因子".to_string(),
        score,
        weight: 0.20,
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
            weight: 0.15,
            description: "未检测到明显K线形态".to_string(),
            status: FactorStatus::Neutral,
        };
    }
    
    // 按可靠性加权评分
    for pattern in patterns {
        let pattern_score = pattern.strength * pattern.reliability * 100.0;
        
        match pattern.direction {
            Direction::Bullish => {
                score += pattern_score * 0.5;
                reasons.push(format!("看涨形态: {}", pattern.description));
            }
            Direction::Bearish => {
                score -= pattern_score * 0.5;
                reasons.push(format!("看跌形态: {}", pattern.description));
            }
            Direction::Neutral => {
                // 中性形态，略微降分（不确定性）
                score -= 5.0;
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
        weight: 0.15,
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
    
    // RSI评分
    if rsi < 30.0 {
        score += 20.0;
        reasons.push("RSI超卖".to_string());
    } else if rsi > 70.0 {
        score -= 20.0;
        reasons.push("RSI超买".to_string());
    } else if rsi > 45.0 && rsi < 55.0 {
        score += 5.0;
        reasons.push("RSI中性偏强".to_string());
    }
    
    // MACD评分
    if macd_dif > macd_dea {
        if macd_dif > 0.0 && macd_dea > 0.0 {
            score += 20.0;
            reasons.push("MACD双线上穿零轴".to_string());
        } else {
            score += 15.0;
            reasons.push("MACD金叉".to_string());
        }
    } else {
        if macd_dif < 0.0 && macd_dea < 0.0 {
            score -= 20.0;
            reasons.push("MACD双线下穿零轴".to_string());
        } else {
            score -= 15.0;
            reasons.push("MACD死叉".to_string());
        }
    }
    
    // MACD柱状图变化
    let macd_bar = macd_dif - macd_dea;
    if macd_bar > 0.0 {
        score += 10.0;
        reasons.push("MACD红柱".to_string());
    } else {
        score -= 10.0;
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
        weight: 0.15,
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
    
    // 计算距离最近支撑位的距离
    if let Some(&nearest_support) = support_levels.first() {
        let distance_pct = (current_price - nearest_support) / current_price * 100.0;
        
        if distance_pct < 2.0 {
            score += 25.0;
            reasons.push(format!("接近强支撑({:.2}元)", nearest_support));
        } else if distance_pct < 5.0 {
            score += 15.0;
            reasons.push("靠近支撑区域".to_string());
        }
    }
    
    // 计算距离最近压力位的距离
    if let Some(&nearest_resistance) = resistance_levels.first() {
        let distance_pct = (nearest_resistance - current_price) / current_price * 100.0;
        
        if distance_pct < 2.0 {
            score -= 25.0;
            reasons.push(format!("接近强压力({:.2}元)", nearest_resistance));
        } else if distance_pct < 5.0 {
            score -= 15.0;
            reasons.push("靠近压力区域".to_string());
        } else if distance_pct > 10.0 {
            score += 10.0;
            reasons.push("上方空间充足".to_string());
        }
    }
    
    // 位置评估
    if !support_levels.is_empty() && !resistance_levels.is_empty() {
        let support = support_levels.first().unwrap();
        let resistance = resistance_levels.first().unwrap();
        let range = resistance - support;
        
        if range > 0.0 {
            let position = (current_price - support) / range;
            if position < 0.3 {
                score += 10.0;
                reasons.push("位于区间底部".to_string());
            } else if position > 0.7 {
                score -= 10.0;
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
        weight: 0.10,
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
    
    // 共振级别评分（0-3级）
    score += resonance_level as f64 * 12.0; // 每级+12分，最高+36分
    
    if resonance_level >= 2 {
        reasons.push(format!("{}级共振", resonance_level));
    }
    
    // 共振方向
    if resonance_direction.contains("多头") {
        score += 15.0;
        reasons.push("多周期多头共振".to_string());
    } else if resonance_direction.contains("空头") {
        score -= 15.0;
        reasons.push("多周期空头共振".to_string());
    }
    
    // 信号质量
    score += (signal_quality - 50.0) * 0.3; // 信号质量影响
    
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
        weight: 0.15,
        description: if reasons.is_empty() { 
            "无明显共振".to_string() 
        } else { 
            reasons.join("; ") 
        },
        status,
    }
}

/// 计算综合评分
pub fn calculate_multi_factor_score(factors: Vec<FactorScore>) -> MultiFactorScore {
    // 加权平均计算总分
    let total_score: f64 = factors.iter()
        .map(|f| f.score * f.weight)
        .sum();
    
    let signal_quality = SignalQuality::from_score(total_score);
    
    // 生成操作建议
    let operation_suggestion = if total_score >= 75.0 {
        "强烈建议买入，设置止损位后建仓".to_string()
    } else if total_score >= 65.0 {
        "可以考虑买入，注意控制仓位".to_string()
    } else if total_score >= 55.0 {
        "可以轻仓试探，严格止损".to_string()
    } else if total_score >= 45.0 {
        "观望为主，等待更好时机".to_string()
    } else if total_score >= 35.0 {
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