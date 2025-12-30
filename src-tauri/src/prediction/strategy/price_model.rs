//! 增强价格预测模型
//! 
//! 核心特性：
//! 1. 动态均值回归 - 基于VWAP和布林带
//! 2. 趋势动量结合 - 考虑动量持续性
//! 3. 波动率自适应 - 高波动时预测保守
//! 4. 支撑阻力敏感 - 接近关键位时调整预测

use crate::prediction::analysis::{
    market_regime::{MarketRegime, VolatilityLevel, StrategyType},
    TrendState, SupportResistance,
};

/// 价格预测上下文
pub struct PricePredictionContext {
    pub current_price: f64,
    pub volatility: f64,
    pub regime: MarketRegime,
    pub volatility_level: VolatilityLevel,
    pub trend: TrendState,
    pub vwap_deviation: f64,
    pub bollinger_position: f64,
    pub support_resistance: SupportResistance,
    pub recent_momentum: f64,
}

/// 价格预测结果
#[derive(Debug, Clone)]
pub struct PricePredictionResult {
    /// 预期变化百分比
    pub expected_change: f64,
    /// 预测区间（低, 高）
    pub prediction_range: (f64, f64),
    /// 预测置信度
    pub confidence: f64,
    /// 各模型贡献
    pub model_contributions: ModelContributions,
    /// 风险调整因子
    pub risk_adjustment: f64,
}

/// 各模型贡献详情
#[derive(Debug, Clone)]
pub struct ModelContributions {
    pub trend_contribution: f64,
    pub mean_reversion_contribution: f64,
    pub momentum_contribution: f64,
    pub volatility_adjustment: f64,
    pub sr_adjustment: f64,
}

/// 计算增强价格预测
pub fn calculate_enhanced_price_prediction(ctx: &PricePredictionContext) -> PricePredictionResult {
    let strategy = ctx.regime.recommended_strategy();
    
    // 1. 趋势成分
    let trend_contribution = calculate_trend_contribution(&ctx.trend, &ctx.regime);
    
    // 2. 均值回归成分
    let mean_reversion = calculate_mean_reversion_contribution(
        ctx.vwap_deviation,
        ctx.bollinger_position,
        &strategy,
    );
    
    // 3. 动量成分
    let momentum_contribution = calculate_momentum_contribution(ctx.recent_momentum, &ctx.regime);
    
    // 4. 波动率调整
    let volatility_adjustment = calculate_volatility_adjustment(&ctx.volatility_level, ctx.volatility);
    
    // 5. 支撑阻力调整
    let sr_adjustment = calculate_sr_adjustment(
        ctx.current_price,
        &ctx.support_resistance,
        trend_contribution + momentum_contribution,
    );
    
    // 6. 综合计算预期变化
    let (weights, expected_change) = combine_predictions(
        trend_contribution,
        mean_reversion,
        momentum_contribution,
        volatility_adjustment,
        sr_adjustment,
        &strategy,
    );
    
    // 7. 计算预测区间
    let prediction_range = calculate_prediction_range(
        expected_change,
        ctx.volatility,
        &ctx.volatility_level,
    );
    
    // 8. 计算置信度
    let confidence = calculate_prediction_confidence(
        &ctx.regime,
        &ctx.volatility_level,
        trend_contribution.abs(),
    );
    
    // 9. 风险调整
    let risk_adjustment = calculate_risk_adjustment(&ctx.volatility_level, sr_adjustment.abs());
    
    PricePredictionResult {
        expected_change,
        prediction_range,
        confidence,
        model_contributions: ModelContributions {
            trend_contribution: trend_contribution * weights.0,
            mean_reversion_contribution: mean_reversion * weights.1,
            momentum_contribution: momentum_contribution * weights.2,
            volatility_adjustment,
            sr_adjustment,
        },
        risk_adjustment,
    }
}

/// 计算趋势贡献
fn calculate_trend_contribution(trend: &TrendState, regime: &MarketRegime) -> f64 {
    let base: f64 = match trend {
        TrendState::StrongBullish => 2.5,
        TrendState::Bullish => 1.5,
        TrendState::Neutral => 0.0,
        TrendState::Bearish => -1.5,
        TrendState::StrongBearish => -2.5,
    };
    
    // 市场状态放大/缩小
    let regime_factor = match regime {
        MarketRegime::StrongUptrend | MarketRegime::StrongDowntrend => 1.3,
        MarketRegime::ModerateUptrend | MarketRegime::ModerateDowntrend => 1.1,
        MarketRegime::Ranging => 0.5,
        _ => 0.8,
    };
    
    base * regime_factor
}

/// 计算均值回归贡献
fn calculate_mean_reversion_contribution(
    vwap_deviation: f64,
    bollinger_position: f64,
    strategy: &StrategyType,
) -> f64 {
    // VWAP回归力量
    let vwap_reversion = if vwap_deviation.abs() > 3.0 {
        -vwap_deviation * 0.4  // 强回归
    } else if vwap_deviation.abs() > 1.5 {
        -vwap_deviation * 0.25  // 中等回归
    } else {
        -vwap_deviation * 0.1  // 弱回归
    };
    
    // 布林带回归力量
    // bollinger_position: -1 到 1，-1表示在下轨，1表示在上轨
    let bb_reversion = if bollinger_position.abs() > 0.8 {
        -bollinger_position * 1.5  // 强回归
    } else if bollinger_position.abs() > 0.5 {
        -bollinger_position * 0.8  // 中等回归
    } else {
        -bollinger_position * 0.3  // 弱回归
    };
    
    // 根据策略调整权重
    match strategy {
        StrategyType::MeanReversion => (vwap_reversion * 0.6 + bb_reversion * 0.4) * 1.3,
        StrategyType::TrendFollowing => (vwap_reversion * 0.4 + bb_reversion * 0.3) * 0.5,
        StrategyType::Reversal => (vwap_reversion * 0.5 + bb_reversion * 0.5) * 1.1,
    }
}

/// 计算动量贡献
fn calculate_momentum_contribution(momentum: f64, regime: &MarketRegime) -> f64 {
    // 动量持续性
    // momentum > 0 表示上涨动量
    let base = momentum * 100.0;  // 转为百分比
    
    // 趋势市场动量更重要
    let regime_weight = match regime {
        MarketRegime::StrongUptrend if momentum > 0.0 => 1.4,
        MarketRegime::StrongDowntrend if momentum < 0.0 => 1.4,
        MarketRegime::ModerateUptrend if momentum > 0.0 => 1.2,
        MarketRegime::ModerateDowntrend if momentum < 0.0 => 1.2,
        MarketRegime::Ranging => 0.6,
        _ => 0.8,
    };
    
    // 限制极端值
    (base * regime_weight * 0.3).clamp(-2.0, 2.0)
}

/// 计算波动率调整
fn calculate_volatility_adjustment(level: &VolatilityLevel, volatility: f64) -> f64 {
    match level {
        VolatilityLevel::VeryHigh => 0.6,   // 大幅降低预测
        VolatilityLevel::High => 0.8,
        VolatilityLevel::Normal => 1.0,
        VolatilityLevel::Low => 1.05,
        VolatilityLevel::VeryLow => {
            // 极低波动可能预示突破
            if volatility < 0.01 { 0.9 } else { 1.0 }
        }
    }
}

/// 计算支撑阻力调整
fn calculate_sr_adjustment(
    current_price: f64,
    sr: &SupportResistance,
    predicted_direction: f64,
) -> f64 {
    let mut adjustment = 0.0;
    
    // 检查最近的阻力位
    if let Some(&nearest_resistance) = sr.resistance_levels.iter()
        .filter(|&&r| r > current_price)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
    {
        let distance_pct = (nearest_resistance - current_price) / current_price * 100.0;
        
        // 如果预测向上且接近阻力位，减小预测
        if predicted_direction > 0.0 && distance_pct < 3.0 {
            adjustment -= (3.0 - distance_pct) * 0.3;
        }
    }
    
    // 检查最近的支撑位
    if let Some(&nearest_support) = sr.support_levels.iter()
        .filter(|&&s| s < current_price)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
    {
        let distance_pct = (current_price - nearest_support) / current_price * 100.0;
        
        // 如果预测向下且接近支撑位，减小预测
        if predicted_direction < 0.0 && distance_pct < 3.0 {
            adjustment += (3.0 - distance_pct) * 0.3;
        }
    }
    
    adjustment
}

/// 综合各预测成分
fn combine_predictions(
    trend: f64,
    mean_reversion: f64,
    momentum: f64,
    volatility_adj: f64,
    sr_adj: f64,
    strategy: &StrategyType,
) -> ((f64, f64, f64), f64) {
    // 根据策略设置权重
    let (trend_weight, mr_weight, momentum_weight) = match strategy {
        StrategyType::TrendFollowing => (0.45, 0.20, 0.35),
        StrategyType::MeanReversion => (0.20, 0.50, 0.30),
        StrategyType::Reversal => (0.25, 0.40, 0.35),
    };
    
    // 加权组合
    let raw_prediction = trend * trend_weight + mean_reversion * mr_weight + momentum * momentum_weight;
    
    // 应用波动率调整
    let volatility_adjusted = raw_prediction * volatility_adj;
    
    // 应用支撑阻力调整
    let final_prediction = volatility_adjusted + sr_adj;
    
    // 限制在合理范围（A股涨跌停）
    let clamped = final_prediction.clamp(-9.5, 9.5);
    
    ((trend_weight, mr_weight, momentum_weight), clamped)
}

/// 计算预测区间
fn calculate_prediction_range(
    expected: f64,
    volatility: f64,
    level: &VolatilityLevel,
) -> (f64, f64) {
    let base_width = volatility * 100.0 * 2.0;  // 基于波动率
    
    // 根据波动率水平调整区间宽度
    let width_multiplier = match level {
        VolatilityLevel::VeryHigh => 1.8,
        VolatilityLevel::High => 1.4,
        VolatilityLevel::Normal => 1.0,
        VolatilityLevel::Low => 0.8,
        VolatilityLevel::VeryLow => 0.6,
    };
    
    let half_width = base_width * width_multiplier;
    
    let lower = (expected - half_width).max(-9.5);
    let upper = (expected + half_width).min(9.5);
    
    (lower, upper)
}

/// 计算预测置信度
fn calculate_prediction_confidence(
    regime: &MarketRegime,
    volatility: &VolatilityLevel,
    signal_strength: f64,
) -> f64 {
    // 基础置信度
    let base = match regime {
        MarketRegime::StrongUptrend | MarketRegime::StrongDowntrend => 0.75,
        MarketRegime::ModerateUptrend | MarketRegime::ModerateDowntrend => 0.65,
        MarketRegime::Ranging => 0.50,
        _ => 0.55,
    };
    
    // 波动率影响
    let volatility_factor = match volatility {
        VolatilityLevel::VeryHigh => 0.70,
        VolatilityLevel::High => 0.85,
        VolatilityLevel::Normal => 1.0,
        VolatilityLevel::Low => 1.05,
        VolatilityLevel::VeryLow => 0.95,
    };
    
    // 信号强度影响
    let signal_factor = (0.7 + signal_strength * 0.3).min(1.1);
    
    (base * volatility_factor * signal_factor).clamp(0.30, 0.90)
}

/// 计算风险调整因子
fn calculate_risk_adjustment(volatility: &VolatilityLevel, sr_pressure: f64) -> f64 {
    let base = match volatility {
        VolatilityLevel::VeryHigh => 0.5,
        VolatilityLevel::High => 0.7,
        VolatilityLevel::Normal => 1.0,
        VolatilityLevel::Low => 1.1,
        VolatilityLevel::VeryLow => 1.0,
    };
    
    // 接近支撑阻力时降低风险
    let sr_factor = if sr_pressure > 0.5 { 0.9 } else { 1.0 };
    
    base * sr_factor
}

/// 计算基于历史的动态权重
/// 使用简单的自适应学习
pub fn calculate_adaptive_weights(
    historical_predictions: &[(f64, f64)],  // (预测, 实际)
) -> (f64, f64, f64) {
    if historical_predictions.len() < 10 {
        return (0.35, 0.35, 0.30);  // 默认权重
    }
    
    // 分析预测误差模式
    let errors: Vec<f64> = historical_predictions.iter()
        .map(|(pred, actual)| actual - pred)
        .collect();
    
    // 计算偏差
    let avg_error = errors.iter().sum::<f64>() / errors.len() as f64;
    let error_variance = errors.iter()
        .map(|e| (e - avg_error).powi(2))
        .sum::<f64>() / errors.len() as f64;
    
    // 根据误差模式调整权重
    // 如果持续高估，增加均值回归权重
    // 如果持续低估，增加趋势权重
    let trend_adj: f64 = if avg_error < -0.5 { 0.05 } else if avg_error > 0.5 { -0.05 } else { 0.0 };
    let mr_adj: f64 = if error_variance > 4.0 { 0.05 } else { -0.02 };
    
    let trend_weight = (0.35_f64 + trend_adj).clamp(0.25, 0.45);
    let mr_weight = (0.35_f64 + mr_adj).clamp(0.25, 0.45);
    let momentum_weight = 1.0_f64 - trend_weight - mr_weight;
    
    (trend_weight, mr_weight, momentum_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trend_contribution() {
        let strong_bullish = calculate_trend_contribution(
            &TrendState::StrongBullish,
            &MarketRegime::StrongUptrend,
        );
        assert!(strong_bullish > 2.0);
        
        let bearish = calculate_trend_contribution(
            &TrendState::Bearish,
            &MarketRegime::Ranging,
        );
        assert!(bearish < 0.0);
    }
}

