//! 市场状态分类器
//! 
//! 基于华尔街量化策略，自动识别市场处于何种状态：
//! - 趋势市（上升/下降）
//! - 震荡市
//! - 转折点
//! 
//! 不同市场状态需要使用不同的预测策略

use serde::{Deserialize, Serialize};
use crate::prediction::indicators::{bollinger, macd, rsi};

/// 市场状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    /// 强势上涨趋势
    StrongUptrend,
    /// 温和上涨趋势
    ModerateUptrend,
    /// 震荡整理
    Ranging,
    /// 温和下跌趋势
    ModerateDowntrend,
    /// 强势下跌趋势
    StrongDowntrend,
    /// 潜在顶部转折
    PotentialTop,
    /// 潜在底部转折
    PotentialBottom,
}

impl MarketRegime {
    pub fn to_string(&self) -> String {
        match self {
            Self::StrongUptrend => "强势上涨".to_string(),
            Self::ModerateUptrend => "温和上涨".to_string(),
            Self::Ranging => "震荡整理".to_string(),
            Self::ModerateDowntrend => "温和下跌".to_string(),
            Self::StrongDowntrend => "强势下跌".to_string(),
            Self::PotentialTop => "潜在顶部".to_string(),
            Self::PotentialBottom => "潜在底部".to_string(),
        }
    }
    
    /// 是否应使用趋势跟踪策略
    pub fn is_trending(&self) -> bool {
        matches!(self, 
            Self::StrongUptrend | Self::ModerateUptrend |
            Self::StrongDowntrend | Self::ModerateDowntrend
        )
    }
    
    /// 是否看涨环境
    pub fn is_bullish(&self) -> bool {
        matches!(self, Self::StrongUptrend | Self::ModerateUptrend | Self::PotentialBottom)
    }
    
    /// 是否看跌环境
    pub fn is_bearish(&self) -> bool {
        matches!(self, Self::StrongDowntrend | Self::ModerateDowntrend | Self::PotentialTop)
    }
    
    /// 获取推荐的策略类型
    pub fn recommended_strategy(&self) -> StrategyType {
        match self {
            Self::StrongUptrend | Self::ModerateUptrend => StrategyType::TrendFollowing,
            Self::StrongDowntrend | Self::ModerateDowntrend => StrategyType::TrendFollowing,
            Self::Ranging => StrategyType::MeanReversion,
            Self::PotentialTop | Self::PotentialBottom => StrategyType::Reversal,
        }
    }
}

/// 策略类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyType {
    /// 趋势跟踪
    TrendFollowing,
    /// 均值回归
    MeanReversion,
    /// 反转策略
    Reversal,
}

/// 市场状态分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRegimeAnalysis {
    /// 当前市场状态
    pub regime: MarketRegime,
    /// 状态置信度 (0-1)
    pub confidence: f64,
    /// 趋势强度 (-1 到 1，负数表示下跌)
    pub trend_strength: f64,
    /// 波动率水平
    pub volatility_level: VolatilityLevel,
    /// 波动率百分位 (0-100)
    pub volatility_percentile: f64,
    /// ADX趋势强度指标
    pub adx_value: f64,
    /// 布林带宽度 (波动率指标)
    pub bollinger_width: f64,
    /// 市场阶段描述
    pub description: String,
    /// 推荐策略
    pub recommended_strategy: StrategyType,
    /// 各项指标评分
    pub indicator_scores: RegimeIndicatorScores,
}

/// 波动率水平
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolatilityLevel {
    VeryLow,
    Low,
    Normal,
    High,
    VeryHigh,
}

impl VolatilityLevel {
    pub fn to_string(&self) -> String {
        match self {
            Self::VeryLow => "极低".to_string(),
            Self::Low => "低".to_string(),
            Self::Normal => "正常".to_string(),
            Self::High => "高".to_string(),
            Self::VeryHigh => "极高".to_string(),
        }
    }
    
    /// 获取波动率调整系数
    pub fn adjustment_factor(&self) -> f64 {
        match self {
            Self::VeryLow => 0.6,
            Self::Low => 0.8,
            Self::Normal => 1.0,
            Self::High => 1.3,
            Self::VeryHigh => 1.6,
        }
    }
}

/// 状态分类指标评分
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeIndicatorScores {
    /// 均线排列得分 (-1 到 1)
    pub ma_alignment_score: f64,
    /// ADX趋势得分 (0 到 1)
    pub adx_score: f64,
    /// 价格位置得分 (相对于布林带)
    pub price_position_score: f64,
    /// 动量得分 (-1 到 1)
    pub momentum_score: f64,
    /// 波动率收敛得分 (0 到 1，用于检测突破)
    pub volatility_contraction_score: f64,
}

/// 分析市场状态
pub fn classify_market_regime(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
) -> MarketRegimeAnalysis {
    let len = prices.len();
    
    if len < 60 {
        return default_regime_analysis();
    }
    
    // 1. 计算均线系统
    let ma5 = calculate_ma(prices, 5);
    let ma10 = calculate_ma(prices, 10);
    let ma20 = calculate_ma(prices, 20);
    let ma60 = calculate_ma(prices, 60);
    
    // 2. 计算均线排列得分
    let ma_alignment_score = calculate_ma_alignment_score(
        prices.last().copied().unwrap_or(0.0),
        ma5, ma10, ma20, ma60
    );
    
    // 3. 计算ADX（趋势强度）
    let adx_value = calculate_adx(highs, lows, prices, 14);
    let adx_score = (adx_value / 50.0).min(1.0);
    
    // 4. 计算布林带位置
    let bb = bollinger::calculate_bollinger_bands(prices, 20, 2.0);
    let current_price = *prices.last().unwrap();
    let price_position_score = (current_price - bb.middle) / (bb.upper - bb.lower).max(0.001);
    
    // 5. 计算布林带宽度（波动率指标）
    let bollinger_width = (bb.upper - bb.lower) / bb.middle.max(0.001);
    
    // 6. 计算动量得分
    let momentum_score = calculate_momentum_score(prices);
    
    // 7. 计算波动率及其百分位
    let current_volatility = calculate_volatility(prices, 20);
    let volatility_percentile = calculate_volatility_percentile(prices, current_volatility);
    let volatility_level = classify_volatility_level(volatility_percentile);
    
    // 8. 波动率收敛检测（用于预测突破）
    let volatility_contraction_score = calculate_volatility_contraction(prices, 20);
    
    // 9. 检测潜在转折点
    let (is_potential_top, is_potential_bottom) = detect_turning_points(
        prices, highs, lows, &ma_alignment_score, momentum_score
    );
    
    // 10. 综合判断市场状态
    let (regime, confidence) = determine_regime(
        ma_alignment_score,
        adx_value,
        momentum_score,
        is_potential_top,
        is_potential_bottom,
        volatility_level,
    );
    
    // 11. 计算趋势强度
    let trend_strength = calculate_trend_strength_value(
        ma_alignment_score, adx_value, momentum_score, &regime
    );
    
    // 12. 生成描述
    let description = generate_regime_description(&regime, adx_value, volatility_level, ma_alignment_score);
    
    let indicator_scores = RegimeIndicatorScores {
        ma_alignment_score,
        adx_score,
        price_position_score,
        momentum_score,
        volatility_contraction_score,
    };
    
    MarketRegimeAnalysis {
        regime,
        confidence,
        trend_strength,
        volatility_level,
        volatility_percentile,
        adx_value,
        bollinger_width,
        description,
        recommended_strategy: regime.recommended_strategy(),
        indicator_scores,
    }
}

/// 计算简单移动平均
fn calculate_ma(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period {
        return prices.last().copied().unwrap_or(0.0);
    }
    let start = prices.len() - period;
    prices[start..].iter().sum::<f64>() / period as f64
}

/// 计算均线排列得分
fn calculate_ma_alignment_score(
    current_price: f64,
    ma5: f64,
    ma10: f64,
    ma20: f64,
    ma60: f64,
) -> f64 {
    let mut score: f64 = 0.0;
    
    // 价格与MA5的关系
    if current_price > ma5 {
        score += 0.15;
    } else {
        score -= 0.15;
    }
    
    // MA5与MA10的关系
    if ma5 > ma10 {
        score += 0.20;
    } else {
        score -= 0.20;
    }
    
    // MA10与MA20的关系
    if ma10 > ma20 {
        score += 0.25;
    } else {
        score -= 0.25;
    }
    
    // MA20与MA60的关系
    if ma20 > ma60 {
        score += 0.30;
    } else {
        score -= 0.30;
    }
    
    // 完美多头/空头排列额外加分
    if current_price > ma5 && ma5 > ma10 && ma10 > ma20 && ma20 > ma60 {
        score += 0.10;
    } else if current_price < ma5 && ma5 < ma10 && ma10 < ma20 && ma20 < ma60 {
        score -= 0.10;
    }
    
    score.clamp(-1.0, 1.0)
}

/// 计算ADX（简化版本）
fn calculate_adx(highs: &[f64], lows: &[f64], prices: &[f64], period: usize) -> f64 {
    if highs.len() < period + 1 || lows.len() < period + 1 || prices.len() < period + 1 {
        return 25.0; // 默认中性值
    }
    
    let len = prices.len();
    let mut plus_dm_sum = 0.0;
    let mut minus_dm_sum = 0.0;
    let mut tr_sum = 0.0;
    
    for i in (len - period)..len {
        if i == 0 {
            continue;
        }
        
        let high_diff = highs[i] - highs[i - 1];
        let low_diff = lows[i - 1] - lows[i];
        
        // +DM
        if high_diff > low_diff && high_diff > 0.0 {
            plus_dm_sum += high_diff;
        }
        // -DM
        if low_diff > high_diff && low_diff > 0.0 {
            minus_dm_sum += low_diff;
        }
        
        // True Range
        let tr = (highs[i] - lows[i])
            .max((highs[i] - prices[i - 1]).abs())
            .max((lows[i] - prices[i - 1]).abs());
        tr_sum += tr;
    }
    
    if tr_sum == 0.0 {
        return 25.0;
    }
    
    let plus_di = (plus_dm_sum / tr_sum) * 100.0;
    let minus_di = (minus_dm_sum / tr_sum) * 100.0;
    
    let di_sum = plus_di + minus_di;
    if di_sum == 0.0 {
        return 25.0;
    }
    
    let dx = ((plus_di - minus_di).abs() / di_sum) * 100.0;
    
    // 返回DX作为ADX的近似值
    dx.clamp(0.0, 100.0)
}

/// 计算动量得分
fn calculate_momentum_score(prices: &[f64]) -> f64 {
    let len = prices.len();
    if len < 20 {
        return 0.0;
    }
    
    let current = *prices.last().unwrap();
    
    // 短期动量（5日）
    let momentum_5d = (current - prices[len - 5]) / prices[len - 5];
    
    // 中期动量（10日）
    let momentum_10d = (current - prices[len - 10]) / prices[len - 10];
    
    // 长期动量（20日）
    let momentum_20d = (current - prices[len - 20]) / prices[len - 20];
    
    // RSI动量
    let rsi = rsi::calculate_rsi(&prices[len.saturating_sub(15)..]);
    let rsi_score = (rsi - 50.0) / 50.0;
    
    // 综合动量得分
    let score = momentum_5d * 0.30 + momentum_10d * 0.25 + momentum_20d * 0.20 + rsi_score * 0.25;
    
    score.clamp(-1.0, 1.0)
}

/// 计算波动率
fn calculate_volatility(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period + 1 {
        return 0.02;
    }
    
    let start = prices.len() - period;
    let mut returns = Vec::new();
    
    for i in (start + 1)..prices.len() {
        let ret = (prices[i] - prices[i - 1]) / prices[i - 1];
        returns.push(ret);
    }
    
    if returns.is_empty() {
        return 0.02;
    }
    
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;
    
    variance.sqrt()
}

/// 计算波动率百分位
fn calculate_volatility_percentile(prices: &[f64], current_volatility: f64) -> f64 {
    let len = prices.len();
    if len < 120 {
        return 50.0;
    }
    
    // 计算历史波动率分布
    let mut historical_vols = Vec::new();
    for i in 60..len {
        let vol = calculate_volatility(&prices[..i], 20);
        historical_vols.push(vol);
    }
    
    if historical_vols.is_empty() {
        return 50.0;
    }
    
    // 计算百分位
    let below_count = historical_vols.iter()
        .filter(|&&v| v < current_volatility)
        .count();
    
    (below_count as f64 / historical_vols.len() as f64) * 100.0
}

/// 分类波动率水平
fn classify_volatility_level(percentile: f64) -> VolatilityLevel {
    if percentile < 15.0 {
        VolatilityLevel::VeryLow
    } else if percentile < 35.0 {
        VolatilityLevel::Low
    } else if percentile < 65.0 {
        VolatilityLevel::Normal
    } else if percentile < 85.0 {
        VolatilityLevel::High
    } else {
        VolatilityLevel::VeryHigh
    }
}

/// 计算波动率收敛程度（用于预测突破）
fn calculate_volatility_contraction(prices: &[f64], period: usize) -> f64 {
    let len = prices.len();
    if len < period * 2 {
        return 0.5;
    }
    
    let current_vol = calculate_volatility(prices, period);
    let prev_vol = calculate_volatility(&prices[..len - period], period);
    
    if prev_vol == 0.0 {
        return 0.5;
    }
    
    let contraction_ratio = 1.0 - (current_vol / prev_vol);
    contraction_ratio.clamp(0.0, 1.0)
}

/// 检测潜在转折点
fn detect_turning_points(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    ma_alignment: &f64,
    momentum: f64,
) -> (bool, bool) {
    let len = prices.len();
    if len < 30 {
        return (false, false);
    }
    
    let current = *prices.last().unwrap();
    
    // 计算RSI
    let rsi = rsi::calculate_rsi(&prices[len.saturating_sub(15)..]);
    
    // 计算MACD
    let (dif, dea, hist) = macd::calculate_macd_full(prices);
    let prev_hist = if len > 27 {
        let (_, _, h) = macd::calculate_macd_full(&prices[..len - 1]);
        h
    } else {
        hist
    };
    
    // 近期高低点
    let recent_high = highs[len.saturating_sub(10)..].iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let recent_low = lows[len.saturating_sub(10)..].iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));
    
    // 潜在顶部信号
    let is_potential_top = 
        (rsi > 70.0 && momentum > 0.3 && *ma_alignment > 0.6) ||  // RSI超买 + 强势环境
        (current > recent_high * 0.98 && hist < prev_hist && hist > 0.0) ||  // 接近高点 + MACD柱缩小
        (rsi > 75.0 && dif < dea);  // 极度超买 + MACD死叉
    
    // 潜在底部信号
    let is_potential_bottom =
        (rsi < 30.0 && momentum < -0.3 && *ma_alignment < -0.6) ||  // RSI超卖 + 弱势环境
        (current < recent_low * 1.02 && hist > prev_hist && hist < 0.0) ||  // 接近低点 + MACD柱缩小
        (rsi < 25.0 && dif > dea);  // 极度超卖 + MACD金叉
    
    (is_potential_top, is_potential_bottom)
}

/// 综合判断市场状态
fn determine_regime(
    ma_alignment: f64,
    adx: f64,
    momentum: f64,
    is_potential_top: bool,
    is_potential_bottom: bool,
    volatility_level: VolatilityLevel,
) -> (MarketRegime, f64) {
    // 优先检测转折点
    if is_potential_top && adx > 25.0 {
        return (MarketRegime::PotentialTop, 0.70);
    }
    if is_potential_bottom && adx > 25.0 {
        return (MarketRegime::PotentialBottom, 0.70);
    }
    
    // 趋势强度判断
    let is_strong_trend = adx > 35.0;
    let is_moderate_trend = adx > 20.0;
    
    // 方向判断
    let is_bullish = ma_alignment > 0.3 && momentum > 0.1;
    let is_strongly_bullish = ma_alignment > 0.6 && momentum > 0.2;
    let is_bearish = ma_alignment < -0.3 && momentum < -0.1;
    let is_strongly_bearish = ma_alignment < -0.6 && momentum < -0.2;
    
    // 状态分类
    let (regime, base_confidence) = if is_strongly_bullish && is_strong_trend {
        (MarketRegime::StrongUptrend, 0.85)
    } else if is_bullish && is_moderate_trend {
        (MarketRegime::ModerateUptrend, 0.75)
    } else if is_strongly_bearish && is_strong_trend {
        (MarketRegime::StrongDowntrend, 0.85)
    } else if is_bearish && is_moderate_trend {
        (MarketRegime::ModerateDowntrend, 0.75)
    } else {
        (MarketRegime::Ranging, 0.65)
    };
    
    // 根据波动率调整置信度
    let volatility_factor: f64 = match volatility_level {
        VolatilityLevel::VeryHigh => 0.85,
        VolatilityLevel::High => 0.90,
        VolatilityLevel::Normal => 1.0,
        VolatilityLevel::Low => 1.05,
        VolatilityLevel::VeryLow => 0.95,  // 极低波动率可能预示突破
    };
    
    let confidence = (base_confidence * volatility_factor).clamp(0.3_f64, 0.95_f64);
    
    (regime, confidence)
}

/// 计算趋势强度值
fn calculate_trend_strength_value(
    ma_alignment: f64,
    adx: f64,
    momentum: f64,
    regime: &MarketRegime,
) -> f64 {
    let direction = if regime.is_bullish() { 1.0 } else if regime.is_bearish() { -1.0 } else { 0.0 };
    
    let strength = (ma_alignment.abs() * 0.40 + 
                   (adx / 50.0).min(1.0) * 0.35 + 
                   momentum.abs() * 0.25).min(1.0);
    
    direction * strength
}

/// 生成状态描述
fn generate_regime_description(
    regime: &MarketRegime,
    adx: f64,
    volatility: VolatilityLevel,
    ma_alignment: f64,
) -> String {
    let trend_desc = if adx > 35.0 {
        "强趋势"
    } else if adx > 20.0 {
        "中等趋势"
    } else {
        "弱趋势/震荡"
    };
    
    let ma_desc = if ma_alignment > 0.6 {
        "均线多头排列"
    } else if ma_alignment > 0.2 {
        "均线偏多"
    } else if ma_alignment < -0.6 {
        "均线空头排列"
    } else if ma_alignment < -0.2 {
        "均线偏空"
    } else {
        "均线粘合"
    };
    
    format!(
        "{} | {} | ADX={:.1} | 波动率:{} | {}",
        regime.to_string(),
        trend_desc,
        adx,
        volatility.to_string(),
        ma_desc
    )
}

/// 默认状态分析（数据不足时）
fn default_regime_analysis() -> MarketRegimeAnalysis {
    MarketRegimeAnalysis {
        regime: MarketRegime::Ranging,
        confidence: 0.30,
        trend_strength: 0.0,
        volatility_level: VolatilityLevel::Normal,
        volatility_percentile: 50.0,
        adx_value: 25.0,
        bollinger_width: 0.04,
        description: "数据不足，无法准确判断市场状态".to_string(),
        recommended_strategy: StrategyType::MeanReversion,
        indicator_scores: RegimeIndicatorScores {
            ma_alignment_score: 0.0,
            adx_score: 0.5,
            price_position_score: 0.0,
            momentum_score: 0.0,
            volatility_contraction_score: 0.5,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ma_alignment() {
        let score = calculate_ma_alignment_score(100.0, 99.0, 98.0, 97.0, 95.0);
        assert!(score > 0.8, "Perfect uptrend should have high score");
        
        let score = calculate_ma_alignment_score(100.0, 101.0, 102.0, 103.0, 105.0);
        assert!(score < -0.8, "Perfect downtrend should have low score");
    }
}

