//! 市场状态分类器
//!
//! 基于华尔街量化策略，自动识别市场处于何种状态：
//! - 趋势市（上升/下降）
//! - 震荡市
//! - 转折点
//!
//! 不同市场状态需要使用不同的预测策略
//!
//! 子模块拆分：
//! - [`indicators`]：均线/ADX/动量等指标
//! - [`volatility`]：波动率及其百分位、收敛
//! - [`classifier`]：转折点检测与状态判定

use crate::prediction::indicators::bollinger;
use serde::{Deserialize, Serialize};

mod classifier;
mod indicators;
mod volatility;

use classifier::{
    calculate_trend_strength_value, detect_turning_points, determine_regime,
    generate_regime_description,
};
use indicators::{calculate_adx, calculate_ma, calculate_ma_alignment_score, calculate_momentum_score};
use volatility::{
    calculate_volatility, calculate_volatility_contraction, calculate_volatility_percentile,
    classify_volatility_level,
};

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
        matches!(
            self,
            Self::StrongUptrend
                | Self::ModerateUptrend
                | Self::StrongDowntrend
                | Self::ModerateDowntrend
        )
    }

    /// 是否看涨环境
    pub fn is_bullish(&self) -> bool {
        matches!(
            self,
            Self::StrongUptrend | Self::ModerateUptrend | Self::PotentialBottom
        )
    }

    /// 是否看跌环境
    pub fn is_bearish(&self) -> bool {
        matches!(
            self,
            Self::StrongDowntrend | Self::ModerateDowntrend | Self::PotentialTop
        )
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
pub fn classify_market_regime(prices: &[f64], highs: &[f64], lows: &[f64]) -> MarketRegimeAnalysis {
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
    let ma_alignment_score =
        calculate_ma_alignment_score(prices.last().copied().unwrap_or(0.0), ma5, ma10, ma20, ma60);

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
    let (is_potential_top, is_potential_bottom) =
        detect_turning_points(prices, highs, lows, &ma_alignment_score, momentum_score);

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
    let trend_strength =
        calculate_trend_strength_value(ma_alignment_score, adx_value, momentum_score, &regime);

    // 12. 生成描述
    let description =
        generate_regime_description(&regime, adx_value, volatility_level, ma_alignment_score);

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
