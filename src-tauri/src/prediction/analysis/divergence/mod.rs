//! 专业级背离检测模块
//!
//! 背离是华尔街量化交易中最重要的反转信号之一
//!
//! 背离类型：
//! - 常规背离：预示趋势反转
//! - 隐藏背离：预示趋势延续
//! - 三重背离：连续三次背离，极强信号
//!
//! 检测指标：
//! - RSI背离
//! - MACD背离
//! - OBV背离（量价背离）
//! - Williams %R背离
//! - ROC背离
//!
//! 子模块拆分：
//! - [`extremes`]：极值点查找与强度/置信度计算
//! - [`detectors`]：各指标背离检测
//! - [`checks`]：常规/隐藏背离判定
//! - [`action`]：建议生成

use serde::{Deserialize, Serialize};

mod action;
mod checks;
mod detectors;
mod extremes;

pub use detectors::{
    detect_macd_divergence, detect_obv_divergence, detect_roc_divergence, detect_rsi_divergence,
    detect_rsi_divergence_enhanced, detect_williams_divergence,
};

use action::generate_divergence_action_enhanced;

/// 背离类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DivergenceType {
    /// 常规看涨背离：价格创新低，指标未创新低
    RegularBullish,
    /// 常规看跌背离：价格创新高，指标未创新高
    RegularBearish,
    /// 隐藏看涨背离：价格未创新低，指标创新低（趋势延续）
    HiddenBullish,
    /// 隐藏看跌背离：价格未创新高，指标创新高（趋势延续）
    HiddenBearish,
}

impl DivergenceType {
    pub fn to_string(&self) -> String {
        match self {
            Self::RegularBullish => "常规底背离".to_string(),
            Self::RegularBearish => "常规顶背离".to_string(),
            Self::HiddenBullish => "隐藏底背离".to_string(),
            Self::HiddenBearish => "隐藏顶背离".to_string(),
        }
    }

    pub fn is_bullish(&self) -> bool {
        matches!(self, Self::RegularBullish | Self::HiddenBullish)
    }

    pub fn is_reversal_signal(&self) -> bool {
        matches!(self, Self::RegularBullish | Self::RegularBearish)
    }
}

/// 背离信号强度
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DivergenceStrength {
    Strong,   // 强背离
    Moderate, // 中等背离
    Weak,     // 弱背离
}

impl DivergenceStrength {
    pub fn to_score(&self) -> f64 {
        match self {
            Self::Strong => 1.0,
            Self::Moderate => 0.7,
            Self::Weak => 0.4,
        }
    }
}

/// 单个背离信号
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceSignal {
    /// 背离类型
    pub divergence_type: DivergenceType,
    /// 指标名称
    pub indicator: String,
    /// 信号强度
    pub strength: DivergenceStrength,
    /// 置信度 (0-1)
    pub confidence: f64,
    /// 价格变化幅度
    pub price_change: f64,
    /// 指标变化幅度
    pub indicator_change: f64,
    /// 背离持续周期数
    pub duration_bars: usize,
    /// 描述
    pub description: String,
}

/// 综合背离分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceAnalysis {
    /// 是否检测到有效背离
    pub has_divergence: bool,
    /// RSI背离信号
    pub rsi_divergence: Option<DivergenceSignal>,
    /// MACD背离信号
    pub macd_divergence: Option<DivergenceSignal>,
    /// OBV背离信号（量价）
    pub obv_divergence: Option<DivergenceSignal>,
    /// Williams %R背离信号
    pub williams_divergence: Option<DivergenceSignal>,
    /// ROC背离信号
    pub roc_divergence: Option<DivergenceSignal>,
    /// 综合背离得分 (-1到1，正数看涨，负数看跌)
    pub composite_score: f64,
    /// 背离数量
    pub divergence_count: usize,
    /// 主要信号方向
    pub primary_direction: String,
    /// 综合置信度
    pub overall_confidence: f64,
    /// 建议动作
    pub suggested_action: String,
    /// 是否为三重背离（极强信号）
    pub is_triple_divergence: bool,
    /// 隐藏背离数量
    pub hidden_divergence_count: usize,
}

impl Default for DivergenceAnalysis {
    fn default() -> Self {
        Self {
            has_divergence: false,
            rsi_divergence: None,
            macd_divergence: None,
            obv_divergence: None,
            williams_divergence: None,
            roc_divergence: None,
            composite_score: 0.0,
            divergence_count: 0,
            primary_direction: "中性".to_string(),
            overall_confidence: 0.0,
            suggested_action: "观望".to_string(),
            is_triple_divergence: false,
            hidden_divergence_count: 0,
        }
    }
}

/// 综合背离分析
pub fn analyze_all_divergences(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    volumes: &[i64],
) -> DivergenceAnalysis {
    let len = prices.len();
    if len < 30 {
        return DivergenceAnalysis::default();
    }

    // 检测RSI背离（包括隐藏背离）
    let rsi_divergence = detect_rsi_divergence_enhanced(prices);

    // 检测MACD背离
    let macd_divergence = detect_macd_divergence(prices);

    // 检测OBV（量价）背离
    let obv_divergence = detect_obv_divergence(prices, volumes);

    // 检测Williams %R背离
    let williams_divergence = detect_williams_divergence(highs, lows, prices);

    // 检测ROC背离
    let roc_divergence = detect_roc_divergence(prices);

    // 汇总背离信号
    let mut bullish_score = 0.0;
    let mut bearish_score = 0.0;
    let mut divergence_count = 0;
    let mut total_confidence = 0.0;
    let mut hidden_divergence_count = 0;

    // 记录各个背离的方向用于检测三重背离
    let mut divergence_directions: Vec<bool> = Vec::new(); // true = bullish

    if let Some(ref div) = rsi_divergence {
        divergence_count += 1;
        total_confidence += div.confidence;
        divergence_directions.push(div.divergence_type.is_bullish());
        if !div.divergence_type.is_reversal_signal() {
            hidden_divergence_count += 1;
        }
        if div.divergence_type.is_bullish() {
            bullish_score += div.strength.to_score() * div.confidence * 1.2;
        } else {
            bearish_score += div.strength.to_score() * div.confidence * 1.2;
        }
    }

    if let Some(ref div) = macd_divergence {
        divergence_count += 1;
        total_confidence += div.confidence;
        divergence_directions.push(div.divergence_type.is_bullish());
        if !div.divergence_type.is_reversal_signal() {
            hidden_divergence_count += 1;
        }
        if div.divergence_type.is_bullish() {
            bullish_score += div.strength.to_score() * div.confidence * 1.0;
        } else {
            bearish_score += div.strength.to_score() * div.confidence * 1.0;
        }
    }

    if let Some(ref div) = obv_divergence {
        divergence_count += 1;
        total_confidence += div.confidence;
        divergence_directions.push(div.divergence_type.is_bullish());
        if div.divergence_type.is_bullish() {
            bullish_score += div.strength.to_score() * div.confidence * 0.8;
        } else {
            bearish_score += div.strength.to_score() * div.confidence * 0.8;
        }
    }

    if let Some(ref div) = williams_divergence {
        divergence_count += 1;
        total_confidence += div.confidence;
        divergence_directions.push(div.divergence_type.is_bullish());
        if !div.divergence_type.is_reversal_signal() {
            hidden_divergence_count += 1;
        }
        if div.divergence_type.is_bullish() {
            bullish_score += div.strength.to_score() * div.confidence * 0.7;
        } else {
            bearish_score += div.strength.to_score() * div.confidence * 0.7;
        }
    }

    if let Some(ref div) = roc_divergence {
        divergence_count += 1;
        total_confidence += div.confidence;
        divergence_directions.push(div.divergence_type.is_bullish());
        if div.divergence_type.is_bullish() {
            bullish_score += div.strength.to_score() * div.confidence * 0.6;
        } else {
            bearish_score += div.strength.to_score() * div.confidence * 0.6;
        }
    }

    // 检测三重背离（3个或以上同向背离）
    let bullish_div_count = divergence_directions.iter().filter(|&&d| d).count();
    let bearish_div_count = divergence_directions.iter().filter(|&&d| !d).count();
    let is_triple_divergence = bullish_div_count >= 3 || bearish_div_count >= 3;

    // 三重背离加成
    if is_triple_divergence {
        if bullish_div_count >= 3 {
            bullish_score *= 1.5;
        } else {
            bearish_score *= 1.5;
        }
    }

    // 计算综合得分
    let composite_score = if bullish_score > 0.0 || bearish_score > 0.0 {
        (bullish_score - bearish_score) / (bullish_score + bearish_score).max(1.0)
    } else {
        0.0
    };

    // 计算综合置信度
    let overall_confidence = if divergence_count > 0 {
        let base_conf = total_confidence / divergence_count as f64;
        // 三重背离提升置信度
        if is_triple_divergence {
            (base_conf * 1.2).min(0.95)
        } else {
            base_conf
        }
    } else {
        0.0
    };

    // 确定主要方向
    let primary_direction = if composite_score > 0.3 {
        "看涨".to_string()
    } else if composite_score < -0.3 {
        "看跌".to_string()
    } else {
        "中性".to_string()
    };

    // 生成建议
    let suggested_action = generate_divergence_action_enhanced(
        composite_score,
        divergence_count,
        overall_confidence,
        is_triple_divergence,
        hidden_divergence_count,
    );

    DivergenceAnalysis {
        has_divergence: divergence_count > 0,
        rsi_divergence,
        macd_divergence,
        obv_divergence,
        williams_divergence,
        roc_divergence,
        composite_score,
        divergence_count,
        primary_direction,
        overall_confidence,
        suggested_action,
        is_triple_divergence,
        hidden_divergence_count,
    }
}
