//! 增强信号确认系统
//!
//! 核心功能：
//! 1. 多信号冲突检测与调和
//! 2. 动态信号权重调整
//! 3. 信号一致性评估
//! 4. 假信号过滤
//!
//! 子模块拆分：
//! - [`sources`]：收集各技术指标信号
//! - [`conflict`]：冲突检测与假信号过滤
//! - [`weights`]：动态权重
//! - [`combination`]：信号综合、一致性、确认级别与建议

use crate::prediction::analysis::market_regime::{MarketRegime, VolatilityLevel};
use crate::prediction::analysis::{TrendState, VolumePriceSignal};
use crate::prediction::indicators::TechnicalIndicatorValues;
use serde::{Deserialize, Serialize};

mod combination;
mod conflict;
mod sources;
mod weights;

use combination::{
    calculate_combined_signal, calculate_consistency, determine_confirmation_level,
    find_dominant_signal, generate_action_suggestion,
};
use conflict::{check_false_signal, detect_conflicts};
use sources::collect_all_signals;
use weights::apply_dynamic_weights;

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
    Trend,      // 趋势信号
    Momentum,   // 动量信号
    Oscillator, // 震荡指标信号
    Volume,     // 量能信号
    Pattern,    // 形态信号
    Sentiment,  // 情绪信号
}

impl SignalType {
    /// 获取信号类型在不同市场状态下的基础权重
    pub fn base_weight(&self, regime: &MarketRegime) -> f64 {
        match regime {
            MarketRegime::StrongUptrend | MarketRegime::StrongDowntrend => match self {
                Self::Trend => 0.30,
                Self::Momentum => 0.25,
                Self::Oscillator => 0.15,
                Self::Volume => 0.20,
                Self::Pattern => 0.05,
                Self::Sentiment => 0.05,
            },
            MarketRegime::ModerateUptrend | MarketRegime::ModerateDowntrend => match self {
                Self::Trend => 0.25,
                Self::Momentum => 0.22,
                Self::Oscillator => 0.18,
                Self::Volume => 0.20,
                Self::Pattern => 0.08,
                Self::Sentiment => 0.07,
            },
            MarketRegime::Ranging => match self {
                Self::Trend => 0.10,
                Self::Momentum => 0.15,
                Self::Oscillator => 0.28,
                Self::Volume => 0.17,
                Self::Pattern => 0.18,
                Self::Sentiment => 0.12,
            },
            MarketRegime::PotentialTop | MarketRegime::PotentialBottom => match self {
                Self::Trend => 0.12,
                Self::Momentum => 0.18,
                Self::Oscillator => 0.22,
                Self::Volume => 0.20,
                Self::Pattern => 0.15,
                Self::Sentiment => 0.13,
            },
        }
    }
}

/// 信号冲突类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConflictType {
    NoConflict,       // 无冲突
    MinorConflict,    // 轻微冲突
    ModerateConflict, // 中等冲突
    SevereConflict,   // 严重冲突
    CompleteConflict, // 完全对立
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
    let confirmation_level =
        determine_confirmation_level(&signals, consistency_score, &conflict_type);

    // 7. 检测假信号
    let is_potential_false_signal = check_false_signal(&signals, consistency_score, &conflict_type);

    // 8. 找出主导信号
    let dominant_signal = find_dominant_signal(&signals);

    // 9. 计算置信度因子
    let confidence_factor =
        conflict_type.confidence_discount() * consistency_score * confirmation_level.to_weight();

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
