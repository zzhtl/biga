//! 专业级预测引擎
//!
//! 基于华尔街顶级量化策略设计，核心理念：
//! 1. 多维度信号验证 - 多个独立信号同向确认才发出信号
//! 2. 市场状态自适应 - 趋势市用趋势跟踪，震荡市用均值回归
//! 3. 动态风险管理 - 根据波动率调整预测范围
//! 4. 概率思维 - 输出置信区间而非单一点预测
//! 5. 信号强度分级 - 只执行高置信度信号
//!
//! 子模块拆分：
//! - [`signals`]：信号收集
//! - [`direction`]：信号确认级别、预测方向、综合置信度
//! - [`change`]：预期涨跌幅与 A 股涨跌停限制
//! - [`risk`]：风险评估
//! - [`output`]：关键因素与操作建议

use crate::prediction::analysis::{
    divergence::DivergenceAnalysis,
    market_regime::{MarketRegime, MarketRegimeAnalysis, StrategyType},
    PatternRecognition, SupportResistance, TrendAnalysis, VolumePriceSignal,
};
use crate::prediction::indicators::TechnicalIndicatorValues;
use crate::prediction::strategy::multi_factor::MultiFactorScore;
use serde::{Deserialize, Serialize};

mod change;
mod direction;
mod output;
mod risk;
mod signals;

use change::calculate_expected_change;
use direction::{
    calculate_comprehensive_confidence, calculate_signal_confirmation,
    determine_prediction_direction,
};
use output::{generate_key_factors, generate_suggested_action};
use risk::assess_risk;
use signals::collect_all_signals;

// =============================================================================
// 核心类型定义
// =============================================================================

/// 信号确认级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalConfirmation {
    /// 强确认：3+个独立信号同向
    StrongConfirm,
    /// 中等确认：2个独立信号同向
    ModerateConfirm,
    /// 弱确认：仅1个信号
    WeakConfirm,
    /// 无确认/信号冲突
    NoConfirm,
}

impl SignalConfirmation {
    pub fn to_weight(&self) -> f64 {
        match self {
            Self::StrongConfirm => 1.0,
            Self::ModerateConfirm => 0.75,
            Self::WeakConfirm => 0.45,
            Self::NoConfirm => 0.25,
        }
    }
}

/// 预测方向
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionDirection {
    StrongBullish,
    Bullish,
    Neutral,
    Bearish,
    StrongBearish,
}

impl PredictionDirection {
    pub fn to_string(&self) -> String {
        match self {
            Self::StrongBullish => "强烈看涨".to_string(),
            Self::Bullish => "看涨".to_string(),
            Self::Neutral => "中性".to_string(),
            Self::Bearish => "看跌".to_string(),
            Self::StrongBearish => "强烈看跌".to_string(),
        }
    }

    pub fn to_bias(&self) -> f64 {
        match self {
            Self::StrongBullish => 1.0,
            Self::Bullish => 0.5,
            Self::Neutral => 0.0,
            Self::Bearish => -0.5,
            Self::StrongBearish => -1.0,
        }
    }
}

/// 专业预测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfessionalPredictionResult {
    /// 预测方向
    pub direction: PredictionDirection,
    /// 预期涨跌幅（中值）
    pub expected_change: f64,
    /// 预测区间（低, 高）
    pub prediction_range: (f64, f64),
    /// 综合置信度 (0-1)
    pub confidence: f64,
    /// 信号确认级别
    pub signal_confirmation: SignalConfirmation,
    /// 市场状态
    pub market_regime: MarketRegime,
    /// 使用的策略类型
    pub strategy_used: StrategyType,
    /// 信号汇总
    pub signal_summary: SignalSummary,
    /// 风险评估
    pub risk_assessment: RiskAssessment,
    /// 关键因素
    pub key_factors: Vec<String>,
    /// 建议操作
    pub suggested_action: String,
}

/// 信号汇总
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalSummary {
    /// 看涨信号数量
    pub bullish_signals: i32,
    /// 看跌信号数量
    pub bearish_signals: i32,
    /// 信号详情
    pub signal_details: Vec<SignalDetail>,
    /// 净信号得分 (-1到1)
    pub net_signal_score: f64,
}

/// 单个信号详情
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalDetail {
    pub source: String,
    pub direction: String,
    pub strength: f64,
    pub description: String,
}

/// 风险评估
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// 风险等级
    pub risk_level: String,
    /// 波动率风险
    pub volatility_risk: f64,
    /// 支撑位距离
    pub support_distance: f64,
    /// 阻力位距离
    pub resistance_distance: f64,
    /// 建议止损幅度
    pub suggested_stop_loss: f64,
    /// 建议止盈幅度
    pub suggested_take_profit: f64,
    /// 风险收益比
    pub risk_reward_ratio: f64,
}

// =============================================================================
// 预测引擎上下文
// =============================================================================

/// 预测引擎上下文（汇聚所有分析结果）
pub struct PredictionContext {
    pub current_price: f64,
    pub market_regime: MarketRegimeAnalysis,
    pub trend_analysis: TrendAnalysis,
    pub volume_signal: VolumePriceSignal,
    pub divergence: DivergenceAnalysis,
    pub indicators: TechnicalIndicatorValues,
    pub patterns: Vec<PatternRecognition>,
    pub support_resistance: SupportResistance,
    pub multi_factor_score: MultiFactorScore,
    pub volatility: f64,
}

// =============================================================================
// A股涨跌停限制规则
// =============================================================================

/// A股涨跌停限制常量
pub mod a_share_limits {
    /// 主板涨停限制
    pub const MAIN_BOARD_LIMIT_UP: f64 = 10.0;
    /// 主板跌停限制
    pub const MAIN_BOARD_LIMIT_DOWN: f64 = -10.0;
    /// ST股涨停限制
    pub const ST_LIMIT_UP: f64 = 5.0;
    /// ST股跌停限制
    pub const ST_LIMIT_DOWN: f64 = -5.0;
    /// 科创板/创业板涨停限制
    pub const KC_CY_LIMIT_UP: f64 = 20.0;
    /// 科创板/创业板跌停限制
    pub const KC_CY_LIMIT_DOWN: f64 = -20.0;
    /// 默认预测限制（保守值，主板留裕度）
    pub const DEFAULT_LIMIT_UP: f64 = 9.5;
    pub const DEFAULT_LIMIT_DOWN: f64 = -9.5;
}

/// 根据股票代码判断市场类型并返回对应的涨跌停限制
pub fn get_stock_price_limits(stock_code: Option<&str>) -> (f64, f64) {
    match stock_code {
        Some(code) => {
            // 科创板：688开头
            if code.starts_with("688") {
                (a_share_limits::KC_CY_LIMIT_DOWN, a_share_limits::KC_CY_LIMIT_UP)
            }
            // 创业板：300开头
            else if code.starts_with("300") || code.starts_with("301") {
                (a_share_limits::KC_CY_LIMIT_DOWN, a_share_limits::KC_CY_LIMIT_UP)
            }
            // ST股：名称中包含ST（这里简化处理，实际应查询数据库）
            // 暂时无法判断，使用主板规则
            else {
                // 主板：60/00开头
                (a_share_limits::DEFAULT_LIMIT_DOWN, a_share_limits::DEFAULT_LIMIT_UP)
            }
        }
        None => {
            // 未知股票，使用保守的主板规则
            (a_share_limits::DEFAULT_LIMIT_DOWN, a_share_limits::DEFAULT_LIMIT_UP)
        }
    }
}

// =============================================================================
// 核心预测函数
// =============================================================================

/// 执行专业级预测
pub fn execute_professional_prediction(ctx: &PredictionContext) -> ProfessionalPredictionResult {
    // 1. 收集所有信号
    let signal_summary = collect_all_signals(ctx);

    // 2. 计算信号确认级别
    let signal_confirmation = calculate_signal_confirmation(&signal_summary);

    // 3. 确定预测方向
    let direction =
        determine_prediction_direction(&signal_summary, &ctx.market_regime, &ctx.divergence);

    // 4. 根据市场状态选择策略并计算预期变化
    let (expected_change, prediction_range) =
        calculate_expected_change(ctx, &direction, &signal_confirmation);

    // 5. 计算综合置信度
    let confidence = calculate_comprehensive_confidence(ctx, &signal_confirmation, &direction);

    // 6. 风险评估
    let risk_assessment = assess_risk(ctx, expected_change);

    // 7. 生成关键因素
    let key_factors = generate_key_factors(ctx, &signal_summary, &direction);

    // 8. 生成建议操作
    let suggested_action = generate_suggested_action(
        &direction,
        &signal_confirmation,
        confidence,
        &risk_assessment,
        &ctx.market_regime.regime,
    );

    ProfessionalPredictionResult {
        direction,
        expected_change,
        prediction_range,
        confidence,
        signal_confirmation,
        market_regime: ctx.market_regime.regime,
        strategy_used: ctx.market_regime.regime.recommended_strategy(),
        signal_summary,
        risk_assessment,
        key_factors,
        suggested_action,
    }
}
