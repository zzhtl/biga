//! 专业级预测引擎
//! 
//! 基于华尔街顶级量化策略设计，核心理念：
//! 1. 多维度信号验证 - 多个独立信号同向确认才发出信号
//! 2. 市场状态自适应 - 趋势市用趋势跟踪，震荡市用均值回归
//! 3. 动态风险管理 - 根据波动率调整预测范围
//! 4. 概率思维 - 输出置信区间而非单一点预测
//! 5. 信号强度分级 - 只执行高置信度信号

use serde::{Deserialize, Serialize};
use crate::prediction::analysis::{
    market_regime::{MarketRegime, MarketRegimeAnalysis, StrategyType, VolatilityLevel},
    divergence::DivergenceAnalysis,
    TrendAnalysis, VolumePriceSignal, PatternRecognition, SupportResistance,
};
use crate::prediction::indicators::TechnicalIndicatorValues;
use crate::prediction::strategy::multi_factor::MultiFactorScore;

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
// 核心预测函数
// =============================================================================

/// 执行专业级预测
pub fn execute_professional_prediction(ctx: &PredictionContext) -> ProfessionalPredictionResult {
    // 1. 收集所有信号
    let signal_summary = collect_all_signals(ctx);
    
    // 2. 计算信号确认级别
    let signal_confirmation = calculate_signal_confirmation(&signal_summary);
    
    // 3. 确定预测方向
    let direction = determine_prediction_direction(
        &signal_summary,
        &ctx.market_regime,
        &ctx.divergence,
    );
    
    // 4. 根据市场状态选择策略并计算预期变化
    let (expected_change, prediction_range) = calculate_expected_change(
        ctx,
        &direction,
        &signal_confirmation,
    );
    
    // 5. 计算综合置信度
    let confidence = calculate_comprehensive_confidence(
        ctx,
        &signal_confirmation,
        &direction,
    );
    
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

/// 收集所有信号
fn collect_all_signals(ctx: &PredictionContext) -> SignalSummary {
    let mut bullish_signals = 0;
    let mut bearish_signals = 0;
    let mut signal_details = Vec::new();
    let mut weighted_score = 0.0;
    let mut total_weight = 0.0;
    
    // 1. 趋势信号 (权重: 0.25)
    let trend_weight = 0.25;
    total_weight += trend_weight;
    match &ctx.trend_analysis.overall_trend {
        super::super::analysis::TrendState::StrongBullish => {
            bullish_signals += 2;
            weighted_score += 1.0 * trend_weight;
            signal_details.push(SignalDetail {
                source: "趋势分析".to_string(),
                direction: "看涨".to_string(),
                strength: 1.0,
                description: "强势多头趋势".to_string(),
            });
        }
        super::super::analysis::TrendState::Bullish => {
            bullish_signals += 1;
            weighted_score += 0.5 * trend_weight;
            signal_details.push(SignalDetail {
                source: "趋势分析".to_string(),
                direction: "看涨".to_string(),
                strength: 0.7,
                description: "温和上涨趋势".to_string(),
            });
        }
        super::super::analysis::TrendState::Bearish => {
            bearish_signals += 1;
            weighted_score -= 0.5 * trend_weight;
            signal_details.push(SignalDetail {
                source: "趋势分析".to_string(),
                direction: "看跌".to_string(),
                strength: 0.7,
                description: "温和下跌趋势".to_string(),
            });
        }
        super::super::analysis::TrendState::StrongBearish => {
            bearish_signals += 2;
            weighted_score -= 1.0 * trend_weight;
            signal_details.push(SignalDetail {
                source: "趋势分析".to_string(),
                direction: "看跌".to_string(),
                strength: 1.0,
                description: "强势空头趋势".to_string(),
            });
        }
        _ => {}
    }
    
    // 2. 技术指标信号 (权重: 0.20)
    let tech_weight = 0.20;
    total_weight += tech_weight;
    
    // MACD信号
    if ctx.indicators.macd_golden_cross {
        bullish_signals += 1;
        weighted_score += 0.8 * tech_weight * 0.4;
        signal_details.push(SignalDetail {
            source: "MACD".to_string(),
            direction: "看涨".to_string(),
            strength: 0.8,
            description: "MACD金叉".to_string(),
        });
    } else if ctx.indicators.macd_death_cross {
        bearish_signals += 1;
        weighted_score -= 0.8 * tech_weight * 0.4;
        signal_details.push(SignalDetail {
            source: "MACD".to_string(),
            direction: "看跌".to_string(),
            strength: 0.8,
            description: "MACD死叉".to_string(),
        });
    }
    
    // RSI信号
    if ctx.indicators.rsi < 30.0 {
        bullish_signals += 1;
        weighted_score += 0.7 * tech_weight * 0.3;
        signal_details.push(SignalDetail {
            source: "RSI".to_string(),
            direction: "看涨".to_string(),
            strength: 0.7,
            description: format!("RSI超卖({:.1})", ctx.indicators.rsi),
        });
    } else if ctx.indicators.rsi > 70.0 {
        bearish_signals += 1;
        weighted_score -= 0.7 * tech_weight * 0.3;
        signal_details.push(SignalDetail {
            source: "RSI".to_string(),
            direction: "看跌".to_string(),
            strength: 0.7,
            description: format!("RSI超买({:.1})", ctx.indicators.rsi),
        });
    }
    
    // KDJ信号
    if ctx.indicators.kdj_golden_cross && ctx.indicators.kdj_oversold {
        bullish_signals += 1;
        weighted_score += 0.75 * tech_weight * 0.3;
        signal_details.push(SignalDetail {
            source: "KDJ".to_string(),
            direction: "看涨".to_string(),
            strength: 0.75,
            description: "KDJ超卖金叉".to_string(),
        });
    } else if ctx.indicators.kdj_death_cross && ctx.indicators.kdj_overbought {
        bearish_signals += 1;
        weighted_score -= 0.75 * tech_weight * 0.3;
        signal_details.push(SignalDetail {
            source: "KDJ".to_string(),
            direction: "看跌".to_string(),
            strength: 0.75,
            description: "KDJ超买死叉".to_string(),
        });
    }
    
    // 3. 量价信号 (权重: 0.18)
    let volume_weight = 0.18;
    total_weight += volume_weight;
    
    match ctx.volume_signal.direction.as_str() {
        "上涨" => {
            let strength = ctx.volume_signal.confidence;
            bullish_signals += 1;
            weighted_score += strength * volume_weight;
            signal_details.push(SignalDetail {
                source: "量价分析".to_string(),
                direction: "看涨".to_string(),
                strength,
                description: ctx.volume_signal.signal.clone(),
            });
        }
        "下跌" => {
            let strength = ctx.volume_signal.confidence;
            bearish_signals += 1;
            weighted_score -= strength * volume_weight;
            signal_details.push(SignalDetail {
                source: "量价分析".to_string(),
                direction: "看跌".to_string(),
                strength,
                description: ctx.volume_signal.signal.clone(),
            });
        }
        _ => {}
    }
    
    // 4. 背离信号 (权重: 0.15，华尔街非常重视背离)
    let divergence_weight = 0.15;
    total_weight += divergence_weight;
    
    if ctx.divergence.has_divergence {
        let div_score = ctx.divergence.composite_score;
        if div_score > 0.2 {
            bullish_signals += 1;
            weighted_score += div_score * divergence_weight;
            signal_details.push(SignalDetail {
                source: "背离检测".to_string(),
                direction: "看涨".to_string(),
                strength: div_score.abs(),
                description: ctx.divergence.suggested_action.clone(),
            });
        } else if div_score < -0.2 {
            bearish_signals += 1;
            weighted_score += div_score * divergence_weight;
            signal_details.push(SignalDetail {
                source: "背离检测".to_string(),
                direction: "看跌".to_string(),
                strength: div_score.abs(),
                description: ctx.divergence.suggested_action.clone(),
            });
        }
    }
    
    // 5. K线形态信号 (权重: 0.12)
    let pattern_weight = 0.12;
    total_weight += pattern_weight;
    
    for pattern in &ctx.patterns {
        if pattern.is_bullish && pattern.reliability > 0.6 {
            bullish_signals += 1;
            weighted_score += pattern.reliability * pattern_weight;
            signal_details.push(SignalDetail {
                source: "K线形态".to_string(),
                direction: "看涨".to_string(),
                strength: pattern.reliability,
                description: pattern.description.clone(),
            });
        } else if !pattern.is_bullish && pattern.reliability > 0.6 {
            bearish_signals += 1;
            weighted_score -= pattern.reliability * pattern_weight;
            signal_details.push(SignalDetail {
                source: "K线形态".to_string(),
                direction: "看跌".to_string(),
                strength: pattern.reliability,
                description: pattern.description.clone(),
            });
        }
    }
    
    // 6. 市场状态信号 (权重: 0.10)
    let regime_weight = 0.10;
    total_weight += regime_weight;
    
    match ctx.market_regime.regime {
        MarketRegime::StrongUptrend | MarketRegime::ModerateUptrend => {
            bullish_signals += 1;
            weighted_score += ctx.market_regime.trend_strength.abs() * regime_weight;
            signal_details.push(SignalDetail {
                source: "市场状态".to_string(),
                direction: "看涨".to_string(),
                strength: ctx.market_regime.confidence,
                description: ctx.market_regime.description.clone(),
            });
        }
        MarketRegime::StrongDowntrend | MarketRegime::ModerateDowntrend => {
            bearish_signals += 1;
            weighted_score -= ctx.market_regime.trend_strength.abs() * regime_weight;
            signal_details.push(SignalDetail {
                source: "市场状态".to_string(),
                direction: "看跌".to_string(),
                strength: ctx.market_regime.confidence,
                description: ctx.market_regime.description.clone(),
            });
        }
        MarketRegime::PotentialBottom => {
            bullish_signals += 1;
            weighted_score += 0.6 * regime_weight;
            signal_details.push(SignalDetail {
                source: "市场状态".to_string(),
                direction: "看涨".to_string(),
                strength: 0.6,
                description: "检测到潜在底部".to_string(),
            });
        }
        MarketRegime::PotentialTop => {
            bearish_signals += 1;
            weighted_score -= 0.6 * regime_weight;
            signal_details.push(SignalDetail {
                source: "市场状态".to_string(),
                direction: "看跌".to_string(),
                strength: 0.6,
                description: "检测到潜在顶部".to_string(),
            });
        }
        _ => {}
    }
    
    // 计算净信号得分
    let net_signal_score = if total_weight > 0.0 {
        (weighted_score / total_weight).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    
    SignalSummary {
        bullish_signals,
        bearish_signals,
        signal_details,
        net_signal_score,
    }
}

/// 计算信号确认级别
fn calculate_signal_confirmation(summary: &SignalSummary) -> SignalConfirmation {
    let net_signals = (summary.bullish_signals as i32 - summary.bearish_signals as i32).abs();
    let dominant_signals = summary.bullish_signals.max(summary.bearish_signals);
    
    if dominant_signals >= 4 && net_signals >= 3 {
        SignalConfirmation::StrongConfirm
    } else if dominant_signals >= 3 && net_signals >= 2 {
        SignalConfirmation::ModerateConfirm
    } else if dominant_signals >= 2 && net_signals >= 1 {
        SignalConfirmation::WeakConfirm
    } else {
        SignalConfirmation::NoConfirm
    }
}

/// 确定预测方向
fn determine_prediction_direction(
    summary: &SignalSummary,
    regime: &MarketRegimeAnalysis,
    divergence: &DivergenceAnalysis,
) -> PredictionDirection {
    let net_score = summary.net_signal_score;
    
    // 考虑背离信号的特殊权重（背离往往预示反转）
    let divergence_adjustment = if divergence.has_divergence {
        divergence.composite_score * 0.2
    } else {
        0.0
    };
    
    // 考虑市场状态
    let regime_adjustment = match regime.regime {
        MarketRegime::PotentialTop => -0.15,
        MarketRegime::PotentialBottom => 0.15,
        _ => 0.0,
    };
    
    let adjusted_score = net_score + divergence_adjustment + regime_adjustment;
    
    if adjusted_score > 0.6 {
        PredictionDirection::StrongBullish
    } else if adjusted_score > 0.25 {
        PredictionDirection::Bullish
    } else if adjusted_score < -0.6 {
        PredictionDirection::StrongBearish
    } else if adjusted_score < -0.25 {
        PredictionDirection::Bearish
    } else {
        PredictionDirection::Neutral
    }
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

/// 根据A股规则限制预测幅度
fn apply_a_share_limits(change: f64, stock_code: Option<&str>) -> f64 {
    let (limit_down, limit_up) = get_stock_price_limits(stock_code);
    change.clamp(limit_down, limit_up)
}

/// 计算预期变化和预测区间
fn calculate_expected_change(
    ctx: &PredictionContext,
    direction: &PredictionDirection,
    confirmation: &SignalConfirmation,
) -> (f64, (f64, f64)) {
    let strategy = ctx.market_regime.regime.recommended_strategy();
    
    // 基础预期变化
    let base_change = match strategy {
        StrategyType::TrendFollowing => calculate_trend_following_change(ctx, direction),
        StrategyType::MeanReversion => calculate_mean_reversion_change(ctx, direction),
        StrategyType::Reversal => calculate_reversal_change(ctx, direction),
    };
    
    // 根据信号确认级别调整
    let confirmation_multiplier = confirmation.to_weight();
    let adjusted_change = base_change * confirmation_multiplier;
    
    // 应用A股涨跌停限制
    let limited_change = apply_a_share_limits(adjusted_change, None);
    
    // 根据波动率计算预测区间
    let volatility_multiplier = ctx.market_regime.volatility_level.adjustment_factor();
    let range_width = ctx.volatility * 100.0 * volatility_multiplier * 1.5;
    
    // 预测区间也要遵守涨跌停限制
    let lower = apply_a_share_limits(limited_change - range_width, None);
    let upper = apply_a_share_limits(limited_change + range_width, None);
    
    (limited_change, (lower, upper))
}

/// 趋势跟踪策略计算
fn calculate_trend_following_change(ctx: &PredictionContext, direction: &PredictionDirection) -> f64 {
    let trend_strength = ctx.trend_analysis.trend_strength;
    let regime_strength = ctx.market_regime.trend_strength;
    
    // 基础变化基于趋势强度
    let base = (trend_strength + regime_strength) / 2.0 * 2.0; // 百分比
    
    // 技术指标调整
    let tech_adjustment = if ctx.indicators.macd_histogram > 0.0 {
        0.3
    } else if ctx.indicators.macd_histogram < 0.0 {
        -0.3
    } else {
        0.0
    };
    
    // 量价配合调整
    let volume_adjustment = match ctx.volume_signal.direction.as_str() {
        "上涨" if ctx.volume_signal.confidence > 0.7 => 0.4,
        "下跌" if ctx.volume_signal.confidence > 0.7 => -0.4,
        _ => 0.0,
    };
    
    let change = base + tech_adjustment + volume_adjustment;
    
    // 根据方向限制变化范围
    match direction {
        PredictionDirection::StrongBullish => change.max(0.5).min(5.0),
        PredictionDirection::Bullish => change.max(0.2).min(3.0),
        PredictionDirection::StrongBearish => change.min(-0.5).max(-5.0),
        PredictionDirection::Bearish => change.min(-0.2).max(-3.0),
        PredictionDirection::Neutral => change.clamp(-1.0, 1.0),
    }
}

/// 均值回归策略计算
fn calculate_mean_reversion_change(ctx: &PredictionContext, _direction: &PredictionDirection) -> f64 {
    // 使用布林带位置和RSI计算均值回归
    let rsi = ctx.indicators.rsi;
    
    // RSI偏离度
    let rsi_deviation = (50.0 - rsi) / 50.0;
    
    // 基于偏离度的回归预期
    let base = rsi_deviation * 1.5;
    
    // CCI调整
    let cci_adjustment = if ctx.indicators.cci > 100.0 {
        -0.3
    } else if ctx.indicators.cci < -100.0 {
        0.3
    } else {
        0.0
    };
    
    let change = base + cci_adjustment;
    
    // 震荡市变化幅度相对较小
    change.clamp(-2.5, 2.5)
}

/// 反转策略计算
fn calculate_reversal_change(ctx: &PredictionContext, direction: &PredictionDirection) -> f64 {
    // 基于背离和转折点信号
    let divergence_component = ctx.divergence.composite_score * 2.0;
    
    // 基于RSI极端值
    let rsi_component = if ctx.indicators.rsi > 80.0 {
        -1.5
    } else if ctx.indicators.rsi < 20.0 {
        1.5
    } else {
        0.0
    };
    
    // 基于市场状态
    let regime_component = match ctx.market_regime.regime {
        MarketRegime::PotentialTop => -1.0,
        MarketRegime::PotentialBottom => 1.0,
        _ => 0.0,
    };
    
    let change = divergence_component + rsi_component + regime_component;
    
    // 反转信号通常预期较大变化
    match direction {
        PredictionDirection::StrongBullish | PredictionDirection::StrongBearish => {
            change.clamp(-4.0, 4.0)
        }
        _ => change.clamp(-2.5, 2.5)
    }
}

/// 计算综合置信度
fn calculate_comprehensive_confidence(
    ctx: &PredictionContext,
    confirmation: &SignalConfirmation,
    direction: &PredictionDirection,
) -> f64 {
    // 基础置信度来自信号确认
    let base_confidence = confirmation.to_weight() * 0.5;
    
    // 市场状态置信度
    let regime_confidence = ctx.market_regime.confidence * 0.15;
    
    // 趋势一致性加成
    let trend_consistency = if ctx.market_regime.regime.is_trending() {
        ctx.trend_analysis.trend_confidence * 0.15
    } else {
        0.1
    };
    
    // 背离信号加成/减成
    let divergence_factor = if ctx.divergence.has_divergence {
        // 背离与预测方向一致时加成
        let divergence_bullish = ctx.divergence.composite_score > 0.0;
        let prediction_bullish = matches!(direction, PredictionDirection::StrongBullish | PredictionDirection::Bullish);
        
        if divergence_bullish == prediction_bullish {
            ctx.divergence.overall_confidence * 0.10
        } else {
            -0.05 // 背离与预测方向冲突时减少置信度
        }
    } else {
        0.0
    };
    
    // 波动率影响（高波动降低置信度）
    let volatility_penalty = match ctx.market_regime.volatility_level {
        VolatilityLevel::VeryHigh => -0.10,
        VolatilityLevel::High => -0.05,
        VolatilityLevel::Normal => 0.0,
        VolatilityLevel::Low => 0.03,
        VolatilityLevel::VeryLow => 0.05,
    };
    
    let confidence = base_confidence + regime_confidence + trend_consistency 
                    + divergence_factor + volatility_penalty;
    
    confidence.clamp(0.25, 0.92)
}

/// 风险评估
fn assess_risk(ctx: &PredictionContext, expected_change: f64) -> RiskAssessment {
    // 波动率风险
    let volatility_risk = match ctx.market_regime.volatility_level {
        VolatilityLevel::VeryHigh => 0.9,
        VolatilityLevel::High => 0.7,
        VolatilityLevel::Normal => 0.5,
        VolatilityLevel::Low => 0.3,
        VolatilityLevel::VeryLow => 0.2,
    };
    
    // 计算支撑阻力距离
    let support_distance = if ctx.support_resistance.support_levels.is_empty() {
        5.0
    } else {
        let nearest_support = ctx.support_resistance.support_levels
            .iter()
            .filter(|&&s| s < ctx.current_price)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(ctx.current_price * 0.95);
        ((ctx.current_price - nearest_support) / ctx.current_price * 100.0).abs()
    };
    
    let resistance_distance = if ctx.support_resistance.resistance_levels.is_empty() {
        5.0
    } else {
        let nearest_resistance = ctx.support_resistance.resistance_levels
            .iter()
            .filter(|&&r| r > ctx.current_price)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(ctx.current_price * 1.05);
        ((nearest_resistance - ctx.current_price) / ctx.current_price * 100.0).abs()
    };
    
    // 建议止损（基于ATR和波动率）
    let base_stop = ctx.volatility * 100.0 * 2.0;
    let suggested_stop_loss = base_stop.max(2.0).min(8.0);
    
    // 建议止盈（基于风险收益比目标2:1）
    let suggested_take_profit = if expected_change.abs() > 0.5 {
        expected_change.abs() * 2.0
    } else {
        suggested_stop_loss * 2.0
    };
    
    // 风险收益比
    let risk_reward_ratio = if suggested_stop_loss > 0.0 {
        suggested_take_profit / suggested_stop_loss
    } else {
        2.0
    };
    
    // 风险等级
    let risk_level = if volatility_risk > 0.7 && support_distance > 4.0 {
        "高风险"
    } else if volatility_risk > 0.5 || support_distance > 3.0 {
        "中等风险"
    } else {
        "低风险"
    };
    
    RiskAssessment {
        risk_level: risk_level.to_string(),
        volatility_risk,
        support_distance,
        resistance_distance,
        suggested_stop_loss,
        suggested_take_profit,
        risk_reward_ratio,
    }
}

/// 生成关键因素
fn generate_key_factors(
    ctx: &PredictionContext,
    summary: &SignalSummary,
    _direction: &PredictionDirection,
) -> Vec<String> {
    let mut factors = Vec::new();
    
    // 添加市场状态
    factors.push(format!("市场状态: {}", ctx.market_regime.regime.to_string()));
    
    // 添加趋势信息
    factors.push(format!("趋势强度: {:.0}%", ctx.trend_analysis.trend_confidence * 100.0));
    
    // 添加最强信号
    if let Some(strongest) = summary.signal_details.iter()
        .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap())
    {
        factors.push(format!("主导信号: {} - {}", strongest.source, strongest.description));
    }
    
    // 添加背离信息
    if ctx.divergence.has_divergence {
        factors.push(format!("背离信号: {}", ctx.divergence.primary_direction));
    }
    
    // 添加量价配合
    factors.push(format!("量价关系: {}", ctx.volume_signal.signal));
    
    // 添加关键指标
    factors.push(format!("RSI: {:.1} | MACD柱: {:.4}", 
        ctx.indicators.rsi, ctx.indicators.macd_histogram));
    
    factors
}

/// 生成建议操作
fn generate_suggested_action(
    direction: &PredictionDirection,
    confirmation: &SignalConfirmation,
    confidence: f64,
    risk: &RiskAssessment,
    regime: &MarketRegime,
) -> String {
    // 信号不足时保守操作
    if matches!(confirmation, SignalConfirmation::NoConfirm) || confidence < 0.45 {
        return "信号不明确，建议观望等待更清晰信号".to_string();
    }
    
    // 高风险时降低建议力度
    if risk.risk_level == "高风险" {
        return match direction {
            PredictionDirection::StrongBullish | PredictionDirection::Bullish => {
                format!("谨慎看涨，建议小仓位试探，止损{:.1}%", risk.suggested_stop_loss)
            }
            PredictionDirection::StrongBearish | PredictionDirection::Bearish => {
                format!("谨慎看跌，建议减仓观望，注意反弹风险")
            }
            PredictionDirection::Neutral => {
                "高波动震荡，建议暂时观望".to_string()
            }
        };
    }
    
    // 正常信号建议
    match (direction, confirmation) {
        (PredictionDirection::StrongBullish, SignalConfirmation::StrongConfirm) => {
            format!(
                "强烈看涨信号({:.0}%置信度)，建议积极做多，目标涨幅{:.1}%，止损{:.1}%",
                confidence * 100.0, risk.suggested_take_profit, risk.suggested_stop_loss
            )
        }
        (PredictionDirection::StrongBullish, _) | 
        (PredictionDirection::Bullish, SignalConfirmation::StrongConfirm) => {
            format!(
                "看涨信号({:.0}%置信度)，可考虑分批建仓，止损{:.1}%",
                confidence * 100.0, risk.suggested_stop_loss
            )
        }
        (PredictionDirection::Bullish, _) => {
            format!(
                "温和看涨({:.0}%置信度)，可小仓位参与，严格止损{:.1}%",
                confidence * 100.0, risk.suggested_stop_loss
            )
        }
        (PredictionDirection::StrongBearish, SignalConfirmation::StrongConfirm) => {
            format!(
                "强烈看跌信号({:.0}%置信度)，建议减仓或对冲，预计下跌{:.1}%",
                confidence * 100.0, risk.suggested_take_profit
            )
        }
        (PredictionDirection::StrongBearish, _) |
        (PredictionDirection::Bearish, SignalConfirmation::StrongConfirm) => {
            format!(
                "看跌信号({:.0}%置信度)，建议降低仓位，等待企稳",
                confidence * 100.0
            )
        }
        (PredictionDirection::Bearish, _) => {
            format!(
                "温和看跌({:.0}%置信度)，建议谨慎持有，设好止损",
                confidence * 100.0
            )
        }
        (PredictionDirection::Neutral, _) => {
            if regime.is_trending() {
                "趋势中继整理，建议持有等待方向明确".to_string()
            } else {
                "震荡行情，可考虑区间操作，高抛低吸".to_string()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_signal_confirmation() {
        let summary = SignalSummary {
            bullish_signals: 5,
            bearish_signals: 1,
            signal_details: vec![],
            net_signal_score: 0.7,
        };
        let confirmation = calculate_signal_confirmation(&summary);
        assert_eq!(confirmation, SignalConfirmation::StrongConfirm);
    }
}

