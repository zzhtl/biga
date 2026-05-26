//! 信号收集

use super::{PredictionContext, SignalDetail, SignalSummary};
use crate::prediction::analysis::market_regime::MarketRegime;
use crate::prediction::analysis::support_resistance::{is_breakdown, is_breakout};
use crate::prediction::analysis::TrendState;

/// 收集所有信号
pub(super) fn collect_all_signals(ctx: &PredictionContext) -> SignalSummary {
    let mut bullish_signals = 0;
    let mut bearish_signals = 0;
    let mut signal_details = Vec::new();
    let mut weighted_score = 0.0;
    let mut total_weight = 0.0;

    // 1. 趋势信号 (权重: 0.25)
    let trend_weight = 0.25;
    total_weight += trend_weight;
    match &ctx.trend_analysis.overall_trend {
        TrendState::StrongBullish => {
            bullish_signals += 2;
            weighted_score += 1.0 * trend_weight;
            signal_details.push(SignalDetail {
                source: "趋势分析".to_string(),
                direction: "看涨".to_string(),
                strength: 1.0,
                description: "强势多头趋势".to_string(),
            });
        }
        TrendState::Bullish => {
            bullish_signals += 1;
            weighted_score += 0.5 * trend_weight;
            signal_details.push(SignalDetail {
                source: "趋势分析".to_string(),
                direction: "看涨".to_string(),
                strength: 0.7,
                description: "温和上涨趋势".to_string(),
            });
        }
        TrendState::Bearish => {
            bearish_signals += 1;
            weighted_score -= 0.5 * trend_weight;
            signal_details.push(SignalDetail {
                source: "趋势分析".to_string(),
                direction: "看跌".to_string(),
                strength: 0.7,
                description: "温和下跌趋势".to_string(),
            });
        }
        TrendState::StrongBearish => {
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

    // 7. 量价突破信号 (权重: 0.10，用真实量比验证突破有效性)
    let breakout_weight = 0.10;
    total_weight += breakout_weight;
    let price = ctx.current_price;
    let volume_ratio = ctx.indicators.volume_ratio;

    // 价格上方刚被突破的阻力位（取价格下方最近的阻力）
    let broken_resistance = ctx
        .support_resistance
        .resistance_levels
        .iter()
        .filter(|&&r| r < price)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .copied();
    // 价格下方刚被跌破的支撑位（取价格上方最近的支撑）
    let broken_support = ctx
        .support_resistance
        .support_levels
        .iter()
        .filter(|&&s| s > price)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .copied();

    if let Some(res) = broken_resistance {
        if is_breakout(price, res, volume_ratio) {
            bullish_signals += 1;
            weighted_score += 0.8 * breakout_weight;
            signal_details.push(SignalDetail {
                source: "量价突破".to_string(),
                direction: "看涨".to_string(),
                strength: 0.8,
                description: format!("放量突破阻力位(量比{volume_ratio:.1})"),
            });
        }
    }
    if let Some(sup) = broken_support {
        if is_breakdown(price, sup, volume_ratio) {
            bearish_signals += 1;
            weighted_score -= 0.8 * breakout_weight;
            signal_details.push(SignalDetail {
                source: "量价突破".to_string(),
                direction: "看跌".to_string(),
                strength: 0.8,
                description: format!("放量跌破支撑位(量比{volume_ratio:.1})"),
            });
        }
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
