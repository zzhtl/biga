//! 关键因素与操作建议生成

use super::{
    PredictionContext, PredictionDirection, RiskAssessment, SignalConfirmation, SignalSummary,
};
use crate::prediction::analysis::market_regime::MarketRegime;

/// 生成关键因素
pub(super) fn generate_key_factors(
    ctx: &PredictionContext,
    summary: &SignalSummary,
    _direction: &PredictionDirection,
) -> Vec<String> {
    let mut factors = Vec::new();

    // 添加市场状态
    factors.push(format!("市场状态: {}", ctx.market_regime.regime.to_string()));

    // 添加趋势信息
    factors.push(format!(
        "趋势强度: {:.0}%",
        ctx.trend_analysis.trend_confidence * 100.0
    ));

    // 添加最强信号
    if let Some(strongest) = summary
        .signal_details
        .iter()
        .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap())
    {
        factors.push(format!(
            "主导信号: {} - {}",
            strongest.source, strongest.description
        ));
    }

    // 添加背离信息
    if ctx.divergence.has_divergence {
        factors.push(format!("背离信号: {}", ctx.divergence.primary_direction));
    }

    // 添加量价配合
    factors.push(format!("量价关系: {}", ctx.volume_signal.signal));

    // 添加关键指标
    factors.push(format!(
        "RSI: {:.1} | MACD柱: {:.4}",
        ctx.indicators.rsi, ctx.indicators.macd_histogram
    ));

    factors
}

/// 生成建议操作
pub(super) fn generate_suggested_action(
    direction: &PredictionDirection,
    confirmation: &SignalConfirmation,
    confidence: f64,
    risk: &RiskAssessment,
    regime: &MarketRegime,
) -> String {
    // 这里只描述技术状态与风险关注点，不把信号强度包装成方向命中概率。
    if matches!(confirmation, SignalConfirmation::NoConfirm) || confidence < 0.45 {
        return "技术信号相互冲突，当前状态不支持方向性结论".to_string();
    }

    if risk.risk_level == "高风险" {
        return match direction {
            PredictionDirection::StrongBullish | PredictionDirection::Bullish => {
                format!(
                    "技术状态偏多，但波动风险较高；下行失效参考幅度约{:.1}%",
                    risk.suggested_stop_loss
                )
            }
            PredictionDirection::StrongBearish | PredictionDirection::Bearish => {
                "技术状态偏空且波动风险较高，需重点关注95%压力区间".to_string()
            }
            PredictionDirection::Neutral => "高波动震荡，方向信号不足".to_string(),
        };
    }

    match (direction, confirmation) {
        (PredictionDirection::StrongBullish, SignalConfirmation::StrongConfirm) => {
            format!(
                "技术状态明显偏多，信号强度{:.0}%；下行失效参考幅度约{:.1}%",
                confidence * 100.0,
                risk.suggested_stop_loss
            )
        }
        (PredictionDirection::StrongBullish, _)
        | (PredictionDirection::Bullish, SignalConfirmation::StrongConfirm) => {
            format!(
                "技术状态偏多，信号强度{:.0}%；下行失效参考幅度约{:.1}%",
                confidence * 100.0,
                risk.suggested_stop_loss
            )
        }
        (PredictionDirection::Bullish, _) => {
            format!(
                "技术状态温和偏多，信号强度{:.0}%；不代表上涨概率",
                confidence * 100.0
            )
        }
        (PredictionDirection::StrongBearish, SignalConfirmation::StrongConfirm) => {
            format!(
                "技术状态明显偏空，信号强度{:.0}%；重点观察下方支撑与压力区间",
                confidence * 100.0
            )
        }
        (PredictionDirection::StrongBearish, _)
        | (PredictionDirection::Bearish, SignalConfirmation::StrongConfirm) => {
            format!(
                "技术状态偏空，信号强度{:.0}%；需关注趋势失效条件",
                confidence * 100.0
            )
        }
        (PredictionDirection::Bearish, _) => {
            format!(
                "技术状态温和偏空，信号强度{:.0}%；不代表下跌概率",
                confidence * 100.0
            )
        }
        (PredictionDirection::Neutral, _) => {
            if regime.is_trending() {
                "趋势中继整理，当前方向证据不足".to_string()
            } else {
                "震荡状态，当前方向证据不足".to_string()
            }
        }
    }
}
