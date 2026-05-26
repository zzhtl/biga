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
    // 信号不足时保守操作
    if matches!(confirmation, SignalConfirmation::NoConfirm) || confidence < 0.45 {
        return "信号不明确，建议观望等待更清晰信号".to_string();
    }

    // 高风险时降低建议力度
    if risk.risk_level == "高风险" {
        return match direction {
            PredictionDirection::StrongBullish | PredictionDirection::Bullish => {
                format!(
                    "谨慎看涨，建议小仓位试探，止损{:.1}%",
                    risk.suggested_stop_loss
                )
            }
            PredictionDirection::StrongBearish | PredictionDirection::Bearish => {
                format!("谨慎看跌，建议减仓观望，注意反弹风险")
            }
            PredictionDirection::Neutral => "高波动震荡，建议暂时观望".to_string(),
        };
    }

    // 正常信号建议
    match (direction, confirmation) {
        (PredictionDirection::StrongBullish, SignalConfirmation::StrongConfirm) => {
            format!(
                "强烈看涨信号({:.0}%置信度)，建议积极做多，目标涨幅{:.1}%，止损{:.1}%",
                confidence * 100.0,
                risk.suggested_take_profit,
                risk.suggested_stop_loss
            )
        }
        (PredictionDirection::StrongBullish, _)
        | (PredictionDirection::Bullish, SignalConfirmation::StrongConfirm) => {
            format!(
                "看涨信号({:.0}%置信度)，可考虑分批建仓，止损{:.1}%",
                confidence * 100.0,
                risk.suggested_stop_loss
            )
        }
        (PredictionDirection::Bullish, _) => {
            format!(
                "温和看涨({:.0}%置信度)，可小仓位参与，严格止损{:.1}%",
                confidence * 100.0,
                risk.suggested_stop_loss
            )
        }
        (PredictionDirection::StrongBearish, SignalConfirmation::StrongConfirm) => {
            format!(
                "强烈看跌信号({:.0}%置信度)，建议减仓或对冲，预计下跌{:.1}%",
                confidence * 100.0,
                risk.suggested_take_profit
            )
        }
        (PredictionDirection::StrongBearish, _)
        | (PredictionDirection::Bearish, SignalConfirmation::StrongConfirm) => {
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
