//! 信号综合、一致性、确认级别与行动建议

use super::{ConfirmationLevel, ConflictType, SignalSource};

/// 计算综合信号
pub(super) fn calculate_combined_signal(weighted: &[(f64, f64)]) -> (f64, f64) {
    let total_weight: f64 = weighted.iter().map(|(_, w)| w).sum();
    if total_weight == 0.0 {
        return (0.0, 0.0);
    }

    let weighted_sum: f64 = weighted.iter().map(|(s, w)| s * w).sum();
    let direction = (weighted_sum / total_weight).clamp(-1.0, 1.0);
    let strength = direction.abs();

    (direction, strength)
}

/// 计算信号一致性
pub(super) fn calculate_consistency(signals: &[SignalSource], overall_direction: f64) -> f64 {
    if signals.is_empty() {
        return 0.5;
    }

    let consistent_signals: f64 = signals
        .iter()
        .filter(|s| (s.direction > 0.0) == (overall_direction > 0.0) || s.direction.abs() < 0.2)
        .map(|s| s.reliability)
        .sum();

    let total_reliability: f64 = signals.iter().map(|s| s.reliability).sum();

    if total_reliability == 0.0 {
        return 0.5;
    }

    (consistent_signals / total_reliability).clamp(0.3, 1.0)
}

/// 确定确认级别
pub(super) fn determine_confirmation_level(
    signals: &[SignalSource],
    consistency: f64,
    conflict: &ConflictType,
) -> ConfirmationLevel {
    let strong_signals = signals
        .iter()
        .filter(|s| s.strength > 0.5 && s.reliability > 0.7)
        .count();

    match conflict {
        ConflictType::CompleteConflict => ConfirmationLevel::Invalid,
        ConflictType::SevereConflict => {
            if consistency > 0.7 {
                ConfirmationLevel::Weak
            } else {
                ConfirmationLevel::Invalid
            }
        }
        _ => {
            if strong_signals >= 4 && consistency > 0.75 {
                ConfirmationLevel::Strong
            } else if strong_signals >= 3 && consistency > 0.6 {
                ConfirmationLevel::Moderate
            } else if strong_signals >= 2 && consistency > 0.5 {
                ConfirmationLevel::Weak
            } else {
                ConfirmationLevel::Invalid
            }
        }
    }
}

/// 找出主导信号
pub(super) fn find_dominant_signal(signals: &[SignalSource]) -> String {
    signals
        .iter()
        .max_by(|a, b| {
            a.weighted_score()
                .abs()
                .partial_cmp(&b.weighted_score().abs())
                .unwrap()
        })
        .map(|s| s.name.clone())
        .unwrap_or_else(|| "无".to_string())
}

/// 生成行动建议
pub(super) fn generate_action_suggestion(
    direction: f64,
    strength: f64,
    level: &ConfirmationLevel,
    is_false: bool,
) -> String {
    if is_false {
        return "信号冲突严重，建议观望等待信号清晰".to_string();
    }

    match level {
        ConfirmationLevel::Invalid => "信号无效，不宜操作".to_string(),
        ConfirmationLevel::Weak => {
            if direction > 0.3 {
                format!("弱看涨信号(强度{:.0}%)，可小仓位试探", strength * 100.0)
            } else if direction < -0.3 {
                format!("弱看跌信号(强度{:.0}%)，注意风险", strength * 100.0)
            } else {
                "方向不明，建议观望".to_string()
            }
        }
        ConfirmationLevel::Moderate => {
            if direction > 0.3 {
                format!("中等看涨信号(强度{:.0}%)，可适量建仓", strength * 100.0)
            } else if direction < -0.3 {
                format!("中等看跌信号(强度{:.0}%)，考虑减仓", strength * 100.0)
            } else {
                "信号中性，保持观望".to_string()
            }
        }
        ConfirmationLevel::Strong => {
            if direction > 0.5 {
                format!("强烈看涨信号(强度{:.0}%)，积极做多", strength * 100.0)
            } else if direction < -0.5 {
                format!("强烈看跌信号(强度{:.0}%)，建议规避", strength * 100.0)
            } else {
                "多空均衡，持币观望".to_string()
            }
        }
    }
}
