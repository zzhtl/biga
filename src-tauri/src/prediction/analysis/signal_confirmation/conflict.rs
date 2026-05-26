//! 冲突检测与假信号过滤

use super::{ConflictType, SignalSource, SignalType};

/// 检测信号冲突
pub(super) fn detect_conflicts(signals: &[SignalSource]) -> ConflictType {
    let mut bullish_count = 0;
    let mut bearish_count = 0;
    let mut bullish_strength = 0.0;
    let mut bearish_strength = 0.0;

    for signal in signals {
        if signal.direction > 0.2 {
            bullish_count += 1;
            bullish_strength += signal.direction * signal.strength;
        } else if signal.direction < -0.2 {
            bearish_count += 1;
            bearish_strength += signal.direction.abs() * signal.strength;
        }
    }

    let total_signals = bullish_count + bearish_count;
    if total_signals == 0 {
        return ConflictType::NoConflict;
    }

    let balance = (bullish_strength - bearish_strength).abs()
        / (bullish_strength + bearish_strength).max(0.01);
    let direction_diff = if bullish_count > bearish_count {
        bullish_count - bearish_count
    } else {
        bearish_count - bullish_count
    };
    let direction_ratio = direction_diff as f64 / total_signals as f64;

    if direction_ratio > 0.7 && balance > 0.6 {
        ConflictType::NoConflict
    } else if direction_ratio > 0.5 && balance > 0.4 {
        ConflictType::MinorConflict
    } else if direction_ratio > 0.3 && balance > 0.2 {
        ConflictType::ModerateConflict
    } else if direction_ratio > 0.1 {
        ConflictType::SevereConflict
    } else {
        ConflictType::CompleteConflict
    }
}

/// 检测假信号
pub(super) fn check_false_signal(
    signals: &[SignalSource],
    consistency: f64,
    conflict: &ConflictType,
) -> bool {
    // 假信号特征：
    // 1. 高冲突 + 低一致性
    // 2. 震荡指标和趋势指标严重分歧
    // 3. 量能不配合

    if consistency < 0.45
        && matches!(
            conflict,
            ConflictType::SevereConflict | ConflictType::CompleteConflict
        )
    {
        return true;
    }

    // 检查震荡指标和趋势指标分歧
    let trend_signals: Vec<_> = signals
        .iter()
        .filter(|s| matches!(s.signal_type, SignalType::Trend | SignalType::Momentum))
        .collect();
    let oscillator_signals: Vec<_> = signals
        .iter()
        .filter(|s| matches!(s.signal_type, SignalType::Oscillator))
        .collect();

    if !trend_signals.is_empty() && !oscillator_signals.is_empty() {
        let trend_avg: f64 =
            trend_signals.iter().map(|s| s.direction).sum::<f64>() / trend_signals.len() as f64;
        let osc_avg: f64 = oscillator_signals.iter().map(|s| s.direction).sum::<f64>()
            / oscillator_signals.len() as f64;

        // 严重分歧
        if (trend_avg > 0.3 && osc_avg < -0.3) || (trend_avg < -0.3 && osc_avg > 0.3) {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflict_detection() {
        let signals = vec![
            SignalSource {
                name: "test1".to_string(),
                direction: 0.8,
                strength: 0.9,
                reliability: 0.85,
                signal_type: SignalType::Trend,
                regime_aligned: true,
            },
            SignalSource {
                name: "test2".to_string(),
                direction: -0.7,
                strength: 0.8,
                reliability: 0.80,
                signal_type: SignalType::Oscillator,
                regime_aligned: true,
            },
        ];

        let conflict = detect_conflicts(&signals);
        assert!(matches!(
            conflict,
            ConflictType::SevereConflict | ConflictType::CompleteConflict
        ));
    }
}
