//! 非线性变换、信号确认与信号生成

use crate::prediction::analysis::market_regime::MarketRegime;

/// Sigmoid变换（平滑极端值）
pub(super) fn sigmoid_transform(x: f64) -> f64 {
    // 将0-1范围的值通过sigmoid平滑
    let centered = (x - 0.5) * 4.0; // 中心化并放大
    1.0 / (1.0 + (-centered).exp())
}

/// 计算信号确认数量
pub(super) fn count_signal_confirmations(
    trend: f64,
    volume: f64,
    momentum: f64,
    pattern: f64,
    sentiment: f64,
) -> i32 {
    let bullish_threshold = 0.6;
    let bearish_threshold = 0.4;

    let scores = [trend, volume, momentum, pattern, sentiment];

    let bullish_count = scores.iter().filter(|&&s| s > bullish_threshold).count();
    let bearish_count = scores.iter().filter(|&&s| s < bearish_threshold).count();

    // 返回优势方向的确认数量
    if bullish_count > bearish_count {
        bullish_count as i32
    } else if bearish_count > bullish_count {
        -(bearish_count as i32)
    } else {
        0
    }
}

/// 应用确认加成/惩罚
pub(super) fn apply_confirmation_adjustment(score: f64, confirmations: i32) -> f64 {
    let abs_conf = confirmations.abs();

    let adjustment = match abs_conf {
        4..=5 => 8.0, // 强确认：+8分
        3 => 5.0,     // 中等确认：+5分
        2 => 2.0,     // 弱确认：+2分
        _ => -3.0,    // 无确认/冲突：-3分
    };

    // 根据方向应用调整
    let direction_factor = if confirmations > 0 && score > 50.0 {
        1.0
    } else if confirmations < 0 && score < 50.0 {
        1.0
    } else if confirmations == 0 {
        1.0
    } else {
        0.5 // 方向冲突时减半调整
    };

    (score + adjustment * direction_factor).clamp(0.0, 100.0)
}

/// 生成增强信号
pub(super) fn generate_enhanced_signal(
    score: f64,
    confirmations: i32,
    regime: Option<&MarketRegime>,
) -> (String, f64) {
    // 基础信号阈值
    let (signal, base_strength) = if score >= 78.0 {
        ("强烈看涨", 0.92)
    } else if score >= 65.0 {
        ("看涨", 0.75)
    } else if score >= 55.0 {
        ("温和看涨", 0.60)
    } else if score >= 45.0 {
        ("中性", 0.50)
    } else if score >= 35.0 {
        ("温和看跌", 0.60)
    } else if score >= 22.0 {
        ("看跌", 0.75)
    } else {
        ("强烈看跌", 0.92)
    };

    // 根据确认数量调整强度
    let confirmation_factor: f64 = match confirmations.abs() {
        4..=5 => 1.1,
        3 => 1.05,
        2 => 1.0,
        1 => 0.95,
        _ => 0.85,
    };

    // 根据市场状态微调
    let regime_factor: f64 = match regime {
        Some(MarketRegime::StrongUptrend) | Some(MarketRegime::StrongDowntrend) => 1.05,
        Some(MarketRegime::PotentialTop) | Some(MarketRegime::PotentialBottom) => 1.08,
        _ => 1.0,
    };

    let final_strength =
        (base_strength * confirmation_factor * regime_factor).clamp(0.3_f64, 0.95_f64);

    (signal.to_string(), final_strength)
}

/// 生成基础信号（向后兼容）
#[allow(dead_code)]
pub(super) fn generate_signal(total_score: f64) -> (String, f64) {
    generate_enhanced_signal(total_score, 0, None)
}
