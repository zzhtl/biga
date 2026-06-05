//! 校准区间预测：方向不可测但波动可测。
//!
//! 实证（见 .claude/CLAUDE.md / 记忆 no-faking-prediction-volatility）：单股次日方向无
//! alpha，引擎点预测已被压成"几乎不动"（这是 MAE 最优且诚实的）。与其伪造无预测力的
//! 日间涨跌，不如诚实给出**校准的涨跌区间带**。
//!
//! 方法：用近 20 日已实现日波动率 σ，第 d 个预测日累计区间半宽 = z·σ·√d（独立增量近似下
//! H 日收益的标准差）。z 倍数经 `examples/interval_calibration.rs` 在 40 票、h=1/5/10 的
//! walk-forward 上校准：realized20 法名义 80% 带（z≈1.34）经验覆盖 ~80%，90%→z≈1.90，
//! 95%→z≈2.50（A 股收益轻微厚尾，故 90/95 档 z 大于正态值）。

use crate::prediction::analysis::volatility_forecast::calculate_realized_volatility;
use crate::prediction::types::{Prediction, PredictionInterval};

/// 已实现波动率回看窗口（交易日）
const REALIZED_VOL_WINDOW: usize = 20;

/// 默认名义覆盖率
pub const DEFAULT_COVERAGE: f64 = 0.80;

/// 校准 z 倍数（realized20 法，经 interval_calibration 实测的经验分位）。
pub fn calibrated_z(confidence: f64) -> f64 {
    if confidence >= 0.95 {
        2.50
    } else if confidence >= 0.90 {
        1.90
    } else {
        1.34 // 默认 80%
    }
}

/// 近 N 日已实现日波动率（小数，如 0.02 = 2%）。数据不足时回退一个温和默认。
pub fn realized_daily_vol(closes: &[f64]) -> f64 {
    if closes.len() < 5 {
        return 0.02;
    }
    let start = closes.len().saturating_sub(REALIZED_VOL_WINDOW);
    let v = calculate_realized_volatility(&closes[start..]);
    if v.is_finite() && v > 1e-6 {
        v
    } else {
        0.02
    }
}

/// 为每个预测日填充校准区间带。
///
/// - `closes`：发起日（含）之前的收盘价序列，用于估计已实现波动率
/// - `base_price`：发起日真实收盘价，区间相对它度量累计涨跌
/// - `confidence`：名义覆盖率（默认 [`DEFAULT_COVERAGE`]）
///
/// 区间居中于各日点预测价（点预测已近乎不动，区间表达真实不确定性）。
pub fn attach_prediction_intervals(
    predictions: &mut [Prediction],
    closes: &[f64],
    base_price: f64,
    confidence: f64,
) {
    if base_price <= 0.0 || !base_price.is_finite() {
        return;
    }
    let sigma = realized_daily_vol(closes);
    let z = calibrated_z(confidence);
    for (idx, prediction) in predictions.iter_mut().enumerate() {
        let day = (idx + 1) as f64;
        // 累计半宽（相对 base 的百分点）
        let half_pct = z * sigma * day.sqrt() * 100.0;
        let cum_change = (prediction.predicted_price - base_price) / base_price * 100.0;
        let lower_change = cum_change - half_pct;
        let upper_change = cum_change + half_pct;
        prediction.interval = Some(PredictionInterval {
            confidence,
            lower_change_percent: lower_change,
            upper_change_percent: upper_change,
            lower_price: (base_price * (1.0 + lower_change / 100.0)).max(0.0),
            upper_price: base_price * (1.0 + upper_change / 100.0),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prediction::types::Prediction;

    fn make_predictions(base: f64, daily: f64, days: usize) -> Vec<Prediction> {
        let mut last = base;
        (1..=days)
            .map(|d| {
                last *= 1.0 + daily / 100.0;
                Prediction {
                    target_date: format!("2026-01-{d:02}"),
                    predicted_price: last,
                    predicted_change_percent: daily,
                    confidence: 0.3,
                    trading_signal: None,
                    signal_strength: None,
                    technical_indicators: None,
                    prediction_reason: None,
                    key_factors: None,
                    interval: None,
                }
            })
            .collect()
    }

    #[test]
    fn test_interval_widens_with_horizon_and_contains_point() {
        // 恒定 1% 日波动的合成价
        let closes: Vec<f64> = (0..40).map(|i| 100.0 * 1.01_f64.powi(i)).collect();
        let base = *closes.last().unwrap();
        let mut preds = make_predictions(base, 0.0, 5);
        attach_prediction_intervals(&mut preds, &closes, base, DEFAULT_COVERAGE);

        let widths: Vec<f64> = preds
            .iter()
            .map(|p| {
                let iv = p.interval.as_ref().unwrap();
                iv.upper_change_percent - iv.lower_change_percent
            })
            .collect();
        // 区间随预测天数单调变宽（√d）
        for w in widths.windows(2) {
            assert!(w[1] > w[0], "区间应随 horizon 变宽: {:?}", widths);
        }
        // 点预测落在自身区间内，且区间对称
        for p in &preds {
            let iv = p.interval.as_ref().unwrap();
            assert!(iv.lower_price < p.predicted_price && p.predicted_price < iv.upper_price);
            assert!((iv.confidence - 0.80).abs() < 1e-9);
        }
    }

    #[test]
    fn test_zero_base_price_is_noop() {
        let mut preds = make_predictions(100.0, 0.0, 3);
        attach_prediction_intervals(&mut preds, &[], 0.0, DEFAULT_COVERAGE);
        assert!(preds.iter().all(|p| p.interval.is_none()));
    }

    #[test]
    fn test_calibrated_z_tiers() {
        assert!((calibrated_z(0.80) - 1.34).abs() < 1e-9);
        assert!((calibrated_z(0.90) - 1.90).abs() < 1e-9);
        assert!((calibrated_z(0.95) - 2.50).abs() < 1e-9);
    }
}
