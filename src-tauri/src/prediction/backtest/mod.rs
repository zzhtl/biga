//! 真实走步回测框架
//!
//! 对历史数据做 walk-forward 回测：每个交易日仅用其之前的数据运行生产预测路径
//! （[`crate::prediction::model::inference::predict_from_historical`]），将预测涨跌幅与未来真实涨跌幅
//! 对比，量化方向准确率、误差与简单策略收益。

pub mod metrics;

use crate::db::models::HistoricalData;
use crate::prediction::model::inference::{predict_from_historical, MAX_ANALYSIS_DAYS};
use crate::prediction::types::{PredictionInterval, PredictionRequest, PredictionResponse};
use chrono::NaiveDate;
use metrics::{compute_metrics, BacktestMetrics, BacktestSample};

/// 回测最小回看窗口（分析管线要求 ≥60 个交易日）
pub const MIN_LOOKBACK: usize = 60;

/// 回测报告
#[derive(Debug, Clone)]
pub struct BacktestReport {
    pub stock_code: String,
    /// 预测周期（交易日）
    pub horizon: usize,
    /// 评估指标
    pub metrics: BacktestMetrics,
    /// 样本明细，用于前端展示和按日期核验
    pub observations: Vec<BacktestObservation>,
}

/// 单次走步预测明细
#[derive(Debug, Clone)]
pub struct BacktestObservation {
    pub prediction_date: NaiveDate,
    pub target_date: NaiveDate,
    pub base_price: f64,
    pub predicted_price: f64,
    pub predicted_daily_changes: Vec<f64>,
    pub actual_price: f64,
    pub predicted_change: f64,
    pub actual_change: f64,
    pub confidence: f64,
    pub key_factors: Vec<String>,
    pub prediction_reason: Option<String>,
    pub interval: Option<PredictionInterval>,
    pub stress_interval: Option<PredictionInterval>,
}

/// 走步回测。
///
/// - `lookback`：每次预测使用的最小历史窗口（自动不低于 [`MIN_LOOKBACK`]）
/// - `horizon`：向前预测的交易日数，与未来真实涨跌幅对比
/// - `step`：滑动步长（≥1）
pub fn run_backtest(
    stock_code: &str,
    historical: &[HistoricalData],
    lookback: usize,
    horizon: usize,
    step: usize,
) -> Result<BacktestReport, String> {
    run_backtest_window(stock_code, historical, lookback, horizon, step, None, None)
}

/// 按预测发起日期窗口做走步回测。
///
/// 窗口只限制评估样本；每个样本仍会使用窗口之前可见的历史数据，避免丢失真实回看期。
pub fn run_backtest_window(
    stock_code: &str,
    historical: &[HistoricalData],
    lookback: usize,
    horizon: usize,
    step: usize,
    start_date: Option<NaiveDate>,
    end_date: Option<NaiveDate>,
) -> Result<BacktestReport, String> {
    run_backtest_window_with_predictor(
        stock_code,
        historical,
        lookback,
        horizon,
        step,
        start_date,
        end_date,
        predict_from_historical,
    )
}

/// 按预测发起日期窗口做走步回测，并允许调用方注入生产预测函数。
#[allow(clippy::too_many_arguments)]
pub fn run_backtest_window_with_predictor(
    stock_code: &str,
    historical: &[HistoricalData],
    lookback: usize,
    horizon: usize,
    step: usize,
    start_date: Option<NaiveDate>,
    end_date: Option<NaiveDate>,
    mut predict: impl FnMut(&PredictionRequest, &[HistoricalData]) -> Result<PredictionResponse, String>,
) -> Result<BacktestReport, String> {
    let lookback = lookback.max(MIN_LOOKBACK);
    let step = step.max(1);

    if horizon == 0 {
        return Err("预测周期 horizon 必须 ≥1".to_string());
    }
    if historical.len() < lookback + horizon + 1 {
        return Err(format!(
            "历史数据不足：需要至少 {} 条，实际 {}",
            lookback + horizon + 1,
            historical.len()
        ));
    }

    let mut samples = Vec::new();
    let mut observations = Vec::new();
    let mut interval_80_total = 0usize;
    let mut interval_80_covered = 0usize;
    let mut interval_80_width_sum = 0.0;
    let mut stress_95_total = 0usize;
    let mut stress_95_covered = 0usize;
    let mut stress_95_width_sum = 0.0;
    let mut t = lookback;
    while t + horizon <= historical.len() {
        let prediction_date = historical[t - 1].date;
        if start_date.is_some_and(|start| prediction_date < start)
            || end_date.is_some_and(|end| prediction_date > end)
        {
            t += step;
            continue;
        }

        // 仅使用预测日前可见数据，并裁到生产预测同款最大窗口。
        let visible_start = visible_history_start(t, MAX_ANALYSIS_DAYS);
        let request = PredictionRequest {
            stock_code: stock_code.to_string(),
            model_name: None,
            prediction_days: horizon,
            use_candle: false,
        };
        let response = predict(&request, &historical[visible_start..t])?;
        let prediction = response
            .predictions
            .last()
            .ok_or_else(|| "未生成预测结果".to_string())?;

        let base = historical[t - 1].close;
        let predicted_price = prediction.predicted_price;
        let predicted_change = if base > 0.0 {
            (predicted_price - base) / base * 100.0
        } else {
            0.0
        };
        let future = historical[t - 1 + horizon].close;
        let actual_change = if base > 0.0 {
            (future - base) / base * 100.0
        } else {
            0.0
        };

        if let Some(interval) = prediction.interval.as_ref() {
            interval_80_total += 1;
            interval_80_width_sum +=
                interval.upper_change_percent - interval.lower_change_percent;
            if (interval.lower_change_percent..=interval.upper_change_percent)
                .contains(&actual_change)
            {
                interval_80_covered += 1;
            }
        }
        if let Some(interval) = prediction.stress_interval.as_ref() {
            stress_95_total += 1;
            stress_95_width_sum +=
                interval.upper_change_percent - interval.lower_change_percent;
            if (interval.lower_change_percent..=interval.upper_change_percent)
                .contains(&actual_change)
            {
                stress_95_covered += 1;
            }
        }

        samples.push(BacktestSample {
            predicted_change,
            actual_change,
        });
        observations.push(BacktestObservation {
            prediction_date,
            target_date: historical[t - 1 + horizon].date,
            base_price: base,
            predicted_price,
            predicted_daily_changes: response
                .predictions
                .iter()
                .map(|prediction| prediction.predicted_change_percent)
                .collect(),
            actual_price: future,
            predicted_change,
            actual_change,
            confidence: prediction.confidence,
            key_factors: prediction.key_factors.clone().unwrap_or_default(),
            prediction_reason: prediction.prediction_reason.clone(),
            interval: prediction.interval.clone(),
            stress_interval: prediction.stress_interval.clone(),
        });
        t += step;
    }

    let mut metrics = compute_metrics(&samples);
    metrics.interval_80_total = interval_80_total;
    metrics.interval_80_coverage = ratio(interval_80_covered, interval_80_total);
    metrics.stress_95_total = stress_95_total;
    metrics.stress_95_coverage = ratio(stress_95_covered, stress_95_total);
    metrics.average_interval_80_width = average(interval_80_width_sum, interval_80_total);
    metrics.average_stress_95_width = average(stress_95_width_sum, stress_95_total);

    Ok(BacktestReport {
        stock_code: stock_code.to_string(),
        horizon,
        metrics,
        observations,
    })
}

fn ratio(count: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        count as f64 / total as f64
    }
}

fn average(sum: f64, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        sum / total as f64
    }
}

fn visible_history_start(end: usize, max_window: usize) -> usize {
    end.saturating_sub(max_window.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn synthetic_history(days: usize) -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        (0..days)
            .map(|i| {
                let close = 100.0 + i as f64 * 0.1;
                HistoricalData {
                    symbol: "test".to_string(),
                    date: start + Duration::days(i as i64),
                    open: close,
                    close,
                    high: close + 0.5,
                    low: close - 0.5,
                    volume: 10_000 + i as i64,
                    amount: close * 10_000.0,
                    amplitude: 1.0,
                    turnover_rate: 1.0,
                    volume_ratio: 1.0,
                    change_percent: 0.1,
                    change: 0.1,
                }
            })
            .collect()
    }

    #[test]
    fn test_backtest_window_end_date_is_prediction_date() {
        let historical = synthetic_history(75);
        let prediction_date = historical[64].date;

        let report = run_backtest_window(
            "test",
            &historical,
            60,
            5,
            1,
            Some(prediction_date),
            Some(prediction_date),
        )
        .unwrap();

        assert_eq!(report.observations.len(), 1);
        assert_eq!(report.observations[0].prediction_date, prediction_date);
        assert_eq!(report.observations[0].target_date, historical[69].date);
    }

    #[test]
    fn test_visible_history_start_matches_production_window_limit() {
        assert_eq!(visible_history_start(100, 3000), 0);
        assert_eq!(visible_history_start(3500, 3000), 500);
        assert_eq!(visible_history_start(10, 0), 9);
    }

    #[test]
    fn test_backtest_window_uses_injected_predictor() {
        let historical = synthetic_history(75);
        let prediction_date = historical[64].date;
        let mut calls = 0usize;

        let report = run_backtest_window_with_predictor(
            "test",
            &historical,
            60,
            5,
            1,
            Some(prediction_date),
            Some(prediction_date),
            |_request, visible_history| {
                calls += 1;
                let last = visible_history.last().unwrap();
                Ok(PredictionResponse {
                    predictions: vec![crate::prediction::types::Prediction {
                        target_date: "2026-01-01".to_string(),
                        predicted_price: last.close * 1.10,
                        predicted_change_percent: 10.0,
                        confidence: 0.8,
                        trading_signal: Some("看涨".to_string()),
                        signal_strength: Some(0.8),
                        technical_indicators: None,
                        prediction_reason: Some("injected".to_string()),
                        key_factors: None,
                        interval: None,
                        stress_interval: None,
                    }],
                    last_real_data: None,
                    diagnostics: None,
                })
            },
        )
        .unwrap();

        assert_eq!(calls, 1);
        assert!((report.observations[0].predicted_change - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_backtest_records_interval_coverage() {
        let historical = synthetic_history(75);
        let prediction_date = historical[64].date;
        let report = run_backtest_window_with_predictor(
            "test",
            &historical,
            60,
            5,
            1,
            Some(prediction_date),
            Some(prediction_date),
            |_request, visible_history| {
                let last = visible_history.last().unwrap();
                let interval = PredictionInterval {
                    confidence: 0.80,
                    lower_change_percent: -10.0,
                    upper_change_percent: 10.0,
                    lower_price: last.close * 0.9,
                    upper_price: last.close * 1.1,
                    method: "test".to_string(),
                    lookback_days: 20,
                };
                let mut stress = interval.clone();
                stress.confidence = 0.95;
                stress.lower_change_percent = -20.0;
                stress.upper_change_percent = 20.0;
                Ok(PredictionResponse {
                    predictions: vec![crate::prediction::types::Prediction {
                        target_date: "2026-01-01".to_string(),
                        predicted_price: last.close,
                        predicted_change_percent: 0.0,
                        confidence: 0.5,
                        trading_signal: Some("中性".to_string()),
                        signal_strength: Some(0.5),
                        technical_indicators: None,
                        prediction_reason: None,
                        key_factors: None,
                        interval: Some(interval),
                        stress_interval: Some(stress),
                    }],
                    last_real_data: None,
                    diagnostics: None,
                })
            },
        )
        .unwrap();

        assert_eq!(report.metrics.interval_80_total, 1);
        assert_eq!(report.metrics.stress_95_total, 1);
        assert!((report.metrics.interval_80_coverage - 1.0).abs() < 1e-9);
        assert!((report.metrics.stress_95_coverage - 1.0).abs() < 1e-9);
        assert!((report.metrics.average_interval_80_width - 20.0).abs() < 1e-9);
    }
}
