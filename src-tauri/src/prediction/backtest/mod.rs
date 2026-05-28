//! 真实走步回测框架
//!
//! 对历史数据做 walk-forward 回测：每个交易日仅用其之前的数据运行完整分析管线
//! （[`crate::prediction::model::inference::analyze`]），将预测涨跌幅与未来真实涨跌幅
//! 对比，量化方向准确率、误差与简单策略收益。

pub mod metrics;

use crate::db::models::HistoricalData;
use crate::prediction::model::inference::{
    analyze, calibrate_professional_result, AnalysisOptions,
};
use chrono::NaiveDate;
use metrics::{compute_metrics, BacktestMetrics, BacktestSample};

/// 回测最小回看窗口（分析管线要求 ≥60 个交易日）
pub const MIN_LOOKBACK: usize = 60;
const BACKTEST_ANALYSIS_WINDOW: usize = 800;

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
    pub actual_price: f64,
    pub predicted_change: f64,
    pub actual_change: f64,
    pub confidence: f64,
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

    let closes: Vec<f64> = historical.iter().map(|h| h.close).collect();
    let highs: Vec<f64> = historical.iter().map(|h| h.high).collect();
    let lows: Vec<f64> = historical.iter().map(|h| h.low).collect();
    let volumes: Vec<i64> = historical.iter().map(|h| h.volume).collect();
    let opens: Vec<f64> = historical.iter().map(|h| h.open).collect();

    let mut samples = Vec::new();
    let mut observations = Vec::new();
    let mut t = lookback;
    while t + horizon <= historical.len() {
        let prediction_date = historical[t - 1].date;
        if start_date.is_some_and(|start| prediction_date < start)
            || end_date.is_some_and(|end| prediction_date > end)
        {
            t += step;
            continue;
        }

        // 仅使用 [0, t) 的数据（最后一条索引 t-1）
        let analysis_start = t.saturating_sub(BACKTEST_ANALYSIS_WINDOW);
        let bundle = analyze(
            &closes[analysis_start..t],
            &highs[analysis_start..t],
            &lows[analysis_start..t],
            &volumes[analysis_start..t],
            &opens[analysis_start..t],
            AnalysisOptions {
                turnover_rate: historical[t - 1].turnover_rate,
                prediction_days: horizon,
                stock_code: Some(stock_code),
            },
        );
        let mut professional_result = bundle.professional_result;
        calibrate_professional_result(
            &historical[..t],
            &mut professional_result,
            horizon,
            Some(stock_code),
        );

        let predicted_change = professional_result.expected_change;
        let base = closes[t - 1];
        let predicted_price = base * (1.0 + predicted_change / 100.0);
        let future = closes[t - 1 + horizon];
        let actual_change = if base > 0.0 {
            (future - base) / base * 100.0
        } else {
            0.0
        };

        samples.push(BacktestSample {
            predicted_change,
            actual_change,
        });
        observations.push(BacktestObservation {
            prediction_date,
            target_date: historical[t - 1 + horizon].date,
            base_price: base,
            predicted_price,
            actual_price: future,
            predicted_change,
            actual_change,
            confidence: professional_result.confidence,
        });
        t += step;
    }

    Ok(BacktestReport {
        stock_code: stock_code.to_string(),
        horizon,
        metrics: compute_metrics(&samples),
        observations,
    })
}
