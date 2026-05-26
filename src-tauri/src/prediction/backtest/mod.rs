//! 真实走步回测框架
//!
//! 对历史数据做 walk-forward 回测：每个交易日仅用其之前的数据运行完整分析管线
//! （[`crate::prediction::model::inference::analyze`]），将预测涨跌幅与未来真实涨跌幅
//! 对比，量化方向准确率、误差与简单策略收益。

pub mod metrics;

use crate::db::models::HistoricalData;
use crate::prediction::model::inference::analyze;
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
    let mut t = lookback;
    while t + horizon <= historical.len() {
        // 仅使用 [0, t) 的数据（最后一条索引 t-1）
        let bundle = analyze(
            &closes[..t],
            &highs[..t],
            &lows[..t],
            &volumes[..t],
            &opens[..t],
            historical[t - 1].turnover_rate,
            horizon,
        );

        let predicted_change = bundle.professional_result.expected_change;
        let base = closes[t - 1];
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
        t += step;
    }

    Ok(BacktestReport {
        stock_code: stock_code.to_string(),
        horizon,
        metrics: compute_metrics(&samples),
    })
}
