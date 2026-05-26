//! 端到端集成测试：用内嵌 fixture 历史数据走完整分析管线与回测，不依赖在线 API / 数据库。

use biga_lib::db::models::HistoricalData;
use biga_lib::prediction::backtest::run_backtest;
use biga_lib::prediction::model::inference::analyze;
use chrono::{Duration, NaiveDate};

/// 构造带趋势 + 周期波动的合成历史数据
fn fixture(n: usize) -> Vec<HistoricalData> {
    let start = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
    (0..n)
        .map(|i| {
            let t = i as f64;
            // 线性上行 + 正弦波动
            let close = 20.0 + t * 0.05 + (t / 6.0).sin() * 1.5;
            let prev = if i == 0 {
                close
            } else {
                20.0 + (t - 1.0) * 0.05 + ((t - 1.0) / 6.0).sin() * 1.5
            };
            let high = close.max(prev) + 0.3;
            let low = close.min(prev) - 0.3;
            let change = close - prev;
            HistoricalData {
                symbol: "test".to_string(),
                date: start + Duration::days(i as i64),
                open: prev,
                close,
                high,
                low,
                volume: 1_000_000 + (i as i64 % 11) * 50_000,
                amount: close * 1_000_000.0,
                amplitude: (high - low) / prev * 100.0,
                turnover_rate: 3.0 + (i as f64 % 5.0),
                volume_ratio: 1.0,
                change_percent: change / prev * 100.0,
                change,
            }
        })
        .collect()
}

#[test]
fn test_analyze_pipeline_runs() {
    let h = fixture(120);
    let closes: Vec<f64> = h.iter().map(|x| x.close).collect();
    let highs: Vec<f64> = h.iter().map(|x| x.high).collect();
    let lows: Vec<f64> = h.iter().map(|x| x.low).collect();
    let volumes: Vec<i64> = h.iter().map(|x| x.volume).collect();
    let opens: Vec<f64> = h.iter().map(|x| x.open).collect();

    let bundle = analyze(&closes, &highs, &lows, &volumes, &opens, 3.5, 5);

    // 量比/换手率应被填充到指标中
    assert!(bundle.tech_indicators.volume_ratio > 0.0);
    assert!((bundle.tech_indicators.turnover_rate - 3.5).abs() < 1e-9);
    // 专业引擎应产出有限的预期涨跌幅与置信度
    assert!(bundle.professional_result.expected_change.is_finite());
    assert!((0.0..=1.0).contains(&bundle.professional_result.confidence));
}

#[test]
fn test_backtest_end_to_end() {
    let h = fixture(220);
    let report = run_backtest("test", &h, 60, 5, 5).expect("回测应成功");

    assert!(report.metrics.total > 0, "应产生回测样本");
    assert!((0.0..=1.0).contains(&report.metrics.direction_accuracy));
    assert!(report.metrics.mean_abs_error.is_finite());
    assert_eq!(report.horizon, 5);
}

#[test]
fn test_backtest_insufficient_data() {
    let h = fixture(50); // 少于 lookback + horizon
    let result = run_backtest("test", &h, 60, 5, 1);
    assert!(result.is_err(), "数据不足应返回错误");
}
