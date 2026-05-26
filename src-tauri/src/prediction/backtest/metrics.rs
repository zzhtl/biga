//! 回测评估指标

/// 单次预测样本：预测涨跌幅 vs 实际涨跌幅（均为百分点）
#[derive(Debug, Clone, Copy)]
pub struct BacktestSample {
    pub predicted_change: f64,
    pub actual_change: f64,
}

/// 高置信度阈值：|预测涨跌幅| ≥ 该值（百分点）才计入高置信子集
pub const HIGH_CONVICTION_THRESHOLD: f64 = 1.5;

/// 回测指标汇总
#[derive(Debug, Clone)]
pub struct BacktestMetrics {
    /// 有效样本数
    pub total: usize,
    /// 方向判断正确数
    pub direction_correct: usize,
    /// 方向准确率 (0-1)
    pub direction_accuracy: f64,
    /// 涨跌幅平均绝对误差（百分点）
    pub mean_abs_error: f64,
    /// 涨跌幅均方根误差（百分点）
    pub rmse: f64,
    /// 按预测方向做多/做空的累计收益（百分点，简单加总）
    pub strategy_return: f64,
    /// 策略胜率 (0-1)
    pub win_rate: f64,
    /// 高置信子集样本数（|预测| ≥ 阈值）
    pub high_conviction_total: usize,
    /// 高置信子集方向准确率 (0-1)
    pub high_conviction_accuracy: f64,
    /// 预测为上涨的比例（检测是否单边预测）
    pub predicted_up_ratio: f64,
    /// 实际上涨的比例（数据本身的上涨基率）
    pub actual_up_ratio: f64,
    /// 朴素基准准确率 = 总是猜多数类（max(上涨率, 下跌率)）
    pub baseline_accuracy: f64,
}

impl BacktestMetrics {
    /// 相对朴素基准的超额准确率（>0 才说明模型有真实预测价值）
    pub fn edge(&self) -> f64 {
        self.direction_accuracy - self.baseline_accuracy
    }
}

impl Default for BacktestMetrics {
    fn default() -> Self {
        Self {
            total: 0,
            direction_correct: 0,
            direction_accuracy: 0.0,
            mean_abs_error: 0.0,
            rmse: 0.0,
            strategy_return: 0.0,
            win_rate: 0.0,
            high_conviction_total: 0,
            high_conviction_accuracy: 0.0,
            predicted_up_ratio: 0.0,
            actual_up_ratio: 0.0,
            baseline_accuracy: 0.0,
        }
    }
}

/// 由样本计算回测指标
pub fn compute_metrics(samples: &[BacktestSample]) -> BacktestMetrics {
    if samples.is_empty() {
        return BacktestMetrics::default();
    }

    let total = samples.len();
    let mut direction_correct = 0usize;
    let mut abs_error_sum = 0.0;
    let mut sq_error_sum = 0.0;
    let mut strategy_return = 0.0;
    let mut wins = 0usize;
    let mut hc_total = 0usize;
    let mut hc_correct = 0usize;
    let mut predicted_up = 0usize;
    let mut actual_up = 0usize;

    for s in samples {
        // 方向：同号视为正确
        let same_dir = (s.predicted_change > 0.0 && s.actual_change > 0.0)
            || (s.predicted_change < 0.0 && s.actual_change < 0.0);
        if same_dir {
            direction_correct += 1;
        }
        if s.predicted_change > 0.0 {
            predicted_up += 1;
        }
        if s.actual_change > 0.0 {
            actual_up += 1;
        }

        // 高置信子集：仅当预测幅度足够大
        if s.predicted_change.abs() >= HIGH_CONVICTION_THRESHOLD {
            hc_total += 1;
            if same_dir {
                hc_correct += 1;
            }
        }

        let err = (s.predicted_change - s.actual_change).abs();
        abs_error_sum += err;
        sq_error_sum += err * err;

        // 策略：按预测方向持仓，收益 = 方向 × 实际涨跌幅
        let position = s.predicted_change.signum();
        let trade_return = position * s.actual_change;
        strategy_return += trade_return;
        if trade_return > 0.0 {
            wins += 1;
        }
    }

    BacktestMetrics {
        total,
        direction_correct,
        direction_accuracy: direction_correct as f64 / total as f64,
        mean_abs_error: abs_error_sum / total as f64,
        rmse: (sq_error_sum / total as f64).sqrt(),
        strategy_return,
        win_rate: wins as f64 / total as f64,
        high_conviction_total: hc_total,
        high_conviction_accuracy: if hc_total > 0 {
            hc_correct as f64 / hc_total as f64
        } else {
            0.0
        },
        predicted_up_ratio: predicted_up as f64 / total as f64,
        actual_up_ratio: actual_up as f64 / total as f64,
        baseline_accuracy: {
            let up = actual_up as f64 / total as f64;
            up.max(1.0 - up)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_and_edge() {
        // 4 个样本：3 涨 1 跌 → 基率 0.75（总是猜涨）
        let samples = vec![
            BacktestSample { predicted_change: 1.0, actual_change: 2.0 },
            BacktestSample { predicted_change: -1.0, actual_change: 1.0 },
            BacktestSample { predicted_change: 1.0, actual_change: 1.0 },
            BacktestSample { predicted_change: 1.0, actual_change: -1.0 },
        ];
        let m = compute_metrics(&samples);
        assert!((m.actual_up_ratio - 0.75).abs() < 1e-9);
        assert!((m.baseline_accuracy - 0.75).abs() < 1e-9);
        // 方向正确：样本1(对) 2(错) 3(对) 4(错) = 2/4 = 0.5 → edge = 0.5-0.75 < 0
        assert!((m.direction_accuracy - 0.5).abs() < 1e-9);
        assert!(m.edge() < 0.0);
    }

    #[test]
    fn test_metrics_all_correct() {
        let samples = vec![
            BacktestSample { predicted_change: 2.0, actual_change: 3.0 },
            BacktestSample { predicted_change: -1.0, actual_change: -2.0 },
        ];
        let m = compute_metrics(&samples);
        assert_eq!(m.total, 2);
        assert_eq!(m.direction_correct, 2);
        assert!((m.direction_accuracy - 1.0).abs() < 1e-9);
        // 策略收益 = 1*3 + (-1)*(-2) = 5
        assert!((m.strategy_return - 5.0).abs() < 1e-9);
        assert!((m.win_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_direction_wrong() {
        let samples = vec![BacktestSample { predicted_change: 2.0, actual_change: -1.0 }];
        let m = compute_metrics(&samples);
        assert_eq!(m.direction_correct, 0);
        // |2 - (-1)| = 3
        assert!((m.mean_abs_error - 3.0).abs() < 1e-9);
        // 策略收益 = 1 * (-1) = -1
        assert!((m.strategy_return + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_empty() {
        let m = compute_metrics(&[]);
        assert_eq!(m.total, 0);
        assert_eq!(m.direction_accuracy, 0.0);
    }
}
