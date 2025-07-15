use serde::{Deserialize, Serialize};
use crate::stock_prediction::backtest::BacktestReport;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub dropout: f64,
    pub hidden_layers: Vec<usize>,
    pub lookback_days: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub config: HyperparameterConfig,
    pub expected_accuracy: f64,
    pub confidence_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterOptimizer {
    pub current_config: HyperparameterConfig,
    pub optimization_history: Vec<OptimizationResult>,
}

impl HyperparameterOptimizer {
    pub fn new(current_config: HyperparameterConfig) -> Self {
        Self {
            current_config,
            optimization_history: Vec::new(),
        }
    }
    
    /// 基于回测结果建议优化参数
    pub fn suggest_optimization(
        &mut self,
        backtest_report: &BacktestReport,
    ) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
        let mut optimized_config = self.current_config.clone();
        let mut expected_improvement = 0.0;
        
        // 1. 学习率优化
        if should_adjust_learning_rate(backtest_report) {
            optimized_config.learning_rate = optimize_learning_rate(
                &self.current_config,
                backtest_report,
            )?;
            expected_improvement += 0.05;
        }
        
        // 2. 批处理大小优化
        if should_adjust_batch_size(backtest_report) {
            optimized_config.batch_size = optimize_batch_size(
                &self.current_config,
                backtest_report,
            )?;
            expected_improvement += 0.03;
        }
        
        // 3. 训练轮数优化
        if should_adjust_epochs(backtest_report) {
            optimized_config.epochs = optimize_epochs(
                &self.current_config,
                backtest_report,
            )?;
            expected_improvement += 0.04;
        }
        
        // 4. Dropout率优化
        if should_adjust_dropout(backtest_report) {
            optimized_config.dropout = optimize_dropout(
                &self.current_config,
                backtest_report,
            )?;
            expected_improvement += 0.06;
        }
        
        // 5. 网络结构优化
        if should_adjust_architecture(backtest_report) {
            optimized_config.hidden_layers = optimize_architecture(
                &self.current_config,
                backtest_report,
            )?;
            expected_improvement += 0.08;
        }
        
        // 6. 历史数据窗口优化
        if should_adjust_lookback(backtest_report) {
            optimized_config.lookback_days = optimize_lookback_days(
                &self.current_config,
                backtest_report,
            )?;
            expected_improvement += 0.07;
        }
        
        let expected_accuracy = backtest_report.overall_price_accuracy + expected_improvement;
        let confidence_score = calculate_confidence_score(backtest_report, expected_improvement);
        
        let result = OptimizationResult {
            config: optimized_config,
            expected_accuracy,
            confidence_score,
        };
        
        self.optimization_history.push(result.clone());
        Ok(result)
    }
}

fn should_adjust_learning_rate(backtest_report: &BacktestReport) -> bool {
    // 如果准确率趋势不稳定或者误差较大，需要调整学习率
    backtest_report.average_prediction_error > 0.06 || 
    is_accuracy_unstable(&backtest_report.accuracy_trend)
}

fn optimize_learning_rate(
    current_config: &HyperparameterConfig,
    backtest_report: &BacktestReport,
) -> Result<f64, Box<dyn std::error::Error>> {
    let current_lr = current_config.learning_rate;
    
    // 基于回测表现调整学习率
    if backtest_report.average_prediction_error > 0.08 {
        // 误差太大，降低学习率
        Ok(current_lr * 0.7)
    } else if backtest_report.average_prediction_error < 0.03 && 
              backtest_report.overall_price_accuracy < 0.75 {
        // 误差小但准确率不高，可能学习率太小
        Ok(current_lr * 1.3)
    } else {
        // 微调
        Ok(current_lr * 0.9)
    }
}

fn should_adjust_batch_size(backtest_report: &BacktestReport) -> bool {
    // 如果训练不稳定或者收敛慢，考虑调整批处理大小
    backtest_report.overall_price_accuracy < 0.65 ||
    is_accuracy_unstable(&backtest_report.accuracy_trend)
}

fn optimize_batch_size(
    current_config: &HyperparameterConfig,
    backtest_report: &BacktestReport,
) -> Result<usize, Box<dyn std::error::Error>> {
    let current_batch = current_config.batch_size;
    
    if backtest_report.overall_price_accuracy < 0.6 {
        // 准确率低，尝试较小的批处理大小以提高梯度更新频率
        Ok((current_batch / 2).max(8))
    } else if is_accuracy_unstable(&backtest_report.accuracy_trend) {
        // 不稳定，尝试较大的批处理大小以平滑梯度
        Ok((current_batch * 2).min(128))
    } else {
        Ok(current_batch)
    }
}

fn should_adjust_epochs(backtest_report: &BacktestReport) -> bool {
    // 基于准确率趋势判断是否需要调整训练轮数
    is_accuracy_declining(&backtest_report.accuracy_trend) ||
    backtest_report.overall_price_accuracy < 0.7
}

fn optimize_epochs(
    current_config: &HyperparameterConfig,
    backtest_report: &BacktestReport,
) -> Result<usize, Box<dyn std::error::Error>> {
    let current_epochs = current_config.epochs;
    
    if is_accuracy_declining(&backtest_report.accuracy_trend) {
        // 准确率下降，可能过拟合，减少训练轮数
        Ok((current_epochs as f64 * 0.8) as usize)
    } else if backtest_report.overall_price_accuracy < 0.65 {
        // 准确率低，可能欠拟合，增加训练轮数
        Ok((current_epochs as f64 * 1.3) as usize)
    } else {
        Ok(current_epochs)
    }
}

fn should_adjust_dropout(backtest_report: &BacktestReport) -> bool {
    // 基于过拟合迹象判断是否需要调整dropout
    is_accuracy_declining(&backtest_report.accuracy_trend) ||
    backtest_report.average_prediction_error > 0.05
}

fn optimize_dropout(
    current_config: &HyperparameterConfig,
    backtest_report: &BacktestReport,
) -> Result<f64, Box<dyn std::error::Error>> {
    let current_dropout = current_config.dropout;
    
    if is_accuracy_declining(&backtest_report.accuracy_trend) {
        // 可能过拟合，增加dropout
        Ok((current_dropout + 0.1).min(0.5))
    } else if backtest_report.overall_price_accuracy < 0.6 {
        // 准确率低，可能dropout太高
        Ok((current_dropout - 0.05).max(0.0))
    } else {
        Ok(current_dropout)
    }
}

fn should_adjust_architecture(backtest_report: &BacktestReport) -> bool {
    // 基于整体表现判断是否需要调整网络结构
    backtest_report.overall_price_accuracy < 0.65 ||
    backtest_report.average_prediction_error > 0.06
}

fn optimize_architecture(
    current_config: &HyperparameterConfig,
    backtest_report: &BacktestReport,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut new_layers = current_config.hidden_layers.clone();
    
    if backtest_report.overall_price_accuracy < 0.6 {
        // 准确率低，增加网络复杂度
        if new_layers.len() < 3 {
            new_layers.push(64);
        } else {
            // 增加现有层的神经元数量
            for layer in &mut new_layers {
                *layer = (*layer as f64 * 1.2) as usize;
            }
        }
    } else if is_accuracy_declining(&backtest_report.accuracy_trend) {
        // 可能过拟合，减少复杂度
        if new_layers.len() > 1 {
            new_layers.pop();
        } else {
            // 减少神经元数量
            for layer in &mut new_layers {
                *layer = (*layer as f64 * 0.8) as usize;
            }
        }
    }
    
    Ok(new_layers)
}

fn should_adjust_lookback(backtest_report: &BacktestReport) -> bool {
    // 基于时间序列相关性判断是否需要调整历史窗口
    backtest_report.overall_direction_accuracy < 0.7 ||
    has_temporal_correlation_issues(backtest_report)
}

fn optimize_lookback_days(
    current_config: &HyperparameterConfig,
    backtest_report: &BacktestReport,
) -> Result<usize, Box<dyn std::error::Error>> {
    let current_lookback = current_config.lookback_days;
    
    if backtest_report.overall_direction_accuracy < 0.65 {
        // 方向预测不准确，可能需要更长的历史窗口
        Ok((current_lookback as f64 * 1.25) as usize)
    } else if backtest_report.average_prediction_error > 0.06 {
        // 预测误差大，可能历史窗口太长造成噪声
        Ok((current_lookback as f64 * 0.8) as usize)
    } else {
        Ok(current_lookback)
    }
}

fn is_accuracy_unstable(accuracy_trend: &[f64]) -> bool {
    if accuracy_trend.len() < 5 {
        return false;
    }
    
    let variance = calculate_variance(accuracy_trend);
    variance > 0.05 // 方差大于5%认为不稳定
}

fn is_accuracy_declining(accuracy_trend: &[f64]) -> bool {
    if accuracy_trend.len() < 3 {
        return false;
    }
    
    let recent_avg = accuracy_trend[accuracy_trend.len()-3..].iter().sum::<f64>() / 3.0;
    let early_avg = accuracy_trend[..3].iter().sum::<f64>() / 3.0;
    
    recent_avg < early_avg - 0.05
}

fn has_temporal_correlation_issues(backtest_report: &BacktestReport) -> bool {
    // 简化判断：如果某些时间段的准确率明显偏低
    let avg_accuracy = backtest_report.daily_accuracy.iter()
        .map(|day| day.price_accuracy)
        .sum::<f64>() / backtest_report.daily_accuracy.len() as f64;
    
    backtest_report.daily_accuracy.iter()
        .filter(|day| day.price_accuracy < avg_accuracy - 0.2)
        .count() > backtest_report.daily_accuracy.len() / 4
}

fn calculate_variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    variance
}

fn calculate_confidence_score(backtest_report: &BacktestReport, expected_improvement: f64) -> f64 {
    let base_confidence = backtest_report.overall_price_accuracy;
    let stability_factor = 1.0 - calculate_variance(&backtest_report.accuracy_trend);
    let improvement_factor = (expected_improvement * 2.0).min(1.0);
    
    (base_confidence + stability_factor + improvement_factor) / 3.0
} 