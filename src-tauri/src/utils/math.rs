//! 数学工具函数

use crate::config::constants::{A_STOCK_LIMIT_DOWN, A_STOCK_LIMIT_UP};

/// A股涨跌停限制
pub fn clamp_daily_change(change_percent: f64) -> f64 {
    change_percent.clamp(A_STOCK_LIMIT_DOWN, A_STOCK_LIMIT_UP)
}

/// 计算标准差
pub fn calculate_std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    
    variance.sqrt()
}

/// 计算移动平均
pub fn calculate_ma(values: &[f64], period: usize) -> f64 {
    if values.len() < period || period == 0 {
        return values.last().copied().unwrap_or(0.0);
    }
    
    values[values.len() - period..].iter().sum::<f64>() / period as f64
}

/// 计算指数移动平均 (EMA)
pub fn calculate_ema(values: &[f64], period: usize) -> f64 {
    if values.is_empty() || period == 0 || values.len() < period {
        return 0.0;
    }
    
    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = values[0..period].iter().sum::<f64>() / period as f64;
    
    for i in period..values.len() {
        ema = (values[i] - ema) * multiplier + ema;
    }
    
    ema
}

/// 计算 EMA 序列
pub fn calculate_ema_series(values: &[f64], period: usize) -> Vec<f64> {
    if values.is_empty() || period == 0 || values.len() < period {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(values.len() - period + 1);
    let multiplier = 2.0 / (period as f64 + 1.0);
    
    let mut ema = values[0..period].iter().sum::<f64>() / period as f64;
    result.push(ema);
    
    for i in period..values.len() {
        ema = (values[i] - ema) * multiplier + ema;
        result.push(ema);
    }
    
    result
}

/// 数据平滑处理 - 移除价格异常值
pub fn smooth_prices(prices: &[f64]) -> Vec<f64> {
    let mut smoothed = prices.to_vec();
    
    for i in 2..smoothed.len().saturating_sub(2) {
        let window: Vec<f64> = smoothed[i - 2..=i + 2].to_vec();
        let mut sorted = window.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[2];
        
        // 如果当前值与中位数相差超过20%，用中位数替换
        if (smoothed[i] - median).abs() / median > 0.2 {
            smoothed[i] = median;
        }
    }
    
    smoothed
}

/// 数据平滑处理 - 移除成交量异常值
pub fn smooth_volumes(volumes: &[i64]) -> Vec<i64> {
    let mut smoothed = volumes.to_vec();
    
    for i in 2..smoothed.len().saturating_sub(2) {
        let window: Vec<i64> = smoothed[i - 2..=i + 2].to_vec();
        let avg = window.iter().sum::<i64>() as f64 / window.len() as f64;
        
        // 如果当前值与平均值相差超过5倍，用平均值替换
        if (smoothed[i] as f64 - avg).abs() / avg > 5.0 {
            smoothed[i] = avg as i64;
        }
    }
    
    smoothed
}

/// 归一化数据到 [0, 1] 区间
pub fn normalize(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    
    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max - min;
    
    if range == 0.0 {
        return vec![0.5; values.len()];
    }
    
    values.iter().map(|&v| (v - min) / range).collect()
}

/// 标准化数据 (z-score)
pub fn standardize(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let std_dev = calculate_std_dev(values);
    
    if std_dev == 0.0 {
        return vec![0.0; values.len()];
    }
    
    values.iter().map(|&v| (v - mean) / std_dev).collect()
}

/// 计算线性回归斜率
pub fn calculate_slope(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    
    let n = values.len() as f64;
    let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xy: f64 = values.iter().enumerate().map(|(i, &v)| i as f64 * v).sum();
    let sum_x2: f64 = (0..values.len()).map(|i| (i * i) as f64).sum();
    
    (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
}

/// 限制值在指定范围内
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_ma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(calculate_ma(&values, 3), 4.0);
    }

    #[test]
    fn test_calculate_std_dev() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = calculate_std_dev(&values);
        assert!((std - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize() {
        let values = vec![0.0, 50.0, 100.0];
        let normalized = normalize(&values);
        assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
    }
}

