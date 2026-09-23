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

/// 标准正态分布 CDF Φ(x)。
///
/// Zelen & Severo 有理近似（A&S 26.2.17），绝对误差 < 7.5e-8——远小于金融数据本身的
/// 噪声，够用。仅作兜底：主路径应当用实测的经验残差分位，因为 A 股收益已被实测为厚尾
/// （见 `prediction::analysis::prediction_interval` 的 z 倍数大于正态值）。
pub fn normal_cdf(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_nan() {
            f64::NAN
        } else if x > 0.0 {
            1.0
        } else {
            0.0
        };
    }
    const P: f64 = 0.231_641_9;
    const B: [f64; 5] = [
        0.319_381_530,
        -0.356_563_782,
        1.781_477_937,
        -1.821_255_978,
        1.330_274_429,
    ];
    let sign_negative = x < 0.0;
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let pdf = (-0.5 * ax * ax).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let poly = B
        .iter()
        .rev()
        .fold(0.0, |acc, &b| (acc + b) * t)
        .max(0.0);
    let upper_tail = pdf * poly;
    if sign_negative {
        upper_tail
    } else {
        1.0 - upper_tail
    }
}

/// 已排序序列的经验分位（线性插值）。
///
/// `p` 取 [0, 1]。空序列返回 `None`。用线性插值而不是四舍五入取整，是因为校准表要在
/// 样本量不大的分桶上取分位，取整会引入可见的阶梯误差。
pub fn empirical_quantile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() || !p.is_finite() {
        return None;
    }
    let n = sorted.len();
    if n == 1 {
        return Some(sorted[0]);
    }
    let pos = p.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    Some(sorted[lo] + frac * (sorted[hi] - sorted[lo]))
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

    #[test]
    fn test_normal_cdf_known_points() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-9);
        assert!((normal_cdf(1.0) - 0.841_344_75).abs() < 1e-6);
        assert!((normal_cdf(-1.0) - 0.158_655_25).abs() < 1e-6);
        assert!((normal_cdf(1.959_964) - 0.975).abs() < 1e-6);
        // 对称性
        for x in [0.3_f64, 1.1, 2.7, 4.0] {
            assert!((normal_cdf(x) + normal_cdf(-x) - 1.0).abs() < 1e-7);
        }
        // 单调且不越界
        assert!(normal_cdf(-40.0) >= 0.0 && normal_cdf(40.0) <= 1.0);
        assert!(normal_cdf(0.1) < normal_cdf(0.2));
    }

    #[test]
    fn test_empirical_quantile_interpolates() {
        let sorted = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(empirical_quantile(&sorted, 0.0), Some(0.0));
        assert_eq!(empirical_quantile(&sorted, 1.0), Some(4.0));
        assert_eq!(empirical_quantile(&sorted, 0.5), Some(2.0));
        // 0.25 * 4 = 1.0 → 恰好落在第二个次序统计量上
        assert_eq!(empirical_quantile(&sorted, 0.25), Some(1.0));
        // 0.3 * 4 = 1.2 → 在 1.0 和 2.0 之间线性插值
        let q = empirical_quantile(&sorted, 0.3).unwrap();
        assert!((q - 1.2).abs() < 1e-9, "得到 {q}");
        assert_eq!(empirical_quantile(&[], 0.5), None);
        assert_eq!(empirical_quantile(&[7.0], 0.9), Some(7.0));
    }
}

