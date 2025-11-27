//! KDJ 指标计算
//! 
//! KDJ（随机指标）
//! - RSV = (收盘价 - N日最低价) / (N日最高价 - N日最低价) × 100
//! - K = 2/3 × 前K + 1/3 × RSV
//! - D = 2/3 × 前D + 1/3 × K
//! - J = 3K - 2D

use serde::{Deserialize, Serialize};

/// KDJ 数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KdjData {
    pub k: f64,
    pub d: f64,
    pub j: f64,
}

/// 计算 KDJ 指标
pub fn calculate_kdj(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: usize,
) -> (f64, f64, f64) {
    if highs.len() < period || lows.len() < period || closes.len() < period {
        return (50.0, 50.0, 50.0);
    }
    
    let len = highs.len();
    let mut k = 50.0;
    let mut d = 50.0;
    
    for i in period..=len {
        let start = i.saturating_sub(period);
        let end = i;
        
        let highest = highs[start..end].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = lows[start..end].iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if highest == lowest {
            continue;
        }
        
        let close = closes[end - 1];
        let rsv = (close - lowest) / (highest - lowest) * 100.0;
        
        k = (2.0 / 3.0) * k + (1.0 / 3.0) * rsv;
        d = (2.0 / 3.0) * d + (1.0 / 3.0) * k;
    }
    
    let j = 3.0 * k - 2.0 * d;
    
    (k, d, j)
}

/// 计算 KDJ 数据结构
pub fn calculate_kdj_data(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> KdjData {
    let (k, d, j) = calculate_kdj(highs, lows, closes, period);
    KdjData { k, d, j }
}

/// 计算随机指标 K 值
pub fn calculate_stochastic_k(prices: &[f64], current_price: f64) -> f64 {
    if prices.is_empty() {
        return 0.5;
    }
    
    let highest = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let lowest = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    if highest == lowest {
        0.5
    } else {
        (current_price - lowest) / (highest - lowest)
    }
}

/// 判断 KDJ 金叉
pub fn is_kdj_golden_cross(prev_k: f64, prev_d: f64, curr_k: f64, curr_d: f64) -> bool {
    prev_k <= prev_d && curr_k > curr_d
}

/// 判断 KDJ 死叉
pub fn is_kdj_death_cross(prev_k: f64, prev_d: f64, curr_k: f64, curr_d: f64) -> bool {
    prev_k >= prev_d && curr_k < curr_d
}

/// 判断超买
pub fn is_overbought(j: f64, threshold: f64) -> bool {
    j > threshold
}

/// 判断超卖
pub fn is_oversold(j: f64, threshold: f64) -> bool {
    j < threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kdj_calculation() {
        let highs = vec![10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0];
        let lows = vec![9.0, 10.0, 10.5, 10.0, 11.0, 11.5, 12.0, 12.5, 13.0];
        let closes = vec![9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5];
        
        let (k, d, j) = calculate_kdj(&highs, &lows, &closes, 9);
        
        assert!(k >= 0.0 && k <= 100.0);
        assert!(d >= 0.0 && d <= 100.0);
    }

    #[test]
    fn test_kdj_cross() {
        assert!(is_kdj_golden_cross(30.0, 40.0, 45.0, 40.0));
        assert!(is_kdj_death_cross(60.0, 50.0, 45.0, 50.0));
    }
}

