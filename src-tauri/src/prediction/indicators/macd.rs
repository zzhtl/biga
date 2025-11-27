//! MACD 指标计算
//! 
//! MACD（Moving Average Convergence Divergence）
//! - DIF = EMA(12) - EMA(26)
//! - DEA = EMA(DIF, 9)
//! - MACD柱 = 2 × (DIF - DEA)

use crate::utils::math::{calculate_ema, calculate_ema_series};
use serde::{Deserialize, Serialize};

/// MACD 数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacdData {
    pub dif: f64,
    pub dea: f64,
    pub histogram: f64,
}

/// 计算 MACD (DIF 值)
pub fn calculate_macd(prices: &[f64]) -> f64 {
    if prices.len() < 26 {
        return 0.0;
    }
    
    let ema12 = calculate_ema(prices, 12);
    let ema26 = calculate_ema(prices, 26);
    ema12 - ema26
}

/// 计算完整 MACD 指标 (DIF, DEA, MACD柱)
pub fn calculate_macd_full(prices: &[f64]) -> (f64, f64, f64) {
    if prices.len() < 26 {
        return (0.0, 0.0, 0.0);
    }
    
    // 计算 EMA 序列
    let ema12_series = calculate_ema_series(prices, 12);
    let ema26_series = calculate_ema_series(prices, 26);
    
    if ema12_series.is_empty() || ema26_series.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    
    // 计算 DIF 序列（对齐两个 EMA 序列）
    let mut dif_series = Vec::new();
    let offset = 14; // 26 - 12 = 14
    
    for i in 0..ema26_series.len() {
        let ema12_idx = offset + i;
        if ema12_idx < ema12_series.len() {
            let dif = ema12_series[ema12_idx] - ema26_series[i];
            dif_series.push(dif);
        }
    }
    
    if dif_series.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    
    let dif = *dif_series.last().unwrap();
    
    // 计算 DEA (DIF 的 9 日 EMA)
    let dea = if dif_series.len() >= 9 {
        calculate_ema(&dif_series, 9)
    } else {
        dif
    };
    
    // 计算 MACD 柱状图
    let histogram = 2.0 * (dif - dea);
    
    (dif, dea, histogram)
}

/// 计算 MACD 数据结构
pub fn calculate_macd_data(prices: &[f64]) -> MacdData {
    let (dif, dea, histogram) = calculate_macd_full(prices);
    MacdData { dif, dea, histogram }
}

/// 判断 MACD 金叉
pub fn is_golden_cross(prev_dif: f64, prev_dea: f64, curr_dif: f64, curr_dea: f64) -> bool {
    prev_dif <= prev_dea && curr_dif > curr_dea
}

/// 判断 MACD 死叉
pub fn is_death_cross(prev_dif: f64, prev_dea: f64, curr_dif: f64, curr_dea: f64) -> bool {
    prev_dif >= prev_dea && curr_dif < curr_dea
}

/// 判断 MACD 上穿零轴
pub fn is_zero_cross_up(prev_histogram: f64, curr_histogram: f64) -> bool {
    prev_histogram <= 0.0 && curr_histogram > 0.0
}

/// 判断 MACD 下穿零轴
pub fn is_zero_cross_down(prev_histogram: f64, curr_histogram: f64) -> bool {
    prev_histogram >= 0.0 && curr_histogram < 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macd_calculation() {
        let prices: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let (dif, dea, hist) = calculate_macd_full(&prices);
        
        // 上涨趋势，MACD 应该为正
        assert!(dif > 0.0);
        assert!(hist != 0.0);
    }

    #[test]
    fn test_golden_cross() {
        assert!(is_golden_cross(-1.0, 0.0, 0.5, 0.0));
        assert!(!is_golden_cross(1.0, 0.0, 0.5, 0.0));
    }
}

