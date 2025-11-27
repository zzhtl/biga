//! 布林带指标计算
//! 
//! Bollinger Bands
//! - 中轨 = N日移动平均线
//! - 上轨 = 中轨 + K × N日标准差
//! - 下轨 = 中轨 - K × N日标准差

use crate::utils::math::calculate_std_dev;
use serde::{Deserialize, Serialize};

/// 布林带数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BollingerBands {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
}

/// 计算布林带
pub fn calculate_bollinger_bands(prices: &[f64], period: usize, std_dev_multiplier: f64) -> BollingerBands {
    if prices.len() < period {
        let avg = if prices.is_empty() {
            0.0
        } else {
            prices.iter().sum::<f64>() / prices.len() as f64
        };
        return BollingerBands {
            upper: avg,
            middle: avg,
            lower: avg,
        };
    }
    
    let recent = &prices[prices.len() - period..];
    let middle = recent.iter().sum::<f64>() / period as f64;
    let std_dev = calculate_std_dev(recent);
    
    let upper = middle + std_dev_multiplier * std_dev;
    let lower = middle - std_dev_multiplier * std_dev;
    
    BollingerBands { upper, middle, lower }
}

/// 计算价格在布林带中的位置
/// 返回值：0 = 下轨，0.5 = 中轨，1 = 上轨
pub fn calculate_bollinger_position(prices: &[f64], current_price: f64) -> f64 {
    if prices.len() < 20 {
        return 0.0;
    }
    
    let bands = calculate_bollinger_bands(prices, 20, 2.0);
    
    if bands.upper == bands.lower {
        0.0
    } else {
        (current_price - bands.lower) / (bands.upper - bands.lower) - 0.5
    }
}

/// 计算布林带宽度
pub fn calculate_bandwidth(bands: &BollingerBands) -> f64 {
    if bands.middle == 0.0 {
        0.0
    } else {
        (bands.upper - bands.lower) / bands.middle * 100.0
    }
}

/// 判断是否触及上轨
pub fn is_touching_upper(current_price: f64, upper: f64, tolerance: f64) -> bool {
    current_price >= upper * (1.0 - tolerance)
}

/// 判断是否触及下轨
pub fn is_touching_lower(current_price: f64, lower: f64, tolerance: f64) -> bool {
    current_price <= lower * (1.0 + tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bollinger_bands() {
        let prices = vec![10.0, 11.0, 10.5, 11.5, 10.0, 12.0, 11.0, 11.5, 12.0, 11.0,
                         10.5, 11.0, 12.0, 11.5, 10.5, 11.0, 12.5, 11.0, 10.5, 11.0];
        
        let bands = calculate_bollinger_bands(&prices, 20, 2.0);
        
        assert!(bands.upper > bands.middle);
        assert!(bands.middle > bands.lower);
    }

    #[test]
    fn test_bollinger_position() {
        let prices = vec![10.0; 20];
        let position = calculate_bollinger_position(&prices, 10.0);
        assert!((position - 0.0).abs() < 0.01);
    }
}

