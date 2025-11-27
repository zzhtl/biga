//! 支撑阻力位分析模块

use serde::{Deserialize, Serialize};

/// 支撑阻力位
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportResistance {
    pub support_levels: Vec<f64>,
    pub resistance_levels: Vec<f64>,
    pub current_position: String,
}

/// 计算支撑阻力位
pub fn calculate_support_resistance(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    current_price: f64,
) -> SupportResistance {
    if prices.len() < 20 {
        return SupportResistance {
            support_levels: Vec::new(),
            resistance_levels: Vec::new(),
            current_position: "数据不足".to_string(),
        };
    }
    
    let n = prices.len();
    let mut all_levels = Vec::new();
    
    // 1. 计算均线支撑/阻力
    let calc_ma = |window: usize| -> f64 {
        if n >= window {
            prices[n - window..].iter().sum::<f64>() / window as f64
        } else {
            current_price
        }
    };
    
    all_levels.push(calc_ma(5));
    all_levels.push(calc_ma(10));
    all_levels.push(calc_ma(20));
    all_levels.push(calc_ma(60));
    
    // 2. 历史高低点
    let lookback = n.min(60);
    let recent_high = highs[n - lookback..].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let recent_low = lows[n - lookback..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    all_levels.push(recent_high);
    all_levels.push(recent_low);
    
    // 3. 斐波那契回撤位
    let fib_range = recent_high - recent_low;
    all_levels.push(recent_high - fib_range * 0.382);
    all_levels.push(recent_high - fib_range * 0.500);
    all_levels.push(recent_high - fib_range * 0.618);
    
    // 去重并排序
    all_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_levels.dedup_by(|a, b| (*a - *b).abs() < current_price * 0.01);
    
    // 分类支撑和阻力
    let mut support_levels: Vec<f64> = all_levels.iter()
        .filter(|&&l| l < current_price && l > current_price * 0.85)
        .copied()
        .collect();
    
    let mut resistance_levels: Vec<f64> = all_levels.iter()
        .filter(|&&l| l > current_price && l < current_price * 1.15)
        .copied()
        .collect();
    
    // 按距离排序
    support_levels.sort_by(|a, b| (current_price - a).partial_cmp(&(current_price - b)).unwrap());
    resistance_levels.sort_by(|a, b| (a - current_price).partial_cmp(&(b - current_price)).unwrap());
    
    support_levels.truncate(5);
    resistance_levels.truncate(5);
    
    // 当前位置描述
    let current_position = if !support_levels.is_empty() && !resistance_levels.is_empty() {
        let to_support = ((current_price - support_levels[0]) / current_price * 100.0).abs();
        let to_resistance = ((resistance_levels[0] - current_price) / current_price * 100.0).abs();
        
        if to_support < 2.0 {
            "接近关键支撑".to_string()
        } else if to_resistance < 2.0 {
            "接近关键压力".to_string()
        } else if to_support < to_resistance {
            format!("中性偏下，距支撑{:.2}%", to_support)
        } else {
            format!("中性偏上，距压力{:.2}%", to_resistance)
        }
    } else {
        "中性区域".to_string()
    };
    
    SupportResistance {
        support_levels,
        resistance_levels,
        current_position,
    }
}

/// 计算价格与支撑阻力位的关系
pub fn calculate_sr_influence(
    current_price: f64,
    support: Option<f64>,
    resistance: Option<f64>,
) -> f64 {
    let support_influence = support
        .map(|s| (current_price - s) / current_price)
        .unwrap_or(0.0);
    
    let resistance_influence = resistance
        .map(|r| (r - current_price) / current_price)
        .unwrap_or(0.0);
    
    ((resistance_influence - support_influence) * 0.5).clamp(-0.03, 0.03)
}

/// 判断是否突破阻力位
pub fn is_breakout(current_price: f64, resistance: f64, volume_ratio: f64) -> bool {
    current_price > resistance * 1.01 && volume_ratio > 1.2
}

/// 判断是否跌破支撑位
pub fn is_breakdown(current_price: f64, support: f64, volume_ratio: f64) -> bool {
    current_price < support * 0.99 && volume_ratio > 1.2
}

