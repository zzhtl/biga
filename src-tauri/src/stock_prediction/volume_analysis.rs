/// 增强的量价关系分析
/// 基于日K数据计算OBV、量比等指标

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeAnalysisResult {
    pub obv: Vec<f64>,                  // 能量潮指标
    pub volume_ratio: Vec<f64>,         // 量比（当日量/近5日均量）
    pub volume_trend: String,           // 量能趋势：放量/缩量/平量
    pub volume_price_sync: bool,        // 量价配合是否良好
    pub abnormal_volume_days: Vec<usize>, // 异常放量的日期索引
    pub accumulation_signal: f64,       // 吸筹信号强度 0-100
}

/// 计算OBV（能量潮）指标
/// OBV原理：价涨则加量，价跌则减量
pub fn calculate_obv(prices: &[f64], volumes: &[i64]) -> Vec<f64> {
    if prices.is_empty() || volumes.is_empty() {
        return Vec::new();
    }
    
    let mut obv = Vec::with_capacity(prices.len());
    obv.push(volumes[0] as f64);
    
    for i in 1..prices.len() {
        if prices[i] > prices[i - 1] {
            // 价格上涨，OBV加上成交量
            obv.push(obv[i - 1] + volumes[i] as f64);
        } else if prices[i] < prices[i - 1] {
            // 价格下跌，OBV减去成交量
            obv.push(obv[i - 1] - volumes[i] as f64);
        } else {
            // 价格不变，OBV保持
            obv.push(obv[i - 1]);
        }
    }
    
    obv
}

/// 计算量比（当日成交量相对于近期平均成交量的比值）
pub fn calculate_volume_ratio(volumes: &[i64], window: usize) -> Vec<f64> {
    let mut ratios = Vec::with_capacity(volumes.len());
    
    for i in 0..volumes.len() {
        if i < window {
            // 数据不足，使用已有数据计算平均
            let avg = volumes[0..=i].iter().sum::<i64>() as f64 / (i + 1) as f64;
            let ratio = if avg > 0.0 {
                volumes[i] as f64 / avg
            } else {
                1.0
            };
            ratios.push(ratio);
        } else {
            // 计算过去window天的平均成交量
            let avg = volumes[i - window..i].iter().sum::<i64>() as f64 / window as f64;
            let ratio = if avg > 0.0 {
                volumes[i] as f64 / avg
            } else {
                1.0
            };
            ratios.push(ratio);
        }
    }
    
    ratios
}

/// 检测异常放量（量比 > 2.0 视为异常放量）
pub fn detect_abnormal_volume(volume_ratios: &[f64], threshold: f64) -> Vec<usize> {
    volume_ratios
        .iter()
        .enumerate()
        .filter(|(_, &ratio)| ratio >= threshold)
        .map(|(idx, _)| idx)
        .collect()
}

/// 判断量价配合情况
/// 理想情况：价涨量增、价跌量缩
pub fn check_volume_price_sync(
    prices: &[f64], 
    volumes: &[i64], 
    lookback: usize
) -> bool {
    if prices.len() < lookback || volumes.len() < lookback {
        return false;
    }
    
    let start = prices.len() - lookback;
    let mut sync_count = 0;
    
    for i in (start + 1)..prices.len() {
        let price_up = prices[i] > prices[i - 1];
        let volume_up = volumes[i] > volumes[i - 1];
        
        // 价涨量增 或 价跌量缩 = 配合良好
        if (price_up && volume_up) || (!price_up && !volume_up) {
            sync_count += 1;
        }
    }
    
    // 超过60%的时间量价配合良好
    sync_count as f64 / (lookback - 1) as f64 > 0.6
}

/// 判断量能趋势
pub fn identify_volume_trend(volumes: &[i64], window: usize) -> String {
    if volumes.len() < window * 2 {
        return "数据不足".to_string();
    }
    
    let n = volumes.len();
    let recent_avg = volumes[n - window..].iter().sum::<i64>() as f64 / window as f64;
    let previous_avg = volumes[n - window * 2..n - window].iter().sum::<i64>() as f64 / window as f64;
    
    let change_pct = (recent_avg - previous_avg) / previous_avg;
    
    if change_pct > 0.2 {
        "放量".to_string()
    } else if change_pct < -0.2 {
        "缩量".to_string()
    } else {
        "平量".to_string()
    }
}

/// 检测主力吸筹信号
/// 特征：缩量横盘 + OBV持续上升
pub fn detect_accumulation_signal(
    prices: &[f64],
    volumes: &[i64],
    obv: &[f64],
    window: usize
) -> f64 {
    if prices.len() < window || obv.len() < window {
        return 0.0;
    }
    
    let n = prices.len();
    let recent_prices = &prices[n - window..];
    let recent_obv = &obv[n - window..];
    let recent_volumes = &volumes[n - window..];
    
    // 1. 价格波动小（横盘）
    let price_max = recent_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let price_min = recent_prices.iter().cloned().fold(f64::INFINITY, f64::min);
    let price_volatility = (price_max - price_min) / price_min;
    let low_volatility = price_volatility < 0.08; // 波动<8%
    
    // 2. 成交量萎缩
    let avg_volume = recent_volumes.iter().sum::<i64>() as f64 / window as f64;
    let prev_avg = if n >= window * 2 {
        volumes[n - window * 2..n - window].iter().sum::<i64>() as f64 / window as f64
    } else {
        avg_volume
    };
    let volume_shrinking = avg_volume < prev_avg * 0.8;
    
    // 3. OBV持续上升
    let obv_rising = recent_obv.last().unwrap() > recent_obv.first().unwrap();
    
    // 综合评分
    let mut score = 0.0;
    if low_volatility { score += 30.0; }
    if volume_shrinking { score += 30.0; }
    if obv_rising { score += 40.0; }
    
    score
}

/// 综合量价分析
pub fn analyze_volume_price(
    prices: &[f64],
    volumes: &[i64],
) -> VolumeAnalysisResult {
    // 计算OBV
    let obv = calculate_obv(prices, volumes);
    
    // 计算量比
    let volume_ratio = calculate_volume_ratio(volumes, 5);
    
    // 检测异常放量
    let abnormal_volume_days = detect_abnormal_volume(&volume_ratio, 2.0);
    
    // 判断量价配合
    let volume_price_sync = check_volume_price_sync(prices, volumes, 10);
    
    // 判断量能趋势
    let volume_trend = identify_volume_trend(volumes, 10);
    
    // 检测吸筹信号
    let accumulation_signal = detect_accumulation_signal(prices, volumes, &obv, 20);
    
    VolumeAnalysisResult {
        obv,
        volume_ratio,
        volume_trend,
        volume_price_sync,
        abnormal_volume_days,
        accumulation_signal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_obv_calculation() {
        let prices = vec![10.0, 11.0, 10.5, 11.5, 11.0];
        let volumes = vec![1000, 1200, 800, 1500, 1000];
        
        let obv = calculate_obv(&prices, &volumes);
        
        assert_eq!(obv.len(), 5);
        assert_eq!(obv[0], 1000.0);
        assert_eq!(obv[1], 2200.0); // 价涨，加量
        assert_eq!(obv[2], 1400.0); // 价跌，减量
    }
    
    #[test]
    fn test_volume_ratio() {
        let volumes = vec![1000, 1000, 1000, 1000, 1000, 2000];
        let ratios = calculate_volume_ratio(&volumes, 5);
        
        // 最后一天量比应该是2.0
        assert!((ratios[5] - 2.0).abs() < 0.01);
    }
} 