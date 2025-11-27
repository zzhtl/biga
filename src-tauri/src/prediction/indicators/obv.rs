//! OBV 能量潮指标计算
//! 
//! OBV（On Balance Volume）
//! - 价格上涨：OBV += 成交量
//! - 价格下跌：OBV -= 成交量
//! - 价格不变：OBV 不变

/// 计算 OBV 指标
pub fn calculate_obv(prices: &[f64], volumes: &[i64]) -> f64 {
    if prices.len() < 2 || volumes.len() < 2 {
        return 0.0;
    }
    
    let mut obv = 0.0;
    let len = prices.len().min(volumes.len());
    
    for i in 1..len {
        if prices[i] > prices[i - 1] {
            obv += volumes[i] as f64;
        } else if prices[i] < prices[i - 1] {
            obv -= volumes[i] as f64;
        }
    }
    
    obv
}

/// 计算 OBV 趋势
pub fn calculate_obv_trend(prices: &[f64], volumes: &[i64], period: usize) -> f64 {
    if prices.len() < period || volumes.len() < period {
        return 0.0;
    }
    
    let len = prices.len();
    let recent_prices = &prices[len - period..];
    let recent_volumes = &volumes[len - period..];
    
    let obv = calculate_obv(recent_prices, recent_volumes);
    let avg_volume = recent_volumes.iter().sum::<i64>() as f64 / period as f64;
    
    if avg_volume == 0.0 {
        0.0
    } else {
        obv / (avg_volume * period as f64)
    }
}

/// 判断 OBV 上升趋势
pub fn is_obv_rising(obv_values: &[f64], period: usize) -> bool {
    if obv_values.len() < period + 1 {
        return false;
    }
    
    let len = obv_values.len();
    let recent = &obv_values[len - period..];
    let prev = &obv_values[len - period - 1..len - 1];
    
    let recent_avg = recent.iter().sum::<f64>() / period as f64;
    let prev_avg = prev.iter().sum::<f64>() / period as f64;
    
    recent_avg > prev_avg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obv_calculation() {
        // 价格上涨，OBV 应该增加
        let prices = vec![10.0, 11.0, 12.0, 13.0];
        let volumes = vec![100, 200, 300, 400];
        let obv = calculate_obv(&prices, &volumes);
        assert!(obv > 0.0);
        
        // 价格下跌，OBV 应该减少
        let prices = vec![13.0, 12.0, 11.0, 10.0];
        let obv = calculate_obv(&prices, &volumes);
        assert!(obv < 0.0);
    }
}

