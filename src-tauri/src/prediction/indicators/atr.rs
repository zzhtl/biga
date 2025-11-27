//! ATR 平均真实波幅指标计算
//! 
//! ATR（Average True Range）
//! TR = max(最高价-最低价, |最高价-前收盘|, |最低价-前收盘|)
//! ATR = TR 的 N 日移动平均

/// 计算 ATR 指标
pub fn calculate_atr(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: usize,
) -> f64 {
    if highs.len() < period || lows.len() < period || closes.len() < period {
        return 0.0;
    }
    
    let mut trs = Vec::new();
    
    for i in 1..closes.len() {
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i - 1]).abs();
        let low_close = (lows[i] - closes[i - 1]).abs();
        let tr = high_low.max(high_close).max(low_close);
        trs.push(tr);
    }
    
    if trs.len() < period {
        return 0.0;
    }
    
    trs[trs.len() - period..].iter().sum::<f64>() / period as f64
}

/// 计算 ATR 百分比（相对于当前价格）
pub fn calculate_atr_percent(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: usize,
) -> f64 {
    let atr = calculate_atr(highs, lows, closes, period);
    let current_price = closes.last().copied().unwrap_or(1.0);
    
    if current_price == 0.0 {
        0.0
    } else {
        (atr / current_price) * 100.0
    }
}

/// 判断高波动市场
pub fn is_high_volatility(atr_percent: f64, threshold: f64) -> bool {
    atr_percent > threshold
}

/// 判断低波动市场
pub fn is_low_volatility(atr_percent: f64, threshold: f64) -> bool {
    atr_percent < threshold
}

/// 波动率水平描述
pub fn volatility_level(atr_percent: f64) -> &'static str {
    if atr_percent < 1.0 {
        "极低波动"
    } else if atr_percent < 2.0 {
        "低波动"
    } else if atr_percent < 3.0 {
        "正常波动"
    } else if atr_percent < 5.0 {
        "高波动"
    } else {
        "极高波动"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atr_calculation() {
        let highs = vec![10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5,
                        16.0, 15.5, 17.0, 16.5, 18.0];
        let lows = vec![9.0, 10.0, 10.5, 10.0, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5,
                       14.0, 14.5, 15.0, 15.5, 16.0];
        let closes = vec![9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0,
                         15.5, 15.0, 16.5, 16.0, 17.5];
        
        let atr = calculate_atr(&highs, &lows, &closes, 14);
        
        assert!(atr > 0.0);
    }

    #[test]
    fn test_atr_percent() {
        let highs = vec![10.0; 15];
        let lows = vec![9.0; 15];
        let closes = vec![9.5; 15];
        
        let atr_pct = calculate_atr_percent(&highs, &lows, &closes, 14);
        
        // ATR = 1.0, 当前价格 = 9.5, ATR% ≈ 10.5%
        assert!(atr_pct > 10.0);
    }
}

