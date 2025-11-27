//! RSI 指标计算
//! 
//! RSI（相对强弱指数）
//! RS = 平均上涨幅度 / 平均下跌幅度
//! RSI = 100 - 100 / (1 + RS)

/// 计算 RSI 指标
pub fn calculate_rsi(prices: &[f64]) -> f64 {
    calculate_rsi_with_period(prices, 14)
}

/// 计算指定周期的 RSI
pub fn calculate_rsi_with_period(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period + 1 {
        return 50.0;
    }
    
    // 计算第一个周期的平均涨跌幅
    let mut first_gain = 0.0;
    let mut first_loss = 0.0;
    
    for i in 1..=period {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            first_gain += change;
        } else {
            first_loss += -change;
        }
    }
    
    let mut avg_gain = first_gain / period as f64;
    let mut avg_loss = first_loss / period as f64;
    
    // Wilder 平滑法计算后续的平均涨跌幅
    for i in (period + 1)..prices.len() {
        let change = prices[i] - prices[i - 1];
        
        if change > 0.0 {
            avg_gain = (avg_gain * (period - 1) as f64 + change) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64) / period as f64;
        } else {
            avg_gain = (avg_gain * (period - 1) as f64) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + (-change)) / period as f64;
        }
    }
    
    if avg_loss == 0.0 {
        100.0
    } else {
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
}

/// 判断超买
pub fn is_overbought(rsi: f64, threshold: f64) -> bool {
    rsi > threshold
}

/// 判断超卖
pub fn is_oversold(rsi: f64, threshold: f64) -> bool {
    rsi < threshold
}

/// RSI 信号强度
pub fn rsi_signal_strength(rsi: f64) -> f64 {
    if rsi >= 70.0 {
        -(rsi - 70.0) / 30.0 // 超买，看空
    } else if rsi <= 30.0 {
        (30.0 - rsi) / 30.0 // 超卖，看多
    } else {
        0.0 // 中性
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi_calculation() {
        // 持续上涨，RSI 应该高
        let up_prices: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let rsi = calculate_rsi(&up_prices);
        assert!(rsi > 70.0);
        
        // 持续下跌，RSI 应该低
        let down_prices: Vec<f64> = (1..=20).rev().map(|x| x as f64).collect();
        let rsi = calculate_rsi(&down_prices);
        assert!(rsi < 30.0);
    }

    #[test]
    fn test_rsi_overbought_oversold() {
        assert!(is_overbought(75.0, 70.0));
        assert!(!is_overbought(65.0, 70.0));
        assert!(is_oversold(25.0, 30.0));
        assert!(!is_oversold(35.0, 30.0));
    }
}

