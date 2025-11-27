//! CCI 商品通道指数计算
//! 
//! CCI（Commodity Channel Index）
//! TP = (最高价 + 最低价 + 收盘价) / 3
//! MA = TP 的 N 日简单移动平均
//! MD = TP 的 N 日平均偏差
//! CCI = (TP - MA) / (0.015 × MD)

/// 计算 CCI 指标
pub fn calculate_cci(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
    if highs.len() < period || lows.len() < period || closes.len() < period {
        return 0.0;
    }
    
    let start = highs.len().saturating_sub(period);
    let mut tp_values = Vec::with_capacity(period);
    
    for i in start..highs.len() {
        let tp = (highs[i] + lows[i] + closes[i]) / 3.0;
        tp_values.push(tp);
    }
    
    let ma = tp_values.iter().sum::<f64>() / tp_values.len() as f64;
    let md = tp_values.iter()
        .map(|&tp| (tp - ma).abs())
        .sum::<f64>() / tp_values.len() as f64;
    
    if md == 0.0 {
        return 0.0;
    }
    
    let current_tp = (highs.last().unwrap() + lows.last().unwrap() + closes.last().unwrap()) / 3.0;
    (current_tp - ma) / (0.015 * md)
}

/// 判断 CCI 超买
pub fn is_cci_overbought(cci: f64, threshold: f64) -> bool {
    cci > threshold
}

/// 判断 CCI 超卖
pub fn is_cci_oversold(cci: f64, threshold: f64) -> bool {
    cci < -threshold
}

/// CCI 信号强度
pub fn cci_signal_strength(cci: f64) -> f64 {
    if cci > 100.0 {
        -((cci - 100.0) / 200.0).min(1.0) // 超买，看空
    } else if cci < -100.0 {
        ((-cci - 100.0) / 200.0).min(1.0) // 超卖，看多
    } else {
        0.0 // 中性
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cci_calculation() {
        let highs = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                        20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0];
        let lows = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                       19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0];
        let closes = vec![9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5,
                         19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5];
        
        let cci = calculate_cci(&highs, &lows, &closes, 20);
        
        // 上涨趋势，CCI 应该为正
        assert!(cci > 0.0);
    }
}

