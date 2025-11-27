//! DMI 动向指标计算
//! 
//! DMI（Directional Movement Index）
//! - +DM = 当日最高价 - 前日最高价（若为正且大于-DM）
//! - -DM = 前日最低价 - 当日最低价（若为正且大于+DM）
//! - TR = max(最高价-最低价, |最高价-前收盘|, |最低价-前收盘|)
//! - +DI = +DM / TR × 100
//! - -DI = -DM / TR × 100
//! - ADX = MA(|(+DI)-(-DI)| / ((+DI)+(-DI)) × 100)

use serde::{Deserialize, Serialize};

/// DMI 数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DmiData {
    pub di_plus: f64,
    pub di_minus: f64,
    pub adx: f64,
    pub dx: f64,
}

/// 计算 DMI 指标
pub fn calculate_dmi(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: usize,
) -> (f64, f64, f64, f64) {
    if highs.len() < period + 1 || lows.len() < period + 1 || closes.len() < period + 1 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    let mut tr_values = Vec::new();
    let mut dm_plus_values = Vec::new();
    let mut dm_minus_values = Vec::new();
    
    for i in 1..highs.len() {
        // True Range
        let h_l = highs[i] - lows[i];
        let h_pc = (highs[i] - closes[i - 1]).abs();
        let l_pc = (lows[i] - closes[i - 1]).abs();
        let tr = h_l.max(h_pc).max(l_pc);
        tr_values.push(tr);
        
        // Directional Movement
        let up_move = highs[i] - highs[i - 1];
        let down_move = lows[i - 1] - lows[i];
        
        let dm_plus = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
        let dm_minus = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };
        
        dm_plus_values.push(dm_plus);
        dm_minus_values.push(dm_minus);
    }
    
    if tr_values.len() < period {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    let start = tr_values.len().saturating_sub(period);
    let atr = tr_values[start..].iter().sum::<f64>() / period as f64;
    let adm_plus = dm_plus_values[start..].iter().sum::<f64>() / period as f64;
    let adm_minus = dm_minus_values[start..].iter().sum::<f64>() / period as f64;
    
    if atr == 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    let di_plus = (adm_plus / atr) * 100.0;
    let di_minus = (adm_minus / atr) * 100.0;
    
    let dx = if di_plus + di_minus == 0.0 {
        0.0
    } else {
        ((di_plus - di_minus).abs() / (di_plus + di_minus)) * 100.0
    };
    
    let adx = dx;
    
    (di_plus, di_minus, adx, dx)
}

/// 计算 DMI 数据结构
pub fn calculate_dmi_data(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: usize,
) -> DmiData {
    let (di_plus, di_minus, adx, dx) = calculate_dmi(highs, lows, closes, period);
    DmiData { di_plus, di_minus, adx, dx }
}

/// 判断多头趋势
pub fn is_bullish_trend(di_plus: f64, di_minus: f64, adx: f64, threshold: f64) -> bool {
    di_plus > di_minus && adx > threshold
}

/// 判断空头趋势
pub fn is_bearish_trend(di_plus: f64, di_minus: f64, adx: f64, threshold: f64) -> bool {
    di_minus > di_plus && adx > threshold
}

/// ADX 趋势强度
pub fn adx_trend_strength(adx: f64) -> &'static str {
    if adx < 20.0 {
        "无趋势"
    } else if adx < 25.0 {
        "弱趋势"
    } else if adx < 50.0 {
        "趋势形成"
    } else if adx < 75.0 {
        "强趋势"
    } else {
        "极强趋势"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dmi_calculation() {
        let highs = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                        20.0, 21.0, 22.0, 23.0, 24.0];
        let lows = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                       19.0, 20.0, 21.0, 22.0, 23.0];
        let closes = vec![9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5,
                         19.5, 20.5, 21.5, 22.5, 23.5];
        
        let (di_plus, di_minus, adx, _) = calculate_dmi(&highs, &lows, &closes, 14);
        
        // 上涨趋势，+DI 应该大于 -DI
        assert!(di_plus >= di_minus);
    }
}

